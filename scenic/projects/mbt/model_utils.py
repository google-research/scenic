# Copyright 2024 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model utils for MBT."""

from typing import Any

from absl import logging
import flax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.common_lib import debug_utils
from scenic.projects.vivit import model_utils as vivit_utils
import scipy


# Shared utils with ViViT
central_frame_initializer = vivit_utils.central_frame_initializer
average_frame_initializer = vivit_utils.average_frame_initializer
tile_positional_embeddings = vivit_utils.tile_positional_embeddings


def interpolate_positional_embeddings(restored_posemb_grid, n_tokens):
  """Interpolate positional embeddings from one size to another.

  Args:
    restored_posemb_grid: Positional embeddings from restored model. Shape is
      [n_restored_tokens, d]. It is assumed that the restored model used square
      image patches.
    n_tokens: Number of tokens in the target model. Can be a scalar if the
      target image is square, otherwise should be a tuple of 2.

  Returns:
    positional embedding resized to match n_tokens. Shape is [1, n_tokens, d]
  """

  restored_gs = int(np.sqrt(len(restored_posemb_grid)))
  if isinstance(n_tokens, tuple):
    gh, gw = n_tokens
  else:
    if n_tokens == len(restored_posemb_grid):
      # No need to interpolate
      return np.expand_dims(restored_posemb_grid, axis=0)
    gh = int(np.sqrt(n_tokens))
    gw = n_tokens // gh
    assert gh * gw == n_tokens
  logging.info('Resizing grid-size from (%s, %s) to (%s, %s).',
               restored_gs, restored_gs, gh, gw)
  restored_posemb_grid = restored_posemb_grid.reshape(restored_gs, restored_gs,
                                                      -1)
  zoom = (gh / restored_gs, gw / restored_gs, 1)
  restored_posemb_grid = scipy.ndimage.zoom(restored_posemb_grid, zoom, order=1)
  restored_posemb_grid = restored_posemb_grid.reshape(1, gh * gw, -1)
  return restored_posemb_grid


def initialise_from_train_state(
    config,
    train_state: Any,
    restored_train_state: Any,
    restored_model_cfg: ml_collections.ConfigDict,
    restore_output_proj: bool,
    mbt_transformer_key: str = 'Transformer',
    log_initialised_param_shapes: bool = True,
    one_config: bool = True,
    prefix_path: Any = None) -> Any:
  """Updates the train_state with data from restored_train_state.

  This function is written to be used for 'fine-tuning' experiments. Here, we
  do some surgery to support larger resolutions (longer sequence length) in
  the transformer block, with respect to the learned pos-embeddings.

  Args:
    config: Configurations for the model being updated, or tuple of configs.
    train_state: A raw TrainState for the model.
    restored_train_state: A TrainState that is loaded with parameters/state of a
      pretrained model.
    restored_model_cfg: Configuration of the model from which the
      restored_train_state come from. Usually used for some asserts.
    restore_output_proj: If true, load the final output projection. Set
      to False if finetuning to a new dataset.
    mbt_transformer_key: The key used for storing the subtree in the
      parameters that keeps Transformer weights, that are supposed to be
      initialized from the given pre-trained model.
    log_initialised_param_shapes: If true, print tabular summary of all the
      variables in the model once they have been initialised.
    one_config: If true, we have only a single config. If false, we get a tuple
      of configs in the order [init_config, model_config, dataset_config]. This
      is useful for works that build upon MBT and have different models in their
      config.
    prefix_path: If parameters are in a subtree.

  Returns:
    Updated train_state.
  """
  # Split up configs
  if one_config:
    init_config = config.init_from
    model_config = config.model
    dataset_config = config.dataset_configs
  else:
    init_config, model_config, dataset_config = config

  # Inspect and compare the parameters of the model with the init-model
  params = flax.core.unfreeze(train_state.optimizer.target)
  logging.info('Parameters in the target model are: %s', params)

  if init_config.get('checkpoint_format', 'scenic') == 'big_vision':
    restored_params = restored_train_state.optimizer['target']
  else:
    restored_params = restored_train_state.optimizer.target
  restored_params = flax.core.unfreeze(restored_params)
  if init_config.get('init_from_vit', True):
    if prefix_path:
      video_params = params[prefix_path]
    else:
      video_params = params
    # Start moving parameters, one-by-one and apply changes if needed
    for m_key, m_params in restored_params.items():
      if m_key == 'output_projection':
        if restore_output_proj:
          video_params[m_key] = m_params
        else:
          pass
      elif m_key == 'pre_logits':
        if model_config.representation_size is None:
          # We don't have representation_size in the new model, so let's ignore
          #   if from the pretained model, in case it has it.
          # Note, removing the key from the dictionary is necessary to prevent
          #   obscure errors from the Flax optimizer.
          video_params.pop(m_key, None)
        else:
          assert restored_model_cfg.model.representation_size
          video_params[m_key] = m_params

      elif m_key in ['Transformer']:
        for tm_key, tm_params in m_params.items():
          if tm_key == 'posembed_input':  # Might need resolution change
            init_posemb(
                video_params[mbt_transformer_key],
                m_params,
                init_config,
                model_config,
                dataset_config,
                restored_model_cfg,
                'posembed_input',
                prefix_path=prefix_path)
            init_posemb(
                video_params[mbt_transformer_key],
                m_params,
                init_config,
                model_config,
                dataset_config,
                restored_model_cfg,
                'posembed_input_spectrogram',
                prefix_path=prefix_path)
            init_posemb(
                video_params,
                m_params,
                init_config,
                model_config,
                dataset_config,
                restored_model_cfg,
                'bottleneck',
                prefix_path=prefix_path)
          elif 'encoderblock' in tm_key:
            logging.info('Loading encoder parameters.')
            init_encoderblock(video_params[mbt_transformer_key], m_params,
                              tm_key)
          else:  # Other parameters of the Transformer encoder
            video_params[mbt_transformer_key][tm_key] = tm_params
      elif m_key == 'embedding':
        init_embedding(video_params, m_params, init_config, model_config,
                       'embedding')
        init_embedding(video_params, m_params, init_config, model_config,
                       'embedding_spectrogram')
      else:
        if m_key in train_state.optimizer.target:
          video_params[m_key] = m_params
        if '%s_spectrogram' % m_key in train_state.optimizer.target:
          video_params['%s_spectrogram' % m_key] = m_params
        else:
          logging.info('Skipping %s. In restored model but not in target',
                       m_key)
  else:
    for m_key, m_params in restored_params.items():
      if m_key == 'output_projection':
        if restore_output_proj:
          params[m_key] = m_params
        else:
          pass
      elif m_key == 'pre_logits':
        if model_config.representation_size is None:
          # We don't have representation_size in the new model, so let's ignore
          #   if from the pretained model, in case it has it.
          # Note, removing the key from the dictionary is necessary to prevent
          #   obscure errors from the Flax optimizer.
          params.pop(m_key, None)
        else:
          assert restored_model_cfg.model.representation_size
          params[m_key] = m_params
      else:
        if m_key in train_state.optimizer.target:
          params[m_key] = m_params
        else:
          logging.info('Skipping %s. In restored model but not in target',
                       m_key)

  if log_initialised_param_shapes:
    logging.info('Parameter summary after initialising from train state')
    debug_utils.log_param_shapes(params)
  return train_state.replace(
      optimizer=train_state.optimizer.replace(target=flax.core.freeze(params)))


def init_posemb(to_params, from_params, init_config, model_config,
                dataset_config, restored_model_cfg, name, prefix_path=None):
  """Initialize the positional embeddings."""
  if name not in to_params:
    logging.info('No %s in target model', name)
  elif init_config.restore_positional_embedding:
    if name == 'bottleneck':
      posemb = to_params[name]
    else:
      posemb = to_params[name]['pos_embedding']
    restored_posemb = from_params['posembed_input']['pos_embedding']
    if restored_posemb.shape != posemb.shape:
      # Rescale the grid of pos, embeddings.
      # Default parameter shape is (1, N, 768)
      logging.info('Adapting positional embeddings %s from %s to %s',
                   name, restored_posemb.shape, posemb.shape)
      ntok = posemb.shape[1]
      if prefix_path:
        # MBT is part of a larger model
        classifier = restored_model_cfg.mbt.model.classifier
      else:
        classifier = restored_model_cfg.model.classifier
      if classifier == 'token':
        # the first token is the CLS token
        cls_tok = restored_posemb[:, :1]
        restored_posemb_grid = restored_posemb[0, 1:]
      else:
        cls_tok = restored_posemb[:, :0]
        restored_posemb_grid = restored_posemb[0]
      if model_config.classifier == 'token':
        ntok -= 1

      size_change = init_config.positional_embed_size_change
      if name == 'bottleneck':
        restored_posemb_grid = interpolate_positional_embeddings(
            restored_posemb_grid, ntok)
      elif size_change == 'tile':
        restored_posemb_grid = tile_positional_embeddings(
            restored_posemb_grid, ntok)
      elif size_change in ['resize_tile', 'resize']:
        temp_encoding = model_config.temporal_encoding_config
        if name.find('spectrogram') > -1:
          gh = ((dataset_config.spec_shape[0] *
                 dataset_config.num_spec_frames) //
                model_config.patches.size[0])
          gw = (dataset_config.spec_shape[1] //
                model_config.patches.size[1])
          tokens_per_frame = (gh, gw)
        elif temp_encoding.method == 'temporal_sampling':
          tokens_per_frame = int(ntok / temp_encoding.n_sampled_frames)
        elif temp_encoding.method == '3d_conv':
          # This is for RGB only.
          n_frames = (
              dataset_config.num_frames //
              model_config.patches.size[2])
          tokens_per_frame = ntok // n_frames
        else:
          raise AssertionError(
              f'Unknown temporal encoding {temp_encoding.method}')

        restored_posemb_grid = interpolate_positional_embeddings(
            restored_posemb_grid, tokens_per_frame)
        if size_change == 'resize_tile' and ntok != tokens_per_frame:
          restored_posemb_grid = restored_posemb_grid[0]
          restored_posemb_grid = tile_positional_embeddings(
              restored_posemb_grid, ntok)
      else:
        raise AssertionError(
            'Unknown positional embedding size changing method')
      # attach the CLS token again
      if model_config.classifier == 'token':
        restored_posemb = jnp.array(
            np.concatenate([cls_tok, restored_posemb_grid], axis=1))
      else:
        restored_posemb = restored_posemb_grid

    if name == 'bottleneck':
      to_params[name] = restored_posemb
    else:
      to_params[name]['pos_embedding'] = restored_posemb
  else:
    logging.info('Not restoring positional encodings from pretrained model')


def init_embedding(to_params, from_params, init_config, model_config, name):
  """Initialize input embedding."""
  if name not in to_params:
    logging.info('No %s in target model', name)
  elif init_config.get('restore_input_embedding', True):
    input_kernel = to_params[name]['kernel']
    restored_kernel = from_params['kernel']
    restored_bias = from_params['bias']

    if input_kernel.shape != restored_kernel.shape:
      kernel_init_method = model_config.temporal_encoding_config.kernel_init_method
      if input_kernel.shape == restored_kernel.shape[1:]:
        # Deflates a ViViT 3D embedder to work with 2D spectrogram inputs.
        restored_kernel = np.mean(restored_kernel, axis=0)
      elif input_kernel.shape[1:] != restored_kernel.shape:
        # Kernel dimensions are [t, c_in, c_out]
        restored_kernel = np.reshape(restored_kernel, input_kernel.shape)
      elif input_kernel.shape[0] == 1:
        # Kernel dimensions are [t, h, w, c_in, c_out]
        restored_kernel = np.expand_dims(restored_kernel, axis=0)
      elif kernel_init_method == 'average_frame_initializer':
        # This corresponds to "filter inflation" in
        # J Carreira and A Zisserman. Quo vadis, action recognition?
        # A new model and the kinetics dataset. CVPR 2017"
        logging.info('Initializing input kernel with filter inflation.')
        t = input_kernel.shape[0]
        restored_kernel = np.expand_dims(restored_kernel, axis=0)
        restored_kernel = np.tile(restored_kernel, [t, 1, 1, 1, 1]) / t
      elif kernel_init_method == 'average_arp_frame_initializer':
        # This corresponds to a combination of filter inflation and
        # the approximate rank pooling described in
        # H Bilen et al. Action Recognition with Dynamic Image Networks.
        # PAMI 2017.
        logging.info('Initialzing input kernel with ARP inflation')
        t = input_kernel.shape[0]
        restored_kernel = np.expand_dims(restored_kernel, axis=0)
        restored_kernel = np.tile(restored_kernel, [t, 1, 1, 1, 1])

        def average_arp(length):
          # Implements Equation 3 of Bilen et al. PAMI 2017
          array = np.arange(1, length + 1)

          harmonic = np.zeros((length + 1))
          harmonic[1:] = np.cumsum(1.0 / array)

          array = 2 * (length - array + 1) - (length + 1) * (
              harmonic[-1] - harmonic[:-1])
          return array

        normalizer = average_arp(t) / t
        normalizer = np.reshape(normalizer, [t, 1, 1, 1, 1])
        restored_kernel = restored_kernel * normalizer
      elif kernel_init_method == 'central_frame_initializer':
        logging.info('Initializing input kernel to select centre frame.')
        central_time_index = input_kernel.shape[0] // 2
        temp = np.zeros(input_kernel.shape)
        temp[central_time_index] = restored_kernel.copy()
        restored_kernel = temp
      else:
        raise AssertionError(
            'Unknown input kernel initialization {}'.format(kernel_init_method))

    to_params[name]['kernel'] = restored_kernel
    to_params[name]['bias'] = restored_bias
  else:
    logging.info('Not restoring input embedding parameters')


def init_encoderblock(to_params, from_params, tm_key):
  """Initialize encoder_block_parameters."""
  # Explicitly enumerate over the keys in the encoder-block. Don't just
  # assign the dictionary. It is possible for the target model to
  # contain keys that are not in the restored model.
  for enc_key in from_params[tm_key].keys():
    restoring_params = False
    if tm_key in to_params:
      assert enc_key in to_params[tm_key], '%s not in to_params[%s]' % (
          enc_key, tm_key)
      to_params[tm_key][enc_key] = from_params[tm_key][enc_key]
      restoring_params = True
    if '%s_spectrogram' % tm_key in to_params:
      assert enc_key in to_params['%s_spectrogram' %
                                  tm_key], '%s not in to_params[%s]' % (
                                      enc_key, '%s_spectrogram' % tm_key)
      to_params['%s_spectrogram' %
                tm_key][enc_key] = from_params[tm_key][enc_key]
      restoring_params = True
    if not restoring_params:
      logging.info('Warning: Not restoring encoder parameters.')
