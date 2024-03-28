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

"""Model utils for PolyViT."""

from typing import Any, Optional

from absl import logging
import flax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.common_lib import debug_utils
import scenic.projects.mbt.model_utils as mbt_utils
import scenic.projects.vivit.model_utils as vivit_utils
import scipy


def initialize_from_polyvit_train_state(
    train_state: Any,
    restored_train_state: Any,
    tokenizer_to_init_from: Optional[str] = None,
    tokenizer_to_init: Optional[str] = None,
    resolution_to_init: Optional[Any] = None,
    initialize_heads: bool = False) -> Any:
  """Initializes PolyViT with other PolyViT body."""

  params = flax.core.unfreeze(train_state.params)
  restored_params = restored_train_state.params
  restored_params = flax.core.unfreeze(restored_params)

  for m_key, m_params in restored_params.items():
    if m_key == 'vit_encoder':
      params[m_key] = m_params
    elif m_key == 'tokenizer':
      # Tokenizer(s) initialization.
      if tokenizer_to_init_from is None:
        # Only adding tokenizers which are also in `params`.
        for tm_key, tm_params in m_params.items():
          if tm_key in params[m_key]:
            params[m_key][tm_key] = tm_params
      else:
        # Initializing `tokenizer_to_init` from `tokenizer_to_init_from` and
        # adapting all needed shapes.
        params['tokenizer'][tokenizer_to_init]['cls'] = m_params[
            tokenizer_to_init_from]['cls']
        params['tokenizer'][tokenizer_to_init]['embedding']['bias'] = m_params[
            tokenizer_to_init_from]['embedding']['bias']
        # Below we assume that that patch h x w is the same for
        # tokenizer_to_init_from and for tokenizer_to_init.
        if tokenizer_to_init_from == 'tokenizer3d':
          params['tokenizer'][tokenizer_to_init]['embedding'][
              'kernel'] = m_params['tokenizer3d']['embedding']['kernel'].sum(
                  axis=0)
        elif tokenizer_to_init == 'tokenizer3d':
          # This is hardcoded since only this shape was used in linear eval of
          # video tasks.
          kernel3d = np.zeros([4, 16, 16, 3, 768])
          # Initializing the middle frame embedding as in ViViT.
          kernel3d[
              2, :, :] = m_params[tokenizer_to_init_from]['embedding']['kernel']
          params['tokenizer']['tokenizer3d']['embedding']['kernel'] = kernel3d
        else:
          params['tokenizer'][tokenizer_to_init]['embedding'][
              'kernel'] = m_params[tokenizer_to_init_from]['embedding'][
                  'kernel']

        # Positional embedding initialization.
        pos_embedding = m_params[tokenizer_to_init_from]['posembed_input'][
            'pos_embedding']
        # Excluding cls token embedding.
        new_pos_embedding = pos_embedding[:, 1:]
        if tokenizer_to_init_from == 'tokenizer3d':
          # This is hardcoded, 8 = 32 / 4 (number of frames in the input /
          # number of frames in the patch).
          new_pos_embedding = new_pos_embedding.reshape(1, 8, -1,
                                                        pos_embedding.shape[2])
          # Averaging per-frame positional embeddings to obtain a 2d
          # initialization.
          new_pos_embedding = new_pos_embedding.mean(axis=1)
        # 2d interpolation if needed.
        if resolution_to_init is not None:
          # This is hardcoded. We assume that image resolution is
          # 384 x 384, video frame resolution is 224 x 224 and audio spectrogram
          # resolution is 800 x 128. We assume that patch size is 16 x 16.
          if tokenizer_to_init_from == 'tokenizer2d':
            new_pos_embedding = new_pos_embedding.reshape(24, 24, -1)
            zoom = (resolution_to_init[0] / 384, resolution_to_init[1] / 384, 1)
          elif tokenizer_to_init_from == 'tokenizer_spec':
            new_pos_embedding = new_pos_embedding.reshape(50, 8, -1)
            zoom = (resolution_to_init[0] / 800, resolution_to_init[1] / 128, 1)
          else:
            new_pos_embedding = new_pos_embedding.reshape(14, 14, -1)
            zoom = (resolution_to_init[0] / 224, resolution_to_init[1] / 224, 1)
          new_pos_embedding = scipy.ndimage.zoom(
              new_pos_embedding, zoom, order=1)
          new_pos_embedding = new_pos_embedding.reshape(
              1, new_pos_embedding.shape[0] * new_pos_embedding.shape[1],
              -1)
        if tokenizer_to_init == 'tokenizer3d':
          # This is hardcoded, 8 = 32 / 4 (number of frames in the input /
          # number of frames in the patch).
          new_pos_embedding = np.tile(new_pos_embedding, (1, 8, 1))
        # Concatenating with the cls embedding.
        new_pos_embedding = np.concatenate(
            [pos_embedding[:, :1], new_pos_embedding], axis=1)
        params['tokenizer'][tokenizer_to_init][
            'posembed_input']['pos_embedding'] = new_pos_embedding
    elif initialize_heads and m_key in params:
      # Initializing heads if needed.
      params[m_key] = m_params

  return train_state.replace(params=flax.core.freeze(params))


def initialize_from_mbt_train_state(
    train_state: Any,
    restored_train_state: Any,
    tokenizer_to_init: str = 'tokenizer_spec',
    resolution_to_init: Optional[Any] = None,
    initialize_head: bool = False,
) -> Any:
  """Initializes PolyViT with AViT body."""

  params = flax.core.unfreeze(train_state.params)
  restored_params = flax.core.unfreeze(restored_train_state.params)

  for m_key, m_params in restored_params.items():
    if m_key == 'Transformer':
      for tm_key, tm_params in m_params.items():
        if tm_key == 'posembed_input_spec':
          # Initializing positional embeddings.
          if tokenizer_to_init == 'tokenizer_spec':
            params['tokenizer']['tokenizer_spec']['posembed_input'] = tm_params
          else:
            # Adapting positional embedding shapes if needed.
            pos_embedding = tm_params['pos_embedding']
            # Excluding cls token embedding.
            new_pos_embedding = pos_embedding[:, 1:]
            # 2d interpolation if needed.
            if resolution_to_init is not None:
              # Assuming that spectrogram is of shape 800 x 128 and patch size
              # is 16 x 16. Reshaping to (800 / 16) x (128 / 16).
              new_pos_embedding = new_pos_embedding.reshape(50, 8, -1)
              zoom = (resolution_to_init[0] / 800, resolution_to_init[1] / 128,
                      1)
              new_pos_embedding = scipy.ndimage.zoom(
                  new_pos_embedding, zoom, order=1)
              new_pos_embedding = new_pos_embedding.reshape(
                  1, new_pos_embedding.shape[0] * new_pos_embedding.shape[1],
                  -1)
            if tokenizer_to_init == 'tokenizer3d':
              # This is hardcoded, 8 = 32 / 4 (number of frames in the input /
              # number of frames in the patch).
              new_pos_embedding = np.tile(new_pos_embedding, (1, 8, 1))
            # Concatenating with the cls embedding.
            new_pos_embedding = np.concatenate(
                [pos_embedding[:, :1], new_pos_embedding], axis=1)
            params['tokenizer'][tokenizer_to_init][
                'posembed_input']['pos_embedding'] = new_pos_embedding
        elif tm_key.startswith('encoderblock'):
          # Removing '_spectrogram' suffix.
          params['vit_encoder'][tm_key[:-5]] = tm_params
        elif tm_key in params['vit_encoder']:
          params['vit_encoder'][tm_key] = tm_params
    elif m_key == 'cls':
      params['tokenizer'][tokenizer_to_init]['cls'] = m_params
    elif m_key == 'embedding_spec':
      # Initializing patch embedding.
      if tokenizer_to_init in ['tokenizer2d', 'tokenizer_spec']:
        params['tokenizer'][tokenizer_to_init]['embedding'] = m_params
      else:
        # This is hardcoded since only this shape was used in linear eval of
        # video tasks.
        kernel3d = np.zeros([4, 16, 16, 3, 768])
        # Initializing the middle frame embedding as in ViViT.
        kernel3d[2, :, :] = m_params['kernel']
        params['tokenizer'][tokenizer_to_init]['embedding'][
            'kernel'] = kernel3d
        params['tokenizer'][tokenizer_to_init]['embedding']['bias'] = m_params[
            'bias']
    elif m_key == 'output_projection' and initialize_head:
      # Initializing head if needed.
      head_name = [
          x for x in params.keys() if x not in ['tokenizer', 'vit_encoder']
      ][0]
      params[head_name]['output_projection'] = m_params

  return train_state.replace(params=flax.core.freeze(params))


def initialize_from_vivit_train_state(
    train_state: Any,
    restored_train_state: Any,
    tokenizer_to_init: str = 'tokenizer3d',
    resolution_to_init: Optional[Any] = None,
    initialize_head: bool = False) -> Any:
  """Initializes PolyViT with ViViT body."""

  params = flax.core.unfreeze(train_state.params)
  restored_params = flax.core.unfreeze(restored_train_state.params)

  for m_key, m_params in restored_params.items():
    # Not initializing heads.
    if m_key == 'Transformer':
      for tm_key, tm_params in m_params.items():
        if tm_key == 'posembed_input':
          # Initializing positional embeddings.
          if tokenizer_to_init == 'tokenizer3d':
            params['tokenizer']['tokenizer3d']['posembed_input'] = tm_params
          else:
            # Adapting positional embedding shapes if needed.
            pos_embedding = tm_params['pos_embedding']
            # Excluding cls token embedding and reshaping into per-frame
            # embeddings. 8 = 32 / 4 (number of frames in the input /
            # number of frames in the patch).
            new_pos_embedding = pos_embedding[:, 1:].reshape(
                1, 8, -1, pos_embedding.shape[2])
            # Averaging per-frame positional embeddings to obtain a 2d
            # initialization.
            new_pos_embedding = new_pos_embedding.mean(axis=1)
            # 2d interpolation if needed.
            if resolution_to_init is not None:
              # This is hardcoded. We assume that video frame resolution is
              # 224 x 224 and patch size is 16 x 16.
              new_pos_embedding = new_pos_embedding.reshape(14, 14, -1)
              zoom = (resolution_to_init[0] / 224, resolution_to_init[1] / 224,
                      1)
              new_pos_embedding = scipy.ndimage.zoom(
                  new_pos_embedding, zoom, order=1)
              new_pos_embedding = new_pos_embedding.reshape(
                  1, new_pos_embedding.shape[0] * new_pos_embedding.shape[1],
                  -1)
            # Concatenating with the cls embedding.
            new_pos_embedding = np.concatenate(
                [pos_embedding[:, :1], new_pos_embedding], axis=1)
            params['tokenizer'][tokenizer_to_init][
                'posembed_input']['pos_embedding'] = new_pos_embedding
        elif tm_key in params['vit_encoder']:
          params['vit_encoder'][tm_key] = tm_params
    elif m_key == 'cls':
      params['tokenizer'][tokenizer_to_init]['cls'] = m_params
    elif m_key == 'embedding':
      # Patch embedding initialization.
      if tokenizer_to_init == 'tokenizer3d':
        params['tokenizer']['tokenizer3d']['embedding'] = m_params
      else:
        params['tokenizer'][tokenizer_to_init]['embedding']['bias'] = m_params[
            'bias']
        # 2d patch embedding is a sum along the frame dimension of the 3d video
        # patch embedding.
        params['tokenizer'][tokenizer_to_init]['embedding'][
            'kernel'] = m_params['kernel'].sum(axis=0)
    elif m_key == 'output_projection' and initialize_head:
      # Initializing head if needed.
      head_name = [
          x for x in params.keys() if x not in ['tokenizer', 'vit_encoder']
      ][0]
      params[head_name]['output_projection'] = m_params

  return train_state.replace(params=flax.core.freeze(params))


def initialise_from_vit_train_state(
    config,
    train_state: Any,
    restored_train_state: Any,
    restored_model_cfg: ml_collections.ConfigDict,
    log_initialised_param_shapes: bool = True) -> Any:
  """Updates the train_state with data from restored_train_state (ViT model).

  This function is written to be used for 'fine-tuning' experiments. Here, we
  do some surgery to support larger resolutions (longer sequence length) in
  the transformer block, with respect to the learned pos-embeddings.

  Args:
    config: Configurations for the model being updated.
    train_state: A raw TrainState for the model.
    restored_train_state: A TrainState that is loaded with parameters/state of a
      pretrained model.
    restored_model_cfg: Configuration of the model from which the
      restored_train_state come from. Usually used for some asserts.
    log_initialised_param_shapes: If true, print tabular summary of all the
      variables in the model once they have been initialised.

  Returns:
    Updated train_state.
  """
  # Inspect and compare the parameters of the model with the init-model.
  params = flax.core.unfreeze(train_state.params)
  restored_params = flax.core.unfreeze(restored_train_state.params)

  # Start moving parameters, one-by-one and apply changes if needed.
  for m_key, m_params in restored_params.items():
    if m_key in ['Transformer', 'SpatialTransformer']:
      for tm_key, tm_params in m_params.items():
        if tm_key == 'posembed_input':  # Might need resolution change.
          if 'tokenizer2d' in params['tokenizer']:
            init_posemb(params['tokenizer']['tokenizer2d'], m_params, config,
                        restored_model_cfg, 'resize')
          if 'tokenizer3d' in params['tokenizer']:
            init_posemb(params['tokenizer']['tokenizer3d'], m_params, config,
                        restored_model_cfg,
                        config.init_from.positional_embed_size_change)
          if 'tokenizer_spec' in params['tokenizer']:
            init_spec_posemb(params['tokenizer']['tokenizer_spec'], m_params,
                             config,
                             restored_model_cfg)
        elif 'encoderblock' in tm_key:
          init_encoderblock(params, m_params, tm_key)
        else:  # Other parameters of the Transformer encoder.
          params['vit_encoder'][tm_key] = tm_params
    elif m_key == 'cls':
      for tokenizer_name in ['tokenizer2d', 'tokenizer3d', 'tokenizer_spec']:
        if tokenizer_name in params['tokenizer']:
          params['tokenizer'][tokenizer_name]['cls'] = m_params
    elif m_key == 'embedding':
      for tokenizer_name in ['tokenizer2d', 'tokenizer_spec']:
        if tokenizer_name in params['tokenizer']:
          params['tokenizer'][tokenizer_name]['embedding'] = m_params
      if 'tokenizer3d' in params['tokenizer']:
        init_embedding(params['tokenizer']['tokenizer3d'], m_params, config)
    else:
      if m_key in train_state.params:
        params[m_key] = m_params
      else:
        logging.info('Skipping %s. In restored model but not in target', m_key)

  if log_initialised_param_shapes:
    logging.info('Parameter summary after initialising from train state')
    debug_utils.log_param_shapes(params)
  return train_state.replace(params=flax.core.freeze(params))


def init_posemb(to_params, from_params, config, restored_model_cfg,
                positional_embed_size_change):
  """Initialize the positional embeddings."""
  with_cls_token, num_video_frames = get_cls_token_and_video_frames(config)
  restored_with_cls_token, _ = get_cls_token_and_video_frames(
      restored_model_cfg)
  if config.init_from.restore_positional_embedding:
    posemb = to_params['posembed_input']['pos_embedding']
    restored_posemb = from_params['posembed_input']['pos_embedding']
    if restored_posemb.shape != posemb.shape:
      # Rescale the grid of pos, embeddings.
      # Default parameter shape is (1, N, 768)
      logging.info('Adapting positional embeddings from %s to %s',
                   restored_posemb.shape, posemb.shape)
      ntok = posemb.shape[1]
      if restored_with_cls_token:
        # The first token is the CLS token.
        cls_tok = restored_posemb[:, :1]
        restored_posemb_grid = restored_posemb[0, 1:]
      else:
        cls_tok = restored_posemb[:, :0]
        restored_posemb_grid = restored_posemb[0]
      if with_cls_token:
        ntok -= 1
      restored_gs = int(np.sqrt(len(restored_posemb_grid)))
      gs = int(np.sqrt(ntok))
      if with_cls_token != restored_with_cls_token:
        logging.warning('Only one of target and restored model uses'
                        'classification token')
        if restored_gs == gs:
          # In case the following `if` is not going to run, lets add batch dim:
          restored_posemb = restored_posemb_grid[None, ...]

      if restored_gs != gs:  # We need resolution change.
        if positional_embed_size_change == 'resize':
          restored_posemb_grid = vivit_utils.interpolate_positional_embeddings(
              restored_posemb_grid, ntok)

        elif positional_embed_size_change == 'tile':
          restored_posemb_grid = vivit_utils.tile_positional_embeddings(
              restored_posemb_grid, ntok)

        elif positional_embed_size_change == 'resize_tile':
          n_frames = (
              num_video_frames // config.model.modalities.video.patches.size[2])
          tokens_per_frame = ntok // n_frames
          restored_posemb_grid = vivit_utils.interpolate_positional_embeddings(
              restored_posemb_grid, tokens_per_frame)
          restored_posemb_grid = restored_posemb_grid[0]
          restored_posemb_grid = vivit_utils.tile_positional_embeddings(
              restored_posemb_grid, ntok)

        else:
          raise AssertionError(
              'Unknown positional embedding size changing method')
        # Attach the CLS token again.
        if with_cls_token:
          restored_posemb = jnp.array(
              np.concatenate([cls_tok, restored_posemb_grid], axis=1))
        else:
          restored_posemb = restored_posemb_grid

    to_params['posembed_input']['pos_embedding'] = restored_posemb
  else:
    logging.info('Not restoring positional encodings from pretrained model')


def init_spec_posemb(to_params, from_params, config, restored_model_cfg):
  """Initialize the spectrogram positional embeddings."""
  with_cls_token, _ = get_cls_token_and_video_frames(config)
  restored_with_cls_token, _ = get_cls_token_and_video_frames(
      restored_model_cfg)
  if config.init_from.restore_positional_embedding:
    posemb = to_params['posembed_input']['pos_embedding']
    restored_posemb = from_params['posembed_input']['pos_embedding']
    # Rescale the grid of pos, embeddings.
    # Default parameter shape is (1, N, 768)
    logging.info('Adapting spectrogram positional embeddings from %s to %s',
                 restored_posemb.shape, posemb.shape)
    ntok = posemb.shape[1]
    if restored_with_cls_token:
      # The first token is the CLS token.
      cls_tok = restored_posemb[:, :1]
      restored_posemb_grid = restored_posemb[0, 1:]
    else:
      cls_tok = restored_posemb[:, :0]
      restored_posemb_grid = restored_posemb[0]
    if with_cls_token:
      ntok -= 1

    gh = ((config.model.modalities.audio.spec_shape[0] *
           config.model.modalities.audio.num_spec_frames) //
          config.model.modalities.audio.patches.size[0])
    gw = (config.model.modalities.audio.spec_shape[1] //
          config.model.modalities.audio.patches.size[1])
    tokens_per_frame = (gh, gw)

    restored_posemb_grid = mbt_utils.interpolate_positional_embeddings(
        restored_posemb_grid, tokens_per_frame
    )
    restored_posemb_grid = restored_posemb_grid[0]
    restored_posemb_grid = mbt_utils.tile_positional_embeddings(
        restored_posemb_grid, ntok
    )

    # Attach the CLS token again.
    if with_cls_token:
      restored_posemb = jnp.array(
          np.concatenate([cls_tok, restored_posemb_grid], axis=1))
    else:
      restored_posemb = restored_posemb_grid

    to_params['posembed_input']['pos_embedding'] = restored_posemb
  else:
    logging.info('Not restoring positional encodings from pretrained model')


def init_encoderblock(to_params, from_params, tm_key):
  """Initialize encoder_block_parameters."""
  # Explicitly enumerate over the keys in the encoder-block. Don't just
  # assign the dictionary. It is possible for the target model to
  # contain keys that are not in the restored model.
  for enc_key in from_params[tm_key].keys():
    if tm_key in to_params['vit_encoder']:
      to_params['vit_encoder'][tm_key][enc_key] = from_params[tm_key][enc_key]
    else:
      for tokenizer_name in ['tokenizer2d', 'tokenizer3d', 'tokenizer_spec']:
        if tokenizer_name in to_params['tokenizer']:
          to_params['tokenizer'][tokenizer_name][tm_key][enc_key] = from_params[
              tm_key][enc_key]


def init_embedding(to_params, from_params, config):
  """Initialize input embedding."""
  if config.init_from.get('restore_input_embedding', True):
    input_kernel = to_params['embedding']['kernel']
    restored_kernel = from_params['kernel']
    restored_bias = from_params['bias']
    if input_kernel.shape != restored_kernel.shape:
      kernel_init_method = config.model.modalities.video.kernel_init_method
      if kernel_init_method == 'average_frame_initializer':
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
          # Implements Equation 3 of Bilen et al. PAMI 2017.
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

    to_params['embedding']['kernel'] = restored_kernel
    to_params['embedding']['bias'] = restored_bias
  else:
    logging.info('Not restoring input embedding parameters')


def get_cls_token_and_video_frames(config):
  """Returns whether there is CLS token and the number of video frames."""

  has_cls_token = False
  num_video_frames = None

  for ds_name, cfg in config.datasets.items():
    # TODO(vlikhosherstov): Add more datasets.
    if ds_name in ['kinetics400', 'moments_in_time', 'epic_kitchens']:
      num_video_frames = cfg.num_frames

  for head_type, head_cfg in config.model.heads.items():
    for cfg in head_cfg.values():
      if head_type in ['label', 'multilabel', 'bow'
                      ] and cfg.classifier in ['token', '0']:
        has_cls_token = True

  return has_cls_token, num_video_frames
