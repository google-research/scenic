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

"""Contains model utility functions."""

from typing import Any, Dict, List, Optional, Sequence

from absl import logging
import flax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.projects.vivit import model_utils as vivit_utils
from scenic.train_lib import train_utils
import scipy.ndimage

_POSITIONAL_EMBEDDING_KEY = 'posembed_input_view'
_BOTTLENECK_KEY = 'bottleneck'


def get_input_token_temporal_dims(
    num_frames: int,
    view_configs: Sequence[ml_collections.ConfigDict]) -> List[int]:
  """Returns temporal dims of input tokens for each view.

  Args:
    num_frames: Number of frames in the input video.
    view_configs: Configurations of each view in the MTV model.

  Returns:
    Temporal dimensions of input tokens from each view.
  """
  return [num_frames // view['patches']['size'][2] for view in view_configs]


def get_temporal_dims_merged_into_space(
    num_frames: int,
    view_configs: Sequence[ml_collections.ConfigDict]) -> List[int]:
  """Returns temporal dims merged into spatial dim for each view.

  In MTV, different views have different temporal dimensions. Before feeding
  tokens from different views into the multiview encoder, we reshape the tokens
  so that they have the same temporal dimension, part of the temporal dimensions
  are folded into the spatial dimesion.

  Args:
    num_frames: Number of frames in the input video.
    view_configs: Configurations of each view in the MTV model.

  Returns:
    Temporal dimensions that were merged into the spatial dimensions.
  """
  dims = get_input_token_temporal_dims(num_frames, view_configs)
  return [d // min(dims) for d in dims]


def interpolate_input_embedding(
    embedding_params: Dict[str, jnp.ndarray],
    restored_embedding_params: Dict[str, jnp.ndarray],
):
  """Interpolates input embedding.

  This function is used to initialize input embeddings for MTV models when the
  current model uses a different size of tubelets than the pretrained one.

  Args:
    embedding_params: A dict of embedding parameters to be updated containing a
      kernel, which is a 5D float tensor of shape (new_kernel_t, new_kernel_h,
      new_kernel_w, in_channels, out_channels).
    restored_embedding_params: A dict of embedding parameters from which we load
      the weights. It has a `kernel` parameter, which is a 5D float tensor of
      shape (kernel_t, kernel_h, kernel_w, in_channels, out_channels).

  Returns:
    A dict of updated parameters. Only `kernel` is updated.
  """
  kernel = embedding_params['kernel']
  restored_kernel = restored_embedding_params['kernel']
  logging.info('Resizing input embedding kernel from %s to %s.',
               restored_kernel.shape, kernel.shape)
  zoom = (
      kernel.shape[0] / restored_kernel.shape[0],
      kernel.shape[1] / restored_kernel.shape[1],
      kernel.shape[2] / restored_kernel.shape[2],
      1,
      1,
  )
  embedding_params['kernel'] = scipy.ndimage.zoom(
      restored_kernel, zoom, order=1)
  embedding_params['bias'] = restored_embedding_params['bias']


def interpolate_cls_tokens(cls: jnp.ndarray,
                           restored_cls: jnp.ndarray) -> jnp.ndarray:
  """Interpolates CLS tokens.

  This function is used to initialize CLS tokens for MTV models when the current
  CLS tokens have a different shape than the pretrained ones.

  Args:
    cls: A 4D float tensor of shape (1, new_time, 1, channels) representing the
      CLS tokens to be updated.
    restored_cls: A 4D float tensor of shape (1, old_time, 1, channels)
      representing the CLS tokens from which we load the weights.

  Returns:
    A 4D float tensor of shape (1, new_time, 1, channels) representing the
    resized CLS tokens.
  """
  logging.info('Resizing CLS tokens from %s to %s.', restored_cls.shape,
               cls.shape)
  zoom = (1, cls.shape[1] / restored_cls.shape[1], 1, 1)
  return scipy.ndimage.zoom(restored_cls, zoom, order=1)


def _get_view_index(key: str) -> int:
  return int(key[len(_POSITIONAL_EMBEDDING_KEY):])


def init_bottleneck(params: Dict[str, Any], restored_bottleneck: jnp.ndarray):
  """Initialize bottleneck tokens from a pretrained model."""
  bottleneck = params[_BOTTLENECK_KEY]
  if bottleneck.shape != restored_bottleneck.shape:
    logging.info('Resizing bottleneck tokens from %s to %s.',
                 restored_bottleneck.shape, bottleneck.shape)
    zoom = (1, bottleneck.shape[1] / restored_bottleneck.shape[1], 1, 1)
    params[_BOTTLENECK_KEY] = scipy.ndimage.zoom(
        restored_bottleneck, zoom, order=1)
  else:
    params[_BOTTLENECK_KEY] = restored_bottleneck


def central_frame_init_embedding(
    to_params: Dict[str, Any],
    from_params: Dict[str, Any],
    view_idx: int,
    config: ml_collections.ConfigDict,
):
  """Initialize input embedding from a ViT model.

  This function is adapted from scenic.projects.vivit.google.model_utils. Here,
  we add support to interpolate the input embeddings if the current model has
  different spatial patch sizes than the pretrained model.

  Args:
    to_params: Parameters of the current model.
    from_params: Model parameters where we load the weights from.
    view_idx: View index.
    config: Current model config.
  """
  if config.init_from.get('restore_input_embedding', True):
    name = f'embedding_view{view_idx}'
    input_kernel = to_params[name]['kernel']
    restored_kernel = from_params['kernel']
    restored_bias = from_params['bias']
    logging.info('Initializing input kernel to select centre frame.')
    if input_kernel.shape[1:] != restored_kernel.shape:
      logging.info('Kernel sizes do not match. Interpolation is performed.')
      (restored_kernel_h, restored_kernel_w, in_depth, out_depth) = (
          restored_kernel.shape
      )
      reshaped_kernel = restored_kernel.reshape(restored_kernel_h,
                                                restored_kernel_w, -1)
      patch_size = config.model.view_configs[view_idx].patches.size
      zoom = (patch_size[0] / restored_kernel_h,
              patch_size[1] / restored_kernel_w, 1)
      resized_kernel = scipy.ndimage.zoom(reshaped_kernel, zoom, order=1)
      resized_kernel = resized_kernel.reshape(
          (patch_size[0], patch_size[1], in_depth, out_depth))
    else:
      resized_kernel = restored_kernel
    central_time_index = input_kernel.shape[0] // 2
    temp = np.zeros(input_kernel.shape)
    temp[central_time_index] = resized_kernel.copy()
    to_params[name]['kernel'] = temp
    to_params[name]['bias'] = restored_bias


def initialize_from_mtv_parameters(
    config: ml_collections.ConfigDict,
    params: Dict[str, Any],
    restored_model_cfg: ml_collections.ConfigDict,
    restored_params: Dict[str, Any],
    restore_output_projection: bool,
    model_prefix_path: Optional[List[str]] = None,
):
  """Initialize MTV parameters from a MTV model.

  Args:
    config: Configuration for the model being updated.
    params: The parameters of the model.
    restored_model_cfg: Configuration of the model from which the
      restored_train_state come from. Usually used for some asserts.
    restored_params: Restored parameters from the given pretrained checkpoint.
    restore_output_projection: Whether or not to restore the weights from output
      projection.
    model_prefix_path: The parent keys in the model dict where the restored
      model should reside.
  """
  if model_prefix_path:
    to_params = params[model_prefix_path[0]]
    for prefix in model_prefix_path[1:]:
      to_params = to_params[prefix]
  else:
    to_params = params
  for m_key, m_params in restored_params.items():
    if m_key == 'output_projection':
      if restore_output_projection:
        to_params[m_key] = m_params
    elif m_key == _BOTTLENECK_KEY:
      init_bottleneck(to_params, m_params)
    elif 'cls' in m_key:
      if config.model.classifier != 'token':
        logging.info('Skipping %s since classifier != `token`.', m_key)
        continue
      if to_params[m_key].shape[1] != m_params.shape[1]:
        to_params[m_key] = interpolate_cls_tokens(to_params[m_key], m_params)
      else:
        to_params[m_key] = m_params
    elif 'embedding' in m_key:
      if to_params[m_key]['kernel'].shape != m_params['kernel'].shape:
        interpolate_input_embedding(to_params[m_key], m_params)
      else:
        to_params[m_key] = m_params
    elif m_key == 'global_encoder':
      for ge_key, ge_params in m_params.items():
        if 'posembed_input' in ge_key:  # Might need resolution change
          vivit_utils.init_posemb(
              to_params[m_key],
              m_params,
              config,
              restored_model_cfg,
              is_temporal=True)
        else:
          to_params[m_key][ge_key] = ge_params
    elif m_key == 'MultiviewEncoder':
      for tm_key, tm_params in m_params.items():
        if 'posembed_input' in tm_key:  # Might need resolution change
          vivit_utils.init_posemb(
              to_params['MultiviewEncoder'],
              m_params,
              config,
              restored_model_cfg,
              is_temporal=False,
              posemb_name=tm_key,
              restored_posemb_name=tm_key)
        else:
          to_params[m_key][tm_key] = tm_params
    else:
      to_params[m_key] = m_params


def initialize_from_mtv_train_state(
    config: ml_collections.ConfigDict,
    train_state: train_utils.TrainState,
    restored_train_state: train_utils.TrainState,
    restored_model_cfg: ml_collections.ConfigDict,
    restore_output_projection: bool,
    model_prefix_path: Optional[List[str]] = None,
) -> train_utils.TrainState:
  """Updates MTV's train_state with a pretrained MTV weights.

  Args:
    config: Configurations for the model being updated.
    train_state: A raw TrainState for the model.
    restored_train_state: A dict. Each key is a Ti/S/B/L ViT model and the
      corresponding value is a TrainState that is loaded with parameters/state
      of pretrained models.
    restored_model_cfg: Configurations of models from which the
      restored_train_states come from. Often only the classifier information is
      used for interpolating the positional embeddings.
    restore_output_projection: Whether or not to restore the weights from output
      projection.
    model_prefix_path: The parent keys in the model dict where the restored
      model should reside.

  Returns:
    Updated train_state.
  """

  params = flax.core.unfreeze(train_state.params)
  restored_params = flax.core.unfreeze(restored_train_state.params)
  initialize_from_mtv_parameters(config, params, restored_model_cfg,
                                 restored_params, restore_output_projection,
                                 model_prefix_path)
  return train_state.replace(params=flax.core.freeze(params))


def initialize_one_view_from_vit_parameters(
    config: ml_collections.ConfigDict,
    params: Dict[str, Any],
    restored_model_cfg: ml_collections.ConfigDict,
    restored_params: Dict[str, Any],
    view_idx: int,
    transformer_key: str = 'MultiviewEncoder'):
  """Initialize one view of MTV from a ViT model.

  Args:
    config: Configuration for the model being updated.
    params: The parameters of the model.
    restored_model_cfg: Configuration of the model from which the
      restored_train_state come from. Usually used for some asserts.
    restored_params: Restored parameters from the given pretrained checkpoint.
    view_idx: The index of the view for which we restore the model.
    transformer_key: The key of transformer whose weights are being updated.
  """

  for m_key, m_params in restored_params.items():
    if m_key == 'output_projection':
      pass
    elif m_key == 'pre_logits':
      # We don't do a linear projection in this model. pre_logits is generated
      # as an identity transformation.
      pass
    elif m_key == 'cls':
      if config.model.classifier == 'token':
        # The CLS token from a ViT model has a shape of (1, 1, channels) while
        # the CLS token from the MTV model has a shape of (1, temporal_dims, 1,
        # channels).
        cls_key = f'cls_view{view_idx}'
        temporal_dims = params[cls_key].shape[1]
        params[cls_key] = jnp.tile(m_params[jnp.newaxis, ...],
                                   [1, temporal_dims, 1, 1])
    elif m_key == 'Transformer':
      for tm_key, tm_params in m_params.items():
        if 'posembed_input' == tm_key:  # Might need resolution change
          vivit_utils.init_posemb(
              params[transformer_key],
              m_params,
              config,
              restored_model_cfg,
              is_temporal=False,
              posemb_name=f'posembed_input_view{view_idx}')
        elif 'encoderblock' in tm_key:
          msa_encoderblock_name = f'{tm_key}_view{view_idx}'
          cross_view_encoderblock_name = f'cross_view_{tm_key}'
          if msa_encoderblock_name in params[transformer_key]:
            params[transformer_key][msa_encoderblock_name] = tm_params
          elif cross_view_encoderblock_name in params[transformer_key]:
            cross_view_encoderblock_params = params[transformer_key][
                cross_view_encoderblock_name]
            # In vit.Encoder1DBlock(), default names are used. For example,
            # `LayerNorm_0` stores the params for the layer norm before MSA and
            # `LayerNorm_1` stores the params for the layer norm before MLP.
            cross_view_encoderblock_params[
                f'msa_ln_view{view_idx}'] = tm_params['LayerNorm_0']
            cross_view_encoderblock_params[f'msa_view{view_idx}'] = tm_params[
                'MultiHeadDotProductAttention_0']
            cross_view_encoderblock_params[
                f'mlp_ln_view{view_idx}'] = tm_params['LayerNorm_1']
            cross_view_encoderblock_params[f'mlp_view{view_idx}'] = tm_params[
                'MlpBlock_0']
        else:
          logging.info(
              'Skipping restoring `%s`, in restored model but not in the'
              ' target.', tm_key)
    elif m_key == 'embedding':
      central_frame_init_embedding(params, m_params, view_idx, config)
    else:
      logging.info('Skipping `%s`, in restored model but not in target', m_key)


def initialize_from_vit_train_states(
    config: ml_collections.ConfigDict,
    train_state: train_utils.TrainState,
    restored_train_states: Sequence[train_utils.TrainState],
    restored_model_cfgs: Sequence[ml_collections.ConfigDict],
    restored_model_formats: Sequence[str],
) -> train_utils.TrainState:
  """Updates MTV's train_state with pretrained ViT weights.

  Args:
    config: Configurations for the model being updated.
    train_state: A raw TrainState for the model.
    restored_train_states: A dict. Each key is a Ti/S/B/L ViT model and the
      corresponding value is a TrainState that is loaded with parameters/state
      of pretrained models.
    restored_model_cfgs: Configurations of models from which the
      restored_train_states come from. Often only the classifier information is
      used for interpolating the positional embeddings.
    restored_model_formats: A list of pretrained model formats. The format can
      only be `big_vision` or 'scenic'.

  Returns:
    Updated train_state.
  """
  assert len(restored_train_states) == len(restored_model_formats), (
      'restored_train_states must have the same dimension as '
      'restored_model_formats.')
  params = flax.core.unfreeze(train_state.params)
  for view_idx, restored_state in enumerate(restored_train_states):
    restored_model_params = flax.core.unfreeze(restored_state.params)
    initialize_one_view_from_vit_parameters(config, params,
                                            restored_model_cfgs[view_idx],
                                            restored_model_params, view_idx)

  return train_state.replace(params=flax.core.freeze(params))
