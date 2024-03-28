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

"""Contains model utilities."""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
from absl import logging
import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.projects.vivit import model_utils as vivit_utils
from scenic.train_lib import train_utils

PyModule = Any
Array = Union[jnp.ndarray, np.ndarray]


def extract_pyramid_video_tokens(tokens: jnp.ndarray, num_pyramid_levels: int,
                                 feature_pyramid_downsample_stride: int,
                                 num_video_tokens_level0: int,
                                 num_text_tokens: int) -> jnp.ndarray:
  """Extracts video tokens from a feature pyramid, removing text tokens.

  Args:
    tokens: Concatenated tokens from each pyramid level in shape (batch,
      seq_len_level0 + seq_len_level1 + ..., channels). This include the text
      tokens.
    num_pyramid_levels: Number of feature pyramid levels.
    feature_pyramid_downsample_stride: Downsample stride used to create the
      feature pyramid.
    num_video_tokens_level0: Number of video tokens in the first level of
      feature pyramid.
    num_text_tokens: Number fo text tokens.

  Returns:
    Concatenated video tokens from each pyramid level in shape (batch,
      seq_len_level0 + seq_len_level1 + ..., channels).
  """

  num_video_tokens_per_level = [
      (num_video_tokens_level0 // (feature_pyramid_downsample_stride**idx))
      for idx in range(num_pyramid_levels)
  ]
  tokens_per_level = split_pyramid_features(
      tokens,
      num_video_tokens_level0,
      num_pyramid_levels,
      feature_pyramid_downsample_stride,
      num_text_tokens,
      np_backend=jnp)
  video_tokens_per_level = []
  for n, t in zip(num_video_tokens_per_level, tokens_per_level):
    video_tokens_per_level.append(t[:, :n])
  return jnp.concatenate(video_tokens_per_level, axis=1)


def create_pyramid_split_indices(
    num_features_level0: int,
    num_pyramid_levels: int,
    feature_pyramid_downsample_stride: int,
    num_extra_features_per_level: int = 0,
):
  """Generates split indices for each pyramid level."""
  num_features_per_level = [
      (num_features_level0 //
       (feature_pyramid_downsample_stride**idx)) + num_extra_features_per_level
      for idx in range(num_pyramid_levels)
  ]
  return np.cumsum(num_features_per_level)[:-1]


def split_pyramid_features(
    tokens: Array,
    num_features_level0: int,
    num_pyramid_levels: int,
    feature_pyramid_downsample_stride: int,
    num_extra_features_per_level: int = 0,
    axis: int = 1,
    np_backend: PyModule = jnp,
):
  """Split tokens into a list based on pyramid levels."""
  indices = create_pyramid_split_indices(num_features_level0,
                                         num_pyramid_levels,
                                         feature_pyramid_downsample_stride,
                                         num_extra_features_per_level)
  return np_backend.split(tokens, indices, axis=axis)


def create_pyramid_input_masks(input_mask: jnp.ndarray,
                               num_features_level0: int,
                               num_pyramid_levels: int,
                               feature_pyramid_downsample_stride: int,
                               num_text_tokens: int) -> List[jnp.ndarray]:
  """Splits input mask into a list of them based on pyramid levels."""
  split_indices = create_pyramid_split_indices(
      num_features_level0,
      num_pyramid_levels,
      feature_pyramid_downsample_stride,
      num_extra_features_per_level=num_text_tokens)
  return np.split(input_mask, split_indices, axis=1)


def merge_pyramid_input_masks(
    input_masks: Sequence[jnp.ndarray],
    input_text_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Merges input masks from different pyramid levels into one."""
  if input_text_mask is None:
    return jnp.concatenate(input_masks, axis=1)
  all_masks = []
  for mask in input_masks:
    all_masks.append(jnp.concatenate([mask, input_text_mask], axis=1))
  return jnp.concatenate(all_masks, axis=1)


def extract_emb(x: jnp.ndarray,
                classifier: str,
                keepdims: bool = False,
                input_mask: Optional[jnp.ndarray] = None,
                input_word_ids: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Extracts an embedding from a sequence of tokens.

  Args:
    x: A 3D float tensor of shape (batch_size, sequence_length, channels)
      representing the tokens where we extract the embedding. If x is 2D, we
      return as it is.
    classifier: 'token', 'eos', or 'gap'.
    keepdims: Whether or not to make the output embedding as the same dimension
      as the input tokens.
    input_mask: Optional. A 2D binary tensor of shape (batch_size,
      sequence_length) representing the input mask. This arg is only used when
      classifier = 'gap'.
    input_word_ids: Optional. A 2D binary tensor of shape (batch_size,
      sequence_length) representing the input word ids. This arg is only used
      when classifier = 'eos'. 'eos' tokens are assumed to have the largest ids.

  Returns:
    A 3D float tensor of shape (batch_size, 1, channels) if
    keepdims=True or a 2D float tensor of shape (batch_size, channels) if
    keepdims=False.
  """
  if x.ndim == 2:
    logging.info('Input is 2D and return as it is.')
    return x
  if classifier == 'token':
    return x[:, :1] if keepdims else x[:, 0]
  elif classifier == 'eos' and input_word_ids is not None:
    x = x[jnp.arange(x.shape[0]), input_word_ids.argmax(-1)]
    if keepdims:
      x = jnp.expand_dims(x, axis=1)
    return x
  else:
    if input_mask is not None:
      return jnp.sum(
          jnp.multiply(x, jnp.expand_dims(input_mask, axis=-1)),
          axis=1,
          keepdims=keepdims) / jnp.expand_dims(
              input_mask.sum(axis=-1, keepdims=keepdims), axis=-1)
    return jnp.mean(x, axis=1, keepdims=keepdims)


def l2_normalize(x: jnp.ndarray, eps: float = 1e-9) -> jnp.ndarray:
  """Normalizes along dimension `axis` using an L2 norm."""
  return x * jax.lax.rsqrt((x * x).sum(axis=-1, keepdims=True) + eps)


def init_text_posemb(
    to_posemb: jnp.ndarray,
    from_posemb: jnp.ndarray,
) -> jnp.ndarray:
  """Clips or pads positional embeddings.

  This function is used to adjust positional embeddings for text encoders when
  the current model has a different sequence length from the pretrained model.
  When the current model uses a shorter sequence length, we clip the restored
  positional embeddings to match the new length. When the current model uses a
  longer sequence, we copy the first N values from the restored embeddings where
  N is the length of restored positional embeddings.

  Note that interpolation is another possibility here.

  Args:
    to_posemb: The current positional embedding.
    from_posemb: The positional embedding from the pretrained model.

  Returns:
    The adjusted positional embedding.
  """
  to_seq_len = to_posemb.shape[0]
  from_seq_len = from_posemb.shape[0]
  if to_seq_len < from_seq_len:
    logging.info('Clip positional embedding to be a length of %s', to_seq_len)
    return from_posemb[:to_seq_len, :]
  elif to_seq_len > from_seq_len:
    logging.warning(
        'The current sequence length is longer than the restored model. Only '
        'the first %s elements are initialized.', from_seq_len)
    return jnp.concatenate([from_posemb, to_posemb[from_seq_len:, :]], axis=0)
  else:
    return from_posemb


def initialize_text_encoder_from_clip_params(
    params: Dict[str, Any],
    restored_params: Mapping[str, Any],
    load_projection: bool = False,
    model_prefix_path: Optional[List[str]] = None,
):
  """Initialize text encoder from a CLIP model."""
  if model_prefix_path:
    to_params = params[model_prefix_path[0]]
    for prefix in model_prefix_path[1:]:
      to_params = to_params[prefix]
  else:
    to_params = params
  for m_key, m_params in restored_params.items():
    if m_key == 'positional_embedding':
      to_params[m_key] = init_text_posemb(to_params[m_key], m_params)
    elif m_key == 'text_projection':
      if load_projection:
        to_params[m_key] = m_params
    else:
      logging.info('Loading `%s` from checkpoint.', m_key)
      to_params[m_key] = m_params


def init_class_embedding(to_params: Dict[str, Any],
                         class_embedding: jnp.ndarray):
  """Initialize class embedding.

  The class embedding of the current model has a shape of (num_frames,
  channels).

  Args:
    to_params: Params of the current model.
    class_embedding: A 1D float tensor of shape (channels,) representing the
      class embedding from the pretrained model.
  """
  num_frames = to_params['class_embedding'].shape[0]
  to_params['class_embedding'] = jnp.tile(
      jnp.expand_dims(class_embedding, axis=0), [num_frames, 1])


def init_posembed(restored_posemb: jnp.ndarray, posemb: jnp.ndarray,
                  restored_classifier: str, classifier: str):
  """Initialize positional embedding."""
  if restored_posemb.shape != posemb.shape:
    logging.info('Adapting positional embeddings from %s to %s',
                 restored_posemb.shape, posemb.shape)
    ntok = posemb.shape[0]
    if restored_classifier == 'token':
      # The first token is the CLS token.
      restored_posemb_grid = restored_posemb[1:, :]
      if classifier == 'token':
        # CLS token in restored model and in target.
        cls_tok = restored_posemb[:1]
        ntok -= 1
      else:
        # CLS token in restored model, but not target.
        cls_tok = restored_posemb[:0]
    else:
      restored_posemb_grid = restored_posemb
      if classifier == 'token':
        # CLS token in target, but not restored model.
        cls_tok = posemb[:1]
        ntok -= 1
      else:
        # CLS token not in target or restored model.
        cls_tok = restored_posemb[:0]
    restored_posemb_grid = vivit_utils.interpolate_positional_embeddings(
        restored_posemb_grid, ntok)[0]  # Squeeze first dimension.
    # Attach the CLS token again.
    if classifier == 'token':
      restored_posemb = jnp.array(
          np.concatenate([cls_tok, restored_posemb_grid], axis=0))
    else:
      restored_posemb = restored_posemb_grid

  return restored_posemb


def init_conv1(from_conv1: Dict[str, Any], to_conv1: Dict[str, Any]):
  """Initialize the first 3D conv parameters.

  Initializes the first 3D conv layer from an image model.

  Args:
    from_conv1: The 2D conv weights from a pretrained model.
      from_conv1['kernel'] has a shape of (h, w, in_channels, out_channels).
    to_conv1: The linear projection weights from the current model.
      to_conv1['kernel'] has a shape of (h*w*in_channels, out_channels).
  """
  input_kernel = to_conv1['kernel']
  restored_kernel = from_conv1['kernel']
  if input_kernel.shape[0] != np.prod(restored_kernel.shape[:-1]):
    raise ValueError(
        'conv1 kernel shapes mismatch during initialization. from_conv1: %s and'
        'to_conv1: %s' % (restored_kernel.shape, input_kernel.shape)
    )
  to_conv1['kernel'] = jnp.reshape(restored_kernel, input_kernel.shape)


def initialize_video_encoder_from_clip_params(
    config: ml_collections.ConfigDict,
    params: Dict[str, Any],
    restored_params: Mapping[str, Any],
    load_projection: bool = False,
    video_modality_name: str = 'video',
    model_prefix_path: Optional[List[str]] = None,
):
  """Initialize video encoder from a CLIP model."""
  if model_prefix_path:
    to_params = params[model_prefix_path[0]]
    for prefix in model_prefix_path[1:]:
      to_params = to_params[prefix]
  else:
    to_params = params
  if config.model.video_tower_config.get('modality_configs'):
    classifier = config.model.video_tower_config.modality_configs[
        video_modality_name
    ].encoder_config.image_encoder_config.classifier
  else:  # backward compatibility for single
    classifier = (
        config.model.video_tower_config.encoder_config.image_encoder_config.classifier
    )
  for m_key, m_params in restored_params.items():
    if m_key == 'class_embedding':
      if 'class_embedding' in to_params:
        init_class_embedding(to_params, m_params)
    elif m_key == 'conv1':
      init_conv1(m_params, to_params['conv1'])
    elif m_key == 'positional_embedding':
      to_params['VisionTransformer'][m_key] = init_posembed(
          m_params,
          to_params['VisionTransformer'][m_key],
          restored_classifier='token',
          classifier=classifier,
      )
    elif m_key == 'proj':
      if load_projection:
        to_params['proj'] = m_params
    else:
      to_params['VisionTransformer'][m_key] = m_params


def initialize_from_clip_model(
    config: ml_collections.ConfigDict,
    train_state: train_utils.TrainState,
    restored_params: Dict[str, Any],
    load_image_tower: bool = True,
    load_text_tower: bool = True,
    video_modality_name: str = 'video',
    text_encoder_name: str = 'text_encoder',
) -> train_utils.TrainState:
  """Initializes a video-text model from an pretrained image-text model."""
  params = flax.core.unfreeze(train_state.params)
  if load_image_tower:
    if config.init_from.get('video_encoder'):
      load_image_tower_projection = config.init_from.video_encoder.get(
          'load_projection'
      ) and config.model.video_tower_config.get('projection_size')
    else:
      load_image_tower_projection = config.init_from.video_encoders[
          video_modality_name
      ].get(
          'load_projection'
      ) and config.model.video_tower_config.modality_configs[
          video_modality_name
      ].get(
          'projection_size'
      )
    initialize_video_encoder_from_clip_params(
        config,
        params,
        restored_params['params']['visual'],
        load_projection=False,
        video_modality_name=video_modality_name,
        model_prefix_path=[f'{video_modality_name}_encoder'],
    )
    if load_image_tower_projection:
      params[f'{video_modality_name}_projection'] = restored_params['params'][
          'visual'
      ]['proj']
  if load_text_tower:
    load_text_tower_projection = (
        config.init_from.text_encoder.get('load_projection') and
        config.model.text_tower_config.get('projection_size'))
    initialize_text_encoder_from_clip_params(
        params,
        restored_params['params']['text'],
        load_projection=False,
        model_prefix_path=[text_encoder_name],
    )
    if load_text_tower_projection:
      params['text_projection'] = restored_params['params']['text'][
          'text_projection']
  return train_state.replace(params=flax.core.freeze(params))
