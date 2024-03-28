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

"""Contains video-text fusion modules."""

import functools
from typing import Callable, List, Optional, Sequence, Tuple
from absl import logging
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.layers import attention_layers
from scenic.projects.baselines import vit
from scenic.projects.unloc import encoders
from scenic.projects.unloc import model_utils

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


def prepend_cls(x: jnp.ndarray, cls: jnp.ndarray) -> jnp.ndarray:
  """Prepend a CLS token."""
  assert x.ndim == 3 and cls.ndim == 3
  n, _, _ = x.shape
  cls = jnp.tile(cls, [n, 1, 1])
  return jnp.concatenate([cls, x], axis=1)


def append_one(input_mask: jnp.ndarray) -> jnp.ndarray:
  """Appends ones to the input mask.

  Args:
    input_mask: Mask assumed to be of shape [batch, tokens, ...].

  Returns:
    Input mask appended with one in shape [batch, tokens+1, ...].
  """
  return jnp.concatenate(
      [input_mask,
       jnp.ones((input_mask.shape[0], 1), dtype=input_mask.dtype)],
      axis=1)


def prepend_one(input_mask: jnp.ndarray) -> jnp.ndarray:
  """Prepends ones to the input mask.

  Args:
    input_mask: Mask assumed to be of shape [batch, tokens, ...].

  Returns:
    Input mask prepended with one in shape [batch, 1+tokens, ...].
  """
  return jnp.concatenate(
      [jnp.ones((input_mask.shape[0], 1), dtype=input_mask.dtype), input_mask],
      axis=1)


class FeaturePyramidEncoder(nn.Module):
  """Transformer feature pyramid encoder.

  Attributes:
    num_layers: Number of layers.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of attention heads.
    feature_pyramid_config: Feature pyramid config.
    downsample_strategy: Strategy to downsample video tokens. Options are:
      'subsample', 'avg_pool', or 'max_pool'.
    positional_embedding: 'learned', 'sinusoid', or 'none'.
    positional_embedding_max_length: If set, the positional embeddings/encodings
      are applied to the first N elements in the input sequence. If not set, the
      positional embeddings/encodings are added to the entire input sequence.
    dropout_rate: Dropout rate.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value. Our implementation of stochastic depth follows the
      timm library, which does per-example layer dropping and uses independent
      dropping patterns for each skip-connection.
    window_size: Window size for window attention blocks.
    window_block_indexes: Tuple. Indexes for blocks using window attention.
    dtype: Dtype of activations.
  """
  num_layers: int
  mlp_dim: int
  num_heads: int
  feature_pyramid_config: ml_collections.ConfigDict
  downsample_strategy: str = 'max_pool'
  positional_embedding: str = 'learned'
  positional_embedding_max_length: Optional[int] = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  window_size: int = 0
  window_block_indexes: Sequence[int] = (0, 1, 3, 4)
  dtype: jnp.dtype = jnp.float32

  def _add_positional_embedding(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Adds positional embedding."""
    posemb = jnp.zeros_like(inputs)
    if self.positional_embedding == 'learned':
      posemb = vit.AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='posembed_input')(
              posemb)
    elif self.positional_embedding == 'sinusoid':
      posemb = attention_layers.Add1DPositionEmbedding(posemb_init=None)(posemb)
    elif self.positional_embedding == 'none':
      logging.info('No positional embedding is used.')
    else:
      raise ValueError(
          f'Invalid positional_embedding: {self.positional_embedding}.')

    if self.positional_embedding_max_length is not None:
      max_len = min(inputs.shape[1], self.positional_embedding_max_length)
      inputs = inputs.at[:, :max_len].add(posemb[:, :max_len])
    else:
      inputs += posemb
    return inputs

  def _subsample(self, x: jnp.ndarray,
                 sampled_indices: np.ndarray) -> jnp.ndarray:
    return x[:, sampled_indices]

  def _max_pool(self, x: jnp.ndarray) -> jnp.ndarray:
    stride = self.feature_pyramid_config.feature_pyramid_downsample_stride
    return nn.max_pool(x, window_shape=(stride,), strides=(stride,))

  def _avg_pool(self, x: jnp.ndarray) -> jnp.ndarray:
    stride = self.feature_pyramid_config.feature_pyramid_downsample_stride
    return nn.avg_pool(x, window_shape=(stride,), strides=(stride,))

  def _strided_depthwise_conv(self, x: jnp.ndarray) -> jnp.ndarray:
    stride = self.feature_pyramid_config.feature_pyramid_downsample_stride
    return nn.Conv(
        features=x.shape[-1],
        kernel_size=(3,),
        strides=(stride,),
        feature_group_count=x.shape[-1],
    )(x)

  def _strided_conv(self, x: jnp.ndarray) -> jnp.ndarray:
    stride = self.feature_pyramid_config.feature_pyramid_downsample_stride
    return nn.Conv(features=x.shape[-1], kernel_size=(3,), strides=(stride,))(x)

  def _downsample_video_tokens(self, x: jnp.ndarray, num_text_tokens: int,
                               sampled_indices: np.ndarray) -> jnp.ndarray:
    """Subsamples video tokens."""

    downsample_fns = {
        'max_pool': self._max_pool,
        'avg_pool': self._avg_pool,
        'strided_depthwise_conv': self._strided_depthwise_conv,
        'strided_conv': self._strided_conv,
        'subsample': functools.partial(
            self._subsample, sampled_indices=sampled_indices
        ),
    }
    if num_text_tokens > 0:
      text_tokens = x[:, -num_text_tokens:]
      y = downsample_fns[self.downsample_strategy](x[:, :-num_text_tokens])
      x = jnp.concatenate([y, text_tokens], axis=1)
    else:
      x = downsample_fns[self.downsample_strategy](x)
    return x

  def _build_top_down_path(self, xs: List[jnp.ndarray]) -> List[jnp.ndarray]:
    """Pass information from top level to bottom in a feature pyramid."""

    for idx in range(len(xs) - 1, 0, -1):
      x = jax.image.resize(
          xs[idx], xs[idx - 1].shape, method='nearest', antialias=False)
      xs[idx - 1] += x
    return xs

  def _apply_output_conv(self, xs: List[jnp.ndarray]) -> List[jnp.ndarray]:
    """Apply depthwise convolutions and layer norm."""
    for idx, x in enumerate(xs):
      # Depthwise convolution.
      y = nn.Conv(
          features=x.shape[-1],
          kernel_size=(3,),
          feature_group_count=x.shape[-1],
          name=f'output_conv_{idx}')(
              x)
      xs[idx] = nn.LayerNorm(name=f'output_ln_{idx}')(y)
    return xs

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               input_mask: Optional[jnp.ndarray] = None,
               train: bool = False):
    """Applies Transformer model on the inputs."""

    num_pyramid_levels = len(self.feature_pyramid_config.feature_pyramid_levels)
    assert self.num_layers >= num_pyramid_levels
    assert inputs.ndim == 3  # Shape is `[batch, len, emb]`.
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)
    x = self._add_positional_embedding(inputs)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    num_text_tokens = (
        inputs.shape[1] - self.feature_pyramid_config.num_features_level0)
    if input_mask is not None:
      input_mask_per_level = model_utils.create_pyramid_input_masks(
          input_mask,
          num_features_level0=self.feature_pyramid_config.num_features_level0,
          num_pyramid_levels=num_pyramid_levels,
          feature_pyramid_downsample_stride=self.feature_pyramid_config
          .feature_pyramid_downsample_stride,
          num_text_tokens=num_text_tokens)
    else:
      input_mask_per_level = [None] * num_pyramid_levels

    cur_pyramid_level = 0
    xs = []
    for lyr in range(self.num_layers):
      x = encoders.TransformerEncoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_depth=(lyr / max(self.num_layers - 1, 1))
          * self.stochastic_depth,
          name=f'encoderblock_{lyr}',
          dtype=dtype,
      )(
          x,
          input_mask=input_mask_per_level[cur_pyramid_level],
          deterministic=not train,
      )
      if lyr in self.feature_pyramid_config.feature_pyramid_levels:
        # Convolution only applies to video tokens.
        if num_text_tokens:
          xs.append(x[:, :-num_text_tokens])
        else:
          xs.append(x)
        cur_pyramid_level += 1
        if (self.feature_pyramid_config.feature_pyramid_downsample_stride > 1
            and lyr < self.num_layers - 1):
          sampled_indices = np.arange(
              0,
              self.feature_pyramid_config.num_features_level0,
              self.feature_pyramid_config.feature_pyramid_downsample_stride**
              cur_pyramid_level,
              dtype=np.int32)
          x = self._downsample_video_tokens(
              x, num_text_tokens, sampled_indices=sampled_indices)
    if num_text_tokens:
      text_tokens = x[:, -num_text_tokens:]
    else:
      # No text tokens.
      text_tokens = x[:, :0]
    xs = self._build_top_down_path(xs)
    xs = self._apply_output_conv(xs)
    xs.append(text_tokens)
    return jnp.concatenate(xs, axis=1)


class SimplePyramidEncoder(FeaturePyramidEncoder):
  """A simple pyramid transformer encoder.

  This structure is inspired by ViTDet (https://arxiv.org/abs/2203.16527). The
  feature pyramid is built using the output from the last layer in the encoder
  and downsampling is performed via strided depthwise convolution. This simple
  design allows us to share the same architecture as the one used in
  classification task.
  """

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      input_mask: Optional[jnp.ndarray] = None,
      train: bool = False,
  ):
    """Applies Transformer model on the inputs."""

    num_pyramid_levels = len(self.feature_pyramid_config.feature_pyramid_levels)
    assert self.num_layers >= num_pyramid_levels
    assert inputs.ndim == 3  # Shape is `[batch, len, emb]`.
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)
    x = self._add_positional_embedding(inputs)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    num_text_tokens = (
        inputs.shape[1] - self.feature_pyramid_config.num_features_level0
    )
    if input_mask is not None:
      input_mask_per_level = model_utils.create_pyramid_input_masks(
          input_mask,
          num_features_level0=self.feature_pyramid_config.num_features_level0,
          num_pyramid_levels=num_pyramid_levels,
          feature_pyramid_downsample_stride=(
              self.feature_pyramid_config.feature_pyramid_downsample_stride
          ),
          num_text_tokens=num_text_tokens,
      )
      # No downsampling is done in the encoder.
      input_mask = input_mask_per_level[0]

    for lyr in range(self.num_layers):
      x = encoders.TransformerEncoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_depth=(lyr / max(self.num_layers - 1, 1))
          * self.stochastic_depth,
          name=f'encoderblock_{lyr}',
          dtype=dtype,
      )(
          x,
          input_mask=input_mask,
          deterministic=not train,
      )
    if num_text_tokens:
      text_tokens = x[:, -num_text_tokens:]
    else:
      # No text tokens.
      text_tokens = x[:, :0]
    # Convolution only applies to video tokens.
    x = x[:, : self.feature_pyramid_config.num_features_level0]
    xs = []
    for lyr in range(num_pyramid_levels):
      stride = (
          1
          if lyr == 0
          else self.feature_pyramid_config.feature_pyramid_downsample_stride
      )
      x = nn.Conv(
          features=x.shape[-1],
          kernel_size=(3,),
          strides=(stride,),
          feature_group_count=x.shape[-1],  # Depthwise convolution
          name=f'output_conv_{lyr}',
      )(x)
      x = nn.LayerNorm(name=f'output_ln_{lyr}')(x)
      xs.append(x)
    xs.append(text_tokens)
    return jnp.concatenate(xs, axis=1)


_VIDEO_TEXT_ENCODER = {
    'transformer': encoders.TransformerEncoder,
    'fpn': FeaturePyramidEncoder,
    'simple_pyramid': SimplePyramidEncoder,
}


class VideoTextEmbSelfAttentionFusion(nn.Module):
  """Implements video-text fusion by self attention.

  We append the text CLS token to the video tokens and then feed the
  concatenated sequence into a Transformer.

  Attributes:
    self_attention_encoder_config: The config of the self attention encoder.
    self_attention_encoder_name: The type of self attention encoder.
  """

  self_attention_encoder_config: ml_collections.ConfigDict
  self_attention_encoder_name: str = 'transformer'  # or 'fpn'

  def _self_attention_encode_per_text_emb(
      self, video_tokens: jnp.ndarray, video_input_mask: jnp.ndarray,
      text_emb: jnp.ndarray, encoder: nn.Module, task: str,
      train: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Fuses video and one text embedding via self attention.

    Args:
      video_tokens: A 3D float tensor of shape (batch_size, sequence_length,
        channels) representing the video tokens.
      video_input_mask: A 2D binary tensor of shape (batch_size,
        sequence_length).
      text_emb: A 1D float tensor of shape (channels,) representing the text
        embedding.
      encoder: The Transformer encoder.
      task: 'action_segmentation', 'temporal_localization', or
        'moment_retrieval'.
      train: Whether or not the model is under training.

    Returns:
      Video tokens in shape (batch size, sequence_length, channels) and text
      token in shape (batch_size, channels).
    """
    assert task in {
        'action_segmentation',
        'moment_retrieval',
        'temporal_localization',
    }
    tiled_text_emb = jnp.tile(text_emb[None, None, :],
                              [video_tokens.shape[0], 1, 1])
    tokens = jnp.concatenate([
        video_tokens,
        tiled_text_emb,
    ], axis=1)
    feature_pyramid_config = self.self_attention_encoder_config.get(
        'feature_pyramid_config')
    if feature_pyramid_config is None:
      video_input_masks = [video_input_mask]
    else:
      video_input_masks = model_utils.create_pyramid_input_masks(
          video_input_mask,
          num_features_level0=feature_pyramid_config.num_features_level0,
          num_pyramid_levels=len(feature_pyramid_config.feature_pyramid_levels),
          feature_pyramid_downsample_stride=feature_pyramid_config
          .feature_pyramid_downsample_stride,
          num_text_tokens=0)
    input_mask = model_utils.merge_pyramid_input_masks(
        video_input_masks,
        input_text_mask=jnp.ones((video_tokens.shape[0], 1), dtype=jnp.int32))
    tokens = encoder(tokens, input_mask=input_mask, train=train)
    return tokens[:, :-1], tokens[:, -1]

  @nn.compact
  def __call__(self,
               video_tokens: jnp.ndarray,
               text_embs: jnp.ndarray,
               task: str,
               input_word_ids: Optional[jnp.ndarray] = None,
               text_input_mask: Optional[jnp.ndarray] = None,
               video_input_mask: Optional[jnp.ndarray] = None,
               train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Fuses video text by self attention.

    Args:
      video_tokens: A 3D float tensor of shape (batch_size, sequence_length,
        channels) representing the video tokens.
      text_embs: A 2D float tensor of shape (num_classes or batch_size *
        max_num_captions, channels) representing the text CLS token for each
        class. The first dimension is batch_size * max_num_captions if task =
        `moment_retrieval`. Otherwise, it is num_classes.
      task: 'action_segmentation', 'temporal_localization',
        'highlight_detection', or 'moment_retrieval'.
      input_word_ids: None.
      text_input_mask: None.
      video_input_mask: None or a 2D binary tensor of shape (batch_size,
        sequence_length).
      train: Whether or not it is in training.

    Returns:
      A 4D float tensor of shape (batch_size, batch_size * max_num_captions,
      sequence_length, channels) representing the pre_logits for each class if
      task = `moment_retrieval`.

      A 4D float tensor of shape (batch_size, num_classes, sequence_length,
      channels) representing the pre_logits for each class if task is
      `temporal_localization`.

      Video tokens in shape (batch_size, sequence_length, num_classes,
      channels) and text tokens in shape (batch_size, num_classes, channels) if
      task is `action_segmentation`.
    """
    encoder = _VIDEO_TEXT_ENCODER[self.self_attention_encoder_name](
        name='video_text_encoder', **self.self_attention_encoder_config)
    if video_input_mask is None:
      feature_pyramid_config = self.self_attention_encoder_config.get(
          'feature_pyramid_config')
      if feature_pyramid_config is None:
        video_input_mask = jnp.ones(video_tokens.shape[:2], dtype=jnp.int32)
      else:
        feature_pyramid_levels = feature_pyramid_config.feature_pyramid_levels
        num_features_level0 = feature_pyramid_config.num_features_level0
        feature_pyramid_downsample_stride = (
            feature_pyramid_config.feature_pyramid_downsample_stride
        )
        fpn_video_token_len = sum([
            num_features_level0 // (feature_pyramid_downsample_stride**idx)
            for idx in range(len(feature_pyramid_levels))
        ])
        video_input_mask = jnp.ones(
            (video_tokens.shape[0], fpn_video_token_len), dtype=jnp.int32)
    video_tokens, text_embs = jax.vmap(
        functools.partial(
            self._self_attention_encode_per_text_emb,
            encoder=encoder,
            task=task,
            train=train,
        ),
        in_axes=[None, None, 0],
    )(video_tokens, video_input_mask, text_embs)
    if task == 'temporal_localization':
      # Converts video_tokens from (num_classes, batch_size, num_frames,
      # channels) to (batch_size, num_classes, num_frames, channels).
      return (
          jnp.transpose(video_tokens, [1, 0, 2, 3]),
          jnp.transpose(text_embs, [1, 0, 2]),
      )
    elif task == 'action_segmentation':
      # Converts video_tokens from (num_classes, batch_size, num_frames,
      # channels) to (batch_size, num_frames, num_classes, channels).
      return (
          jnp.transpose(video_tokens, [1, 2, 0, 3]),
          jnp.transpose(text_embs, [1, 0, 2]),
      )
    elif task == 'moment_retrieval':
      # Converts video_toekns from (batch_size * max_num_captions, batch_size,
      # num_frames, channels) to (batch_size, batch_size * max_num_captions,
      # num_frames, channels).
      return (
          jnp.transpose(video_tokens, [1, 0, 2, 3]),
          jnp.transpose(text_embs, [1, 0, 2]),
      )
    else:
      raise ValueError(f'Unexpected task `{task}`.')


class VideoTextSelfAttentionFusion(nn.Module):
  """Implements video-text fusion by self attention.

  We concatenate all text tokens (or only the CLS token) with video tokens and
  then feed the concatenated sequence into a Transformer.

  Attributes:
    text_tower_classifier: 'token' (take the first token), 'eos' (take the last
    token).
    self_attention_encoder_config: The config of the self attention encoder.
    use_all_text_tokens: Whether or not to fuse with all text tokens. If False,
      we only fuse with the text CLS token.
  """

  text_tower_classifier: str
  self_attention_encoder_config: ml_collections.ConfigDict
  use_all_text_tokens: bool
  self_attention_encoder_name: str = 'transformer'  # or 'fpn'

  def _self_attention_encode_all_video_text_pairs(
      self,
      video_tokens: jnp.ndarray,
      video_input_mask: jnp.ndarray,
      text_tokens: jnp.ndarray,
      input_word_ids: jnp.ndarray,
      text_input_mask: jnp.ndarray,
      encoder: nn.Module,
      train: bool,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Fuses all video-text pairs via self attention.

    Args:
      video_tokens: A 3D float tensor of shape (batch_size, sequence_length,
        channels) representing the video tokens.
      video_input_mask: A 2D binary tensor of shape (batch_size,
        sequence_length).
      text_tokens: A 3D float tensor of shape (batch_size, sequence_length,
        channels) representing the text tokens.
      input_word_ids: A 2D int tensor of shape (batch_size, sequence_length)
        representing the input word indices.
      text_input_mask: A 2D binary tensor of shape (batch_size, sequence_length)
        representing the mask of the text inputs.
      encoder: The Transformer encoder.
      train: Whether or not the model is under training.

    Returns:
      video_tokens: A 4D float tensor of shape (batch size, 1, sequence_length,
        channels) representing the fused frame embedding.
      text_emb: a 3D float tensor of shape (batch_size, 1, channels)
        representing the text CLS token.
    """

    feature_pyramid_config = self.self_attention_encoder_config.get(
        'feature_pyramid_config'
    )
    if feature_pyramid_config is None:
      video_input_masks = [video_input_mask]
    else:
      video_input_masks = model_utils.create_pyramid_input_masks(
          video_input_mask,
          num_features_level0=feature_pyramid_config.num_features_level0,
          num_pyramid_levels=len(feature_pyramid_config.feature_pyramid_levels),
          feature_pyramid_downsample_stride=feature_pyramid_config.feature_pyramid_downsample_stride,
          num_text_tokens=0,
      )
    if self.use_all_text_tokens:
      tokens = jnp.concatenate([video_tokens, text_tokens], axis=1)
      input_mask = model_utils.merge_pyramid_input_masks(
          video_input_masks, input_text_mask=text_input_mask
      )
      num_text_tokens = text_tokens.shape[1]
    else:
      tokens = jnp.concatenate(
          [
              video_tokens,
              model_utils.extract_emb(
                  text_tokens,
                  self.text_tower_classifier,
                  keepdims=True,
                  input_mask=text_input_mask,
                  input_word_ids=input_word_ids,
              ),
          ],
          axis=1,
      )
      input_mask = model_utils.merge_pyramid_input_masks(
          video_input_masks,
          input_text_mask=jnp.ones((video_tokens.shape[0], 1), dtype=jnp.int32),
      )
      num_text_tokens = 1

    tokens = encoder(tokens, input_mask=input_mask, train=train)
    if self.use_all_text_tokens:
      text_emb = model_utils.extract_emb(
          tokens[:, -num_text_tokens:],
          self.text_tower_classifier,
          keepdims=False,
          input_mask=text_input_mask,
          input_word_ids=input_word_ids,
      )
    else:
      text_emb = tokens[:, -1]
    video_tokens = tokens[:, :-num_text_tokens]
    return (
        jnp.expand_dims(video_tokens, axis=1),
        jnp.expand_dims(text_emb, axis=1),
    )

  def _self_attention_encode_per_text(
      self, video_tokens: jnp.ndarray, video_input_mask: jnp.ndarray,
      text_tokens: jnp.ndarray, input_word_ids: jnp.ndarray,
      text_input_mask: jnp.ndarray, encoder: nn.Module, task: str,
      train: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Fuses video and text tokens per class via self attention.

    Args:
      video_tokens: A 3D float tensor of shape (batch_size, sequence_length,
        channels) representing the video tokens.
      video_input_mask: A 2D binary tensor of shape (batch_size,
        sequence_length).
      text_tokens: A 2D float tensor of shape (sequence_length, channels)
        representing the text tokens.
      input_word_ids: A 1D int tensor of shape (sequence_length) representing
        the input word indices.
      text_input_mask: A 1D binary tensor of shape (sequence_length)
        representing the mask of the text inputs.
      encoder: The Transformer encoder.
      task: 'action_segmentation', 'moment_retrieval', 'highlight_detection', or
        'temporal_localization'.
      train: Whether or not the model is under training.

    Returns:
      video_tokens: A 3D float tensor of shape (batch size, sequence_length,
        channels) representing the fused frame embedding.
      text_emb: A 2D float tensor of shape (batch_size, channels) representing
        the text CLS token.
    """
    tiled_text_tokens = jnp.tile(
        jnp.expand_dims(text_tokens, axis=0), [video_tokens.shape[0], 1, 1])
    tiled_text_mask = jnp.tile(
        jnp.expand_dims(text_input_mask, axis=0), [video_tokens.shape[0], 1])
    tiled_input_word_ids = jnp.tile(
        jnp.expand_dims(input_word_ids, axis=0), [video_tokens.shape[0], 1])
    feature_pyramid_config = self.self_attention_encoder_config.get(
        'feature_pyramid_config')
    if feature_pyramid_config is None:
      video_input_masks = [video_input_mask]
    else:
      video_input_masks = model_utils.create_pyramid_input_masks(
          video_input_mask,
          num_features_level0=feature_pyramid_config.num_features_level0,
          num_pyramid_levels=len(feature_pyramid_config.feature_pyramid_levels),
          feature_pyramid_downsample_stride=feature_pyramid_config
          .feature_pyramid_downsample_stride,
          num_text_tokens=0)
    if self.use_all_text_tokens:
      tokens = jnp.concatenate([video_tokens, tiled_text_tokens], axis=1)
      input_mask = model_utils.merge_pyramid_input_masks(
          video_input_masks,
          input_text_mask=tiled_text_mask)
      num_text_tokens = text_tokens.shape[0]
    else:
      tokens = jnp.concatenate([
          video_tokens,
          model_utils.extract_emb(
              tiled_text_tokens,
              self.text_tower_classifier,
              keepdims=True,
              input_mask=tiled_text_mask,
              input_word_ids=tiled_input_word_ids)
      ],
                               axis=1)
      input_mask = model_utils.merge_pyramid_input_masks(
          video_input_masks,
          input_text_mask=jnp.ones((video_tokens.shape[0], 1), dtype=jnp.int32))
      num_text_tokens = 1

    tokens = encoder(tokens, input_mask=input_mask, train=train)
    if self.use_all_text_tokens:
      text_emb = model_utils.extract_emb(
          tokens[:, -num_text_tokens:],
          self.text_tower_classifier,
          keepdims=False,
          input_mask=tiled_text_mask,
          input_word_ids=tiled_input_word_ids)
    else:
      text_emb = tokens[:, -1]
    if task in {
        'action_segmentation',
        'temporal_localization',
        'moment_retrieval',
    }:
      return tokens[:, :-num_text_tokens], text_emb
    else:
      raise ValueError(f'Unexpected task `{task}`.')

  @nn.compact
  def __call__(self,
               video_tokens: jnp.ndarray,
               text_tokens: jnp.ndarray,
               task: str,
               input_word_ids: Optional[jnp.ndarray] = None,
               text_input_mask: Optional[jnp.ndarray] = None,
               video_input_mask: Optional[jnp.ndarray] = None,
               train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Executes video-text fusion by self attention.

    Args:
      video_tokens: A 3D float tensor of shape (batch_size, sequence_length,
        channels) representing the video tokens.
      text_tokens: A 3D float tensor of shape (num_texts, sequence_length,
        channels) representing the text tokens. The first dimension is
        batch_size * max_num_captions if task = `moment_retrieval`. Otherwise,
        it is num_classes.
      task: 'action_segmentation', 'temporal_localization', 'moment_retrieval'
        or 'highlight_detection.
      input_word_ids: A 2D int tensor of shape (num_texts, sequence_length)
        representing the input word indices. This arg is used to find EOS id.
      text_input_mask: A 2D binary tensor of shape (num_texts, sequence_length)
        representing the mask of the text inputs.
      video_input_mask: A 2D binary tensor of shape (batch_size,
        sequence_length) representing the mask of the video inputs.
      train: Whether or not it is in training.

    Returns:
      video_tokens:
        A 4D float tensor of shape (batch_size, num_classes, sequence_length,
        channels) representing the fused tokens for each class if task is
        'temporal_localization'.

        A 4D float tensor of shape (batch_size, 1, sequence_length, channels)
        representing the fused frame tokens if task is 'highlight_detection'.

        A 3D float tensor of shape (batch_size, sequence_length, num_classes)
        representing the fused tokens for each class if task is
        'action_segmentation'.

        A 4D float tensor of shape (batch_size, batch_size * max_num_captions,
        sequence_length, channels) representing the fused tokens for each class
        if task = 'moment_retrieval'.
      text_tokens:
        Not used. A 3D float tensor of shape (batch_size, num_texts, channels)
        representing the text CLS token for each class. The second dimension is
        batch_size * max_num_captions if task = `moment_retrieval`. Otherwise,
        it is num_classes.
    """

    encoder = _VIDEO_TEXT_ENCODER[self.self_attention_encoder_name](
        name='video_text_encoder', **self.self_attention_encoder_config)
    if video_input_mask is None:
      video_input_mask = jnp.ones(video_tokens.shape[:2], dtype=jnp.int32)
    if task == 'highlight_detection':
      return self._self_attention_encode_all_video_text_pairs(
          video_tokens,
          video_input_mask,
          text_tokens,
          input_word_ids,
          text_input_mask,
          encoder,
          train,
      )

    video_tokens, text_tokens = jax.vmap(
        functools.partial(
            self._self_attention_encode_per_text,
            encoder=encoder,
            task=task,
            train=train,
        ),
        in_axes=[None, None, 0, 0, 0],
    )(
        video_tokens,
        video_input_mask,
        text_tokens,
        input_word_ids,
        text_input_mask,
    )
    if task == 'temporal_localization':
      # Convert video_tokens from (num_classes, batch_size, num_frames,
      # channels) to (batch_size, num_classes, num_frames, channels).
      return (jnp.transpose(video_tokens, [1, 0, 2, 3]),
              jnp.transpose(text_tokens, [1, 0, 2]))
    elif task == 'action_segmentation':
      # Convert video_tokens from (num_classes, batch_size, num_frames,
      # channels) to (batch_size, num_frames, num_classes, channels).
      return (jnp.transpose(video_tokens, [1, 2, 0, 3]),
              jnp.transpose(text_tokens, [1, 0, 2]))
    elif task == 'moment_retrieval':
      # Convert video_tokens from (batch_size * max_num_captions, batch_size,
      # num_frames, channels) to (batch_size, batch_size * max_num_captions,
      # num_frames, channels).
      return (jnp.transpose(video_tokens, [1, 0, 2, 3]),
              jnp.transpose(text_tokens, [1, 0, 2]))
    else:
      raise ValueError(f'Unexpected task `{task}`.')


FUSION_MODELS = {
    'video_text_emb_self_attention': VideoTextEmbSelfAttentionFusion,
    'video_text_self_attention': VideoTextSelfAttentionFusion,
}
