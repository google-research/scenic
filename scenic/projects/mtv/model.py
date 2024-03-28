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

"""Implements the MTV model."""
import functools
from typing import Any, List, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import vit
from scenic.projects.mtv import model_utils
from scenic.projects.vivit import model as vivit_model
from scenic.train_lib import train_utils

_DEFAULT_MTV_CONFIG = ml_collections.ConfigDict({
    'dataset_configs': {
        'num_frames': 8,
    },
    'model':
        dict(
            view_configs=[
                ml_collections.ConfigDict({
                    'hidden_size': 16,
                    'patches': {
                        'size': (4, 4, 2)
                    },
                    'num_heads': 2,
                    'mlp_dim': 32,
                    'num_layers': 1,
                })
            ],
            cross_view_fusion=None,
            temporal_encoding_config=ml_collections.ConfigDict({
                'method': '3d_conv',
                'kernel_init_method': 'central_frame_initializer',
            }),
            global_encoder_config=ml_collections.ConfigDict({
                'num_layers': 2,
                'mlp_dim': 8,
                'num_heads': 2,
                'hidden_size': 8,
            }),
            dropout_rate=0.,
            attention_dropout_rate=0.,
            classifier='token',
            data_dtype_str='float32')
})


def get_model_cls(model_name):
  """"Selects MTV model type."""
  if model_name == 'mtv_multiclass_classification':
    return MTVClassificationModel
  elif model_name == 'mtv_multihead_classification':
    return MTVMultiheadClassificationModel
  else:
    raise ValueError('Unrecognized model: {}'.format(model_name))


class CrossViewAttentionEncoderBlock(nn.Module):
  """Crossview Transformer encoder layer.

  The encoder architecture for each view is as follows:
  Layer norm
  cross attention (out projection weights are initialized with zeros)
  residual connection
  Layer norm
  self attention (initialized with pretrained ViT weights)
  residual connection
  Layer norm
  MLP (initialized with pretrained ViT weights)
  residual connection

  We apply cross attention in a sequential fashion and limit it to only take
  place in neighboring views. For example, view[i-1] is used as the query and
  view[i] is used as key and value. This design is based on the assumption
  that the tubelet sizes grow from 0th view to the nth view. We initialize cross
  attention's weights with zeros and self attention and MLP weights are
  initialized with pretrained ViTs.

  Attributes:
    view_configs: Model configs for each view (e.g., num_heads, mlp_dim, etc).
    cross_view_fusion: Cross view fusion config.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value.

  Returns:
    output after transformer encoder block.
  """
  view_configs: Sequence[ml_collections.ConfigDict]
  cross_view_fusion: ml_collections.ConfigDict
  dtype: Any = jnp.float32
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  stochastic_depth: float = 0.0

  def _get_stochastic_depth_rate(self, cur_layer, view_idx):
    """Returns the stochastic depth rate for the current layer and view."""
    max_layer = max(self.view_configs[view_idx]['num_layers'] - 1, 1)
    return (cur_layer / max_layer) * self.stochastic_depth

  def _apply_self_attentions(self, tokens: List[jnp.ndarray], cur_layer: int,
                             deterministic: bool) -> List[jnp.ndarray]:
    """Applies self attentions for each view."""
    for view_idx, x in enumerate(tokens):
      if cur_layer >= self.view_configs[view_idx]['num_layers']:
        continue
      y = nn.LayerNorm(dtype=self.dtype, name=f'msa_ln_view{view_idx}')(x)
      config = self.view_configs[view_idx]
      y = nn.MultiHeadDotProductAttention(
          num_heads=config['num_heads'],
          dtype=self.dtype,
          broadcast_dropout=False,
          deterministic=deterministic,
          dropout_rate=self.attention_dropout_rate,
          name=f'msa_view{view_idx}')(y, y)
      y = nn.Dropout(rate=self.dropout_rate)(y, deterministic)
      r = self._get_stochastic_depth_rate(cur_layer, view_idx)
      tokens[view_idx] += nn_layers.StochasticDepth(r)(y, deterministic)
    return tokens

  def _apply_cross_attention(
      self,
      tokens: List[jnp.ndarray],
      cur_layer: int,
      deterministic: bool,
      fuse_in_descending_order: bool,
  ) -> List[jnp.ndarray]:
    """Applies cross view attention."""
    xs = [
        nn.LayerNorm(dtype=self.dtype, name=f'cross_attention_ln_view{idx}')(x)
        for idx, x in enumerate(tokens)
    ]
    view_indices = (
        range(len(xs) -
              1, 0, -1) if fuse_in_descending_order else range(len(xs) - 1))
    for view_index in view_indices:
      query_view_index = (
          view_index - 1 if fuse_in_descending_order else view_index + 1)
      key_value_view_index = view_index
      query = xs[query_view_index]
      key_value = xs[key_value_view_index]
      num_heads = (
          self.view_configs[query_view_index]['num_heads']
          if self.cross_view_fusion.use_query_config else
          self.view_configs[key_value_view_index]['num_heads'])
      qkv_features = (
          query.shape[-1]
          if self.cross_view_fusion.use_query_config else key_value.shape[-1])

      y = attention_layers.MultiHeadAttention(
          num_heads=num_heads,
          dtype=self.dtype,
          qkv_features=qkv_features,
          out_kernel_init=nn.initializers.zeros,
          dropout_rate=self.attention_dropout_rate,
          name=f'cross_attention_view{query_view_index}_{key_value_view_index}'
      )(query, key_value, deterministic=deterministic)
      y = nn.Dropout(rate=self.dropout_rate)(y, deterministic)
      r = self._get_stochastic_depth_rate(cur_layer, view_index)
      tokens[query_view_index] += nn_layers.StochasticDepth(r)(y, deterministic)

    return tokens

  def _apply_mlp(self, tokens: List[jnp.ndarray], cur_layer: int,
                 deterministic: bool) ->List[jnp.ndarray]:
    """Applies MLP block."""
    for view_idx, x in enumerate(tokens):
      if cur_layer >= self.view_configs[view_idx]['num_layers']:
        continue
      y = nn.LayerNorm(dtype=self.dtype, name=f'mlp_ln_view{view_idx}')(x)
      y = attention_layers.MlpBlock(
          mlp_dim=self.view_configs[view_idx]['mlp_dim'],
          dtype=self.dtype,
          dropout_rate=self.dropout_rate,
          activation_fn=nn.gelu,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6),
          name=f'mlp_view{view_idx}')(
              y, deterministic=deterministic)
      r = self._get_stochastic_depth_rate(cur_layer, view_idx)
      tokens[view_idx] += nn_layers.StochasticDepth(r)(y, deterministic)
    return tokens

  @nn.compact
  def __call__(self, tokens: List[jnp.ndarray], cur_layer: int,
               deterministic: bool) -> List[jnp.ndarray]:
    """Applies CrossViewAttentionEncoderBlock module.

    Args:
      tokens: Input tokens from each view.
      cur_layer: Which layer we apply cross attention.
      deterministic: Deterministic or not (to apply dropout).

    Returns:
      Output tokens for each view.
    """
    tokens = self._apply_cross_attention(
        tokens, cur_layer, deterministic,
        self.cross_view_fusion.get('fuse_in_descending_order', True))
    tokens = self._apply_self_attentions(tokens, cur_layer, deterministic)
    tokens = self._apply_mlp(tokens, cur_layer, deterministic)
    return tokens


class MultiviewEncoder(nn.Module):
  """Multiview Transformer Encoder.

  Attributes:
    view_configs: Model configs for each view (e.g., num_heads, mlp_dim, etc).
    cross_view_fusion: Cross view fusion config.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value. Our implementation of stochastic depth follows the
      timm library, which does per-example layer dropping and uses independent
      dropping patterns for each skip-connection.
    dtype: Any of activations.
  """
  view_configs: Sequence[ml_collections.ConfigDict]
  cross_view_fusion: ml_collections.ConfigDict
  input_token_temporal_dims: Sequence[int]
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  dtype: Any = jnp.float32

  def _split_tokens_and_bottleneck(
      self, tokens: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Removes bottleneck tokens from input."""
    return (tokens[:, :-self.cross_view_fusion.bottleneck_tokens],
            tokens[:, -self.cross_view_fusion.bottleneck_tokens:])

  def _add_posembed(self, tokens: Sequence[jnp.ndarray]) -> List[jnp.ndarray]:
    """Adds positional embeddings."""
    temporal_dims_after_alignment = [
        t // min(self.input_token_temporal_dims)
        for t in self.input_token_temporal_dims
    ]
    xs = []
    for idx, t in enumerate(tokens):
      bs, spacetime, channels = t.shape
      reshaped_t = t.reshape(
          (bs, temporal_dims_after_alignment[idx], -1, channels))
      add_posembed_fn = vit.AddPositionEmbs(name=f'posembed_input_view{idx}')
      x = jax.vmap(add_posembed_fn, in_axes=1, out_axes=1)(reshaped_t)
      xs.append(x.reshape(bs, spacetime, channels))
    return xs

  def _build_with_bottleneck(
      self,
      xs: List[jnp.ndarray],
      bottleneck: jnp.ndarray,
      fusion_layers: Sequence[int],
      max_num_layers: int,
      train: bool,
      dtype: Any,
  ) -> List[jnp.ndarray]:
    """Builds the encoder with bottlenecks."""
    view_indices = list(range(len(self.view_configs)))
    if self.cross_view_fusion.get('fuse_in_descending_order', True):
      view_indices.reverse()
    for lyr in range(max_num_layers):
      for view_idx in view_indices:
        view_config = self.view_configs[view_idx]
        if lyr >= view_config['num_layers']:
          continue
        if lyr in fusion_layers:
          if xs[view_idx].shape[-1] != bottleneck.shape[-1]:
            bottleneck = nn.Dense(
                xs[view_idx].shape[-1],
                kernel_init=nn.initializers.xavier_uniform(),
                name=f'bottleneck_linear_{lyr}_view{view_idx}')(
                    bottleneck)
          xs[view_idx] = jnp.concatenate([xs[view_idx], bottleneck], axis=1)

        xs[view_idx] = vit.Encoder1DBlock(
            mlp_dim=view_config['mlp_dim'],
            num_heads=view_config['num_heads'],
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            stochastic_depth=(lyr / max(view_config['num_layers'] - 1, 1)) *
            self.stochastic_depth,
            name=f'encoderblock_{lyr}_view{view_idx}',
            dtype=self.dtype)(
                xs[view_idx], deterministic=not train)
        if lyr in fusion_layers:
          xs[view_idx], bottleneck = self._split_tokens_and_bottleneck(
              xs[view_idx])
    return xs

  def _build_with_cross_view_attention(
      self,
      xs: List[jnp.ndarray],
      fusion_layers: Sequence[int],
      max_num_layers: int,
      train: bool,
      dtype: Any,
  ) -> List[jnp.ndarray]:
    """Builds the encoder with bottlenecks."""
    for lyr in range(max_num_layers):
      if lyr in fusion_layers:
        xs = CrossViewAttentionEncoderBlock(
            view_configs=self.view_configs,
            cross_view_fusion=self.cross_view_fusion,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            stochastic_depth=self.stochastic_depth,
            name=f'cross_view_encoderblock_{lyr}')(
                xs, lyr, deterministic=not train)
      else:
        for view_idx, view_config in enumerate(self.view_configs):
          if lyr >= view_config['num_layers']:
            continue
          xs[view_idx] = vit.Encoder1DBlock(
              mlp_dim=view_config['mlp_dim'],
              num_heads=view_config['num_heads'],
              dropout_rate=self.dropout_rate,
              attention_dropout_rate=self.attention_dropout_rate,
              stochastic_depth=(lyr / max(view_config['num_layers'] - 1, 1)) *
              self.stochastic_depth,
              name=f'encoderblock_{lyr}_view{view_idx}',
              dtype=self.dtype)(
                  xs[view_idx], deterministic=not train)
    return xs

  @nn.compact
  def __call__(self,
               tokens: Sequence[jnp.ndarray],
               bottleneck: Union[jnp.ndarray, None],
               train: bool = False) -> List[jnp.ndarray]:
    """Applies Transformer model on the tokens.

    This function will be called within a vmap along the time axis. Before
    calling this function, we need to make sure all elements in the list have
    the same temporal dimension.

    Args:
      tokens: A sequence of input tubelet tokens. Each one is a 3D float tensor
        of shape (batch, sequence_len, channels). We assume that tokens[0]
        contains tokens from the largest view while tokens[-1] are from the
        smallest view. We define a view as a representation of the input video
        composed of tubelets. A larger view corresponds to larger tubelets.
      bottleneck: A 3D float tensor of shape (batch, num_tokens, channels)
        representing a set of tokens used for fusing information among views.
      train: Whether or not it is in training.

    Returns:
      A list of activations after encoding for each view. They have the same
      shapes as their input counterparts.
    """

    for t in tokens:
      assert t.ndim == 3  # Shape is `[batch, len, emb]`.
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)
    xs = self._add_posembed(tokens)
    max_num_layers = max([config['num_layers'] for config in self.view_configs])
    fusion_layers = ([] if self.cross_view_fusion is None else
                     self.cross_view_fusion.fusion_layers)

    if (self.cross_view_fusion is None or
        self.cross_view_fusion.type == 'cross_view_attention'):
      return self._build_with_cross_view_attention(xs, fusion_layers,
                                                   max_num_layers, train, dtype)
    if self.cross_view_fusion.type == 'bottleneck':
      return self._build_with_bottleneck(xs, bottleneck, fusion_layers,
                                         max_num_layers, train, dtype)
    raise ValueError(
        f'Invalid cross view fusion type: {self.cross_view_fusion.type}.')


class MTV(nn.Module):
  """MTV model."""
  view_configs: Sequence[ml_collections.ConfigDict]
  cross_view_fusion: ml_collections.ConfigDict
  temporal_encoding_config: ml_collections.ConfigDict
  global_encoder_config: ml_collections.ConfigDict
  input_token_temporal_dims: Sequence[int]
  num_classes: int
  classifier: str
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  keep_spatiotemporal_features: bool = False
  final_endpoint: str = 'logits'
  dtype: Any = jnp.float32

  def _add_cls_token(self, x: jnp.ndarray, name: str) -> jnp.ndarray:
    """Prepends CLS token.

    Args:
      x: A 3D float tensor of shape (batch, sequence_len, channels) representing
        the tokens.
      name: Parameter name of the added CLS token.

    Returns:
      A 3D float tensor with prepended CLS token. Its new shape is (batch,
      sequence_len+1, channels).
    """
    if self.classifier == 'token':
      bs, _, c = x.shape
      cls = self.param(name, nn.initializers.zeros, (1, 1, c), x.dtype)
      cls = jnp.tile(cls, [bs, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
    return x

  def _add_cls_tokens_all_frames(self, x: jnp.ndarray,
                                 name: str) -> jnp.ndarray:
    """Prepends CLS token for all frames.

    Args:
      x: A 4D float tensor of shape (batch, time, sequence_len, channels)
        representing the tokens.
      name: Parameter name of the added CLS token.

    Returns:
      A 4D float tensor with prepended CLS token. Its new shape is (batch, time,
      sequence_len+1, channels).
    """
    if self.classifier == 'token':
      bs, time, _, c = x.shape
      cls = self.param(name, nn.initializers.zeros, (1, time, 1, c), x.dtype)
      cls = jnp.tile(cls, [bs, 1, 1, 1])
      x = jnp.concatenate([cls, x], axis=2)
    return x

  def _add_cls_tokens_for_all_views(
      self, tokens: Sequence[jnp.ndarray]) -> List[jnp.ndarray]:
    """Prepends CLS tokens for all views.

    Args:
      tokens: Tokens from all views. Each one has a shape of (batch, time,
        sequence_len, channels)

    Returns:
      A list of tokens with CLS tokens added. Each one has a new shape of
      (batch, time, sequence_len+1, channels).
    """
    outputs = []
    for idx, x in enumerate(tokens):
      outputs.append(self._add_cls_tokens_all_frames(x, name=f'cls_view{idx}'))
    return outputs

  def _extract_encoder_output(self,
                              x: jnp.ndarray,
                              axis: int = 1) -> jnp.ndarray:
    """Extracts encoder output."""
    if self.classifier in ['token', '0']:
      x = x.take(indices=0, axis=axis)
    elif self.classifier in ('gap', 'gmp', 'gsp'):
      fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
      x = fn(x, axis=list(range(axis, x.ndim - 1)))
    return x

  def _tokenize(self, x: jnp.ndarray) -> List[jnp.ndarray]:
    """Creates tokens for each view.

    Args:
      x: A 5D float tensor of shape (batch, time, height, width, channels)
        representing the input video.

    Returns:
      Tokens for each view and each one has a shape of (batch, time,
      sequence_len, channels).
    """
    tokens = []
    for idx, config in enumerate(self.view_configs):
      view_tokens, _ = vivit_model.temporal_encode(
          x,
          self.temporal_encoding_config,
          ml_collections.ConfigDict(config['patches']),
          config['hidden_size'],
          return_1d=False,
          name=f'embedding_view{idx}')
      bs, t, h, w, c = view_tokens.shape
      view_tokens = view_tokens.reshape(bs, t, h * w, c)
      tokens.append(view_tokens)
    return tokens

  def _align_temporal_dimension_across_views(
      self, tokens: Sequence[jnp.ndarray]) -> List[jnp.ndarray]:
    """Reshapes tokens from each view so they have the same temporal dim."""
    min_temporal_dim = min(self.input_token_temporal_dims)
    outputs = []
    for t in tokens:
      bs, time, n, c = t.shape
      outputs.append(
          t.reshape(bs, min_temporal_dim, (n * time) // min_temporal_dim, c))
    return outputs

  def _merge_views_along_time_axis(self, tokens: Sequence[jnp.ndarray],
                                   hidden_size: int) -> jnp.ndarray:
    """Merges tokens from each view along the time axis."""
    projected_tokens = []
    for view_idx, x in enumerate(tokens):
      bs, time, n, c = x.shape
      x = x.reshape(bs, self.input_token_temporal_dims[view_idx],
                    (time * n) // self.input_token_temporal_dims[view_idx], c)
      if not self.keep_spatiotemporal_features:
        x = self._extract_encoder_output(x, axis=2)
      projected_tokens.append(
          nn.Dense(
              hidden_size,
              kernel_init=nn.initializers.xavier_uniform(),
              name=f'global_encoder_linear_view{view_idx}')(x))
    return jnp.concatenate(projected_tokens, axis=1)

  def _merge_views_along_channel_axis(
      self, tokens: Sequence[jnp.ndarray]) -> jnp.ndarray:
    """Merges tokens from each view along the channel axis."""
    max_temporal_dim = max(self.input_token_temporal_dims)
    xs = []
    for idx, x in enumerate(tokens):
      bs, time, n, c = x.shape
      x = x.reshape(bs, self.input_token_temporal_dims[idx],
                    (time * n) // self.input_token_temporal_dims[idx], c)
      if self.keep_spatiotemporal_features:
        xs.append(jnp.tile(x, (1, max_temporal_dim // x.shape[1], 1, 1)))
      else:
        x = self._extract_encoder_output(x, axis=2)
        xs.append(jnp.tile(x, (1, max_temporal_dim // x.shape[1], 1)))
    return jnp.concatenate(xs, axis=-1)

  def _global_encode(self, tokens: Sequence[jnp.ndarray],
                     is_train: bool) -> jnp.ndarray:
    """Applies the global encoder.

    We support two strategies to merge encoded tokens from each view:

    In the first strategy, we extract the CLS tokens from each view (we apply
    pooling when other classifiers are used), apply tiling to match the temporal
    dimension, and concatenate them in the channel dimension.

    In the second strategy, after we extract the CLS tokens we linear project
    them into the same dimension and concatenate them along the temporal
    dimension.

    The global encoder is implemented as a ViT encoder.

    Args:
      tokens: A list of tokens from each view. Each one has a shape of (batch,
        time, sequence_len, channels).
      is_train: Whether or not it is in training.

    Returns:
      A 2D float tensor representing the embedding from the global encoder.
    """
    encoder_config = self.global_encoder_config.to_dict()
    encoder_config.update(
        dict(
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            stochastic_depth=self.stochastic_depth,
            dtype=self.dtype,
            name='global_encoder'))
    merge_axis = encoder_config.pop('merge_axis', 'channel')
    hidden_size = encoder_config.pop('hidden_size')
    if merge_axis == 'time':
      x = self._merge_views_along_time_axis(tokens, hidden_size)
    elif merge_axis == 'channel':
      x = self._merge_views_along_channel_axis(tokens)
    else:
      raise ValueError(f'Invalid merge_axis: {merge_axis}.')
    x = self._add_cls_token(x, name='cls_global')
    encoder = vit.Encoder(**encoder_config)
    if self.keep_spatiotemporal_features:
      x = jax.vmap(
          functools.partial(encoder, train=is_train), in_axes=2, out_axes=2)(
              x)
    else:
      x = encoder(x, train=is_train)
    return (x if self.keep_spatiotemporal_features else
            self._extract_encoder_output(x))

  def _encode_per_time(
      self,
      tokens: Sequence[jnp.ndarray],
      bottleneck: Union[jnp.ndarray, None],
      is_train: bool,
  ) -> List[jnp.ndarray]:
    """Encodes input tokens on a per-time basis."""

    tokens = MultiviewEncoder(
        view_configs=self.view_configs,
        cross_view_fusion=self.cross_view_fusion,
        input_token_temporal_dims=self.input_token_temporal_dims,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
        dtype=self.dtype,
        name='MultiviewEncoder')(
            tokens, bottleneck=bottleneck, train=is_train)
    return tokens

  def _check_config(self, x: jnp.ndarray):
    """Checks configuration errors."""
    if self.keep_spatiotemporal_features and self.classifier == 'token':
      raise ValueError('Classifier cannot be `token` when '
                       '`keep_spatiotemporal_features` is True.')
    heights = [config['patches']['size'][0] for config in self.view_configs]
    widths = [config['patches']['size'][1] for config in self.view_configs]
    if self.keep_spatiotemporal_features and (len(set(heights)) > 1 or
                                              len(set(widths)) > 1):
      raise ValueError('Patches from different views must have the same height '
                       'and width when `keep_spatiotemporal_features` is True.')

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               *,
               train: bool = True,
               debug: bool = False):
    """Executes MTV model.

    Args:
      x: A 5D float tensor of shape (batch, time, height, width, channels)
        representing the input video.
      train: Whether or not it is in training.
      debug: Whether or not it is in debug mogde. Not used here.

    Returns:
      The logits produced by the MTV model.
    """
    del debug
    self._check_config(x)
    tokens = self._tokenize(x)
    tokens = self._add_cls_tokens_for_all_views(tokens)
    tokens = self._align_temporal_dimension_across_views(tokens)
    if (self.cross_view_fusion is not None and
        self.cross_view_fusion.type == 'bottleneck'):
      if self.cross_view_fusion.get('fuse_in_descending_order', True):
        channels = tokens[-1].shape[-1]
      else:
        channels = tokens[0].shape[-1]
      bottleneck = self.param(
          'bottleneck', nn.initializers.normal(stddev=0.02),
          (1, tokens[0].shape[1], self.cross_view_fusion.bottleneck_tokens,
           channels), self.dtype)
      bottleneck = jnp.tile(bottleneck, [x.shape[0], 1, 1, 1])
      tokens = jax.vmap(
          functools.partial(self._encode_per_time, is_train=train),
          in_axes=(1, 1),
          out_axes=1)(tokens, bottleneck)
    else:
      tokens = jax.vmap(
          functools.partial(
              self._encode_per_time, bottleneck=None, is_train=train),
          in_axes=1,
          out_axes=1)(
              tokens)
    tokens = self._global_encode(tokens, train)
    if self.keep_spatiotemporal_features:
      bs, _, h, w, _ = x.shape
      tokens = tokens.reshape(
          (bs, tokens.shape[1], h // self.view_configs[0].patches.size[0],
           w // self.view_configs[0].patches.size[1], -1))
    pre_logits = nn_layers.IdentityLayer(name='pre_logits')(tokens)
    if self.final_endpoint == 'pre_logits':
      return pre_logits
    if self.keep_spatiotemporal_features:
      pre_logits = self._extract_encoder_output(pre_logits, axis=1)
    logits = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(
            pre_logits)
    if self.final_endpoint == 'logits':
      return logits
    raise ValueError(f'Final endpoint `{self.final_endpoint}` not recognized.')


class MTVClassificationModel(vivit_model.ViViTClassificationModel):
  """MTV model for multiclass classification task."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return MTV(
        view_configs=self.config.model.view_configs,
        cross_view_fusion=self.config.model.cross_view_fusion,
        temporal_encoding_config=self.config.model.temporal_encoding_config,
        global_encoder_config=self.config.model.global_encoder_config,
        input_token_temporal_dims=model_utils.get_input_token_temporal_dims(
            self.config.dataset_configs.num_frames,
            self.config.model.view_configs),
        num_classes=self.dataset_meta_data['num_classes'],
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.0),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.1),
        stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
        keep_spatiotemporal_features=self.config.model.get(
            'keep_spatiotemporal_features', False),
        final_endpoint=self.config.model.get('final_endpoint', 'logits'),
        dtype=model_dtype)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return _DEFAULT_MTV_CONFIG

  def init_from_train_state(
      self,
      train_state: train_utils.TrainState,
      restored_train_state: train_utils.TrainState,
      restored_model_cfg: ml_collections.ConfigDict,
      restore_output_proj: bool = False) -> train_utils.TrainState:
    """Updates the train_state with data from restored_train_state.

    This function is writen to be used for 'fine-tuning' experiments. The input
    embeddings and positional embeddings are resized if the current model uses
    a different size of tubelets than the pretrained model.

    Args:
      train_state: A raw TrainState for the model.
      restored_train_state: A TrainState that is loaded with parameters/state of
        a  pretrained model.
      restored_model_cfg: Configuration of the model from which the
        restored_train_state come from. Usually used for some asserts.
      restore_output_proj: Whether or not to restore output projection weights.

    Returns:
      Updated train_state.
    """
    return model_utils.initialize_from_mtv_train_state(
        self.config,
        train_state,
        restored_train_state,
        restored_model_cfg,
        restore_output_projection=restore_output_proj)

  def init_from_vit_train_states(
      self,
      train_state: train_utils.TrainState,
      restored_train_states: Sequence[train_utils.TrainState],
      restored_model_cfgs: Sequence[ml_collections.ConfigDict],
      restored_model_formats: Sequence[str],
  ) -> train_utils.TrainState:
    """Updates the train_state with data from restored_train_states.

    This function is used to initialize a MTV model from a list of ViT
    checkpoints. We assume that the number of restored_train_states is equal to
    the number of views.

    Args:
      train_state: A raw TrainState for the model.
      restored_train_states: A sequence of TrainStates that is loaded with
        parameters/state of a pretrained ViT model.
      restored_model_cfgs: A sequence of model configuration of the pretrained
        ViT models. Usually used for some asserts.
      restored_model_formats: The checkpoint format of each model. The format
        can be 'scenic' or 'big_vision'.

    Returns:
      Updated train_state.
    """
    return model_utils.initialize_from_vit_train_states(self.config,
                                                        train_state,
                                                        restored_train_states,
                                                        restored_model_cfgs,
                                                        restored_model_formats)


class MTVMultiheadClassificationModel(
    vivit_model.ViViTMultiHeadClassificationModel, MTVClassificationModel):
  """MTV model for multi-classification tasks.

  When methods are overriden by both parents, the implementation follows the
  first parent, which is ViViTMultiHeadClassificationModel in this case. For
  build_flax_model() and default_flax_model_config(), we explicitly call the
  methods from MTVClassificationModel.
  """

  def build_flax_model(self) -> nn.Module:
    return MTVClassificationModel.build_flax_model(self)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return MTVClassificationModel.default_flax_model_config(self)
