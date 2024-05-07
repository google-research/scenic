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

"""ViViT: Vision Transformer for Video."""

import functools
from typing import Any, Optional, Callable, Sequence

from absl import logging
import flax.linen as nn
from flax.linen.linear import default_kernel_init
from immutabledict import immutabledict
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.common_lib import video_utils
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import classification_model
from scenic.model_lib.base_models import model_utils as base_model_utils
from scenic.model_lib.base_models.classification_model import ClassificationModel
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import vit
from scenic.projects.vivit import model_utils

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


def get_model_cls(model_name):
  """"Selects Vivit model type."""
  if model_name == 'vivit_multilabel_classification':
    return ViViTMultilabelClassificationModel
  elif model_name == 'vivit_classification':
    return ViViTClassificationModel
  elif model_name == 'vivit_multihead_classification':
    return ViViTMultiHeadClassificationModel
  else:
    raise ValueError('Unrecognized model: {}'.format(model_name))


_AXIS_TO_NAME = immutabledict({
    1: 'time',
    2: 'space',
})

KERNEL_INITIALIZERS = immutabledict({
    'zero': nn.initializers.zeros,
    'xavier': nn.initializers.xavier_uniform(),
})

ViViT_CLASSIFICATION_METRICS_BASIC = immutabledict({
    'accuracy': (base_model_utils.weighted_correctly_classified,
                 base_model_utils.num_examples),
    'loss': (base_model_utils.weighted_unnormalized_softmax_cross_entropy,
             base_model_utils.num_examples)
})

ViViT_CLASSIFICATION_METRICS = immutabledict({
    **ViViT_CLASSIFICATION_METRICS_BASIC,
    'accuracy_top_5': (functools.partial(
        base_model_utils.weighted_topk_correctly_classified,
        k=5), base_model_utils.num_examples),
})


def _reshape_to_time_space(x, temporal_dims):
  if x.ndim == 3:
    b, thw, d = x.shape
    assert thw % temporal_dims == 0
    hw = thw // temporal_dims
    x = jnp.reshape(x, [b, temporal_dims, hw, d])
  assert x.ndim == 4
  return x


def embed_2d_patch(x, patches, embedding_dim):
  """Standard ViT method of embedding input patches."""

  n, h, w, c = x.shape

  assert patches.get('size') is not None, ('patches.size is now the only way'
                                           'to define the patches')

  fh, fw = patches.size
  gh, gw = h // fh, w // fw

  if embedding_dim:
    x = nn.Conv(
        embedding_dim, (fh, fw),
        strides=(fh, fw),
        padding='VALID',
        name='embedding')(x)
  else:
    # This path often results in excessive padding: b/165788633
    x = jnp.reshape(x, [n, gh, fh, gw, fw, c])
    x = jnp.transpose(x, [0, 1, 3, 2, 4, 5])
    x = jnp.reshape(x, [n, gh, gw, -1])

  return x


def embed_3d_patch(x,
                   patches,
                   embedding_dim,
                   kernel_init_method,
                   name='embedding'):
  """Embed 3D input patches into tokens."""

  assert patches.get('size') is not None, 'patches.size must be defined'
  assert len(patches.size) == 3, 'patches.size must have 3 elements'
  assert embedding_dim, 'embedding_dim must be specified'

  fh, fw, ft = patches.size

  if kernel_init_method == 'central_frame_initializer':
    kernel_initializer = model_utils.central_frame_initializer()
    logging.info('Using central frame initializer for input embedding')
  elif kernel_init_method == 'average_frame_initializer':
    kernel_initializer = model_utils.average_frame_initializer()
    logging.info('Using average frame initializer for input embedding')
  else:
    kernel_initializer = default_kernel_init
    logging.info('Using default initializer for input embedding')

  x = nn.Conv(
      embedding_dim, (ft, fh, fw),
      strides=(ft, fh, fw),
      padding='VALID',
      name=name,
      kernel_init=kernel_initializer)(
          x)

  return x


def temporal_encode(x,
                    temporal_encoding_config,
                    patches,
                    hidden_size,
                    return_1d=True,
                    name='embedding'):
  """Encode video for feeding into ViT."""

  n, _, in_h, in_w, c = x.shape

  if temporal_encoding_config.method == 'temporal_sampling':
    n_sampled_frames = temporal_encoding_config.n_sampled_frames
    x = video_utils.sample_frames_uniformly(x, n_sampled_frames)
    t_s = x.shape[1]
    x = jnp.reshape(x, [n, t_s * in_h, in_w, c])

    x = embed_2d_patch(x, patches, hidden_size)
    temporal_dims = t_s
    if return_1d:
      n, th, w, c = x.shape
      x = jnp.reshape(x, [n, th * w, c])
    else:
      n, th, w, c = x.shape
      x = jnp.reshape(x, [n, t_s, -1, w, c])

  elif temporal_encoding_config.method == '3d_conv':
    kernel_init_method = temporal_encoding_config.get('kernel_init_method',
                                                      None)
    x = embed_3d_patch(x, patches, hidden_size, kernel_init_method, name)
    temporal_dims = x.shape[1]
    if return_1d:
      n, t, h, w, c = x.shape
      x = jnp.reshape(x, [n, t * h * w, c])

  else:
    raise AssertionError('Unknown temporal encoding method.')

  assert x.size > 0, ('Found zero tokens after temporal encoding. '
                      'Perhaps one of the patch sizes is such that '
                      'floor(dim_size / patch_size) = 0?')

  return x, temporal_dims


class EncoderBlock(nn.Module):
  """Transformer encoder block.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of heads.
    attention_axis: Axis over which we run attention.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    droplayer_p: Probability of dropping a layer.
    attention_kernel_initializer: Initializer to use for attention
      layers.
    deterministic: Deterministic or not (to apply dropout).
    attention_fn: dot_product_attention or compatible function. Accepts query,
      key, value, and returns output of shape `[bs, dim1, dim2, ..., dimN,,
      num_heads, value_channels]``
    dtype: The dtype of the computation (default: float32).

  Returns:
    Output after transformer encoder block.
  """
  mlp_dim: int
  num_heads: int
  dtype: jnp.dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  attention_kernel_initializer: Initializer = nn.initializers.xavier_uniform()
  attention_fn: Any = nn.dot_product_attention
  droplayer_p: float = 0.0

  def get_drop_pattern(self, x, deterministic):
    if not deterministic and self.droplayer_p:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng('dropout'), self.droplayer_p, shape).astype('float32')
    else:
      return 0.0

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
    """Applies Encoder1DBlock module."""

    # Attention block.
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=self.attention_kernel_initializer,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        attention_fn=self.attention_fn,
        dtype=self.dtype)(
            x, x, deterministic=deterministic)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)

    drop_pattern = self.get_drop_pattern(x, deterministic)
    x = x * (1.0 - drop_pattern) + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = attention_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(
            y, deterministic=deterministic)

    drop_pattern = self.get_drop_pattern(x, deterministic)
    return y * (1.0 - drop_pattern) + x


class EncoderFactorizedSelfAttentionBlock(nn.Module):
  """Encoder with facctorized self attention block.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of heads.
    temporal_dims: Number of temporal dimensions in the flattened input
    attention_kernel_initializer: Initializer to use for attention layers.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    droplayer_p: Probability of dropping a layer.
    attention_order: The order to do the attention. Choice of {time_space,
      space_time}.
    dtype: the dtype of the computation (default: float32).
  """
  mlp_dim: int
  num_heads: int
  temporal_dims: int
  attention_kernel_initializer: Initializer
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  droplayer_p: Optional[float] = None
  attention_order: str = 'time_space'
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, deterministic: bool):
    """Applies Encoder1DBlock module."""
    b, thw, d = inputs.shape
    inputs = _reshape_to_time_space(inputs, self.temporal_dims)
    self_attention = functools.partial(
        nn.SelfAttention,
        num_heads=self.num_heads,
        kernel_init=self.attention_kernel_initializer,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        dtype=self.dtype)

    if self.attention_order == 'time_space':
      attention_axes = (1, 2)
    elif self.attention_order == 'space_time':
      attention_axes = (2, 1)
    else:
      raise ValueError(f'Invalid attention order {self.attention_order}.')

    def _run_attention_on_axis(inputs, axis, two_d_shape):
      """Reshapes the input and run attention on the given axis."""
      inputs = model_utils.reshape_to_1d_factorized(inputs, axis=axis)
      x = nn.LayerNorm(
          dtype=self.dtype, name='LayerNorm_{}'.format(_AXIS_TO_NAME[axis]))(
              inputs)
      x = self_attention(
          name='MultiHeadDotProductAttention_{}'.format(_AXIS_TO_NAME[axis]))(
              x, deterministic=deterministic)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
      x = x + inputs
      return model_utils.reshape_to_2d_factorized(
          x, axis=axis, two_d_shape=two_d_shape)

    x = inputs
    two_d_shape = inputs.shape
    for axis in attention_axes:
      x = _run_attention_on_axis(x, axis, two_d_shape)

    # MLP block.
    x = jnp.reshape(x, [b, thw, d])
    y = nn.LayerNorm(dtype=self.dtype, name='LayerNorm_mlp')(x)
    y = attention_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        name='MlpBlock')(
            y, deterministic=deterministic)
    return x + y


class Encoder(nn.Module):
  """Transformer Encoder.

  Attributes:
    inputs: nd-array, Input data
    temporal_dims: Number of temporal dimensions in the input.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of attention heads.
    attention_config: Has parameters for the type of attention.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_droplayer_rate: Probability of dropping a layer linearly
      grows from 0 to the provided value. Our implementation of stochastic
      depth follows timm library, which does per-example layer dropping and
      uses independent dropping patterns for each skip-connection.
    positional_embedding: The type of positional embedding to use. Supported
      values are {learned_1d, sinusoidal_1d, sinusoidal_3d, none}.
    normalise_output: If True, perform layernorm on the output.
  """

  temporal_dims: Optional[int]
  mlp_dim: int
  num_layers: int
  num_heads: int
  attention_config: ml_collections.ConfigDict = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_droplayer_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32
  positional_embedding: str = 'learned_1d'
  normalise_output: bool = True

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, train: bool):
    """Applies Transformer model on the inputs."""
    assert inputs.ndim == 3  # (batch, len, emb)
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)

    if self.positional_embedding == 'learned_1d':
      x = vit.AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='posembed_input')(inputs)
    elif self.positional_embedding == 'sinusoidal_1d':
      x = attention_layers.Add1DPositionEmbedding(
          posemb_init=None)(inputs)
    elif self.positional_embedding == 'sinusoidal_3d':
      batch, num_tokens, hidden_dim = inputs.shape
      height = width = int(np.sqrt(num_tokens // self.temporal_dims))
      if height * width * self.temporal_dims != num_tokens:
        raise ValueError('Input is assumed to be square for sinusoidal init.')
      inputs_reshape = inputs.reshape([batch, self.temporal_dims, height, width,
                                       hidden_dim])
      x = attention_layers.AddFixedSinCosPositionEmbedding()(inputs_reshape)
      x = x.reshape([batch, num_tokens, hidden_dim])
    elif self.positional_embedding == 'none':
      x = inputs
    else:
      raise ValueError(
          f'Unknown positional embedding {self.positional_embedding}')
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    if self.attention_config is None or self.attention_config.type in [
        'spacetime', 'factorized_encoder'
    ]:
      encoder_block = EncoderBlock
    elif self.attention_config.type == 'factorized_self_attention_block':
      encoder_block = functools.partial(
          EncoderFactorizedSelfAttentionBlock,
          attention_order=self.attention_config.attention_order,
          attention_kernel_initializer=KERNEL_INITIALIZERS[
              self.attention_config.get('attention_kernel_init_method',
                                        'xavier')],
          temporal_dims=self.temporal_dims)
    elif self.attention_config.type == 'factorized_dot_product_attention':
      b, thw, d = x.shape
      x = _reshape_to_time_space(x, self.temporal_dims)  # [b, t, hw, d]
      encoder_block = functools.partial(
          EncoderBlock,
          attention_fn=functools.partial(
              model_utils.factorized_dot_product_attention))
    else:
      raise ValueError(f'Unknown attention type {self.attention_config.type}')

    # Input Encoder
    for lyr in range(self.num_layers):
      droplayer_p = (
          lyr / max(self.num_layers - 1, 1)) * self.stochastic_droplayer_rate
      x = encoder_block(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          droplayer_p=droplayer_p,
          name=f'encoderblock_{lyr}',
          dtype=dtype)(
              x, deterministic=not train)

    if self.attention_config.type == 'factorized_dot_product_attention':
      # Reshape back to 3D:
      x = jnp.reshape(x, [b, thw, d])

    if self.normalise_output:
      encoded = nn.LayerNorm(name='encoder_norm')(x)
    else:
      encoded = x

    return encoded


class ViViT(nn.Module):
  """Vision Transformer model for Video.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_classes: Number of output classes.
    num_heads: Number of self-attention heads.
    num_layers: Number of layers.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    representation_size: Size of the representation layer in the model's head.
      if None, we skip the extra projection + tanh activation at the end.
    temporal_encoding_config: ConfigDict which defines the type of input
      encoding when tokenising the video.
    attention_config: ConfigDict which defines the type of spatio-temporal
      attention applied in the model.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_droplayer_rate: Probability of dropping a layer. Linearly
      increases from 0 to the provided value..
    classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
      'token'.
    return_prelogits: If true, return the final representation of the network
      before the classification head. Useful when using features for a
      downstream task.
    return_preclassifier: If true, return the representation after the
      transformer encoder. Useful if using this as the backbone stem as part
      of a bigger architecture.
    dtype: JAX data type for activations.
  """

  mlp_dim: int
  num_layers: int
  num_heads: int
  num_classes: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  temporal_encoding_config: ml_collections.ConfigDict
  attention_config: ml_collections.ConfigDict
  representation_size: Optional[int] = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_droplayer_rate: float = 0.
  classifier: str = 'gap'
  return_prelogits: bool = False
  return_preclassifier: bool = False
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, debug: bool = False):

    assert self.classifier in ['token', '0', 'gap', 'gmp', 'gsp']
    attention_type = self.attention_config.get('type', 'spacetime')
    if attention_type in [
        'factorized_transformer_block', 'factorized_self_attention_block',
        'factorized_dot_product_attention'
    ]:
      assert self.classifier not in ['token', '0'], (
          'For factorized_transformer_block, factorized_self_attention_block'
          'and factorized_dot_product_attention, the token classifier is not'
          'implemented.')

    x, temporal_dims = temporal_encode(
        x, self.temporal_encoding_config, self.patches, self.hidden_size)

    # If we want to add a class token, add it here.
    if self.classifier in ['token']:
      n, _, c = x.shape
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    x = Encoder(
        temporal_dims=temporal_dims,
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        attention_config=self.attention_config,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_droplayer_rate=self.stochastic_droplayer_rate,
        dtype=self.dtype,
        name='Transformer')(
            x, train=train)

    if self.return_preclassifier:
      return x

    if self.classifier in ['token', '0']:
      x = x[:, 0]
    elif self.classifier in ('gap', 'gmp', 'gsp'):
      fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
      x = fn(x, axis=list(range(1, x.ndim - 1)))

    if self.representation_size is not None:
      x = nn.Dense(self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = nn_layers.IdentityLayer(name='pre_logits')(x)

    if self.return_prelogits:
      return x
    else:
      x = nn.Dense(
          self.num_classes,
          kernel_init=nn.initializers.zeros,
          name='output_projection')(x)
      return x


class SpaceTimeViViT(nn.Module):
  """ViT model for Video with factorized space-time attention."""

  spatial_mlp_dim: int
  spatial_num_layers: int
  spatial_num_heads: int
  temporal_mlp_dim: int
  temporal_num_layers: int
  temporal_num_heads: int
  num_classes: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  temporal_encoding_config: ml_collections.ConfigDict
  attention_config: ml_collections.ConfigDict
  representation_size: Optional[int] = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_droplayer_rate: float = 0.
  classifier: str = 'gap'
  return_prelogits: bool = False
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, debug: bool = False):

    del debug
    x, _ = temporal_encode(
        x, self.temporal_encoding_config, self.patches, self.hidden_size,
        return_1d=False)
    bs, t, h, w, c = x.shape
    x = x.reshape(bs, t, h * w, c)

    def vit_body(x, mlp_dim, num_layers, num_heads, encoder_name='Transformer'):
      # If we want to add a class token, add it here.
      if self.classifier in ['token']:
        n, _, c = x.shape
        cls = self.param(f'cls_{encoder_name}', nn.initializers.zeros,
                         (1, 1, c), x.dtype)
        cls = jnp.tile(cls, [n, 1, 1])
        x = jnp.concatenate([cls, x], axis=1)

      x = Encoder(
          temporal_dims=None,  # This is unused for Factorised-Encoder
          mlp_dim=mlp_dim,
          num_layers=num_layers,
          num_heads=num_heads,
          attention_config=self.attention_config,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_droplayer_rate=self.stochastic_droplayer_rate,
          dtype=self.dtype,
          name=encoder_name)(x, train=train)

      if self.classifier in ['token', '0']:
        x = x[:, 0]
      elif self.classifier in ('gap', 'gmp', 'gsp'):
        fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
        x = fn(x, axis=list(range(1, x.ndim - 1)))
      return x

    # run attention across spacec, per frame
    x = jax.vmap(
        functools.partial(
            vit_body,
            mlp_dim=self.spatial_mlp_dim,
            num_layers=self.spatial_num_layers,
            num_heads=self.spatial_num_heads,
            encoder_name='SpatialTransformer'),
        in_axes=1,
        out_axes=1,
        axis_name='time')(
            x)
    assert x.ndim == 3 and x.shape[:2] == (bs, t)

    # run attention across time, over all frames
    if not self.attention_config.get('spatial_only_baseline', False):
      x = vit_body(
          x,
          mlp_dim=self.temporal_mlp_dim,
          num_layers=self.temporal_num_layers,
          num_heads=self.temporal_num_heads,
          encoder_name='TemporalTransformer')
    else:
      # Do global average pooling instead, as method of combining temporal info.
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))

    if self.representation_size is not None:
      x = nn.Dense(self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = nn_layers.IdentityLayer(name='pre_logits')(x)

    if self.return_prelogits:
      return x
    else:
      x = nn.Dense(
          self.num_classes,
          kernel_init=nn.initializers.zeros,
          name='output_projection')(x)
      return x


class ViViTClassificationModel(ClassificationModel):
  """Video Transformer model for n-way classification."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    attention_type = self.config.model.attention_config.get(
        'type', 'spacetime')
    if attention_type in [
        'spacetime', 'factorized_transformer_block',
        'factorized_self_attention_block', 'factorized_dot_product_attention'
    ]:
      return ViViT(
          num_classes=self.dataset_meta_data['num_classes'],
          mlp_dim=self.config.model.mlp_dim,
          num_layers=self.config.model.num_layers,
          num_heads=self.config.model.num_heads,
          representation_size=self.config.model.representation_size,
          patches=self.config.model.patches,
          hidden_size=self.config.model.hidden_size,
          temporal_encoding_config=self.config.model.temporal_encoding_config,
          attention_config=self.config.model.attention_config,
          classifier=self.config.model.classifier,
          dropout_rate=self.config.model.get('dropout_rate', 0.1),
          attention_dropout_rate=self.config.model.get(
              'attention_dropout_rate', 0.1),
          stochastic_droplayer_rate=self.config.model.get(
              'stochastic_droplayer_rate', 0),
          return_prelogits=self.config.model.get('return_prelogits', False),
          return_preclassifier=self.config.model.get(
              'return_preclassifier', False),
          dtype=model_dtype,
      )
    elif attention_type == 'factorized_encoder':
      # TODO(dehghani): Rewrite this as a type of attention in ViViT Encoder.
      return SpaceTimeViViT(
          num_classes=self.dataset_meta_data['num_classes'],
          spatial_mlp_dim=self.config.model.spatial_transformer.mlp_dim,
          spatial_num_layers=self.config.model.spatial_transformer.num_layers,
          spatial_num_heads=self.config.model.spatial_transformer.num_heads,
          temporal_mlp_dim=self.config.model.temporal_transformer.mlp_dim,
          temporal_num_layers=self.config.model.temporal_transformer
          .num_layers,
          temporal_num_heads=self.config.model.temporal_transformer.num_heads,
          representation_size=self.config.model.representation_size,
          patches=self.config.model.patches,
          hidden_size=self.config.model.hidden_size,
          temporal_encoding_config=self.config.model.temporal_encoding_config,
          attention_config=self.config.model.attention_config,
          classifier=self.config.model.classifier,
          dropout_rate=self.config.model.get('dropout_rate', 0.1),
          attention_dropout_rate=self.config.model.get(
              'attention_dropout_rate', 0.1),
          stochastic_droplayer_rate=self.config.model.get(
              'stochastic_droplayer_rate', 0),
          return_prelogits=self.config.model.get('return_prelogits', False),
          dtype=model_dtype,
      )
    else:
      raise ValueError(f'Attention type {attention_type} does not exist.')

  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one
        of the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      label, weights)```
    """
    del split  # for all splits, we return the same metric functions

    metrics = ViViT_CLASSIFICATION_METRICS
    if self.dataset_meta_data.get('num_classes', -1) <= 5:
      metrics = ViViT_CLASSIFICATION_METRICS_BASIC
    return functools.partial(
        classification_model.classification_metrics_function,
        target_is_onehot=self.dataset_meta_data.get('target_is_onehot', False),
        metrics=metrics)

  def init_from_train_state(self,
                            train_state: Any,
                            restored_train_state: Any,
                            restored_model_cfg: ml_collections.ConfigDict,
                            restore_output_proj: bool = False) -> Any:
    """Updates the train_state with data from restored_train_state."""
    attention_type = self.config.model.attention_config.get(
        'type', 'spacetime')
    if attention_type in [
        'spacetime', 'factorized_transformer_block',
        'factorized_self_attention_block', 'factorized_dot_product_attention'
    ]:
      vivit_transformer_key = 'Transformer'
    elif attention_type == 'factorized_encoder':
      vivit_transformer_key = 'SpatialTransformer'
    else:
      raise ValueError(f'Attention type {attention_type} does not exist.')
    return model_utils.initialise_from_train_state(
        self.config,
        train_state,
        restored_train_state,
        restored_model_cfg,
        restore_output_proj,
        vivit_transformer_key=vivit_transformer_key)


class ViViTMultilabelClassificationModel(vit.ViTMultiLabelClassificationModel):
  """Video Transformer model for multi-class classification."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    attention_type = self.config.model.attention_config.get(
        'type', 'spacetime')
    if attention_type in [
        'spacetime', 'factorized_transformer_block',
        'factorized_self_attention_block', 'factorized_dot_product_attention'
    ]:
      return ViViT(
          num_classes=self.dataset_meta_data['num_classes'],
          mlp_dim=self.config.model.mlp_dim,
          num_layers=self.config.model.num_layers,
          num_heads=self.config.model.num_heads,
          representation_size=self.config.model.representation_size,
          patches=self.config.model.patches,
          hidden_size=self.config.model.hidden_size,
          temporal_encoding_config=self.config.model.temporal_encoding_config,
          attention_config=self.config.model.attention_config,
          classifier=self.config.model.classifier,
          dropout_rate=self.config.model.get('dropout_rate', 0.1),
          attention_dropout_rate=self.config.model.get(
              'attention_dropout_rate', 0.1),
          stochastic_droplayer_rate=self.config.model.get(
              'stochastic_droplayer_rate', 0),
          return_prelogits=self.config.model.get('return_prelogits', False),
          return_preclassifier=self.config.model.get(
              'return_preclassifier', False),
          dtype=model_dtype,
      )
    elif attention_type == 'factorized_encoder':
      # TODO(dehghani): Rewrite this as a type of attention in ViViT Encoder.
      return SpaceTimeViViT(
          num_classes=self.dataset_meta_data['num_classes'],
          spatial_mlp_dim=self.config.model.spatial_transformer.mlp_dim,
          spatial_num_layers=self.config.model.spatial_transformer.num_layers,
          spatial_num_heads=self.config.model.spatial_transformer.num_heads,
          temporal_mlp_dim=self.config.model.temporal_transformer.mlp_dim,
          temporal_num_layers=self.config.model.temporal_transformer
          .num_layers,
          temporal_num_heads=self.config.model.temporal_transformer.num_heads,
          representation_size=self.config.model.representation_size,
          patches=self.config.model.patches,
          hidden_size=self.config.model.hidden_size,
          temporal_encoding_config=self.config.model.temporal_encoding_config,
          attention_config=self.config.model.attention_config,
          classifier=self.config.model.classifier,
          dropout_rate=self.config.model.get('dropout_rate', 0.1),
          attention_dropout_rate=self.config.model.get(
              'attention_dropout_rate', 0.1),
          stochastic_droplayer_rate=self.config.model.get(
              'stochastic_droplayer_rate', 0),
          return_prelogits=self.config.model.get('return_prelogits', False),
          dtype=model_dtype,
      )
    else:
      raise ValueError(f'Attention type {attention_type} does not exist.')

  def init_from_train_state(self,
                            train_state: Any,
                            restored_train_state: Any,
                            restored_model_cfg: ml_collections.ConfigDict,
                            restore_output_proj: bool = False) -> Any:
    """Updates the train_state with data from restored_train_state."""
    attention_type = self.config.model.attention_config.get(
        'type', 'spacetime')
    if attention_type in [
        'spacetime', 'factorized_transformer_block',
        'factorized_self_attention_block', 'factorized_dot_product_attention'
    ]:
      vivit_transformer_key = 'Transformer'
    elif attention_type == 'factorized_encoder':
      vivit_transformer_key = 'SpatialTransformer'
    else:
      raise ValueError(f'Attention type {attention_type} does not exist.')
    return model_utils.initialise_from_train_state(
        self.config,
        train_state,
        restored_train_state,
        restored_model_cfg,
        restore_output_proj,
        vivit_transformer_key=vivit_transformer_key)


class ViViTMultiHeadClassificationModel(ViViTClassificationModel):
  """Video Transformer model for multiple n-way classification."""

  def __init__(self, config, dataset_meta_data):
    super().__init__(config, dataset_meta_data)

    assert self.config.dataset_configs.get('class_splits'), (
        'dataset_configs.class_splits must be specified')
    self.class_splits = np.cumsum(self.config.dataset_configs.class_splits)
    if self.config.dataset_configs.get('split_names'):
      self.split_names = self.config.dataset_configs.split_names
    else:
      self.split_names = [str(x + 1) for x in range(len(self.class_splits))]

    assert not config.get('multicrop_softmax_logits', False), (
        'Returning softmaxed logits during multicrop evaluation is not '
        'supported for this model.')

  def loss_function(self,
                    logits: jnp.ndarray,
                    batch: base_model.Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Return softmax cross entropy loss with an L2 penalty on the weights."""
    weights = batch.get('batch_mask')

    if self.dataset_meta_data.get('target_is_onehot', False):
      one_hot_targets = batch['label']
    else:
      raise ValueError('Target labels should be one-hot.')

    if logits.shape[-1] != self.class_splits[-1]:
      raise AssertionError('Logit dimension must be equal to number of classes')

    logit_splits = jnp.split(logits, self.class_splits, axis=-1)[:-1]
    one_hot_target_splits = jnp.split(
        one_hot_targets, self.class_splits, axis=-1)[:-1]
    label_smoothing = self.config.get('label_smoothing')

    sof_ce_losses = [
        base_model_utils.weighted_softmax_cross_entropy(
            logits, one_hot_targets, weights, label_smoothing)
        for logits, one_hot_targets in zip(logit_splits, one_hot_target_splits)
    ]
    sof_ce_loss = jnp.mean(jnp.array(sof_ce_losses))

    if self.config.get('l2_decay_factor') is None:
      total_loss = sof_ce_loss
    else:
      l2_loss = base_model_utils.l2_regularization(model_params)
      total_loss = sof_ce_loss + 0.5 * self.config.l2_decay_factor * l2_loss
    return total_loss

  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one
        of the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      label, weights)```
    """
    del split  # for all splits, we return the same metric functions

    num_classes_in_each_head = (
        self.dataset_meta_data.get('class_splits', [-1]))
    minimal_num_classes = min(num_classes_in_each_head)
    def classification_metrics_function(logits, batch, metrics, class_splits,
                                        split_names):

      one_hot_targets = batch['label']
      weights = batch.get('batch_mask')  # batch_mask might not be defined

      logit_splits = jnp.split(logits, class_splits, axis=-1)[:-1]
      one_hot_target_splits = jnp.split(
          one_hot_targets, class_splits, axis=-1)[:-1]

      evaluated_metrics = {}
      total_loss = [0.0, 0.0]
      for logits_i, one_hot_targets_i, name in zip(logit_splits,
                                                   one_hot_target_splits,
                                                   split_names):
        for key, val in metrics.items():
          evaluated_metrics[
              f'{name}_{key}'] = base_model_utils.psum_metric_normalizer(
                  (val[0](logits_i, one_hot_targets_i,
                          weights), val[1](logits_i, one_hot_targets_i,
                                           weights)))
          if key == 'loss':
            total_loss[0] += evaluated_metrics[f'{name}_{key}'][0]
            total_loss[1] += evaluated_metrics[f'{name}_{key}'][1]
      evaluated_metrics['total_loss'] = total_loss

      if len(class_splits) == 2:
        pairwise_acc = base_model_utils.psum_metric_normalizer(
            (model_utils.joint_accuracy(logits, one_hot_targets, class_splits,
                                        weights),
             base_model_utils.num_examples(logits, one_hot_targets, weights)))
        eval_name = f'{split_names[0]}-{split_names[1]}'
        evaluated_metrics[f'{eval_name}_accuracy'] = pairwise_acc
        if minimal_num_classes > 5:
          pairwise_top_five = base_model_utils.psum_metric_normalizer(
              (model_utils.joint_top_k(
                  logits, one_hot_targets, class_splits, k=5, weights=weights),
               base_model_utils.num_examples(logits, one_hot_targets, weights)))
          evaluated_metrics[f'{eval_name}_accuracy_top_5'] = pairwise_top_five

      return evaluated_metrics
    metrics = ViViT_CLASSIFICATION_METRICS
    if minimal_num_classes <= 5:
      metrics = ViViT_CLASSIFICATION_METRICS_BASIC
    return functools.partial(
        classification_metrics_function,
        metrics=metrics,
        class_splits=self.class_splits,
        split_names=self.split_names)
