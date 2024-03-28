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

"""Fast Attention Models utilities."""

import abc
import enum
import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Iterable

from absl import logging
from flax import linen as nn
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np
from scenic.model_lib.layers import attention_layers

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


class LinformerEncoderAttention(nn.Module):
  """Linformer Encoder only multi-head dot-product self-attention.

  Attributes:
    num_heads: Number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    qkv_features: Dimension of the key, query, and value.
    out_features: Dimension of the last projection.
    broadcast_dropout: Use a broadcasted dropout along batch dims.
    dropout_rate: Dropout rate.
    kernel_init: Initializer for the kernel of the Dense layers.
    bias_init: Initializer for the bias of the Dense layers.
    bias: Whether pointwise QKVO dense transforms use bias.
    dtype: The dtype of the computation.
    precision: Numerical precision of the computation see `jax.lax.Precision`
      for details.
    low_rank_features: Low rank features.
    proj_mode: Supports "linear",  "mlp", or "cnn" projections.
    downsample: Supports downsampling query too.
    proj_configs: Configurations used in the low-rank projection.
    qk_attention_fn: A function that given multi-headed key, query, and value
      computes the attention and generates the new values.
  """
  num_heads: int
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.1
  kernel_init: Initializer = nn.linear.default_kernel_init
  bias_init: Initializer = nn.initializers.zeros
  use_bias: bool = True
  dtype: jnp.dtype = jnp.float32
  precision: Optional[jax.lax.Precision] = None
  low_rank_features: int = 8
  proj_mode: str = 'linear'
  downsample: bool = False
  proj_configs: Optional[Dict[Any, Any]] = None
  print('attnbro', dir(attention_layers))
  qk_attention_fn: Callable[
      ..., jnp.ndarray] = attention_layers.dot_product_attention

  @nn.compact
  def __call__(self,  # pytype: disable=annotation-type-mismatch  # jax-ndarray
               inputs_q: jnp.ndarray,
               inputs_kv: jnp.ndarray = None,
               *,
               deterministic: bool) -> jnp.ndarray:
    """Applies Linformer multi-head dot product self-attention.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: Input query of shape `[batch_sizes..., length, features]`.
      inputs_kv: Input key-vale, which is ignored in linformer.
      deterministic: Whether the model is run in deterministic mode (if so, do
        not apply dropout).

    Returns:
      Output of shape `[batch_sizes..., length features]`.
    """
    if inputs_kv is not None:
      logging.warning(
          'Ignoring inputs_kv as Linformer only supports self-attention.')
    x = inputs_q
    features = self.out_features or x.shape[-1]
    qkv_features = self.qkv_features or x.shape[-1]

    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    # Project inputs_q to multi-headed q/k/v.
    # Dimensions are then [bs, dims..., n_heads, n_features_per_head].
    dense = functools.partial(
        nn.DenseGeneral,
        features=(self.num_heads, head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision)

    query, key, value = (dense(dtype=self.dtype, name='query')(x),
                         dense(dtype=self.dtype, name='key')(x),
                         dense(dtype=self.dtype, name='value')(x))

    def _linear_low_rank_projection(key,
                                    value,
                                    features,
                                    activation=None,
                                    transpose=True,
                                    query=None):
      # By default, shared projections.
      if transpose:
        # Transpose if input is already transposed.
        key = key.transpose((0, 3, 2, 1))
        value = value.transpose((0, 3, 2, 1))
        if query is not None:
          query = query.transpose((0, 3, 2, 1))
      dense_proj = functools.partial(
          nn.DenseGeneral,
          features=features,
          axis=-1,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          use_bias=None,
          precision=self.precision,
          dtype=self.dtype)
      shared_dense_proj = dense_proj()
      key = shared_dense_proj(key)
      value = shared_dense_proj(value)
      if query is not None:
        query = shared_dense_proj(query)
      if activation is not None:
        key = activation(key)
        value = activation(value)
        if query is not None:
          query = activation(query)
      if transpose:
        key = key.transpose((0, 3, 2, 1))
        value = value.transpose((0, 3, 2, 1))
        if query is not None:
          query = query.transpose((0, 3, 2, 1))
      return key, value, query

    def _mlp_low_rank_projection(key, value, features):
      """MLP-based low rank projection function."""
      # Handle transpose outside (before and after linear low rank projections).
      key = key.transpose((0, 3, 2, 1))
      value = value.transpose((0, 3, 2, 1))
      for f in features[:-1]:
        key, value, _ = _linear_low_rank_projection(
            key, value, features=f, activation=nn.relu, transpose=False)
      # Don't apply activation on the last layer.
      key, value, _ = _linear_low_rank_projection(
          key, value, features=features[-1], activation=None, transpose=False)
      key = key.transpose((0, 3, 2, 1))
      value = value.transpose((0, 3, 2, 1))
      return key, value

    if self.proj_mode == 'linear':
      logging.info('Using linear low-rank projectors')
      if self.downsample:
        key, value, query = _linear_low_rank_projection(
            key,
            value,
            features=self.low_rank_features,
            transpose=True,
            query=query)
      else:
        key, value, _ = _linear_low_rank_projection(
            key, value, features=self.low_rank_features, transpose=True)
    elif self.proj_mode == 'mlp':
      # Note: do not support downsampling.
      logging.info('Using MLP low-rank projectors')
      key, value = _mlp_low_rank_projection(
          key, value, features=[self.low_rank_features, self.low_rank_features])
    else:
      raise NotImplementedError('This low-rank projection is not supported.')

    attention_bias = None
    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # Apply attention.
    x = self.qk_attention_fn(
        query,
        key,
        value,
        bias=attention_bias,
        broadcast_dropout=self.broadcast_dropout,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        deterministic=deterministic,
        dtype=self.dtype,
        precision=self.precision)

    # Project back to the original inputs dimensions.
    out = nn.DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        name='out')(
            x)
    return out


class PerformerEncoderAttention(nn.Module):
  """Encoder only multi-head dot-product self-attention based on Performer.

  based on: https://arxiv.org/abs/2009.14794

  Attributes:
    num_heads: Number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    qkv_features: Dimension of the key, query, and value.
    out_features: Dimension of the last projection.
    broadcast_dropout: Use a broadcasted dropout along batch dims.
    dropout_rate: Dropout rate.
    kernel_init: Initializer for the kernel of the Dense layers.
    bias_init: Initializer for the bias of the Dense layers.
    use_bias: Whether pointwise QKVO dense transforms use bias.
    dtype: The dtype of the computation.
    precision: Numerical precision of the computation see `jax.lax.Precision`
      for details.
    attention_fn_cls: Name of the attention function that is used by performer,
      which can be 'softmax' or 'generalized'.
    num_kernel_features: Number of kernel features.
    redraw: Whether to redraw (valid only if random featurees are used).
    attention_fn_configs: Configurations that is passed to the performer
      attention function.
  """
  num_heads: int
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.1
  kernel_init: Initializer = nn.linear.default_kernel_init
  bias_init: Initializer = nn.initializers.zeros
  use_bias: bool = True
  dtype: jnp.dtype = jnp.float32
  precision: Optional[jax.lax.Precision] = None
  attention_fn_cls: str = 'generalized'
  num_kernel_features: int = 256
  redraw: bool = True
  attention_fn_configs: Optional[Dict[Any, Any]] = None

  @nn.compact
  def __call__(self, inputs_q: jnp.ndarray, inputs_kv: Optional[jnp.ndarray], *,
               deterministic: bool) -> jnp.ndarray:
    """Applies multi-head dot product self-attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: Input of shape `[batch_sizes..., length, features]`.
      inputs_kv: Memory input of shape `[batch_sizes..., kv length, features]`.
      deterministic: Whether the model is running in deterministic mode (if so,
        do not apply dropout).

    Returns:
      Output of shape `[batch_sizes..., length features]`.
    """
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    if self.attention_fn_cls == 'softmax':
      qk_attention_fn = functools.partial(
          make_fast_softmax_attention,
          nb_features=self.num_kernel_features,
          redraw_features=self.redraw)
    elif self.attention_fn_cls == 'generalized':
      qk_attention_fn = make_fast_generalized_attention
    else:
      raise ValueError(f'Unknown attention_fn_cls: {self.attention_fn_cls}.')

    qk_attention_fn = (
        qk_attention_fn if self.attention_fn_configs is None else
        functools.partial(qk_attention_fn, **self.attention_fn_configs))  # pylint: disable=not-a-mapping

    return attention_layers.MultiHeadAttention(
        num_heads=self.num_heads,
        qkv_features=qkv_features,
        out_features=self.out_features,
        broadcast_dropout=self.broadcast_dropout,
        dropout_rate=self.dropout_rate,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        attention_fn=qk_attention_fn(
            qkv_features // self.num_heads, unidirectional=False),
    )(inputs_q=inputs_q, inputs_kv=inputs_kv, deterministic=deterministic)


class AttentionFunctionName(enum.Enum):
  """Defines name assigned to self attention modules."""
  STANDARD = 'standard'
  LINFORMER = 'linformer'
  PERFORMER = 'performer'


def _get_attention_module(name: str, is_self_attention=True) -> Any:
  """Returns an attention module."""
  function_name = AttentionFunctionName(name)
  if function_name == AttentionFunctionName.STANDARD:
    return attention_layers.MultiHeadAttention
  elif function_name == AttentionFunctionName.LINFORMER:
    if not is_self_attention:
      raise NotImplementedError
    else:
      return LinformerEncoderAttention
  elif function_name == AttentionFunctionName.PERFORMER:
    return PerformerEncoderAttention


def _get_variant_args(name: str) -> Any:
  """Return self-attention variant specific list of attn args."""

  standard_args = [
      'num_heads', 'x', 'qkv_features', 'out_features', 'broadcast_dropout',
      'dropout_rate', 'deterministic', 'kernel_init', 'bias_init', 'bias',
      'dtype', 'precision', 'qkv_attention_fn'
  ]

  if name == 'performer':
    return ['attention_fn_cls'] + ['num_kernel_features'] + ['redraw'
                                                            ] + standard_args
  elif name == 'linformer':
    return ['low_rank_features', 'downsample', 'proj_mode', 'proj_configs'
           ] + standard_args
  elif name == 'standard':
    return standard_args


def get_axial_1d_input(x: jnp.ndarray, axis: int):
  """Converts 2d inputs to 1d for axial attention."""

  assert x.ndim == 4, ('The input dimention should be '
                       '[batch_size, height, width, channel]')
  batch_size, height, width, channel = x.shape
  if axis == 1:
    return x.transpose((0, 2, 1, 3)).reshape(batch_size * width, height,
                                             channel)
  elif axis == 2:
    return x.reshape(batch_size * height, width, channel)


def get_axial_2d_input(x: jnp.ndarray, axis: int, two_d_shape: Tuple[int, int,
                                                                     int, int]):
  """Converts 1d inputs back to 2d after axial attention."""
  assert x.ndim == 3, ('The input dimention should be '
                       '[batch_size, height*width, channel]')
  batch_size, height, width, channel = two_d_shape
  if axis == 1:
    assert x.shape[0] == batch_size * width
    return x.reshape((batch_size, width, height, channel)).transpose(
        (0, 2, 1, 3))
  elif axis == 2:
    assert x.shape[0] == batch_size * height
    return x.reshape(two_d_shape)


class Encoder1DBlock(nn.Module):
  """1-Dimensional Transformer encoder block.

  Attributes:
    mlp_dim: dimension of the MLP on top of attention block.
    attention_configs: Configs pass to the self-attention func.
    attention_fn: Type of the self-attention function.
    dropout_rate: Dropout used in the MLP block.
    attention_dropout_rate: Dropout for attention heads.
    num_kernel_features: Number of kernel features used.
    redraw: Whether to redraw (valid only if random faturees are used).
    post_sa_fn: Function to be applied on the output of self-attention block.
    dtype: The dtype of the computation.
  """
  mlp_dim: int
  attention_configs: Dict[Any, Any]
  attention_fn: str
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  num_kernel_features: int = 256
  redraw: bool = True
  post_sa_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
  droplayer_p: float = 0.0
  dtype: jnp.ndarray = jnp.float32

  def get_drop_pattern(self, x, deterministic):
    if not deterministic and self.droplayer_p:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng('dropout'), self.droplayer_p, shape).astype('float32')
    else:
      return 0.0

  @nn.compact
  def __call__(self, inputs_q: jnp.ndarray, inputs_kv: jnp.ndarray, *,
               deterministic: bool) -> jnp.ndarray:
    """Applies Encoder1DBlock module.

    Args:
      inputs_q: Input data in shape of `[bs, len, c]`.
      inputs_kv: Memory data in shape of `[bs, memory len, c]`.
      deterministic: Whether the model is in deterministic mode (if so, do not
        apply dropout).

    Returns:
      Output after 1-d transformer encoder block.
    """
    assert inputs_q.ndim == 3
    if self.attention_fn:
      is_self_attention = inputs_kv is None

      # Attention block.
      valid_args = _get_variant_args(self.attention_fn)
      # Remove args that are potentially not needed for variant.
      attention_configs = {
          x: self.attention_configs[x]
          for x in valid_args
          if x in self.attention_configs
      }
      x = nn.LayerNorm(dtype=self.dtype)(inputs_q)
      if not is_self_attention:
        assert inputs_kv.ndim == 3
        inputs_kv = nn.LayerNorm(dtype=self.dtype)(inputs_kv)

      # Prepare the input for the attention modole.
      # We shouldn't pass memory if it is self-attention.
      init_arg_to_attention_module = {
          'kernel_init': nn.initializers.xavier_uniform(),
          'broadcast_dropout': False,
          'dtype': self.dtype,
          'dropout_rate': self.attention_dropout_rate,
      }
      inputs_to_attention_module = {
          'inputs_q': x,
          'deterministic': deterministic,
      }
      if is_self_attention:
        inputs_to_attention_module['inputs_kv'] = x
      else:
        inputs_to_attention_module['inputs_kv'] = inputs_kv

      x = _get_attention_module(
          self.attention_fn,
          is_self_attention)(**init_arg_to_attention_module,
                             **attention_configs)(**inputs_to_attention_module)

      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

      if x.shape[-2] != inputs_q.shape[-2]:
        # TODO(yitay): Support case where we downsample. How do we handle res?
        # Currently bypassing this causes training problems...
        raise ValueError('Shape not identical. Cannot add residual connection.')

      drop_pattern = self.get_drop_pattern(x, deterministic)
      x = x * (1.0 - drop_pattern) + inputs_q
      if self.post_sa_fn is not None:
        x = self.post_sa_fn(x)  # pylint: disable=not-callable
    else:
      x = inputs_q

    if self.mlp_dim is None:
      # Skip the MLP block.
      return x

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    mlp_dim = self.mlp_dim
    if isinstance(self.mlp_dim, int):
      mlp_dim = (mlp_dim,)
    for mlp_d in mlp_dim:
      y = attention_layers.MlpBlock(
          mlp_dim=mlp_d,
          dtype=self.dtype,
          dropout_rate=self.dropout_rate,
          activation_fn=nn.gelu,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6))(
              y, deterministic=deterministic)

    drop_pattern = self.get_drop_pattern(x, deterministic)
    return x + y * (1.0 - drop_pattern)


class EncoderAxialBlock(nn.Module):
  """2-Dimensional Transformer encoder block with Axial attention.

  This block is similar to Encoder1DBlock, where instead of `self-attention+MLP`
  we have `row-self-attention + col-self-attention + MLP`.

  Attributes:
    mlp_dim: dimension of the mlp on top of attention block.
    attention_configs: Configs pass to the self-attention func.
    attention_fn: Type of the sel-attention function.
    dropout_rate: Dropout used in the mlp block.
    attention_dropout_rate: Dropout for attention heads.
    factorization_axis: Axis over which we run attention.
    post_sa_fn: Function to be applied on the output of self-attention block.
    dtype: The dtype of the computation.
  """
  mlp_dim: int
  attention_configs: Dict[Any, Any]
  attention_fn: str
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  factorization_axis: Tuple[int, ...] = (1, 2)
  post_sa_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
  droplayer_p: float = 0.0
  dtype: jnp.ndarray = jnp.float32

  def get_drop_pattern(self, x, deterministic):
    if not deterministic and self.droplayer_p:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng('dropout'), self.droplayer_p, shape).astype('float32')
    else:
      return 0.0

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *,
               deterministic: bool) -> jnp.ndarray:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data in shape of `[bs, len, c]`.
      deterministic: Whether the model is in deterministic mode (if so, do not
        apply dropout).

    Returns:
      Output after axial attention encoder block.
    """

    def _run_attention_on_axis(inputs, axis, two_d_shape):
      inputs = get_axial_1d_input(inputs, axis=axis)
      x = nn.LayerNorm(dtype=self.dtype)(inputs)
      init_arg_to_attention_module = {
          'kernel_init': nn.initializers.xavier_uniform(),
          'broadcast_dropout': False,
          'dtype': self.dtype,
          'dropout_rate': self.attention_dropout_rate,
      }
      # Attention block.
      valid_args = _get_variant_args(self.attention_fn)
      # Remove args that are potentially not needed for variant.
      attention_configs = {
          x: self.attention_configs[x]
          for x in valid_args
          if x in self.attention_configs
      }
      x = _get_attention_module(
          self.attention_fn,
          is_self_attention=True)(**init_arg_to_attention_module,
                                  **attention_configs)(
                                      x, deterministic=deterministic)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
      drop_pattern = self.get_drop_pattern(x, deterministic)
      x = x * (1.0 - drop_pattern) + inputs
      return get_axial_2d_input(x, axis=axis, two_d_shape=two_d_shape)

    x = inputs
    if self.attention_fn:
      # Row attention block.
      two_d_shape = inputs.shape

      for ax in self.factorization_axis:
        x = _run_attention_on_axis(x, ax, two_d_shape)

      if self.post_sa_fn is not None:
        x = self.post_sa_fn(x)  # pylint: disable=not-callable

    if self.mlp_dim is None:
      # Skip the MLP block.
      return x

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    mlp_dim = self.mlp_dim
    if isinstance(self.mlp_dim, int):
      mlp_dim = (mlp_dim,)
    for mlp_d in mlp_dim:
      y = attention_layers.MlpBlock(
          mlp_dim=mlp_d,
          dtype=self.dtype,
          dropout_rate=self.dropout_rate,
          activation_fn=nn.gelu,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6))(
              y, deterministic=deterministic)

    drop_pattern = self.get_drop_pattern(x, deterministic)
    return x + y * (1.0 - drop_pattern)


def sample_categorical(rng: jnp.ndarray,
                       logits: jnp.ndarray,
                       num_samples: int,
                       *,
                       replacement: bool = True):
  """Sample catogorical with or without replacement for the top-k selector.

  Args:
    rng: JAX PRNG key.
    logits: Categorical distribution logits of shape [batch_dims, num_classes].
    num_samples: Number of samples to produce.
    replacement: If True, sampling is done with replacement.

  Returns:
    Categorial samples of shape [batch_dims, num_samples].
  """
  rng = jax.random.split(rng, num_samples)
  if replacement:
    samples = jax.vmap(jax.random.categorical, in_axes=(0, None))(rng, logits)
  else:
    num_categories = logits.shape[-1]
    if num_categories < num_samples:
      raise ValueError(f'Number of samples ({num_samples}) must be <= number of'
                       f' categories ({num_categories}) when sampling without'
                       f' replacement.')

    def sample_one(logits, scan_rng):
      samples = jax.random.categorical(scan_rng, logits, axis=-1)
      mask = jax.nn.one_hot(samples, num_categories, dtype=jnp.bool_)
      logits = jnp.where(mask, -1e10, logits)
      return logits, samples

    _, samples = jax.lax.scan(sample_one, logits, rng)

  # Restore original shape.
  ndim = samples.ndim
  if ndim > 1:
    samples = jnp.transpose(samples, axes=tuple(range(1, ndim)) + (0,))
  return samples


class TopKTokenSelector(nn.Module):
  """A layer that selects top-k tokens.

  Note that if `pool_unselected_tokens` is set to True, it pools all the
  unselected tokens and appends it as an extra tokens and returns k+1 tokens.

  Attributes:
    top_k: Number of tokens we select.
    sample_tokens: Whether sample the top-k tokens given their scores or just
      take the top-k.
    pool_unselected_tokens: Whether we pool the unselected tokens and attach the
      pooled version as an extra token to the selected tokens.
    exclude_cls: If set to True, it assumes the token at position 0 is CLS token
      and should be excluded from the selection process and be attached back at
      the end.
    score_net_kernel_init: Kernel initialization for the score net.
    dtype: Jax dtype.
  """
  top_k: int
  sample_tokens: bool
  pool_unselected_tokens: bool
  exclude_cls: bool = False
  score_net_kernel_init: Initializer = nn.linear.default_kernel_init
  dtype: jnp.ndarray = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, train: bool) -> jnp.ndarray:
    if self.exclude_cls:
      cls, inputs = jnp.split(inputs, (1,), axis=1)
    input_len = inputs.shape[1]
    if self.top_k > input_len:
      raise ValueError(f'The value of top_{self.top_k} should be less than'
                       f'input length:{input_len}.')
    logging.info('Selecting %d tokens out of %d tokens.', self.top_k, input_len)
    # TODO(dehghani): Explore if adding a non-linearity to the score_net helps.
    score_logits = jnp.squeeze(
        nn.Dense(
            features=1,
            dtype=self.dtype,
            kernel_init=self.score_net_kernel_init,
            # No bias is needed since it gets removed during normalization.
            use_bias=False,
            name='score_net')(inputs),
        axis=-1)

    if train and self.sample_tokens and self.top_k < input_len:
      # We use dropout rng for sampling, which is always provided.
      rng = self.make_rng('dropout')
      selected_index = sample_categorical(
          rng, score_logits, self.top_k, replacement=False)
      selected_logits = jax.vmap(jnp.take, (0, 0, None))(score_logits,
                                                         selected_index, 0)
    else:
      selected_logits, selected_index = jax.lax.top_k(score_logits, self.top_k)

    # Take selected tokens:
    selected_tokens = jax.vmap(jnp.take, (0, 0, None))(inputs, selected_index,
                                                       0)
    # Normalize "selected logits" and used as weights for selected tokens:
    selected_tokens = selected_tokens * jax.nn.softmax(selected_logits)[
        ..., jnp.newaxis]

    if self.pool_unselected_tokens and self.top_k < input_len:
      # Extract index of unselected tokens:
      selected_index_one_hot = jax.nn.one_hot(
          selected_index, num_classes=input_len, dtype=jnp.bool_)
      unselected_index_one_hot = jnp.any(
          jnp.logical_not(selected_index_one_hot), axis=1)
      _, unselected_index = jax.lax.top_k(unselected_index_one_hot,
                                          input_len - self.top_k)

      # Take unselected tokens:
      unselected_tokens = jax.vmap(jnp.take, (0, 0, None))(inputs,
                                                           unselected_index, 0)
      unselected_logits = jax.vmap(jnp.take, (0, 0, None))(score_logits,
                                                           unselected_index, 0)
      # Normalize "unselected logits" and used as weights for unselected tokens:
      weighted_unselected_tokens = (
          unselected_tokens *
          jax.nn.softmax(unselected_logits)[..., jnp.newaxis])
      unselected_tokens_rep = jnp.sum(
          weighted_unselected_tokens, axis=1, keepdims=True)

      selected_tokens = jnp.concatenate(
          [selected_tokens, unselected_tokens_rep], axis=1)

    if self.exclude_cls:
      selected_tokens = jnp.concatenate([cls, selected_tokens], axis=1)
    return selected_tokens


###### PERFORMER FUNCTIONS:
def nonnegative_softmax_kernel_feature_creator(data,
                                               projection_matrix,
                                               attention_dims_t,
                                               batch_dims_t,
                                               precision,
                                               is_query,
                                               normalize_data=True,
                                               eps=0.0001):
  """Constructs nonnegative kernel features for fast softmax attention.

  Args:
    data: input for which features are computes
    projection_matrix: random matrix used to compute features
    attention_dims_t: tuple of attention dimensions
    batch_dims_t: tuple of batch dimensions
    precision: precision parameter
    is_query: predicate indicating whether input data corresponds to queries or
      keys
    normalize_data: predicate indicating whether data should be normalized,
    eps: numerical stabilizer.

  Returns:
    Random features for fast softmax attention.
  """

  if normalize_data:
    # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
    # w_norm = w * data_normalizer for w in {q,k}.
    data_normalizer = 1.0 / (jnp.sqrt(jnp.sqrt(data.shape[-1])))
  else:
    data_normalizer = 1.0
  ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
  data_mod_shape = data.shape[0:len(batch_dims_t)] + projection_matrix.shape
  data_thick_random_matrix = jnp.zeros(data_mod_shape) + projection_matrix

  data_dash = lax.dot_general(
      data_normalizer * data,
      data_thick_random_matrix,
      (((data.ndim - 1,), (data_thick_random_matrix.ndim - 1,)),
       (batch_dims_t, batch_dims_t)),
      precision=precision)

  diag_data = jnp.square(data)
  diag_data = jnp.sum(diag_data, axis=data.ndim - 1)
  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
  diag_data = jnp.expand_dims(diag_data, axis=data.ndim - 1)

  last_dims_t = (len(data_dash.shape) - 1,)
  if is_query:
    data_dash = ratio * (
        jnp.exp(data_dash - diag_data -
                jnp.max(data_dash, axis=last_dims_t, keepdims=True)) + eps)
  else:
    data_dash = ratio * (
        jnp.exp(data_dash - diag_data - jnp.max(
            data_dash, axis=last_dims_t + attention_dims_t, keepdims=True)) +
        eps)

  return data_dash


def sincos_softmax_kernel_feature_creator(data,
                                          projection_matrix,
                                          attention_dims_t,
                                          batch_dims_t,
                                          precision,
                                          normalize_data=True):
  """Constructs kernel sin-cos features for fast softmax attention.

  Args:
    data: input for which features are computes
    projection_matrix: random matrix used to compute features
    attention_dims_t: tuple of attention dimensions
    batch_dims_t: tuple of batch dimensions
    precision: precision parameter
    normalize_data: predicate indicating whether data should be normalized.

  Returns:
    Random features for fast softmax attention.
  """
  if normalize_data:
    # We have: exp(qk^T/sqrt{d}) = exp(|q|^2/2sqrt{d}) * exp(|k|^2/2sqrt{d}) *
    # exp(-(|q*c-k*c|^2)/2), where c = 1.0 / sqrt{sqrt{d}}.
    data_normalizer = 1.0 / (jnp.sqrt(jnp.sqrt(data.shape[-1])))
  else:
    data_normalizer = 1.0
  ratio = 1.0 / jnp.sqrt(projection_matrix.shape[0])
  data_mod_shape = data.shape[0:len(batch_dims_t)] + projection_matrix.shape
  data_thick_random_matrix = jnp.zeros(data_mod_shape) + projection_matrix

  data_dash = lax.dot_general(
      data_normalizer * data,
      data_thick_random_matrix,
      (((data.ndim - 1,), (data_thick_random_matrix.ndim - 1,)),
       (batch_dims_t, batch_dims_t)),
      precision=precision)
  data_dash_cos = ratio * jnp.cos(data_dash)
  data_dash_sin = ratio * jnp.sin(data_dash)
  data_dash = jnp.concatenate((data_dash_cos, data_dash_sin), axis=-1)

  # Constructing D_data and data^{'}
  diag_data = jnp.square(data)
  diag_data = jnp.sum(diag_data, axis=data.ndim - 1)
  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
  diag_data = jnp.expand_dims(diag_data, axis=data.ndim - 1)
  # Additional renormalization for numerical stability
  data_renormalizer = jnp.max(diag_data, attention_dims_t, keepdims=True)
  diag_data -= data_renormalizer
  diag_data = jnp.exp(diag_data)
  data_prime = data_dash * diag_data
  return data_prime


def generalized_kernel_feature_creator(data, projection_matrix, batch_dims_t,
                                       precision, kernel_fn, kernel_epsilon,
                                       normalize_data):
  """Constructs kernel features for fast generalized attention.

  Args:
    data: input for which features are computes
    projection_matrix: matrix used to compute features
    batch_dims_t: tuple of batch dimensions
    precision: precision parameter
    kernel_fn: kernel function used
    kernel_epsilon: additive positive term added to every feature for numerical
      stability
    normalize_data: predicate indicating whether data should be normalized.

  Returns:
    Random features for fast generalized attention.
  """
  if normalize_data:
    data_normalizer = 1.0 / (jnp.sqrt(jnp.sqrt(data.shape[-1])))
  else:
    data_normalizer = 1.0
  if projection_matrix is None:
    return kernel_fn(data_normalizer * data) + kernel_epsilon
  else:
    data_mod_shape = data.shape[0:len(batch_dims_t)] + projection_matrix.shape
    data_thick_random_matrix = jnp.zeros(data_mod_shape) + projection_matrix
    data_dash = lax.dot_general(
        data_normalizer * data,
        data_thick_random_matrix,
        (((data.ndim - 1,), (data_thick_random_matrix.ndim - 1,)),
         (batch_dims_t, batch_dims_t)),
        precision=precision)
  data_prime = kernel_fn(data_dash) + kernel_epsilon
  return data_prime


def make_fast_softmax_attention(qkv_dim,
                                renormalize_attention=True,
                                numerical_stabilizer=0.000001,
                                nb_features=256,
                                ortho_features=True,
                                ortho_scaling=0.0,
                                redraw_features=True,
                                unidirectional=False,
                                nonnegative_features=True,
                                lax_scan_unroll=1):
  """Construct a fast softmax attention method."""
  '''
  logging.info(
      'Fast softmax attention: %s features and orthogonal=%s, renormalize=%s',
      nb_features, ortho_features, renormalize_attention)
  '''
  if ortho_features:
    matrix_creator = functools.partial(
        GaussianOrthogonalRandomMatrix,
        nb_features,
        qkv_dim,
        scaling=ortho_scaling)
  else:
    matrix_creator = functools.partial(GaussianUnstructuredRandomMatrix,
                                       nb_features, qkv_dim)
  if nonnegative_features:

    def kernel_feature_creator(data,
                               projection_matrix,
                               attention_dims_t,
                               batch_dims_t,
                               precision,
                               is_query,
                               normalize_data=True):
      return nonnegative_softmax_kernel_feature_creator(
          data, projection_matrix, attention_dims_t, batch_dims_t, precision,
          is_query, normalize_data, numerical_stabilizer)
  else:

    def kernel_feature_creator(data,
                               projection_matrix,
                               attention_dims_t,
                               batch_dims_t,
                               precision,
                               is_query,
                               normalize_data=True):
      del is_query
      return sincos_softmax_kernel_feature_creator(data, projection_matrix,
                                                   attention_dims_t,
                                                   batch_dims_t, precision,
                                                   normalize_data)

  attention_fn = FastAttentionviaLowRankDecomposition(
      matrix_creator,
      kernel_feature_creator,
      renormalize_attention=renormalize_attention,
      numerical_stabilizer=numerical_stabilizer,
      redraw_features=redraw_features,
      unidirectional=unidirectional,
      lax_scan_unroll=lax_scan_unroll).dot_product_attention
  return attention_fn


def make_fast_generalized_attention(qkv_dim,
                                    renormalize_attention=True,
                                    numerical_stabilizer=0.0,
                                    nb_features=256,
                                    features_type='deterministic',
                                    kernel_fn=jax.nn.relu,
                                    kernel_epsilon=0.001,
                                    redraw_features=False,
                                    unidirectional=False,
                                    lax_scan_unroll=1):
  """Construct a fast generalized attention menthod."""
  '''
  logging.info('Fast generalized attention.: %s features and renormalize=%s',
               nb_features, renormalize_attention)
  '''
  if features_type == 'ortho':
    matrix_creator = functools.partial(
        GaussianOrthogonalRandomMatrix, nb_features, qkv_dim, scaling=False)
  elif features_type == 'iid':
    matrix_creator = functools.partial(GaussianUnstructuredRandomMatrix,
                                       nb_features, qkv_dim)
  elif features_type == 'deterministic':
    matrix_creator = None
  else:
    raise ValueError('Unknown feature value type')

  def kernel_feature_creator(data,
                             projection_matrix,
                             attention_dims_t,
                             batch_dims_t,
                             precision,
                             is_query,
                             normalize_data=False):
    del attention_dims_t
    del is_query
    return generalized_kernel_feature_creator(data, projection_matrix,
                                              batch_dims_t, precision,
                                              kernel_fn, kernel_epsilon,
                                              normalize_data)

  attention_fn = FastAttentionviaLowRankDecomposition(
      matrix_creator,
      kernel_feature_creator,
      renormalize_attention=renormalize_attention,
      numerical_stabilizer=numerical_stabilizer,
      redraw_features=redraw_features,
      unidirectional=unidirectional,
      lax_scan_unroll=lax_scan_unroll).dot_product_attention
  return attention_fn


class RandomMatrix(metaclass=abc.ABCMeta):
  """Abstract class providing a method for constructing 2D random arrays.

  Class is responsible for constructing 2D random arrays.
  """

  @abc.abstractmethod
  def get_2d_array(self):
    raise NotImplementedError('Abstract method')


class GaussianUnstructuredRandomMatrix(RandomMatrix):

  def __init__(self, nb_rows, nb_columns, key):
    self.nb_rows = nb_rows
    self.nb_columns = nb_columns
    self.key = key

  def get_2d_array(self):
    return random.normal(self.key, (self.nb_rows, self.nb_columns))


class GaussianOrthogonalRandomMatrix(RandomMatrix):
  r"""Class providing a method to create Gaussian orthogonal matrix.

  Class is responsible for constructing 2D Gaussian orthogonal arrays.
  """

  def __init__(self, nb_rows, nb_columns, key, scaling=0):
    self.nb_rows = nb_rows
    self.nb_columns = nb_columns
    self.key = key
    self.scaling = scaling

  def get_2d_array(self):
    nb_full_blocks = int(self.nb_rows / self.nb_columns)
    block_list = []
    rng = self.key
    for _ in range(nb_full_blocks):
      rng, rng_input = jax.random.split(rng)
      unstructured_block = random.normal(rng_input,
                                         (self.nb_columns, self.nb_columns))
      q, _ = jnp.linalg.qr(unstructured_block)
      q = jnp.transpose(q)
      block_list.append(q)
    remaining_rows = self.nb_rows - nb_full_blocks * self.nb_columns
    if remaining_rows > 0:
      rng, rng_input = jax.random.split(rng)
      unstructured_block = random.normal(rng_input,
                                         (self.nb_columns, self.nb_columns))
      q, _ = jnp.linalg.qr(unstructured_block)
      q = jnp.transpose(q)
      block_list.append(q[0:remaining_rows])
    final_matrix = jnp.vstack(block_list)

    if self.scaling == 0:
      multiplier = jnp.linalg.norm(
          random.normal(self.key, (self.nb_rows, self.nb_columns)), axis=1)
    elif self.scaling == 1:
      multiplier = jnp.sqrt(float(self.nb_columns)) * jnp.ones((self.nb_rows))
    else:
      raise ValueError('Scaling must be one of {0, 1}. Was %s' % self.scaling)

    return jnp.matmul(jnp.diag(multiplier), final_matrix)


class FastAttention(metaclass=abc.ABCMeta):
  """Abstract class providing a method for fast attention.

  Class is responsible for providing a method <dot_product_attention> for fast
  approximate attention.
  """

  @abc.abstractmethod
  def dot_product_attention(self,
                            query,
                            key,
                            value,
                            dtype=jnp.float32,
                            bias=None,
                            axis=None,
                            broadcast_dropout=True,
                            dropout_rng=None,
                            dropout_rate=0.,
                            deterministic=False,
                            precision=None):
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying fast approximate dot-product
    attention. It calculates the attention weights given query and key and
    combines the values using the attention weights. This function supports
    multi-dimensional inputs.
    Args:
      query: queries for calculating attention with shape of [batch_size, dim1,
        dim2, ..., dimN, num_heads, mem_channels].
      key: keys for calculating attention with shape of [batch_size, dim1, dim2,
        ..., dimN, num_heads, mem_channels].
      value: values to be used in attention with shape of [batch_size, dim1,
        dim2,..., dimN, num_heads, value_channels].
      dtype: the dtype of the computation (default: float32)
      bias: bias for the attention weights. This can be used for incorporating
        autoregressive mask, padding mask, proximity bias.
      axis: axises over which the attention is applied.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout.
      dropout_rate: dropout rate.
      deterministic: bool, deterministic or not (to apply dropout).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.

    Returns:
      Output of shape [bs, dim1, dim2, ..., dimN,, num_heads, value_channels].
    """
    raise NotImplementedError('Abstract method')


def _numerator(z_slice_shape, precision, unroll=1):
  """Computes the numartor."""

  def fwd(qs, ks, vs):

    def body(p, qkv):
      (q, k, v) = qkv
      p += jnp.einsum('...m,...d->...md', k, v, precision=precision)
      x_slice = jnp.einsum('...m,...md->...d', q, p, precision=precision)
      return p, x_slice

    init_value = jnp.zeros(z_slice_shape)
    p, w = lax.scan(body, init_value, (qs, ks, vs), unroll=unroll)
    return w, (p, qs, ks, vs)

  def bwd(pqkv, w_ct):

    def body(carry, qkv_xct):
      p, p_ct = carry
      q, k, v, x_ct = qkv_xct
      q_ct = jnp.einsum('...d,...md->...m', x_ct, p, precision=precision)
      p_ct += jnp.einsum('...d,...m->...md', x_ct, q, precision=precision)
      k_ct = jnp.einsum('...md,...d->...m', p_ct, v, precision=precision)
      v_ct = jnp.einsum('...md,...m->...d', p_ct, k, precision=precision)
      p -= jnp.einsum('...m,...d->...md', k, v, precision=precision)
      return (p, p_ct), (q_ct, k_ct, v_ct)

    p, qs, ks, vs = pqkv
    _, (qs_ct, ks_ct, vs_ct) = lax.scan(
        body, (p, jnp.zeros_like(p)), (qs, ks, vs, w_ct),
        reverse=True,
        unroll=unroll)
    return qs_ct, ks_ct, vs_ct

  @jax.custom_vjp
  def _numerator_impl(qs, ks, vs):
    w, _ = fwd(qs, ks, vs)
    return w

  _numerator_impl.defvjp(fwd, bwd)

  return _numerator_impl


def _denominator(t_slice_shape, precision, unroll=1):
  """Computes the denominator."""

  def fwd(qs, ks):

    def body(p, qk):
      q, k = qk
      p += k
      x = jnp.einsum('...m,...m->...', q, p, precision=precision)
      return p, x

    p = jnp.zeros(t_slice_shape)
    p, r = lax.scan(body, p, (qs, ks), unroll=unroll)
    return r, (qs, ks, p)

  def bwd(qkp, r_ct):

    def body(carry, qkx):
      p, p_ct = carry
      q, k, x_ct = qkx
      q_ct = jnp.einsum('...,...m->...m', x_ct, p, precision=precision)
      p_ct += jnp.einsum('...,...m->...m', x_ct, q, precision=precision)
      k_ct = p_ct
      p -= k
      return (p, p_ct), (q_ct, k_ct)

    qs, ks, p = qkp
    _, (qs_ct, ks_ct) = lax.scan(
        body, (p, jnp.zeros_like(p)), (qs, ks, r_ct),
        reverse=True,
        unroll=unroll)
    return (qs_ct, ks_ct)

  @jax.custom_vjp
  def _denominator_impl(qs, ks):
    r, _ = fwd(qs, ks)
    return r

  _denominator_impl.defvjp(fwd, bwd)

  return _denominator_impl


class FastAttentionviaLowRankDecomposition(FastAttention):
  """Class providing a method for fast attention via low rank decomposition.

  Class is responsible for providing a method <dot_product_attention> for fast
  dot-product attention with the use of low rank decomposition (e.g. with
  random feature maps).
  """

  def __init__(self,
               matrix_creator,
               kernel_feature_creator,
               renormalize_attention,
               numerical_stabilizer,
               redraw_features,
               unidirectional,
               lax_scan_unroll=1):  # For optimal GPU performance, set to 16.
    rng = random.PRNGKey(0)
    self.matrix_creator = matrix_creator
    self.projection_matrix = self.draw_weights(rng)
    self.kernel_feature_creator = kernel_feature_creator
    self.renormalize_attention = renormalize_attention
    self.numerical_stabilizer = numerical_stabilizer
    self.redraw_features = redraw_features
    self.unidirectional = unidirectional
    self.lax_scan_unroll = lax_scan_unroll

  def draw_weights(self, key):
    if self.matrix_creator is None:
      return None
    matrixrng, _ = random.split(key)
    projection_matrix = self.matrix_creator(key=matrixrng).get_2d_array()
    return projection_matrix

  def dot_product_attention(self,
                            query,
                            key,
                            value,
                            dtype=jnp.float32,
                            bias=None,
                            axis=None,
                            broadcast_dropout=True,
                            dropout_rng=None,
                            dropout_rate=0.,
                            deterministic=False,
                            precision=None):

    assert key.shape[:-1] == value.shape[:-1]
    assert (query.shape[0:1] == key.shape[0:1] and
            query.shape[-1] == key.shape[-1])
    if axis is None:
      axis = tuple(range(1, key.ndim - 2))
    if not isinstance(axis, Iterable):
      axis = (axis,)
    assert key.ndim == query.ndim
    assert key.ndim == value.ndim
    for ax in axis:
      if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
        raise ValueError('Attention axis must be between the batch '
                         'axis and the last-two axes.')
    n = key.ndim

    # Constructing projection tensor.
    if self.redraw_features:
      query_seed = lax.convert_element_type(
          jnp.ceil(jnp.sum(query) * 10000000.0), jnp.int32)
      rng = random.PRNGKey(query_seed)
      self.projection_matrix = self.draw_weights(rng)

    # batch_dims is  <bs, <non-attention dims>, num_heads>
    batch_dims = tuple(np.delete(range(n), axis + (n - 1,)))
    # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
    qk_perm = batch_dims + axis + (n - 1,)
    k_extra_perm = axis + batch_dims + (n - 1,)
    key_extra = key.transpose(k_extra_perm)
    key = key.transpose(qk_perm)
    query = query.transpose(qk_perm)
    # v -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
    v_perm = batch_dims + axis + (n - 1,)
    value = value.transpose(v_perm)
    batch_dims_t = tuple(range(len(batch_dims)))
    attention_dims_t = tuple(
        range(len(batch_dims),
              len(batch_dims) + len(axis)))

    # Constructing tensors Q^{'} and K^{'}.
    query_prime = self.kernel_feature_creator(query, self.projection_matrix,
                                              attention_dims_t, batch_dims_t,
                                              precision, True)
    query_prime = query_prime.astype(dtype)
    key_prime = self.kernel_feature_creator(key, self.projection_matrix,
                                            attention_dims_t, batch_dims_t,
                                            precision, False)
    key_prime = key_prime.astype(dtype)

    if self.unidirectional:
      index = attention_dims_t[0]
      z_slice_shape = key_prime.shape[0:len(batch_dims_t)] + (
          key_prime.shape[-1],) + (value.shape[-1],)

      numerator_fn = _numerator(z_slice_shape, precision, self.lax_scan_unroll)
      w = numerator_fn(
          jnp.moveaxis(query_prime, index, 0),
          jnp.moveaxis(key_prime, index, 0), jnp.moveaxis(value, index, 0))

      # Constructing w = (Q^{'}(K^{'})^{T})_{masked}V
      w = jnp.moveaxis(w, 0, index)

      if not self.renormalize_attention:
        # Unidirectional, not-normalized attention.
        perm_inv = _invert_perm(qk_perm)
        result = w.transpose(perm_inv)
        return result
      else:
        # Unidirectional, normalized attention.
        thick_all_ones = jnp.zeros(key.shape[0:-1]) + jnp.ones(
            key_extra.shape[0:len(axis)])

        index = attention_dims_t[0]
        t_slice_shape = key_prime.shape[0:len(batch_dims_t)] + (
            key_prime.shape[-1],)
        denominator_fn = _denominator(t_slice_shape, precision,
                                      self.lax_scan_unroll)
        r = denominator_fn(
            jnp.moveaxis(query_prime, index, 0),
            jnp.moveaxis(key_prime, index, 0))

        r = jnp.moveaxis(r, 0, index)
    else:
      contract_query = tuple(
          range(len(batch_dims) + len(axis),
                len(batch_dims) + len(axis) + 1))
      contract_z = tuple(range(len(batch_dims), len(batch_dims) + 1))
      # Constructing  z = (K^{'})^{T}V
      #  z (bs, <non-attention dims>, num_heads, channels_m, channels_v)
      z = lax.dot_general(
          key_prime,
          value,
          ((attention_dims_t, attention_dims_t), (batch_dims_t, batch_dims_t)),
          precision=precision)
      # Constructing w = Q^{'} z = Q^{'}(K^{'})^{T}V
      # q (bs, <non-attention dims>, num_heads, <attention dims>, channels_m)
      #  z (bs, <non-attention dims>, num_heads, channels_m, channels_v)
      # w (bs,  <non-attention dims>, num_heads, <attention dims>, channels_v)
      w = lax.dot_general(
          query_prime,
          z, ((contract_query, contract_z), (batch_dims_t, batch_dims_t)),
          precision=precision)
      if not self.renormalize_attention:
        # Bidirectional, not-normalized attention.
        perm_inv = _invert_perm(qk_perm)
        result = w.transpose(perm_inv)
        return result
      else:
        # Bidirectional, normalized attention.
        thick_all_ones = jnp.zeros(key.shape[0:-1]) + jnp.ones(
            key_extra.shape[0:len(axis)])
        thick_all_ones = thick_all_ones.astype(dtype)
        contract_key = tuple(
            range(len(batch_dims),
                  len(batch_dims) + len(axis)))
        contract_thick_all_ones = tuple(
            range(thick_all_ones.ndim - len(axis), thick_all_ones.ndim))
        # Construct t = (K^{'})^{T} 1_L
        # k (bs, <non-attention dims>, num_heads, <attention dims>, channels)
        t = lax.dot_general(
            key_prime,
            thick_all_ones, ((contract_key, contract_thick_all_ones),
                             (batch_dims_t, batch_dims_t)),
            precision=precision)

        # Construct partition function: r = Q^{'} t = Q^{'}(K^{'})^{T} 1_L
        # q_p (bs, <non-attention dims>, num_heads, <attention dims>, channs_m)
        # t   (bs, <non-attention dims>, num_heads, channels_m)
        r = lax.dot_general(
            query_prime,
            t, (((query_prime.ndim - 1,), (t.ndim - 1,)),
                (batch_dims_t, range(0,
                                     len(t.shape) - 1))),
            precision=precision)

    r = r + 2 * self.numerical_stabilizer * (
        jnp.abs(r) <= self.numerical_stabilizer)
    r = jnp.reciprocal(r)
    r = jnp.expand_dims(r, len(r.shape))
    # w (bs, <non-attention dims>, num_heads, <attention dims>, channels_v)
    # r (bs, <non-attention dims>, num_heads, <attention dims>, extra_channel)
    result = w * r
    # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
    perm_inv = _invert_perm(qk_perm)
    result = result.transpose(perm_inv)
    return result


def _invert_perm(perm):
  perm_inv = [0] * len(perm)
  for i, j in enumerate(perm):
    perm_inv[j] = i
  return tuple(perm_inv)
