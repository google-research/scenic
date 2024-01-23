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

"""Common attention modules.

Conventions:
- Pass `deterministic` and `rng` as an argument. `rng` is optional and defaults
  to `self.make_rng()`.
- `train` and `deterministic` should not have a default.
- Do not define `rng`, `deterministic` or `train` as attributes.
- `rng`, `deterministic`, `train` should always be keyword only arguments.
- Prefer `use_bias` over `bias`.
"""
import functools
from typing import Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from scenic.model_lib.layers import nn_layers

# TODO(mrit): Upstream this to jax.nn.initializers
# Inputs are PRNGKey, input shape and dtype.
Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]
Shape = Sequence[int]


def _attention_dropout(attn_weights: jnp.ndarray,
                       *,
                       rate: float,
                       broadcast: bool = True,
                       dropout_rng: jnp.ndarray) -> jnp.ndarray:
  """Applies dropout on attention weights.

  This always applies the dropout. There is no `deterministic` parameter.

  Arguments:
    attn_weights: Attention weights.
    rate: The dropout rate. (_not_ the keep rate!)
    broadcast: Whether to broadcast on first and second last axis.
    dropout_rng: RNG.

  Returns:
    Weights after dropout.
  """
  keep_prob = 1.0 - rate
  if broadcast:
    # Dropout is broadcast across the batch+head+non-attention dimension.
    dropout_shape = list(attn_weights.shape)
    dropout_shape[0] = 1  # Broadcast batch.
    dropout_shape[-2] = 1  # Broadcast heads.
    keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
  else:
    keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
  multiplier = (
      keep.astype(attn_weights.dtype) /
      jnp.asarray(keep_prob, dtype=attn_weights.dtype))
  return attn_weights * multiplier


def dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    *,
    bias: Optional[jnp.ndarray] = None,
    bias_kv: Optional[jnp.ndarray] = None,
    broadcast_dropout: bool = True,
    dropout_rate: float = 0.1,
    dtype: jnp.dtype = jnp.float32,
    precision: Optional[jax.lax.Precision] = None,
    deterministic: bool,
    dropout_rng: Optional[jnp.ndarray] = None,
    capture_attention_weights: bool = True) -> jnp.ndarray:
  """Computes the dot-product attention given query, key and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Note: query, key, value needn't have any batch dimensions.

  Args:
    query: Queries for calculating attention with shape of `[batch..., q_length,
      num_heads, qk_depth_per_head]`.
    key: Keys for calculating attention with shape of `[batch..., kv_length,
      num_heads, qk_depth_per_head]`.
    value: Values to be used in attention with shape of `[batch..., kv_length,
      num_heads, v_depth_per_head]`.
    bias: Bias for the attention weights. This should be
      broadcastable to the shape: `[batch..., num_heads, q_length, kv_length]`
        This can be used for incorporating causal masks, padding masks,
        proximity bias, etc.
    bias_kv: Attention bias defined for keys only which has shape
      `[batch..., kv_length]`. Can be used for masking elements in k/v.
    broadcast_dropout: Use a broadcasted dropout along batch dims.
    dropout_rate: Dropout rate.
    dtype: The dtype of the computation (default: float32).
    precision: Numerical precision of the computation see `jax.lax.Precision`
      for details.
    deterministic: Deterministic or not (to apply dropout).
    dropout_rng: Optional JAX PRNGKey to be used for dropout.
    capture_attention_weights: Whether to add an identity layer to tag the
      attention weights to be used for capturing them using Flax
      capture_intermediate, e.g. for visualization. Note that if this is set to
      True, this function can be only called within a Flax module.

  Returns:
    Output of shape `[batch..., length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      'q, k, v num_heads must match.')
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # Calculate attention matrix.
  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)
  # attn weight shape is (batch..., num_heads, q_length, kv_length)
  attn_weights = jnp.einsum(
      '...qhd,...khd->...hqk', query, key, precision=precision)

  # Apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias
  if bias_kv is not None:
    bias_kv = bias_kv[..., jnp.newaxis, jnp.newaxis, :]
    attn_weights += bias_kv

  # Normalize the attention weights.
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  if capture_attention_weights:
    # Tag the intermediate weights for logging/visualization.
    attn_weights = nn_layers.IdentityLayer(name='attn_weights')(attn_weights)

  # Apply attention dropout.
  if not deterministic and dropout_rate > 0.:
    if dropout_rng is None:
      raise ValueError('Did not provide `rng` to dot_product_attention().')
    attn_weights = _attention_dropout(
        attn_weights,
        rate=dropout_rate,
        broadcast=broadcast_dropout,
        dropout_rng=dropout_rng)

  # Return weighted sum over values for each query position.
  return jnp.einsum(
      '...hqk,...khd->...qhd', attn_weights, value, precision=precision)


def axial_dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    *,
    bias: Optional[jnp.ndarray] = None,
    bias_kv: Optional[jnp.ndarray] = None,
    broadcast_dropout: bool = True,
    dropout_rate: float = 0.1,
    dtype: jnp.dtype = jnp.float32,
    precision: Optional[jax.lax.Precision] = None,
    deterministic: bool,
    dropout_rng: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Applies masked, head-axial qkv dot-product attention.

  Assigns different heads for different axes which is more efficient and
  allows for having attention on all axes in every layer.

  Args:
    query: Queries for calculating attention with shape of `[batch, ...,
      num_heads, qk_depth_per_head]`.
    key: Keys for calculating attention with shape of `[batch, ..., num_heads,
      qk_depth_per_head]`.
    value: Values to be used in attention with shape of `[batch, ..., num_heads,
      v_depth_per_head]`.
    bias: Bias is not supported and will raise an error if passed.
    bias_kv: Bias for the attention weights. This should be
      broadcastable to the shape: `[batch, ...]`. This can be used for
        incorporating causal masks, padding masks, proximity bias, etc. Default
        is None, which means no bias is applied on attention matrix.
    broadcast_dropout: Use a broadcasted dropout along batch dims.
    dropout_rate: Dropout rate.
    dtype: The dtype of the computation (default: float32).
    precision: Numerical precision of the computation see `jax.lax.Precision`
      for details.
    deterministic: Deterministic or not (to apply dropout).
    dropout_rng: Optional JAX PRNGKey to be used for dropout.

  Returns:
    Output of shape `[bs, ..., num_heads, features]`.
  """
  if query.shape != key.shape:
    raise ValueError('Axial dot product attention only supports '
                     'query and key with the same shape.')
  if bias is not None:
    raise NotImplementedError('Bias passed to axial attention.')
  if bias_kv is not None:
    # expand padding mask for head dimension, and last dimension, which will
    # be broadcasted. [batch, ..., 1, 1]
    bias_kv = bias_kv[..., jnp.newaxis, jnp.newaxis]
  # Normalize the query with the squre of its depth.
  query = query / jnp.sqrt(query.shape[-1]).astype(dtype)
  prefix_str = 'abcdefghijk'
  # Split heads for each axial attention dimension.
  num_attn_dimensions = query.ndim - 3  # all dims but bs, heads, and channel.
  if query.shape[-2] % num_attn_dimensions != 0:
    raise ValueError(f'In head-axial dot-product attention, number of '
                     f'heads ({query.shape[-2]}) should be divisible by number '
                     f'of attention dimensions ({num_attn_dimensions})!')

  queries = jnp.split(query, num_attn_dimensions, axis=-2)
  keys = jnp.split(key, num_attn_dimensions, axis=-2)
  values = jnp.split(value, num_attn_dimensions, axis=-2)

  outputs = []
  for i, (query, key, value) in enumerate(zip(queries, keys, values)):
    axis = i + 1  # + 1 for batch
    batch_dims = prefix_str[:axis]
    einsum_str = f'{batch_dims}x...z,{batch_dims}y...z->{batch_dims}x...y'
    attn_logits = jnp.einsum(einsum_str, query, key, precision=precision)
    if bias_kv is not None:
      # put attention axis into last dimension
      attn_logits += jnp.swapaxes(bias_kv, axis, -1)  # {batch_dims}1...y
    attn_weights = jax.nn.softmax(attn_logits, axis=-1)

    # Apply dropout.
    if not deterministic and dropout_rate > 0.:
      attn_weights = _attention_dropout(
          attn_weights,
          rate=dropout_rate,
          broadcast=broadcast_dropout,
          dropout_rng=dropout_rng)
    einsum_str = f'{batch_dims}x...y,{batch_dims}y...z->{batch_dims}x...z'
    outputs.append(
        jnp.einsum(einsum_str, attn_weights, value, precision=precision))

  return jnp.concatenate(outputs, axis=-2)


class MultiHeadAttention(nn.Module):
  """Customized multi-head attention for scenic.

  Attributes:
    num_heads: Number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    qkv_features: Dimension of the key, query, and value.
    out_features: Dimension of the last projection.
    dropout_rate: Dropout rate.
    broadcast_dropout: Use a broadcasted dropout along batch dims.
    kernel_init: Initializer for the kernel of the Dense layers.
    bias_init: Initializer for the bias of the Dense layers.
    out_kernel_init: Initializer for the kernel of the output Dense layers. If
      None, kernel_init will be used.
    use_bias: Whether pointwise QKV dense transforms use bias.
    precision: Numerical precision of the computation see `jax.lax.Precision`
      for details.
    attention_fn: Defaults to dot_product_attention. Other function of the
      same signature are possible.
    dtype: the dtype of the computation (default: float32).
    enforce_hidden_size_divisible_by_heads: Whether or not we allow the hidden
      size to not be divisible by the number of heads.
  """
  num_heads: int
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  dropout_rate: float = 0.
  broadcast_dropout: bool = False
  kernel_init: Initializer = nn.linear.default_kernel_init
  bias_init: Initializer = nn.initializers.zeros
  out_kernel_init: Optional[Initializer] = None
  use_bias: bool = True
  attention_fn: Callable[..., jnp.ndarray] = dot_product_attention
  precision: Optional[jax.lax.Precision] = None
  dtype: jnp.dtype = jnp.float32
  enforce_hidden_size_divisible_by_heads: bool = True

  @nn.compact
  def __call__(self,
               inputs_q: jnp.ndarray,
               inputs_kv: Optional[jnp.ndarray],
               *,
               pos_emb_q: Optional[jnp.ndarray] = None,
               pos_emb_k: Optional[jnp.ndarray] = None,
               pos_emb_v: Optional[jnp.ndarray] = None,
               attention_bias: Optional[jnp.ndarray] = None,
               attention_bias_kv: Optional[jnp.ndarray] = None,
               deterministic: bool = False) -> jnp.ndarray:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    This can be used for encoder-decoder attention by specifying both `inputs_q`
    and `inputs_kv` or for self-attention by only specifying `inputs_q` and
    setting `inputs_kv` to None.

    Args:
      inputs_q: Input queries of shape  `[bs, ..., len_q, features]`.
      inputs_kv: Key/values of shape `[bs, ..., len_k, features]` or None for
        self-attention, in which case key/values will be derived from inputs_q.
      pos_emb_q: Positional embedding to be added to the query.
      pos_emb_k: Positional embedding to be added to the key.
      pos_emb_v: Positional embedding to be added to the value.
      attention_bias: Full attention bias. Should be broadcastable to:
        inputs_q.shape[:-2] + (num_heads, len_q, len_k).
      attention_bias_kv: Attention bias for keys independent of queries which
        has shape (bs, ..., len_k).
      deterministic: Run deterministically or with dropout.

    Returns:
      Output of shape `[bs, ..., features]`.
    """
    if inputs_kv is None:
      inputs_kv = inputs_q

    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]

    if self.enforce_hidden_size_divisible_by_heads:
      assert qkv_features % self.num_heads == 0, (
          'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    def add_positional_emb(x, pos):
      return x + pos if pos is not None else x

    query, key, value = (
        add_positional_emb(inputs_q, pos_emb_q),
        add_positional_emb(inputs_kv, pos_emb_k),
        add_positional_emb(inputs_kv, pos_emb_v))

    dense = functools.partial(
        nn.DenseGeneral,
        axis=-1,
        features=(self.num_heads, head_dim),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        precision=self.precision)
    # Project inputs_q to multi-headed q/k/v.
    # Dimensions are then [..., l, n_heads, n_features_per_head].
    query, key, value = (dense(name='query')(query),
                         dense(name='key')(key),
                         dense(name='value')(value))

    # pylint: disable=too-many-function-args
    attn_kwargs = {}
    if attention_bias_kv is not None:
      # Not necessarily supported by all underlying functions.
      attn_kwargs['bias_kv'] = attention_bias_kv
    if not deterministic and self.dropout_rate > 0:
      attn_kwargs['dropout_rng'] = self.make_rng('dropout')

    x = self.attention_fn(
        query,
        key,
        value,
        bias=attention_bias,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=deterministic,
        dtype=self.dtype,
        precision=self.precision,
        **attn_kwargs)
    # pylint: enable=too-many-function-args

    # Back to the original inputs dimensions.
    out_kernel_init = (self.out_kernel_init if self.out_kernel_init is not None
                       else self.kernel_init)
    out = nn.DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=out_kernel_init,
        bias_init=self.bias_init,
        use_bias=True,
        dtype=self.dtype,
        precision=self.precision,
        name='out')(x)

    return out


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  use_bias: bool = True
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.normal(stddev=1e-6)
  activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
  precision: Optional[jax.lax.Precision] = None
  dtype: jnp.ndarray = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, deterministic: bool):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        self.mlp_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision)(
            inputs)
    x = nn_layers.IdentityLayer(name='mlp1')(self.activation_fn(x))
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision)(
            x)
    output = nn_layers.IdentityLayer(name='mlp2')(output)
    output = nn.Dropout(rate=self.dropout_rate)(
        output, deterministic=deterministic)
    return output


def sinusoidal_init(max_len: int, max_timescale: float = 1.0e4):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input.
      max_timescale: Maximum time scale.

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key: jnp.ndarray,
           shape: Sequence[int],
           dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """Sinusoidal init.

    The defined API by JAX for a custom initializer is:
      `def init(key, shape, dtype)`

    Even though some of args might be not used, the signature should follow
    this API as JAX passes all the three arguments (key, shape, dtype)
    to the initializers.

    Args:
      key: JAXPRNG key.
      shape: Shape used for making the initialized values.
      dtype: JAX data type.

    Returns:
      Initialized values
    """
    del key, dtype
    d_feature = shape[-1]
    pos_emb = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_feature, 2) * -(np.log(max_timescale) / d_feature))
    pos_emb[:, 0::2] = np.sin(position * div_term)
    pos_emb[:, 1::2] = np.cos(position * div_term)
    pe = pos_emb[np.newaxis, :, :]  #  Shape: `[1, max_len, d_feature]`.
    return jnp.array(pe)

  return init


class Add1DPositionEmbedding(nn.Module):
  """Adds 1-dimensional positional embeddings to the inputs.

  Attributes:
    rescale_from: tuple; If not None, embeddings are rescaled from this shape.
    max_len: int; Maximum possible length for the input. If None, the max_len is
      set to the inputs sequence length.
    posemb_init: Positional embedding initializer.
    param_name: The name of the parameter that stores the positional embedding.
  """

  rescale_from: Optional[Sequence[int]] = None
  max_len: Optional[int] = None
  posemb_init: Optional[Initializer] = None
  param_name: str = 'pos_embedding'

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies Add1DPositionEmbedding module.

    Args:
      inputs: nd-arrary; Input data.

    Returns:
      Output: `(bs, timesteps, in_dim)`.
    """
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    max_len = self.max_len or length
    embedding_length = max_len

    if self.rescale_from:  # Shape: `[len, c]`.
      embedding_length = self.rescale_from[0]

    pos_emb_shape = (1, embedding_length, inputs.shape[-1])
    if self.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=embedding_length)(None,  # pytype: disable=wrong-arg-types  # jax-ndarray
                                                                pos_emb_shape,
                                                                None)
    else:
      pos_embedding = self.param(self.param_name, self.posemb_init,
                                 pos_emb_shape)
    pe = pos_embedding[:, :length, :]

    if max_len != embedding_length:
      pe = jax.image.resize(
          pe, (1, max_len, pe.shape[-1]), method='bilinear', antialias=False)
      pe = jnp.reshape(pe, (1, max_len, -1))
    return inputs + pe


class Add2DPositionEmbedding(nn.Module):
  """Adds 2-dimensional positional embeddings to the inputs.

  Attributes:
    rescale_from: tuple; If not None, embeddings are rescaled from this shape.
    posemb_init: Positional embedding initializer.
  """

  rescale_from: Optional[Tuple[int, ...]] = None
  posemb_init: Initializer = nn.initializers.normal(stddev=0.02)  # From BERT.

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies Add2DPositionEmbedding module.

    Args:
      inputs: nd-arrary; Input data.

    Returns:
      Output: `(bs, h, w, c)`.
    """
    assert inputs.ndim == 4, ('Number of dimensions should be 4,'
                              ' but it is: %d' % inputs.ndim)
    _, h, w, c = inputs.shape
    embedding_h, embedding_w = h, w
    if self.rescale_from:  # `[h, w, c]`
      embedding_h, embedding_w = self.rescale_from[0], self.rescale_from[1]

    row_pos_embed = self.param('row_pos_embedding', self.posemb_init,
                               (embedding_w, c // 2))
    col_pos_embed = self.param('col_pos_embedding', self.posemb_init,
                               (embedding_h, c // 2))
    # To `[h, w, c//2]`.
    x_pos_emb = jnp.tile(
        jnp.expand_dims(row_pos_embed, axis=0), (embedding_h, 1, 1))
    # To `[h, w, c//2]`.
    y_pos_emb = jnp.tile(
        jnp.expand_dims(col_pos_embed, axis=1), (1, embedding_w, 1))
    # To `[h, w, c]`.
    pos = jnp.concatenate((x_pos_emb, y_pos_emb), axis=-1)

    if w != embedding_w or h != embedding_h:
      pos = jax.image.resize(pos, (h, w, c), method='bilinear', antialias=False)

    # To `[1, h, w, c]`.
    pos = jnp.expand_dims(pos, axis=0)

    return inputs + pos


def get_fixed_sincos_position_embedding(x_shape: Shape,
                                        temperature: float = 10_000,
                                        dtype: jnp.dtype = jnp.float32):
  """Provides a fixed position encoding for 2D and 3D coordinates.

  The embedding follows the initialisation method used in multiple papers such
  as "Attention is All You Need", https://arxiv.org/abs/1706.03762 and
  "Better plain ViT baselines for ImageNet-1k", https://arxiv.org/abs/2205.01580

  Arguments:
    x_shape: the shape of the input for which a position embedding is needed.
    temperature: Temperature parameter.
    dtype: dtype of the position encoding.
  Returns:
    Matrix of position embeddings, has shape [1, ...], where ... = x_shape[1:].
  """
  assert len(x_shape) in (4, 5), f'Unsupported input shape: {x_shape}'
  num_parts = 4 if len(x_shape) == 4 else 6
  channels = x_shape[-1]
  assert channels % num_parts == 0, f'Channels must be multiple of {num_parts}'
  omega = jnp.arange(
      channels // num_parts, dtype=jnp.float32) / (channels / num_parts)
  omega = 1. / (temperature**omega)

  if len(x_shape) == 4:  # 2D input.
    _, h, w, _ = x_shape
    y, x = jnp.mgrid[:h, :w]
    y = jnp.einsum('m,d->md', y.flatten(), omega)
    x = jnp.einsum('m,d->md', x.flatten(), omega)
    p = [jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)]
    shape = (1, h, w, channels)
  elif len(x_shape) == 5:  # 3D input.
    _, t, h, w, _ = x_shape
    z, y, x = jnp.mgrid[:t, :h, :w]
    z = jnp.einsum('m,d->md', z.flatten(), omega)
    y = jnp.einsum('m,d->md', y.flatten(), omega)
    x = jnp.einsum('m,d->md', x.flatten(), omega)
    p = [jnp.sin(z), jnp.cos(z),
         jnp.sin(x), jnp.cos(x),
         jnp.sin(y), jnp.cos(y)]
    shape = (1, t, h, w, channels)
  else:  # Should never reach there because of assert at beginning.
    raise ValueError(f'Unsupported input shape: {x_shape}')

  assert (shape[0] == 1) and (shape[1:] == x_shape[1:])
  pe = jnp.concatenate(p, axis=1)
  return jnp.asarray(pe, dtype).reshape(*shape)


class AddFixedSinCosPositionEmbedding(nn.Module):
  """Provides a fixed position encoding for 2D and 3D coordinates.

  The embedding follows the initialisation method used in multiple papers such
  as "Attention is All You Need", https://arxiv.org/abs/1706.03762 and
  "Better plain ViT baselines for ImageNet-1k", https://arxiv.org/abs/2205.01580

  Attributes:
    temperature: Temperature parameter.
    dtype: dtype of the position encoding.
  """
  temperature: float = 10_000
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Adds the fixed embedding to the inputs.

    Args:
      inputs: Either an [N, W, H, C] or [N, T, W, H, C] input array.

    Returns:
      inputs with position encodings added to them.
    """
    return inputs + get_fixed_sincos_position_embedding(
        inputs.shape, self.temperature, self.dtype)


class RelativeAttentionBias(nn.Module):
  """Provides learnable NxN relative attention bias.

  Attributes:
    num_heads: Number of heads for which to provide relative attention.
    nd_shape: Shape for which to provided relative attention bias. For instance,
      for images we we would provide a 2D shape. Note that batch and feature
      dimensions should be excluded here.
    initializer: Initializer for the bias.
  """

  num_heads: int
  nd_shape: Sequence[int]
  initializer: Initializer = nn.initializers.zeros

  @nn.compact
  def __call__(self) -> jnp.ndarray:
    """Creates relative attention bias that factorizes over dimensions.

    length = prod(nd_shape)

    Returns:
      Bias of shape `[num_heads, length, length]`.
    """
    length = np.prod(self.nd_shape)
    tile = 1
    biases = []
    for i, l in enumerate(self.nd_shape):
      # Relative attention in every dimension separately.
      if l > 1:
        new_bias = self.relative_attn_bias(l, self.num_heads, f'bias_{i}')
        repeat = length // (tile * l)
        if repeat > 1:
          new_bias = new_bias[:, :, jnp.newaxis, :, jnp.newaxis]
          new_bias = jnp.tile(new_bias, [1, tile, repeat, tile, repeat])
          new_bias = jnp.reshape(new_bias, [self.num_heads, length, length])
        elif tile > 1:
          new_bias = jnp.tile(new_bias, [1, tile, tile])
        tile *= l
        biases.append(new_bias)

    return sum(biases)

  def relative_attn_bias(self, length, num_heads, name):
    """Computes attention bias based on relative positions.

    Content-based relative position attention bias was used in:
      https://arxiv.org/pdf/1803.02155.
    Non-content-based relative position attention bias was used in:
      https://arxiv.org/abs/1606.01933.

    Args:
      length: Length of self-attention window for relative attention.
      num_heads: Number of attention heads.
      name: Name of the parameter to be created.

    Returns:
      A `[num_heads, length, length]` tensor with queries.
    """
    # Actually we need only 2 * length - 1 relative positions, but we need at
    # least another entry as padding for relative shift of each row to the right
    num_rel_pos = 2 * length

    rel_bias = self.param(
        name, self.initializer, (self.num_heads, num_rel_pos))

    # Now we have to shift in order to compute relative biases.
    # Example: length = 3
    # Say we want:  [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
    # Start: [[-2, -1, 0, 1, 2, 3], [-2, -1, 0, 1, 2, 3], [-2, -1, 0, 1, 2, 3]]
    # We linearize: [-2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3]
    # We slice: [-2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, 3, -2, -1, 0]
    # We reshape: [[-2, -1, 0, 1, 2], [3, -2, -1, 0, 1], [2, 3, -2, -1, 0]]
    # We slice: [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]
    # Tadaaa!

    # [heads, length * num_rel_pos]
    rel_bias = jnp.tile(rel_bias, [1, length])

    # [heads, length * (num_rel_pos - 1)]
    num_rel_pos -= 1
    rel_bias = rel_bias[..., :length * num_rel_pos]

    # [heads, length, num_rel_pos - 1]
    # Now every row is shifted by 1 to the right.
    rel_bias = rel_bias.reshape(num_heads, length, num_rel_pos)

    # [heads, length, length]
    # Slice the overlapping elements from start.
    rel_bias = rel_bias[..., num_rel_pos - length:]

    return rel_bias
