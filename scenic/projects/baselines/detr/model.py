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

"""Implementation of DETR architecture.

End-to-End Object Detection with Transformers: https://arxiv.org/abs/2005.12872
Implementation is based on: https://github.com/facebookresearch/detr
"""

# pylint: disable=not-callable

import functools
from typing import Any, Callable, Dict, Tuple, Optional

import flax.linen as nn
import jax
from jax.nn import initializers
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.layers import attention_layers
from scenic.projects.baselines import resnet
from scenic.projects.baselines.detr import detr_base_model

# TODO(scenic): add support for DC5 in the backbone

pytorch_kernel_init = functools.partial(initializers.variance_scaling, 1. / 3.,
                                        'fan_in', 'uniform')


def uniform_initializer(minval, maxval, dtype=jnp.float32):

  def init(key, shape, dtype=dtype):
    return jax.random.uniform(key, shape, dtype, minval=minval, maxval=maxval)

  return init


class QueryPosEmbedding(nn.Module):
  """Creates learned positional embeddings for object queries.

  Attributes:
    hidden_dim: Hidden dimension for the pos embeddings.
    num_queries: Number of object queries.
    posemb_init: Positional embeddings initializer.
    dtype: Jax dtype; The dtype of the computation (default: float32).
  """
  hidden_dim: int
  num_queries: int
  posemb_init: Callable[..., Any] = initializers.normal(stddev=1.0)
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self) -> jnp.ndarray:
    """Creates the positional embeddings for queries.

    Returns:
      Positional embedding for object queries.
    """
    query_pos = self.param('query_emb', self.posemb_init,
                           (self.num_queries, self.hidden_dim))
    query_pos = jnp.expand_dims(query_pos, 0)
    return jnp.asarray(query_pos, self.dtype)


class InputPosEmbeddingLearned(nn.Module):
  """Creates learned positional embeddings for inputs.

  Attributes:
    inputs_shape: Shape of the 2D input, before flattening.
    hidden_dim: hidden dimension for the pos embeddings.
    max_h_w: Maximum height and width for the transformer inputs.
    posemb_init: Positional embeddings initializer.
    dtype: The dtype of the computation (default: float32).
  """

  inputs_shape: Tuple[int, int, int, int]
  hidden_dim: Optional[int] = None
  max_h_w: Optional[int] = None
  posemb_init: Callable[..., Any] = initializers.normal(stddev=1.0)
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self) -> jnp.ndarray:
    """Creates the positional embeddings for transformer inputs.

    Returns:
      Positional embedding for inputs and queries.
    """
    _, h, w, c = self.inputs_shape
    max_h_w = self.max_h_w or np.max((h, w))
    assert h <= max_h_w and w <= max_h_w

    hidden_dim = self.hidden_dim or c

    row_pos_embed = self.param('row_pos_embed', self.posemb_init,
                               (max_h_w, hidden_dim // 2))
    col_pos_embed = self.param('col_pos_embed', self.posemb_init,
                               (max_h_w, hidden_dim // 2))

    # to `[h, w, hidden_dim//2]`
    x_pos_emb = jnp.tile(jnp.expand_dims(col_pos_embed[:w], axis=0), (h, 1, 1))
    # to `[h, w, hidden_dim//2]`
    y_pos_emb = jnp.tile(jnp.expand_dims(row_pos_embed[:h], axis=1), (1, w, 1))
    # to `[1, h*w, hidden_dim]`
    pos = jnp.expand_dims(
        jnp.concatenate((x_pos_emb, y_pos_emb), axis=-1).reshape(
            (h * w, hidden_dim)),
        axis=0)

    return jnp.asarray(pos, self.dtype)


class InputPosEmbeddingSine(nn.Module):
  """Creates sinusoidal positional embeddings for inputs."""

  hidden_dim: int
  dtype: jnp.dtype = jnp.float32
  scale: Optional[float] = None
  temperature: float = 10000

  @nn.compact
  def __call__(self, padding_mask: jnp.ndarray) -> jnp.ndarray:
    """Creates the positional embeddings for transformer inputs.

    Args:
      padding_mask: Binary matrix with 0 at padded image regions. Shape is
        [batch, height, width]

    Returns:
      Positional embedding for inputs.

    Raises:
      ValueError if `hidden_dim` is not an even number.
    """
    if self.hidden_dim % 2:
      raise ValueError('`hidden_dim` must be an even number.')

    mask = padding_mask.astype(jnp.float32)
    y_embed = jnp.cumsum(mask, axis=1)
    x_embed = jnp.cumsum(mask, axis=2)

    # Normalization:
    eps = 1e-6
    scale = self.scale if self.scale is not None else 2 * jnp.pi
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    num_pos_feats = self.hidden_dim // 2
    dim_t = jnp.arange(num_pos_feats, dtype=jnp.float32)
    dim_t = self.temperature**(2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, jnp.newaxis] / dim_t
    pos_y = y_embed[:, :, :, jnp.newaxis] / dim_t
    pos_x = jnp.stack([
        jnp.sin(pos_x[:, :, :, 0::2]),
        jnp.cos(pos_x[:, :, :, 1::2]),
    ],
                      axis=4).reshape(padding_mask.shape + (-1,))
    pos_y = jnp.stack([
        jnp.sin(pos_y[:, :, :, 0::2]),
        jnp.cos(pos_y[:, :, :, 1::2]),
    ],
                      axis=4).reshape(padding_mask.shape + (-1,))

    pos = jnp.concatenate([pos_y, pos_x], axis=3)
    b, h, w = padding_mask.shape
    pos = jnp.reshape(pos, [b, h * w, self.hidden_dim])
    return jnp.asarray(pos, self.dtype)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  kernel_init: Callable[..., Any] = nn.initializers.xavier_uniform()
  bias_init: Callable[..., Any] = nn.initializers.normal(stddev=1e-6)
  activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
  dtype: jnp.ndarray = jnp.float32

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               deterministic: bool = True) -> jnp.ndarray:
    """Applies Transformer MlpBlock model."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(
            inputs)
    x = self.activation_fn(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(
            x)
    output = nn.Dropout(rate=self.dropout_rate)(
        output, deterministic=deterministic)
    return output


class MultiHeadDotProductAttention(nn.Module):
  """DETR Customized Multi-head dot-product attention.

  Attributes:
    num_heads: Number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    pos_emb_q: Positional embedding to be added to the query.
    pos_emb_k: Positional embedding to be added to the key.
    pos_emb_v: Positional embedding to be added to the value.
    qkv_features: dimension of the key, query, and value.
    out_features: dimension of the last projection
    dropout_rate: dropout rate
    broadcast_dropout: use a broadcasted dropout along batch dims.
    kernel_init: initializer for the kernel of the Dense layers.
    bias_init: initializer for the bias of the Dense layers.
    use_bias: bool: whether pointwise QKV dense transforms use bias. In DETR
      they always have a bias on the output.
    dtype: the dtype of the computation (default: float32)
  """

  num_heads: int
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  dropout_rate: float = 0.
  broadcast_dropout: bool = False
  kernel_init: Callable[..., Any] = initializers.xavier_uniform()
  bias_init: Callable[..., Any] = initializers.zeros
  use_bias: bool = True
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs_q: jnp.ndarray,
               inputs_kv: Optional[jnp.ndarray] = None,
               *,
               pos_emb_q: Optional[jnp.ndarray] = None,
               pos_emb_k: Optional[jnp.ndarray] = None,
               pos_emb_v: Optional[jnp.ndarray] = None,
               key_padding_mask: Optional[jnp.ndarray] = None,
               train: bool = False) -> jnp.ndarray:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    This can be used for encoder-decoder attention by specifying both `inputs_q`
    and `inputs_kv` or for self-attention by only specifying `inputs_q` and
    setting `inputs_kv` to None.

    Args:
      inputs_q: Input queries of shape  `[bs, len, features]`.
      inputs_kv: Key/values of shape `[bs, len, features]` or None for
        self-attention, in which case key/values will be derived from inputs_q.
      pos_emb_q: Positional embedding to be added to the query.
      pos_emb_k: Positional embedding to be added to the key.
      pos_emb_v: Positional embedding to be added to the value.
      key_padding_mask: Binary array. Key-value tokens that are padded are 0,
        and 1 otherwise.
      train: Train or not (to apply dropout).

    Returns:
      output of shape `[bs, len, features]`.
    """
    if inputs_kv is None:
      inputs_kv = inputs_q

    assert inputs_kv.ndim == inputs_q.ndim == 3
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]

    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    def add_positional_emb(x, pos_emb_x):
      return x if pos_emb_x is None else x + pos_emb_x

    query, key, value = (add_positional_emb(inputs_q, pos_emb_q),
                         add_positional_emb(inputs_kv, pos_emb_k),
                         add_positional_emb(inputs_kv, pos_emb_v))

    dense = functools.partial(
        nn.DenseGeneral,
        axis=-1,
        features=(self.num_heads, head_dim),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=self.dtype)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [bs, l, n_heads, n_features_per_head]
    query, key, value = (dense(name='query')(query), dense(name='key')(key),
                         dense(name='value')(value))

    # create attention masks
    if key_padding_mask is not None:
      attention_bias = (1 - key_padding_mask) * -1e10
      # add head and query dimension.
      attention_bias = jnp.expand_dims(attention_bias, -2)
      attention_bias = jnp.expand_dims(attention_bias, -2)
    else:
      attention_bias = None

    # apply attention
    x = attention_layers.dot_product_attention(
        query,
        key,
        value,
        dtype=self.dtype,
        bias=attention_bias,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        dropout_rng=self.make_rng('dropout') if train else None,
        deterministic=not train,
        capture_attention_weights=True)

    # back to the original inputs dimensions
    out = nn.DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=True,
        dtype=self.dtype,
        name='out')(
            x)

    return out


class EncoderBlock(nn.Module):
  """DETR Transformer encoder block.

  Attributes:
    num_heads: Number of heads.
    qkv_dim: Dimension of the query/key/value.
    mlp_dim: Dimension of the mlp on top of attention block.
    pre_norm: If use LayerNorm before attention/mlp blocks.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout rate for attention weights.
    dtype: Data type of the computation (default: float32).
  """

  num_heads: int
  qkv_dim: int
  mlp_dim: int
  pre_norm: bool = False
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               pos_embedding: Optional[jnp.ndarray] = None,
               padding_mask: Optional[jnp.ndarray] = None,
               train: bool = False) -> jnp.ndarray:
    """Applies EncoderBlock module.

    Args:
      inputs: Input data of shape [batch_size, len, features].
      pos_embedding: Positional Embedding to be added to the queries and keys in
        the self-attention operation.
      padding_mask: Binary mask containing 0 for padding tokens.
      train: Train or not (to apply dropout).

    Returns:
      Output after transformer encoder block.
    """
    self_attn = MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=self.qkv_dim,
        dropout_rate=self.attention_dropout_rate,
        broadcast_dropout=False,
        kernel_init=initializers.xavier_uniform(),
        bias_init=initializers.zeros,
        use_bias=True,
        dtype=self.dtype)

    mlp = MlpBlock(  # pytype: disable=wrong-arg-types  # jnp-type
        mlp_dim=self.mlp_dim,
        activation_fn=nn.relu,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate)

    assert inputs.ndim == 3

    if self.pre_norm:
      x = nn.LayerNorm(dtype=self.dtype)(inputs)
      x = self_attn(
          inputs_q=x,
          pos_emb_q=pos_embedding,
          pos_emb_k=pos_embedding,
          pos_emb_v=None,
          key_padding_mask=padding_mask,
          train=train)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
      x = x + inputs
      y = nn.LayerNorm(dtype=self.dtype)(x)
      y = mlp(y, deterministic=not train)
      out = x + y

    else:
      x = self_attn(
          inputs_q=inputs,
          pos_emb_q=pos_embedding,
          pos_emb_k=pos_embedding,
          pos_emb_v=None,
          key_padding_mask=padding_mask,
          train=train)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
      x = x + inputs
      x = nn.LayerNorm(dtype=self.dtype)(x)
      y = mlp(x, deterministic=not train)
      y = x + y
      out = nn.LayerNorm(dtype=self.dtype)(y)

    return out


class DecoderBlock(nn.Module):
  """DETR Transformer decoder block.

  Attributes:
    num_heads: Number of heads.
    qkv_dim: Dimension of the query/key/value.
    mlp_dim: Dimension of the mlp on top of attention block.
    pre_norm: If use LayerNorm before attention/mlp blocks.
    dropout_rate:Dropout rate.
    attention_dropout_rate:Dropout rate for attention weights.
    dtype: Data type of the computation (default: float32).
  """

  num_heads: int
  qkv_dim: int
  mlp_dim: int
  pre_norm: bool = False
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               obj_queries: jnp.ndarray,
               encoder_output: jnp.ndarray,
               *,
               pos_embedding: Optional[jnp.ndarray] = None,
               query_pos_emb: Optional[jnp.ndarray] = None,
               key_padding_mask: Optional[jnp.ndarray] = None,
               train: bool = False):
    """Applies DecoderBlock module.

    Args:
      obj_queries: Input data for decoder.
      encoder_output: Output of encoder, which are encoded inputs.
      pos_embedding: Positional Embedding to be added to the keys in
        cross-attention.
      query_pos_emb: Positional Embedding to be added to the queries.
      key_padding_mask: Binary mask containing 0 for pad tokens in key.
      train: Train or not (to apply dropout)

    Returns:
      Output after transformer decoder block.
    """

    assert query_pos_emb is not None, ('Given that object_queries are zeros '
                                       'and not learnable, we should add '
                                       'learnable query_pos_emb to them.')
    # Seems in DETR the self-attention in the first layer basically does
    # nothing, as the  value vector is a zero vector and we add no learnable
    # positional embedding to it!
    self_attn = MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=self.qkv_dim,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        kernel_init=initializers.xavier_uniform(),
        bias_init=initializers.zeros,
        use_bias=True,
        dtype=self.dtype)

    cross_attn = MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=self.qkv_dim,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        kernel_init=initializers.xavier_uniform(),
        bias_init=initializers.zeros,
        use_bias=True,
        dtype=self.dtype)

    mlp = MlpBlock(  # pytype: disable=wrong-arg-types  # jnp-type
        mlp_dim=self.mlp_dim,
        activation_fn=nn.relu,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate)

    assert obj_queries.ndim == 3
    if self.pre_norm:
      # self attention block
      x = nn.LayerNorm(dtype=self.dtype)(obj_queries)
      x = self_attn(
          inputs_q=x,
          pos_emb_q=query_pos_emb,
          pos_emb_k=query_pos_emb,
          pos_emb_v=None,
          train=train)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
      x = x + obj_queries
      # cross attention block
      y = nn.LayerNorm(dtype=self.dtype)(x)
      y = cross_attn(
          inputs_q=y,
          inputs_kv=encoder_output,
          pos_emb_q=query_pos_emb,
          pos_emb_k=pos_embedding,
          pos_emb_v=None,
          key_padding_mask=key_padding_mask,
          train=train)
      y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
      y = y + x
      # mlp block
      z = nn.LayerNorm(dtype=self.dtype)(y)
      z = mlp(z, deterministic=not train)
      out = y + z

    else:
      # self attention block
      x = self_attn(
          inputs_q=obj_queries,
          pos_emb_q=query_pos_emb,
          pos_emb_k=query_pos_emb,
          pos_emb_v=None,
          train=train)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
      x = x + obj_queries
      x = nn.LayerNorm(dtype=self.dtype)(x)
      # cross attention block
      y = cross_attn(
          inputs_q=x,
          inputs_kv=encoder_output,
          pos_emb_q=query_pos_emb,
          pos_emb_k=pos_embedding,
          pos_emb_v=None,
          key_padding_mask=key_padding_mask,
          train=train)
      y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
      y = y + x
      y = nn.LayerNorm(dtype=self.dtype)(y)
      # mlp block
      z = mlp(y, deterministic=not train)
      z = y + z
      out = nn.LayerNorm(dtype=self.dtype)(z)

    return out


class Encoder(nn.Module):
  """DETR Transformer Encoder.

  Attributes:
    num_heads: Number of heads.
    num_layers: Number of layers.
    qkv_dim: Dimension of the query/key/value.
    mlp_dim: Dimension of the mlp on top of attention block.
    normalize_before: If use LayerNorm before attention/mlp blocks.
    norm: normalization layer to be applied on the output.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout rate for attention weights.
    dtype: Data type of the computation (default: float32).
  """

  num_heads: int
  num_layers: int
  qkv_dim: int
  mlp_dim: int
  normalize_before: bool = False
  norm: Optional[Callable[..., Any]] = None
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               pos_embedding: Optional[jnp.ndarray] = None,
               padding_mask: Optional[jnp.ndarray] = None,
               train: bool = False) -> jnp.ndarray:
    """Applies Encoder on the inputs.

    Args:
      inputs: Input data.
      pos_embedding: Positional Embedding to be added to the queries and keys in
        the self-attention operation.
      padding_mask: Binary mask containing 0 for padding tokens, and 1
        otherwise.
      train: Whether it is training.

    Returns:
      Output of the transformer encoder.
    """
    assert inputs.ndim == 3  # `[batch, height*width, features]`
    x = inputs

    # input Encoder
    for lyr in range(self.num_layers):
      x = EncoderBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          pre_norm=self.normalize_before,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          dtype=self.dtype)(
              x,
              pos_embedding=pos_embedding,
              padding_mask=padding_mask,
              train=train)

    if self.norm is not None:
      x = self.norm(x)
    return x


class Decoder(nn.Module):
  """DETR Transformer Decoder.

  Attributes:
    num_heads: Number of heads.
    num_layers: Number of layers.
    qkv_dim: Dimension of the query/key/value.
    mlp_dim: Dimension of the mlp on top of attention block.
    normalize_before: If use LayerNorm before attention/mlp blocks.
    return_intermediate: If return the outputs from intermediate layers.
    padding_mask: Binary mask containing 0 for padding tokens.
    dropout_rate:Dropout rate.
    attention_dropout_rate:Dropout rate for attention weights.
    dtype: Data type of the computation (default: float32).
  """

  num_heads: int
  num_layers: int
  qkv_dim: int
  mlp_dim: int
  normalize_before: bool = False
  norm: Optional[Callable[..., Any]] = None
  return_intermediate: bool = False
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               obj_queries: jnp.ndarray,
               encoder_output: jnp.ndarray,
               *,
               key_padding_mask: Optional[jnp.ndarray] = None,
               pos_embedding: Optional[jnp.ndarray] = None,
               query_pos_emb: Optional[jnp.ndarray] = None,
               train: bool = False) -> jnp.ndarray:
    """Applies Decoder on the inputs.

    Args:
      obj_queries: Input data for decoder.
      encoder_output: Output of encoder, which are encoded inputs.
      key_padding_mask: Binary mask containing 0 for padding tokens in the keys.
      pos_embedding: Positional Embedding to be added to the keys.
      query_pos_emb: Positional Embedding to be added to the queries.
      train: Whether it is training.

    Returns:
      Output of a transformer decoder.
    """
    assert encoder_output.ndim == 3  # `[batch, len, features]`
    assert obj_queries.ndim == 3  # `[batch, num queries, embedding size]`
    y = obj_queries
    outputs = []
    for lyr in range(self.num_layers):
      y = DecoderBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          pre_norm=self.normalize_before,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          dtype=self.dtype,
          name=f'decoderblock_{lyr}')(
              y,
              encoder_output,
              pos_embedding=pos_embedding,
              query_pos_emb=query_pos_emb,
              key_padding_mask=key_padding_mask,
              train=train)
      if self.return_intermediate:
        outputs.append(y)

    if self.return_intermediate:
      y = jnp.stack(outputs, axis=0)
    return y if self.norm is None else self.norm(y)


class DETRTransformer(nn.Module):
  """DETR Transformer.

  Attributes:
    num_queries: Number of object queries. query_emb_size; Size of the embedding
      learned for object queries.
    query_emb_size: Size of the embedding learned for object queries.
    num_heads: Number of heads.
    num_encoder_layers: Number of encoder layers.
    num_decoder_layers: Number of decoder layers.
    qkv_dim: Dimension of the query/key/value.
    mlp_dim: Dimension of the mlp on top of attention block.
    return_intermediate_dec: If return the outputs from intermediate layers of
      the decoder.
    normalize_before: If use LayerNorm before attention/mlp blocks.
    dropout_rate: Dropout rate.
    attention_dropout_rate:Dropout rate for attention weights.
    dtype: Data type of the computation (default: float32).
  """

  num_queries: int = 100
  query_emb_size: Optional[int] = None
  num_heads: int = 8
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  return_intermediate_dec: bool = False
  normalize_before: bool = False
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               padding_mask: Optional[jnp.ndarray] = None,
               pos_embedding: Optional[jnp.ndarray] = None,
               query_pos_emb: Optional[jnp.ndarray] = None,
               train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Applies DETRTransformer on the inputs.

    Args:
      inputs: Input data.
      padding_mask: Binary mask containing 0 for padding tokens.
      pos_embedding: Positional Embedding to be added to the inputs.
      query_pos_emb: Positional Embedding to be added to the queries.
      train: Whether it is training.

    Returns:
      Output of the DETR transformer and output of the encoder.
    """
    encoder_norm = nn.LayerNorm() if self.normalize_before else None
    encoded = Encoder(
        num_heads=self.num_heads,
        num_layers=self.num_encoder_layers,
        qkv_dim=self.qkv_dim,
        mlp_dim=self.mlp_dim,
        normalize_before=self.normalize_before,
        norm=encoder_norm,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        dtype=self.dtype,
        name='encoder')(
            inputs,
            padding_mask=padding_mask,
            pos_embedding=pos_embedding,
            train=train)

    query_dim = self.query_emb_size or inputs.shape[-1]
    obj_query_shape = tuple([inputs.shape[0], self.num_queries, query_dim])
    # Note that we always learn query_pos_embed, so we simply use constant
    # zero vectors for obj_queries and later when applying attention, we have:
    # query = query_pos_embed + obj_queries
    obj_queries = jnp.zeros(obj_query_shape)

    decoder_norm = nn.LayerNorm()
    output = Decoder(
        num_heads=self.num_heads,
        num_layers=self.num_decoder_layers,
        qkv_dim=self.qkv_dim,
        mlp_dim=self.mlp_dim,
        normalize_before=self.normalize_before,
        return_intermediate=self.return_intermediate_dec,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        norm=decoder_norm,
        dtype=self.dtype,
        name='decoder')(
            obj_queries,
            encoded,
            key_padding_mask=padding_mask,
            pos_embedding=pos_embedding,
            query_pos_emb=query_pos_emb,
            train=train)
    return output, encoded


class BBoxCoordPredictor(nn.Module):
  """FFN block for predicting bounding box coordinates."""
  mlp_dim: int
  num_layers: int = 3
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies FFN MLP block to inputs.

    Args:
      x: Input tensor.

    Returns:
      Output of FFN MLP block.
    """
    for _ in range(self.num_layers - 1):
      # This is like pytorch initializes biases in linear layers.
      bias_range = 1 / np.sqrt(x.shape[-1])
      x = nn.Dense(
          self.mlp_dim,
          kernel_init=pytorch_kernel_init(dtype=self.dtype),
          bias_init=uniform_initializer(
              -bias_range, bias_range, dtype=self.dtype),
          dtype=self.dtype)(
              x)
      x = nn.relu(x)

    bias_range = 1 / np.sqrt(x.shape[-1])
    x = nn.Dense(
        4,
        kernel_init=pytorch_kernel_init(dtype=self.dtype),
        bias_init=uniform_initializer(
            -bias_range, bias_range, dtype=self.dtype))(
                x)
    output = nn.sigmoid(x)
    return output


class ObjectClassPredictor(nn.Module):
  """Linear Projection block for predicting classification."""
  num_classes: int
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies Linear Projection to inputs.

    Args:
      inputs: Input data.

    Returns:
      Output of Linear Projection block.
    """
    bias_range = 1. / np.sqrt(inputs.shape[-1])
    return nn.Dense(
        self.num_classes,
        kernel_init=pytorch_kernel_init(dtype=self.dtype),
        bias_init=uniform_initializer(-bias_range, bias_range, self.dtype),
        dtype=self.dtype)(
            inputs)


class DETR(nn.Module):
  """Detection Transformer (DETR) model.

  Attributes:
    num_classes: Number of object classes.
    hidden_dim: Hidden dimension of the inputs to the model.
    num_queries: Number of object queries, ie detection slot. This is the
      maximal number of objects DETR can detect in a single image. For COCO,
      DETR paper recommends 100 queries.
    query_emb_size: Size of the embedding learned for object queries.
    transformer_num_heads: Number of transformer heads.
    transformer_num_encoder_layers: Number of transformer encoder layers.
    transformer_num_decoder_layers: Number of transformer decoder layers.
    transformer_qkv_dim: Dimension of the transformer query/key/value.
    transformer_mlp_dim: Dimension of the mlp on top of attention block.
    transformer_normalize_before: If use LayerNorm before attention/mlp blocks.
    backbone_num_filters: Num filters in the ResNet backbone.
    backbone_num_layers: Num layers in the ResNet backbone.
    aux_loss: If train with auxiliary loss.
    dropout_rate:Dropout rate.
    attention_dropout_rate:Attention dropout rate.
    dtype: Data type of the computation (default: float32).
  """

  num_classes: int
  hidden_dim: int = 512
  num_queries: int = 100
  query_emb_size: Optional[int] = None
  transformer_num_heads: int = 8
  transformer_num_encoder_layers: int = 6
  transformer_num_decoder_layers: int = 6
  transformer_qkv_dim: int = 512
  transformer_mlp_dim: int = 2048
  transformer_normalize_before: bool = False
  backbone_num_filters: int = 64
  backbone_num_layers: int = 50
  aux_loss: bool = False
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               train: bool,
               *,
               padding_mask: Optional[jnp.ndarray] = None,
               update_batch_stats: bool = False,
               debug: bool = False) -> Dict[str, Any]:
    """Applies DETR model on the input.

    Args:
      inputs: Input data.
      train:  Whether it is training.
      padding_mask: Binary matrix with 0 at padded image regions.
      update_batch_stats: Whether update the batch statistics for the BatchNorms
        in the backbone. if None, the value of `train` flag will be used, i.e.
        we update the batch stat if we are in the train mode.
      debug: Whether the debug mode is enabled. debug=True enables model
        specific logging/storing some values using jax.host_callback.

    Returns:
      Output: dit; that has 'pred_logits' and 'pred_boxes', and potentially
      'aux_outputs'.
    """
    # for now, we only support this case
    assert self.hidden_dim == self.transformer_qkv_dim

    if update_batch_stats is None:
      update_batch_stats = train

    backbone_features = resnet.ResNet(
        num_outputs=None,
        num_filters=self.backbone_num_filters,
        num_layers=self.backbone_num_layers,
        dtype=self.dtype,
        name='backbone')(
            inputs, train=update_batch_stats)
    x = backbone_features['stage_4']

    bs, h, w, _ = x.shape

    if padding_mask is None:
      padding_mask_downsampled = jnp.ones((bs, h, w), dtype=jnp.bool_)
    else:
      padding_mask_downsampled = jax.image.resize(
          padding_mask.astype(jnp.float32), shape=[bs, h, w],
          method='nearest').astype(jnp.bool_)
    pos_emb = InputPosEmbeddingSine(hidden_dim=self.hidden_dim)(
        padding_mask_downsampled)

    query_pos_emb = QueryPosEmbedding(
        hidden_dim=self.hidden_dim, num_queries=self.num_queries)()

    # project and reshape to 3 dimensions and project
    x = nn.Conv(features=self.hidden_dim, kernel_size=(1, 1), strides=(1, 1))(x)
    x = x.reshape(bs, h * w, self.hidden_dim)
    transformer_input = x

    return_intermediate = self.aux_loss
    transformer = DETRTransformer(
        num_queries=self.num_queries,
        query_emb_size=self.query_emb_size,
        num_heads=self.transformer_num_heads,
        num_encoder_layers=self.transformer_num_encoder_layers,
        num_decoder_layers=self.transformer_num_decoder_layers,
        qkv_dim=self.transformer_qkv_dim,
        mlp_dim=self.transformer_mlp_dim,
        return_intermediate_dec=return_intermediate,
        normalize_before=self.transformer_normalize_before,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate)
    decoder_output, encoder_output = transformer(
        x,
        padding_mask=jnp.reshape(padding_mask_downsampled, [bs, h * w]),
        pos_embedding=pos_emb,
        query_pos_emb=query_pos_emb,
        train=train)

    def output_projection(model_output):
      # classification head
      pred_logits = ObjectClassPredictor(num_classes=self.num_classes)(
          model_output)
      # bounding box detection head
      pred_boxes = BBoxCoordPredictor(mlp_dim=self.hidden_dim)(model_output)
      return pred_logits, pred_boxes

    if not return_intermediate:
      pred_logits, pred_boxes = output_projection(decoder_output)
      return {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}

    pred_logits, pred_boxes = jax.vmap(output_projection)(decoder_output)
    output = {
        'pred_logits': pred_logits[-1],
        'pred_boxes': pred_boxes[-1],
        'transformer_input': transformer_input,
        'backbone_features': backbone_features,
        'encoder_output': encoder_output,
        'decoder_output': decoder_output[-1],
        'padding_mask': padding_mask_downsampled,
    }

    if self.aux_loss:
      output['aux_outputs'] = []
      for lgts, bxs in zip(pred_logits[:-1], pred_boxes[:-1]):
        output['aux_outputs'].append({'pred_logits': lgts, 'pred_boxes': bxs})

    return output


class DETRModel(detr_base_model.ObjectDetectionWithMatchingModel):
  """Detr model for object detection task."""

  def build_flax_model(self):
    return DETR(
        num_classes=self.dataset_meta_data['num_classes'],
        hidden_dim=self.config.get('hidden_dim', 512),
        num_queries=self.config.get('num_queries', 100),
        query_emb_size=self.config.get('query_emb_size', None),
        transformer_num_heads=self.config.get('transformer_num_heads', 8),
        transformer_num_encoder_layers=self.config.get(
            'transformer_num_encoder_layers', 6),
        transformer_num_decoder_layers=self.config.get(
            'transformer_num_decoder_layers', 6),
        transformer_qkv_dim=self.config.get('transformer_qkv_dim', 512),
        transformer_mlp_dim=self.config.get('transformer_mlp_dim', 2048),
        transformer_normalize_before=self.config.get(
            'transformer_normalize_before', False),
        backbone_num_filters=self.config.get('backbone_num_filters', 64),
        backbone_num_layers=self.config.get('backbone_num_layers', 50),
        aux_loss=self.config.get('aux_loss', False),
        dropout_rate=self.config.get('dropout_rate', 0.0),
        attention_dropout_rate=self.config.get('attention_dropout_rate', 0.0),
        dtype=jnp.float32)

  def default_flax_model_config(self):
    return ml_collections.ConfigDict(
        dict(
            hidden_dim=32,
            num_queries=8,
            query_emb_size=None,
            transformer_num_heads=2,
            transformer_num_encoder_layers=1,
            transformer_num_decoder_layers=1,
            transformer_qkv_dim=32,
            transformer_mlp_dim=32,
            transformer_normalize_before=False,
            backbone_num_filters=32,
            backbone_num_layers=1,
            aux_loss=False,
            dropout_rate=0.0,
            attention_dropout_rate=0.0))
