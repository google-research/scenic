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

"""Implementation of Transformer architecture.

Implementation is based on DETR.
"""

# pylint: disable=not-callable

import functools
from typing import Any, Callable, Optional

import flax.linen as nn
from jax.nn import initializers
import jax.numpy as jnp


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
  """LayoutViT Customized Multi-head dot-product attention.

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
      train: Train or not (to apply dropout)

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
    dropout_rng = self.make_rng('dropout') if train else None
    x = nn.attention.dot_product_attention(
        query,
        key,
        value,
        dtype=self.dtype,
        bias=attention_bias,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=not train)

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
  """LayoutViT Transformer encoder block.

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

    mlp = MlpBlock(
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
  """LayoutViT Transformer decoder block.

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
               query_padding_mask: Optional[jnp.ndarray] = None,
               train: bool = False):
    """Applies DecoderBlock module.

    Args:
      obj_queries: Input data for decoder.
      encoder_output: Output of encoder, which are encoded inputs.
      pos_embedding: Positional Embedding to be added to the keys in
        cross-attention.
      query_pos_emb: Positional Embedding to be added to the queries.
      key_padding_mask: Binary mask containing 0 for pad tokens in key.
      query_padding_mask: Binary mask containing 0 for pad tokens in queries.
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

    mlp = MlpBlock(
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
          key_padding_mask=query_padding_mask,
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
          key_padding_mask=query_padding_mask,
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
  """LayoutViT Transformer Encoder.

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
  """LayoutViT Transformer Decoder.

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
               query_padding_mask: Optional[jnp.ndarray] = None,
               pos_embedding: Optional[jnp.ndarray] = None,
               query_pos_emb: Optional[jnp.ndarray] = None,
               train: bool = False) -> jnp.ndarray:
    """Applies Decoder on the inputs.

    Args:
      obj_queries: Input data for decoder.
      encoder_output: Output of encoder, which are encoded inputs.
      key_padding_mask: Binary mask containing 0 for padding tokens in the keys.
      query_padding_mask: Binary mask containing 0 for padding tokens in the
        queries.
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
              query_padding_mask=query_padding_mask,
              train=train)
      if self.return_intermediate:
        outputs.append(y)

    if self.return_intermediate:
      y = jnp.stack(outputs, axis=0)
    return y if self.norm is None else self.norm(y)
