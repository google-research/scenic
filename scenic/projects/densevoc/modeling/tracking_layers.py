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

"""GTR Transformer.


Implementation of transformer from Global Tracking Transformers (GTR).
Mostly the same as DETRTransformer, except that:
* no layer normalization.
* no position encoding.
* typically a single layer for encoder and decoder.

Pytorch reference:
https://github.com/xingyizhou/GTR/blob/master/gtr/modeling/roi_heads/
transformer.py
Reference: https://arxiv.org/pdf/2203.13250.pdf
"""

import functools
from typing import Any, Callable, Optional

import einops
import flax.linen as nn
import jax
from jax.nn import initializers
import jax.numpy as jnp
from scenic.model_lib.layers import attention_layers
from scenic.projects.baselines.detr import model as detr_model

Array = jnp.ndarray
PyTree = Any

# Match PyTorch LayerNorm.
LayerNorm = functools.partial(nn.LayerNorm, epsilon=1e-5)


class GTRAssoHead(nn.Module):
  """Association head for Global Tracking Transformer."""

  dim: int = 512
  num_layers: int = 2

  @nn.compact
  def __call__(self, x: PyTree) -> PyTree:
    x = einops.rearrange(x, '... h w c -> ... (h w c)')
    for i in range(self.num_layers):
      x = nn.Dense(
          features=self.dim,
          kernel_init=initializers.variance_scaling(
              1, mode='fan_in', distribution='uniform'
          ),
          bias_init=initializers.zeros,
          name=f'fc{i+1}',
      )(x)
      x = jax.nn.relu(x)
    return x


class GTREncoderLayer(nn.Module):
  """GTR Encoder Layer.

  Assumes post-normalization and defaults to disabling layer norm.

  Attributes:
    num_heads: Number of heads.
    num_features: Feature dimension.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout rate for attention weights.
    dtype: Data type of the computation (default: float32).
  """

  num_heads: int
  num_features: int
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  norm: bool = False
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      *,
      pos_embedding: Optional[jnp.ndarray] = None,
      padding_mask: Optional[jnp.ndarray] = None,
      train: bool = False,
  ) -> jnp.ndarray:
    """Applies GTREncoderLayer module.

    Args:
      inputs: Input data of shape (batch_size, len, features).
      pos_embedding: Positional Embedding to be added to the queries and keys in
        the self-attention operation.
      padding_mask: Binary mask containing 0 for padding tokens.
      train: Train or not (to apply dropout).

    Returns:
      Output: (batch_size, len, features).
    """
    self_attn = detr_model.MultiHeadDotProductAttention(
        name='self_attn',
        num_heads=self.num_heads,
        qkv_features=self.num_features,
        dropout_rate=self.attention_dropout_rate,
        broadcast_dropout=False,
        kernel_init=initializers.xavier_uniform(),
        bias_init=initializers.zeros,
        use_bias=True,
        dtype=self.dtype,
    )

    mlp = attention_layers.MlpBlock(  # pytype: disable=wrong-arg-types  # jnp-type
        name='mlp',
        mlp_dim=self.num_features,
        activation_fn=nn.relu,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
    )

    assert inputs.ndim == 3
    x = self_attn(
        inputs_q=inputs,
        pos_emb_q=pos_embedding,
        pos_emb_k=pos_embedding,
        pos_emb_v=None,
        key_padding_mask=padding_mask,
        train=train,
    )
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
    x = x + inputs
    if self.norm:
      x = LayerNorm(dtype=self.dtype)(x)
    y = mlp(x, deterministic=not train)
    y = x + y
    out = y
    if self.norm:
      out = LayerNorm(dtype=self.dtype)(out)

    return out


class GTRDecoderLayer(nn.Module):
  """GTRTransformer decoder layer.

  Attributes:
    num_heads: Number of heads.
    num_features: Dimension of the query/key/value.
    dropout_rate:Dropout rate.
    attention_dropout_rate:Dropout rate for attention weights.
    dtype: Data type of the computation (default: float32).
  """

  num_heads: int
  num_features: int
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  use_self_attn: bool = False
  norm: bool = False
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      obj_queries: jnp.ndarray,
      encoder_output: jnp.ndarray,
      *,
      pos_embedding: Optional[jnp.ndarray] = None,
      query_pos_emb: Optional[jnp.ndarray] = None,
      key_padding_mask: Optional[jnp.ndarray] = None,
      train: bool = False,
  ):
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

    # Seems in DETR the self-attention in the first layer basically does
    # nothing, as the  value vector is a zero vector and we add no learnable
    # positional embedding to it!
    self_attn = detr_model.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=self.num_features,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        kernel_init=initializers.xavier_uniform(),
        bias_init=initializers.zeros,
        use_bias=True,
        dtype=self.dtype,
    )

    cross_attn = detr_model.MultiHeadDotProductAttention(
        name='cross_attn',
        num_heads=self.num_heads,
        qkv_features=self.num_features,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        kernel_init=initializers.xavier_uniform(),
        bias_init=initializers.zeros,
        use_bias=True,
        dtype=self.dtype,
    )

    mlp = attention_layers.MlpBlock(  # pytype: disable=wrong-arg-types  # jnp-type
        name='mlp',
        mlp_dim=self.num_features,
        activation_fn=nn.relu,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
    )

    assert obj_queries.ndim == 3

    # Optional self attention block.
    if self.use_self_attn:
      x = self_attn(
          inputs_q=obj_queries,
          pos_emb_q=query_pos_emb,
          pos_emb_k=query_pos_emb,
          pos_emb_v=None,
          train=train,
      )
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
      x = x + obj_queries
      if self.norm:
        x = LayerNorm(dtype=self.dtype)(x)
    else:
      x = obj_queries

    # cross attention block
    y = cross_attn(
        inputs_q=x,
        inputs_kv=encoder_output,
        pos_emb_q=query_pos_emb,
        pos_emb_k=pos_embedding,
        pos_emb_v=None,
        key_padding_mask=key_padding_mask,
        train=train,
    )

    y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
    y = y + x
    if self.norm:
      y = LayerNorm(dtype=self.dtype)(y)

    # mlp block
    z = mlp(y, deterministic=not train)
    z = y + z
    if self.norm:
      z = LayerNorm(dtype=self.dtype)(z)

    return z


class GTREncoder(nn.Module):
  """GTRTransformer Encoder.

  Attributes:
    num_heads: Number of heads.
    num_layers: Number of layers.
    num_features: Dimension of the query/key/value.
    norm: normalization layer to be applied on the output.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout rate for attention weights.
    dtype: Data type of the computation (default: float32).
  """

  num_heads: int
  num_layers: int
  num_features: int
  norm: Callable[..., Any] = lambda x: x
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      *,
      pos_embedding: Optional[jnp.ndarray] = None,
      padding_mask: Optional[jnp.ndarray] = None,
      train: bool = False,
  ) -> jnp.ndarray:
    """Applies Encoder on the inputs.

    Args:
      inputs: (batch_size, num_objs, D)
      pos_embedding: Positional Embedding to be added to the queries and keys in
        the self-attention operation.
      padding_mask: Binary mask containing 0 for padding tokens, and 1
        otherwise.
      train: Whether it is training.

    Returns:
      Output of the transformer encoder.
    """
    assert inputs.ndim == 3
    x = inputs

    # Input encoder
    for lyr in range(self.num_layers):
      x = GTREncoderLayer(
          num_heads=self.num_heads,
          num_features=self.num_features,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          dtype=self.dtype,
      )(x, pos_embedding=pos_embedding, padding_mask=padding_mask, train=train)

    x = self.norm(x)
    return x


class GTRDecoder(nn.Module):
  """GTR Transformer Decoder.

  Attributes:
    num_heads: Number of heads.
    num_layers: Number of layers.
    num_features: Dimension of the query/key/value.
    return_intermediate: If return the outputs from intermediate layers.
    padding_mask: Binary mask containing 0 for padding tokens.
    dropout_rate:Dropout rate.
    attention_dropout_rate:Dropout rate for attention weights.
    dtype: Data type of the computation (default: float32).
  """

  num_heads: int
  num_layers: int
  num_features: int
  norm: Callable[..., Any] = lambda x: x
  return_intermediate: bool = False
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      obj_queries: jnp.ndarray,
      encoder_output: jnp.ndarray,
      *,
      key_padding_mask: Optional[jnp.ndarray] = None,
      pos_embedding: Optional[jnp.ndarray] = None,
      query_pos_emb: Optional[jnp.ndarray] = None,
      train: bool = False,
  ) -> jnp.ndarray:
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
      y = GTRDecoderLayer(
          num_features=self.num_features,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          dtype=self.dtype,
          name=f'decoderblock_{lyr}',
      )(
          y,
          encoder_output,
          pos_embedding=pos_embedding,
          query_pos_emb=query_pos_emb,
          key_padding_mask=key_padding_mask,
          train=train,
      )
      if self.return_intermediate:
        outputs.append(y)

    if self.return_intermediate:
      y = jnp.stack(outputs, axis=0)
    return self.norm(y)


class GTRTransformer(nn.Module):
  """GTR Transformer.

  Attributes:
    num_heads: Number of heads.
    num_encoder_layers: Number of encoder layers.
    num_decoder_layers: Number of decoder layers.
    num_features: Dimension of the query/key/value.
    dropout_rate: Dropout rate.
    attention_dropout_rate:Dropout rate for attention weights.
    dtype: Data type of the computation (default: float32).
  """

  num_heads: int = 8
  num_encoder_layers: int = 1
  num_decoder_layers: int = 1
  num_features: int = 512
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: Array, train: bool = False) -> Array:
    """Applies GTRTransformer on the inputs.

    Args:
      inputs: batch_size x num_tot_objects x D. make sure batch_size = 1.
      train: Whether it is training.

    Returns:
      Output: association matrix: batch_size x num_tot_objects x num_tot_objects
    """
    queries = inputs

    encoded = GTREncoder(
        num_heads=self.num_heads,
        num_layers=self.num_encoder_layers,
        num_features=self.num_features,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        dtype=self.dtype,
        name='encoder',
    )(inputs, train=train)  # (batch_size, num_tot_objects, D)

    output = GTRDecoder(
        num_heads=self.num_heads,
        num_layers=self.num_decoder_layers,
        num_features=self.num_features,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        dtype=self.dtype,
        name='decoder',
    )(queries, encoded, train=train)  # (batch_size, num_tot_objects, D)
    # pred_asso: (batch_size, num_tot_objects, num_tot_objects)
    pred_asso = jnp.einsum('bnc,bmc->bnm', output, encoded)
    return pred_asso
