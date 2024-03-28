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

r"""Sam transformer for running cross-attention.

Pytorch reference:

https://github.com/facebookresearch/segment-anything/blob/HEAD/\
segment_anything/modeling/transformer.py

"""

import math
from typing import Any

import flax.linen as nn
import jax.numpy as jnp


class TwoWayTransformer(nn.Module):
  """Transformer with query and key/ value inputs."""

  depth: int = 2
  embedding_dim: int = 256
  num_heads: int = 8
  mlp_dim: int = 2048
  activation: Any = nn.relu
  attention_downsample_rate: int = 2

  def setup(self):
    layers = []
    for i in range(self.depth):
      layer = TwoWayAttentionBlock(
          embedding_dim=self.embedding_dim,
          num_heads=self.num_heads,
          mlp_dim=self.mlp_dim,
          activation=self.activation,
          attention_downsample_rate=self.attention_downsample_rate,
          skip_first_layer_pe=(i == 0),
          name=f'layers.{i}')
      layers.append(layer)
    self.layers = layers

    self.final_attn_token_to_image = Attention(
        self.embedding_dim, self.num_heads, self.attention_downsample_rate,
        name='final_attn_token_to_image')
    self.norm_final_attn = nn.LayerNorm(epsilon=1e-5, name='norm_final_attn')

  def __call__(self, image_embedding, image_pe, point_embedding):
    """Forward pass.

    Args:
      image_embedding: (batch_size, h, w, embedding_dim)
      image_pe: (batch_size, h, w, embedding_dim)
      point_embedding: (batch_size, num_points, embedding_dim)
    Returns:
    """
    batch_size, c = image_embedding.shape[0], image_embedding.shape[-1]
    image_embedding = image_embedding.reshape((batch_size, -1, c))
    image_pe = image_pe.reshape((batch_size, -1, c))

    # Prepare queries
    queries = point_embedding
    keys = image_embedding

    # Apply transformer blocks and final layernorm
    for layer in self.layers:
      queries, keys = layer(
          queries=queries,
          keys=keys,
          query_pe=point_embedding,
          key_pe=image_pe,
      )

    # Apply the final attention layer from the points to the image
    q = queries + point_embedding
    k = keys + image_pe
    attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
    queries = queries + attn_out
    queries = self.norm_final_attn(queries)

    return queries, keys


class TwoWayAttentionBlock(nn.Module):
  """Transformer block."""

  embedding_dim: int
  num_heads: int
  mlp_dim: int = 2048
  activation: Any = nn.relu
  attention_downsample_rate: int = 2
  skip_first_layer_pe: bool = False

  def setup(self):
    self.self_attn = Attention(
        self.embedding_dim, self.num_heads, name='self_attn')
    self.norm1 = nn.LayerNorm(epsilon=1e-5, name='norm1')

    self.cross_attn_token_to_image = Attention(
        self.embedding_dim, self.num_heads, self.attention_downsample_rate,
        name='cross_attn_token_to_image')
    self.norm2 = nn.LayerNorm(epsilon=1e-5, name='norm2')

    self.mlp = MLPBlock(
        self.embedding_dim, self.mlp_dim, self.activation,
        name='mlp')
    self.norm3 = nn.LayerNorm(epsilon=1e-5, name='norm3')

    self.norm4 = nn.LayerNorm(epsilon=1e-5, name='norm4')
    self.cross_attn_image_to_token = Attention(
        self.embedding_dim, self.num_heads, self.attention_downsample_rate,
        name='cross_attn_image_to_token')

  def __call__(self, queries, keys, query_pe, key_pe):
    """Forward two-way attention block.

    Args:
      queries: (batch_size, query_tokens, embedding_dim)
      keys: (batch_size, key_tokens, embedding_dim)
      query_pe: (batch_size, query_tokens, embedding_dim)
      key_pe: (batch_size, key_tokens, embedding_dim)
    Returns:
      queries: (batch_size, query_tokens, embedding_dim)
      keys: (batch_size, key_tokens, embedding_dim)
    """
    # Self attention block
    if self.skip_first_layer_pe:
      queries = self.self_attn(q=queries, k=queries, v=queries)
    else:
      q = queries + query_pe
      attn_out = self.self_attn(q=q, k=q, v=queries)
      queries = queries + attn_out
    queries = self.norm1(queries)

    # Cross attention block, tokens attending to image embedding
    q = queries + query_pe
    k = keys + key_pe
    attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
    queries = queries + attn_out
    queries = self.norm2(queries)

    # MLP block
    mlp_out = self.mlp(queries)
    queries = queries + mlp_out
    queries = self.norm3(queries)

    # Cross attention block, image embedding attending to tokens
    q = queries + query_pe
    k = keys + key_pe
    attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
    keys = keys + attn_out
    keys = self.norm4(keys)

    return queries, keys


class Attention(nn.Module):
  """Attention module."""
  embedding_dim: int
  num_heads: int
  downsample_rate: int = 1

  def setup(self):
    self.internal_dim = self.embedding_dim // self.downsample_rate
    assert self.internal_dim % self.num_heads == 0, (
        'num_heads must divide embedding_dim.')

    self.q_proj = nn.Dense(self.internal_dim, name='q_proj')
    self.k_proj = nn.Dense(self.internal_dim, name='k_proj')
    self.v_proj = nn.Dense(self.internal_dim, name='v_proj')
    self.out_proj = nn.Dense(self.embedding_dim, name='out_proj')

  def _separate_heads(self, x):
    b, n, c = x.shape
    x = x.reshape(b, n, self.num_heads, c // self.num_heads)
    return x.transpose((0, 2, 1, 3))  # B x N_heads x N_tokens x C_per_head

  def _recombine_heads(self, x):
    b, n_heads, n_tokens, c_per_head = x.shape
    x = x.transpose((0, 2, 1, 3))
    return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

  def __call__(self, q, k, v):
    """Forward attention module.

    Args:
      q: (batch_size, query_tokens, embedding_dim)
      k: (batch_size, key_tokens, embedding_dim)
      v: (batch_size, key_tokens, embedding_dim)
    Returns:
      out: (batch_size, query_tokens, embedding_dim)
    """
    # Input projections
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)

    # Separate into heads
    q = self._separate_heads(q)  # (batch_size, num_heads, n, c_per_head)
    k = self._separate_heads(k)  # (batch_size, num_heads, m, c_per_head)
    v = self._separate_heads(v)  # (batch_size, num_heads, m, c_per_head)

    # Attention
    _, _, _, c_per_head = q.shape
    attn = jnp.matmul(
        q, k.transpose((0, 1, 3, 2)))  # B x N_heads x N_tokens x N_tokens
    attn = attn / math.sqrt(c_per_head)
    attn = nn.softmax(attn, axis=-1)

    # Get output
    out = jnp.matmul(attn, v)
    out = self._recombine_heads(out)
    out = self.out_proj(out)

    return out


class MLPBlock(nn.Module):
  embedding_dim: int
  mlp_dim: int
  activation: Any = nn.relu  # Confirmed in the original code.

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.mlp_dim, name='lin1')(x)
    x = self.activation(x)
    x = nn.Dense(self.embedding_dim, name='lin2')(x)
    return x
