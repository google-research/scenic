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

"""Implement Object-aware attention block."""

import functools
from typing import Any, Callable, Iterable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.layers import attention_layers


Initializer = Callable[[jnp.ndarray, Iterable[int], jnp.dtype], jnp.ndarray]


class ObjectBlock(nn.Module):
  """The same as vivit block. Supports object attention and masked tokens."""
  mlp_dim: Optional[int]
  num_heads: int
  dtype: jnp.dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  attention_kernel_initializer: Initializer = nn.initializers.xavier_uniform()
  mlp_kernel_initializer: Initializer = nn.initializers.xavier_uniform()
  mlp_bias_initializer: Initializer = nn.initializers.normal(stddev=1e-6)
  attention_fn: Any = nn.dot_product_attention
  droplayer_p: float = 0.0
  use_approximate_gelu: bool = True
  configs: ml_collections.ConfigDict = ml_collections.ConfigDict()

  def get_drop_pattern(self, x, deterministic):
    if not deterministic and self.droplayer_p:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng('dropout'), self.droplayer_p, shape).astype('float32')
    else:
      return 0.0

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, deterministic: bool,
      token_scores: jnp.ndarray,
    ) -> jnp.ndarray:
    """Applies Encoder1DBlock module.

    Args:
      inputs: batch x num_fg_tokens x hidden_dim;
      deterministic: bool
      token_scores: optional: batch x num_objs x num_fg_tokens
    Returns:
      x: batch x num_fg_tokens x hidden_dim
    """
    num_tokens_per_object = self.configs.get('num_tokens_per_object', 8)
    norm_objects = self.configs.get('norm_objects', False)
    with_objects_linear = self.configs.get('with_objects_linear', True)
    with_traj_emb = self.configs.get('with_traj_emb', True)

    batch, num_fg_tokens, hidden_dim = inputs.shape
    num_objs = token_scores.shape[1]

    patch_tokens = inputs
    object_tokens = patch_tokens[:, None] * token_scores[..., None]
    # batch x num_objs x N x D

    if norm_objects:
      object_tokens = object_tokens.sum(axis=2) / (
          token_scores[..., None].sum(axis=2) + 1e-4)
      object_tokens = object_tokens[:, :, None, :]
      # batch x num_objs x 1 x D

    if with_objects_linear:
      object_tokens = nn.Dense(hidden_dim // 2, use_bias=False)(object_tokens)
      object_tokens = nn.relu(object_tokens)
      object_tokens = nn.Dense(hidden_dim, use_bias=False)(object_tokens)
      object_tokens = nn.relu(object_tokens)  # batch x num_objs x N x D

    if not norm_objects:
      object_tokens = object_tokens.transpose(
          0, 1, 3, 2)  # batch x num_objs x D x N
      object_tokens, _ = jax.lax.top_k(
          object_tokens, k=num_tokens_per_object)  # batch x num_objs x D x k
      object_tokens = object_tokens.transpose(
          0, 1, 3, 2)  # batch x num_objs x k x D

    if with_traj_emb:
      box_categories = self.param(
          'box_categories', nn.initializers.zeros,
          (1, num_objs, 1, hidden_dim), jnp.float32)
      object_tokens = object_tokens + box_categories

    object_tokens = object_tokens.reshape(batch, -1, hidden_dim)
    all_tokens = jnp.concatenate(
        [patch_tokens, object_tokens], axis=1,
    )  # batch x (num_fg_tokens + num_objs * k) x D

    # Attention block.
    x = nn.LayerNorm(dtype=self.dtype)(all_tokens)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=self.attention_kernel_initializer,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        attention_fn=self.attention_fn,
        dtype=self.dtype)(
            x, x, deterministic=deterministic)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)

    x = x[:, :num_fg_tokens, :]

    drop_pattern = self.get_drop_pattern(x, deterministic)
    x = x * (1.0 - drop_pattern) + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = attention_layers.MlpBlock(  # pytype: disable=wrong-arg-types  # jnp-type
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=functools.partial(
            nn.gelu, approximate=self.use_approximate_gelu),
        kernel_init=self.mlp_kernel_initializer,
        bias_init=self.mlp_bias_initializer)(
            y, deterministic=deterministic)

    drop_pattern = self.get_drop_pattern(x, deterministic)
    ret = y * (1.0 - drop_pattern) + x
    return ret
