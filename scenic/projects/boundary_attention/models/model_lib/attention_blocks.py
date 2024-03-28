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

"""Transformer block for boundary attention."""

from typing import Optional
import flax.linen as nn
import jax.numpy as jnp
from scenic.projects.boundary_attention.models.model_lib import model_utils


class TransformerBlock(nn.Module):
  """Finds next hidden state given junction features and prev hidden state."""

  M: int = 3
  C: int = 3
  patchsize: int = 5
  encoding_dim: int = 256
  num_heads: Optional[int] = 4
  num_layers: Optional[int] = 1
  attn_dropout_prob: Optional[float] = 0

  @nn.compact
  def __call__(self,
               hidden_state: jnp.ndarray,
               embedded_hidden_state: jnp.ndarray,
               train: bool = True):

    hidden_dim = embedded_hidden_state.shape[-1]

    # Extract patches from the input
    hidden_state_patches, bin_mask = model_utils.extract_patches(
        embedded_hidden_state, (self.patchsize, self.patchsize), 1)

    # Add positional encoding to patches
    pos_embedding = self.param('PositionalEmbedding',
                               nn.initializers.lecun_normal(),
                               (self.patchsize, self.patchsize, hidden_dim))
    hidden_state_patches = hidden_state_patches + jnp.expand_dims(
        pos_embedding, [0, 1, 2])

    # Flatten patches and prepare binary mask
    hidden_state_patches_flattened = hidden_state_patches.reshape(
        [-1, hidden_state_patches.shape[1], hidden_state_patches.shape[2],
         hidden_state_patches.shape[3]*hidden_state_patches.shape[4],
         hidden_state_patches.shape[5]])
    bin_mask_flattened = bin_mask.reshape(
        [-1, bin_mask.shape[1], bin_mask.shape[2], 1,
         bin_mask.shape[3]*bin_mask.shape[4], bin_mask.shape[5]]).squeeze(-1)

    # Process through transformer encoder layers
    for _ in range(self.num_layers):
      hidden_state = EncoderBlock(hidden_state.shape[-1],
                                  self.encoding_dim, self.num_heads,
                                  self.attn_dropout_prob)(
                                      jnp.expand_dims(hidden_state, 3),
                                      hidden_state_patches_flattened,
                                      jnp.expand_dims(bin_mask_flattened,
                                                      -2).astype(bool),
                                      train=train)

    # Return new hidden state
    return hidden_state


class EncoderBlock(nn.Module):
  """Encoder block for transformer."""

  hidden_dim: int
  dim_conv: int
  num_heads: int = 4
  dropout_prob: float = 0

  @nn.compact
  def __call__(self,
               hidden_state: jnp.ndarray,
               hidden_state_kv: jnp.ndarray,
               attn_mask: jnp.ndarray,
               train=True):

    next_hidden_state = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads)(hidden_state, hidden_state_kv,
                                  mask=attn_mask, deterministic=not train)

    hidden_state = hidden_state + next_hidden_state
    hidden_state = nn.LayerNorm()(hidden_state)

    # MLP
    mlp_out = nn.Dense(self.dim_conv)(hidden_state)
    mlp_out = nn.Dropout(self.dropout_prob,
                         name='MLPDropout')(mlp_out, deterministic=not train)
    mlp_out = nn.relu(mlp_out)
    mlp_out = nn.Dense(self.hidden_dim)(mlp_out)

    hidden_state = hidden_state + mlp_out
    hidden_state = nn.LayerNorm()(hidden_state)

    return hidden_state.squeeze(-2)
