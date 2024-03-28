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

"""Mask Adapter."""

import flax.linen as nn
import jax.numpy as jnp
from scenic.projects.baselines.segment_anything.modeling import transformer


class SamMaskAdaptor(nn.Module):
  """Mask Adaptor."""

  depth: int = 2
  input_projection: bool = True
  transformer_dim: int = 256
  num_outputs: int = 2
  head_hidden_dim: int = 512
  output_dim: int = 256
  gating: bool = True

  def setup(self):
    self.output_tokens = self.param(
        'output_tokens.weight',
        nn.initializers.normal(stddev=1.0),
        (self.num_outputs, self.transformer_dim),
    )
    # NOTE(jiaruixu): borrow the arch for fast impl
    self.transformer = transformer.TwoWayTransformer(
        depth=self.depth, embedding_dim=self.transformer_dim, name='transformer'
    )
    if self.input_projection:
      self.visual_projection_mlp = MLP(
          hidden_dim=self.head_hidden_dim,
          output_dim=self.transformer_dim,
          num_layers=1,
          name='visual_projection_mlp',
      )
      self.textual_projection_mlp = MLP(
          hidden_dim=self.head_hidden_dim,
          output_dim=self.transformer_dim,
          num_layers=1,
          name='textual_projection_mlp',
      )
    if self.gating:
      self.alpha_output = self.param(
          'alpha_output.weight',
          nn.initializers.constant(0.0),
          (self.num_outputs, self.output_dim),
      )

  @nn.compact
  def __call__(self, sparse_embedding, visual_features, textual_features):
    """Predict sam sparse embedding based on text_features.

    Args:
      sparse_embedding: (B, N, O, C)
      visual_features: (B, N, L1, C)
      textual_features: (B, N, L2, C)

    Returns:
      output_embed: (B, N, O, C)
    """
    if self.input_projection:
      visual_features = self.visual_projection_mlp(visual_features)
      textual_features = self.textual_projection_mlp(textual_features)

    batch_size, num_prompts, max_cap_len, embed_dim = textual_features.shape
    assert visual_features.shape[:2] == (batch_size, num_prompts)
    assert visual_features.shape[-1] == embed_dim

    # [B*N, L2, C]
    textual_features = jnp.reshape(
        textual_features, (batch_size * num_prompts, max_cap_len, embed_dim)
    )
    visual_features = jnp.reshape(
        visual_features, (batch_size * num_prompts, -1, embed_dim)
    )

    # [B*N, O, C]
    output_tokens = jnp.broadcast_to(
        self.output_tokens[None],
        (batch_size * num_prompts, self.num_outputs, self.transformer_dim),
    )

    # [B*N, L1+L2, C]
    concat_features = jnp.concatenate(
        [visual_features, textual_features], axis=1
    )
    output_embed, _ = self.transformer(
        concat_features, jnp.zeros_like(concat_features), output_tokens
    )
    # [B, N, O, C]
    output_embed = jnp.reshape(
        output_embed,
        (batch_size, num_prompts, self.num_outputs, self.output_dim),
    )

    if self.gating:
      output_embed = jnp.tanh(self.alpha_output) * output_embed

    output_embed += sparse_embedding

    return output_embed


class MLP(nn.Module):
  """MLP with pre-norm."""
  hidden_dim: int
  output_dim: int
  num_layers: int
  pre_norm: bool = True
  activation: str = 'gelu'

  @nn.compact
  def __call__(self, x):
    if self.pre_norm:
      x = nn.LayerNorm(epsilon=1e-6)(x)
    for i in range(self.num_layers - 1):
      x = nn.Dense(self.hidden_dim, name=f'layers.{i}')(x)
      if self.activation == 'gelu':
        x = nn.gelu(x, approximate=False)
      elif self.activation == 'relu':
        x = nn.relu(x)
      else:
        raise NotImplementedError(self.activation)
    x = nn.Dense(self.output_dim, name=f'layers.{self.num_layers - 1}')(x)
    return x
