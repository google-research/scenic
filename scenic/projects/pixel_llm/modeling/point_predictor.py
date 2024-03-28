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

"""Point location prediction module."""

import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
  """MLP module."""
  hidden_dim: int
  output_dim: int
  num_layers: int
  zero_out: bool = False
  activation: str = 'relu'

  @nn.compact
  def __call__(self, x):
    for i in range(self.num_layers - 1):
      x = nn.Dense(self.hidden_dim, name=f'layers.{i}')(x)
      if self.activation == 'gelu':
        x = nn.gelu(x, approximate=False)
      elif self.activation == 'relu':
        x = nn.relu(x)
      else:
        raise NotImplementedError(self.activation)
    if self.zero_out:
      x = nn.Dense(
          self.output_dim,
          kernel_init=nn.initializers.normal(0.0001),
          bias_init=nn.initializers.zeros_init(),
          name=f'layers.{self.num_layers - 1}',
      )(x)
    else:
      x = nn.Dense(self.output_dim, name=f'layers.{self.num_layers - 1}')(x)
    return x


class MlpPointPredictor(nn.Module):
  """Predict points for each token."""

  hidden_dim: int = 512
  depth: int = 3
  num_output_points: int = 4
  num_classes: int = 1
  zero_out: bool = False
  pre_norm: bool = False
  mlp_activation: str = 'relu'

  def setup(self):
    self.mlp = MLP(
        hidden_dim=self.hidden_dim,
        # x,y + background
        output_dim=self.num_output_points * (3 + self.num_classes),
        num_layers=self.depth,
        zero_out=self.zero_out,
        activation=self.mlp_activation,
        name='mlp',
    )

  @nn.compact
  def __call__(self, visual_features, text_feat):
    """Predicot points for each token.

    Args:
      visual_features: (B, N, L1, C), not used
      text_feat: (batch_size, ..., hidden_size)

    Returns:
      pred_points: (batch_size, ..., num_points, 2)
    """
    del visual_features
    if self.pre_norm:
      text_feat = nn.LayerNorm(epsilon=1e-6)(text_feat)
    pred_points = self.mlp(text_feat)
    pred_points = jnp.reshape(
        pred_points,
        pred_points.shape[:-1] + (self.num_output_points, 3 + self.num_classes),
    )
    point_coords = pred_points[..., :2]
    logits = pred_points[..., 2:]
    # point_coords = jnp.clip(point_coords, -1, 1)

    point_coords = (point_coords + 1) * 0.5

    return point_coords, logits
