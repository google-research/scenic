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

"""Layer utilis."""

import flax.linen as nn


class LinearProjectLayers(nn.Module):
  """Linear projection layer."""
  emb_dim: int = 1024
  use_projection_ln: bool = True

  @nn.compact
  def __call__(self, x, train=False):
    # The name `visual_projection.x` is for a historical reason to load
    # weights for other decoders. This is not meaningful here now.
    x = nn.Dense(
        self.emb_dim, name='visual_projection.0',
        kernel_init=nn.initializers.normal(stddev=0.02))(
            x)  # (batch_size, feature_length, hidden_size)
    if self.use_projection_ln:
      x = nn.LayerNorm(
          epsilon=1e-5, name='visual_projection.1')(x)
    return x
