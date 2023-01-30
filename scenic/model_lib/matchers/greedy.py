# Copyright 2022 The Scenic Authors.
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

"""Greedy matcher."""
import jax
import jax.numpy as jnp
import numpy as np


def greedy_matcher(cost):
  """Computes Greedy Matching on cost matrix for a batch of datapoints.

  Args:
    cost: jnp.ndarray; Cost matrix for the matching of shape [B, N, M].

  Returns:
    An assignment of size [B, 2, min(N, M)].
  """
  # Break potential equalities.
  cost += jax.random.uniform(jax.random.PRNGKey(0), cost.shape, maxval=1e-4)
  grid = np.stack(
      np.meshgrid(
          np.arange(cost.shape[1]), np.arange(cost.shape[2]), indexing='ij'),
      -1)
  grid = jnp.array(grid, dtype=jnp.int32)

  def greedy_select(cost, _):
    min_element = jnp.min(cost, axis=[1, 2], keepdims=True)
    selected = cost == min_element
    indices = jnp.einsum('bnm,nm2->b2', selected, grid)
    mask = jnp.maximum(
        selected.max(axis=1, keepdims=True),
        selected.max(axis=2, keepdims=True))
    cost += mask * 1e6
    return cost, indices

  _, indices = jax.lax.scan(greedy_select, cost, None,
                            min(cost.shape[1], cost.shape[2]))
  return indices.transpose([1, 2, 0])
