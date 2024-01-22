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

"""Greedy matcher."""
import jax
import jax.numpy as jnp


@jax.vmap
def greedy_matcher(cost):
  """Computes greedy bipartite matching given a cost matrix.

  The code applies to a single matrix; vmap is applied for batching.

  Args:
    cost: jnp.ndarray; Cost matrix for the matching of shape [N, M].

  Returns:
    An assignment of size [2, min(N, M)].
  """
  # Ensure that the shorter dimension comes first:
  transposed = cost.shape[0] > cost.shape[1]
  if transposed:
    cost = jnp.transpose(cost)

  # Max cost is used for masking:
  max_cost = jnp.max(cost)

  def select(cost, _):
    min_index_flat = jnp.argmin(cost)
    min_row, min_col = jnp.unravel_index(min_index_flat, cost.shape)
    cost = cost.at[min_row, :].set(max_cost + 1)
    cost = cost.at[:, min_col].set(max_cost + 1)
    return cost, jnp.array([min_row, min_col])

  _, indices = jax.lax.scan(f=select, init=cost, xs=None, length=cost.shape[0])

  if transposed:
    indices = jnp.flip(indices, axis=1)

  # From [N/M, 2] to [2, N/M]:
  return jnp.transpose(indices)
