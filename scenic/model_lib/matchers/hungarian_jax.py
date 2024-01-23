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

"""JAX-based Hungarian matcher implementation."""

import jax
import jax.numpy as jnp


def hungarian_single(cost):
  """Hungarian matcher for a single example."""

  is_transpose = cost.shape[0] > cost.shape[1]
  if is_transpose:
    cost = cost.T

  n, m = cost.shape
  one_hot_m = jnp.eye(m + 1)

  def row_scan_fn(state, i):
    """Loop over the rows of the cost matrix."""
    u, v, parent = state

    # parent[0] = i; note that i runs from 1 to n inclusive
    parent = jax.lax.dynamic_update_index_in_dim(parent, i, 0, axis=0)

    def dfs_body_fn(state):
      # Row potential, column potential, used array, support array, path array
      # column index j0.
      u, v, used, minv, way, j0 = state

      # Mark column as used
      # used = jax.lax.dynamic_update_index_in_dim(used, True, j0, axis=0)
      # used = jnp.logical_or(used, jnp.arange(m + 1) == j0)
      used = jnp.logical_or(used, one_hot_m[j0])
      used_slice = used[1:]

      # Row paired to column j0
      i0 = parent[j0]

      # Update minv and path to it
      cur = cost[i0 - 1, :] - u[i0] - v[1:]
      cur = jnp.where(used_slice, jnp.full_like(cur, 1e10), cur)
      way = jnp.where(cur < minv, jnp.full_like(way, j0), way)
      minv = jnp.where(cur < minv, cur, minv)

      # When finding an index with minimal minv, we need to mask out the visited
      # rows
      masked_minv = jnp.where(used_slice, jnp.full_like(minv, 1e10), minv)
      j1 = jnp.argmin(masked_minv) + 1
      delta = jnp.min(minv, initial=1e10, where=jnp.logical_not(used_slice))

      # Update potentials
      indices = jnp.where(used, parent, n + 1)  # deliberately out of bounds
      u = u.at[indices].add(delta)
      v = jnp.where(used, v - delta, v)
      minv = jnp.where(jnp.logical_not(used_slice), minv - delta, minv)

      return (u, v, used, minv, way, j1)

    def dfs_cond_fn(state):
      _, _, _, _, _, j0 = state
      return parent[j0] != 0

    # Run the inner while loop (i.e. DFS)
    way = jnp.zeros((m,), dtype=jnp.int32)
    used = jnp.zeros((m + 1,), dtype=jnp.bool_)
    minv = jnp.full((m,), 1e10, dtype=jnp.float32)
    init_state = (u, v, used, minv, way, 0)

    state = jax.lax.while_loop(dfs_cond_fn, dfs_body_fn, init_state)
    u, v, _, _, way, j0 = state

    def update_parent_body_fn(state):
      """Update parents based on the DFS path."""
      parent, j0 = state
      j1 = way[j0 - 1]
      parent = jax.lax.dynamic_update_index_in_dim(
          parent, parent[j1], j0, axis=0)
      return (parent, j1)

    def update_parent_cond_fn(state):
      """Condition function counterpart."""
      _, j0 = state
      return j0 != 0

    # Backtrack the DFS path
    init_state = (parent, j0)
    parent, _ = jax.lax.while_loop(
        update_parent_cond_fn, update_parent_body_fn, init_state)

    return (u, v, parent), None

  # Define the initial state
  u = jnp.zeros((n + 2,), dtype=jnp.float32)
  v = jnp.zeros((m + 1,), dtype=jnp.float32)
  parent = jnp.zeros((m + 1,), dtype=jnp.int32)

  init_state = (u, v, parent)
  (u, v, parent), _ = jax.lax.scan(
      row_scan_fn, init_state, jnp.arange(1, n + 1))

  # -v[0] is the matching cost, but not returned to match the signature all
  # other matchers.
  if n != m:
    # This is a costly operation, so skip it when possible (i.e. for square cost
    # matrices).
    parent, indices = jax.lax.top_k(parent[1:], n)
  else:
    parent, indices = parent[1:], jnp.arange(n)
  parent = parent - 1  # Switch back to 0-based indexing.

  if is_transpose:
    return jnp.stack([indices, parent], axis=0)
  return jnp.stack([parent, indices], axis=0)


def hungarian_scan(cost):
  """A scan-based batch version of the hungarian matching."""
  def hungarian_fn(_, cost):
    return None, hungarian_single(cost)
  _, indices = jax.lax.scan(hungarian_fn, None, cost, unroll=1)
  return indices


hungarian_tpu_matcher = jax.jit(jax.vmap(hungarian_single))
hungarian_scan_tpu_matcher = jax.jit(hungarian_scan)
