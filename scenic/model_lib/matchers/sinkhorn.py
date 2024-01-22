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

"""Sinkhorn matcher.

"""


from typing import Optional
import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry import geometry
from ott.tools import transport


def idx2permutation(row_ind, col_ind):
  """Constructs a permutation matrix from the column and row indices of ones."""
  bs, dim = row_ind.shape[:2]
  perm = jnp.zeros(shape=(bs, dim, dim), dtype='float32')
  perm = jax.vmap((lambda x, idx, y: x.at[idx].set(y)),
                  (0, 0, None))(perm, (row_ind, col_ind), 1.)
  return perm


def sample_permutation(key, coupling):
  """Samples a permutation matrix from a doubly stochastic coupling matrix.

  CAREFUL: the couplings that come out of the Sinkhorn solver are not doubly
  stochastic but 1/dim * doubly_stochastic.

  See **Convex Relaxations for Permutation Problems** paper for rough
  explanation of the algorithm.

  Best to use by drawing multiple samples and picking the permutation with
  lowest cost as sometimes permutations seem to be drawn with high cost. The
  sample_best_permutation method does this.

  Args:
    key: jnp.ndarray; Functions as a PRNG key.
    coupling: jnp.ndarray; Has shape [N, N] which must have marginals such that
      coupling.sum(0) == 1. and coupling.sum(1) == 1. Note that in Sinkhorn we
      usually output couplings with marginals that sum to 1/N.

  Returns:
    Permutation matrix: jnp.ndarray of shape [N, N] of floating dtype.
  """
  bs, dim = coupling.shape[:2]

  # Random monotonic vector v without duplicates.
  v = jax.random.choice(key, 10 * dim, shape=(bs, dim), replace=False)
  v = jnp.sort(v, axis=-1) * 10.

  w = jnp.einsum('bnm,bm->bn', coupling, v)
  # Sorting w will give the row indices of the permutation matrix.
  row_ind = jnp.argsort(w, axis=-1)
  col_ind = jnp.tile(jnp.arange(0, dim)[None, :], [bs, 1])

  # Compute permutation matrix from row and column indices.
  perm = idx2permutation(row_ind, col_ind)
  return perm


def sample_best_permutation(key, coupling, cost, num_trials=10):
  """Samples permutation matrices and returns the one with lowest cost.

  See **Convex Relaxations for Permutation Problems** paper for rough
  explanation of the algorithm.

  Args:
    key: jnp.ndarray; functions as a PRNG key.
    coupling: jnp.ndarray; has shape [N, N].
    cost: jnp.ndarray; has shape [N, N].
    num_trials: int; determines the amount of times we sample a permutation.

  Returns:
    Permutation matrix: jnp.ndarray of shape [N, N] of floating point type.
      This is the permutation matrix with lowest optimal transport cost.
  """
  vec_sample_permutation = jax.vmap(sample_permutation, in_axes=(0, None),
                                    out_axes=0)
  key = jax.random.split(key, num_trials)
  perms = vec_sample_permutation(key, coupling)

  # Pick the permutation with minimal ot cost
  ot = jnp.einsum('nbij,bij->nb', perms, cost)
  min_idx = jnp.argmin(ot, axis=0)
  out_perm = jax.vmap(jnp.take, (1, 0, None))(perms, min_idx, 0)
  return out_perm


def sinkhorn_matcher(cost: jnp.ndarray,
                     rng: Optional[jnp.ndarray] = None,
                     epsilon: float = 0.001,
                     init: float = 50,
                     decay: float = 0.9,
                     num_iters: int = 1000,
                     num_permutations: int = 100,
                     threshold: float = 1e-2,
                     chg_momentum_from: int = 100):
  """Computes Sinkhorn Matching on cost matrix for a batch of datapoints.

  Args:
    cost: Cost matrix for the matching of shape [B, N, N].
    rng: Random generator for sampling.
    epsilon: Level of entropic regularization wanted.
    init: Multiplier for epsilon decay at the first iteration.
    decay: How much to decay epsilon between two iterations.
    num_iters: Number of Sinkhorn iterations.
    num_permutations: Number of random permutations to sample for
      selecting the best.
    threshold: Convergence threshold for Sinkhorn algorithm.
    chg_momentum_from: Iteration from which to trigger the momemtum in Sinkhorn.

  Returns:
    An assignment of size [B, 2, N].
  """
  def coupling_fn(c):
    geom = geometry.Geometry(
        cost_matrix=c, epsilon=epsilon, init=init, decay=decay)
    return transport.solve(geom,
                           max_iterations=num_iters,
                           chg_momentum_from=chg_momentum_from,
                           threshold=threshold).matrix

  coupling = jax.vmap(coupling_fn)(cost)

  if rng is None:
    # Use fixed key to make sampling deterministic.
    rng = jax.random.PRNGKey(0)

  permutation = sample_best_permutation(rng, coupling, cost, num_permutations)
  permutation = jnp.array(permutation, dtype=jnp.int32)
  grid = np.stack(
      np.meshgrid(
          np.arange(cost.shape[1], dtype=np.int32),
          np.arange(cost.shape[2], dtype=np.int32),
          indexing='ij'),
      axis=0)
  indices = jnp.einsum('bnm,2nm->b2n', permutation, grid)

  return indices
