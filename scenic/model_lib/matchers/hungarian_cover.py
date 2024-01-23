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

"""TPU-friendly Hungarian matching algorithm.

JAX implementation the Linear Sum Assignment problem solver. This implementation
builds off of the Hungarian Matching Algorithm
(https://www.cse.ust.hk/~golin/COMP572/Notes/Matching.pdf).

Based on the original implementation by Jiquan Ngiam <jngiam@google.com>.
"""


from typing import Tuple, Optional, Dict

import jax
import jax.numpy as jnp


def _prepare(weights: jnp.ndarray) -> jnp.ndarray:
  """Prepare the cost matrix.

  To speed up computational efficiency of the algorithm, all weights are shifted
  to be non-negative. Each element is reduced by the row / column minimum. Note
  that neither operation will effect the resulting solution but will provide a
  better starting point for the greedy assignment. Note this corresponds to the
  pre-processing and step 1 of the Hungarian algorithm from Wikipedia.

  Args:
    weights: A float32 [b, n, m] array, where each inner matrix represents
    weights to be use for matching.

  Returns:
    A prepared weights array of the same shape and dtype.
  """
  # Since every worker needs a job and every job needs a worker, we can subtract
  # the minimum from each.
  assert weights.ndim == 3

  weights = weights - jnp.min(weights, axis=2, keepdims=True)
  weights = weights - jnp.min(weights, axis=1, keepdims=True)
  return weights


def _greedy_assignment(adj_matrix: jnp.ndarray) -> jnp.ndarray:
  """Greedily assigns workers to jobs based on an adjaceny matrix.

  Starting with an adjacency matrix representing the available connections in
  the bi-partite graph, this function greedily chooses elements such that each
  worker is matched to at most one job (or each job is assigned to at most one
  worker). Note, if the adjacency matrix has no available values for a
  particular row/column, the corresponding job/worker may go unassigned.

  Args:
    adj_matrix: A bool [b, n, m] array, where each element of the inner matrix
      represents whether the worker (row) can be matched to the job (column).

  Returns:
    A bool [b, n, m] array, where each element of the inner matrix represents
    whether the worker has been matched to the job. Each row and column can have
    at most one true element. Some of the rows and columns may not be matched.
  """
  b, _, m = adj_matrix.shape

  # To (n, b, m)
  adj_matrix = jnp.transpose(adj_matrix, axes=(1, 0, 2))

  # Iteratively assign each row using jax.lax.scan. Intuitively, this is a loop
  # over rows, where we incrementally assign each row.
  def _assign_row(col_assigned, row_adj):
    # Viable candidates cannot already be assigned to another job.
    candidates = jnp.logical_and(row_adj, jnp.logical_not(col_assigned))

    # Deterministically assign to the candidates of the highest index count.
    max_candidate_idx = jnp.argmax(candidates, axis=1)
    candidates_indicator = jax.nn.one_hot(max_candidate_idx, m, dtype=jnp.bool_)
    candidates_indicator = jnp.logical_and(candidates_indicator, candidates)

    # Make assignment to the column.
    col_assigned = jnp.logical_or(candidates_indicator, col_assigned)

    return col_assigned, candidates_indicator

  # Store the elements assigned to each column to update each iteration.
  col_assigned = jnp.zeros((b, m), dtype=jnp.bool_)
  _, assignment = jax.lax.scan(
      _assign_row, col_assigned, adj_matrix)

  # To (b, n, m)
  assignment = jnp.transpose(assignment, axes=(1, 0, 2))
  return assignment


def _find_augmenting_path(assignment: jnp.ndarray,
                          adj_matrix: jnp.ndarray) -> Dict[str, jnp.ndarray]:
  """Finds an augmenting path given an assignment and an adjacency matrix.

  The augmenting path search starts from the unassigned workers, then goes on
  to find jobs (via an unassigned pairing), then back again to workers (via an
  existing pairing), and so on. The path alternates between unassigned and
  existing pairings. Returns the state after the search.

  Note: In the state the worker and job, indices are 1-indexed so that we can
  use 0 to represent unreachable nodes. State contains the following keys:

  - jobs: A [b, 1, m] array containing the highest index unassigned worker that
    can reach this job through a path.
  - jobs_from_worker: A [b, n] array containing the worker reached immediately
    before this job.
  - workers: A [b, n, 1] array containing the highest index unassigned worker
    that can reach this worker through a path.
  - workers_from_job: A [b, m] array containing the job reached immediately
    before this worker.
  - new_jobs: A bool [b, m] array containing True if the unassigned job can be
    reached via a path.

  State can be used to recover the path via backtracking.

  Args:
    assignment: A bool [b, n, m] array, where each element of the inner matrix
      represents whether the worker has been matched to the job. This may be a
      partial assignment.
    adj_matrix: A bool [b, n, m] array, where each element of the inner matrix
      represents whether the worker (row) can be matched to the job (column).

  Returns:
    A state dictionary, which represents the outcome of running an augmenting
    path search on the graph given the assignment.
  """
  b, n, m = assignment.shape
  unassigned_workers = jnp.logical_not(
      jnp.any(assignment, axis=2, keepdims=True))
  unassigned_jobs = jnp.logical_not(jnp.any(assignment, axis=1, keepdims=True))

  unassigned_pairings = jnp.logical_and(
      adj_matrix, jnp.logical_not(assignment)).astype(jnp.int32)
  existing_pairings = assignment.astype(jnp.int32)

  # Initialize unassigned workers to have non-zero ids, assigned workers will
  # have ids = 0.
  worker_indices = jnp.arange(1, n + 1, dtype=jnp.int32)
  init_workers = jnp.tile(
      worker_indices[jnp.newaxis, :, jnp.newaxis], (b, 1, 1))
  init_workers = init_workers * unassigned_workers.astype(jnp.int32)

  state = {'jobs': jnp.zeros((b, 1, m), dtype=jnp.int32),
           'jobs_from_worker': jnp.zeros((b, m), dtype=jnp.int32),
           'workers': init_workers,
           'workers_from_job': jnp.zeros((b, n), dtype=jnp.int32),}

  def _has_active_workers(arg):
    """Check if there are still active workers."""
    _, cur_workers = arg
    return jnp.sum(cur_workers) > 0

  def _augment_step(arg):
    """Performs one search step."""
    state, cur_workers = arg

    # Find potential jobs using current workers.
    potential_jobs = cur_workers * unassigned_pairings
    curr_jobs = jnp.max(potential_jobs, axis=1, keepdims=True)
    curr_jobs_from_worker = 1 + jnp.argmax(potential_jobs, axis=1)

    # Remove already accessible jobs from curr_jobs.
    default_jobs = jnp.zeros_like(state['jobs'])
    curr_jobs = jnp.where(state['jobs'] > 0, default_jobs, curr_jobs)
    curr_jobs_from_worker = (curr_jobs_from_worker *
                             (curr_jobs > 0).astype(jnp.int32)[:, 0, :])

    # Find potential workers from current jobs.
    potential_workers = curr_jobs * existing_pairings
    cur_workers = jnp.max(potential_workers, axis=2, keepdims=True)
    cur_workers_from_job = 1 + jnp.argmax(potential_workers, axis=2)

    # Remove already accessible workers from cur_workers.
    default_workers = jnp.zeros_like(state['workers'])
    cur_workers = jnp.where(
        state['workers'] > 0, default_workers, cur_workers)
    cur_workers_from_job = (cur_workers_from_job *
                            (cur_workers > 0).astype(jnp.int32)[:, :, 0])

    # Update state so that we can backtrack later.
    state['jobs'] = jnp.maximum(state['jobs'], curr_jobs)
    state['jobs_from_worker'] = jnp.maximum(
        state['jobs_from_worker'], curr_jobs_from_worker)
    state['workers'] = jnp.maximum(state['workers'], cur_workers)
    state['workers_from_job'] = jnp.maximum(
        state['workers_from_job'], cur_workers_from_job)

    return state, cur_workers

  state, _ = jax.lax.while_loop(
      _has_active_workers, _augment_step, (state, init_workers))

  # Compute new jobs, this is useful for determnining termnination of the
  # maximum bi-partite matching and initialization for backtracking.
  new_jobs = jnp.logical_and(state['jobs'] > 0, unassigned_jobs)
  state['new_jobs'] = new_jobs[:, 0, :]
  return state


def _improve_assignment(assignment: jnp.ndarray,
                        state: Dict[str, jnp.ndarray]) -> jnp.ndarray:
  """Improves an assignment by backtracking the augmented path using state.

  Args:
    assignment: A bool [b, n, m] array, where each element of the inner matrix
      represents whether the worker has been matched to the job. This may be a
      partial assignment.
    state: Represents the outcome of running an augmenting path search on the
    graph given the assignment.

  Returns:
    A new assignment array of the same shape and type as assignment, where the
    assignment has been updated using the augmented path found.
  """
  b, n, m = assignment.shape

  # We store the current job id and iteratively backtrack using jobs_from_worker
  # and workers_from_job until we reach an unassigned worker. We flip all the
  # assignments on this path to discover a better overall assignment.

  # Note: The indices in state are 1-indexed, where 0 represents that the
  # worker / job cannot be reached.

  # Obtain initial job indices based on new_jobs.
  curr_job_idx = jnp.argmax(state['new_jobs'], axis=1)

  # Track whether an example is actively being backtracked. Since we are
  # operating on a batch, not all examples in the batch may be active.
  simple_gather = jax.vmap(lambda x, idx: x[idx], in_axes=[0, 0], out_axes=0)
  active = simple_gather(state['new_jobs'], curr_job_idx)
  batch_range = jnp.arange(0, b, dtype=jnp.int32)

  # Flip matrix tracks which assignments we need to flip - corresponding to the
  # augmenting path taken. We use an integer array here so that we can use
  # array_scatter_nd_add to update the array, and then cast it back to bool
  # after the loop.
  flip_matrix = jnp.zeros((b, n, m), dtype=jnp.int32)

  def _has_active_backtracks(arg):
    """Check if there are still active workers."""
    _, active, _ = arg
    return jnp.any(active)

  dimension_numbers = jax.lax.ScatterDimensionNumbers(
      update_window_dims=(),
      inserted_window_dims=(0, 1, 2),
      scatter_dims_to_operand_dims=(0, 1, 2))

  def _backtrack_one_step(arg):
    """Take one step in backtracking."""
    flip_matrix, active, curr_job_idx = arg
    # Discover the worker that the job originated from, note that this worker
    # must exist by construction.
    curr_worker_idx = simple_gather(state['jobs_from_worker'], curr_job_idx) - 1
    curr_worker_idx = jnp.maximum(curr_worker_idx, 0)
    update_indices = jnp.stack([batch_range, curr_worker_idx, curr_job_idx],
                               axis=1)
    update_indices = jnp.maximum(update_indices, 0)
    flip_matrix = jax.lax.scatter_add(
        flip_matrix,
        update_indices,
        active.astype(jnp.int32),
        dimension_numbers,
        unique_indices=True)

    # Discover the (potential) job that the worker originated from.
    curr_job_idx = simple_gather(state['workers_from_job'], curr_worker_idx) - 1

    # Note that jobs may not be active, and we track that here (before
    # adjusting indices so that they are all >= 0 for gather).
    active = jnp.logical_and(active, curr_job_idx >= 0)
    curr_job_idx = jnp.maximum(curr_job_idx, 0)
    update_indices = jnp.stack([batch_range, curr_worker_idx, curr_job_idx],
                               axis=1)
    update_indices = jnp.maximum(update_indices, 0)
    flip_matrix = jax.lax.scatter_add(
        flip_matrix,
        update_indices,
        active.astype(jnp.int32),
        dimension_numbers,
        unique_indices=True)

    return flip_matrix, active, curr_job_idx

  flip_matrix, _, _ = jax.lax.while_loop(
      _has_active_backtracks,
      _backtrack_one_step,
      (flip_matrix, active, curr_job_idx))

  assignment = jnp.logical_xor(assignment, flip_matrix > 0)
  return assignment


def _maximum_bipartite_matching(
    adj_matrix: jnp.ndarray, assignment: Optional[jnp.ndarray] = None
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
  """Performs maximum bipartite matching using augmented paths.

  Args:
    adj_matrix: A bool [b, n, m] array, where each element of the inner matrix
      represents whether the worker (row) can be matched to the job (column).
    assignment: An optional bool [b, n, m] array, where each element of the
      inner matrix represents whether the worker has been matched to the job.
      This may be a partial assignment. If specified, this assignment will be
      used to seed the iterative algorithm.

  Returns:
    A state dict representing the final augmenting path state search, and
    a maximum bipartite matching assignment array. Note that the state outcome
    can be used to compute a minimum vertex cover for the bipartite graph.
  """
  if assignment is None:
    assignment = _greedy_assignment(adj_matrix)
  state = _find_augmenting_path(assignment, adj_matrix)

  def _has_new_jobs(arg):
    state, _ = arg
    return jnp.any(state['new_jobs'])

  def _improve_assignment_and_find_new_path(arg):
    state, assignment = arg
    assignment = _improve_assignment(assignment, state)
    state = _find_augmenting_path(assignment, adj_matrix)
    return state, assignment

  state, assignment = jax.lax.while_loop(
      _has_new_jobs,
      _improve_assignment_and_find_new_path,
      (state, assignment))

  return state, assignment


def _compute_cover(
    state: Dict[str, jnp.ndarray], assignment: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes a cover for the bipartite graph.

  We compute a cover using the construction provided at
  https://en.wikipedia.org/wiki/K%C5%91nig%27s_theorem_(graph_theory)#Proof
  which uses the outcome from the alternating path search.

  Args:
    state: A state dict, which represents the outcome of running an augmenting
      path search on the graph given the assignment.
    assignment: An optional bool [b, n, m] array, where each element of the
      inner matrix represents whether the worker has been matched to the job.
      This may be a partial assignment. If specified, this assignment will be
      used to seed the iterative algorithm.

  Returns:
    Row and column covers for the bipartite graph: workers_cover is a boolean
    array of shape [b, n, 1] and jobs_cover is a array array of shape
    [b, 1, m].
  """
  assigned_workers = jnp.any(assignment, axis=2, keepdims=True)
  assigned_jobs = jnp.any(assignment, axis=1, keepdims=True)

  workers_cover = jnp.logical_and(assigned_workers, state['workers'] <= 0)
  jobs_cover = jnp.logical_and(assigned_jobs, state['jobs'] > 0)

  return workers_cover, jobs_cover


def _update_weights_using_cover(workers_cover: jnp.ndarray,
                                jobs_cover: jnp.ndarray,
                                weights: jnp.ndarray) -> jnp.ndarray:
  """Updates weights for hungarian matching using a cover.

  We first find the minimum uncovered weight. Then, we subtract this from all
  the uncovered weights, and add it to all the doubly covered weights.

  Args:
    workers_cover: A boolean array of shape [b, n, 1].
    jobs_cover: A boolean array of shape [b, 1, n].
    weights: A float32 [b, n, n] array, where each inner matrix represents
      weights to be use for matching.

  Returns:
    A new weight matrix with elements adjusted by the cover.
  """
  max_value = jnp.max(weights)

  covered = jnp.logical_or(workers_cover, jobs_cover)
  double_covered = jnp.logical_and(workers_cover, jobs_cover)

  uncovered_weights = jnp.where(
      covered, jnp.full_like(weights, max_value), weights)
  min_weight = jnp.min(uncovered_weights, axis=(-2, -1), keepdims=True)

  add_weight = jnp.where(double_covered,
                         jnp.full_like(weights, min_weight),
                         jnp.zeros_like(weights))
  sub_weight = jnp.where(covered, jnp.zeros_like(weights),
                         jnp.full_like(weights, min_weight))

  return weights + add_weight - sub_weight


def hungarian_cover_matcher(weights: jnp.ndarray,
                            eps: float = 1e-8) -> jnp.ndarray:
  """Computes the minimum linear sum assignment using the Hungarian algorithm.

  Args:
    weights: A float32 [b, n, m] array, where each inner matrix represents
    weights to be use for matching.
    eps: Small number to test for equality to 0.

  Returns:
    Jobs and workers matching indices as [b, 2, m] .
  """
  b, n, m = weights.shape
  should_transpose = n > m
  if should_transpose:
    weights = jnp.transpose(weights, axes=(0, 2, 1))
    n, m = m, n

  # TODO(agritsenko): Figure out a more efficient way of correctly handling
  # rectangular cost matrices.
  if n != m:  # So n < m based on the code block above.
    pad_n = 1  # `m - n` is guaranteed to be correct, but 1 also works.
    pad_values = jnp.max(weights, axis=(1, 2), keepdims=True) * 1.1
    pad_values = jnp.broadcast_to(pad_values, (b, pad_n, m))
    weights = jnp.concatenate((weights, pad_values), axis=1)
    n += pad_n
  else:
    pad_n = 0

  weights = _prepare(weights)
  adj_matrix = jnp.abs(weights) < eps
  state, assignment = _maximum_bipartite_matching(adj_matrix)
  workers_cover, jobs_cover = _compute_cover(state, assignment)

  def _cover_incomplete(arg):
    workers_cover, jobs_cover, _, _ = arg
    cover_sum = (jnp.sum(workers_cover, dtype=jnp.int32) +
                 jnp.sum(jobs_cover, dtype=jnp.int32))
    return cover_sum < b * n

  def _update_weights_and_match(arg):
    workers_cover, jobs_cover, weights, assignment = arg
    weights = _update_weights_using_cover(workers_cover, jobs_cover, weights)
    adj_matrix = jnp.abs(weights) < eps
    state, assignment = _maximum_bipartite_matching(adj_matrix, assignment)
    workers_cover, jobs_cover = _compute_cover(state, assignment)
    return workers_cover, jobs_cover, weights, assignment

  workers_cover, jobs_cover, weights, assignment = jax.lax.while_loop(
      _cover_incomplete,
      _update_weights_and_match,
      (workers_cover, jobs_cover, weights, assignment))

  workers_ind = jnp.broadcast_to(jnp.arange(n), (b, n))
  jobs_ind = jnp.argmax(assignment, axis=2)

  # Remove padded indices.
  workers_ind = workers_ind[:, :n - pad_n]
  jobs_ind = jobs_ind[:, :n - pad_n]

  if not should_transpose:
    ind = jnp.stack([workers_ind, jobs_ind], axis=1)
  else:
    ind = jnp.stack([jobs_ind, workers_ind], axis=1)
  return ind


hungarian_cover_tpu_matcher = jax.jit(hungarian_cover_matcher)
