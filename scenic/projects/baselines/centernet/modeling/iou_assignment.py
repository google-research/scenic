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

"""Label assignment based on IoUs.

This implementation is provided by Yuxin Wu.
"""

import enum

import jax
import jax.numpy as jnp


class Assignment(enum.IntEnum):
  """Assignment result of each anchor after matching them with ground truth."""

  IGNORE = -1  # The anchor is ignored / excluded from training.
  NEGATIVE = 0  # The anchor is negative / does not match objects.
  POSITIVE = 1  # The anchor is positive / matches some object.


def label_assignment(
    iou_matrix: jnp.ndarray,
    thresholds: list[float],
    assignments: list[Assignment],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Match groundtruth boxes with detection boxes to assign objectness labels.

  The matched groundtruth (GT) of each detection box (DT) is the GT with maximum
  IoU. After a match is found, the label of this DT is determined by the
  threshold range that the IoU value falls into: if
  thresholds[i] <= IoU < thresholds[i+1], then the DT is given assignments[i+1].

  Args:
    iou_matrix: a (N, M) IoU matrix where N, M are the number of ground truth
      and detection (or anchor) boxes. Values are expected to be non-negative.
    thresholds: a sorted list of iou thresholds to determine which label to
      assign to each detection box.
    assignments: a list of assignments. Must have length == len(thresholds) + 1.

  Returns:
    A length-M integer vector, the matched GT index of every DT box.
    A length-M integer vector, the assigned label of every DT box.
  """
  if len(assignments) != len(thresholds) + 1:
    raise ValueError('Invalid length of assignments & thresholds!')
  if thresholds != sorted(thresholds):
    raise ValueError('Thresholds must be sorted!')
  thresholds = jnp.array(thresholds)
  matches = jnp.argmax(iou_matrix, axis=0)
  matched_max = jnp.max(iou_matrix, axis=0)  # Best IoU for each DT.
  # For each IoU, find its position inside "thresholds"
  if len(thresholds) == 1:
    assignments_per_box = jnp.where(matched_max < thresholds[0], assignments[0],
                                    assignments[1])
  elif len(thresholds) == 2:
    assignments_per_box = jnp.where(
        matched_max < thresholds[0], assignments[0],
        jnp.where(matched_max < thresholds[1], assignments[1], assignments[2]))
  else:
    # Handle the generic case, but much slower.
    indices = jnp.searchsorted(thresholds, matched_max, side='right')
    assignments_per_box = jnp.array(assignments, dtype=jnp.int32)[indices]

  return matches, assignments_per_box


def random_top_k(vec: jnp.ndarray, k: int,
                 prng_key: jnp.ndarray) -> jnp.ndarray:
  """Select top k elements from the given vector x, randomly breaking ties.

  Args:
    vec: input vector.
    k: number of elements to select
    prng_key: jax PRNG key

  Returns:
    A bool vector same size as `x` indicating the k elements that are selected.
  """
  if k < 0:
    raise ValueError(f'Cannot use k={k}!')
  if k == 0:
    return jnp.zeros_like(vec, dtype=bool)
  n = vec.size

  if vec.dtype == bool:
    vec = vec.astype(jnp.int32)
  perm_array = jnp.stack([vec, jnp.arange(n, dtype=vec.dtype)], axis=0)
  # Get permuted vector together with permutation index (avoid extra gather).
  perm_array = jax.random.permutation(prng_key, perm_array, axis=1)
  permuted_vec, permutation = jnp.split(perm_array, 2, axis=0)
  permutation = permutation[0].astype(jnp.int32)

  # Find the topk elements under permuted indices.
  _, topk_inds = jax.lax.top_k(permuted_vec[0], k)
  inds = permutation[topk_inds]  # Get back the original indices.
  return jnp.zeros_like(
      vec, dtype=bool).at[inds].set(
          True, mode='promise_in_bounds', unique_indices=True)


def subsample_assignments(assignments: jnp.ndarray, num_samples: int,
                          positive_fraction: float,
                          prng_key: jnp.ndarray) -> jnp.ndarray:
  """Randomly sample a subset from `assignments` for training.

  Args:
    assignments: an integer vector. Value must belong to `Assignment`.
    num_samples: number of samples to take.
    positive_fraction: the desired fraction of positive samples in the result.
      Will sample this amount of positive samples as long as there is enough.
      The rest of samples will be negative as long as there is enough.
    prng_key: jax PRNG key

  Returns:
    Result assignment vector with the same meaning as input. Items that are
    not sampled are assigned IGNORE.
  """
  pos_mask = assignments == Assignment.POSITIVE
  sampled_pos_mask = random_top_k(
      pos_mask, int(num_samples * positive_fraction), prng_key) & pos_mask
  # Ignore the positives that are not selected. sampled_assignments now have
  # desired number of positives, but possibly too many negatives.
  sampled_assignments = jnp.where(
      pos_mask & jnp.logical_not(sampled_pos_mask),
      Assignment.IGNORE, assignments)
  # Now pick top `num_samples` from it. This assumes order in Assignment enum.
  assert Assignment.IGNORE < Assignment.NEGATIVE < Assignment.POSITIVE
  final_mask = random_top_k(sampled_assignments, num_samples, prng_key)
  return jnp.where(final_mask, sampled_assignments, Assignment.IGNORE)
