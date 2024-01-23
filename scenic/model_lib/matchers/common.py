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

"""Common functions for computing matchings."""

import jax
import jax.numpy as jnp
import numpy as np


def slicer(cost, n_present_col, matching_fn):
  """Maps matching_fn over examples after removing padding to speed up matching.

  Args:
    cost: Cost matrix or batch of cost matrices with any number of batch
      dimensions. Requires n_row >= n_col.
    n_present_col: Number of non-padding columns of the cost matrices, or None
      if padding should not be removed.
    matching_fn: A matching function that operates on a single cost matrix.

  Returns:
    Matchings of shape [batch, 2, n_col].

  Raises:
    ValueError if n_row < n_col and n_present_col is not None.
  """
  batch_shape = cost.shape[:-2]
  cost = cost.reshape(-1, *cost.shape[-2:])

  if n_present_col is None:
    matches = np.stack([matching_fn(c) for c in cost])
    return matches.reshape(*batch_shape, *matches.shape[1:])

  n_present_col = n_present_col.reshape(-1)
  assert cost.shape[:1] == n_present_col.shape, (
      cost.shape,
      n_present_col.shape,
  )

  batch, n_row, n_col = cost.shape
  if n_row < n_col:
    raise ValueError(
        f'Slicer requires that n_row ({n_row}) >= n_col ({n_col}).')

  eye = np.eye(n_row, dtype=bool)
  matches = []
  for i in range(batch):
    present_col = max(n_present_col[i], 1)  # One col even if all are padded.
    cost_m = cost[i, :, :present_col]  # Slicing should avoid a copy.

    row, col = matching_fn(cost_m)

    # Add padded matches (if padding was done correctly these can be random).
    unmatched_row = np.where(~eye[row].max(axis=0))[0]  # Faster than setdiff1d.
    unmatched_row = unmatched_row.astype(np.int32)
    unmatched_col = np.arange(present_col, n_col, dtype=np.int32)

    # Assume n_row >= n_col >= n_present_col.
    n_common = n_col - present_col
    unmatched_row = unmatched_row[:n_common]

    # Reconstruct the matching.
    row = np.concatenate([row, unmatched_row], axis=0)
    col = np.concatenate([col, unmatched_col], axis=0)

    matches.append(np.stack([row, col], axis=0))

  matches = np.stack(matches)

  return matches.reshape(*batch_shape, *matches.shape[1:])


def cpu_matcher(matching_fn):
  """Wraps matching function to be usable within jitted functions.

  Args:
    matching_fn: function; A matching function that aligns the predictions of
      the model with targets.

  Returns:
    Matching function with host callback that can be jitted.
  """
  # The callback function can only take a single argument.
  def slice_and_match(args):
    cost, ncol = args
    return slicer(cost, ncol, matching_fn)

  @jax.custom_vjp
  def matching_fn_hcb(cost, n_cols=None):
    *b, n, m = cost.shape
    return jax.pure_callback(
        slice_and_match,
        jax.ShapeDtypeStruct(b + [2, min(n, m)], jnp.int32),
        (cost, n_cols),
        vectorized=True)

  # Define forward and backward passes.
  def matching_fn_hcb_vjp_fwd(cost, n_cols):
    return matching_fn_hcb(cost, n_cols), None

  def matching_fn_hcb_vjp_bwd(*_):
    return (None,)  # Return no gradient.

  matching_fn_hcb.defvjp(matching_fn_hcb_vjp_fwd, matching_fn_hcb_vjp_bwd)

  # Note: When called from TPU, errors in the callback will NOT cause the code
  # to fail, but will produce nonsensical matching outputs. Test the callback
  # carefully with jax.jit(callback, backend='cpu') first.
  return matching_fn_hcb
