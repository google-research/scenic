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

"""Common functions for computing matchings."""

import jax
import jax.experimental.host_callback as hcb
import jax.numpy as jnp
import numpy as np


def slicer(cost, n_present_col, matching_fn):
  """Slices cost matrices per-example and exploits padding to compute faster.

  Args:
    cost: np.ndarray[batch, n_row, n_col]; Cost matrices for which matchings
      must be computed. It is assumed that n_row >= n_col.
    n_present_col: np.ndarray[batch]; Number of trailing padded columns of the
      cost matrices.
    matching_fn: Matching function to call to compute matching on the unpadded
      cost matrices.

  Returns:
    Matchings of shape [batch, 2, n_col].
  """
  batch, n_row, n_col = cost.shape
  if n_row < n_col:
    raise ValueError(
        f'Slicer requires that n_row ({n_row}) >= n_col ({n_col}).')

  eye = np.eye(n_row, dtype=bool)
  matches = []
  for i in range(batch):
    present_col = max(n_present_col[i], 1)  # One col even if all are padded.
    cost_m = cost[i : i + 1, :, :present_col]  # Slicing should avoid a copy.
    indices = matching_fn(cost_m)[0]
    row, col = indices[0], indices[1]

    # Add padded matches (if padding was done correctly these can be random).
    unmatched_row = np.where(~eye[row].max(axis=0))[0]  # Faster than setdiff1d.
    unmatched_col = np.arange(present_col, n_col)

    # Assume n_row >= n_col >= n_present_col.
    n_common = n_col - present_col
    unmatched_row = unmatched_row[:n_common]

    # Reconstruct the matching.
    row = np.concatenate([row, unmatched_row], axis=0)
    col = np.concatenate([col, unmatched_col], axis=0)

    indices = np.stack([row, col], axis=0)
    matches.append(indices)
  return np.stack(matches)


def cpu_matcher(matching_fn):
  """Wraps matching function to be usable within jitted functions.

  Args:
    matching_fn: function; A matching function that aligns the predictions of
      the model with targets.

  Returns:
    Matching function with host callback that can be jitted.
  """
  # The callback function can only take a single argument.
  def maybe_slice_and_match(args):
    cost, ncol = args
    if ncol is None:
      return matching_fn(cost)
    else:
      return slicer(cost, ncol, matching_fn)

  @jax.custom_vjp
  def matching_fn_hcb(cost, n_cols=None):
    bs, n, m = cost.shape
    return hcb.call(
        maybe_slice_and_match, (cost, n_cols),
        result_shape=jax.ShapeDtypeStruct([bs, 2, min(n, m)], jnp.int32))

  # Define forward and backward passes.
  def matching_fn_hcb_vjp_fwd(cost, n_cols):
    return matching_fn_hcb(cost, n_cols), None

  def matching_fn_hcb_vjp_bwd(*_):
    return (None,)  # Return no gradient.

  matching_fn_hcb.defvjp(matching_fn_hcb_vjp_fwd, matching_fn_hcb_vjp_bwd)

  return matching_fn_hcb
