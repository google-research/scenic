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

"""Linearly interpolates the values of an array with a set of points."""

import functools
import jax
import jax.numpy as jnp


@functools.partial(jax.jit)
def linearly_interpolate(arr, points):
  """Linearly interpolates the values of an array with a set of points.

  Args:
    arr: An array of shape [batch_size, H, W, C]
    points: An array of shape [batch_size, H, W, num_points, 2]

  Returns:
    An array of shape [batch_size, H, W, num_points, C]
  """

  # Flatten the batch and grid dimensions of arr
  batch_size, height, width, channels = arr.shape
  arr_flat = arr.reshape(batch_size, height * width, channels)

  # Flatten the corresponding dimensions in points
  points_flat = points.reshape(batch_size, height * width, -1, 2)

  # Ensure points are within the bounds
  points_flat = points_flat.clip(0, jnp.array([[height - 1, width - 1]]))

  # Splitting points into h and w components
  h, w = points_flat[..., 0], points_flat[..., 1]

  # Identify neighboring points
  h_floor = jnp.floor(h).astype(jnp.int32)
  w_floor = jnp.floor(w).astype(jnp.int32)
  h_ceil = jnp.ceil(h).astype(jnp.int32).clip(0, height - 1)
  w_ceil = jnp.ceil(w).astype(jnp.int32).clip(0, width - 1)

  # Compute 1D indices for the flattened array
  idx_floor_floor = h_floor * width + w_floor
  idx_floor_ceil = h_floor * width + w_ceil
  idx_ceil_floor = h_ceil * width + w_floor
  idx_ceil_ceil = h_ceil * width + w_ceil

  # Compute the interpolation weights
  lh = height - h_floor
  lw = width - w_floor
  hh = 1 - lh
  hw = 1 - lw

  # Gather the values from the four corners for each point
  top_left = arr_flat[..., idx_floor_floor, :]
  top_right = arr_flat[..., idx_floor_ceil, :]
  bottom_left = arr_flat[..., idx_ceil_floor, :]
  bottom_right = arr_flat[..., idx_ceil_ceil, :]

  # Perform interpolation
  interpolated_values = (top_left * hh[..., None] * hw[..., None] +
                         top_right * hh[..., None] * lw[..., None] +
                         bottom_left * lh[..., None] * hw[..., None] +
                         bottom_right * lh[..., None] * lw[..., None])

  # Reshape back to original grid and batch dimensions
  return interpolated_values.reshape(batch_size, height, width, -1, channels)
