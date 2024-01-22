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

"""Common neural network funcitonality that doesn't require parameters."""

from typing import Callable, Sequence
import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


def extract_image_patches(lhs,
                          rhs_shape,
                          strides,
                          padding,
                          rhs_dilation,
                          data_format='NHWC'):
  """Extract patches of size `rhs_shape` from `lhs`.

  Args:
    lhs: A 4-D Tensor;  With shape `[batch, in_rows, in_cols, depth].
    rhs_shape: tuple; Size of the sliding window for each dimension of `lhs`.
    strides: tuple; How far the centers of two consecutive patches are in the
      lhs. Must be: `[1, stride_rows, stride_cols, 1]`.
    padding: str; The type of padding algorithm to use.
      We specify the size-related attributes as: ```python ksizes = [1,
        ksize_rows, ksize_cols, 1] strides = [1, strides_rows, strides_cols, 1]
        rates = [1, rates_rows, rates_cols, 1]```
    rhs_dilation: A 1-D Tensor of length 4; Must be: `[1, rate_rows, rate_cols,
      1]`. This is the input stride, specifying how far two consecutive patch
      samples are in the input. Equivalent to extracting patches with
      `patch_sizes_eff = patch_sizes + (patch_sizes - 1) * (rates - 1)`,
      followed by subsampling them spatially by a factor of `rates`. This is
      equivalent to `rate` in dilated (a.k.a. Atrous) convolutions.
    data_format: str; The format of the `lhs`. Must be either `'NHWC'` or
      `'NCHW'`.

  Returns:
    A 4-D Tensor. Has the same type and data format as `lhs`, and with shape
    `[batch, num_patches_col, num_patches_row, rhs_shape[1], rhs_shape[2], C]`.
  """
  num_dims = lhs.ndim
  num_spatial_dims = num_dims - 2

  batch_dim = data_format.index('N')
  feature_dim = data_format.index('C')
  depth = lhs.shape[feature_dim]

  if rhs_shape[batch_dim] != 1 or rhs_shape[feature_dim] != 1:
    raise NotImplementedError(
        'Current implementation does not yet support window sizes > 1 in '
        'the batch and depth dimensions.')

  if strides[batch_dim] != 1 or strides[feature_dim] != 1:
    raise NotImplementedError(
        'Current implementation does not support strides in the batch '
        'and depth dimensions.')

  if rhs_dilation[batch_dim] != 1 or rhs_dilation[feature_dim] != 1:
    raise NotImplementedError(
        'Current implementation does not support dilations in the batch '
        'and depth dimensions.')

  # Replicating tensorflow's implementation.
  lhs_perm = lax.conv_general_permutations(
      (data_format, 'HWIO', data_format))[0]
  kernel_shape = [rhs_shape[i] for i in lhs_perm[2:]]

  kernel_size = np.prod(kernel_shape)
  conv_filter_shape = kernel_shape[:]
  conv_filter_shape.append(1)
  conv_filter_shape.append(kernel_size * depth)

  iota_kernel_shape = (kernel_size, depth, kernel_size)

  conv_filter = lax.eq(
      lax.broadcasted_iota(jnp.int32, iota_kernel_shape, 0),
      lax.broadcasted_iota(jnp.int32, iota_kernel_shape, 2),
  )
  conv_filter = lax.convert_element_type(conv_filter, lhs.dtype)
  conv_filter = lax.reshape(conv_filter, conv_filter_shape)

  dim_num = lax.conv_dimension_numbers(lhs.shape, conv_filter.shape,
                                       (data_format, 'HWIO', data_format))
  conv_strides = [0] * num_spatial_dims
  conv_rhs_dilation = [0] * num_spatial_dims
  for i in range(num_spatial_dims):
    dim = dim_num.lhs_spec[i + 2]
    conv_strides[i] = strides[dim]
    conv_rhs_dilation[i] = rhs_dilation[dim]

  conv = lax.conv_general_dilated(lhs, conv_filter, conv_strides, padding, None,
                                  conv_rhs_dilation, dim_num, depth)

  conv_dims = list(conv.shape[:-1])
  conv_dims.append(depth)
  conv_dims.extend(kernel_shape)
  conv = lax.reshape(conv, conv_dims)

  permutation = list(range(len(conv_dims)))
  depth_dim = permutation.pop(-3)
  permutation.append(depth_dim)

  return lax.transpose(conv, permutation)


def extract_patches(lhs, rhs_shape, strides=(1, 1)):
  """Extracts patches from an image using a convolution operator.

  Args:
    lhs: A tensor of images of shapes (B, H, W, C).
    rhs_shape: The size of the patches to extract (h, w).
    strides: The shift between extracted patches (s1, s2)

  Returns:
    All the patches in a tensor of dimension
      (B, (H - h + 1) // s1, (W - w + 1) // s2, h, w, C).
  """
  # [batch, channels, height, width]
  lhs = jnp.moveaxis(lhs, -1, 1)
  d = lhs.shape[1]
  h, w = rhs_shape

  # Construct the lookup conv weights.
  dim_out = jnp.arange(d * h * w).reshape((-1, 1, 1, 1))
  dim_in = jnp.arange(d).reshape((1, -1, 1, 1))
  i = jnp.arange(h).reshape((1, 1, -1, 1))
  j = jnp.arange(w).reshape((1, 1, 1, -1))
  weights = ((w * i + j) * d + dim_in == dim_out).astype(jnp.float32)

  # [batch, h * w * d, (H - h + 1) // s1, (W - w + 1) // s2]
  concatenated_patches = lax.conv(
      lhs, weights, window_strides=strides, padding='VALID')

  # [batch, (H - h + 1) // s1, (W - w + 1) // s2, h * w * d]
  concatenated_patches = jnp.moveaxis(concatenated_patches, 1, -1)

  # [batch, (H - h + 1) // s1, (W - w + 1) // s2, h, w, d]
  shape = concatenated_patches.shape[:3] + (h, w, d)
  return concatenated_patches.reshape(shape)


def compute_relative_positions(query_spatial_shape,
                               key_spatial_shape,
                               spatial_axis=None):
  """Generate relative positions of queries and keys.


  For relative attention, the pairwise positional distance between each query
  and key point is used in the attention weight computation. This function
  generates the positional distances between each query-key pair, given the
  offset of first position in the query with respect to first position in the
  key.

  For example, if the query and key are 1d and query has 2 entries and the key
  has 3 entries, the relative distance matrix is:
    [[0, 1, 2],
     [-1, 0, 1]]
  where each [i, j] entry = j - i (j = key index, i = query index). Note that
  the values in this matrix are being used by an embedding lookup, so we shift
  them such that the smallest index is zero:
    [[1, 2, 3],
     [0, 1, 2]]

  This function produces the multi-dimensional distance for a query and key.
  It factorizes the distance computation such that there is a positional
  distance per dimension. An input with 3 dimensions will have a total of
  3 distances, 1 per dimension.

  Args:
    query_spatial_shape: tuple; Indicating the spatial shape of the query.
    key_spatial_shape: tuple; Indicating the spatial shape of the key.
    spatial_axis: tuple; The axis over which the distance is calculated. Default
      is None, which means distances over all axis is calculated.

  Returns:
    a numpy (np) int array of shape [len(spatial_axis),
      query_spatial_shape(spatial_axis), key_spatial_shape(spatial_axis)]
      holding the distance between each query  and key pair across dimensions
      that are determined by `spatial_axis`,  where the query and key are
      indexed by their position. The smallest value in the array is zero.
  """
  assert len(query_spatial_shape) == len(key_spatial_shape)
  if spatial_axis is None:
    spatial_axis = range(len(query_spatial_shape))
  for sa in spatial_axis:
    if not 0 <= sa < len(query_spatial_shape):
      raise ValueError('Element of `spatial_axis` should be between 0 and '
                       'length of `query_spatial_shape`.')

  num_dims = len(spatial_axis)
  # Keep only dimensions we are iterested in.
  query_spatial_shape = tuple([query_spatial_shape[a] for a in spatial_axis])
  key_spatial_shape = tuple([key_spatial_shape[a] for a in spatial_axis])

  total_queries = np.prod(query_spatial_shape)

  total_keys = np.prod(key_spatial_shape)
  # A distance per dimension in the flattened query-key arrays.

  relative_positions = np.empty((num_dims, total_queries, total_keys),
                                dtype=np.int32)

  # Convert flattened indices to multi-dimension coordinate indices.
  coordinates_query = np.unravel_index(
      range(total_queries), query_spatial_shape)
  coordinates_key = np.unravel_index(range(total_keys), key_spatial_shape)

  # Compute distances between each query-key point.
  for dim in range(num_dims):
    for flat_index_query in range(total_queries):
      for flat_index_key in range(total_keys):
        relative_positions[dim, flat_index_query, flat_index_key] = (
            coordinates_key[dim][flat_index_key] -
            coordinates_query[dim][flat_index_query])
    relative_positions[dim] = relative_positions[dim]

  # These indices are being used by an embedding lookup, so shift the indices
  # such that the smallest index is zero.
  relative_positions -= np.amin(relative_positions, axis=(1, 2), keepdims=True)
  # Reshape to original dim.
  relative_positions = relative_positions.reshape((num_dims,) +
                                                  query_spatial_shape +
                                                  key_spatial_shape)
  return relative_positions


def patch_image(inputs,
                inputs_shape,
                patch_size,
                strides=None,
                padding='VALID',
                mode='i2p'):
  """Applies patching operation on the input.

  Args:
    inputs: Input data.
    inputs_shape: tuple; Shape of the input data.
    patch_size: tuple; size of the patch: (height, width).
    strides: tuple; Specifies how far two consecutive patches are in the
      input.
    padding: str; The type of padding algorithm to use.
    mode: str; Either 'i2p' to convert the input image to patches or 'p2i' to
      convert the patched image to the original shape.

  Returns:
    Patched image if mode='i2p', original image if mode='p2i'.
  """
  strides = strides or patch_size

  def i2p(x):
    return extract_image_patches(
        lhs=x.astype(jnp.float64),
        rhs_shape=(1,) + patch_size + (1,),
        strides=(1,) + strides + (1,),
        padding=padding,
        rhs_dilation=(1,) * inputs.ndim,
        data_format='NHWC')

  if mode == 'i2p':
    _, inputs_w, inputs_h, _ = inputs.shape
    patch_w, patch_h = patch_size
    if (inputs_w < patch_w or inputs_h < patch_h):
      raise ValueError(f'Patch height and width ({patch_w} and  {patch_h}) '
                       'should be smaller thatn inputs height and width'
                       f' ({inputs_w} and  {inputs_h}).')
    outputs = i2p(inputs)

  elif mode == 'p2i':
    _, fn_vjp = jax.vjp(i2p, jnp.ones(inputs_shape))
    overlap_count = fn_vjp(jnp.ones_like(inputs))[0]
    outputs = fn_vjp(inputs)[0] / overlap_count

  else:
    raise ValueError()
  return outputs


def space_to_depth(inputs, window_shape, strides=None, padding='VALID'):
  """Applies space to depth.

  Args:
    inputs: Input data with dimensions `[bs, window dims, ..., features]`.
    window_shape: tuple; Defining the window to reduce over.
    strides: tuple, A sequence of `n` integers, representing the inter-window
        strides (default: window_shape).
    padding: str; Either `'SAME'`, `'VALID'`, or a sequence of `n` `(low,
      high)` integer pairs that give the padding to
      apply before and after each spatial dimension (default: `'VALID'`).

  Returns:
    An output image with less or equal spacial dimensions as inputs.

  """
  strides = strides or window_shape
  patched = extract_image_patches(
      lhs=inputs.astype(jnp.float64),
      rhs_shape=(1,) + window_shape + (1,),
      strides=(1,) + strides + (1,),
      padding=padding,
      rhs_dilation=(1,) * inputs.ndim,
      data_format='NHWC')

  bs, n_patch_h, n_patch_w, _, _, _ = patched.shape
  return patched.reshape(bs, n_patch_h, n_patch_w, -1)


def pooling(inputs,
            window_shape,
            pooling_configs=None,
            strides=None,
            padding='VALID'):
  """Applies configurable pooling.

  Args:
    inputs: an nd-array; Thego shape of inputs is `[bs, <window dims>,
      features]` and for presence_weights, the shape is `[bs, <window dims>]`.
    window_shape: tuple; Defining the window to reduce over.
    pooling_configs: dict; Configuration for the optional pooling operation.
    strides: tuple, A sequence of `n` integers, representing the inter-window
        strides (default: window_shape).
    padding: str; Either `'SAME'`, `'VALID'`, or a sequence of `n` `(low, high)`
      integer pairs that give the padding to
      apply before and after each spatial dimension (default: `'VALID'`).

  Returns:
    An output image with less or equal spacial dimensions as inputs.
  """
  # TODO(dehghani): add positional embedding to other type of pooling?
  strides = strides or window_shape

  pooling_type = pooling_configs.get('pooling_type')
  if pooling_type == 'avg_pooling':
    x = nn.avg_pool(inputs, window_shape, strides=strides, padding=padding)

  elif pooling_type == 'max_pooling':
    x = nn.max_pool(inputs, window_shape, strides=strides, padding=padding)

  elif pooling_type == 'space_to_depth':
    x = space_to_depth(inputs, window_shape, strides=strides, padding=padding)

  else:
    raise ValueError('Pooling type {} is not defined.'.format(pooling_type))
  return x


def weighted_max_pool(inputs,
                      weights,
                      window_shape,
                      strides=None,
                      padding='VALID',
                      return_pooled_weights=False):
  """Pools the input by taking max over a window, w.r.t their inputs' weights.

  Args:
    inputs: Input data with dimensions (batch, <window dims>, features).
    weights: Input weights with dimensions (batch, <window dims>).
    window_shape: tuple; A shape tuple defining the window to reduce over.
    strides: tuple; A sequence of `n` integers, representing the inter-window
        strides (default: `(1, ..., 1)`).
    padding: str/list(tuple); Either the string `'SAME'`, the string `'VALID'`,
      or a sequence of `n` `(low, high)` integer pairs that give the padding to
      apply before and after each spatial dimension (default: `'VALID'`).
    return_pooled_weights: bool; Also return the pooled weight

  Returns:
    The maximum of each window slice. If return_pooled_weights is True, it also
    returns the maximum of pooled weights.
  """
  assert inputs.shape[:-1] == weights.shape
  weights = jnp.expand_dims(weights, -1)
  inputs = inputs * weights
  outputs = nn.max_pool(inputs, window_shape, strides=strides, padding=padding)
  if return_pooled_weights:
    max_weights = nn.max_pool(
        weights, window_shape, strides=strides, padding=padding)
    return outputs, max_weights.squeeze(axis=-1)
  return outputs


def weighted_avg_pool(inputs,
                      weights,
                      window_shape,
                      strides=None,
                      padding='VALID',
                      return_pooled_weights=False):
  """Pools the input by averaging over a window, w.r.t their inputs' weights.

  Args:
    inputs: Input data with dimensions (batch, <window dims>, features).
    weights: Input weights with dimensions (batch, <window dims>).
    window_shape: tuple; A shape tuple defining the window to reduce over.
    strides: tuple; A sequence of `n` integers, representing the inter-window
        strides (default: `(1, ..., 1)`).
    padding: str/list(tuple); Either the string `'SAME'`, the string `'VALID'`,
      or a sequence of `n` `(low, high)` integer pairs that give the padding to
      apply before and after each spatial dimension (default: `'VALID'`).
    return_pooled_weights: bool; Also return the pooled weight

  Returns:
    The average for each window slice. If return_pooled_weights is True, it also
    returns the sum of pooled weights.
  """
  assert inputs.shape[:-1] == weights.shape
  weights = jnp.expand_dims(weights, -1)
  inputs = inputs * weights
  y = nn.pooling.pool(inputs, 0., lax.add, window_shape, strides, padding)
  pooled_weights = nn.pooling.pool(weights, 0., lax.add, window_shape, strides,
                                   padding)
  outputs = y / pooled_weights
  if return_pooled_weights:
    return outputs, (pooled_weights.squeeze(axis=-1) / np.prod(window_shape))
  return outputs


def upscale2x_nearest_neighbor(inputs):
  """Doubles image size by repeating every pixel 2x2 times.

  Args:
    inputs: nd-array: Inputs in shape of `[bs, height, width, channels]'

  Returns:
    Upscaled inputs, in shape of  `[bs, 2*height, 2*width, channels]'
  """
  input_channels = inputs.shape[-1]
  input_h, input_w = inputs.shape[1], inputs.shape[2]
  input_nchw = jnp.transpose(inputs, (0, 3, 1, 2))
  flat_input_shape = (-1, input_h, input_w, 1)
  flat_input = jnp.reshape(input_nchw, flat_input_shape)

  height_scale, width_scale = 2, 2
  resize_kernel = jnp.ones((height_scale, width_scale, 1, 1))
  strides = (height_scale, width_scale)
  flat_output = lax.conv_transpose(
      flat_input, resize_kernel, strides, padding='VALID')

  output_nchw_shape = (-1, input_channels, height_scale * input_h,
                       width_scale * input_w)
  output_nchw = jnp.reshape(flat_output, output_nchw_shape)
  resized_x = jnp.transpose(output_nchw, (0, 2, 3, 1))  # Output: nhwc.
  return resized_x


def central_crop(inputs, target_shape):
  """Returns a central crop in axis (1, 2).

  Args:
    inputs: nd-array; Inputs in shape of `[bs, height, width, channels]'.
    target_shape: tuple(int); Target shape after crop.

  Returns:
    Cropped image.
  """
  h, w = target_shape[1:3]
  assert h <= inputs.shape[1], f'{h} > {inputs.shape[1]}'
  assert w <= inputs.shape[2], f'{w} > {inputs.shape[2]}'
  h0 = (inputs.shape[1] - h) // 2
  w0 = (inputs.shape[2] - w) // 2
  return inputs[:, h0:(h0 + h), w0:(w0 + w)]


def compute_1d_relative_distance(query_len: int, key_len: int) -> np.ndarray:
  """Generate relative positions of queries and keys for relative attention.

  Args:
    query_len: Length of the query.
    key_len: Length of the key.

  Returns:
    A numpy (np) int array of shape [len_q, len_k]   holding the distance
    between each query  and key pair, where the query and key are
      indexed by their position. The smallest value in the array is zero.
  """
  # A distance per dimension in the query-key arrays.
  relative_positions = (
      np.arange(key_len)[np.newaxis, :] - np.arange(query_len)[:, np.newaxis])
  # These indices are being used by an embedding lookup, so shift the indices
  # such that the smallest index is zero.
  relative_positions -= np.min(relative_positions)
  return relative_positions


def truncated_normal_initializer(stddev: float = 1e-2,
                                 dtype: jnp.dtype = jnp.float_) -> Initializer:
  """Returns a truncated normal parameter initializer.

  The truncation bounds are -2 and +2 standard deviations.

  Args:
    stddev: The standard deviation of the truncated normal distribution.
    dtype: The data type to use.

  Returns:
    Initializer function compatible with Flax modules.
  """
  def init(key, shape, dtype=dtype):
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    if jnp.issubdtype(dtype, jnp.floating):
      # constant is stddev of standard normal truncated to (-2, 2)
      s = stddev / jnp.array(.87962566103423978, dtype)
    else:
      # constant is stddev of complex standard normal truncated to 2
      s = stddev / jnp.array(.95311164380491208, dtype)
    return jax.random.truncated_normal(key, -2, 2, shape, dtype) * s
  return init
