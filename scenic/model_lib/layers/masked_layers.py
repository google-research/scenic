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

"""Masked Flax layers.

Useful when images (or more broadly) tensors are padded to maximum size for
batching. E.g. a naive convolution of will introduce edge effects that are
different for the padded and unpadded edges of the tensor; similarly, a naive
batch norm will aggregate accross masked out positions, etc. This module
introduces a Flax layers that do not suffer from this.
"""

import functools
from typing import Optional, Tuple, Union, Callable, Any, Sequence, List, Iterable

import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


def _absolute_dims(rank: int, dims: Iterable[int]):
  return tuple([rank + dim if dim < 0 else dim for dim in dims])


def avg_pool(
    inputs: jnp.ndarray,
    window_shape: Tuple[int, ...],
    strides: Optional[Tuple[int, ...]] = None,
    padding: Union[str, Sequence[Tuple[int, int]]] = 'VALID',
    spatial_shape: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
  """Pools the input by taking the average over a window.

  Args:
    inputs: Input data with dimensions (batch, window_dims..., features).
    window_shape: Shape tuple defining the window to reduce over.
    strides: A sequence of `n` integers, representing the inter-window
      strides (default: `(1, ..., 1)`).
    padding: Either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension (default: `'VALID'`).
    spatial_shape: Per-input spatial shape of *unpadded* input data with
      dimensions (batch, window_dims).

  Returns:
    The average for each window slice.
  """
  inputs = nn.avg_pool(inputs, window_shape, strides=strides, padding=padding)
  if spatial_shape is not None:
    if (isinstance(padding, str) and padding.upper() == 'SAME' and
        window_shape != (1,) * len(window_shape)):
      raise NotImplementedError(
          "Padding 'SAME' is not supported by masked mean pool.")
    spatial_shape = _conv_output_shape(spatial_shape=spatial_shape,
                                       kernel_size=window_shape,
                                       input_dilation=None,
                                       kernel_dilation=None,
                                       strides=strides,
                                       padding=padding)
    inputs = apply_spatial_mask(inputs, spatial_shape)

  return inputs, spatial_shape


def max_pool(
    inputs: jnp.ndarray,
    window_shape: Tuple[int, ...],
    strides: Optional[Tuple[int, ...]] = None,
    padding: Union[str, Sequence[Tuple[int, int]]] = 'VALID',
    spatial_shape: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
  """Pools the input by taking the maximum of a window slice.

  Args:
    inputs: Input data with dimensions (batch, window_dims..., features).
    window_shape: A shape tuple defining the window to reduce over.
    strides: A sequence of `n` integers, representing the inter-window
      strides (default: `(1, ..., 1)`).
    padding: Either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension (default: `'VALID'`).
    spatial_shape: Per-input spatial shape of *unpadded* input data with
      dimensions (batch, window_dims).

  Returns:
    The maximum for each window slice.
  """
  if spatial_shape is not None:
    # Unlike for avg pool, for max pool a spatial mask must be applied first.
    inputs = apply_spatial_mask(inputs, spatial_shape, value=-jnp.inf)

  inputs = nn.max_pool(inputs, window_shape, strides=strides, padding=padding)
  if spatial_shape is not None:
    if (isinstance(padding, str) and padding.upper() == 'SAME' and
        window_shape != (1,) * len(window_shape)):
      raise NotImplementedError(
          "Padding 'SAME' is not supported by masked max pool.")
    spatial_shape = _conv_output_shape(spatial_shape=spatial_shape,
                                       kernel_size=window_shape,
                                       input_dilation=None,
                                       kernel_dilation=None,
                                       strides=strides,
                                       padding=padding)
    inputs = apply_spatial_mask(inputs, spatial_shape)

  return inputs, spatial_shape


def _bn_agg_mean_var(
    x: jnp.ndarray,
    axis: Union[Tuple[int, ...], int],
    p_agg: bool, *,
    axis_name: Optional[str] = None,
    axis_index_groups: Optional[Sequence[Sequence[int]]] = None,
    spatial_shape: Optional[jnp.ndarray] = None):
  """Aggregate batch statistics accross devices.

  Args:
    x: Inputs to compute batch statistics on.
    axis: Reduction axes for the stats.
    p_agg: If True, parallel aggregation is performed using psum.
    axis_name: Name of the axis for psum aggregation.
    axis_index_groups: Groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None).
    spatial_shape: Per-input spatial shape of *unpadded* input data with
        dimensions (batch, spatial_dims, channels).

  Returns:
    Batch stats (mean and variance).
  """
  # When using spatial padding, we cannot accumulate the mean directly and
  # instead must aaccumulate the numerator and sum directly.
  acc_sum = jnp.sum(x, axis=axis, keepdims=False)
  acc_sum2 = jnp.sum(lax.square(x), axis=axis, keepdims=False)
  if spatial_shape is None:
    denom = np.prod([x.shape[i] for i in axis])
  else:
    reduction_axis_shifted = tuple(i - 1 for i in axis if i > 0)
    denom = jnp.prod(spatial_shape[:, reduction_axis_shifted], axis=-1)
    denom = jnp.sum(denom)

  if p_agg:
    concatenated_acc_sum = jnp.concatenate([acc_sum, acc_sum2, denom])
    acc_sum, acc_sum2, denom = jnp.split(
        lax.psum(
            concatenated_acc_sum,
            axis_name=axis_name,
            axis_index_groups=axis_index_groups), 3)

  denom = jnp.maximum(denom, 1.)
  mean = acc_sum / denom
  var = acc_sum2 / denom - lax.square(mean)
  return mean, var


class BatchNorm(nn.Module):
  """Masking-aware Batch Normalization layer.

  Attributes:
    use_running_average: If True, the statistics stored in batch_stats
      will be used instead of computing the batch statistics on the input.
    axis: The feature or non-batch axis of the input.
    momentum: Decay rate for the exponential moving average of
      the batch statistics.
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: The dtype of the computation (default: float32).
    use_bias:  If True, bias (beta) is added.
    use_scale: If True, multiply by scale (gamma).
      When the next layer is linear (also e.g. nn.relu), this can be disabled
      since the scaling will be done by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
    spatial_norm: If True, spatial shapes influence group norm weights,
      otherwise every batch element has an equal weight.
    axis_name: Axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
    axis_index_groups: Groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
      examples on the first two and last two devices. See `jax.lax.psum` for
      more details.
  """

  use_running_average: Optional[bool] = None
  axis: int = -1
  momentum: float = 0.99
  epsilon: float = 1e-5
  dtype: jnp.dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[
      [Any, Tuple[int, ...], Any], Any] = nn.initializers.zeros
  scale_init: Callable[
      [Any, Tuple[int, ...], Any], Any] = nn.initializers.ones
  spatial_norm: bool = True
  axis_name: Optional[str] = None
  axis_index_groups: Optional[Sequence[Sequence[int]]] = None

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      use_running_average: Optional[bool] = None,
      spatial_shape: Optional[jnp.ndarray] = None
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Normalizes the input using batch statistics.

    Args:
      x: The input to be normalized.
      use_running_average: If True, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.
      spatial_shape: Per-input spatial shape of *unpadded* input data with
        dimensions (batch, spatial_dims, channels).

    Returns:
      Normalized inputs (the same shape as inputs) and the spatial shape.
    """
    use_running_average = nn.module.merge_param(
        'use_running_average', self.use_running_average, use_running_average)
    x = jnp.asarray(x, jnp.float32)
    axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
    axis = _absolute_dims(x.ndim, axis)
    feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
    reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
    reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)

    # Detect if we're in initialization via empty variable tree.
    initializing = not self.has_variable('batch_stats', 'mean')

    ra_mean = self.variable('batch_stats', 'mean',
                            lambda s: jnp.zeros(s, jnp.float32),
                            reduced_feature_shape)
    ra_var = self.variable('batch_stats', 'var',
                           lambda s: jnp.ones(s, jnp.float32),
                           reduced_feature_shape)

    if use_running_average:
      mean, var = ra_mean.value, ra_var.value
    else:
      p_agg = ((self.axis_name is not None) and (not initializing)
               and self.spatial_norm)
      mean, var = _bn_agg_mean_var(
          x,
          reduction_axis,
          p_agg,
          axis_name=self.axis_name,
          axis_index_groups=self.axis_index_groups,
          spatial_shape=spatial_shape)

      if not initializing:
        ra_mean.value = (self.momentum * ra_mean.value +
                         (1 - self.momentum) * mean)
        ra_var.value = (self.momentum * ra_var.value +
                        (1 - self.momentum) * var)

    # Apply normaliation.
    mean, var = mean.reshape(feature_shape), var.reshape(feature_shape)
    y = x - mean
    mul = lax.rsqrt(var + self.epsilon)

    if self.use_scale:
      scale = self.param('scale',
                         self.scale_init,
                         reduced_feature_shape).reshape(feature_shape)
      mul = mul * scale
    y = y * mul
    if self.use_bias:
      bias = self.param('bias',
                        self.bias_init,
                        reduced_feature_shape).reshape(feature_shape)
      y = y + bias

    if spatial_shape is not None:
      # Restore spatial mask for the outputs.
      y = apply_spatial_mask(y, spatial_shape)
    return jnp.asarray(y, self.dtype), spatial_shape


class GroupNorm(nn.Module):
  """Masking-aware Group Normalization (arxiv.org/abs/1803.08494).

  This op is similar to batch normalization, but statistics are shared across
  equally-sized groups of channels and not shared across batch dimension.
  Thus, group normalization does not depend on the batch composition and does
  not require maintaining internal state for storing statistics.
  The user should either specify the total number of channel groups or the
  number of channels per group.

  Attributes:
    num_groups: Total number of channel groups. The default value of 32 is
      proposed by the original group normalization paper.
    group_size: The number of channels in a group.
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: The dtype of the computation (default: float32).
    use_bias: If True, bias (beta) is added.
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
    spatial_norm: If True, spatial shapes influence group norm weights,
      otherwise every batch element has an equal weight.
  """

  num_groups: int = 32
  group_size: Optional[int] = None
  epsilon: float = 1e-6
  dtype: jnp.dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[
      [Any, Tuple[int, ...], Any], Any] = nn.initializers.zeros
  scale_init: Callable[
      [Any, Tuple[int, ...], Any], Any] = nn.initializers.ones
  spatial_norm: bool = True

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      spatial_shape: Optional[jnp.ndarray] = None,
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Applies group normalization to the input (arxiv.org/abs/1803.08494).

    Args:
      x: Input of shape N...C, where N is a batch dimension and C is a channels
        dimensions. `...` represents an arbitrary number of extra dimensions
        that are used to accumulate statistics over.
      spatial_shape: Per-input spatial shape of *unpadded* input data with
        dimensions (batch, spatial_dims, channels).

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    x = jnp.asarray(x, jnp.float32)
    if ((self.num_groups is None and self.group_size is None) or
        (self.num_groups is not None and self.group_size is not None)):
      raise ValueError('Either `num_groups` or `group_size` should be '
                       'specified, but not both of them.')
    num_groups = self.num_groups

    channels = x.shape[-1]
    if self.group_size is not None:
      if channels % self.group_size != 0:
        raise ValueError('Number of channels ({}) is not multiple of the '
                         'group size ({}).'.format(channels, self.group_size))
      num_groups = channels // self.group_size

    if num_groups <= 0 or channels % num_groups != 0:
      raise ValueError('Number of groups ({}) does not divide the number'
                       ' of channels ({}).'.format(num_groups, channels))

    input_shape = x.shape
    group_shape = x.shape[:-1] + (num_groups, x.shape[-1] // num_groups)
    x = x.reshape(group_shape)

    reduction_axis = tuple(range(1, x.ndim - 2)) + (x.ndim - 1,)
    mean = jnp.mean(x, axis=reduction_axis, keepdims=True)
    mean_of_squares = jnp.mean(jnp.square(x), axis=reduction_axis,
                               keepdims=True)
    orig_denom = np.prod([x.shape[i] for i in reduction_axis[:-1]])
    if (spatial_shape is not None) and self.spatial_norm:
      reduction_axis_shifted = tuple(
          i - 1 for i in reduction_axis[:-1] if i > 0)
      denom = jnp.prod(spatial_shape[:, reduction_axis_shifted], axis=-1)
      denom = jnp.reshape(denom, (denom.shape[0],) + (1,) * (mean.ndim - 1))
      denom = jnp.maximum(denom, 1.)
      mean = mean * (orig_denom / denom)
      mean_of_squares = mean_of_squares  * (orig_denom / denom)

    var = mean_of_squares - jnp.square(mean)
    x = (x - mean) * lax.rsqrt(var + self.epsilon)
    x = x.reshape(input_shape)

    feature_shape = tuple([1 for d in input_shape[:-1]] + [input_shape[-1]])
    if self.use_scale:
      x *= self.param('scale', self.scale_init, feature_shape)
    if self.use_bias:
      x += self.param('bias', self.bias_init, feature_shape)

    if spatial_shape is not None:
      # Restore spatial mask for the outputs.
      x = apply_spatial_mask(x, spatial_shape)
    return x.astype(self.dtype), spatial_shape


class Conv(nn.Conv):
  """Masked convolution.

  Attributes:
    features: Number of convolution filters.
    kernel_size: Shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer. For all other cases, it
      must be a sequence of integers.
    strides: A sequence of `n` integers, representing the inter-window
      strides.
    padding: Either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension.
    input_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `inputs`.
      Convolution with input dilation `d` is equivalent to transposed
      convolution with stride `d`.
    kernel_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel. Convolution with kernel dilation is also known as 'atrous
      convolution'.
    bias: Whether to add a bias to the output (default: True).
    dtype: The dtype of the computation (default: float32).
    precision: Numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: Initializer for the convolutional kernel.
    bias_init: Initializer for the bias.
  """

  features: int
  kernel_size: Union[int, Tuple[int, ...]]
  strides: Optional[Tuple[int, ...]] = None
  padding: Union[str, Sequence[Tuple[int, int]]] = 'SAME'
  input_dilation: Optional[Tuple[int, ...]] = None
  kernel_dilation: Optional[Tuple[int, ...]] = None
  use_bias: bool = True
  dtype: jnp.dtype = jnp.float32
  precision: Optional[jax.lax.Precision] = None
  kernel_init: Callable[  # pytype: disable=annotation-type-mismatch  # jax-types
      [Any, Sequence[int], Any], Any] = nn.linear.default_kernel_init
  bias_init: Callable[
      [Any, Sequence[int], Any], Any] = nn.initializers.zeros

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      spatial_shape: Optional[jnp.ndarray] = None
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Applies a *masked* convolution to the inputs.

    Args:
      inputs: Input data with dimensions (batch, spatial_dims..., features).
      spatial_shape: Per-input spatial shape of *unpadded* input data with
        dimensions (batch, len(spatial_dims)).

    Returns:
      The convolved data and the output spatial size as a tuple.
    """
    outputs = super(Conv, self).__call__(inputs)
    kernel_size = self.kernel_size
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size,)

    if spatial_shape is not None:
      if (isinstance(self.padding, str) and self.padding.upper() == 'SAME' and
          kernel_size != (1,) * len(kernel_size)):
        # In the case of 'SAME' padding the ammounts padded on the left and
        # right depend on the shape of the input (which is dynamic in our case),
        # the stride and the kernel size. In our case this means that each
        # element of the batch might have to be paded in a different way and
        # cannnot be batched.
        # Because the possible number of dynamic pads for a given stride kernel
        # size is finite, it should be possible in theory to run all of them and
        # then select the correct out for masking. The implementation is left
        # for a time when we have a use case for this.
        raise NotImplementedError(
            "Padding 'SAME' is not supported by masked convolutions.")
      spatial_shape = _conv_output_shape(spatial_shape=spatial_shape,
                                         kernel_size=kernel_size,
                                         input_dilation=self.input_dilation,
                                         kernel_dilation=self.kernel_dilation,
                                         strides=self.strides,
                                         padding=self.padding)
      outputs = apply_spatial_mask(outputs, spatial_shape)
    return outputs, spatial_shape


def apply_spatial_mask(
    inputs: jnp.ndarray,
    spatial_shape: jnp.ndarray,
    value: float = 0.) -> jnp.ndarray:
  """Construct and apply spatial mask to the inputs.

  Args:
    inputs: Input tensor with dimensions [batch, spatial_dim, dim] to which the
      mask should be applied.
    spatial_shape: Per-input spatial shape of *unpadded* input data with
      dimensions (batch, len(spatial_dims)).
    value: Value to use for the mask (default: 0).

  Returns:
    Inputs with masking spatial masking applied to them.
  """
  assert inputs.shape[0] == spatial_shape.shape[0]
  assert spatial_shape.shape[1] == inputs.ndim - 2
  mask = mask_from_spatial(inputs.shape[1:-1], spatial_shape, per_axis=False)
  mask = jnp.expand_dims(mask, axis=-1)
  inputs = jnp.where(mask, inputs, value)
  return inputs


def mask_from_spatial(
    padded_shape: Tuple[int, ...],
    spatial_shape: jnp.ndarray,
    per_axis: bool = False) -> Union[jnp.ndarray, List[jnp.ndarray]]:
  """Create a spatial mask for a given padded shape and spatial size.

  Args:
    padded_shape: Shape of the spatial dimensions padded data that needs to be
      masked.
    spatial_shape: Per-element unpadded spatial size of the data with dimensions
      (batch, len(padded_shape)).
    per_axis: If True, per list of per-spatial-dim masks is returned instead of
      a single mask; that should enable some memory savings as it allows for
      shape boradcasting to happen only when it is required.

  Returns:
    The spatial mask.
  """
  assert spatial_shape.ndim == 2
  assert len(padded_shape) == spatial_shape.shape[1]
  ndim = spatial_shape.shape[1]

  masks = []
  for i in range(ndim):
    # Construct per-axis mask and then broadcast if asked for.
    mask = jnp.arange(0, padded_shape[i], dtype=jnp.int32)
    mask = jnp.reshape(
        mask,
        (1,) * (i + 1) + (padded_shape[i],) + (1,) * (ndim - i - 1))
    threshold = spatial_shape[:, i]
    threshold = jnp.reshape(threshold, (spatial_shape.shape[0],) + (1,) * ndim)
    mask = mask < threshold
    masks.append(mask)

  if not per_axis:
    masks = functools.reduce(jnp.logical_and, masks)
  return masks


def _dilate_shape(shape: jnp.ndarray, dilation: Tuple[int, ...]):
  """Utility function for computing the shape resulting from a dilation.

  Args:
    shape: Shapes (input or kernel i.e. lhr or rhs) to which the dilation should
      be applied.
    dilation: The dilation to apply.

  Returns:
    Dilated input shapes.
  """
  if not np.all(np.greater(dilation, 0)):
    raise TypeError(f'All dilations must be positive, got {dilation}.')
  dilation = (1,) * (shape.shape[1] - len(dilation)) + tuple(dilation)
  dilation = jnp.array(dilation)
  return jnp.where(shape == 0, 0,
                   jnp.multiply(dilation, jnp.subtract(shape, 1)) + 1)


def _conv_output_shape(
    spatial_shape: jnp.ndarray,
    kernel_size: Tuple[int, ...],
    input_dilation: Optional[Tuple[int, ...]],
    kernel_dilation: Optional[Tuple[int, ...]],
    strides: Optional[Tuple[int, ...]],
    padding: Union[str, Sequence[Tuple[int, int]]]) -> jnp.ndarray:
  """Convenience wrapper function for inferring the convolution output shape.

  Args:
    spatial_shape: Input (lhs) shapes for which the output shapes should be
      inferred as array with dimensions (batch, spatial_dims, dims).
    kernel_size: Covolution kernel size (i.e. rhs shape).
    input_dilation: Input (lhs) dilation.
    kernel_dilation: Convolution kernel (rhs) dilation.
    strides: Convolution (rhs) stride.
    padding: Input (lhs) padding.

  Returns:
    Inferred convolution output shapes as array with dimensions
    (batch, len(spatial_dims)).
  """
  strides = strides or (1,) * len(kernel_size)
  if input_dilation is not None:
    spatial_shape = _dilate_shape(spatial_shape, input_dilation)
  if kernel_dilation is not None:
    kernel_size = tuple(
        (k - 1) * r + 1 for k, r in zip(kernel_size, kernel_dilation))
  spatial_shape = jnp.concatenate(
      [jnp.ones((spatial_shape.shape[0], 2), dtype=jnp.int32), spatial_shape],
      axis=-1)
  out_shape = conv_shape_tuple(lhs_shape=spatial_shape,
                               rhs_shape=(1, 1) + kernel_size,
                               strides=strides,
                               pads=padding)
  return out_shape[:, 2:]


def _ceil_divide(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
  """Ceil division of two JAX arrays."""
  return -jnp.floor_divide(jnp.negative(x1), x2)


def padtype_to_pads(
    in_shape: jnp.ndarray,
    window_shape: Tuple[int, ...],
    window_strides: Tuple[int, ...],
    padding: str) -> jnp.ndarray:
  """Convert padding string to list of pairs of pad values.

  Args:
    in_shape: Input (lhs) shapes for which the padding should be inferred as
      array with dimensions (batch, spatial_dims).
    window_shape: Window (kernel; rhs) shape of the convolution.
    window_strides: Window (kernel; rhs) convolution strides.
    padding: Convlution (rhs) padding.

  Returns:
    Inferred lhs paddings as array with dimensions
    (batch, len(spatial_dims), 2).
  """
  xc = jax.lib.xla_client
  if isinstance(padding, str):
    mapping = {'VALID': xc.PaddingType.VALID, 'SAME': xc.PaddingType.SAME}  # pytype: disable=module-attr  # gen-stub-imports
    try:
      padding = mapping[padding.upper()]
    except KeyError as err:
      msg = "Unrecognized padding type: expected 'VALID' or 'SAME', got {}."
      raise RuntimeError(msg.format(padding)) from err

  if padding == xc.PaddingType.SAME:  # pytype: disable=module-attr  # gen-stub-imports
    window_shape = jnp.array(window_shape)
    window_strides = jnp.array(window_strides)
    out_shape = _ceil_divide(in_shape, window_strides)
    pad_sizes = jnp.maximum(
        (out_shape - 1) * window_strides + window_shape - in_shape, 0)
    pad_sizes = jnp.stack([pad_sizes // 2, pad_sizes - pad_sizes // 2], axis=1)
    return pad_sizes
  elif padding == xc.PaddingType.VALID:  # pytype: disable=module-attr  # gen-stub-imports
    return jnp.zeros((in_shape.shape[0], len(window_shape), 2), dtype=jnp.int32)
  raise TypeError(f'Unknown padding type: {padding}.')


def conv_shape_tuple(
    lhs_shape: jnp.ndarray,
    rhs_shape: Tuple[int, ...],
    strides: Tuple[int, ...],
    pads: Union[str, Sequence[Tuple[int, int]]]) -> jnp.ndarray:
  """Compute the shape of a conv given input shapes in canonical order.

  Args:
    lhs_shape: Input (lhs) shapes for which the output shapes should be inferred
      as array with dimensions (batch, spatial_dims, dims).
    rhs_shape: Covolution kernel size (i.e. rhs shape).
    strides: Convolution (rhs) stride.
    pads: Input (lhs) padding.

  Returns:
    Inferred convolution output shapes as array with dimensions
    (batch, len(spatial_dims)).
  """
  if isinstance(pads, str):
    pads = padtype_to_pads(lhs_shape[:, 2:], rhs_shape[2:], strides, pads)
  else:
    pads = jnp.expand_dims(jnp.array(pads), axis=0)

  if pads.shape[1] != lhs_shape.shape[1] - 2:
    msg = 'Wrong number of explicit pads for convolution: expected {}, got {}.'
    raise TypeError(msg.format(lhs_shape.shape[1] - 2, pads.shape[1]))
  lhs_padded = jnp.add(lhs_shape[:, 2:], jnp.sum(pads, axis=2))

  rhs_shape = jnp.array(rhs_shape, dtype=jnp.int32)
  strides = jnp.array(strides, dtype=jnp.int32)
  out_space = jnp.floor_divide(
      jnp.subtract(lhs_padded, rhs_shape[2:]), strides) + 1
  out_space = jnp.maximum(0, out_space)
  out_shape = jnp.stack(
      [lhs_shape[:, 0], jnp.full((lhs_shape.shape[0],), rhs_shape[0])],
      axis=-1)
  return jnp.concatenate([out_shape, out_space], axis=-1)
