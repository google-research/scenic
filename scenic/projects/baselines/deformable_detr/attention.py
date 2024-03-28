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

"""Modules and uttilities for multi-scale deformable attention.

See paper "Deformable DETR: Deformable Transformers for End-to-End Object
Detection" [1] and corresponding code [2].

[1] https://arxiv.org/abs/2010.04159.
[2] https://github.com/fundamentalvision/Deformable-DETR.
"""

import functools
import math
from typing import Callable, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np


Array = Union[jnp.ndarray, np.ndarray]
Shape = Tuple[int, ...]


@functools.partial(jax.jit, static_argnames=('w', 'h'))
def bilinear_interpolate(im: Array, grid: Array, w: int, h: int) -> jnp.ndarray:
  """Performs 2D bilinear interpolation.

  It is assumed that the center of the top-left pixel in `im` has coordinate
  (0.5, 0.5). If you want a different mapping, transform `grid` before calling
  this function. For example, if you want the center of the top-left pixel to
  have coordinate (0, 0), pass `grid + [0.5, 0.5]` into the function instead.

  Args:
    im: [image_height * image_width, nembed] a flattened 2D image.
    grid: [..., 2] normalized sampling grid.
    w: Image width.
    h: Image height.

  Returns:
    [..., nembed] array of interpolated values.
  """
  im = im.reshape(h, w, -1)
  im = jnp.pad(im, ((1, 1), (1, 1), (0, 0)), 'empty')
  im = im.reshape((h + 2) * (w + 2), -1)

  x = grid[..., 0] * w
  y = grid[..., 1] * h
  x -= 0.5
  y -= 0.5
  x0 = jnp.floor(x).astype(int)
  x1 = x0 + 1
  y0 = jnp.floor(y).astype(int)
  y1 = y0 + 1

  # An important observation that can be made is that we can group the gathering
  # of (x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1) together since they use
  # the same indices for gathering. After packing, we perform one gathering with
  # 4x feature dimension instead of four gatherings with 1x feature dimension,
  # which greatly improves the speed on TPUs, since TPUs have very slow
  # gathering and also forces the feature dimension onto the 128-dimensional
  # sublanes.

  # prepare for packing
  indices_y0_offset = (jnp.arange(h + 1) * (w + 2))[:, None]
  indices_y1_offset = (jnp.arange(1, h + 2) * (w + 2))[:, None]
  indices_x0 = jnp.arange(w + 1)
  indices_x1 = jnp.arange(1, w + 2)
  im00 = im[(indices_y0_offset + indices_x0).flatten()]
  im10 = im[(indices_y1_offset + indices_x0).flatten()]
  im01 = im[(indices_y0_offset + indices_x1).flatten()]
  im11 = im[(indices_y1_offset + indices_x1).flatten()]

  # pack: (nembed * 4, (h + 1) * (w + 1))
  im_packed = jnp.concatenate([im00, im10, im01, im11], axis=-1)

  # gather
  indices11 = jnp.clip(y1, 0, h + 1) * (w + 1) + jnp.clip(x1, 0, w + 1)
  im_gathered = im_packed[indices11]

  # unpack
  im_a, im_b, im_c, im_d = jnp.split(im_gathered, 4, axis=-1)

  # Mark indices out-of-bounds.
  x0_out = jnp.logical_or(x0 < 0, x0 > w - 1)
  y0_out = jnp.logical_or(y0 < 0, y0 > h - 1)
  x1_out = jnp.logical_or(x1 < 0, x1 > w - 1)
  y1_out = jnp.logical_or(y1 < 0, y1 > h - 1)
  out00 = jnp.logical_or(x0_out, y0_out)
  out01 = jnp.logical_or(x0_out, y1_out)
  out10 = jnp.logical_or(x1_out, y0_out)
  out11 = jnp.logical_or(x1_out, y1_out)

  # Set weights where weights for out-of-bound pixels are forced to be 0.
  wa = jnp.where(out00, 0, (x1 - x) * (y1 - y))
  wb = jnp.where(out01, 0, (x1 - x) * (y - y0))
  wc = jnp.where(out10, 0, (x - x0) * (y1 - y))
  wd = jnp.where(out11, 0, (x - x0) * (y - y0))

  return (jnp.einsum('...e,...->...e', im_a, wa) +
          jnp.einsum('...e,...->...e', im_b, wb) +
          jnp.einsum('...e,...->...e', im_c, wc) +
          jnp.einsum('...e,...->...e', im_d, wd))


def _map(
    map_fn: Callable[[Array], jnp.ndarray],
    mode: str,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """A versatile vmap-like function."""
  if mode == 'auto':
    # Current tests show that for our purpose 'map' is better than the other
    # options on all platforms ('gpu', 'tpu', 'cpu'). But this may change in the
    # future.
    mode = 'map'
  if mode == 'loop':
    return lambda v: jnp.stack([map_fn(e) for e in v], axis=0)
  elif mode == 'map':
    return lambda v: jax.lax.map(map_fn, v)
  elif mode == 'vmap':
    return jax.vmap(map_fn)
  else:
    raise ValueError('Invalid batching mode.')


@functools.partial(jax.jit, static_argnames=('shapes', 'use_remat', 'mode'))
def deform_attn_sampling_fn(values: Array, sampling_locations: Array,
                            attn_weights: Array, shapes: Tuple[Tuple[int, int],
                                                               ...],
                            use_remat: bool, mode: str) -> jnp.ndarray:
  """Performs deformable attention sampling calculation.

  Args:
    values: [bs, len_v, nembed]-array of values.
    sampling_locations: [bs, nlevels, npoints * len_q, 2]-array of
      sampling locations.
    attn_weights: [bs, len_q, nlevels * npoints]-array of attention weights.
    shapes: Static tuple of image shapes for each level to unflatten len_v.
    use_remat: Flag for rematerialization.
    mode: Determines how batching is performed, can be one of the following:
      'loop', 'map', 'vmap', 'auto'.

  Returns:
    [bs, len_q, nembed]-array
  """
  nembed = values.shape[-1]
  len_q = attn_weights.shape[1]
  split_indices = np.cumsum(np.array([h * w for h, w in shapes]))[:-1]
  # Split values by level.
  values_by_level = jnp.split(values, split_indices, axis=-2)
  sampled_values_all_levels = []

  if use_remat:
    attention_fn = jax.remat(
        bilinear_interpolate,
        policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
        static_argnums=(2, 3))
  else:
    attention_fn = bilinear_interpolate

  def attention_at_index_fn(i, level_idx):
    return attention_fn(
        values_by_level[level_idx][i],
        sampling_locations[:, level_idx][i],
        shapes[level_idx][1],
        shapes[level_idx][0],
    )

  def reshape_attn(fn):
    return lambda *args: jnp.reshape(fn(*args), (-1, len_q, nembed))

  for level_idx in range(len(shapes)):
    fn = functools.partial(attention_at_index_fn, level_idx=level_idx)
    fn = _map(reshape_attn(fn), mode)
    sampled_values_all_levels.append(fn(jnp.array(range(values.shape[0]))))

  # (bs, nlevels * npoints, len_q, nembed)
  sampled_values_all_levels = jnp.concatenate(
      sampled_values_all_levels, axis=1)

  # (bs, len_q, nembed)
  return jnp.einsum('blqe,bql->bqe', sampled_values_all_levels, attn_weights)


class MultiScaleDeformableAttention(nn.Module):
  """Layer for MultiScaleDeformableAttention."""
  spatial_shapes: Sequence[Tuple[int, int]]
  embed_dim: int
  num_levels: int
  num_heads: int
  num_points: int
  compiler_config: ml_collections.ConfigDict
  dtype: jnp.dtype

  def setup(self):
    # (nlevels, 2)
    arr_shapes = jnp.asarray(self.spatial_shapes)
    self.offset_norm = jnp.stack(
        [arr_shapes[:, 1], arr_shapes[:, 0]], -1)
    # (1, 1, 1, nlevels, 1, 2)
    self.offset_norm = self.offset_norm[None, None, None, :, None, :]

  def pos_grid_init(self, key: jax.Array, shape: Shape,
                    dtype: jnp.dtype) -> Array:
    """Initializes deformable attention sampling offsets."""
    del key, shape
    thetas = jnp.arange(
        self.num_heads, dtype=dtype) * (2 * math.pi) / self.num_heads
    grid_init = jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], -1)
    denom = jnp.max(jnp.abs(grid_init), axis=-1, keepdims=True)
    grid_init = (grid_init / denom)
    grid_init = grid_init.reshape(self.num_heads, 1, 1, 2)
    grid_init = jnp.tile(grid_init, (1, self.num_levels, self.num_points, 1))
    for i in range(self.num_points):
      grid_i = grid_init[:, :, i, :] * (i + 1)
      grid_init = grid_init.at[:, :, i, :].set(grid_i)
    return grid_init

  @nn.compact
  def __call__(self, query: jnp.ndarray, ref_points: jnp.ndarray,
               value: jnp.ndarray, pad_mask: jnp.ndarray,
               train: bool) -> jnp.ndarray:
    """Calculates multi-scale multi-head deformable attention.

    Args:
      query: [bs, len_q, embed_dim]-ndarray of queries.
      ref_points: [bs, len_q, num_levels, box_dim]-ndarray of reference points
        for each query.
      value: [bs, len_v, embed_dim]-ndarray of values.
      pad_mask: [bs, len_v]-ndarray of boolean values, where 0 indicates pad.
      train: Whether we are in training mode.

    Returns:
      Attention weighted values based on the queries and reference points
      relevant to the queries.
    """
    assert self.embed_dim % self.num_heads == 0, (
        '`embed_dim` must be divisible by `num_heads`.')
    ref_dim = ref_points.shape[-1]
    bs, len_q, _ = query.shape
    _, len_v, _ = value.shape

    nembed = self.embed_dim // self.num_heads

    # (bs, len_v, nheads, nembed)
    value = nn.DenseGeneral(
        features=(self.num_heads, nembed),
        param_dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.zeros,
        name='value_proj',
    )(
        value)
    # (bs, len_v, nheads, nembed)
    value = jnp.where(pad_mask[..., None, None], value, 0)
    # (bs, nheads, len_v, nembed)
    value = value.transpose(0, 2, 1, 3)
    # (bs * nheads, len_v, nembed)
    value = value.reshape(bs * self.num_heads, len_v, -1)

    # (bs, len_q, nheads, nlevels, npoints, 2)
    sampling_offsets = nn.DenseGeneral(
        features=(self.num_heads, self.num_levels, self.num_points, 2),
        param_dtype=self.dtype,
        kernel_init=nn.initializers.zeros,
        bias_init=self.pos_grid_init,
        name='sampling_offsets',
    )(
        query)

    if ref_dim == 2:
      # (bs, len_q, 1, nlevels, 1, 2)
      ref_xy = ref_points[:, :, None, :, None, :]
      # (bs, len_q, nheads, nlevels, npoints, 2)
      normalized_offsets = sampling_offsets / self.offset_norm
    elif ref_dim >= 4:
      # (bs, len_q, 1, nlevels, 1, 2)
      ref_xy = ref_points[:, :, None, :, None, :2]
      # (bs, len_q, 1, nlevels, 1, 2)
      ref_wh = ref_points[:, :, None, :, None, 2:4]
      # (bs, len_q, nheads, nlevels, npoints, 2)
      normalized_offsets = sampling_offsets / (2 * self.num_points) * ref_wh

    # (bs, len_q, nheads, nlevels, npoints, 2)
    sampling_locations = ref_xy + normalized_offsets
    # (bs, nheads, nlevels, npoints, len_q, 2)
    sampling_locations = sampling_locations.transpose(0, 2, 3, 4, 1, 5)
    # (bs * nheads, nlevels, npoints * len_q, 2)
    sampling_locations = sampling_locations.reshape(bs * self.num_heads,
                                                    self.num_levels,
                                                    self.num_points * len_q, 2)

    # (bs, len_q, nheads, nlevels * npoints)
    attn_weights = nn.DenseGeneral(
        features=(self.num_heads, self.num_levels * self.num_points),
        param_dtype=self.dtype,
        kernel_init=nn.initializers.zeros,
        bias_init=nn.initializers.zeros,
        name='attention_weights',
    )(
        query)
    # (bs, len_q, nheads, nlevels * npoints)
    attn_weights = nn.softmax(attn_weights)
    # (bs, nheads, len_q, nlevels * npoints)
    attn_weights = attn_weights.transpose(0, 2, 1, 3)
    # (bs * nheads, len_q, nlevels * npoints)
    attn_weights = attn_weights.reshape(bs * self.num_heads, len_q,
                                        self.num_levels * self.num_points)
    if train:
      use_remat = self.compiler_config.train_remat
    else:
      use_remat = False
    # (bs * nheads, len_q, nembed)
    x = deform_attn_sampling_fn(
        values=value,
        sampling_locations=sampling_locations,
        attn_weights=attn_weights,
        shapes=self.spatial_shapes,
        use_remat=use_remat,
        mode=self.compiler_config.attention_batching_mode)

    # (bs, nheads, len_q, nembed)
    x = x.reshape(bs, self.num_heads, len_q, nembed)
    # (bs, len_q, embed_dim)
    return nn.DenseGeneral(
        features=self.embed_dim,
        axis=(-3, -1),
        param_dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.zeros,
        name='output_proj',
    )(
        x)
