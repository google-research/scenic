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
import numpy as np

Array = Union[jnp.ndarray, np.ndarray]


def bilinear_interpolate(im: Array,
                         grid: Array,
                         zero_center: bool) -> jnp.ndarray:
  """Perform 2D bilinear interpolation, i.e., im(x, y).

  Uses zero-padding such that all indices outside of the image bounds are 0.

  Args:
    im: [..., num_rows, num_cols] 2D image to interpolate values from.
    grid: [..., 2] normalized sampling grid.
    zero_center: Consider the case of wanting to return the first pixel value
      exactly. For zero_center == False (default) you pass the point (0.5, 0.5),
      whereas for zero_center == True you pass (0, 0), i.e., the center of
      arbitrary pixel is at (i, j) or (i + 0.5, j + 0.5). This matches pytorch
      argument name `align_corners`.

  Returns:
    Interpolated value from im for each (x, y) pair.
  """
  w = im.shape[-1]
  h = im.shape[-2]
  x, y = grid[..., 0] * w, grid[..., 1] * h
  im = im.reshape(im.shape[:-2] + (-1,))
  if not zero_center:
    x -= 0.5
    y -= 0.5
  x0 = jnp.floor(x).astype(int)
  x1 = x0 + 1
  y0 = jnp.floor(y).astype(int)
  y1 = y0 + 1

  # Mark indices out-of-bounds.
  x0_out = jnp.logical_or(x0 < 0, x0 > w - 1)
  y0_out = jnp.logical_or(y0 < 0, y0 > h - 1)
  x1_out = jnp.logical_or(x1 < 0, x1 > w - 1)
  y1_out = jnp.logical_or(y1 < 0, y1 > h - 1)

  # All remaining indices are strictly in-bounds.
  x0_in = jnp.clip(x0, 0, w - 1)
  x1_in = jnp.clip(x1, 0, w - 1)
  y0_in = jnp.clip(y0, 0, h - 1)
  y1_in = jnp.clip(y1, 0, h - 1)

  indices00 = y0_in * w + x0_in
  indices10 = y1_in * w + x0_in
  indices01 = y0_in * w + x1_in
  indices11 = y1_in * w + x1_in

  # Sample from image where out-of-bounds value == 0.
  im_a = jnp.where(jnp.logical_or(x0_out, y0_out), 0, im[..., indices00])
  im_b = jnp.where(jnp.logical_or(x0_out, y1_out), 0, im[..., indices10])
  im_c = jnp.where(jnp.logical_or(x1_out, y0_out), 0, im[..., indices01])
  im_d = jnp.where(jnp.logical_or(x1_out, y1_out), 0, im[..., indices11])

  # Get linear interpolation weights.
  wa = (x1 - x) * (y1 - y)
  wb = (x1 - x) * (y - y0)
  wc = (x - x0) * (y1 - y)
  wd = (x - x0) * (y - y0)

  return im_a * wa + im_b * wb + im_c * wc + im_d * wd


@functools.partial(jax.jit, static_argnums=(3, 4))
def deform_attn_sampling_fn(value: Array, sampling_locations: Array,
                            attn_weights: Array, shapes: Tuple[Tuple[int, int]],
                            train: bool) -> jnp.ndarray:
  """Perform deformable attention sampling calculation.

  Given values, sampling locations and attention weights calculate the attention
  weighted values using given sampling locations as described in [1]. This
  function is mainly focused on doing the interpolated sampling from the
  values and then multiplying them by the given attention weights and
  replaces the custom CUDA kernel from pytorch implementation [2].

  Args:
    value: [bs, len_v, nheads, nembed]-ndarray where nembed is embed per head.
    sampling_locations: [bs, len_q, nheads, nlevels, npoints, 2]-ndarray of 2D
      points for all len_q x nheads x nlevels combinations.
    attn_weights: [bs, len_q, nheads, nlevels, npoints]-ndarray attention
      weights that have already been calculated.
    shapes: Static tuple of image shapes for each level to unflatten len_v.
    train: Whether we are in training mode.

  Returns:
    [bs * nheads, 1, len_q, nlevels * npoints]-ndarray values weighted by the
    attention weights using the deformable locations.
  """
  _, _, _, nembed = value.shape
  bs, len_q, nheads, nlevels, npoints, _ = sampling_locations.shape
  split_indices = np.cumsum(np.array([h * w for h, w in shapes]))[:-1]
  # Split values by level.
  value_by_level = value.split(split_indices, axis=1)

  sampled_values_by_level = []
  for level_idx, (im_h, im_w) in enumerate(shapes):
    # bs, im_h * im_w, nheads, nembed -> bs * nheads, nembed, im_h, im_w
    v = value_by_level[level_idx]
    v = (
        v.reshape(v.shape[:2] + (-1,)).transpose(0, 2, 1).reshape(
            bs * nheads, nembed, im_h, im_w))

    # bs, len_q, nheads, npoints, 2 -> bs * nheads, len_q, npoints, 2
    s = sampling_locations[:, :, :, level_idx].transpose(0, 2, 1, 3, 4)
    s = s.reshape((-1,) + s.shape[2:])

    if train:
      attention_fn = jax.remat(
          bilinear_interpolate,
          policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
          static_argnums=2)
    else:
      attention_fn = bilinear_interpolate

    sampled_values = jax.lax.map(
        lambda i: attention_fn(v[i], s[i], False),  # pylint: disable=cell-var-from-loop
        jnp.array(range(v.shape[0])))
    sampled_values_by_level.append(sampled_values)

  # bs, len_q, nheads, nlevels, npoints ->
  # bs * nheads, 1, len_q, nlevels * npoints
  attn_weights = (
      attn_weights.transpose(0, 2, 1, 3, 4).reshape(bs * nheads, 1, len_q,
                                                    nlevels * npoints))

  out = jnp.stack(sampled_values_by_level, axis=-2)
  out = out.reshape(out.shape[:-2] + (-1,)) * attn_weights
  out = out.sum(-1).reshape(bs, nheads * nembed, len_q)
  out = out.transpose(0, 2, 1)

  return out


def pos_grid_init(num_heads: int, num_levels: int,
                  num_points: int) -> Callable[..., jnp.ndarray]:
  """Initialize deformable attention sampling offsets as grid.

  Args:
    num_heads: Number of attention heads.
    num_levels: Number of feature levels.
    num_points: Number of reference points.

  Returns:
    Init function for sampling offset weights.
  """

  def _pos_grid_init(key, shape, dtype=jnp.float32):
    del key
    thetas = jnp.arange(num_heads, dtype=dtype) * (2 * math.pi) / num_heads
    grid_init = jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], -1)
    denom = jnp.max(jnp.abs(grid_init), axis=-1, keepdims=True)
    grid_init = (grid_init / denom)
    grid_init = grid_init.reshape(num_heads, 1, 1, 2)
    grid_init = jnp.tile(grid_init, (1, num_levels, num_points, 1))
    for i in range(num_points):
      grid_i = grid_init[:, :, i, :] * (i + 1)
      grid_init = grid_init.at[:, :, i, :].set(grid_i)
    return grid_init.reshape(shape)

  return _pos_grid_init


class MultiScaleDeformableAttention(nn.Module):
  """Layer for MultiScaleDeformableAttention."""
  spatial_shapes: Sequence[Tuple[int, int]]
  embed_dim: int = 256
  num_levels: int = 4
  num_heads: int = 8
  num_points: int = 4
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, query: jnp.ndarray, ref_points: jnp.ndarray,
               value: jnp.ndarray, pad_mask: jnp.ndarray,
               train: bool) -> jnp.ndarray:
    """Calculate multi-scale multi-head deformable attention.

    Args:
      query: [bs, len_q, embed_dim]-ndarray of queries.
      ref_points: [bs, len_q, num_levels, box_dim]-ndarray of reference points
        for each query. box_dim is in {2, 4} as ref_points can either be box
        cxcy or cxcywh.
      value: [bs, len_v, embed_dim]-ndarray of queries.
      pad_mask: [bs, len_v]-ndarray of boolean values, where 0 indicates pad.
      train: Whether we are in training mode.

    Returns:
      Attention weighted values based on the queries and reference points
      relevant to the queries.
    """
    assert self.embed_dim % self.num_heads == 0, ('embed_dim must be divisible '
                                                  'by num_heads.')
    ref_dim = ref_points.shape[-1]
    assert ref_dim in {2, 4}, f'Unexpected ref_points dim: {ref_dim}'

    # Embed dim by head.
    c = self.embed_dim // self.num_heads

    bs, len_q, _ = query.shape
    _, len_v, _ = value.shape
    assert np.array(self.spatial_shapes).prod(
        axis=1).sum() == len_v, 'Shapes must match flattened dim.'

    # Value projection.
    value = nn.Dense(self.embed_dim, dtype=self.dtype, name='value_proj')(value)
    # Zero out values for padded pixels.
    value = jnp.where(~pad_mask[..., None], 0, value)
    value = value.reshape(bs, len_v, self.num_heads, c)

    # Sampling offsets embedding.
    num_total_points = self.num_heads * self.num_levels * self.num_points
    sampling_offsets = nn.Dense(
        num_total_points * 2,
        kernel_init=nn.initializers.zeros,
        bias_init=pos_grid_init(self.num_heads, self.num_levels,
                                self.num_points),
        name='sampling_offsets')(
            query)
    sampling_offsets = sampling_offsets.reshape(bs, len_q, self.num_heads,
                                                self.num_levels,
                                                self.num_points, 2)

    # Attention weights embedding.
    attn_weights = nn.Dense(
        num_total_points,
        kernel_init=nn.initializers.zeros,
        name='attention_weights')(
            query)
    attn_weights = attn_weights.reshape(bs, len_q, self.num_heads,
                                        self.num_levels * self.num_points)
    attn_weights = nn.softmax(
        attn_weights, axis=-1).reshape(bs, len_q, self.num_heads,
                                       self.num_levels, self.num_points)

    # Calculate the actual sampling locations.
    if ref_dim == 4:
      ref_xy = ref_points[:, :, None, :, None, :2]
      ref_wh = ref_points[:, :, None, :, None, 2:]
      normalized_offsets = sampling_offsets / self.num_points * ref_wh * 0.5
    else:
      ref_xy = ref_points[:, :, None, :, None, :]
      arr_shapes = np.array(self.spatial_shapes)
      offset_norm = np.stack([arr_shapes[:, 1], arr_shapes[:, 0]], -1)
      normalized_offsets = sampling_offsets / offset_norm[None, None, None, :,
                                                          None, :]
    sampling_locations = ref_xy + normalized_offsets

    x = deform_attn_sampling_fn(value, sampling_locations, attn_weights,
                                self.spatial_shapes, train)

    x = nn.Dense(self.embed_dim, name='output_proj')(x)
    return x
