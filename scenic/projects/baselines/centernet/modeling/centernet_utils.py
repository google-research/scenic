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

"""Util functions for centernet."""

import functools
from typing import Any, List, Tuple

import jax
import jax.numpy as jnp

INF = 100000000
Array = jnp.ndarray


def box_iou(boxes1: Array, boxes2: Array) -> Tuple[Array, Array]:
  """Compute box IoU. Boxes in format [-l, -t, b, r].

  Args:
    boxes1: array in shape B x n x 4 or n x 4
    boxes2: array in shape B x m x 4 or m x 4
  Returns:
    iou: array in shape B x n x m or n x m
    union: array in shape B x n x m or n x m
  """
  wh1 = boxes1[..., 2:] + boxes1[..., :2]
  area1 = wh1[..., 0] * wh1[..., 1]  # [bs, n]
  wh2 = boxes2[..., 2:] + boxes2[..., :2]
  area2 = wh2[..., 0] * wh2[..., 1]  # [bs, m]
  lt = jnp.maximum(- boxes1[..., :, :2], - boxes2[..., :, :2])  # [bs, n, 2]
  rb = jnp.minimum(boxes1[..., :, 2:], boxes2[..., :, 2:])  # [bs, n, 2]
  wh = (rb - lt).clip(0.0)  # [bs, n, 2]
  intersection = wh[..., :, 0] * wh[..., :, 1]  # [bs, n]
  union = area1 + area2 - intersection
  iou = intersection / (union + 1e-6)
  return iou, union


def giou_loss(boxes1: Array, boxes2: Array) -> Array:
  """Compute GIoU loss. Boxes in format [-l, -t, b, r].

  Args:
    boxes1: array in shape B x n x 4 or n x 4
    boxes2: array in shape B x m x 4 or m x 4
  Returns:
    array in shape B x n
  """
  iou, union = box_iou(boxes1, boxes2)
  lt = jnp.minimum(- boxes1[..., :, :2], - boxes2[..., :, :2])  # [bs, n, 2]
  rb = jnp.maximum(boxes1[..., :, 2:], boxes2[..., :, 2:])  # [bs, n, 2]
  wh = (rb - lt).clip(0.0)  # [bs, n, 2]
  area = wh[..., 0] * wh[..., 1]  # [bs, n]
  giou = iou - (area - union) / (area + 1e-6)
  return 1. - giou


def get_grid(shapes: List[Tuple[int, int]], strides: Any) -> List[Array]:
  """Generate the default locations of each output pixels.

  Args:
    shapes: list of (hl, wl) tuples in FPN levels, with length L (number of
      FPN levels). l is the FPN level index. In general hl == h // (2**l) where
      h is the original input size.
    strides: list of integers. The strides in each FPN level.

  Returns:
    grids: List of arrays with length L, each in shape (hl * wl, 2).
  """
  grids = []
  for l, (h, w) in enumerate(shapes):
    shifts_x = jnp.arange(
        0, w * strides[l], step=strides[l], dtype=jnp.float32)
    shifts_y = jnp.arange(
        0, h * strides[l], step=strides[l], dtype=jnp.float32)
    shift_x, shift_y = jnp.meshgrid(shifts_x, shifts_y)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    grids_per_level = jnp.stack(
        (shift_x, shift_y), axis=1) + strides[l] // 2
    grids.append(grids_per_level)
  return grids


def naive_create_heatmaps(
    dist: Array, labels: Array, num_classes: int) -> Array:
  """create per-class heatmaps from normalized distances.

  Args:
    dist: arrays in shape m x n. m is the sum pixels in all levels, n is
      the number of objects.
    labels: int arrays in shape n: the class index (int range [0, C - 1])
    num_classes: C or 0. 0 for class agnostic.

  Returns:
    array in shape m x C or m x 1 (agnostic): the CenterNet heatmaps.
  """
  # output a single-channel heatmap when num_classes == 0 (agnostic)
  out_channels = max(num_classes, 1)
  heatmap = jnp.zeros((out_channels, dist.shape[0]), dtype=jnp.float32)
  dist = jnp.exp(-dist)  # m x n
  for i in range(labels.shape[0]):
    heatmap = heatmap.at[labels[i]].set(
        jnp.maximum(dist[:, i], heatmap[labels[i]]))
  heatmap = heatmap.transpose(1, 0)  # m x C
  return heatmap


def scatter_max(inp, index, src):
  """Jax implementation of torch.scatter(inp, 1, index, src)."""
  # from https://github.com/google/jax/issues/8487
  dnums = jax.lax.ScatterDimensionNumbers(
      update_window_dims=(), inserted_window_dims=(0,),
      scatter_dims_to_operand_dims=(0,))
  scatter = functools.partial(jax.lax.scatter_max, dimension_numbers=dnums)
  scatter = jax.vmap(scatter, in_axes=(0, 0, 0), out_axes=0)
  return scatter(inp, jnp.expand_dims(index, axis=-1), src)


def create_heatmaps(dist: Array, labels: Array, num_classes: int) -> Array:
  """create heatmaps.

  This is equivalent to "naive_create_heatmaps" but runs faster.

  Args:
    dist: arrays in shape m x n. m is the sum pixels in all levels, n is
      the number of objects. The weighted distance (Y_{xyc} in Eq. 1 in paper
      https://arxiv.org/pdf/1904.07850.pdf) between the m output pixels
      and n object centers.
    labels: int arrays in shape n: the class label (int range [0, C - 1]) of
      each ground truth object.
    num_classes: C or 0. 0 for class agnostic.

  Returns:
    array in shape m x C or m x 1 (agnostic): the CenterNet heatmaps.
  """
  # output a single-channel heatmap when num_classes == 0 (agnostic)
  out_channels = max(num_classes, 1)
  heatmap = jnp.zeros((dist.shape[0], out_channels), dtype=jnp.float32)
  heatmap = scatter_max(
      heatmap,
      jnp.broadcast_to(labels[None], (dist.shape[0], dist.shape[1])),
      jnp.exp(-dist))
  return heatmap


def get_center3x3(grids_expanded: Array, centers_discret: Array,
                  strides_expanded: Array) -> Array:
  """Get the 3x3 regions near each discret centers for regression.

  Args:
    grids_expanded: arrays in shape m x n x 2. m is the sum pixels in all
      levels, n is the number of objects.
    centers_discret: arrays in shape m x n x 2
    strides_expanded: arrays in shape m x n x 2

  Returns:
    bool array in shape m x n: if a pixel is within the 3x3 region of an object
      in any of the fpn level.
  """
  dist_x = jnp.absolute(grids_expanded[:, :, 0] - centers_discret[:, :, 0])
  dist_y = jnp.absolute(grids_expanded[:, :, 1] - centers_discret[:, :, 1])
  return (dist_x <= strides_expanded[:, :, 0]) & (
      dist_y <= strides_expanded[:, :, 0])


def assign_fpn_level(reg_target: Array, fpn_range: Array) -> Array:
  """Assign each ground truth object to its FPN level.

  Args:
    reg_target: array in shape m x n x 4. m is the sum pixels in all levels,
      n is the number of objects.
    fpn_range: array in shape m x 2: the range of each pixel.
  Returns:
    a bool array in shape m x n
  """
  diag_length = ((reg_target[:, :, :2] + reg_target[:, :, 2:]) ** 2).sum(
      axis=2) ** 0.5 / 2  # m x n, where all values are the same in m
  is_cared_in_fpn_level = (diag_length >= fpn_range[:, None, 0]) & (
      diag_length <= fpn_range[:, None, 1])  # m x n
  return is_cared_in_fpn_level


def get_reg_targets(reg_target: Array, dist: Array, mask: Array) -> Array:
  """Assign regression gts. Each pixel regress to its "closest" valid object.

  Args:
    reg_target: array in shape m x n x 4. m is the sum pixels in all levels,
      n is the number of objects.
    dist: array in shape m x n: the weighted distance between pixels and objects
      defined in the heatmap.
    mask: bool array in shape m x n: if assign the pixel is valid of the object.
  Returns:
    regs: array in shape m x 4: the regression target of each pixel.
  """
  dist = dist * mask + (1. - mask) * INF  # m x n
  min_dist, min_inds = dist.min(axis=1), dist.argmin(axis=1)  # m
  regs = reg_target[jnp.arange(len(reg_target)), min_inds]  # m x n x 4 -> m x 4
  invalid = (min_dist == INF)  # m
  regs = regs * (1. - invalid[:, None]) - 1. * INF * invalid[:, None]  # m x 4
  return regs


def level_first_to_batch_first(preds) -> Array:
  """Concatenate features from different FPN level.

  Args:
    preds: list of arrays: L x [B, hl, wl, D]. B is the batch size, hl * wl
      is the numbers of pixels in the FPN level, D is the feature dimention.
  Returns:
    array in shape B x m x D, m = sum_l hl * wl
  """
  return jnp.concatenate(
      [x.reshape(x.shape[0], -1, x.shape[-1]) for x in preds], axis=1)
