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

"""Utilities for boxes.

Axis-aligned utils implemented based on:
https://github.com/facebookresearch/detr/blob/master/util/box_ops.py.

Rotated box utils implemented based on:
https://github.com/lilanxiao/Rotated_IoU.
"""
from typing import Any, Union

import jax.numpy as jnp
import numpy as np

PyModule = Any
Array = Union[jnp.ndarray, np.ndarray]


def box_cxcywh_to_xyxy(x: Array, np_backbone: PyModule = jnp) -> Array:
  """Converts boxes from [cx, cy, w, h] format into [x, y, x', y'] format."""
  x_c, y_c, w, h = np_backbone.split(x, 4, axis=-1)
  b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
  return np_backbone.concatenate(b, axis=-1)


def box_cxcywh_to_yxyx(x: Array, np_backbone: PyModule = jnp) -> Array:
  """Converts boxes from [cx, cy, w, h] format into [y, x, y', x'] format."""
  x_c, y_c, w, h = np_backbone.split(x, 4, axis=-1)
  b = [(y_c - 0.5 * h), (x_c - 0.5 * w), (y_c + 0.5 * h), (x_c + 0.5 * w)]
  return np_backbone.concatenate(b, axis=-1)


def box_xyxy_to_cxcywh(x: Array, np_backbone: PyModule = jnp) -> Array:
  """Converts boxes from [x, y, x', y'] format into [cx, cy, w, h] format."""
  x0, y0, x1, y1 = np_backbone.split(x, 4, axis=-1)
  b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
  return np_backbone.concatenate(b, axis=-1)


def box_yxyx_to_cxcywh(x: Array, np_backbone: PyModule = jnp) -> Array:
  """Converts boxes from [y, x, y', x'] format into [cx, cy, w, h] format."""
  y0, x0, y1, x1 = np_backbone.split(x, 4, axis=-1)
  b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
  return np_backbone.concatenate(b, axis=-1)


def box_iou(boxes1: Array,
            boxes2: Array,
            np_backbone: PyModule = jnp,
            all_pairs: bool = True,
            eps: float = 1e-6) -> Array:
  """Computes IoU between two sets of boxes.

  Boxes are in [x, y, x', y'] format [x, y] is top-left, [x', y'] is bottom
  right.

  Args:
    boxes1: Predicted bounding-boxes in shape [bs, n, 4].
    boxes2: Target bounding-boxes in shape [bs, m, 4]. Can have a different
      number of boxes if all_pairs is True.
    np_backbone: numpy module: Either the regular numpy package or jax.numpy.
    all_pairs: Whether to compute IoU between all pairs of boxes or not.
    eps: Epsilon for numerical stability.

  Returns:
    If all_pairs == True, returns the pairwise IoU cost matrix of shape
    [bs, n, m]. If all_pairs == False, returns the IoU between corresponding
    boxes. The shape of the return value is then [bs, n].
  """

  # First, compute box areas. These will be used later for computing the union.
  wh1 = boxes1[..., 2:] - boxes1[..., :2]
  area1 = wh1[..., 0] * wh1[..., 1]  # [bs, n]

  wh2 = boxes2[..., 2:] - boxes2[..., :2]
  area2 = wh2[..., 0] * wh2[..., 1]  # [bs, m]

  if all_pairs:
    # Compute pairwise top-left and bottom-right corners of the intersection
    # of the boxes.
    lt = np_backbone.maximum(boxes1[..., :, None, :2],
                             boxes2[..., None, :, :2])  # [bs, n, m, 2].
    rb = np_backbone.minimum(boxes1[..., :, None, 2:],
                             boxes2[..., None, :, 2:])  # [bs, n, m, 2].

    # intersection = area of the box defined by [lt, rb]
    wh = (rb - lt).clip(0.0)  # [bs, n, m, 2]
    intersection = wh[..., 0] * wh[..., 1]  # [bs, n, m]

    # union = sum of areas - intersection
    union = area1[..., :, None] + area2[..., None, :] - intersection

    iou = intersection / (union + eps)

  else:
    # Compute top-left and bottom-right corners of the intersection between
    # corresponding boxes.
    assert boxes1.shape[1] == boxes2.shape[1], (
        'Different number of boxes when all_pairs is False')
    lt = np_backbone.maximum(boxes1[..., :, :2],
                             boxes2[..., :, :2])  # [bs, n, 2]
    rb = np_backbone.minimum(boxes1[..., :, 2:], boxes2[..., :,
                                                        2:])  # [bs, n, 2]

    # intersection = area of the box defined by [lt, rb]
    wh = (rb - lt).clip(0.0)  # [bs, n, 2]
    intersection = wh[..., :, 0] * wh[..., :, 1]  # [bs, n]

    # union = sum of areas - intersection.
    union = area1 + area2 - intersection

    # Somehow the PyTorch implementation does not use eps to avoid 1/0 cases.
    iou = intersection / (union + eps)

  return iou, union  # pytype: disable=bad-return-type  # jax-ndarray


def generalized_box_iou(boxes1: Array,
                        boxes2: Array,
                        np_backbone: PyModule = jnp,
                        all_pairs: bool = True,
                        eps: float = 1e-6) -> Array:
  """Generalized IoU from https://giou.stanford.edu/.

  The boxes should be in [x, y, x', y'] format specifying top-left and
  bottom-right corners.

  Args:
    boxes1: Predicted bounding-boxes in shape [..., n, 4].
    boxes2: Target bounding-boxes in shape [..., m, 4].
    np_backbone: Numpy module: Either the regular numpy package or jax.numpy.
    all_pairs: Whether to compute generalized IoU from between all-pairs of
      boxes or not. Note that if all_pairs == False, we must have m==n.
    eps: Epsilon for numerical stability.

  Returns:
    If all_pairs == True, returns a [bs, n, m] pairwise matrix, of generalized
    ious. If all_pairs == False, returns a [bs, n] matrix of generalized ious.
  """
  # Degenerate boxes gives inf / nan results, so do an early check.
  # TODO(b/166344282): Figure out how to enable asserts on inputs with jitting:
  # assert (boxes1[:, :, 2:] >= boxes1[:, :, :2]).all()
  # assert (boxes2[:, :, 2:] >= boxes2[:, :, :2]).all()
  iou, union = box_iou(
      boxes1, boxes2, np_backbone=np_backbone, all_pairs=all_pairs, eps=eps)

  # Generalized IoU has an extra term which takes into account the area of
  # the box containing both of these boxes. The following code is very similar
  # to that for computing intersection but the min and max are flipped.
  if all_pairs:
    lt = np_backbone.minimum(boxes1[..., :, None, :2],
                             boxes2[..., None, :, :2])  # [bs, n, m, 2]
    rb = np_backbone.maximum(boxes1[..., :, None, 2:],
                             boxes2[..., None, :, 2:])  # [bs, n, m, 2]

  else:
    lt = np_backbone.minimum(boxes1[..., :, :2],
                             boxes2[..., :, :2])  # [bs, n, 2]
    rb = np_backbone.maximum(boxes1[..., :, 2:], boxes2[..., :,
                                                        2:])  # [bs, n, 2]

  # Now, compute the covering box's area.
  wh = (rb - lt).clip(0.0)  # Either [bs, n, 2] or [bs, n, m, 2].
  area = wh[..., 0] * wh[..., 1]  # Either [bs, n] or [bs, n, m].

  # Finally, compute generalized IoU from IoU, union, and area.
  # Somehow the PyTorch implementation does not use eps to avoid 1/0 cases.
  return iou - (area - union) / (area + eps)


### Rotated Box Utilties ###


def cxcywha_to_corners(cxcywha: Array, np_backbone: PyModule = jnp) -> Array:
  """Convert [cx, cy, w, h, a] to four corners of [x, y].

  Args:
    cxcywha: [..., 5]-ndarray of [center-x, center-y, width, height, angle]
    representation of rotated boxes. Angle is in radians and center of rotation
    is defined by [center-x, center-y] point.
    np_backbone: Numpy module: Either the regular numpy package or jax.numpy.

  Returns:
    [..., 4, 2]-ndarray of four corners of the rotated box as [x, y] points.
  """
  assert cxcywha.shape[-1] == 5, 'Expected [..., [cx, cy, w, h, a] input.'
  bs = cxcywha.shape[:-1]
  cx, cy, w, h, a = np_backbone.split(cxcywha, indices_or_sections=5, axis=-1)
  xs = np_backbone.array([.5, .5, -.5, -.5]) * w
  ys = np_backbone.array([-.5, .5, .5, -.5]) * h
  pts = np_backbone.stack([xs, ys], axis=-1)
  sin = np_backbone.sin(a)
  cos = np_backbone.cos(a)
  rot = np_backbone.concatenate([cos, -sin, sin, cos], axis=-1).reshape(
      (*bs, 2, 2))
  offset = np_backbone.concatenate([cx, cy], -1).reshape((*bs, 1, 2))
  corners = pts @ rot + offset
  return corners


def corners_to_cxcywha(corners: jnp.ndarray,
                       np_backbone: PyModule = jnp) -> jnp.ndarray:
  """Convert four corners of [x, y] to [cx, cy, w, h, a].

  Although the conversion is only guaranteed to produce an exact rbox when given
  vertices that form an rbox, there is some graceful handling of nearly rbox
  vertices by choosing the rbox with corners minimizing the square distance to
  the rbox vertices. This solution is equivalent to taking the average of the
  top and bottom edges (wcorners*) as well as the left and right edges
  (hcornersy).

  Args:
    corners: [..., 4, 2]-ndarray of four corners of the rotated box as [x, y]
      points.
    np_backbone: Numpy module: Either the regular numpy package or jax.numpy.

  Returns:
    [..., 5]-ndarray of [center-x, center-y, width, height, angle]
    representation of rotated boxes. Angle is in radians and center of rotation
    is defined by [center-x, center-y] point.
  """
  assert corners.shape[-2] == 4 and corners.shape[-1] == 2, (
      'Expected four corners [..., 4, 2] input.')

  cornersx, cornersy = corners[..., 0], corners[..., 1]
  cx = np_backbone.mean(cornersx, axis=-1)
  cy = np_backbone.mean(cornersy, axis=-1)
  wcornersx = (
      cornersx[..., 0] + cornersx[..., 1] - cornersx[..., 2] - cornersx[..., 3])
  wcornersy = (
      cornersy[..., 0] + cornersy[..., 1] - cornersy[..., 2] - cornersy[..., 3])
  hcornersy = (-cornersy[..., 0,] + cornersy[..., 1] + cornersy[..., 2] -
               cornersy[..., 3])
  a = -np_backbone.arctan2(wcornersy, wcornersx)
  cos = np_backbone.cos(a)
  w = wcornersx / (2 * cos)
  h = hcornersy / (2 * cos)
  cxcywha = np_backbone.stack([cx, cy, w, h, a], axis=-1)

  return cxcywha


def intersect_line_segments(
    lines1: jnp.ndarray, lines2: jnp.ndarray, eps: float = 1e-8
) -> jnp.ndarray:
  """Intersect two line segments.

  Given two 2D line segments, where a line segment is defined as two 2D points.
  Finds the point of intersection or returns [nan, nan] if no point exists.

  See https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection (Given two
  points on each line segment).

  Performance Note: At the calling point, we expect user to appropriately vmap
  function to work on batches of lines.
  Args:
    lines1: [..., 2, 2]-ndarray, [[x1, y1], [x2, y2]] for lines.
    lines2: [..., 2, 2]-ndarray, [[x3, y3], [x4, y4]] for other lines.
    eps: Epsilon for numerical stability.

  Returns:
    Intersection points [..., 2]-ndarray or [..., [nan, nan]] if no point
    exists. Since we are intersecting line segments in 2D, this happens if
    lines are parallel or the intersection of the infinite line would occur
    outside of both segments.
  """
  assert lines1.shape[-2:] == (2, 2) and lines2.shape[-2:] == (2, 2)
  x1, y1 = jnp.split(lines1[..., 0, :], 2, -1)
  x2, y2 = jnp.split(lines1[..., 1, :], 2, -1)
  x3, y3 = jnp.split(lines2[..., 0, :], 2, -1)
  x4, y4 = jnp.split(lines2[..., 1, :], 2, -1)
  den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
  num_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
  num_u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)
  # t and u are parameterizations of line1 and line2 respectively and are left
  # as variable names from the original algorithm documentation.
  t = num_t / (den + eps)
  u = -num_u / (den + eps)

  intersection_pt = jnp.concatenate([x1 + t * (x2 - x1), y1 + t * (y2 - y1)],
                                    -1)
  are_parallel = jnp.abs(den) < eps
  not_on_line1 = jnp.logical_or(u < 0, u > 1)
  not_on_line2 = jnp.logical_or(t < 0, t > 1)

  not_possible = jnp.any(
      jnp.concatenate([are_parallel, not_on_line1, not_on_line2], -1), -1)
  nan_pt = jnp.ones_like(intersection_pt) * jnp.nan
  return jnp.where(not_possible[..., None], nan_pt, intersection_pt)


def intersect_rbox_edges(corners1: jnp.ndarray,
                         corners2: jnp.ndarray) -> jnp.ndarray:
  """Find intersection points between all four edges of both rotated boxes.

  Note that you are expected to explicitly use vmap to control batching.

  Args:
    corners1: (4, 2)-ndarray of corners for rbox1.
    corners2: (4, 2)-ndarray of corners for rbox2.

  Returns:
    intersections: (4, 4, 2)-ndarray (i, j, :) means intersection of i-th
    edge of rbox1 with j-th of rbox2.
  """
  intersections = []
  # Apparently for-loop is 2-4x faster than vectorized implementation on TPU
  # because it has much higher memory bandwidth. On GPU, the for-loop
  # implementation is 1.5x slower than vectorized.
  for i in range(4):
    line1 = jnp.stack([corners1[i, :], corners1[(i + 1) % 4, :]], axis=0)
    for j in range(4):
      line2 = jnp.stack([corners2[j, :], corners2[(j + 1) % 4, :]], axis=0)
      intersections.append(intersect_line_segments(line1, line2))
  intersections = jnp.reshape(jnp.stack(intersections), (4, 4, 2))
  return intersections
