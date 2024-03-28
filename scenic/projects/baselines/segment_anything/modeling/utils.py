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

"""Util functions for Segment Anything models."""

import jax.numpy as jnp
import numpy as np
from scenic.projects.baselines.segment_anything.modeling import nms as nms_lib


def build_point_grid(points_per_side):
  """Generates a 2D grid of points evenly spaced in [0, 1] x [0, 1]."""
  offset = 1. / (2 * points_per_side)
  points_one_side = jnp.linspace(offset, 1 - offset, points_per_side)
  points_x = jnp.tile(points_one_side[None, :], (points_per_side, 1))
  points_y = jnp.tile(points_one_side[:, None], (1, points_per_side))
  points = jnp.stack([points_x, points_y], axis=-1).reshape(-1, 2)
  return points  # (points_per_side ** 2, 1)


def batched_mask_to_box(masks):
  """Convert binary masks in (n, h, w) to boxes (n, 4)."""
  if masks.shape[0] == 0:
    return jnp.zeros((0, 4), dtype=jnp.float32)

  h, w = masks.shape[-2:]
  in_height = jnp.max(masks, axis=-1)  # (n, h)
  in_height_coords = in_height * jnp.arange(h)[None]  # (n, h)
  bottom_edges = jnp.max(in_height_coords, axis=-1)  # (n, )
  # Mark "0" as "h" so that we can take min.
  in_height_coords = in_height_coords + h * (1 - in_height)  # (n, h)
  top_edges = jnp.min(in_height_coords, axis=-1)  # (n,)

  in_width = jnp.max(masks, axis=-2)  # (n, w)
  in_width_coords = in_width * jnp.arange(w)[None]  # (n, w)
  right_edges = jnp.max(in_width_coords, axis=-1)  # (n,)
  in_width_coords = in_width_coords + w * (1 - in_width)  # (n, w)
  left_edges = jnp.min(in_width_coords, axis=-1)

  # mark empty mask as [0, 0, 0, 0]
  empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
  out = jnp.stack(
      [left_edges, top_edges, right_edges, bottom_edges], axis=-1)  # (n, 4)
  out = out * (1 - empty_filter)[:, None]
  return out


def batched_mask_to_box_np(masks):
  """Convert binary masks in (n, h, w) to boxes (n, 4)."""
  if masks.shape[0] == 0:
    return np.zeros((0, 4), dtype=np.float32)

  h, w = masks.shape[-2:]
  in_height = np.max(masks, axis=-1)  # (n, h)
  in_height_coords = in_height * np.arange(h)[None]  # (n, h)
  bottom_edges = np.max(in_height_coords, axis=-1)  # (n, )
  # Mark "0" as "h" so that we can take min.
  in_height_coords = in_height_coords + h * (1 - in_height)  # (n, h)
  top_edges = np.min(in_height_coords, axis=-1)  # (n,)

  in_width = np.max(masks, axis=-2)  # (n, w)
  in_width_coords = in_width * np.arange(w)[None]  # (n, w)
  right_edges = np.max(in_width_coords, axis=-1)  # (n,)
  in_width_coords = in_width_coords + w * (1 - in_width)  # (n, w)
  left_edges = np.min(in_width_coords, axis=-1)

  # mark empty mask as [0, 0, 0, 0]
  empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
  out = np.stack(
      [left_edges, top_edges, right_edges, bottom_edges], axis=-1)  # (n, 4)
  out = out * (1 - empty_filter)[:, None]
  return out


def calculate_stability_score(
    mask_logits, mask_threshold, stability_score_offset):
  """The stability score measures if the mask changes with different thresh."""
  low = (mask_logits > (mask_threshold + stability_score_offset)).sum(
      axis=-1).sum(axis=-1)
  high = (mask_logits > (mask_threshold - stability_score_offset)).sum(
      axis=-1).sum(axis=-1)
  return low / high


def nms(boxes, scores, iou_threshold, num_outputs=100):
  _, _, keep = nms_lib.non_max_suppression_padded(
      scores[None], boxes[None], num_outputs, iou_threshold,
      return_idx=True)  # pytype: disable=wrong-arg-types
  return keep[0]  # undo batch
