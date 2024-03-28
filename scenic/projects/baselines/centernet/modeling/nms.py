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

r"""Implement NMS functions.

Jax NMS is forked from
https://github.com/mlperf/training_results_v0.7/blob/master/Google/benchmarks/\
ssd/implementations/ssd-research-JAX-tpu-v3-4096/nms.py
"""

from typing import Tuple
import jax
import jax.numpy as jnp
import numpy as np

_NMS_TILE_SIZE = 256


def _bbox_overlap(boxes: jnp.ndarray, gt_boxes: jnp.ndarray):
  """Find Bounding box overlap.

  Args:
    boxes: first set of bounding boxes
    gt_boxes: second set of boxes to compute IOU

  Returns:
    iou: Intersection over union matrix of all input bounding boxes
  """
  bb_y_min, bb_x_min, bb_y_max, bb_x_max = jnp.split(
      ary=boxes, indices_or_sections=4, axis=2)
  gt_y_min, gt_x_min, gt_y_max, gt_x_max = jnp.split(
      ary=gt_boxes, indices_or_sections=4, axis=2)

  # Calculates the intersection area.
  i_xmin = jnp.maximum(bb_x_min, jnp.transpose(gt_x_min, [0, 2, 1]))
  i_xmax = jnp.minimum(bb_x_max, jnp.transpose(gt_x_max, [0, 2, 1]))
  i_ymin = jnp.maximum(bb_y_min, jnp.transpose(gt_y_min, [0, 2, 1]))
  i_ymax = jnp.minimum(bb_y_max, jnp.transpose(gt_y_max, [0, 2, 1]))
  i_area = jnp.maximum((i_xmax - i_xmin), 0) * jnp.maximum((i_ymax - i_ymin), 0)

  # Calculates the union area.
  bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
  gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
  # Adds a small epsilon to avoid divide-by-zero.
  u_area = bb_area + jnp.transpose(gt_area, [0, 2, 1]) - i_area + 1e-8

  # Calculates IoU.
  iou = i_area / u_area

  return iou


def _self_suppression(in_args):
  iou, _, iou_sum = in_args
  batch_size = iou.shape[0]
  can_suppress_others = jnp.reshape(
      jnp.max(iou, 1) <= 0.5, [batch_size, -1, 1]).astype(iou.dtype)
  iou_suppressed = jnp.reshape(
      (jnp.max(can_suppress_others * iou, 1) <= 0.5).astype(iou.dtype),
      [batch_size, -1, 1]) * iou
  iou_sum_new = jnp.sum(iou_suppressed, [1, 2])
  return iou_suppressed, jnp.any(iou_sum - iou_sum_new > 0.5), iou_sum_new


def _cross_suppression(in_args):
  boxes, box_slice, iou_threshold, inner_idx = in_args
  batch_size = boxes.shape[0]
  new_slice = jax.lax.dynamic_slice(
      boxes, [0, inner_idx * _NMS_TILE_SIZE, 0],
      [batch_size, _NMS_TILE_SIZE, 4])
  iou = _bbox_overlap(new_slice, box_slice)
  ret_slice = jnp.expand_dims(
      (jnp.all(iou < iou_threshold, [1])).astype(box_slice.dtype),
      2) * box_slice
  return boxes, ret_slice, iou_threshold, inner_idx + 1


def _suppression_loop_body(in_args):
  """Process boxes in the range [idx*_NMS_TILE_SIZE, (idx+1)*_NMS_TILE_SIZE).

  Args:
    in_args: A tuple of arguments: boxes, iou_threshold, output_size, idx

  Returns:
    boxes: updated boxes.
    iou_threshold: pass down iou_threshold to the next iteration.
    output_size: the updated output_size.
    idx: the updated induction variable.
  """
  boxes, iou_threshold, output_size, idx = in_args
  num_tiles = boxes.shape[1] // _NMS_TILE_SIZE
  batch_size = boxes.shape[0]

  # Iterates over tiles that can possibly suppress the current tile.
  box_slice = jax.lax.dynamic_slice(
      boxes, [0, idx * _NMS_TILE_SIZE, 0],
      [batch_size, _NMS_TILE_SIZE, 4])
  def _loop_cond(in_args):
    _, _, _, inner_idx = in_args
    return inner_idx < idx

  _, box_slice, _, _ = jax.lax.while_loop(
      _loop_cond,
      _cross_suppression, (boxes, box_slice, iou_threshold,
                           0))

  # Iterates over the current tile to compute self-suppression.
  iou = _bbox_overlap(box_slice, box_slice)
  mask = jnp.expand_dims(
      jnp.reshape(jnp.arange(_NMS_TILE_SIZE), [1, -1]) > jnp.reshape(
          jnp.arange(_NMS_TILE_SIZE), [-1, 1]), 0)
  iou *= (jnp.logical_and(mask, iou >= iou_threshold)).astype(iou.dtype)

  def _loop_cond2(in_args):
    _, loop_condition, _ = in_args
    return loop_condition

  suppressed_iou, _, _ = jax.lax.while_loop(
      _loop_cond2, _self_suppression,
      (iou, True,
       jnp.sum(iou, [1, 2])))
  suppressed_box = jnp.sum(suppressed_iou, 1) > 0
  box_slice *= jnp.expand_dims(1.0 - suppressed_box.astype(box_slice.dtype), 2)

  # Uses box_slice to update the input boxes.
  mask = jnp.reshape(
      (jnp.equal(jnp.arange(num_tiles), idx)).astype(boxes.dtype),
      [1, -1, 1, 1])
  boxes = jnp.tile(jnp.expand_dims(
      box_slice, 1), [1, num_tiles, 1, 1]) * mask + jnp.reshape(
          boxes, [batch_size, num_tiles, _NMS_TILE_SIZE, 4]) * (1 - mask)
  boxes = jnp.reshape(boxes, [batch_size, -1, 4])

  # Updates output_size.
  output_size += jnp.sum(
      jnp.any(box_slice > 0, [2]).astype(jnp.int32), [1])
  return boxes, iou_threshold, output_size, idx + 1


def non_max_suppression_padded(scores: jnp.ndarray,
                               boxes: jnp.ndarray,
                               max_output_size: jnp.ndarray,
                               iou_threshold: float,
                               return_idx: bool = False):
  """A wrapper that handles non-maximum suppression.

  Assumption:
    * The boxes are sorted by scores unless the box is a dot (all coordinates
      are zero).
    * Boxes with higher scores can be used to suppress boxes with lower scores.

  The overal design of the algorithm is to handle boxes tile-by-tile:

  boxes = boxes.pad_to_multiply_of(tile_size)
  num_tiles = len(boxes) // tile_size
  output_boxes = []
  for i in range(num_tiles):
    box_tile = boxes[i*tile_size : (i+1)*tile_size]
    for j in range(i - 1):
      suppressing_tile = boxes[j*tile_size : (j+1)*tile_size]
      iou = _bbox_overlap(box_tile, suppressing_tile)
      # if the box is suppressed in iou, clear it to a dot
      box_tile *= _update_boxes(iou)
    # Iteratively handle the diagnal tile.
    iou = _box_overlap(box_tile, box_tile)
    iou_changed = True
    while iou_changed:
      # boxes that are not suppressed by anything else
      suppressing_boxes = _get_suppressing_boxes(iou)
      # boxes that are suppressed by suppressing_boxes
      suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)
      # clear iou to 0 for boxes that are suppressed, as they cannot be used
      # to suppress other boxes any more
      new_iou = _clear_iou(iou, suppressed_boxes)
      iou_changed = (new_iou != iou)
      iou = new_iou
    # remaining boxes that can still suppress others, are selected boxes.
    output_boxes.append(_get_suppressing_boxes(iou))
    if len(output_boxes) >= max_output_size:
      break

  Args:
    scores: a tensor with a shape of [batch_size, anchors].
    boxes: a tensor with a shape of [batch_size, anchors, 4].
    max_output_size: a scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression.
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    return_idx: bool. If true, addtionally return index of the remaining boxes.
  Returns:
    nms_scores: a tensor with a shape of [batch_size, max_output_size].
      It has the same dtype as input scores.
    nms_proposals: a tensor with a shape of [batch_size, max_output_size, 4].
      It has the same dtype as input boxes.
    idx: only return if return_idx == True. A int32 array of shape
      [batch_size,  max_output_size]. The values are in range [0, num_boxes):
      the indexes of the remaining boxes.
  """
  batch_size = boxes.shape[0]
  num_boxes = boxes.shape[1]
  pad = int(np.ceil(float(num_boxes) / _NMS_TILE_SIZE)
           ) * _NMS_TILE_SIZE - num_boxes
  boxes = jnp.pad(boxes.astype(jnp.float32), [[0, 0], [0, pad], [0, 0]])
  scores = jnp.pad(scores.astype(jnp.float32), [[0, 0], [0, pad]])
  num_boxes += pad

  def _loop_cond(in_args):
    unused_boxes, unused_threshold, output_size, idx = in_args
    return jnp.logical_and(
        jnp.min(output_size) < max_output_size,
        idx < num_boxes // _NMS_TILE_SIZE)

  selected_boxes, _, output_size, _ = jax.lax.while_loop(
      _loop_cond, _suppression_loop_body, (
          boxes, iou_threshold,
          jnp.zeros([batch_size], jnp.int32),
          0
      ))
  idx = num_boxes - jax.lax.top_k(  # pytype: disable=wrong-arg-types  # jax-ndarray
      jnp.any(selected_boxes > 0, [2]).astype(jnp.int32) *
      jnp.expand_dims(jnp.arange(num_boxes, 0, -1), 0),
      max_output_size)[0].astype(jnp.int32)
  idx = jnp.minimum(idx, num_boxes - 1)
  idx_return = idx
  idx = jnp.reshape(
      idx + jnp.reshape(jnp.arange(batch_size) * num_boxes, [-1, 1]), [-1])
  boxes = jnp.reshape(
      (jnp.reshape(boxes, [-1, 4]))[idx],
      [batch_size, max_output_size, 4])
  boxes = boxes * (
      jnp.reshape(jnp.arange(max_output_size), [1, -1, 1]) < jnp.reshape(
          output_size, [-1, 1, 1])).astype(boxes.dtype)
  scores = jnp.reshape(
      jnp.reshape(scores, [-1, 1])[idx],
      [batch_size, max_output_size])
  scores = scores * (
      jnp.reshape(jnp.arange(max_output_size), [1, -1]) < jnp.reshape(
          output_size, [-1, 1])).astype(scores.dtype)
  if return_idx:
    return scores, boxes, idx_return
  else:
    return scores, boxes


def batched_nms_jax(
    boxes: jnp.ndarray,
    scores: jnp.ndarray,
    idxs: jnp.ndarray,
    output_size: int,
    iou_threshold: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Class-independent NMS using a coordinate trick.

  Args:
    boxes: array in shape N x 4, in format (x_min, y_min, x_max, y_max). Can be
      eithor absolute or normalized.
    scores: array in shape N.
    idxs: array in shape N: the class index. We only apply NMS within
      the same class.
    output_size: int: the number of remaining boxes.
    iou_threshold: float. A box will be suppressed if there is another box that
      has a higher score and an IoU greater than the threshold.
  Returns:
    nms_boxes: a tensor with a shape of [[output_size, 4].
    nms_scores: a tensor with a shape of [output_size].
    nms_classes: a tensor with a shape of [output_size].
  """
  max_coordinate = boxes.max()
  offsets = idxs * (max_coordinate + 1)
  boxes_for_nms = boxes + offsets[:, None]
  # non_max_suppression_padded uses an additional batch dimension.
  nms_scores, _, keep = non_max_suppression_padded(  # pytype: disable=wrong-arg-types  # jax-ndarray
      scores[None], boxes_for_nms[None], output_size, iou_threshold,
      return_idx=True)
  vmap_index = jax.vmap(lambda x, i: x[i])
  nms_boxes = vmap_index(boxes[None], keep)
  nms_classes = vmap_index(idxs[None], keep)
  # undo batch
  return nms_boxes[0], nms_scores[0], nms_classes[0]
