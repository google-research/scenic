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

r"""Util functions for RoI Head.

The box operation code is forked from
https://github.com/google-research/google-research/blob/master/fvlm/utils/\
box_utils.py

The detection post-processing code is forked from
https://github.com/google-research/google-research/blob/master/fvlm/ops/\
generate_detections.py
"""
import functools
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from scenic.projects.baselines.centernet.modeling import nms

Array = jnp.ndarray
_EPSILON = 1e-7
BBOX_XFORM_CLIP = np.log(1000. / 16.)


def pairwise_iou(boxes1: Array, boxes2: Array) -> Array:
  """Compute pairwise IoU between two batches of boxes.

  Given two Boxes of size N and M, compute the IoU (intersection over
  union) between **all** N x M pairs of boxes.

  Args:
    boxes1: N bounding boxes.
    boxes2: M bounding boxes.

  Returns:
      IoU matrix of size (N, M). Invalid boxes will have a IoU of
      0 with any other boxes.
  """
  if boxes1.ndim != 2 or boxes2.ndim != 2:
    raise ValueError("pairwise_iou only supports 2D Boxes! "
                     "Either flatten your inputs or use vmap.")
  area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
  area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

  x1y1 = jnp.maximum(boxes1[:, None, :2], boxes2[:, :2])
  x2y2 = jnp.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
  intersection = jnp.maximum(x2y2 - x1y1, 0)  # (N, M, 2)
  intersection = intersection[:, :, 0] * intersection[:, :, 1]

  union = area1[:, None] + area2 - intersection
  # Intersection must be 0 when union is 0.
  iou = intersection / jnp.where(union > 0, union, 1.0)
  return iou


def encode_boxes(boxes: Array,
                 anchors: Array,
                 weights: Optional[Sequence[float]] = None) -> Array:
  """Encode boxes to targets.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    anchors: a tensor whose shape is the same as, or `broadcastable` to `boxes`,
      representing the coordinates of anchors in ymin, xmin, ymax, xmax order.
    weights: None or a list of four float numbers used to scale coordinates.

  Returns:
    encoded_boxes: a tensor whose shape is the same as `boxes` representing the
      encoded box targets.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError(
        "boxes.shape[-1] is {:d}, but must be 4.".format(boxes.shape[-1]))

  boxes = boxes.astype(anchors.dtype)
  ymin = boxes[..., 0:1]
  xmin = boxes[..., 1:2]
  ymax = boxes[..., 2:3]
  xmax = boxes[..., 3:4]
  box_h = ymax - ymin
  box_w = xmax - xmin
  box_yc = ymin + 0.5 * box_h
  box_xc = xmin + 0.5 * box_w

  anchor_ymin = anchors[..., 0:1]
  anchor_xmin = anchors[..., 1:2]
  anchor_ymax = anchors[..., 2:3]
  anchor_xmax = anchors[..., 3:4]
  anchor_h = anchor_ymax - anchor_ymin
  anchor_w = anchor_xmax - anchor_xmin
  anchor_yc = anchor_ymin + 0.5 * anchor_h
  anchor_xc = anchor_xmin + 0.5 * anchor_w

  encoded_dy = (box_yc - anchor_yc) / anchor_h
  encoded_dx = (box_xc - anchor_xc) / anchor_w
  encoded_dh = jnp.log(box_h / anchor_h)
  encoded_dw = jnp.log(box_w / anchor_w)
  if weights:
    encoded_dy *= weights[0]
    encoded_dx *= weights[1]
    encoded_dh *= weights[2]
    encoded_dw *= weights[3]

  encoded_boxes = jnp.concatenate(
      [encoded_dy, encoded_dx, encoded_dh, encoded_dw],
      axis=-1)
  return encoded_boxes


def decode_boxes(encoded_boxes: Array,
                 anchors: Array,
                 weights: Optional[Sequence[float]] = None) -> Array:
  """Decode boxes.

  Args:
    encoded_boxes: a tensor whose last dimension is 4 representing the
      coordinates of encoded boxes in ymin, xmin, ymax, xmax order.
    anchors: a tensor whose shape is the same as, or `broadcastable` to `boxes`,
      representing the coordinates of anchors in ymin, xmin, ymax, xmax order.
    weights: None or a list of four float numbers used to scale coordinates.

  Returns:
    encoded_boxes: a tensor whose shape is the same as `boxes` representing the
      decoded box targets.
  """
  if encoded_boxes.shape[-1] != 4:
    raise ValueError(
        "encoded_boxes.shape[-1] is {:d}, but must be 4."
        .format(encoded_boxes.shape[-1]))

  encoded_boxes = encoded_boxes.astype(anchors.dtype)
  dy = encoded_boxes[..., 0:1]
  dx = encoded_boxes[..., 1:2]
  dh = encoded_boxes[..., 2:3]
  dw = encoded_boxes[..., 3:4]
  if weights:
    dy /= weights[0]
    dx /= weights[1]
    dh /= weights[2]
    dw /= weights[3]
  dh = jnp.minimum(dh, BBOX_XFORM_CLIP)
  dw = jnp.minimum(dw, BBOX_XFORM_CLIP)

  anchor_ymin = anchors[..., 0:1]
  anchor_xmin = anchors[..., 1:2]
  anchor_ymax = anchors[..., 2:3]
  anchor_xmax = anchors[..., 3:4]
  anchor_h = anchor_ymax - anchor_ymin
  anchor_w = anchor_xmax - anchor_xmin
  anchor_yc = anchor_ymin + 0.5 * anchor_h
  anchor_xc = anchor_xmin + 0.5 * anchor_w

  decoded_boxes_yc = dy * anchor_h + anchor_yc
  decoded_boxes_xc = dx * anchor_w + anchor_xc
  decoded_boxes_h = jnp.exp(dh) * anchor_h
  decoded_boxes_w = jnp.exp(dw) * anchor_w

  decoded_boxes_ymin = decoded_boxes_yc - 0.5 * decoded_boxes_h
  decoded_boxes_xmin = decoded_boxes_xc - 0.5 * decoded_boxes_w
  decoded_boxes_ymax = decoded_boxes_ymin + decoded_boxes_h
  decoded_boxes_xmax = decoded_boxes_xmin + decoded_boxes_w

  decoded_boxes = jnp.concatenate(
      [decoded_boxes_ymin, decoded_boxes_xmin,
       decoded_boxes_ymax, decoded_boxes_xmax],
      axis=-1)
  return decoded_boxes


def clip_boxes(boxes: Array,
               image_shape: Array) -> Array:
  """Clips boxes to image boundaries. It's called from roi_ops.py.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    image_shape: (batch_size, 1, 2). [height, width].

  Returns:
    clipped_boxes: a tensor whose shape is the same as `boxes` representing the
      clipped boxes in ymin, xmin, ymax, xmax order.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError(
        "boxes.shape[-1] is {:d}, but must be 4.".format(boxes.shape[-1]))

  image_shape = image_shape.astype(boxes.dtype)
  height, width = jnp.split(image_shape, 2, axis=-1)
  y_x_max_list = jnp.concatenate([height, width, height, width], -1)

  return jnp.maximum(jnp.minimum(boxes, y_x_max_list), 0.0)


def generate_detections(
    class_outputs: Array,
    box_outputs: Array,
    pre_nms_num_detections: int = 5000,
    post_nms_num_detections: int = 100,
    nms_threshold: float = 0.3,
    score_threshold: float = 0.05,
    class_box_regression: bool = True,
) -> Tuple[Array, Array, Array, Array]:
  """Generates the detections given anchor boxes and predictions.

  Args:
    class_outputs: An array with shape [batch, num_boxes, num_classes] of class
      logits for each box.
    box_outputs: An array with shape [batch, num_boxes, num_classes, 4] of
      predicted boxes in [ymin, xmin, ymax, xmax] order. Also accept num_classes
      = 1 for class agnostic box outputs.
    pre_nms_num_detections: An integer that specifies the number of candidates
      before NMS.
    post_nms_num_detections: An integer that specifies the number of candidates
      after NMS.
    nms_threshold: A float number to specify the IOU threshold of NMS.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    class_box_regression: Whether to use class-specific box regression or not.
      Default True is to assume box_outputs are class-specific.

  Returns:
    A tuple of arrays corresponding to
      (box coordinates, object categories for each boxes, and box scores).
  """
  batch_size, _, num_classes = jnp.shape(class_outputs)

  final_boxes = []
  final_scores = []
  final_classes = []
  all_valid = []
  for b in range(batch_size):
    nmsed_boxes = []
    nmsed_scores = []
    nmsed_classes = []
    # Skips the background class.
    for i in range(1, num_classes):
      box_idx = i if class_box_regression else 0
      boxes_i = box_outputs[b, :, box_idx]
      scores_i = class_outputs[b, :, i]
      # Filter by threshold.
      above_threshold = scores_i > score_threshold
      scores_i = jnp.where(above_threshold, scores_i, -1)

      # Obtains pre_nms_num_boxes before running NMS.
      scores_i, indices = jax.lax.top_k(
          scores_i, k=min(pre_nms_num_detections, scores_i.shape[-1])
      )
      boxes_i = boxes_i[indices]

      nmsed_scores_i, nmsed_boxes_i = nms.non_max_suppression_padded(  # pytype: disable=wrong-arg-types  # jax-ndarray
          scores=scores_i[None, ...],
          boxes=boxes_i[None, ...],
          max_output_size=post_nms_num_detections,
          iou_threshold=nms_threshold,
      )

      nmsed_classes_i = jnp.ones([post_nms_num_detections]) * i
      nmsed_boxes.append(nmsed_boxes_i[0])
      nmsed_scores.append(nmsed_scores_i[0])
      nmsed_classes.append(nmsed_classes_i)

    # Concats results from all classes and sort them.
    nmsed_boxes = jnp.concatenate(nmsed_boxes, axis=0)
    nmsed_scores = jnp.concatenate(nmsed_scores, axis=0)
    nmsed_classes = jnp.concatenate(nmsed_classes, axis=0)
    nmsed_scores, indices = jax.lax.top_k(
        nmsed_scores, k=post_nms_num_detections)
    nmsed_boxes = nmsed_boxes[indices]
    nmsed_classes = nmsed_classes[indices]
    valid_detections = jnp.sum((nmsed_scores > 0.0).astype(jnp.int32))

    all_valid.append(valid_detections)
    final_classes.append(nmsed_classes)
    final_scores.append(nmsed_scores)
    final_boxes.append(nmsed_boxes)

  return (
      jnp.stack(final_boxes, axis=0),
      jnp.stack(final_scores, axis=0),
      jnp.stack(final_classes, axis=0),
      jnp.stack(all_valid, axis=0),
  )


@functools.partial(jax.vmap, in_axes=[0, 0], out_axes=0)
def batch_gather(x: Array, idx: Array) -> Array:
  """Performs a batched gather of the data.

  Args:
    x: A [batch, num_in, ...] JTensor of data to gather from.
    idx: A [batch, num_out] JTensor of dtype int32 or int64 specifying which
      elements to gather. Every value is expected to be in the range of [0,
      num_in].

  Returns:
    A [batch, num_out, ...] JTensor of gathered data.
  """
  return x[idx]


def generate_detections_vmap(
    class_outputs: Array,
    box_outputs: Array,
    pre_nms_num_detections: int = 5000,
    post_nms_num_detections: int = 100,
    nms_threshold: float = 0.3,
    score_threshold: float = 0.05,
    class_box_regression: bool = True,
) -> Tuple[Array, Array, Array, Array]:
  """Generates the detections given anchor boxes and predictions.


  Args:
    class_outputs: An array with shape [batch, num_boxes, num_classes] of class
      logits for each box.
    box_outputs: An array with shape [batch, num_boxes, num_classes, 4] of
      predicted boxes in [ymin, xmin, ymax, xmax] order. Also accept num_classes
      = 1 for class agnostic box outputs.
    pre_nms_num_detections: An integer that specifies the number of candidates
      before NMS.
    post_nms_num_detections: An integer that specifies the number of candidates
      after NMS.
    nms_threshold: A float number to specify the IOU threshold of NMS.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    class_box_regression: Whether to use class-specific box regression or not.
      Default True is to assume box_outputs are class-specific.

  Returns:
    A tuple of arrays corresponding to
      (box coordinates, object categories for each boxes, and box scores).
  """
  _, _, num_classes = jnp.shape(class_outputs)

  if not class_box_regression:
    if num_classes == 1:
      raise ValueError(
          "If using `class_box_regression=False` we expect num_classes = 1"
      )
    box_outputs = jnp.tile(box_outputs, [1, 1, num_classes, 1])

  box_outputs = box_outputs[:, :, 1:, :]
  box_scores = class_outputs[:, :, 1:]

  def batched_per_class_nms_fn(per_class_boxes, per_class_scores):
    # Transpose the data so the class dim is now a batch dim.
    per_class_boxes = jnp.transpose(per_class_boxes, (1, 0, 2))
    per_class_scores = jnp.transpose(per_class_scores, (1, 0))

    above_threshold = per_class_scores > score_threshold
    per_class_scores = jnp.where(
        above_threshold, per_class_scores, per_class_scores * 0 - 1
    )
    # Obtains pre_nms_num_boxes before running NMS.
    per_class_scores, indices = jax.lax.top_k(
        per_class_scores,
        k=min(pre_nms_num_detections, per_class_scores.shape[-1]),
    )
    per_class_boxes = batch_gather(per_class_boxes, indices)

    # Run NMS where the [num_classes, ...] dim is the batch dim.
    nmsed_scores, nmsed_boxes = nms.non_max_suppression_padded(  # pytype: disable=wrong-arg-types  # jax-ndarray
        scores=per_class_scores,
        boxes=per_class_boxes,
        max_output_size=post_nms_num_detections,
        iou_threshold=nms_threshold,
    )

    nmsed_classes = jnp.ones([num_classes - 1, post_nms_num_detections])
    nmsed_classes *= jnp.arange(1, num_classes, dtype=jnp.int32)[:, None]

    nmsed_boxes = jnp.reshape(nmsed_boxes, [-1, 4])
    nmsed_scores = jnp.reshape(nmsed_scores, [-1])
    nmsed_classes = jnp.reshape(nmsed_classes, [-1])

    nmsed_scores, indices = jax.lax.top_k(
        nmsed_scores, k=post_nms_num_detections)
    nmsed_boxes = nmsed_boxes[indices]
    nmsed_classes = nmsed_classes[indices]
    valid_detections = jnp.sum((nmsed_scores > 0.0).astype(jnp.int32))
    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections

  batched_per_class_nms = jax.vmap(
      batched_per_class_nms_fn, in_axes=0, out_axes=0
  )

  final_boxes, final_scores, final_classes, valid_detections = (
      batched_per_class_nms(box_outputs, box_scores)
  )
  return final_boxes, final_scores, final_classes, valid_detections


def process_and_generate_detections(
    box_outputs: Array,
    class_outputs: Array,
    anchor_boxes: Array,
    image_shape: Array,
    pre_nms_num_detections: int = 5000,
    post_nms_num_detections: int = 100,
    nms_threshold: float = 0.5,
    score_threshold: float = 0.05,
    class_box_regression: bool = True,
    box_weights: Any = (10.0, 10.0, 5.0, 5.0),
    use_vmap: bool = True,
) -> Dict[str, Array]:
  """Generate final detections.

    Move softmax of class_outputs out of this function so that it can multiply
    with proposal scores.

  Args:
    box_outputs: An array of shape of [batch_size, K, num_classes * 4]
      representing the class-specific box coordinates relative to anchors.
    class_outputs: An array of shape of [batch_size, K, num_classes]
      representing the class logits before applying score activiation.
    anchor_boxes: An array of shape of [batch_size, K, 4] representing the
      corresponding anchor boxes w.r.t `box_outputs`.
    image_shape: An array of shape of [batch_size, 2] storing the image height
      and width w.r.t. the scaled image, i.e. the same image space as
      `box_outputs` and `anchor_boxes`.
    pre_nms_num_detections: An integer that specifies the number of candidates
      before NMS.
    post_nms_num_detections: An integer that specifies the number of candidates
      after NMS.
    nms_threshold: A float number to specify the IOU threshold of NMS.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.
    class_box_regression: Whether to use class-specific box regression or not.
      Default True is to assume box_outputs are class-specific.
    box_weights: four float numbers used to scale coordinates.
    use_vmap: bool;

  Returns:
    A dictionary with the following key-value pairs:
      detection_boxes: `float` array of shape [batch_size, max_total_size, 4]
        representing top detected boxes in [y1, x1, y2, x2].
      detection_scores: `float` array of shape [batch_size, max_total_size]
        representing sorted confidence scores for detected boxes. The values are
        between [0, 1].
      detection_classes: `int` array of shape [batch_size, max_total_size]
        representing classes for detected boxes.
      num_detections: `int` array of shape [batch_size] only the top
        `valid_detections` boxes are valid detections.
  """
  _, num_locations, num_classes = class_outputs.shape

  if class_box_regression:
    num_detections = num_locations * num_classes
    box_outputs = box_outputs.reshape(-1, num_detections, 4)
    anchor_boxes = jnp.tile(
        jnp.expand_dims(anchor_boxes, axis=2), [1, 1, num_classes, 1])
    anchor_boxes = anchor_boxes.reshape(-1, num_detections, 4)

  decoded_boxes = decode_boxes(
      box_outputs, anchor_boxes, weights=box_weights)
  decoded_boxes = clip_boxes(
      decoded_boxes, image_shape[:, None, :])
  if class_box_regression:
    decoded_boxes = decoded_boxes.reshape(-1, num_locations, num_classes, 4)
  else:
    decoded_boxes = decoded_boxes[:, :, None, :]

  generate_detections_fn = (
      generate_detections_vmap if use_vmap else generate_detections)

  nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
      generate_detections_fn(
          class_outputs,
          decoded_boxes,
          pre_nms_num_detections,
          post_nms_num_detections,
          nms_threshold,
          score_threshold,
          class_box_regression,
      ))

  return {
      "num_detections": valid_detections,
      "detection_boxes": nmsed_boxes,
      "detection_classes": nmsed_classes,
      "detection_scores": nmsed_scores,
  }
