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

"""Implements metrics."""

from typing import Any, List, Union

import jax.numpy as jnp
import numpy as np

PyModule = Any
Array = Union[jnp.ndarray, np.ndarray]
Scalar = Union[int, float, np.number, np.ndarray, jnp.ndarray]


def frame_accuracy(
    logits: np.ndarray,
    label: np.ndarray,
    background_logit_threshold: float = 0.0,
) -> float:
  """Computes frame accuracy.

  We assume there are background samples where the labels are all zeros.

  Args:
    logits: Class logits in shape (N, num_classes).
    label: Multihot class labels in shape (N, num_classes).
    background_logit_threshold: If the max logit of an example is less than this
      value, this example is predicted as background.

  Returns:
    Accuracy computed as number of correctly predicted frames over total number
    of frames.
  """
  top1_idx = np.argmax(logits, axis=-1)
  background_label = np.sum(label, axis=-1) == 0
  pred_background = (
      np.max(logits, axis=-1) <= background_logit_threshold
  ).astype(np.int32)

  # Extracts the label at the highest logit index for each input.
  top1_correct = np.take_along_axis(label, top1_idx[..., None], axis=-1)
  top1_correct = np.squeeze(top1_correct)
  foreground_correct = ~pred_background.astype(bool) * top1_correct

  # Count correctly classified background samples.
  background_correct = pred_background * background_label
  correct = foreground_correct + background_correct
  return np.sum(correct) / len(logits)


def temporal_iou(pred_displacements: Array,
                 gt_displacements: Array,
                 eps: float = 1e-6,
                 np_backend: PyModule = jnp) -> Array:
  """Computes temporal IoUs.

  The displacements are assumed to be greater or equal to zero.

  Args:
    pred_displacements: A ND array where the last dimension is 2 containing the
      predicted displacements to the start/end times.
    gt_displacements: A ND array where the last dimension is 2 containing the
      ground truth displacements to the start/end times.
    eps: A small value to avoid division by zero.
    np_backend: Numpy backend.

  Returns:
    A (N-1)D array containing the computed IoUs.
  """
  intersection = np_backend.minimum(
      pred_displacements[..., 0],
      gt_displacements[..., 0]) + np_backend.minimum(pred_displacements[..., 1],
                                                     gt_displacements[..., 1])
  union = np_backend.maximum(pred_displacements[..., 0],
                             gt_displacements[..., 0]) + np_backend.maximum(
                                 pred_displacements[..., 1],
                                 gt_displacements[..., 1])
  return intersection / (union + eps)


def center_offset_squared(pred_displacements: Array,
                          gt_displacements: Array,
                          eps: float = 1e-6,
                          np_backend: PyModule = jnp) -> Array:
  """Computes squared offset between centers of temporal segments.

  The displacements are assumed to be greater or equal to zero.

  Args:
    pred_displacements: A ND array where the last dimension is 2 containing the
      predicted displacements to the start/end times.
    gt_displacements: A ND array where the last dimension is 2 containing the
      ground truth displacements to the start/end times.
    eps: A small value to avoid division by zero.
    np_backend: Numpy backend.

  Returns:
    A (N-1)D array of squared offsets between centers of temporal segments.
  """
  union = np_backend.maximum(pred_displacements[..., 0],
                             gt_displacements[..., 0]) + np_backend.maximum(
                                 pred_displacements[..., 1],
                                 gt_displacements[..., 1])
  offset = 0.5 * (
      pred_displacements[..., 1] - pred_displacements[..., 0] -
      (gt_displacements[..., 1] - gt_displacements[..., 0]))
  return np_backend.square(offset / (union + eps))


def normalized_l1(
    pred_displacements: Array,
    gt_displacements: Array,
    eps: float = 1e-6,
    np_backend: PyModule = jnp,
) -> Array:
  """Computes the normalized L1 distance between two temporal segments."""
  union = np_backend.maximum(
      pred_displacements[..., 0], gt_displacements[..., 0]
  ) + np_backend.maximum(pred_displacements[..., 1], gt_displacements[..., 1])
  l1_norm = np_backend.abs(
      gt_displacements[..., 0] - pred_displacements[..., 0]
  ) + np_backend.abs(gt_displacements[..., 1] - pred_displacements[..., 1])
  return l1_norm / (union + eps)


def compute_iou(segment1: Array, segment2: Array) -> Any:
  """Computes the IoU score between two temporal segments."""
  start = max(segment1[0], segment2[0])
  end = min(segment1[1], segment2[1])
  if start >= end:
    return 0.0
  intersection = end - start
  duration1 = segment1[1] - segment1[0]
  duration2 = segment2[1] - segment2[0]
  union = duration1 + duration2 - intersection
  return intersection / union


def compute_recall_at_k(
    ground_truth_segments: List[Array],
    predicted_segments: List[Array],
    predicted_scores: List[Array],
    ranks: List[int],
    iou_thresholds: List[float],
) -> dict[str, Scalar]:
  """Compute recall at k given a iou threshold.

  Compute the recall@k for a set of ground-truth segments, predicted segments,
  and predicted scores, given a list of specific IoU threshold.

  Args:
    ground_truth_segments: Each element represents the ground-truth segment for
      the corresponding caption, with the start and end frame indices.
    predicted_segments: Each element represents the predicted segments for the
      corresponding caption, with the start and end frame indices for each.
    predicted_scores: Each element represents the predicted segments score
      for the corresponding caption.
    ranks: The number of top ranked segments to consider for each caption.
    iou_thresholds: The IoU threshold. A predicted segment is considered
      positive if its IoU with the ground-truth segment is at least this value.

  Returns:
    The mean recall@k over all captions with the given IoU.
  """

  metrics = {}
  num_captions = len(ground_truth_segments)
  for r in ranks:
    for iou_threshold in iou_thresholds:
      recall = 0.0
      for i in range(num_captions):
        ground_truth_segment = ground_truth_segments[i]
        predicted_segment_scores = predicted_scores[i]
        num_predictions = min(r, len(predicted_segment_scores))
        if num_predictions == 0:
          continue
        sorted_indices = np.argsort(predicted_segment_scores)[::-1]
        predicted_segments_sorted = predicted_segments[i][sorted_indices]
        for j in range(num_predictions):
          predicted_segment = predicted_segments_sorted[j]
          iou = compute_iou(ground_truth_segment, predicted_segment)
          if iou >= iou_threshold:
            recall += 1
            break
      metrics.update(
          {
              'R@{},IOU={}'.format(r, iou_threshold): recall / num_captions
          }
      )
  return metrics
