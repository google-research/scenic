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

"""Tests for metrics."""

from absl.testing import parameterized
import numpy as np
from scenic.projects.unloc import metrics
import tensorflow as tf


class MetricsTest(tf.test.TestCase, parameterized.TestCase):

  def test_frame_accuracy(self):
    logits = np.array([
        [1.5, 0.0, -1.0],
        [0.1, 0.7, 0.2],
        [0.1, 0.7, 0.2],
        [0.1, -0.7, 0.2],
        [-0.1, -0.7, -0.2],
    ], np.float32)
    label = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
    ], np.int32)
    actual = metrics.frame_accuracy(logits, label)
    self.assertAlmostEqual(actual, 0.6)

  def test_frame_accuracy_all_background(self):
    logits = np.array(
        [
            [1.5, 0.0, -1.0],
            [0.1, 0.7, 0.2],
            [0.1, 0.7, 0.2],
            [0.1, -0.7, 0.2],
            [-0.1, -0.7, -0.2],
        ],
        np.float32,
    )
    label = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        np.int32,
    )
    actual = metrics.frame_accuracy(logits, label)
    self.assertAlmostEqual(actual, 0.2)

  def test_frame_accuracy_all_foreground(self):
    logits = np.array(
        [
            [1.5, 0.0, -1.0],
            [0.1, 0.7, 0.2],
            [0.1, 0.7, 0.2],
            [0.1, -0.7, -0.2],
            [-0.1, -0.7, -0.2],
        ],
        np.float32,
    )
    label = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        np.int32,
    )
    actual = metrics.frame_accuracy(logits, label)
    self.assertAlmostEqual(actual, 0.4)

  @parameterized.named_parameters(
      ('both_inside_gt', [[0.3, 0.2]], [0.5]),
      ('both_inside_gt_two_items', [[0.3, 0.2], [0.4, 0.4]], [0.5, 0.8]),
      ('both_outside_gt', [[0.6, 0.7]], [1. / 1.3]),
      ('start_outside_end_inside_gt', [[0.6, 0.3]], [0.8 / 1.1]),
      ('end_outside_start_inside_gt', [[0.3, 0.7]], [0.8 / 1.2]),
      ('pred_equal_gt', [[0.5, 0.5]], [1.]),
  )
  def test_temporal_iou(self, pred_displacements, expected_ious):
    pred_displacements = np.array(pred_displacements, dtype=np.float32)
    gt_displacements = np.array([[0.5, 0.5]], dtype=np.float32)
    ious = metrics.temporal_iou(pred_displacements, gt_displacements)
    self.assertAllClose(ious, expected_ious)

  @parameterized.named_parameters(
      ('both_inside_gt', [[0.3, 0.2]], [0.0025]),
      ('both_inside_gt_two_items', [[0.3, 0.2], [0.4, 0.4]], [0.0025, 0.0]),
      ('both_outside_gt', [[0.6, 0.7]], [(0.05 / 1.3)**2]),
      ('start_outside_end_inside_gt', [[0.6, 0.3]], [(0.15 / 1.1)**2]),
      ('end_outside_start_inside_gt', [[0.3, 0.7]], [(0.2 / 1.2)**2]),
      ('pred_equal_gt', [[0.5, 0.5]], [0.]),
  )
  def test_center_offset_squared(self, pred_displacements, expected):
    pred_displacements = np.array(pred_displacements, dtype=np.float32)
    gt_displacements = np.array([[0.5, 0.5]], dtype=np.float32)
    actual = metrics.center_offset_squared(pred_displacements, gt_displacements)
    self.assertAllClose(actual, expected)

  @parameterized.named_parameters(
      ('both_inside_gt', [[0.3, 0.2]], [0.5]),
      ('both_inside_gt_two_items', [[0.3, 0.2], [0.4, 0.4]], [0.5, 0.2]),
      ('both_outside_gt', [[0.6, 0.7]], [0.3 / 1.3]),
      ('start_outside_end_inside_gt', [[0.6, 0.3]], [0.3 / 1.1]),
      ('end_outside_start_inside_gt', [[0.3, 0.7]], [0.4 / 1.2]),
      ('pred_equal_gt', [[0.5, 0.5]], [0.0]),
  )
  def test_normalized_l1(self, pred_displacements, expected):
    pred_displacements = np.array(pred_displacements, dtype=np.float32)
    gt_displacements = np.array([[0.5, 0.5]], dtype=np.float32)
    actual = metrics.normalized_l1(pred_displacements, gt_displacements)
    self.assertAllClose(actual, expected)

  @parameterized.named_parameters(
      ('r@1_iou_0.5', [1], [0.5], [0.5]),
      ('r@2_iou_0.5', [2], [0.5], [0.75]),
      ('r@1_iou_0.75', [1], [0.75], [0.5]),
      ('r@1_5_iou_0.75_0.9', [1, 5], [0.75, 0.9], [0.5, 0.0, 0.75, 0.0]),
      )
  def test_compute_recall_at_k(self, ranks, iou_thresholds, expected):
    ground_truth_segments = [
        np.array([10, 20], dtype=np.float32),
        np.array([30, 40], dtype=np.float32),
        np.array([50, 60], dtype=np.float32),
        np.array([70, 70.5], dtype=np.float32),
        ]
    predicted_segments = [
        np.array([[11, 19], [15, 25], [30, 35], [40, 45]], dtype=np.float32),
        np.array([[31, 39], [32, 38], [35, 40], [20, 30]], dtype=np.float32),
        np.array([[51, 59], [52, 58], [55, 60], [60, 65]], dtype=np.float32),
        np.array([], dtype=np.float32),
    ]
    predicted_scores = [
        np.array([0.9, 0.8, 0.7, 0.6]),
        np.array([0.6, 0.7, 0.8, 0.9]),
        np.array([0.9, 0.8, 0.7, 0.6]),
        np.array([], dtype=np.float32),  # no detections under low resolution.
    ]
    actual = metrics.compute_recall_at_k(
        ground_truth_segments,
        predicted_segments,
        predicted_scores,
        ranks,
        iou_thresholds,
    )
    scores_out = [score for _, score in actual.items()]
    self.assertAllClose(
        scores_out, expected
    )

if __name__ == '__main__':
  tf.test.main()
