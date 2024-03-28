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

"""Tests for postprocessing_utils."""

from absl.testing import parameterized
import ml_collections
import numpy as np
from scenic.projects.unloc import metrics
from scenic.projects.unloc import postprocessing_utils
import tensorflow as tf


class PostprocessingUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_dedup_by_vid_no_frame_mask(self):
    logits = np.array([
        [1.0, -2.0, 3.0],
        [3.0, -2.0, 1.0],
        [3.0, -2.0, 1.0],
    ], np.float32)
    labels = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
    ], np.int32)
    batch_mask = np.array([1, 1, 1], np.int32)
    vids = np.array([0, 1, 1], np.int32)
    deduped_logits, deduped_labels, deduped_vids = (
        postprocessing_utils.dedup_by_vid(
            logits, labels, batch_mask, vids
        )
    )
    self.assertAllClose(deduped_logits, [
        [1.0, -2.0, 3.0],
        [3.0, -2.0, 1.0],
    ])
    self.assertAllEqual(deduped_labels, [
        [1, 0, 0],
        [0, 1, 0],
    ])
    self.assertAllEqual(deduped_vids, [0, 1])

  def test_dedup_by_vid_w_frame_mask(self):
    logits = np.array([
        [[1.0, -2.0, 3.0], [1.0, -2.0, 3.0]],
        [[3.0, -2.0, 1.0], [3.0, -2.0, 1.0]],
        [[3.0, -2.0, 1.0], [3.0, -2.0, 1.0]],
    ], np.float32)
    labels = np.array([
        [[1, 0, 0], [1, 0, 0]],
        [[0, 1, 0], [0, 1, 0]],
        [[0, 1, 0], [0, 1, 0]],
    ], np.int32)
    batch_mask = np.array([1, 1, 1], np.int32)
    frame_mask = np.array([[1, 1], [1, 0], [1, 0]], np.int32)
    vids = np.array([0, 1, 1], np.int32)
    deduped_logits, deduped_labels, deduped_vids = (
        postprocessing_utils.dedup_by_vid(
            logits, labels, batch_mask, vids, frame_mask
        )
    )
    self.assertAllClose(deduped_logits, [
        [1.0, -2.0, 3.0],
        [1.0, -2.0, 3.0],
        [3.0, -2.0, 1.0],
    ])
    self.assertAllEqual(deduped_labels, [
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ])
    self.assertAllEqual(deduped_vids, [0, 1])

  @parameterized.parameters(
      (float('-inf'), np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32),
       np.array([0.9, 0.1, 0.2, 0.1, 0.1, 0.2, 0.6, 0.3, 0.2],
                dtype=np.float32),
       np.array([
           [4., 8.],
           [5.6, 6.4],
           [5.2, 6.8],
           [6.6, 7.8],
           [6.6, 7.8],
           [6.2, 8.2],
           [6., 10.],
           [7.6, 8.4],
           [7.2, 8.8],
       ])),
      (0.5, np.array([0, 0], dtype=np.int32),
       np.array([0.9, 0.6], dtype=np.float32), np.array([[4., 8.], [6., 10.]])),
      (1.0, np.array([], dtype=np.int32), np.array(
          [], dtype=np.float32), np.zeros((0, 2), dtype=np.float32)),
  )
  def test_get_segments_from_frame_predictions(self, score_threshold,
                                               expected_class_indices,
                                               expected_class_probs,
                                               expected_segments):
    class_probs = np.array([[0.9, 0.1, 0.2], [0.1, 0.1, 0.2], [0.6, 0.3, 0.2],
                            [0.1, 0.8, 0.2]])
    displacements = np.array([
        [[0.5, 0.5], [0.1, 0.1], [0.2, 0.2]],
        [[0.1, 0.2], [0.1, 0.2], [0.2, 0.3]],
        [[0.5, 0.5], [0.1, 0.1], [0.2, 0.2]],
        [[0.1, 0.2], [0.1, 0.2], [0.2, 0.3]],
    ])
    input_mask = np.array([1, 1, 1, 0], dtype=np.int32)
    total_frames = 16
    (actual_class_indices, actual_class_probs, actual_segments
    ) = postprocessing_utils.get_segments_from_frame_predictions(
        class_probs,
        displacements,
        input_mask=input_mask,
        total_frames=total_frames,
        stride=1,
        displacement_normalizer='sampled_span',
        secs_per_timestep=1.0,
        score_threshold=score_threshold)
    self.assertAllEqual(actual_class_indices, expected_class_indices)
    self.assertAllClose(actual_class_probs, expected_class_probs)
    self.assertAllClose(actual_segments, expected_segments)

  def test_get_segments_from_frame_predictions_with_fpn(self):
    class_probs = np.array([
        # FPN level 0
        [0.9, 0.1, 0.2],
        [0.1, 0.1, 0.2],
        [0.6, 0.3, 0.2],
        [0.1, 0.8, 0.2],
        [0.9, 0.1, 0.2],
        [0.1, 0.1, 0.2],
        [0.6, 0.3, 0.2],  # mask out
        [0.1, 0.8, 0.2],  # mask out
        # FPN level 1
        [0.9, 0.1, 0.2],
        [0.1, 0.1, 0.2],
        [0.6, 0.3, 0.2],
        [0.1, 0.8, 0.2],  # mask out
    ])
    displacements = np.array([
        # FPN level 0
        [[0.5, 0.5], [0.1, 0.1], [0.2, 0.2]],
        [[0.1, 0.2], [0.1, 0.2], [0.2, 0.3]],
        [[0.5, 0.5], [0.1, 0.1], [0.2, 0.2]],
        [[0.1, 0.2], [0.1, 0.2], [0.2, 0.3]],
        [[0.5, 0.5], [0.1, 0.1], [0.2, 0.2]],
        [[0.1, 0.2], [0.1, 0.2], [0.2, 0.3]],
        [[0.5, 0.5], [0.1, 0.1], [0.2, 0.2]],  # mask out
        [[0.1, 0.2], [0.1, 0.2], [0.2, 0.3]],  # mask out
        # FPN level 1
        [[0.5, 0.5], [0.1, 0.1], [0.2, 0.2]],
        [[0.1, 0.2], [0.1, 0.2], [0.2, 0.3]],
        [[0.5, 0.5], [0.1, 0.1], [0.2, 0.2]],
        [[0.1, 0.2], [0.1, 0.2], [0.2, 0.3]],  # mask out
    ])
    input_mask = np.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0], dtype=np.int32)
    total_frames = 16
    actual_class_indices, actual_class_probs, actual_segments = (
        postprocessing_utils.get_segments_from_frame_predictions(
            class_probs,
            displacements,
            input_mask=input_mask,
            total_frames=total_frames,
            stride=1,
            secs_per_timestep=1.0,
            score_threshold=float('-inf'),
            feature_pyramid_config=ml_collections.ConfigDict({
                'num_features_level0': 8,
                'feature_pyramid_downsample_stride': 2,
                'feature_pyramid_levels': [0, 1],
            }),
        )
    )
    self.assertAllEqual(actual_class_indices, [0, 1, 2] * np.sum(input_mask))
    self.assertAllClose(
        actual_class_probs, class_probs[input_mask.astype(bool)].flatten()
    )
    expected_segments = [
        [0.0, 12.0],
        [2.4, 5.6],
        [0.8, 7.2],
        [3.4, 8.2],
        [3.4, 8.2],
        [1.8, 9.8],
        [0.0, 14.0],
        [4.4, 7.6],
        [2.8, 9.2],
        [5.4, 10.2],
        [5.4, 10.2],
        [3.8, 11.8],
        [0.0, 16.0],
        [6.4, 9.6],
        [4.8, 11.2],
        [7.4, 12.2],
        [7.4, 12.2],
        [5.8, 13.8],
        [0.0, 12.0],
        [2.4, 5.6],
        [0.8, 7.2],
        [4.4, 9.2],
        [4.4, 9.2],
        [2.8, 10.8],
        [0.0, 16.0],
        [6.4, 9.6],
        [4.8, 11.2],
    ]
    self.assertAllClose(actual_segments, expected_segments)

  def test_get_segments_from_frame_predictions_mr(self):
    class_probs = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        ])
    displacements = np.array([
        [
            [-5.00000000e-01, 1.00000000e00],
            [5.00000000e-01, 0.00000000e00],
            [1.50000000e00, -1.00000000e00],
            [2.50000000e00, -2.00000000e00],
            [3.50000000e00, -3.00000000e00],
            [4.50000000e00, -4.00000000e00],
            [5.50000000e00, -5.00000000e00],
            [6.50000000e00, -6.00000000e00],
        ],
        [
            [-1.50000000e00, 2.50000000e00],
            [-5.00000000e-01, 1.50000000e00],
            [5.00000000e-01, 5.00000000e-01],
            [1.50000000e00, -5.00000000e-01],
            [2.50000000e00, -1.50000000e00],
            [3.50000000e00, -2.50000000e00],
            [4.50000000e00, -3.50000000e00],
            [5.50000000e00, -4.50000000e00],
        ],
        [
            [-4.00000000e00, 5.00000000e00],
            [-3.00000000e00, 4.00000000e00],
            [-2.00000000e00, 3.00000000e00],
            [-1.00000000e00, 2.00000000e00],
            [0.00000000e00, 1.00000000e00],
            [1.00000000e00, 0.00000000e00],
            [2.00000000e00, -1.00000000e00],
            [3.00000000e00, -2.00000000e00],
        ],
        [
            [-6.50000000e00, 7.00000000e00],
            [-5.50000000e00, 6.00000000e00],
            [-4.50000000e00, 5.00000000e00],
            [-3.50000000e00, 4.00000000e00],
            [-2.50000000e00, 3.00000000e00],
            [-1.50000000e00, 2.00000000e00],
            [-5.00000000e-01, 1.00000000e00],
            [5.00000000e-01, 0.00000000e00],
        ],
        [
            [5.00000000e02, -5.00000000e02],
            [5.01000000e02, -5.01000000e02],
            [5.02000000e02, -5.02000000e02],
            [5.03000000e02, -5.03000000e02],
            [5.04000000e02, -5.04000000e02],
            [5.05000000e02, -5.05000000e02],
            [5.06000000e02, -5.06000000e02],
            [5.07000000e02, -5.07000000e02],
        ],  # mask out
    ])

    input_mask = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1],
        dtype=np.int32,
    )
    caption_mask = np.array(
        [1, 1, 1, 1, 0],
        dtype=np.int32,
    )
    total_frames = np.array([15], dtype=np.int32)
    actual_class_probs, actual_segments = (
        postprocessing_utils.get_segments_from_frame_predictions_mr(
            class_probs,
            displacements,
            input_mask=input_mask,
            caption_mask=caption_mask,
            total_frames=total_frames,
            stride=1,
            sampling_strategy='linspace',
            displacement_normalizer='none',
            secs_per_timestep=1.0,
            feature_pyramid_config=ml_collections.ConfigDict({
                'num_features_level0': 8,
                'feature_pyramid_downsample_stride': 2,
                'feature_pyramid_levels': [2],
            }),
        )
    )
    cls_probs = class_probs[caption_mask.astype(bool)]
    cls_probs = cls_probs[:, input_mask.astype(bool)]
    self.assertAllClose(actual_class_probs, cls_probs)
    expected_segments = np.array([
        [
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
        ],
        [
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
        ],
        [
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
        ],
        [
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
        ],
    ])
    self.assertAllClose(actual_segments, expected_segments)

  def test_non_max_suppression_mr(self):
    scores = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        ])
    segments = np.array([
        [
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
        ],
        [
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
        ],
        [
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
        ],
        [
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
        ],
    ])
    out_scores, out_segments = (
        postprocessing_utils.non_max_suppression_mr(
            scores,
            segments,
            config=ml_collections.ConfigDict({
                'max_detection': 100,
                'iou_threshold': 0.9,
                'score_threshold': 0.001,
                'soft_nms_sigma': 0.75
            }),
        )
    )
    expected_out_scores = [
        np.array([1.0]),
        np.array([1.0]),
        np.array([1.0, 0.5134171]),
        np.array([1.0]),
    ]
    expected_out_segments = [
        np.array([[1, 2]]),
        np.array([[3, 5]]),
        np.array([[8, 10], [8, 10]]),
        np.array([[13, 14]]),
    ]
    self.assertAllClose(out_scores, expected_out_scores)
    self.assertAllClose(out_segments, expected_out_segments)

    caption_mask = np.array(
        [1, 1, 1, 1, 0],
        dtype=np.int32,
    )
    gt_segments = np.array([[1, 2], [3, 5], [8, 10], [13, 14], [-1000, -1000]])
    gt_segments = gt_segments[caption_mask.astype(bool)]
    result = metrics.compute_recall_at_k(gt_segments, out_segments, out_scores,
                                         [1, 5], [0.5, 0.7])
    scores_out = [score for _, score in result.items()]
    self.assertAllClose(scores_out, [1.0, 1.0, 1.0, 1.0])

  def test_get_segments_from_frame_predictions_mr_with_fpn(self):
    class_probs = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.int32)
    displacements = np.array([
        [
            [-5.00000000e-01, 1.00000000e00],
            [5.00000000e-01, 0.00000000e00],
            [1.50000000e00, -1.00000000e00],
            [2.50000000e00, -2.00000000e00],
            [3.50000000e00, -3.00000000e00],
            [4.50000000e00, -4.00000000e00],
            [5.50000000e00, -5.00000000e00],
            [6.50000000e00, -6.00000000e00],
            [-5.00000000e-01, 1.00000000e00],
            [1.50000000e00, -1.00000000e00],
            [3.50000000e00, -3.00000000e00],
            [5.50000000e00, -5.00000000e00],
            [-5.00000000e-01, 1.00000000e00],
            [3.50000000e00, -3.00000000e00],
        ],
        [
            [-1.50000000e00, 2.50000000e00],
            [-5.00000000e-01, 1.50000000e00],
            [5.00000000e-01, 5.00000000e-01],
            [1.50000000e00, -5.00000000e-01],
            [2.50000000e00, -1.50000000e00],
            [3.50000000e00, -2.50000000e00],
            [4.50000000e00, -3.50000000e00],
            [5.50000000e00, -4.50000000e00],
            [-1.50000000e00, 2.50000000e00],
            [5.00000000e-01, 5.00000000e-01],
            [2.50000000e00, -1.50000000e00],
            [4.50000000e00, -3.50000000e00],
            [-1.50000000e00, 2.50000000e00],
            [2.50000000e00, -1.50000000e00],
        ],
        [
            [-4.00000000e00, 5.00000000e00],
            [-3.00000000e00, 4.00000000e00],
            [-2.00000000e00, 3.00000000e00],
            [-1.00000000e00, 2.00000000e00],
            [0.00000000e00, 1.00000000e00],
            [1.00000000e00, 0.00000000e00],
            [2.00000000e00, -1.00000000e00],
            [3.00000000e00, -2.00000000e00],
            [-4.00000000e00, 5.00000000e00],
            [-2.00000000e00, 3.00000000e00],
            [0.00000000e00, 1.00000000e00],
            [2.00000000e00, -1.00000000e00],
            [-4.00000000e00, 5.00000000e00],
            [0.00000000e00, 1.00000000e00],
        ],
        [
            [-6.50000000e00, 7.00000000e00],
            [-5.50000000e00, 6.00000000e00],
            [-4.50000000e00, 5.00000000e00],
            [-3.50000000e00, 4.00000000e00],
            [-2.50000000e00, 3.00000000e00],
            [-1.50000000e00, 2.00000000e00],
            [-5.00000000e-01, 1.00000000e00],
            [5.00000000e-01, 0.00000000e00],
            [-6.50000000e00, 7.00000000e00],
            [-4.50000000e00, 5.00000000e00],
            [-2.50000000e00, 3.00000000e00],
            [-5.00000000e-01, 1.00000000e00],
            [-6.50000000e00, 7.00000000e00],
            [-2.50000000e00, 3.00000000e00],
        ],
        [
            [5.00000000e02, -5.00000000e02],
            [5.01000000e02, -5.01000000e02],
            [5.02000000e02, -5.02000000e02],
            [5.03000000e02, -5.03000000e02],
            [5.04000000e02, -5.04000000e02],
            [5.05000000e02, -5.05000000e02],
            [5.06000000e02, -5.06000000e02],
            [5.07000000e02, -5.07000000e02],
            [5.00000000e+02, -5.00000000e+02],
            [5.02000000e+02, -5.02000000e+02],
            [5.04000000e+02, -5.04000000e+02],
            [5.06000000e+02, -5.06000000e+02],
            [5.00000000e+02, -5.00000000e+02],
            [5.04000000e+02, -5.04000000e+02],
        ],  # mask out
    ], dtype=np.float32)

    input_mask = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        dtype=np.int32,
    )
    caption_mask = np.array(
        [1, 1, 1, 1, 0],
        dtype=np.int32,
    )
    total_frames = np.array([15], dtype=np.int32)
    actual_class_probs, actual_segments = (
        postprocessing_utils.get_segments_from_frame_predictions_mr(
            class_probs,
            displacements,
            input_mask=input_mask,
            caption_mask=caption_mask,
            total_frames=total_frames,
            stride=1,
            sampling_strategy='linspace',
            displacement_normalizer='none',
            secs_per_timestep=1.0,
            feature_pyramid_config=ml_collections.ConfigDict({
                'num_features_level0': 8,
                'feature_pyramid_downsample_stride': 2,
                'feature_pyramid_levels': [0, 1, 2],
            }),
        )
    )
    cls_probs = class_probs[caption_mask.astype(bool)]
    cls_probs = cls_probs[:, input_mask.astype(bool)]
    self.assertAllClose(actual_class_probs, cls_probs)
    expected_segments = np.array([
        [
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2],
        ],
        [
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
            [3, 5],
        ],
        [
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
            [8, 10],
        ],
        [
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
            [13, 14],
        ],
    ])
    self.assertAllClose(actual_segments, expected_segments)


if __name__ == '__main__':
  tf.test.main()
