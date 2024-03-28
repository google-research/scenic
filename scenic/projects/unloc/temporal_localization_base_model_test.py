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

"""Tests for temporal_localization_base_model."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import jax_utils
import jax
import ml_collections
import numpy as np
from scenic.projects.unloc import temporal_localization_base_model


class MockTemporalLocalizationModel(
    temporal_localization_base_model.TemporalLocalizationModel):
  """A mock temporal localization model for testing purposes."""

  def __init__(self, config: ml_collections.ConfigDict):
    dataset_meta_data = {}
    super().__init__(config, dataset_meta_data)

  def build_flax_model(self):
    pass

  def default_flax_model_config(self):
    pass


class TemporalLocalizationBaseModelTest(parameterized.TestCase):

  @parameterized.parameters(
      (None, np.array([
          [1, 0, 1],
          [0, 0, 1],
      ], dtype=np.int32)),
      (np.array([1, 0], dtype=np.int32),
       np.array([
           [1, 0, 1],
           [0, 0, 0],
       ], dtype=np.int32)),
  )
  def test_weighted_top_one_correctly_classified(self, weights,
                                                 expected_correct):
    logits = np.array([
        [[1.2, -0.2, 0.1], [0.2, -0.4, -0.4], [-1.0, -1.0, -1.0]],
        [[-0.4, 0.5, -1.0], [2.0, 1.0, 3.0], [-1.0, -1.0, -1.0]],
    ],
                      dtype=np.float32)
    multihot_targets = np.array([
        [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ],
                                dtype=np.int32)
    correct = temporal_localization_base_model.weighted_top_one_correctly_classified(  # pylint: disable=line-too-long
        logits, multihot_targets, weights=weights)
    np.testing.assert_equal(correct, expected_correct)

  def test_weighted_top_one_correctly_classified_all_background(self):
    logits = np.array(
        [
            [[1.2, -0.2, 0.1], [0.2, -0.4, -0.4], [-1.0, -1.0, -1.0]],
            [[-0.4, 0.5, -1.0], [2.0, 1.0, 3.0], [-1.0, -1.0, -1.0]],
        ],
        dtype=np.float32,
    )
    multihot_targets = np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=np.int32,
    )
    correct = (
        temporal_localization_base_model.weighted_top_one_correctly_classified(
            logits, multihot_targets, weights=None
        )
    )
    expected_correct = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.int32)
    np.testing.assert_equal(correct, expected_correct)

  def test_weighted_top_one_correctly_classified_all_foreground(self):
    logits = np.array(
        [
            [[1.2, -0.2, 0.1], [0.2, -0.4, -0.4], [-1.0, -1.0, -1.0]],
            [[-0.4, 0.5, -1.0], [2.0, 1.0, 3.0], [-1.0, -1.0, -1.0]],
        ],
        dtype=np.float32,
    )
    multihot_targets = np.array(
        [
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        ],
        dtype=np.int32,
    )
    correct = (
        temporal_localization_base_model.weighted_top_one_correctly_classified(
            logits, multihot_targets, weights=None
        )
    )
    expected_correct = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int32)
    np.testing.assert_equal(correct, expected_correct)

  @parameterized.parameters(
      (None, 'iou'),
      (None, 'l1'),
      (None, 'center_offset_squared+iou'),
      (None, 'l1+iou'),
      (None, '0.5*l1+1.0*iou'),
      (np.ones((2, 8)), 'iou'),
      (np.ones((2, 8)), 'iou+center_offset_squared'),
      (np.ones((2, 8)), 'iou+l1'),
  )
  def test_weighted_unnormalized_iou_loss(self, weights, loss_type):
    batch_size, num_frames, num_classes = 2, 8, 10
    displacements = np.ones((batch_size, num_frames, num_classes, 2),
                            dtype=np.float32)
    gt_displacements = np.ones((batch_size, num_frames, num_classes, 2),
                               dtype=np.float32)
    label = np.zeros((batch_size, num_frames, num_classes), dtype=np.int32)
    label[..., 0] = 1
    box_loss = temporal_localization_base_model.weighted_unnormalized_box_regression_loss(
        displacements, gt_displacements, label, weights, loss_type
    )
    self.assertTupleEqual(box_loss.shape, (batch_size, num_frames, num_classes))

  @parameterized.parameters(
      (None, 2),
      (np.array([1, 0], dtype=np.int32), 1),
  )
  def test_num_positive_frames(self, weights, expected_output):
    label = np.array([
        [[1, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0]],
    ],
                     dtype=np.int32)
    actual = temporal_localization_base_model.num_positive_frames(
        label, weights)
    self.assertEqual(actual, expected_output)

  @parameterized.parameters(
      ('sigmoid', 'iou', True),
      ('focal', 'iou+l1', True),
      ('focal', 'iou+l1', False),
      ('focal', '0.8*iou+l1', True),
      ('focal', 'iou+0.5*center_offset_squared', True),
  )
  def test_temporal_localization_model_loss_function(
      self, cls_loss_type, box_loss_type, output_per_class_displacements
  ):
    config = ml_collections.ConfigDict({
        'classification_loss_type': cls_loss_type,
        'box_loss_type': box_loss_type,
        'output_per_class_displacements': output_per_class_displacements,
    })
    model = MockTemporalLocalizationModel(config)
    batch_size, num_frames, num_classes = 2, 8, 10
    if output_per_class_displacements:
      logits = np.ones(
          (batch_size, num_frames, num_classes * 3), dtype=np.float32
      )
    else:
      logits = np.ones(
          (batch_size, num_frames, num_classes + 2), dtype=np.float32
      )
    batch = {
        'batch_mask':
            np.ones((batch_size), dtype=np.int32),
        'inputs': {
            'input_mask': np.ones((batch_size, num_frames), dtype=np.int32),
        },
        'label':
            np.zeros((batch_size, num_frames, num_classes), dtype=np.int32),
        'displacements':
            np.zeros((batch_size, num_frames, num_classes, 2),
                     dtype=np.float32),
    }
    if output_per_class_displacements:
      batch['displacements'] = np.zeros(
          (batch_size, num_frames, num_classes, 2), dtype=np.float32
      )
    else:
      batch['displacements'] = np.zeros(
          (batch_size, num_frames, 2), dtype=np.float32
      )
    batch['label'][..., 0] = 1
    loss = model.loss_function(logits, batch)
    self.assertGreater(loss, 0.0)

  @parameterized.parameters(
      (
          'sigmoid',
          'iou',
          True,
          {'iou_loss': 2.0 - 8 / 9 - 6 / 8},
          8 / 9 + 6 / 8,
      ),
      (
          'focal',
          'iou+center_offset_squared',
          True,
          {
              'iou_loss': 2.0 - 8 / 9 - 6 / 8,
              'center_offset_squared_loss': (0.05 / 0.9) ** 2,
          },
          8 / 9 + 6 / 8,
      ),
      (
          'focal',
          'iou+l1',
          True,
          {
              'iou_loss': 2.0 - 8 / 9 - 6 / 8,
              'l1_loss': 1 / 9 + 2 / 8,
          },
          8 / 9 + 6 / 8,
      ),
      (
          'focal',
          'iou+l1',
          False,
          {
              'iou_loss': 2.0 - 8 / 9 - 6 / 8,
              'l1_loss': 1 / 9 + 2 / 8,
          },
          8 / 9 + 6 / 8,
      ),
  )
  def test_temporal_localization_model_get_metrics_fn(
      self,
      cls_loss_type,
      box_loss_type,
      output_per_class_displacements,
      expected_box_loss,
      expected_mean_iou,
  ):
    config = ml_collections.ConfigDict({
        'classification_loss_type': cls_loss_type,
        'box_loss_type': box_loss_type,
        'output_per_class_displacements': output_per_class_displacements,
    })
    model = MockTemporalLocalizationModel(config)
    metrics_fn = jax.pmap(model.get_metrics_fn(), axis_name='batch')
    class_logits = np.array([
        [[[1.2], [-0.9], [0.4]], [[-0.4], [-0.8], [-0.1]]],
        [[[1.2], [0.9], [0.4]], [[0.4], [0.8], [0.1]]],
    ])  # shape is (2, 2, 3, 1).
    if output_per_class_displacements:
      pred_displacements = np.array([
          [
              [[0.4, 0.4], [0.0, 0.1], [0.2, 0.0]],
              [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
          ],
          [
              [[1.2, 0.9], [0.4, 0.4], [0.4, 0.2]],
              [[0.4, 0.8], [0.1, 0.3], [0.3, 0.3]],
          ],
      ])  # shape is (2, 2, 3, 2).
      logits = np.concatenate([class_logits, pred_displacements], axis=-1)
      logits = logits.reshape((2, 2, 9))
    else:
      pred_displacements = np.array([
          [[0.4, 0.4], [0.1, 0.2]],
          [[0.4, 0.4], [0.4, 0.8]],
      ])  # shape is (2, 2, 2).
      logits = np.concatenate(
          [np.squeeze(class_logits, axis=-1), pred_displacements], axis=-1
      )
    batch = {
        'batch_mask':
            np.ones((2,), dtype=np.int32),
        'inputs': {
            'input_mask': np.ones((2, 2), dtype=np.int32),
        },
        'label':
            np.array([
                [[1, 0, 0], [0, 0, 0]],
                [[0, 1, 0], [0, 0, 0]],
            ],
                     dtype=np.int32),
        'displacements':
            np.array([
                [
                    [[0.5, 0.4], [0, 0], [0, 0]],
                    [[0, 0], [0, 0], [0, 0]],
                ],
                [
                    [[0, 0], [0.3, 0.3], [0, 0]],
                    [[0, 0], [0, 0], [0, 0]],
                ],
            ],
                     dtype=np.float32),
    }
    if output_per_class_displacements:
      batch['displacements'] = np.array(
          [
              [
                  [[0.5, 0.4], [0, 0], [0, 0]],
                  [[0, 0], [0, 0], [0, 0]],
              ],
              [
                  [[0, 0], [0.3, 0.3], [0, 0]],
                  [[0, 0], [0, 0], [0, 0]],
              ],
          ],
          dtype=np.float32,
      )
    else:
      batch['displacements'] = np.array(
          [
              [[0.5, 0.4], [0, 0]],
              [[0.3, 0.3], [0, 0]],
          ],
          dtype=np.float32,
      )  # shape (2, 2, 2)
    logits, batch = jax_utils.replicate((logits, batch))
    metrics = metrics_fn(logits, batch)
    expected_cls_loss_key = ('sigmoid_classification_loss' if cls_loss_type
                             == 'sigmoid' else 'focal_classification_loss')
    expected_box_loss_keys = set(expected_box_loss.keys())
    self.assertSetEqual(
        set(metrics.keys()),
        {'precision@1', expected_cls_loss_key, 'mean_iou'}
        | expected_box_loss_keys,
    )
    metrics = jax_utils.unreplicate(metrics)
    self.assertAlmostEqual(
        metrics['mean_iou'][0], expected_mean_iou, delta=1e-4)
    self.assertAlmostEqual(metrics['mean_iou'][1], 2)
    for key in expected_box_loss_keys:
      self.assertAlmostEqual(
          metrics[key][0], expected_box_loss[key], delta=1e-4
      )
      self.assertAlmostEqual(metrics[key][1], 2)
    self.assertAlmostEqual(metrics['precision@1'][0], 2.0)
    self.assertAlmostEqual(metrics['precision@1'][1], 4)
    self.assertGreaterEqual(metrics[expected_cls_loss_key][0], 0)
    self.assertAlmostEqual(metrics[expected_cls_loss_key][1], 4)


if __name__ == '__main__':
  absltest.main()
