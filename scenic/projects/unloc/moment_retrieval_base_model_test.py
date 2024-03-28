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

"""Tests for moment_retrieval_base_model."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import jax_utils
import jax
import ml_collections
import numpy as np
from scenic.projects.unloc import moment_retrieval_base_model


class MockMomentRetrievalModel(moment_retrieval_base_model.MomentRetrievalModel
                              ):
  """A mock moment retrieval model for testing purposes."""

  def __init__(self, config: ml_collections.ConfigDict):
    dataset_meta_data = {}
    super().__init__(config, dataset_meta_data)

  def build_flax_model(self):
    pass

  def default_flax_model_config(self):
    pass


class MomentRetrievalBaseModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.class_logits = np.array([
        [
            [[1.2], [0.4], [-0.9]],
            [[-0.4], [0.8], [0.1]],
            [[-1.0], [-1.0], [-1.0]],
            [[-1.0], [-1.0], [-1.0]],
        ],
        [
            [[-1.0], [-1.0], [-1.0]],
            [[-1.0], [-1.0], [-1.0]],
            [[1.2], [0.9], [0.4]],
            [[0.1], [0.1], [0.1]],
        ],
    ])  # shape is (2, 2*2, 3, 1).
    pred_displacements = np.array([
        [
            [[0.4, 0.4], [0.0, 0.1], [0.2, 0.0]],
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[1.2, 0.9], [0.4, 0.4], [0.4, 0.2]],
            [[0.4, 0.8], [0.1, 0.3], [0.3, 0.3]],
        ],
    ])  # shape is (2, 2*2, 3, 2).
    self.logits = np.concatenate([self.class_logits, pred_displacements],
                                 axis=-1)
    self.batch = {
        'batch_mask':
            np.ones((2,), dtype=np.int32),
        'inputs': {
            'input_mask': np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int32),
            'caption_mask': np.array([[1, 1], [1, 0]], dtype=np.int32),
        },
        'label':
            np.array([
                [[[1], [1], [0]], [[0], [1], [1]]],
                [[[1], [1], [0]], [[0], [0], [0]]],
            ],
                     dtype=np.int32),  # shape is (2, 2, 3, 1).
        'displacements':
            np.array([
                [
                    [[0.5, 0.4], [0.3, 0.2], [0, 0]],
                    [[0, 0], [0.4, 0.3], [0.5, 0.6]],
                ],
                [
                    [[0.2, 0.1], [0.3, 0.3], [0, 0]],
                    [[0, 0], [0, 0], [0, 0]],
                ],
            ],
                     dtype=np.float32),  # shape is (2, 2, 3, 2).
    }

  @parameterized.parameters(
      ('sigmoid', 'iou', True),
      ('focal', 'iou+center_offset_squared', True),
      ('sigmoid', 'iou', False),
      ('focal', 'iou+center_offset_squared', False),
  )
  def test_moment_retrieval_model_loss_function(
      self, cls_loss_type, box_loss_type, all_gather_loss
  ):
    config = ml_collections.ConfigDict({
        'classification_loss_type': cls_loss_type,
        'box_loss_type': box_loss_type,
        'all_gather_loss': all_gather_loss,
    })
    model = MockMomentRetrievalModel(config)
    loss = model.loss_function(self.logits, self.batch)
    self.assertGreater(loss, 0.0)

  @parameterized.parameters(
      ('sigmoid', 'iou', True, {'iou_loss': 2.268258}, 3.731742),
      (
          'focal',
          'iou+center_offset_squared',
          True,
          {
              'iou_loss': 2.268258,
              'center_offset_squared_loss': 0.060979,
          },
          3.731742,
      ),
      ('sigmoid', 'iou', False, {'iou_loss': 2.268258}, 3.731742),
      (
          'focal',
          'iou+center_offset_squared',
          False,
          {
              'iou_loss': 2.268258,
              'center_offset_squared_loss': 0.060979,
          },
          3.731742,
      ),
  )
  def test_moment_retrieval_model_get_metrics_fn(
      self, cls_loss_type, box_loss_type, all_gather_loss,
      expected_box_loss, expected_mean_iou
  ):
    config = ml_collections.ConfigDict({
        'classification_loss_type': cls_loss_type,
        'box_loss_type': box_loss_type,
        'all_gather_loss': all_gather_loss,
    })
    model = MockMomentRetrievalModel(config)
    metrics_fn = jax.pmap(model.get_metrics_fn(), axis_name='batch')
    logits, batch = jax_utils.replicate((self.logits, self.batch))
    metrics = metrics_fn(logits, batch)
    expected_cls_loss_key = ('sigmoid_classification_loss' if cls_loss_type
                             == 'sigmoid' else 'focal_classification_loss')
    expected_box_loss_keys = set(expected_box_loss.keys())
    self.assertSetEqual(
        set(metrics.keys()),
        {'accuracy', expected_cls_loss_key, 'mean_iou'}
        | expected_box_loss_keys,
    )
    metrics = jax_utils.unreplicate(metrics)
    self.assertAlmostEqual(
        metrics['mean_iou'][0], expected_mean_iou, delta=1e-4)
    self.assertAlmostEqual(metrics['mean_iou'][1], 6)
    for key in expected_box_loss_keys:
      self.assertAlmostEqual(
          metrics[key][0], expected_box_loss[key], delta=1e-4
      )
      self.assertAlmostEqual(metrics[key][1], 6)
    if all_gather_loss:
      self.assertAlmostEqual(metrics['accuracy'][0], 16)
    else:
      self.assertAlmostEqual(metrics['accuracy'][0], 8)

    if all_gather_loss:
      self.assertAlmostEqual(metrics['accuracy'][1], 16)
    else:
      self.assertAlmostEqual(metrics['accuracy'][1], 8)

    self.assertGreaterEqual(metrics[expected_cls_loss_key][0], 0)

    if all_gather_loss:
      self.assertGreaterEqual(metrics[expected_cls_loss_key][1], 16)
    else:
      self.assertGreaterEqual(metrics[expected_cls_loss_key][1], 8)


if __name__ == '__main__':
  absltest.main()
