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

"""Tests for action_segmentation_base_model."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import jax_utils
import jax
import ml_collections
import numpy as np
from scenic.projects.unloc import action_segmentation_base_model


class MockActionSegmentationModel(
    action_segmentation_base_model.ActionSegmentationModel):
  """A mock action segmentation model for testing purposes."""

  def __init__(self, config: ml_collections.ConfigDict):
    dataset_meta_data = {}
    super().__init__(config, dataset_meta_data)

  def build_flax_model(self):
    pass

  def default_flax_model_config(self):
    pass


class ActionSegmentationBaseModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.logits = np.array([
        [
            [1.2, -0.9, 0.4],  # class 0
            [-0.4, -0.8, -0.1],  # background
            [-1.0, -1.0, -1.0],  # background
            [-1.0, -1.0, -1.0],  # background
        ],
        [
            [-1.0, -1.0, -1.0],  # background
            [-1.0, -1.0, -1.0],  # background
            [1.2, 0.9, 0.4],  # class 0
            [0.4, 0.8, 0.1],  # class 1
        ],
    ])  # shape is (2, 4, 3).
    self.batch = {
        'batch_mask':
            np.ones((2,), dtype=np.int32),
        'inputs': {
            'input_mask':
                np.array([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=np.int32),
        },
        'label':
            np.array(
                [
                    [
                        [1, 0, 0],  # class 0
                        [0, 0, 0],  # background
                        [0, 0, 0],  # background
                        [0, 0, 0],  # background
                    ],
                    [
                        [0, 1, 0],  # class 1
                        [0, 0, 0],  # background
                        [0, 1, 0],  # class 1
                        [0, 1, 0],  # class 1
                    ],
                ],
                dtype=np.int32),  # shape is (2, 4, 3).
    }

  def test_action_segmentation_model_multi_class_loss_function(self):
    config = ml_collections.ConfigDict()
    model = MockActionSegmentationModel(config)
    loss = model.loss_function(self.logits, self.batch)
    self.assertGreater(loss, 0.0)

  def test_action_segmentation_model_get_metrics_fn(self):
    config = ml_collections.ConfigDict()
    model = MockActionSegmentationModel(config)
    metrics_fn = jax.pmap(model.get_metrics_fn(), axis_name='batch')
    logits, batch = jax_utils.replicate((self.logits, self.batch))
    metrics = metrics_fn(logits, batch)
    self.assertSetEqual(
        set(metrics.keys()), {'frame_accuracy', 'sigmoid_classification_loss'})
    metrics = jax_utils.unreplicate(metrics)
    self.assertAlmostEqual(metrics['frame_accuracy'][0], 5)
    self.assertAlmostEqual(metrics['frame_accuracy'][1], 7)
    self.assertGreaterEqual(metrics['sigmoid_classification_loss'][0], 0)
    self.assertAlmostEqual(metrics['sigmoid_classification_loss'][1], 7)

  def test_action_segmentation_model_one_class_loss_function(self):
    logits = np.array([
        [
            [1.2,
             -0.8,
             -1.0,
             -1.0
             ],
        ],
        [
            [-1.0,
             -1.0,
             1.2,
             0.8
             ],
        ],
    ])  # shape is (2, 4).
    batch = {
        'batch_mask':
            np.ones((2,), dtype=np.int32),
        'inputs': {
            'input_mask':
                np.array([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=np.int32),
        },
        'label':
            np.array(
                [
                    [
                        [1],  # class 0
                        [0],  # background
                        [0],  # background
                        [0],  # background
                    ],
                    [
                        [0],  # background
                        [0],  # background
                        [1],  # class 0
                        [0],  # background
                    ],
                ],
                dtype=np.int32),  # shape is (2, 4, 1).
    }
    config = ml_collections.ConfigDict()
    model = MockActionSegmentationModel(config)
    loss = model.loss_function(logits, batch)
    self.assertGreater(loss, 0.0)


if __name__ == '__main__':
  absltest.main()
