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

"""Unit tests for detr input pipeline."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_collections
from scenic.projects.baselines.deformable_detr import input_pipeline_detection
from scenic.projects.baselines.detr.tests import test_util
import tensorflow_datasets as tfds


class DeformableDETRInputPipelineTests(parameterized.TestCase):
  """Unit tests for detr test_input_pipeline_detection.py."""

  def test_dataset_builder_coco_deformable_detr_detection(self):
    """Tests dataset builder for coco_deformable_detr_detection."""
    num_shards = jax.local_device_count()
    batch_size = num_shards * 2
    eval_batch_size = num_shards * 2

    dataset_config = ml_collections.ConfigDict()
    dataset_config.max_size = 1333
    dataset_config.valid_max_size = 1333

    with tfds.testing.mock_data(
        num_examples=50,
        as_dataset_fn=test_util.generate_fake_dataset(num_examples=50)):
      dataset = input_pipeline_detection.get_dataset(
          batch_size=batch_size,
          eval_batch_size=eval_batch_size,
          dataset_configs=dataset_config,
          num_shards=num_shards)

      # A dataset should at least provide `train_iter` and `valid_iter`.
      self.assertIsNotNone(dataset.train_iter)
      self.assertIsNotNone(dataset.valid_iter)

      train_batch = next(dataset.train_iter)
      eval_batch = next(dataset.valid_iter)

    # Check shapes.
    # Tests first two shape dimensions.
    expected_shape = [num_shards, batch_size // num_shards, 1333, 1333, 3]
    expected_shape_eval = [
        num_shards, eval_batch_size // num_shards, 1333, 1333, 3
    ]
    self.assertSequenceEqual(train_batch['inputs'].shape, expected_shape)
    self.assertSequenceEqual(eval_batch['inputs'].shape, expected_shape_eval)

    self.assertEqual(train_batch['inputs'].shape[:-1],
                     train_batch['padding_mask'].shape)
    self.assertEqual(eval_batch['inputs'].shape[:-1],
                     eval_batch['padding_mask'].shape)

  def test_dtypes_input_pipeline_detection(self):
    """Tests data type of dataset coco_deformable_detr_detection."""

    with tfds.testing.mock_data(
        num_examples=50,
        as_dataset_fn=test_util.generate_fake_dataset(num_examples=50)):
      num_shards = jax.local_device_count()
      for dt in ['float32']:
        dataset = input_pipeline_detection.get_dataset(
            batch_size=num_shards * 2,
            eval_batch_size=num_shards * 2,
            num_shards=num_shards,
            dtype_str=dt)

        train_batch = next(dataset.train_iter)
        eval_batch = next(dataset.valid_iter)

        # Check dtype.
        self.assertEqual(train_batch['inputs'].dtype, getattr(jnp, dt))
        self.assertEqual(eval_batch['inputs'].dtype, getattr(jnp, dt))

if __name__ == '__main__':
  absltest.main()
