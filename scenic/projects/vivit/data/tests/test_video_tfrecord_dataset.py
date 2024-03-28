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

"""Unit tests for datasets."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_collections
from scenic.projects.vivit.data import video_tfrecord_dataset


class VideoTFRecordDatsetTest(parameterized.TestCase):
  """Unit tests for video_tfrecord_dataset.py."""

  @parameterized.named_parameters(
      ('1 test clip', 1, False, 0),
      ('1x3 test clips', 1, True, 0),
      ('4 test clips, prefetch', 4, False, 1),
      ('4x3 test clips, prefetch', 4, True, 1))
  def test_dataset_builder(self, num_test_clips, do_three_spatial_crops,
                           prefetch_to_device):
    """Tests dataset builder."""
    num_shards = jax.local_device_count()
    batch_size = num_shards * 3
    eval_batch_size = num_shards * 2

    dataset_configs = ml_collections.ConfigDict()
    dataset_configs.prefetch_to_device = prefetch_to_device
    dataset_configs.num_frames = 8
    dataset_configs.num_test_clips = num_test_clips
    dataset_configs.do_three_spatial_crops = do_three_spatial_crops

    dataset_configs.base_dir = '/path/to/dataset_root/'
    dataset_configs.tables = {
        'train': 'something-something-v2-train.rgb.tfrecord@128',
        'validation': 'something-something-v2-validation.rgb.tfrecord@128',
        'test': 'something-something-v2-validation.rgb.tfrecord@128'
    }
    dataset_configs.examples_per_subset = {
        'train': 168913,
        'validation': 24777,
        'test': 24777
    }
    dataset_configs.num_classes = 174

    print('Please set the correct dataset base directory and run'
          'this test again.')
    return

    dataset = video_tfrecord_dataset.get_dataset(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_shards=num_shards,
        dataset_configs=dataset_configs)

    self.assertIsNotNone(dataset.train_iter)
    self.assertIsNotNone(dataset.valid_iter)
    self.assertIsNotNone(dataset.test_iter)

    train_batch = next(dataset.train_iter)
    eval_batch = next(dataset.valid_iter)
    test_batch = next(dataset.test_iter)

    # Check shapes.
    num_spatial_crops = 3 if do_three_spatial_crops else 1
    expected_shape = jnp.array((num_shards, batch_size // num_shards) +
                               dataset.meta_data['input_shape'][1:])
    expected_shape_eval = jnp.array(
        (num_shards, eval_batch_size // num_shards) +
        dataset.meta_data['input_shape'][1:])
    expected_shape_test = jnp.array(
        (num_shards,
         eval_batch_size * num_test_clips * num_spatial_crops // num_shards) +
        dataset.meta_data['input_shape'][1:])
    self.assertTrue(
        jnp.array_equal(train_batch['inputs'].shape, expected_shape))
    self.assertTrue(
        jnp.array_equal(eval_batch['inputs'].shape, expected_shape_eval))
    self.assertTrue(
        jnp.array_equal(test_batch['inputs'].shape, expected_shape_test))

    # Check number of examples.
    self.assertEqual(dataset.meta_data['num_train_examples'], 168913)
    self.assertEqual(dataset.meta_data['num_eval_examples'], 24777)
    self.assertEqual(dataset.meta_data['num_test_examples'],
                     24777 * num_test_clips * num_spatial_crops)


if __name__ == '__main__':
  absltest.main()
