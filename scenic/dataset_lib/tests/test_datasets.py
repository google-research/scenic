# Copyright 2021 The Scenic Authors.
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

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
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
from scenic.dataset_lib import datasets

EXPECTED_DATASETS = frozenset([
    'cifar10',
    'cityscapes',
    'fashion_mnist',
    'imagenet',
    'oxford_pets',
    'svhn'])
UNEXPECTED_DATASETS = frozenset([
    'does_not_exist1',
    'does_not_exist2'])


class DatasetsTest(parameterized.TestCase):
  """Unit tests for datasets.py."""

  @parameterized.named_parameters(*zip(EXPECTED_DATASETS, EXPECTED_DATASETS))
  def test_available(self, name):
    """Test the a given dataset is available."""
    self.assertIsNotNone(datasets.get_dataset(name))

  @parameterized.named_parameters(
      *zip(UNEXPECTED_DATASETS, UNEXPECTED_DATASETS))
  def test_unavailable(self, name):
    """Test the a given dataset is NOT available."""
    with self.assertRaises(KeyError):
      datasets.get_dataset(name)

  @parameterized.named_parameters(*zip(EXPECTED_DATASETS, EXPECTED_DATASETS))
  def test_dataset_builder(self, ds):
    """Tests dataset builder."""
    num_shards = jax.local_device_count()
    batch_size = num_shards * 2
    eval_batch_size = num_shards * 1
    assert batch_size % num_shards == 0
    assert eval_batch_size % num_shards == 0

    dataset_builder = datasets.get_dataset(ds)
    dataset = dataset_builder(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_shards=num_shards)
    expected_shape = jnp.array([
        num_shards, batch_size // num_shards,
        dataset.meta_data['input_shape'][1],
        dataset.meta_data['input_shape'][2],
        dataset.meta_data['input_shape'][3]
    ])
    expected_shape_eval = jnp.array([
        num_shards, eval_batch_size // num_shards,
        dataset.meta_data['input_shape'][1],
        dataset.meta_data['input_shape'][2],
        dataset.meta_data['input_shape'][3]
    ])

    # A dataset should at least provide train_iter and valid_iter.
    self.assertIsNotNone(dataset.train_iter)
    self.assertIsNotNone(dataset.valid_iter)

    train_batch = next(dataset.train_iter)
    eval_batch = next(dataset.valid_iter)
    if dataset.test_iter:
      test_batch = next(dataset.test_iter)
    else:
      test_batch = None

    # Check shapes.
    self.assertTrue(
        jnp.array_equal(train_batch['inputs'].shape, expected_shape))
    self.assertTrue(
        jnp.array_equal(eval_batch['inputs'].shape, expected_shape_eval))

    # Checks for test_iter, if it is not None:
    if test_batch:
      self.assertTrue(
          jnp.array_equal(test_batch['inputs'].shape, expected_shape_eval))


if __name__ == '__main__':
  absltest.main()
