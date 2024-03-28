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

"""Tests for segmentation_datasets."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import ml_collections
from scenic.projects.robust_segvit.datasets import segmentation_datasets

EXPECTED_DATASETS = [
    ('ade20k', 'ade20k', 'validation'),
]


class SegmentationVariantsTest(parameterized.TestCase):

  @parameterized.named_parameters(EXPECTED_DATASETS)
  def test_available(self, name, val_split):
    """Test we can load a corrupted dataset."""
    num_shards = jax.local_device_count()
    config = ml_collections.ConfigDict()
    config.batch_size = num_shards*2
    config.eval_batch_size = num_shards*2
    config.num_shards = num_shards

    config.rng = jax.random.PRNGKey(0)
    config.dataset_configs = ml_collections.ConfigDict()
    config.dataset_configs.train_target_size = (120, 120)
    config.dataset_configs.name = name
    config.dataset_configs.denoise = None
    config.dataset_configs.use_timestep = 0
    config.dataset_configs.val_split = val_split
    dataset = segmentation_datasets.get_dataset(**config)
    batch = next(dataset.valid_iter)
    self.assertEqual(
        batch['inputs'].shape,
        (num_shards, config.eval_batch_size // num_shards, 120, 120, 3))


if __name__ == '__main__':
  absltest.main()
