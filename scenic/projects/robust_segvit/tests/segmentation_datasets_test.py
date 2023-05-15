"""Tests for segmentation_datasets."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import ml_collections
from scenic.projects.robust_segvit.datasets import segmentation_datasets

EXPECTED_DATASETS = [
    ('cityscapes', 'cityscapes', 'validation[:32]'),
    ('ade20k', 'ade20k', 'validation[:32]'),
    ('street_hazards', 'street_hazards', 'validation[:32]'),
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
    config.dataset_configs.validation_split = val_split
    _, dataset, _, _ = segmentation_datasets.get_dataset(**config)
    batch = next(dataset)
    self.assertEqual(
        batch['inputs'].shape,
        (num_shards, config.eval_batch_size // num_shards, 120, 120, 3))


if __name__ == '__main__':
  absltest.main()
