"""Tests for segmentation_datasets."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import ml_collections
from scenic.projects.robust_segvit.datasets import segmentation_variants

EXPECTED_DATASETS = [
    ('ade20k_ind_c1', 'ade20k_ind_c', 'gaussian_noise', 1, 'validation[:32]'),
    ('ade20k_ind_c2', 'ade20k_ind_c', 'brightness', 1, 'validation[:32]'),
    ('ade20k_ind_c3', 'ade20k_ind_c', 'contrast', 1, 'validation[:32]'),
    ('ade20k_ood_open', 'ade20k_ood_open', None, None, 'validation[:32]'),
    ('street_hazards_open', 'street_hazards_open', None, None, 'validation[:32]'),
    ('cityscapes_c1', 'cityscapes_c', 'gaussian_noise', 1, 'validation[:32]'),
    ('cityscapes_c2', 'cityscapes_c', 'brightness', 1, 'validation[:32]'),
    ('cityscapes_c3', 'cityscapes_c', 'contrast', 1, 'validation[:32]'),
    ('fishyscapes/Static', 'fishyscapes/Static', None, None, 'validation[:30]'),
    ('street_hazards_c1', 'street_hazards_c', 'gaussian_noise', 1, 'validation[:32]'),
    ('street_hazards_c2', 'street_hazards_c', 'brightness', 1, 'validation[:32]'),
    ('street_hazards_c3', 'street_hazards_c', 'contrast', 1, 'validation[:32]'),
]


class SegmentationVariantsTest(parameterized.TestCase):

  @parameterized.named_parameters(EXPECTED_DATASETS)
  def test_available(self, name, corruption_type, corruption_level, val_split):
    """Test we can load a corrupted dataset."""
    num_shards = jax.local_device_count()
    config = ml_collections.ConfigDict()
    config.batch_size = num_shards*2
    config.eval_batch_size = num_shards*2
    config.num_shards = num_shards

    config.rng = jax.random.PRNGKey(0)
    config.dataset_configs = ml_collections.ConfigDict()
    config.dataset_configs.train_target_size = (120, 120)
    if corruption_type:
      config.dataset_configs.name = f'{name}_{corruption_type}_{corruption_level}'
    else:
      config.dataset_configs.name = name
    config.dataset_configs.denoise = None
    config.dataset_configs.use_timestep = 0
    config.dataset_configs.validation_split = val_split
    _, dataset, _, _ = segmentation_variants.get_dataset(**config)
    batch = next(dataset)
    self.assertEqual(
        batch['inputs'].shape,
        (num_shards, config.eval_batch_size // num_shards, 120, 120, 3))


if __name__ == '__main__':
  absltest.main()
