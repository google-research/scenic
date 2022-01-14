"""Unit tests for datasets."""

from absl.testing import absltest
from absl.testing import parameterized
from scenic.dataset_lib import datasets

EXPECTED_DATASETS = frozenset(['layout_denoise'])


class DatasetsTest(parameterized.TestCase):
  """Unit tests for datasets.py."""

  @parameterized.named_parameters(*zip(EXPECTED_DATASETS, EXPECTED_DATASETS))
  def test_available(self, name):
    """Test the a given dataset is available."""
    self.assertIsNotNone(datasets.get_dataset(name))


if __name__ == '__main__':
  absltest.main()
