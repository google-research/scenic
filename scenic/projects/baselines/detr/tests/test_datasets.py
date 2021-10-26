"""Unit tests for datasets."""

from absl.testing import absltest
from absl.testing import parameterized
from scenic.dataset_lib import datasets
from scenic.projects.baselines.detr import input_pipeline_detection  # pylint: disable=unused-import


EXPECTED_DATASETS = frozenset([
    'coco_detr_detection',
])


class DatasetsTest(parameterized.TestCase):
  """Unit tests for datasets.py."""

  @parameterized.named_parameters(*zip(EXPECTED_DATASETS, EXPECTED_DATASETS))
  def test_available(self, name):
    """Test the a given dataset is available."""
    self.assertIsNotNone(datasets.get_dataset(name))


if __name__ == '__main__':
  absltest.main()
