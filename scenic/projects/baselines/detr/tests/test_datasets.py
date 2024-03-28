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
