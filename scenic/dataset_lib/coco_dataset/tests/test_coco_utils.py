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

"""Unit tests for functions in coco_utils.py."""

from absl.testing import absltest
from absl.testing import parameterized
from scenic.dataset_lib.coco_dataset import coco_utils


class CocoUtilsTest(parameterized.TestCase):
  """Test COCO utils."""

  @parameterized.parameters(
      ('coco/2017',),
      ('coco/2017_panoptic',),
      ('lvis',),
  )
  def get_label_map(self, tfds_name):
    """Test get_label_map."""
    label_map = coco_utils.get_label_map(tfds_name)
    self.assertIs(label_map, dict)
    self.assertTrue(all(isinstance(k, int) for k in label_map.keys()),
                    msg='Not all label map keys are of type int.')
    max_label = max(label_map.keys())
    self.assertSequenceEqual(range(max_label), label_map.keys())

  def test_get_label_map_unknown(self):
    """Test get_label_map for unknown TFDS name."""
    with self.assertRaisesWithPredicateMatch(
        ValueError, lambda m: m.args == ('Unsupported TFDS name: unknown',)):
      coco_utils.get_label_map('unknown')

if __name__ == '__main__':
  absltest.main()
