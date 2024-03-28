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

"""Tests for config_utils."""
from absl.testing import parameterized


from scenic.projects.mtv import config_utils
import tensorflow as tf


class ConfigUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('B/2', (16, 16, 2)),
      ('B/18x2', (18, 18, 2)),
      ('B/18x18x2', (18, 18, 2)),
  )
  def test_parse_view_configs_one_view(self, variant, expected_patch_size):
    view_configs = config_utils.parse_view_configs(variant)
    self.assertLen(view_configs, 1)
    self.assertDictEqual(
        view_configs[0].to_dict(), {
            'hidden_size': 768,
            'num_heads': 12,
            'mlp_dim': 3072,
            'num_layers': 12,
            'patches': {
                'size': expected_patch_size
            },
        })

  @parameterized.parameters(
      ('S/8+B/4+H/2', (16, 16, 8), (16, 16, 4), (14, 14, 2)),
      ('S/14x8+B/12x4+H/18x2', (14, 14, 8), (12, 12, 4), (18, 18, 2)),
      ('S/14x14x8+B/12x12x4+H/18x18x2', (14, 14, 8), (12, 12, 4), (18, 18, 2)),
  )
  def test_parse_view_configs_threeview(self, variant,
                                        expected_patch_size_view0,
                                        expected_patch_size_view1,
                                        expected_patch_size_view2):
    view_configs = config_utils.parse_view_configs(variant)
    self.assertLen(view_configs, 3)
    self.assertDictEqual(
        view_configs[0].to_dict(), {
            'hidden_size': 384,
            'num_heads': 6,
            'mlp_dim': 1536,
            'num_layers': 12,
            'patches': {
                'size': expected_patch_size_view0
            },
        })
    self.assertDictEqual(
        view_configs[1].to_dict(), {
            'hidden_size': 768,
            'num_heads': 12,
            'mlp_dim': 3072,
            'num_layers': 12,
            'patches': {
                'size': expected_patch_size_view1
            },
        })
    self.assertDictEqual(
        view_configs[2].to_dict(), {
            'hidden_size': 1280,
            'num_heads': 16,
            'mlp_dim': 5120,
            'num_layers': 32,
            'patches': {
                'size': expected_patch_size_view2
            },
        })


if __name__ == '__main__':
  tf.test.main()
