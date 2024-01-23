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

"""Unit tests for functions in common_utils.py."""

from absl.testing import absltest
from absl.testing import parameterized
from scenic.common_lib import common_utils
from scenic.model_lib.base_models import classification_model as test_model_module  # pylint: disable=unused-import


class RecursiveReloadTest(parameterized.TestCase):
  """Tests recursive_reload."""

  def test_recursive_reload(self):
    """Tests that recursive_reload returns without error."""
    global test_model_module
    test_model_module = common_utils.recursive_reload(
        test_model_module, package_restrict='scenic')

if __name__ == '__main__':
  absltest.main()
