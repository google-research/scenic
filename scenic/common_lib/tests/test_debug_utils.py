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

"""Tests for debug_utils."""

from absl.testing import absltest
from scenic.common_lib import debug_utils


class ConfigDictWithAccessRecordTest(absltest.TestCase):
  """Tests for ConfigDictWithAccessRecord."""

  def test_getattr_access(self):
    config = debug_utils.ConfigDictWithAccessRecord()
    config.a = None
    config.b = None
    config.reset_access_record()
    _ = config.a  # Performs config access that should be recorded.
    self.assertSetEqual(config.get_not_accessed(), {'config.b'})

  def test_getitem_access(self):
    config = debug_utils.ConfigDictWithAccessRecord()
    config.a = None
    config.b = None
    config.reset_access_record()
    _ = config['a']  # Performs config access that should be recorded.
    self.assertSetEqual(config.get_not_accessed(), {'config.b'})

  def test_nested_access(self):
    config = debug_utils.ConfigDictWithAccessRecord()
    config.nested = debug_utils.ConfigDictWithAccessRecord()
    config.nested.a = None
    config.nested.b = None
    config.reset_access_record()
    _ = config.nested.a  # Performs config access that should be recorded.
    self.assertSetEqual(config.get_not_accessed(), {'config.nested.b'})

  def test_reset_access_record(self):
    config = debug_utils.ConfigDictWithAccessRecord()
    config.a = None
    _ = config.a  # Performs config access that should be recorded.
    config.reset_access_record()
    self.assertSetEqual(config.get_not_accessed(), {'config.a'})

  def test_reset_access_record_nested(self):
    config = debug_utils.ConfigDictWithAccessRecord()
    config.nested = debug_utils.ConfigDictWithAccessRecord()
    config.nested.a = None
    _ = config.nested.a  # Performs config access that should be recorded.
    config.reset_access_record()
    self.assertSetEqual(config.get_not_accessed(), {'config.nested.a'})


if __name__ == '__main__':
  absltest.main()
