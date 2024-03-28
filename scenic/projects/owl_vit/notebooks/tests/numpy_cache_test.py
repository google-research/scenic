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

"""Tests for numpy_cache."""

from absl.testing import absltest
import numpy as np
from scenic.projects.owl_vit.notebooks import numpy_cache


class NumpyCacheTest(absltest.TestCase):

  def test_caching(self):
    """Tests that the function code is called only once for the same input."""

    side_effects = 0

    @numpy_cache.lru_cache(maxsize=10)
    def cached_function(a):
      nonlocal side_effects
      side_effects += 1
      return a

    self.assertEqual(side_effects, 0)

    _ = cached_function(np.zeros(1))
    self.assertEqual(side_effects, 1)

    _ = cached_function(np.ones(1))
    self.assertEqual(side_effects, 2)

    _ = cached_function(np.zeros(1))
    self.assertEqual(side_effects, 2)

  def test_non_numpy_args(self):
    """Tests that non-Numpy arguments are passed through correctly."""

    @numpy_cache.lru_cache(maxsize=1)
    def cached_function(a):
      return a

    arg = ('test',)
    out = cached_function(arg)
    self.assertEqual(out, arg)

  def test_kwargs(self):
    """Tests that keyword arguments are passed through correctly."""

    @numpy_cache.lru_cache(maxsize=1)
    def cached_function(*, a, b):
      return a, b

    a, b = cached_function(b='b', a='a')
    self.assertEqual(a, 'a')
    self.assertEqual(b, 'b')


if __name__ == '__main__':
  absltest.main()
