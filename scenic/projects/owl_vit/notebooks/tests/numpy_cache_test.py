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
