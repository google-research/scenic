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

"""A functools.lru_cache that works for functions taking Numpy arguments."""

import functools
import numpy as np


def lru_cache(*args, **kwargs):
  """Wraps functools.lru_cache to make it compatible with Numpy arrays."""

  def decorator(function):

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
      hashable_args = [_make_hashable(arg) for arg in args]
      hashable_kwargs = {k: _make_hashable(v) for k, v in kwargs.items()}
      return cached_wrapper(*hashable_args, **hashable_kwargs)

    @functools.lru_cache(*args, **kwargs)
    def cached_wrapper(*hashable_args, **hashable_kwargs):
      args = [_undo_make_hashable(arg) for arg in hashable_args]
      kwargs = {k: _undo_make_hashable(v) for k, v in hashable_kwargs.items()}
      return function(*args, **kwargs)

    return wrapper

  return decorator


class _HashableNumpyArray:
  """Simple wrapper that makes Numpy arrays hashable."""

  def __init__(self, array: np.ndarray):
    self.array = array
    self._bytes = array.data.tobytes()

  def __hash__(self):
    return hash(self._bytes)

  def __eq__(self, other):
    return isinstance(other, type(self)) and self._bytes == other._bytes


def _make_hashable(x):
  return _HashableNumpyArray(x) if isinstance(x, np.ndarray) else x


def _undo_make_hashable(x):
  return x.array if isinstance(x, _HashableNumpyArray) else x
