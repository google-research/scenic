# Copyright 2026 The Scenic Authors.
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

"""Greedy matching remains one-to-one across the input cost range."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scenic.model_lib.matchers import greedy_matcher


def reference(cost):
  transposed = cost.shape[0] > cost.shape[1]
  if transposed:
    cost = cost.T
  used_rows, used_cols, pairs = set(), set(), []
  for _ in range(cost.shape[0]):
    _, i, j = min(
        (cost[i, j].item(), i, j)
        for i in range(cost.shape[0])
        for j in range(cost.shape[1])
        if i not in used_rows and j not in used_cols
    )
    used_rows.add(i)
    used_cols.add(j)
    pairs.append((j, i) if transposed else (i, j))
  return np.asarray(pairs).T


@pytest.mark.parametrize(
    "dtype,value",
    [
        (jnp.float32, 1e8),
        (jnp.float32, -1e8),
        (jnp.float16, 4000),
        (jnp.bfloat16, 1024),
        (jnp.int32, 2147483647),
        (jnp.uint32, 4294967295),
    ],
)
@pytest.mark.parametrize("shape", [(3, 3), (2, 4), (4, 2)])
def test_saturated_costs_do_not_reuse_rows_or_columns(dtype, value, shape):
  cost = jnp.full((2,) + shape, value, dtype=dtype)
  expected = np.stack([reference(np.asarray(c)) for c in cost])
  for match in (greedy_matcher, jax.jit(greedy_matcher)):
    actual = np.asarray(match(cost))
    np.testing.assert_array_equal(actual, expected)
    for rows, cols in actual:
      assert len(np.unique(rows)) == min(shape)
      assert len(np.unique(cols)) == min(shape)


@pytest.mark.parametrize("shape", [(3, 3), (2, 4), (4, 2), (1, 3)])
def test_ordinary_costs_match_sequential_reference(shape):
  rng = np.random.default_rng(4)
  cost = jnp.asarray(rng.integers(-8, 9, size=(6,) + shape), dtype=jnp.float32)
  expected = np.stack([reference(np.asarray(c)) for c in cost])
  np.testing.assert_array_equal(greedy_matcher(cost), expected)
  np.testing.assert_array_equal(jax.jit(greedy_matcher)(cost), expected)


def test_positive_rescaling_preserves_matching_when_order_is_representable():
  cost = jnp.array([[[1.0, 4.0, 2.0], [6.0, 3.0, 8.0], [5.0, 9.0, 7.0]]])
  expected = greedy_matcher(cost)
  np.testing.assert_array_equal(greedy_matcher(cost * 2**27), expected)
  np.testing.assert_array_equal(greedy_matcher(cost + 2**20), expected)
