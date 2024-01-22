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

"""Lazy matcher."""

import jax
import jax.numpy as jnp


def lazy_matcher(cost: jnp.ndarray) -> jnp.ndarray:
  """Computes lazy Matching on cost matrix for a batch of datapoints.

  This matcher ignores input and matches i-th to i-th input.

  Args:
    cost: Cost matrix for the matching of shape [B, N, M].

  Returns:
    An assignment of size [B, 2, min(N,M)].
  """
  batch_size, n, m = cost.shape
  return jax.lax.broadcast(jnp.arange(0, min(n, m)), (batch_size, 2))
