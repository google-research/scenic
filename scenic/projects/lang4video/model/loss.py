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

"""Loss functions."""

from typing import Optional

from jax import nn
import jax.numpy as jnp


def _rows_to_columns_nce_loss(
    scores: jnp.ndarray,  # Shape: (N, N)
    where: Optional[jnp.ndarray] = None,  # Shape: broadcastable with `scores`
    initial: Optional[float] = None,
) -> jnp.ndarray:  # Shape: (N,)
  """Computes the InfoNCE loss from rows to columns."""
  return -nn.log_softmax(scores, where=where, initial=initial).diagonal()


def nce_loss(
    scores: jnp.ndarray,  # Shape: (N, N)
    where: Optional[jnp.ndarray] = None,  # Shape: broadcastable with (N,)
    initial: Optional[float] = None,
) -> jnp.ndarray:  # Shape: (N,)
  return (_rows_to_columns_nce_loss(scores, where=where, initial=initial) +
          _rows_to_columns_nce_loss(scores.T, where=where, initial=initial))
