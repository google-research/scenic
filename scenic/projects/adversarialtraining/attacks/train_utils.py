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

"""Train utils for attack code."""
from typing import Tuple, Optional

import jax
import jax.numpy as jnp


def bind_rng_to_host_device(rng: jnp.ndarray,
                            axis_name: str,
                            bind_to: Optional[str] = None) -> jnp.ndarray:
  """Bind rng to host device."""
  if bind_to is None:
    return rng
  if bind_to == 'host':
    return jax.random.fold_in(rng, jax.process_index())
  elif bind_to == 'device':
    return jax.random.fold_in(rng, jax.lax.axis_index(axis_name))
  else:
    raise ValueError(
        "`bind_to` should be one of the `[None, 'host', 'device']`")


def unnormalize_imgnet(input_tensors):
  return (input_tensors + 1.) / 2.


def normalize_minmax(tensor):
  mn = jnp.min(tensor, axis=(-1, -2, -3), keepdims=True)
  mx = jnp.max(tensor, axis=(-1, -2, -3), keepdims=True)
  return (tensor - mn) / jnp.clip(mx - mn, a_min=1e-5)


def psum_metric_normalizer(metrics: Tuple[jnp.ndarray, jnp.ndarray]
                           ) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Applies psum over the given tuple of (metric, normalizer)."""
  psumed_metric = jnp.sum(jax.lax.psum(metrics[0], axis_name='batch'))
  psumed_normalizer = jnp.sum(
      jax.lax.psum(metrics[1], axis_name='batch'))
  return (psumed_metric, psumed_normalizer)
