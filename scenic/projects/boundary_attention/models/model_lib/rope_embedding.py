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

"""Rotary embedding."""

import functools
import einops
import jax
import jax.numpy as jnp


class RotaryEmbedding2D:
  """Calculates and applies the rotary embedding."""

  def __init__(self, dim: int):
    self.dim = dim
    assert self.dim // 2 % 2 == 0

  @functools.partial(jax.jit, static_argnums=(0,))
  def calc_and_apply(self, x, x_coords):
    """Calculates and applies the rotary embedding."""

    sinusoid_inp = self.get_pos(coords=x_coords)
    return self.apply_2d_rotary_pos_emb(x, sinusoid_inp)

  # @functools.partial(jax.jit, static_argnums=(0,))
  def get_pos(self, coords):
    """Calculates the position of the rotary embedding."""

    # Half of each feature will get x, the other half will get y
    inv_freq = 1.0 / (10000 ** (jnp.arange(0,
                                           self.dim // 2, 2) / (self.dim // 2)))

    # Take inner product with inverse frequencies
    sinusoid_inp = jnp.einsum("b x y n i , j -> b x y n i j", coords, inv_freq)

    # Reshape so that x and y are now stacked in a single dimension
    sinusoid_inp = jnp.reshape(sinusoid_inp, (*coords.shape[:-1], -1))

    return sinusoid_inp

  def rotate_every_two(self, x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    x = jnp.stack((-x2, x1), axis=-1)

    return einops.rearrange(x, "... d j -> ... (d j)")

  @functools.partial(jax.jit, static_argnums=(0,))
  def apply_2d_rotary_pos_emb(self, x, sinusoid_inp):
    """Applies the rotary embedding to the input."""

    sin = jnp.repeat(jnp.sin(sinusoid_inp)[..., None, :], 2, axis=-1)
    cos = jnp.repeat(jnp.cos(sinusoid_inp)[..., None, :], 2, axis=-1)

    return (x * cos) + (self.rotate_every_two(x) * sin)
