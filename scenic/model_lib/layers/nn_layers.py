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

"""Common neural network modules."""

from typing import Callable, Iterable, Optional, Sequence

import flax.linen as nn
import jax
from jax.nn import initializers
import jax.numpy as jnp
import numpy as np

# Inputs are PRNGKey, input shape and dtype.
Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


class Residual(nn.Module):
  """Residual connection module.

  Attributes:
    residual_type: str; residual connection type. Possible values are [
      'gated', 'sigtanh', 'rezero', 'highway', 'add'].
    dtype: Jax dtype; The dtype of the computation (default: float32).
  """

  residual_type: str = 'add'
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x, y):
    """Applies the residual connection on given input/output of a module.

    Args:
      x: Input of the module.
      y: Output of the module.

    Returns:
      Output: A combination of the x and y.
    """
    if x.shape != y.shape:
      raise ValueError('x and y should be of the same shape.')

    dtype = self.dtype

    if self.residual_type == 'add':
      return x + y

    elif self.residual_type == 'highway':
      features = x.shape[-1]
      hw_gate = nn.sigmoid(
          nn.Dense(
              features=features,
              use_bias=True,
              kernel_init=initializers.zeros,
              bias_init=lambda rng, shape, *_: jnp.full(shape, -10.0),
              dtype=dtype)(x))
      output = jnp.multiply((1 - hw_gate), x) + jnp.multiply(hw_gate, y)

    elif self.residual_type == 'rezero':
      # Based on https://arxiv.org/pdf/2003.04887v1.pdf.
      alpha = self.param('rezero_alpha', initializers.zeros, (1,))
      return x + (alpha * y)

    elif self.residual_type == 'sigtanh':
      # Based on https://arxiv.org/pdf/1606.05328.pdf.
      features = x.shape[-1]
      # sigmoid(W_g.y).
      sigmoid_y = nn.sigmoid(
          nn.Dense(
              features=features,
              use_bias=True,
              kernel_init=initializers.zeros,
              bias_init=lambda rng, shape, *_: jnp.full(shape, -10.0),
              dtype=dtype)(y))
      # tanh(U_g.y).
      tanh_y = nn.tanh(
          nn.Dense(
              features=features,
              use_bias=False,
              kernel_init=initializers.zeros,
              bias_init=initializers.zeros,
              dtype=dtype)(y))
      return x + (sigmoid_y * tanh_y)

    elif self.residual_type == 'gated':
      # Based on https://arxiv.org/pdf/1910.06764.pdf.
      features = x.shape[-1]
      # Reset gate: r = sigmoid(W_r.x + U_r.y).
      r = nn.sigmoid(
          nn.Dense(
              features=features,
              use_bias=False,
              kernel_init=initializers.zeros,
              bias_init=initializers.zeros,
              dtype=dtype)(x) + nn.Dense(
                  features=features,
                  use_bias=False,
                  kernel_init=initializers.zeros,
                  bias_init=initializers.zeros,
                  dtype=dtype)(y))
      # Update gate: z = sigmoid(W_z.x + U_z.y - b_g).
      # NOTE: the paper claims best initializtion for their task for b is 2.
      b_g = self.param('b_g',
                       lambda rng, shape, *_: jnp.full(shape, 10.0),
                       (features,)).astype(dtype)
      z = nn.sigmoid(
          nn.Dense(
              features=features,
              use_bias=False,
              kernel_init=initializers.zeros,
              bias_init=initializers.zeros,
              dtype=dtype)(x) + nn.Dense(
                  features=features,
                  use_bias=False,
                  kernel_init=initializers.zeros,
                  bias_init=initializers.zeros,
                  dtype=dtype)(y) - b_g)
      # Candidate_activation: h' = tanh(W_g.y + U_g.(r*x)).
      h = jnp.tanh(
          nn.Dense(
              features=features,
              use_bias=False,
              kernel_init=initializers.zeros,
              bias_init=initializers.zeros,
              dtype=dtype)(y) + nn.Dense(
                  features=features,
                  use_bias=False,
                  kernel_init=initializers.zeros,
                  bias_init=initializers.zeros,
                  dtype=dtype)(jnp.multiply(r, x)))

      # Output: g = (1-z)*x + z*h.
      output = jnp.multiply((1.0 - z), x) + jnp.multiply(z, h)

    else:
      raise ValueError(f'Residual type {self.residual_type} is not defined.')
    return output


class SqueezeAndExcite(nn.Module):
  """Squeeze-and-Excitation layer.

  Introduced in SENet: https://arxiv.org/abs/1709.01507
  """
  reduction_factor: int = 4

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies SqueezeAndExcite on the 2D inputs.

    Args:
      inputs: Input data in shape of `[bs, height, width, features]`.

    Returns:
      Output in which channel-wise features of the input are recalibrated.
    """
    if inputs.ndim != 4:
      # TODO(dehghani): extend this to N-D inputs with arbitrary spatial dims.
      raise ValueError(
          'Inputs should in shape of `[bs, height, width, features]`')

    # Squeeze.
    x = jnp.mean(inputs, axis=(1, 2))
    x = nn.Dense(features=x.shape[-1] // self.reduction_factor)(x)
    x = nn.relu(x)
    # Back to the original feature size.
    x = nn.Dense(features=inputs.shape[-1])(x)
    x = nn.sigmoid(x)
    x = jax.lax.broadcast_in_dim(
        x, shape=(x.shape[0], 1, 1, x.shape[-1]), broadcast_dimensions=(0, 3))
    # Excite.
    return inputs * x


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return x


def get_constant_initializer(constant: float) -> Initializer:
  """Returns an initializer that initializes everything to a given constant."""

  def init_fn(unused_key: jnp.ndarray,  # pytype: disable=annotation-type-mismatch  # jnp-type
              shape: Iterable[int],
              dtype: jnp.dtype = np.float32) -> np.ndarray:
    return constant * np.ones(shape, dtype=dtype)

  return init_fn  # pytype: disable=bad-return-type  # jax-ndarray


class Affine(nn.Module):
  """Affine transformation layer.

  Described in:
  Touvron et al, "ResMLP: Feedforward networks for image classification
  with data-efficient training", 2021.

  Performs an affine transformation on the final dimension of the input tensor.
  """
  bias_init: Initializer = nn.initializers.zeros
  scale_init: Initializer = nn.initializers.ones
  use_bias: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    n = x.shape[-1]
    scale = self.param('scale', self.scale_init, (n,))
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (n,))
    else:
      bias = 0.0
    return scale * x + bias


class StochasticDepth(nn.Module):
  """Performs layer-dropout (also known as stochastic depth).

  Described in
  Huang & Sun et al, "Deep Networks with Stochastic Depth", 2016
  https://arxiv.org/abs/1603.09382

  Attributes:
    rate: the layer dropout probability (_not_ the keep rate!).
    deterministic: If false (e.g. in training) the inputs are scaled by `1 / (1
      - rate)` and the layer dropout is applied, whereas if true (e.g. in
      evaluation), no stochastic depth is applied and the inputs are returned as
      is.
  """
  rate: float = 0.0
  deterministic: Optional[bool] = None

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               deterministic: Optional[bool] = None) -> jnp.ndarray:
    """Applies a stochastic depth mask to the inputs.

    Args:
      x: Input tensor.
      deterministic: If false (e.g. in training) the inputs are scaled by `1 /
        (1 - rate)` and the layer dropout is applied, whereas if true (e.g. in
        evaluation), no stochastic depth is applied and the inputs are returned
        as is.

    Returns:
      The masked inputs reweighted to preserve mean.
    """
    broadcast_dims = range(1, x.ndim)
    return nn.Dropout(
        rate=self.rate, broadcast_dims=broadcast_dims)(x, deterministic)
