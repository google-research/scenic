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

"""Module introducing several linear low-rank (LLR) attention mechanisms.

Papers: "Rethinking attention with Performers",
        "Learning a Fourier Transform for Linear Relative Positional
         Encodings in Transformers".
"""

from typing import Callable
import jax
import jax.numpy as jnp


def general_kernel_linearization(
    data: jax.Array,
    projection_matrix: jax.Array | None = None,
    numerical_stabilizer: float = 0.001,
    activation_fn: Callable[
        [jax.Array, jax.Array], jax.Array
    ] = lambda P, X: jax.nn.relu(P),
) -> jax.Array:
  r"""Computes general features of kernel's linearization.

  Args:
    data: input data tensor of the shape [B..., L, H, D], where: B - batch
      dimensions, L - attention dimension, H - heads, D - features.
    projection_matrix: projection matrix of the shape [M, D], where M stands
      for the number of projections.
    numerical_stabilizer: small positive constant used for numerical stability.
    activation_fn: activation function taking projected data and data and
      outputting features of the kernel's linearization.

  Returns:
    Corresponding kernel feature map.
  """
  mc_normalizer = 1.0
  if projection_matrix is not None:
    mc_normalizer = 1.0 / jnp.sqrt(projection_matrix.shape[0])
    data_dash = jnp.einsum(
        '...lhd,md->...lhm', data, projection_matrix
    )
  else:
    data_dash = data
  return mc_normalizer * (activation_fn(data_dash, data) + numerical_stabilizer)


def softmax_positive_rfs(
    data: jax.Array,
    projection_matrix: jax.Array | None = None,
    numerical_stabilizer: float = 0.000001,
    is_query: bool = True,

) -> jax.Array:
  r"""Computes positive random features from https://arxiv.org/abs/2009.14794.

  Args:
    data: input data tensor of the shape [B..., L, H, D], where: B - batch
      dimensions, L - attention dimension, H - heads, D - features.
    projection_matrix: Gaussian projection matrix of the shape [M, D], where M
      stands for the number of projections.
    numerical_stabilizer: small positive constant used for numerical stability.
    is_query: determines whether input data tensor is a query- or key-tensor.

  Returns:
    Corresponding kernel feature map used to linearize softmax kernel.
  """
  h = lambda X: jnp.exp(-0.5 * jnp.sum(jnp.square(X), axis=-1, keepdims=True))
  if is_query:
    axis = (-1,)
  else:
    axis = None
  activation_fn = lambda P, X: h(X) * jnp.exp(
      P - jnp.max(P, axis=axis, keepdims=True)
  )
  return general_kernel_linearization(
      data, projection_matrix, numerical_stabilizer, activation_fn
  )


def softmax_hyper_positive_rfs(
    data: jax.Array,
    projection_matrix: jax.Array | None = None,
    numerical_stabilizer: float = 0.000001,
    is_query: bool = True,

) -> jax.Array:
  r"""Computes hyperbolic extension of positive random features.

  Args:
    data: input data tensor of the shape [B..., L, H, D], where: B - batch
      dimensions, L - attention dimension, H - heads, D - features.
    projection_matrix: Gaussian projection matrix of the shape [M, D], where M
      stands for the number of projections.
    numerical_stabilizer: small positive constant used for numerical stability.
    is_query: determines whether input data tensor is a query- or key-tensor.

  Returns:
    Corresponding kernel feature map used to linearize softmax kernel.
  """
  h = lambda X: jnp.exp(-0.5 * jnp.sum(jnp.square(X), axis=-1, keepdims=True))
  if is_query:
    axis = (-1,)
  else:
    axis = None
  m = lambda P: jnp.maximum(
      jnp.max(P, axis=axis, keepdims=True),
      -jnp.min(P, axis=axis, keepdims=True),
  )
  positive_activation_fn = lambda P, X: h(X) * jnp.exp(P - m(P))
  positive_exponential = jnp.sqrt(0.5) * general_kernel_linearization(
      data, projection_matrix, numerical_stabilizer, positive_activation_fn
  )
  negative_activation_fn = lambda P, X: h(X) * jnp.exp(-P - m(P))
  negative_exponential = jnp.sqrt(0.5) * general_kernel_linearization(
      data, projection_matrix, numerical_stabilizer, negative_activation_fn
  )
  return jnp.concatenate((positive_exponential, negative_exponential), axis=-1)

