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

"""Library constructing random projection matrices for softmax-kernel RFs."""

import jax
from jax import random
import jax.numpy as jnp
Array = jax.Array
PRNGKey = jax.Array


def get_gaussian_orth_rand_mat(
    rng: PRNGKey, nb_rows: int, nb_columns: int, scaling: bool = False
) -> Array:
  """Method for constructing structured block-orthogonal Gaussian matrices.

  Args:
    rng: the key used to generate randomness for the construction of the random
      matrices,
    nb_rows: number of rows of the Gaussian matrix to be constructed,
    nb_columns: number of columns of the Gaussian matrix to be constructed,
    scaling: boolean indicating whether the rows of the Gaussian matrix should
      be normalized to the deterministic length sqrt(nb_rows)
  Returns:
    The Gaussian matrix of <nb_rows> rows and <nb_columns> columns.
  """
  nb_full_blocks = int(nb_rows / nb_columns)
  block_list = []
  for _ in range(nb_full_blocks):
    rng, rng_input = jax.random.split(rng)
    unstructured_block = random.normal(rng_input, (nb_columns, nb_columns))
    q, _ = jnp.linalg.qr(unstructured_block)
    q = jnp.transpose(q)
    block_list.append(q)
  remaining_rows = nb_rows - nb_full_blocks * nb_columns
  if remaining_rows > 0:
    rng, rng_input = jax.random.split(rng)
    unstructured_block = random.normal(rng_input, (nb_columns, nb_columns))
    q, _ = jnp.linalg.qr(unstructured_block)
    q = jnp.transpose(q)
    block_list.append(q[:remaining_rows])
  final_matrix = jnp.vstack(block_list)
  if scaling:
    multiplier = jnp.sqrt(float(nb_columns)) * jnp.ones((nb_rows))
  else:
    _, rng_input = jax.random.split(rng)
    multiplier = jnp.linalg.norm(
        random.normal(rng_input, (nb_rows, nb_columns)), axis=1
    )
  return jnp.matmul(jnp.diag(multiplier), final_matrix)


def get_gaussian_simplex_rand_mat(
    rng: PRNGKey,
    nb_rows: int,
    nb_columns: int,
    scaling: bool = False,
) -> Array:
  """Method for constructing 2D Gaussian simplex arrays.

  Method for constructing Gaussian matrix that is block-wise simplex, i.e.
  it consists of square-blocks, where the rows within each block form a simplex.
  Args:
    rng: the key used to generate randomness for the construction of the random
      matrices,
    nb_rows: number of rows of the Gaussian matrix to be constructed,
    nb_columns: number of columns of the Gaussian matrix to be constructed,
    scaling: boolean indicating whether the rows of the Gaussian matrix should
      be normalized to the deterministic length sqrt(nb_rows)
  Returns:
    The Gaussian matrix of <nb_rows> rows and <nb_columns> columns.
  """
  sim_vectors = []
  all_ones_but_last = (
      jnp.ones(nb_columns) - jnp.identity(nb_columns)[nb_columns - 1]
  )
  first_mult = (jnp.sqrt(nb_columns) + 1.0) / jnp.power(nb_columns - 1, 1.5)
  second_mult = 1.0 / jnp.sqrt(nb_columns - 1)
  for i in range(nb_columns - 1):
    sim_vector = (
        jnp.sqrt(nb_columns / (nb_columns - 1)) * jnp.identity(nb_columns)[i]
        - first_mult * all_ones_but_last
    )
    sim_vectors.append(sim_vector)
  sim_vectors.append(second_mult * all_ones_but_last)
  sim_matrix = jnp.transpose(jnp.array(sim_vectors))
  nb_full_blocks = int(nb_rows / nb_columns)
  block_list = []
  for _ in range(nb_full_blocks):
    rng, rng_input = jax.random.split(rng)
    unstructured_block = random.normal(rng_input, (nb_columns, nb_columns))
    q, _ = jnp.linalg.qr(unstructured_block)
    q = jnp.transpose(q)
    block_list.append(jnp.transpose(jnp.matmul(q, sim_matrix)))
  remaining_rows = nb_rows - nb_full_blocks * nb_columns
  if remaining_rows > 0:
    rng, rng_input = jax.random.split(rng)
    unstructured_block = random.normal(rng_input, (nb_columns, nb_columns))
    q, _ = jnp.linalg.qr(unstructured_block)
    q = jnp.transpose(q)
    block_list.append(
        jnp.transpose(jnp.matmul(q, sim_matrix[:, :remaining_rows]))
    )
  final_matrix = jnp.vstack(block_list)
  if scaling:
    multiplier = jnp.sqrt(float(nb_columns)) * jnp.ones((nb_rows))
  else:
    _, rng_input = jax.random.split(rng)
    multiplier = jnp.linalg.norm(
        random.normal(rng_input, (nb_rows, nb_columns)), axis=1
    )
  return jnp.matmul(jnp.diag(multiplier), final_matrix)


