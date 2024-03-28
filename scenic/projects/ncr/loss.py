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

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

Array = Union[jnp.ndarray, np.ndarray]


def pairwise_kl_loss(logits: Array,
                     neighbourhood_logits: Array,
                     knn_indices: Array,
                     knn_similarities: Array,
                     temperature: float,
                     epsilon: float = 1e-16,
                     example_weights: Optional[Array] = None) -> float:
  """KL Divergence loss weighted by similarity within a neighbourhood.

  Args:
    logits: [n, d] array.
    neighbourhood_logits: [m, d] array.
    knn_indices: [n, k] array of nearest neighbours for each logit. Indexes
      into "neighbourhood_logits". Therefore, in each row, each of the k values
      are integers in the interval [0, m).
    knn_similarities: [n, k] array of each of the similarity scores
      corresponding to the knn_indices.
    temperature: A float.
    epsilon: Small constant for numerical stability.
    example_weights: The weight for each example in the batch. Can be used
      to account for TPU padding.

  Returns:
    The KL divergence of each logit to its neighbourhood, weighted by the
      similarity.
  """
  n, d = logits.shape
  k = knn_indices.shape[1]

  knn_logits = neighbourhood_logits[knn_indices.reshape(-1)].reshape([n, k, d])

  t_softmax_temp = jax.nn.softmax(knn_logits / temperature) + epsilon
  s_softmax_temp = jax.nn.log_softmax(logits / temperature)  # [n, d]

  # Normalize the sum of similarities by their sum, so that the new labels
  # will sum up to 1.
  normalized_sim = knn_similarities / jnp.reshape(
      jnp.sum(knn_similarities, axis=-1), (-1, 1))  # [n, k]

  # Multiply the labels by their corresponding similarity value.
  weighted_t_softmax = jnp.squeeze(
      jnp.matmul(jnp.expand_dims(normalized_sim, 1), t_softmax_temp))  # [n, d]

  kldiv_loss_per_pair = weighted_t_softmax * (
      jnp.log(weighted_t_softmax) - s_softmax_temp)  # [n, m]
  kldiv_loss_per_example = (
      jnp.power(temperature, 2) * jnp.sum(kldiv_loss_per_pair, 1))  # [n, 1]

  if example_weights is not None:
    normalization = example_weights.sum()
  else:
    normalization = n
  return jnp.sum(kldiv_loss_per_example) / (normalization + epsilon)  # pytype: disable=bad-return-type  # dataclasses-replace


def l2_normalize(tensor: Array, axis: int = -1, epsilon: float = 1e-6):
  """L2 normalize an input tensor."""

  return tensor / jnp.linalg.norm(tensor, axis=axis, keepdims=True + epsilon)


def get_knn(queries: Array, dataset: Array, k: int,
            zero_negative_similarities: bool = True) -> Tuple[Array, Array]:
  """Return k nearest neighbours from dataset given queries.

  Args:
    queries: An [q, d] array where q is the number of queries, and d the
      dimensionality of each query-vector.
    dataset: An [n, d] array where n is the number of examples.
    k: The number of nearest neighbours to retrieve.
    zero_negative_similarities: If true, negative similarities are set to 0.

  Returns:
    indices: A [q, k] dimensional array. For each query, the k indices of the
      nearest neighbours are returned.
    similarities: A [q, k] dimensional array with the similarities to each
      query. Similarities and corresponding indices are sorted, in descending
      order.
  """
  if k <= 0:
    k = dataset.shape[0]
  if k > dataset.shape[0]:
    k = dataset.shape[0]

  queries = l2_normalize(queries, axis=-1)
  dataset = l2_normalize(dataset, axis=-1)

  all_similarities = jnp.matmul(queries, jnp.transpose(dataset))  # [q, n]
  similarities, indices = jax.lax.top_k(all_similarities, k)
  if zero_negative_similarities:
    similarities = jax.nn.relu(similarities)

  return indices, similarities


def ncr_loss(logits: Array,
             features: Array,
             batch_logits: Array,
             batch_features: Array,
             number_neighbours: int,
             smoothing_gamma: float,
             temperature: float = 1.0,
             example_weights: Optional[Array] = None) -> float:
  """Computes the Neighbourhood Consistency Regularisation loss.

  Details are in: https://arxiv.org/pdf/2202.02200.pdf

  Args:
    logits: An [n_batch, n_classes] array.
    features: An [n_batch, d] array.
    batch_logits: An [m, n_classes] array.
    batch_features: An [m, d] array.
    number_neighbours: The number of neighbours to use. Must be an integer in
      the interval [1, m]
    smoothing_gamma: A value in (0, infinity)
    temperature: Temperature for the KL-Divergence. A value in (0, infinity).
    example_weights: If not None, the weight to apply to the loss of each
      example in the batch. Useful for dealing with TPU padding. If None, the
      mean over the batch is computed.

  Returns:
    The weighted NCR loss computed over the batch.
  """

  indices, similarities = get_knn(features, batch_features,
                                  number_neighbours + 1)
  # Remove the example itself from the list of nearest neighbours.
  indices = indices[:, 1:]
  similarities = similarities[:, 1:]

  similarities = jnp.power(similarities, smoothing_gamma)
  loss = pairwise_kl_loss(logits, batch_logits, indices, similarities,
                          temperature, example_weights=example_weights)
  return loss
