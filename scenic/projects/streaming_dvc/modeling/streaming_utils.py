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

"""Utils for streaming model."""

import jax
import jax.numpy as jnp


def paired_cosine_distance(u, v):
  """Compute pairwise cosine distance.

  Args:
    u: (N, D)
    v: (N, D)
  Returns:
    cosine_distance: (N,)
  """
  u_norm = jnp.linalg.norm(u, axis=-1)  # n
  v_norm = jnp.linalg.norm(v, axis=-1)  # n
  dist = 1. - (u * v).sum(axis=-1) / (u_norm * v_norm + 1e-8)
  return dist


def adjacent_merge(buffer, new_tokens, weights=None):
  """Perform token merging.

  This is the original implementation in MovieChat only merges adjacent frames.
  Reference: https://github.com/rese1f/MovieChat/blob/main/MovieChat/models/
  moviechat.py#L279

  Args:
    buffer: (N, D); The memory tokens.
    new_tokens: (M, D); The incoming tokens. The tokens will be added to memory
      one-by-one in order.
    weights: (N, D) or None; if provided, compute the weighted average between
      existing tokens and the new incoming token, so that exiting tokens which
      are merged from many tokens have more weights.
  Returns:
    buffer: (N, D); The updated memory tokens.
  """
  num_buffers = buffer.shape[0]
  num_new_tokens = new_tokens.shape[0]
  def body_fn(state):
    current_buffer, current_weights, t = state
    new_token = new_tokens[t]
    all_tokens = jnp.concatenate((current_buffer, new_token[None]), axis=0)
    dist = paired_cosine_distance(all_tokens[:-1], all_tokens[1:])  # (N,)
    i = jnp.argmin(dist)  # i < num_buffers
    if current_weights is not None:
      all_weights = jnp.concatenate(
          (current_weights, jnp.ones((1,), dtype=jnp.int32)), axis=0)
      merged_token = (
          all_tokens[i] * all_weights[i] + all_tokens[i + 1] * all_weights[
              i + 1]) / (all_weights[i] + all_weights[i + 1])
      all_tokens = all_tokens.at[i].set(merged_token)
      all_tokens = all_tokens.at[i + 1].set(new_token)
      all_weights = all_weights.at[i].set(all_weights[i] + all_weights[i + 1])
      all_weights = all_weights.at[i + 1].set(1)
      return (all_tokens[:num_buffers], all_weights[:num_buffers], t + 1)
    else:
      merged_token = (all_tokens[i] + all_tokens[i + 1]) / 2
      all_tokens = all_tokens.at[i].set(merged_token)
      all_tokens = all_tokens.at[i + 1].set(new_token)
      return (all_tokens[:num_buffers], None, t + 1)
  state = jax.lax.while_loop(
      lambda s: s[2] < num_new_tokens,
      body_fn,
      (buffer, weights, 0),
  )
  return jax.lax.stop_gradient(state[0]), jax.lax.stop_gradient(state[1])


def kmeans(init_centers, data, weights, num_iters=1):
  """Run kmeans on weighted data.


  Args:
    init_centers: array in shape (k, d);
    data: array in shape (n, d); All data points.
    weights: array in shape (n,); Weights of the data points.
    num_iters: int;
  Returns:
    new_centers: (k, d)
    counts: (k,), num_data assigned to each center
  """
  k = init_centers.shape[0]
  def step_fn(_, centers_counts):
    centers, _ = centers_counts
    # TODO(zhouxy): We might want to try other distance functions.
    distances = jnp.linalg.norm(
        data[:, None] - centers[None, :], axis=2)  # (n, k)
    assignments = jnp.argmin(distances, axis=1)  # (n,)
    weighted_data = data * weights[:, None]  # (n, d)
    one_hot_assignments = jax.nn.one_hot(assignments, k)  # (n, k)
    # NOTE: The following stop_gradient is optional, since both one_hot and
    # argmin are not differentiable anyway.
    # one_hot_assignments = jax.lax.stop_gradient(one_hot_assignments)
    weighted_sums = jnp.dot(
        one_hot_assignments.T, weighted_data)  # (k, d)
    counts = jnp.dot(one_hot_assignments.astype(jnp.int32).T, weights)  # (k,)
    # If the cluster is empty (which can happen from the second iteration),
    # we just retain the original cluster center.
    new_centers = weighted_sums / jnp.maximum(counts[:, None], 1)
    new_centers = jnp.where(
        jnp.broadcast_to(counts[:, None], new_centers.shape) == 0,
        centers, new_centers)

    return new_centers, counts
  new_centers, counts = jax.lax.fori_loop(
      0, num_iters, step_fn,
      (init_centers, jnp.zeros((k,), dtype=jnp.int32)))
  # new_centers = jax.lax.stop_gradient(new_centers)
  # counts = jax.lax.stop_gradient(counts)
  return new_centers, counts
