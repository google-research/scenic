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

"""Utils to do with generating random walks and GRFs for topological masking.
"""

import jax.numpy as jnp
import numpy as np


def get_grid_graph_adjacency_matrix(
    nb_x_patches, nb_y_patches, normalised=False
):
  """get (weighted) adj mat for a square grid graph."""
  adj_mat = jnp.zeros(
      (nb_x_patches * nb_y_patches, nb_x_patches * nb_y_patches)
  )
  for node in range(nb_x_patches * nb_y_patches):
    if node % nb_y_patches != nb_y_patches - 1:
      adj_mat = adj_mat.at[node, node + 1].set(1)
    if node < (nb_x_patches - 1) * nb_y_patches:
      adj_mat = adj_mat.at[node, node + nb_y_patches].set(1)
  adj_mat += adj_mat.T
  if normalised:
    degrees = jnp.sum(adj_mat, axis=1)
    deg = jnp.diag(1 / jnp.sqrt(degrees))
    w_adj_mat = deg @ adj_mat @ deg
    return w_adj_mat
  else:
    return adj_mat


def get_numpy_grid_graph_adjacency_matrix(
    nb_x_patches, nb_y_patches, normalised=False
):
  """Numpy version."""
  adj_matrix = np.zeros(
      (nb_x_patches * nb_y_patches, nb_x_patches * nb_y_patches)
  )
  for node in range(nb_x_patches * nb_y_patches):
    if node % nb_y_patches != nb_y_patches - 1:
      adj_matrix[node, node + 1] = 1
    if node < (nb_x_patches - 1) * nb_y_patches:
      adj_matrix[node, node + nb_y_patches] = 1
  adj_matrix += adj_matrix.T
  if normalised:
    degrees = np.sum(adj_matrix, axis=1)
    deg = np.diag(1 / np.sqrt(degrees))
    w_adj_mat = deg @ adj_matrix @ deg
    return w_adj_mat
  else:
    return adj_matrix


def get_grid_graph_adjacency_lists(nb_x_patches, nb_y_patches):
  """Function to compute graph adj lists."""
  adj_lists = []
  for _ in range(nb_x_patches * nb_y_patches):
    adj_lists.append([])
  for node in range(nb_x_patches * nb_y_patches):
    if node % nb_y_patches != nb_y_patches - 1:
      adj_lists[node].append(node + 1)
      adj_lists[node + 1].append(node)
    if node < (nb_x_patches - 1) * nb_y_patches:
      adj_lists[node].append(node + nb_y_patches)
      adj_lists[node + nb_y_patches].append(node)
  return adj_lists


def generate_walk(p, adj_lists, base_vertex):
  """get a graph walk given the length."""
  length = int(np.floor(np.log(1 - np.random.random()) / np.log(1 - p)))
  walk = [base_vertex]
  vertex = np.copy(base_vertex)
  for _ in range(length):
    neighbours = adj_lists[vertex]
    new_vertex = neighbours[int(np.random.uniform(0, 1) * len(neighbours))]
    vertex = new_vertex
    walk.append(vertex)
  return walk


def walk_to_loads_list(walk, adj_matrix, p_halt, degrees):
  """Pre-compute a bunch of loads and store as a list."""
  loads = []
  load = 1.0
  for step, _ in enumerate(walk):
    loads.append(load)
    if step < (len(walk) - 1):
      load *= (
          np.copy(adj_matrix)[walk[step]][walk[step + 1]]
          * np.copy(degrees)[walk[step]]
          / (1-p_halt)
      )
  return loads


def m_initialiser(
    key,
    nb_heads,
    nb_x_patches,
    nb_y_patches,
    walks_per_vertex,
    max_steps,
    p_halt,
):
  """Function to specify M initialisation."""
  del key
  adj_matrix = get_numpy_grid_graph_adjacency_matrix(
      nb_x_patches, nb_y_patches, normalised=True
  )
  adj_lists = get_grid_graph_adjacency_lists(nb_x_patches, nb_y_patches)
  degrees = np.asarray([len(adj_lists[i]) for i in range(len(adj_lists))])
  nb_vertices = nb_x_patches * nb_y_patches
  m_matrix = np.zeros(
      (nb_heads, nb_vertices, nb_vertices, max_steps)
  )  #  sparse matrix for computing the GRFs

  for head in range(nb_heads):
    all_walks = []  #  list of walks
    for base_vertex in range(nb_vertices):
      this_vertex_walks = []
      for _ in range(walks_per_vertex):
        this_vertex_walks.append(
            tuple(generate_walk(p_halt, adj_lists, base_vertex))
        )
      all_walks.append(tuple(this_vertex_walks))
    all_walks = tuple(all_walks)

    all_loads = []  #  list of loads
    for base_vertex in range(nb_vertices):
      this_vertex_loads = []
      for walk in all_walks[base_vertex]:
        this_vertex_loads.append(
            tuple(walk_to_loads_list(walk, adj_matrix, p_halt, degrees))
        )
      all_loads.append(tuple(this_vertex_loads))
    all_loads = tuple(all_loads)

    for start_index, walks in enumerate(all_walks):
      for walk_index, walk in enumerate(walks):
        if len(walk) > max_steps:  # catch case that walks are too long
          walk = walk[:max_steps]
        for length, end_index in enumerate(walk):
          m_matrix[head, start_index, end_index, length] += all_loads[
              start_index
          ][walk_index][length] / len(all_loads[start_index])

  return jnp.asarray(m_matrix, dtype=jnp.float32)


def get_qmc_walks(
    adj_lists, p_halt, nb_random_walks, antithetic=False, repelling=False
):
  """Function to get antithetic or repelling walks on a graph.

  Args:
    adj_lists: list of lists of neighbours
    p_halt: probability of halting
    nb_random_walks: number of random walks to generate
    antithetic: whether to use antithetic termination
    repelling: whether to use repelling termination

  Returns:
    all_walks: list of lists of walks
  """
  all_walks = []
  for base_vertex in range(len(adj_lists)):
    nb_vertices = len(adj_lists)
    current_vertices = np.asarray(
        np.ones(nb_random_walks) * base_vertex, dtype=int
    )
    vertex_populations = np.zeros(nb_vertices)
    vertex_populations[base_vertex] = nb_random_walks

    term_indicators = np.zeros(nb_random_walks)
    rand_draws = np.zeros(nb_random_walks)

    walks = [[base_vertex] for _ in range(nb_random_walks)]

    while np.sum(term_indicators) < nb_random_walks:  # do the termination bit
      if antithetic:
        if nb_random_walks % 2 != 0:
          raise ValueError('Use an even number of walkers for AT')
        rand_draws[: int(nb_random_walks / 2)] = np.random.uniform(
            0, 1, int(nb_random_walks / 2)
        )
        rand_draws[int(nb_random_walks / 2) :] = np.mod(
            rand_draws[: int(nb_random_walks / 2)] + 0.5, 1
        )
      else:
        rand_draws = np.random.uniform(0, 1, nb_random_walks)

      for i in range(nb_random_walks):
        if term_indicators[i] == 0:
          term_indicators[i] = rand_draws[i] < p_halt
          vertex_populations[current_vertices[i]] -= term_indicators[i]

      if not repelling:
        remaining_walkers = np.where(term_indicators == 0)[0]

        for walker in remaining_walkers:
          current_vertex = current_vertices[walker]
          neighbours = adj_lists[current_vertex]
          rnd_index = int(np.random.uniform(0, 1) * len(neighbours))
          newnode = neighbours[rnd_index]
          current_vertices[walker] = newnode
          walks[walker].append(newnode)

      else:
        occupied_vertices = np.where(vertex_populations > 0)[0]
        current_vertices_copy = np.copy(
            current_vertices
        )  # because should be simultaneous

        for vertex in occupied_vertices:
          walkers = np.where(
              np.logical_and(term_indicators == 0, current_vertices == vertex)
          )[0]
          num_walkers = len(walkers)
          vertex_populations[vertex] -= num_walkers
          neighbours = np.asarray(adj_lists[vertex])
          num_neighbours = len(neighbours)

          if num_walkers > num_neighbours:
            vertex_offsets = np.linspace(0, num_walkers - 1, num_walkers)
            blocks = num_walkers // num_neighbours
            remainder = num_walkers % num_neighbours
            for block in range(blocks):
              vertex_offsets[
                  block * num_neighbours : (block + 1) * num_neighbours
              ] = np.mod(
                  vertex_offsets[:num_neighbours]
                  + np.random.randint(num_neighbours),
                  num_neighbours,
              )
            vertex_offsets[num_walkers - remainder :] = np.mod(
                vertex_offsets[num_walkers - remainder :]
                + np.random.randint(num_neighbours),
                num_neighbours,
            )
            vertex_offsets = np.asarray(vertex_offsets, dtype=int)
          else:
            vertex_offsets = np.asarray(
                np.mod(
                    np.linspace(0, num_walkers - 1, num_walkers) / num_walkers
                    + np.random.uniform(0, 1),
                    1,
                )
                * len(neighbours),
                dtype=int,
            )

          new_vertices = neighbours[vertex_offsets]
          current_vertices_copy[walkers] = new_vertices

          for walker_ind, walker in enumerate(walkers):
            walks[walker].append(new_vertices[walker_ind])

          for vertex in new_vertices:
            vertex_populations[vertex] += 1

        current_vertices = np.copy(
            current_vertices_copy
        )  # now want to update all positions

    all_walks.append(walks)
  all_walks = tuple([
      tuple([tuple(walk) for walk in all_walks[i]])
      for i in range(len(all_walks))
  ])
  return all_walks


def qmc_m_initialiser(
    key,
    nb_heads,
    nb_x_patches,
    nb_y_patches,
    walks_per_vertex,
    max_steps,
    p_halt,
    antithetic=False,
    repelling=False,
):
  """Function to specify M initialisation w QMC RWs.
  """
  del key
  adj_matrix = get_numpy_grid_graph_adjacency_matrix(
      nb_x_patches, nb_y_patches, normalised=True
  )
  adj_lists = get_grid_graph_adjacency_lists(nb_x_patches, nb_y_patches)
  degrees = np.asarray([len(adj_lists[i]) for i in range(len(adj_lists))])
  nb_vertices = nb_x_patches * nb_y_patches
  m_matrix = np.zeros((nb_heads, nb_vertices, nb_vertices, max_steps))

  for head in range(nb_heads):
    all_walks = get_qmc_walks(
        adj_lists, p_halt, walks_per_vertex, antithetic, repelling
    )

    all_loads = []  #  list of loads
    for base_vertex in range(nb_vertices):
      this_vertex_loads = []
      for walk in all_walks[base_vertex]:
        this_vertex_loads.append(
            tuple(walk_to_loads_list(walk, adj_matrix, p_halt, degrees))
        )
      all_loads.append(tuple(this_vertex_loads))
    all_loads = tuple(all_loads)

    for start_index, walks in enumerate(all_walks):
      for walk_index, walk in enumerate(walks):
        if len(walk) > max_steps:  # catch case that walks are too long
          walk = walk[:max_steps]
        for length, end_index in enumerate(walk):
          m_matrix[head, start_index, end_index, length] += all_loads[
              start_index
          ][walk_index][length] / len(all_loads[start_index])

  return jnp.asarray(m_matrix, dtype=jnp.float32)
