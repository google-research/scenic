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

"""Functions for processing the data."""

import json
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tqdm

import tensorflow as tf
from tf.io import gfile

MAX_REL_PER_ENTITY = 10000
# Add path to your data here:
WIKIPEDIA_LINK_PATH = ''
WIKIPEDIA_GRAPH_PATH = ''
WIKIDATA_EDGE_PATH = ''
WIKIDATA_ENTITY_PATH = ''

NUM_SPLIT = 1024


def construct_kg_graph(rel_list) -> Dict[str, Dict[str, List[str]]]:
  """construct a kg graph with reverse link.

  Args:
    rel_list: KG edge list, in the format of <s, r, t> triplets.

  Returns:

  """
  kg_graph = {}
  kg_rel = {}
  for se, r, te in tqdm.tqdm(rel_list):
    if se == te:
      continue
    if se not in kg_graph:
      kg_graph[se] = {}
    if te not in kg_graph[se]:
      kg_graph[se][te] = []
    if r not in kg_graph[se][te]:
      kg_graph[se][te].append(r)

    if r not in kg_rel:
      kg_rel[r] = {}
    if te not in kg_rel[r]:
      kg_rel[r][te] = 0
    kg_rel[r][te] += 1

  for se, r, te in tqdm.tqdm(rel_list):
    if se == te or kg_rel[r][te] >= MAX_REL_PER_ENTITY:
      continue
    if te not in kg_graph:
      kg_graph[te] = {}
    if se not in kg_graph[te]:
      kg_graph[te][se] = []
    if r + '_R' not in kg_graph[te][se]:
      kg_graph[te][se].append(r + '_R')

  return kg_graph


def load_wiki_from_file(file_path) -> List[Dict[str, None]]:
  data_list = []
  with gfile.Open(file_path, 'r') as fopen:
    lines = fopen.readlines()
  for line in lines:
    data = json.loads(line)
    data_list += [data]
  del lines
  return data_list


def extract_2hop_graph(in_context_ents, kg_graph) -> Dict[Any, Dict[Any, bool]]:
  """For each wikipedia page with N in-context entities, extract a subgraph that contains only 2hop paths between any pair of nodes.

  Args:
    in_context_ents: all entities within each wiki-page, stored as dict.
    kg_graph: the global KG (i.e. WikiData Knowledge Graph), stored as dict.

  Returns:
    Extracted 2-hop subgraph for each page.
  """
  all_nodes = {se: [se] for se in in_context_ents}
  for se in in_context_ents:
    if se in kg_graph:
      for te in kg_graph[se]:
        if te not in in_context_ents:
          if te not in all_nodes:
            all_nodes[te] = [se]
          else:
            all_nodes[te] += [se]
  remain_nodes = {e: True for e in all_nodes if len(all_nodes[e]) > 1}
  for e in in_context_ents:
    remain_nodes[e] = True

  two_graph = {}
  for se in remain_nodes:
    if se in kg_graph:
      for te in kg_graph[se]:
        if te in remain_nodes:
          if se not in two_graph:
            two_graph[se] = {}
          two_graph[se][te] = kg_graph[se][te]
          if te in kg_graph and se in kg_graph[te]:
            if te not in two_graph:
              two_graph[te] = {}
            two_graph[te][se] = kg_graph[te][se]
  return two_graph


def plot_graph(in_graph_ents, graph, entity_dict, print_out_label=True) -> None:
  """Function to plot each wikipedia's subgraph.

  Args:
    in_graph_ents: all in-context entities
    graph: subgraph of each wikipage.
    entity_dict: entityID to name
    print_out_label: whether to print the intermediate label.
  """
  if not graph:
    return
  g = nx.Graph()
  all_label = {}
  in_label = {}
  for se in graph:
    all_label[entity_dict[se]] = entity_dict[se]
    if se in in_graph_ents:
      g.add_node(entity_dict[se], color='red', size=2000)
      in_label[entity_dict[se]] = entity_dict[se]
    if se not in in_graph_ents:
      g.add_node(entity_dict[se], color='blue', size=100)
    for te in graph[se]:
      all_label[entity_dict[te]] = entity_dict[te]
      if te in in_graph_ents:
        g.add_node(entity_dict[te], color='red', size=2000)
        in_label[entity_dict[te]] = entity_dict[te]
      if te not in in_graph_ents:
        g.add_node(entity_dict[te], color='blue', size=100)
      g.add_edge(entity_dict[se], entity_dict[te])
  plt.figure(figsize=(10, 10))
  layout = nx.kamada_kawai_layout(g)
  if print_out_label:
    nx.draw_networkx_nodes(
        g,
        pos=layout,
        node_color=nx.get_node_attributes(g, 'color').values(),
        node_size=list(nx.get_node_attributes(g, 'size').values()))
    nx.draw_networkx_labels(g, pos=layout, labels=all_label)
    nx.draw_networkx_edges(g, pos=layout, alpha=0.3, arrows=False)
  else:
    nx.draw_networkx_nodes(
        g,
        pos=layout,
        node_color=nx.get_node_attributes(g, 'color').values(),
        node_size=list(nx.get_node_attributes(g, 'size').values()))
    nx.draw_networkx_labels(g, pos=layout, labels=in_label)
    nx.draw_networkx_edges(g, pos=layout, alpha=0.3, arrows=False)
  xs = np.array(list(layout.values()))[:, 0]
  xmin, xmax = np.min(xs), np.max(xs)
  plt.xlim(xmin - (xmax - xmin) * 0.2, xmax + (xmax - xmin) * 0.2)
  plt.show()
