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

"""Extract 2-hop subgraph from each wikipedia page."""

import json
import multiprocessing as mp
import os

from absl import app
from absl import logging
from scenic.projects.knowledge_visual_language.data.wikidata import data_util
import tqdm

import tensorflow as tf
from tf.io import gfile


def extract_graph_from_file(kg_graph, i) -> None:
  """Extract 2hop graphs for all pages in each file.

  Args:
    kg_graph: the global KG (i.e. WikiData Knowledge Graph), stored as dict.
    i: index of wiki file.

  Returns:
    A list of wikipedia pages with extracted graph.
  """
  d_list = data_util.load_wiki_from_file(
      os.path.join(data_util.WIKIPEDIA_LINK_PATH, 'wiki_%d.txt' % i))
  with gfile.Open(
      os.path.join(data_util.WIKIPEDIA_GRAPH_PATH, 'wiki_graph_%d.txt' % i),
      'w') as fopen:
    for d in d_list:
      if 'link' in d and isinstance(d['link'], dict):
        in_context_ents = {ent: True for ent in d['link'] if ent != 'UNK'}
        d['graph'] = data_util.extract_2hop_graph(in_context_ents, kg_graph)
      fopen.write(json.dumps(d) + '\n')
  del d_list


def extract_graph_mp(kg_graph, n_pool=1) -> None:
  """Use multiprocessing to parallize graph extraction.

  Args:
    kg_graph: the global KG (i.e. WikiData Knowledge Graph), stored as dict.
    n_pool: number of process (worker)
  """
  jobs = []
  if not gfile.Exists(data_util.WIKIPEDIA_GRAPH_PATH):
    gfile.MakeDirs(data_util.WIKIPEDIA_GRAPH_PATH)
  if n_pool == 1:
    for i in range(data_util.NUM_SPLIT):
      extract_graph_from_file(kg_graph, i)
  else:
    with mp.Pool(n_pool) as pool:
      for i in range(data_util.NUM_SPLIT):
        jobs += [pool.apply_async(extract_graph_from_file, args=(kg_graph, i))]
      for job in tqdm.tqdm(jobs):
        job.get()


def main(_) -> None:
  logging.info('Load WikiData relational edges.')
  with gfile.Open(data_util.WIKIDATA_EDGE_PATH) as f:
    rel_list = json.load(f)
  logging.info('Construct KG Graph from edge list.')
  kg_graph = data_util.construct_kg_graph(rel_list)
  logging.info('Extract 2-hop subgraphs within wiki-pages.')
  extract_graph_mp(kg_graph, n_pool=64)


if __name__ == '__main__':
  app.run(main)
