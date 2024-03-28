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

"""Global file to store local knowledge base."""

import functools

from absl import logging
from flax.core.frozen_dict import FrozenDict
import jax
import numpy as np
import tqdm


def static_encode_knowledge(knowledge_batch, train_state, *, flax_model):
  """Function to encode KB knowledge.

  Args:
    knowledge_batch: A single batch of data. The buffer of this argument can be
      donated to the computation.
    train_state: The state of training including the current global_step,
      model_state, rng, and optimizer. The buffer of this argument can be
      donated to the computation.
    flax_model: A Flax model.

  Returns:
    Key (single embedding), Val (compressed list of embedding) and Mask.
  """
  variables = {'params': train_state.params, **train_state.model_state}
  retr_tokens, retr_images = knowledge_batch[
      'knowledge_tokens'], knowledge_batch['image']
  batch_size = retr_images.shape[0]
  keys_head, compressed_val, compressed_mask, _, _ = flax_model.apply(
      variables,
      retr_texts=retr_tokens,
      retr_images=retr_images,
      bsz=batch_size,
      train=False,
      random_drop_image=False,
      method=flax_model.encode_knowledge)
  return keys_head, compressed_val, compressed_mask


class KnowledgeBase:
  """Local Knowledge Base stored in CPU."""

  def __init__(self):
    self.memory = {}
    self.memory_flatten = {}
    self.specs = {}
    self.n_data_per_shard = -1
    self.n_data = -1
    self.ret_specs = None
    self.n_kb_dataset = 1
    self.n_local_device = 1
    self.k = 1

  def set_encode_fn(self, flax_model):
    encode_fn = functools.partial(
        static_encode_knowledge, flax_model=flax_model)
    self.encode_knowledge_pmap = jax.pmap(
        encode_fn,
        axis_name='batch',
        donate_argnums=(0, 1),
    )

  def initialize(self, kb_datasets):
    """Load sharded dataset into CPU."""
    memory_image = []
    memory_text = []
    memory_idxs = []
    self.n_kb_dataset = len(kb_datasets)
    logging.info('Start Loading sharded dataset into CPU.')
    for idx, dataset_name in enumerate(kb_datasets):
      logging.info(dataset_name)
      dataset = kb_datasets[dataset_name]
      n_iter = int(dataset.meta_data['example_per_shard'] //
                   dataset.meta_data['batch_size'])
      for _ in tqdm.tqdm(range(n_iter)):
        kb_batch = next(dataset.train_iter)
        memory_image += [np.asarray(kb_batch['image'])]
        memory_text += [np.asarray(kb_batch['knowledge_tokens'])]
        memory_idxs += [
            idx * np.ones(shape=kb_batch['image'].shape[:2]).astype('int16')
        ]
        del kb_batch
    self.memory['image'] = np.concatenate(memory_image, axis=1)
    self.memory['text'] = np.concatenate(memory_text, axis=1)
    self.n_local_device = self.memory['image'].shape[0]
    self.memory_flatten['idxs'] = np.repeat(
        np.reshape(np.concatenate(memory_idxs, axis=1), (1, -1)),
        self.n_local_device,
        axis=0)
    self.n_data_per_shard = self.memory['image'].shape[1]
    self.n_data = self.n_data_per_shard * self.n_local_device

    self.specs = {
        'image': dataset.meta_data['image_spec'],
        'text': dataset.meta_data['knowledge_spec']
    }

  def update_memory(self, pmap_train_state, bsz, retr_k, data_k,
                    axis_index_groups):
    """Function to update stale embedding as memory.

    Args:
      pmap_train_state: pmaped train state.
      bsz: Global batch size.
      retr_k: number of returned data for retrieval.
      data_k: number of returned data for ranking.
      axis_index_groups: axis groups to gather data.

    Returns:
      updated train_state
    """
    per_bsz = bsz // jax.device_count()
    if axis_index_groups is None:
      per_shard_bsz = bsz
    else:
      per_shard_bsz = bsz // len(axis_index_groups[0])
    logging.info('update memory!!!')
    logging.info(per_bsz)
    memory_key = []
    memory_val = []
    # memory_mask = []
    eval_per_bsz = per_bsz * 4
    for idx in range(int(np.ceil(self.n_data_per_shard / eval_per_bsz))):
      kb_batch = {
          'knowledge_tokens':
              self.memory['text'][:,
                                  idx * eval_per_bsz:(idx + 1) * eval_per_bsz],
          'image':
              self.memory['image'][:,
                                   idx * eval_per_bsz:(idx + 1) * eval_per_bsz],
      }
      keys_head, compressed_val, _ = self.encode_knowledge_pmap(
          kb_batch, pmap_train_state)
      memory_key += [np.asarray(keys_head)]
      memory_val += [np.asarray(compressed_val)]
      # memory_mask += [np.asarray(mask)]
      del kb_batch

    for kw in ['keys', 'values']:
      if kw in self.memory:
        del self.memory[kw]
      if kw in self.memory_flatten:
        del self.memory_flatten[kw]

    self.memory['keys'] = np.concatenate(memory_key, axis=1)
    self.memory['values'] = np.concatenate(memory_val, axis=1)
    # self.memory['masks'] = np.concatenate(memory_mask, axis=1)

    for kw in ['keys', 'values', 'image', 'text']:
      mem = self.memory[kw]
      self.memory_flatten[kw] = mem.reshape((mem.shape[0] * mem.shape[1],) +
                                            mem.shape[2:])

    self.specs['keys'] = (keys_head.shape[2:], keys_head.dtype.name)
    self.specs['values'] = (compressed_val.shape[2:], compressed_val.dtype.name)
    # self.specs['masks'] = (mask.shape[2:], mask.dtype.name)

    self.local_ret_specs = [{
        'keys':
            jax.ShapeDtypeStruct(
                shape=(per_bsz, retr_k - data_k) + self.specs['keys'][0],
                dtype=self.specs['values'][1]),
        'values':
            jax.ShapeDtypeStruct(
                shape=(per_bsz, retr_k - data_k) + self.specs['values'][0],
                dtype=self.specs['values'][1])
    }, {
        'image':
            jax.ShapeDtypeStruct(
                shape=(per_bsz, data_k) + self.specs['image'][0],
                dtype=self.specs['image'][1]),
        'text_tokens':
            jax.ShapeDtypeStruct(
                shape=(per_bsz, data_k) + self.specs['text'][0],
                dtype=self.specs['text'][1])
    }]
    self.k = int(np.ceil(retr_k / len(axis_index_groups)) + 1)
    self.ret_top_specs = [{
        'keys':
            jax.ShapeDtypeStruct(
                shape=(per_shard_bsz, self.k) + self.specs['keys'][0],
                dtype=self.specs['values'][1]),
        'values':
            jax.ShapeDtypeStruct(
                shape=(per_shard_bsz, self.k) + self.specs['values'][0],
                dtype=self.specs['values'][1])
    }, {
        'image':
            jax.ShapeDtypeStruct(
                shape=(per_shard_bsz, self.k) + self.specs['image'][0],
                dtype=self.specs['image'][1]),
        'text_tokens':
            jax.ShapeDtypeStruct(
                shape=(per_shard_bsz, self.k) + self.specs['text'][0],
                dtype=self.specs['text'][1])
    }]

    logging.info(self.local_ret_specs)
    logging.info(self.ret_specs)
    logging.info(self.memory['keys'].shape)
    new_model_state = pmap_train_state.model_state.unfreeze()
    if 'keys' in new_model_state['memory']:
      del new_model_state['memory']['keys']
      del new_model_state['memory']['idxs']
    new_model_state['memory']['keys'] = self.memory['keys']
    new_model_state['memory']['idxs'] = self.memory_flatten['idxs']
    pmap_train_state = pmap_train_state.replace(
        model_state=FrozenDict(new_model_state))
    logging.info('finish update memory!!!')
    return pmap_train_state


def retrieve_memory(args):
  device_id, indexs = args
  return [
      {
          'values': kb.memory['values'][device_id][indexs],
          # 'masks': kb.memory['masks'][device_id][indexs]
      },
      {
          'image': kb.memory['image'][device_id][indexs],
          'text_tokens': kb.memory['text'][device_id][indexs]
      }
  ]


def local_retrieve_memory(args):
  global_data_ids, global_memory_ids = args
  return [
      {
          'keys': kb.memory_flatten['keys'][global_memory_ids],
          'values': kb.memory_flatten['values'][global_memory_ids],
          # 'masks': kb.memory['masks'][device_id][indexs]
      },
      {
          'image': kb.memory_flatten['image'][global_data_ids],
          'text_tokens': kb.memory_flatten['text'][global_data_ids]
      }
  ]


def retrieve_top_memory(args):
  top1_ids = args
  return [
      {
          'keys': kb.memory_flatten['keys'][top1_ids],
          'values': kb.memory_flatten['values'][top1_ids],
          # 'masks': kb.memory['masks'][device_id][indexs]
      },
      {
          'image': kb.memory_flatten['image'][top1_ids],
          'text_tokens': kb.memory_flatten['text'][top1_ids]
      }
  ]


kb = KnowledgeBase()
