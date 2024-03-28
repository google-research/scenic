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

"""Scenic wrapper around Wikipedia dataset from Tensorflow Models."""

import functools
from typing import Optional

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import tensorflow as tf

from official.nlp.data import pretrain_dataloader

_TYPE_VOCAB_SIZE = 2
_VOCAB_SIZE = 30522
_NUM_TRAIN_EXAMPLES = 200_000_000
_NUM_EVAL_EXAMPLES = 344793


def reduce_next_sentence_label_dimension(batch):
  """Change next_sentence_labels's shape from (-1, 1) to (-1,).

  Args:
    batch: A dictionary mapping keys to arrays.

  Returns:
    Updated batch.
  """

  batch['next_sentence_labels'] = batch['next_sentence_labels'][:, 0]
  return batch


@datasets.add_dataset('bert_wikibooks')
def get_dataset(
    *,
    batch_size,
    eval_batch_size,
    num_shards,
    dtype_str='float32',  # pylint: disable=unused-argument
    shuffle_seed=0,
    rng=None,
    dataset_configs=None,
    dataset_service_address: Optional[str] = None):  # pylint: disable=unused-argument
  """Returns generators for the Wikipedia train and validation sets.

  Args:
    batch_size: int; Determines the train batch size.
    eval_batch_size: int; Determines the evaluation batch size.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type for inputs. Not used.
    shuffle_seed: int; Seed for shuffling the training data.
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    dataset_configs: dict; Dataset specific configurations.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address. Not used.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """
  del rng
  assert dataset_configs is not None
  logging.info('Loading train split of the wikibooks dataset.')

  with dataset_configs.unlocked():
    dataset_configs.train_data_loader.seed = shuffle_seed
    dataset_configs.train_data_loader.is_training = True
    dataset_configs.train_data_loader.cache = False

  train_data_loader = pretrain_dataloader.BertPretrainDataLoader(
      dataset_configs.train_data_loader)
  input_context = tf.distribute.InputContext(
      num_input_pipelines=jax.process_count(),
      input_pipeline_id=jax.process_index(),
      num_replicas_in_sync=jax.process_count())

  train_ds = train_data_loader.load(input_context=input_context).prefetch(
      dataset_configs.get('prefetch_to_host', 2))

  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch,
      train=True,
      batch_size=batch_size,
      inputs_key='input_word_ids')
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)

  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(reduce_next_sentence_label_dimension, train_iter)
  train_iter = map(shard_batches, train_iter)
  if dataset_configs.prefetch_to_device:
    train_iter = jax_utils.prefetch_to_device(
        train_iter, dataset_configs.prefetch_to_device)

  logging.info('Loading validation split of the wikibooks dataset.')
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch,
      train=False,
      batch_size=eval_batch_size,
      inputs_key='input_word_ids')

  with dataset_configs.unlocked():
    dataset_configs.val_data_loader.seed = shuffle_seed
    # Some tricks to make sure that the dataset is repeated but not shuffled.
    dataset_configs.val_data_loader.is_training = True
    dataset_configs.val_data_loader.cache = False
    dataset_configs.val_data_loader.shuffle_buffer_size = 1

  val_data_loader = pretrain_dataloader.BertPretrainDataLoader(
      dataset_configs.val_data_loader)
  val_ds = val_data_loader.load(input_context=input_context).prefetch(
      dataset_configs.get('prefetch_to_host', 2))

  valid_iter = iter(val_ds)
  valid_iter = map(dataset_utils.tf_to_numpy, valid_iter)
  valid_iter = map(maybe_pad_batches_eval, valid_iter)
  valid_iter = map(reduce_next_sentence_label_dimension, valid_iter)
  valid_iter = map(shard_batches, valid_iter)
  if dataset_configs.prefetch_to_device:
    valid_iter = jax_utils.prefetch_to_device(
        valid_iter, dataset_configs.prefetch_to_device)

  input_shape = (-1, dataset_configs.train_data_loader.seq_length)

  input_spec = {
      'input_word_ids': (input_shape, jnp.int32),
      'input_mask': (input_shape, jnp.int32),
      'input_type_ids': (input_shape, jnp.int32),
      'masked_lm_positions': (
          (-1, dataset_configs.train_data_loader.max_predictions_per_seq),
          jnp.int32)
  }

  meta_data = {
      'type_vocab_size': _TYPE_VOCAB_SIZE,
      'vocab_size': _VOCAB_SIZE,
      'input_spec': input_spec,
      # TODO(vlikhosherstov): Put the real value.
      'num_train_examples': _NUM_TRAIN_EXAMPLES,
      'num_eval_examples': _NUM_EVAL_EXAMPLES,
  }
  if dataset_configs.get('extra_meta_data'):
    for k, v in dataset_configs.extra_meta_data.items():
      meta_data[k] = v
  return dataset_utils.Dataset(train_iter, valid_iter, None, meta_data)
