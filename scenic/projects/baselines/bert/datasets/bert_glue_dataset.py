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

"""Scenic wrapper around BERT GLUE.

General Language Understanding Evaluation benchmark (GLUE) dataset from
Tensorflow Models.
"""

import functools
import json
from typing import Optional

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import tensorflow as tf

from official.nlp.data.bert import input_pipeline

_TYPE_VOCAB_SIZE = 2
_VOCAB_SIZE = 30522
_LABEL_TYPES_MAP = {'int': tf.int64, 'float': tf.float32}

REGRESSION_TASKS = ['stsb']
SEQ_CLASSIFICATION_TASKS = ['cola', 'sst2']
SEQ_PAIR_CLASSIFICATION_TASKS = [
    'mrpc', 'qqp', 'mnli_matched', 'mnli_mismatched', 'rte', 'wnli', 'qnli'
]
DEBUGGING_TASKS = ['ax']

_SUPPORTED_TASK_NAMES = (
    REGRESSION_TASKS + SEQ_CLASSIFICATION_TASKS +
    SEQ_PAIR_CLASSIFICATION_TASKS + DEBUGGING_TASKS)


def create_classifier_dataset(file_path,
                              seq_length,
                              batch_size,
                              is_training=True,
                              repeats=None,
                              input_pipeline_context=None,
                              label_type=tf.int64,
                              include_sample_weights=False,
                              num_samples=None):
  """Creates input dataset from (tf)records files for train/eval."""
  name_to_features = {
      'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'label_ids': tf.io.FixedLenFeature([], label_type),
  }
  if include_sample_weights:
    name_to_features['weight'] = tf.io.FixedLenFeature([], tf.float32)
  dataset = input_pipeline.single_file_dataset(
      file_path, name_to_features, num_samples=num_samples)

  # The dataset is always sharded by number of hosts.
  # num_input_pipelines is the number of hosts rather than number of cores.
  if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
    dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                            input_pipeline_context.input_pipeline_id)

  def _select_data_from_record(record):
    x = {
        'input_word_ids': record['input_ids'],
        'input_mask': record['input_mask'],
        'input_type_ids': record['segment_ids']
    }
    y = record['label_ids']
    if include_sample_weights:
      w = record['weight']
      return (x, y, w)
    return (x, y)

  if is_training:
    # The correct way would be to repeat and then shuffle but this is how it is
    # done in:
    # https://github.com/tensorflow/models/blob/master/official/nlp/bert/input_pipeline.py#L190
    dataset = dataset.shuffle(100)
    dataset = dataset.repeat(repeats)

  dataset = dataset.map(
      _select_data_from_record,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=is_training)

  if not is_training:
    dataset = dataset.repeat(repeats)
  return dataset


def get_dataset_fn(input_file_pattern,
                   max_seq_length,
                   batch_size,
                   is_training,
                   repeats=None,
                   label_type=tf.int64,
                   include_sample_weights=False,
                   num_samples=None):
  """Gets a closure to create a dataset."""

  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    dataset = create_classifier_dataset(
        tf.io.gfile.glob(input_file_pattern),
        max_seq_length,
        batch_size,
        is_training=is_training,
        repeats=repeats,
        input_pipeline_context=ctx,
        label_type=label_type,
        include_sample_weights=include_sample_weights,
        num_samples=num_samples)
    return dataset

  return _dataset_fn


def postprocess(batch, task):
  """Post process the batch and make it ready to be sent to the model."""
  if task == 'stsb':  # Regression task
    return dict(targets=batch[1][:, None], **batch[0])
  return dict(label=batch[1], **batch[0])


@datasets.add_dataset('bert_glue')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None):
  """Returns generators for the BERT classification task, train and validation.

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
  del dtype_str
  del dataset_service_address
  del shuffle_seed
  assert dataset_configs is not None
  if dataset_configs.task not in _SUPPORTED_TASK_NAMES:
    raise ValueError('dataset_configs.task_name must be one of [{}].'.format(
        ', '.join(_SUPPORTED_TASK_NAMES)))
  with tf.io.gfile.GFile(dataset_configs.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))
  label_type = _LABEL_TYPES_MAP[input_meta_data.get('label_type', 'int')]
  include_sample_weights = input_meta_data.get('has_sample_weights', False)

  logging.info('Loading train split of the %s dataset.', dataset_configs.task)
  train_input_fn = get_dataset_fn(
      dataset_configs.train_data_path,
      input_meta_data['max_seq_length'],
      batch_size,
      is_training=True,
      label_type=label_type,
      include_sample_weights=include_sample_weights,
      num_samples=input_meta_data['train_data_size'])

  input_context = tf.distribute.InputContext(
      num_input_pipelines=jax.process_count(),
      input_pipeline_id=jax.process_index(),
      num_replicas_in_sync=jax.process_count())

  train_ds = train_input_fn(ctx=input_context).prefetch(
      dataset_configs.get('prefetch_to_host', 2))

  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch,
      train=True,
      batch_size=batch_size,
      inputs_key='input_word_ids')
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
  task_postprocess = functools.partial(postprocess, task=dataset_configs.task)

  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(task_postprocess, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(shard_batches, train_iter)
  if dataset_configs.prefetch_to_device:
    train_iter = jax_utils.prefetch_to_device(
        train_iter, dataset_configs.prefetch_to_device)

  logging.info('Loading validation split of the %s dataset.',
               dataset_configs.task)
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch,
      train=False,
      batch_size=eval_batch_size,
      inputs_key='input_word_ids')

  eval_input_fn = get_dataset_fn(
      dataset_configs.eval_data_path,
      input_meta_data['max_seq_length'],
      eval_batch_size,
      is_training=False,
      label_type=label_type,
      include_sample_weights=include_sample_weights)

  val_ds = eval_input_fn(ctx=input_context).prefetch(
      dataset_configs.get('prefetch_to_host', 2))

  valid_iter = iter(val_ds)
  valid_iter = map(dataset_utils.tf_to_numpy, valid_iter)
  valid_iter = map(task_postprocess, valid_iter)
  valid_iter = map(maybe_pad_batches_eval, valid_iter)
  valid_iter = map(shard_batches, valid_iter)

  if dataset_configs.prefetch_to_device:
    valid_iter = jax_utils.prefetch_to_device(
        valid_iter, dataset_configs.prefetch_to_device)

  input_shape = (-1, input_meta_data['max_seq_length'])

  input_spec = {
      'input_word_ids': (input_shape, jnp.int32),
      'input_mask': (input_shape, jnp.int32),
      'input_type_ids': (input_shape, jnp.int32),
  }
  num_classes = (
      None if dataset_configs.task == 'stsb'  # Regression task!
      else input_meta_data['num_labels'])
  meta_data = {
      'type_vocab_size': _TYPE_VOCAB_SIZE,
      'vocab_size': _VOCAB_SIZE,
      'input_spec': input_spec,
      'num_classes': num_classes,
      'num_train_examples': input_meta_data['train_data_size'],
      'num_eval_examples': input_meta_data['eval_data_size'],
      'target_is_onehot': False,
  }
  return dataset_utils.Dataset(train_iter, valid_iter, None, meta_data)
