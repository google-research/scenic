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

"""Utilities for tasseo trainer."""

from absl import logging
import jax
import numpy as np
import tensorflow as tf


def parse_example(serialized_example):
  """Parses feature dictionary from the `serialized_example` proto.

  Args:
    serialized_example: The proto of the current example.

  Returns:
    A parsed example as dict with several elements.
  """
  feature_description = {
      'chrm_img': tf.io.FixedLenFeature([], tf.string, default_value=''),
      'meta_img': tf.io.FixedLenFeature([], tf.string, default_value=''),
      'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
      'chrm_path': tf.io.FixedLenFeature([], tf.string, default_value=''),
  }

  features = tf.io.parse_single_example(serialized_example, feature_description)

  return features


def load_data(prefix, is_train=False, parallel_reads=4):
  """Loads the metaphase dataset.

  Args:
    prefix: str; Dataset path prefix to tfrecords.
    is_train: bool; is the dataset used for training?
    parallel_reads: int; Number of parallel readers (set to 1 for determinism).

  Returns:
    tf.data.Datasets for training, testing and validation.
  """
  num_hosts = jax.process_count()
  host_id = jax.process_index()

  # We shard the data between different hosts and create a Dataset that includes
  # only 1/num_shards of full dataset.
  filenames = tf.io.matching_files(prefix + '*')
  filenames_host_split = np.array_split(filenames, num_hosts)[host_id]
  logging.info('Host id=%d assigned %d out of %d dataset filenames (train=%r).',
               host_id, len(filenames_host_split), len(filenames), is_train)
  if not list(filenames_host_split):
    raise ValueError(
        'Zero dataset filenames assigned to host %d for reading; %d available'
        ' for all hosts' % (host_id, len(filenames)))
  files = tf.data.Dataset.list_files(filenames_host_split)

  data = files.interleave(
      tf.data.TFRecordDataset,
      cycle_length=1 if not is_train else parallel_reads,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  data = data.map(
      parse_example,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return data


def build_dataset(dataset_fn,
                  batch_size=None,
                  shuffle_buffer_size=256,
                  seed=None,
                  strategy=None,
                  **dataset_kwargs):
  """Dataset builder that takes care of strategy, batching and shuffling.

  Args:
    dataset_fn: function; A function that loads the dataset.
    batch_size: int; Size of the batch.
    shuffle_buffer_size: int; Size of the buffer for used for shuffling.
    seed: int; Random seed used for shuffling.
    strategy: TF strategy that handles the distribution policy.
    **dataset_kwargs: dict; Arguments passed to TFDS.

  Returns:
    Dataset.
  """

  def _dataset_fn(input_context=None):
    """Dataset function."""
    replica_batch_size = batch_size
    if input_context:
      replica_batch_size = input_context.get_per_replica_batch_size(batch_size)
    ds = dataset_fn(**dataset_kwargs)
    split = dataset_kwargs.get('split')
    if split == 'train':
      # first repeat then shuffle, then batch
      ds = ds.repeat()
      local_seed = seed  # seed for this machine
      if local_seed is not None and input_context:
        local_seed += input_context.input_pipeline_id
      ds = ds.shuffle(shuffle_buffer_size, seed=local_seed)
      ds = ds.batch(replica_batch_size, drop_remainder=True)
    else:  # test and validation
      # first batch then repeat
      ds = ds.batch(replica_batch_size, drop_remainder=False)
      ds = ds.repeat()
    options = tf.data.Options()
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)
    return ds.prefetch(tf.data.experimental.AUTOTUNE)

  if strategy:
    ds = strategy.experimental_distribute_datasets_from_function(_dataset_fn)
  else:
    ds = _dataset_fn()

  return ds


def pad(
    key,
    max_length=100,
    pad_char='#',
):
  """Pads a key string to a fixed length.

  We have to iteratively find an integer eqaul to key_length tensor to be able
  to multiply it by pad_char and make a proper pad.

  Args:
    key: The original key, usually a path-like string.
    max_length: Length of the desired key.
    pad_char: Character to be used for padding.

  Returns:
    Padded key of the required length.
  """
  padding = ''
  # Transform the key into the basename of the file without the extension.
  key = tf.strings.join([
      tf.strings.split(key, sep='/')[-2], '/',
      tf.strings.split(key, sep='/')[-1]
  ])
  key = tf.strings.regex_replace(key, '.png', '')
  for padding_size in range(1, max_length):
    if tf.math.equal(padding_size, max_length - tf.strings.length(key)):
      padding = padding_size * pad_char
  padded_key = tf.strings.join([padding, key])
  return padded_key


def preprocess(features, label_key, chrm_image_shape, class_names=None):
  """Preprocessing code."""
  if isinstance(label_key, str):
    labels = features[label_key]
  else:
    labels = tuple(features[k] for k in label_key)

  class_names = tf.convert_to_tensor(class_names)
  chrm = tf.reshape(
      tf.io.decode_raw(features['chrm_img'], tf.float32),
      tuple(chrm_image_shape) + (1,))
  labels = labels == class_names  # Creates one-hot label.

  return {
      'inputs': chrm,
      'label': labels,
      'key': tf.strings.unicode_decode(pad(features['chrm_path']), 'UTF-8'),
  }
