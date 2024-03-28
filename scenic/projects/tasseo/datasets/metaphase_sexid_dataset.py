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

"""Data generators for the metaphase sex ID task."""

import functools
from typing import Optional

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import tensorflow as tf


# path to data's base directory
_BASE_DIR = ''


def _parse_example(serialized_example):
  """Parses feature dictionary from the `serialized_example` proto.

  Args:
    serialized_example: The proto of the current example.

  Returns:
    A parsed example as dict with several elements.
  """
  feature_description = {
      'meta_img': tf.io.FixedLenFeature([], tf.string, default_value=''),
      'label': tf.io.FixedLenFeature([], tf.string, default_value='')
  }

  features = tf.io.parse_single_example(serialized_example, feature_description)

  return features


def _resize(image, image_size):
  """Resizes the image.

  Args:
    image: Tensor; Input image.
    image_size: int; Image size.

  Returns:
    Resized image.
  """
  return tf.image.resize([image], [image_size, image_size],
                         method=tf.image.ResizeMethod.BICUBIC)[0]


def load_data(split, parallel_reads=4):
  """Loads a split of metaphase dataset to be processed on a given host.

  Each host runs this function in parallel and loads a sub-split of the data
  based on its host_id.

  Args:
    split: str; One of 'train', 'eval' or 'test'.
    parallel_reads: int; Number of parallel readers (set to 1 for determinism).

  Returns:
    tf.data.Datasets for training, testing and validation. if
    n_validation_shards is 0, the validation dataset will be None.
  """
  base_dir = _BASE_DIR

  if split == 'test':
    path = base_dir + '/10262021_metaphase_sex_id_fold-0014*'
  else:
    path = base_dir + '/10262021_metaphase_sex_id_fold-00*'

  # Each host is responsible for a fixed subset of data. Here we create a
  # Dataset that includes only 1/num_shards of data so the data is
  # splitted between different hosts.

  num_hosts, host_id = jax.process_count(), jax.process_index()
  filenames = tf.io.matching_files(path)

  if len(filenames) >= num_hosts:
    # Sharding on data sources (e.g. filenames). Each hosts reads a different
    # sub-split of the data based its host_id.
    filenames = np.array_split(filenames, num_hosts)[host_id]

  files = tf.data.Dataset.list_files(filenames)

  # cycle_length is the cycle length of interleaving files, if it's None, it's
  # up to tf.data to decide how much parallelism of files reading. If it's 1,
  # will start reading new file only after the first read finishes.
  data = files.interleave(
      tf.data.TFRecordDataset,
      cycle_length=1 if split != 'train' else parallel_reads,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if len(filenames) < num_hosts:
    # Sharding using tf.shard.
    data = data.shard(num_shards=num_hosts, index=host_id)

  data = data.map(
      _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
    print('Getting data split: %s' % split)
    if split == 'train':
      # first repeat then shuffle, then batch
      ds = ds.repeat()
      local_seed = seed  # seed for this machine
      if local_seed is not None and input_context:
        local_seed += input_context.input_pipeline_id
      ds = ds.shuffle(shuffle_buffer_size, seed=local_seed)
      ds = ds.batch(replica_batch_size, drop_remainder=True)
      print('Dropped remainder: %s' % split)
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


def preprocess(features, label_key):
  """Preprocessing code specific to metaphase images."""
  if isinstance(label_key, str):
    labels = features[label_key]
  else:
    labels = tuple(features[k] for k in label_key)

  image = tf.reshape(
      tf.io.decode_raw(features['meta_img'], tf.float32), (512, 512, 3))
  image = _resize(image, 224)  # resizing for resnet

  labels = 0 if labels == b'male' else 1

  return {
      'inputs': image,
      'label': labels,
  }


@datasets.add_dataset('metaphase_sexid')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                prefetch_buffer_size=2,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None):
  """Returns generators for the metaphase sexid train, validation, and test set.

  Args:
    batch_size: int; Determines the train batch size.
    eval_batch_size: int; Determines the evaluation batch size.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image (e.g. 'float32').
    shuffle_seed: int; Seed for shuffling the training data.
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    prefetch_buffer_size: int; Buffer size for the prefetch.
    dataset_configs: dict; Dataset specific configurations.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """
  del rng
  del dataset_configs

  def build_metaphase_dataset(split='train', shuffle=False):
    """dataset_fn called by data.build_dataset(**kwargs)."""
    par_reads = 4 if shuffle else 1
    ds = load_data(split, parallel_reads=par_reads)
    ds = ds.map(lambda x: preprocess(x, 'label'))
    return ds

  # use different seed for each host
  if shuffle_seed is None:
    local_seed = None
  else:
    data_seed = 0
    local_seed = data_seed + jax.process_index()

  train_dataset = build_dataset(
      dataset_fn=build_metaphase_dataset,
      batch_size=batch_size,
      seed=local_seed,
      split='train',
      strategy=None)

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you '
                       'want to run with dataset service.')
    logging.info('Using the tf.data service at %s', dataset_service_address)
    train_dataset = dataset_utils.distribute(train_dataset,
                                             dataset_service_address)

  eval_dataset = build_dataset(
      dataset_fn=build_metaphase_dataset,
      split='valid',
      batch_size=eval_batch_size,
      strategy=None)

  test_dataset = build_dataset(
      dataset_fn=build_metaphase_dataset,
      split='test',
      batch_size=eval_batch_size,
      strategy=None)

  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size)
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch, train=False, batch_size=eval_batch_size)

  train_iter = iter(train_dataset)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(shard_batches, train_iter)
  train_iter = jax_utils.prefetch_to_device(train_iter, prefetch_buffer_size)

  valid_iter = iter(eval_dataset)
  valid_iter = map(dataset_utils.tf_to_numpy, valid_iter)
  valid_iter = map(maybe_pad_batches_eval, valid_iter)
  valid_iter = map(shard_batches, valid_iter)
  valid_iter = jax_utils.prefetch_to_device(valid_iter, prefetch_buffer_size)

  test_iter = iter(test_dataset)
  test_iter = map(dataset_utils.tf_to_numpy, test_iter)
  test_iter = map(maybe_pad_batches_eval, test_iter)
  test_iter = map(shard_batches, test_iter)
  test_iter = jax_utils.prefetch_to_device(test_iter, prefetch_buffer_size)

  num_classes = 2
  image_size = 224
  input_shape = [-1, image_size, image_size, 3]

  meta_data = {
      'num_classes': num_classes,
      'input_shape': input_shape,
      'num_train_examples': 80000,
      'num_eval_examples': 10000,
      'input_dtype': getattr(jnp, dtype_str),
      'target_is_onehot': False,
  }

  return dataset_utils.Dataset(train_iter, valid_iter, test_iter, meta_data)
