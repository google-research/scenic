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

"""Dataset for chromosome ID task with baseline-formatted examples."""

import functools
from typing import Optional

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.projects.tasseo import dataset_utils as ts_dataset_utils
import tensorflow as tf


NUM_TRAIN_EXAMPLES = 368_106
NUM_TEST_EXAMPLES = 40_620
NUM_CLASSES = 24
NUM_CHANNELS = 1

# TODO(shamsiz) Filter out abnormal karyograms in (99,49) and (149,69) datasets.
# path to data's base directory
DATASET_BASE_DIRS = {
    (99, 49): '',
    (149, 69): '',
    (199, 99): '',
}


@datasets.add_dataset('chrmID_baseline')
def get_dataset(
    *,
    batch_size,
    eval_batch_size,
    num_shards,
    dtype_str='float32',
    shuffle_seed=0,
    rng=None,
    prefetch_buffer_size=2,
    dataset_configs=None,
    dataset_service_address: Optional[str] = None):
  """Returns generators for the chrmID train, validation, and test set.

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
  try:
    dataset_base_dir = DATASET_BASE_DIRS[tuple(
        dataset_configs.chrm_image_shape)]
  except KeyError as key_error:
    raise ValueError('No dataset found matching "%s"; options: %r' %
                     (dataset_configs.chrm_image_shape,
                      DATASET_BASE_DIRS.keys())) from key_error

  def build_baseline_dataset(split='train', shuffle=False):
    """dataset_fn called by data.build_dataset(**kwargs)."""
    parallel_reads = 4 if shuffle else 1

    ds = load_data(
        dataset_base_dir,
        split,
        parallel_reads=parallel_reads)
    ds = ds.map(
        lambda x: preprocess(x, 'label', dataset_configs.chrm_image_shape))
    return ds

  # use different seed for each host
  if shuffle_seed is None:
    local_seed = None
  else:
    data_seed = 0
    local_seed = data_seed + jax.process_index()

  train_dataset = build_dataset(
      dataset_fn=build_baseline_dataset,
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
      dataset_fn=build_baseline_dataset,
      split='valid',
      batch_size=eval_batch_size,
      strategy=None)

  test_dataset = build_dataset(
      dataset_fn=build_baseline_dataset,
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

  input_shape = [-1] + list(dataset_configs.chrm_image_shape) + [NUM_CHANNELS]
  meta_data = {
      'num_classes': NUM_CLASSES,
      'input_shape': input_shape,
      'num_train_examples': NUM_TRAIN_EXAMPLES,
      'num_eval_examples': NUM_TEST_EXAMPLES,
      'input_dtype': getattr(jnp, dtype_str),
      'target_is_onehot': True,
  }

  return dataset_utils.Dataset(train_iter, valid_iter, test_iter, meta_data)


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


def load_data(base_dir, split, parallel_reads=4):
  """Loads the metaphase dataset.

  Args:
    base_dir: Base directory containing dataset as tfrecords.
    split: str; One of 'train', 'eval' or 'test'.
    parallel_reads: int; Number of parallel readers (set to 1 for determinism).

  Returns:
    tf.data.Datasets for training, testing and validation.
  """
  num_hosts = jax.process_count()
  host_id = jax.process_index()

  if split == 'train':
    path = base_dir + '[0-7]-00*'
  elif split == 'validation':
    path = base_dir + '8-00*'
  else:
    path = base_dir + '9-00*'

  # We shard the data between different hosts and create a Dataset that includes
  # only 1/num_shards of full dataset.
  filenames = tf.io.matching_files(path)
  filenames_host_split = np.array_split(filenames, num_hosts)[host_id]
  files = tf.data.Dataset.list_files(filenames_host_split)

  data = files.interleave(
      tf.data.TFRecordDataset,
      cycle_length=1 if split != 'train' else parallel_reads,
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


def preprocess(features, label_key, chrm_image_shape):
  """Preprocessing code specific to metaphase images."""
  if isinstance(label_key, str):
    labels = features[label_key]
  else:
    labels = tuple(features[k] for k in label_key)

  class_names = tf.convert_to_tensor([
      b'chrm_%d' % i for i in range(1, 23)] + [b'chrm_X', b'chrm_Y'])
  chrm = tf.reshape(
      tf.io.decode_raw(features['chrm_img'], tf.float32),
      tuple(chrm_image_shape) + (1,))
  labels = labels == class_names  # Creates one-hot label.

  return {
      'inputs':
          chrm,
      'label':
          labels,
      'key':
          tf.strings.unicode_decode(
              ts_dataset_utils.pad(features['chrm_path']), 'UTF-8'),
  }
