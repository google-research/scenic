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

"""Data generators for the coverage pileup images."""

import functools
from typing import Optional, Tuple

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import tensorflow as tf



def _extract_metadata_from_first_example(filename: str) -> Tuple[int]:
  """Extracts the image shape from the first example."""
  raw_example = next(
      iter(
          tf.data.TFRecordDataset(
              filenames=filename,
              compression_type=_filename_to_compression(filename))))
  example = tf.train.Example.FromString(raw_example.numpy())

  return tuple(example.features.feature['image/shape'].int64_list.value)


def _filename_to_compression(filename: str) -> Optional[str]:
  return 'GZIP' if tf.strings.regex_full_match(filename, '.*.gz') else None


def create_coverage_based_dataset(
    filenames: str,
    with_label: bool = True,
) -> tf.data.Dataset:
  """Creates a coverage based pileup dataset from a filepath.

  Args:
    filenames: The data directory/pattern containing data files.
    with_label: whether to load the labels or not.

  Returns:
    tf.data.Dataset
  """
  logging.info('Finding all data files matching the file pattern.')
  dataset_files = tf.io.matching_files(filenames)

  # Extract image shape from the first example
  shape = _extract_metadata_from_first_example(dataset_files[0])

  proto_features = {
      'variant/encoded': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
      'image/encoded': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
      'image/shape': tf.io.FixedLenFeature(shape=(len(shape),), dtype=tf.int64),
  }
  if with_label:
    proto_features['label'] = tf.io.FixedLenFeature(shape=1, dtype=tf.int64)

  def _process_input(proto_string):
    """Helper function for input function that parses a serialized example."""

    parsed_features = tf.io.parse_single_example(
        serialized=proto_string, features=proto_features)

    features = {
        'image': tf.io.parse_tensor(parsed_features['image/encoded'], tf.uint8),
    }

    features['image'].set_shape(shape)

    if with_label:
      return features, parsed_features['label']
    else:
      return features, None

  compression = _filename_to_compression(dataset_files[0])

  logging.info('Loading TFRecords as bytes.')
  dataset = tf.data.Dataset.from_tensor_slices(dataset_files)

  # pylint: disable=g-long-lambda
  # interleave parallelizes the data loading step by interleaving the I/O
  # operation to read the file. It speeds up the I/O step.
  dataset = dataset.interleave(
      lambda filename: tf.data.TFRecordDataset(
          filename,
          compression_type=compression,
      ).map(
          _process_input,
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
      ),
      cycle_length=len(dataset_files),
  )

  return dataset


def preprocess(features, label):
  """Preprocessing code specific to ViT models."""
  label_tensor = tf.cast(tf.squeeze(label, [-1]), tf.int32)
  return {
      'inputs': tf.image.resize(
          features['image'],
          [256, 256]),  # Resize pileups to make side length divisible by 4.
      'label': tf.one_hot(label_tensor, 3)
  }


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
      # First repeat then shuffle, then batch.
      ds = ds.repeat()
      local_seed = seed  # Seed for this machine.
      if local_seed is not None and input_context:
        local_seed += input_context.input_pipeline_id
      ds = ds.shuffle(shuffle_buffer_size, seed=local_seed)
      ds = ds.batch(replica_batch_size, drop_remainder=True)
    else:  # Test and validation.
      # First batch then repeat.
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


@datasets.add_dataset('pileup_coverage')
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
  """Returns generators for the pileup window train, validation, and test set.

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

  def build_pileup_window_dataset(split):
    """dataset_fn called by data.build_dataset(**kwargs)."""

    if split == 'train':
      if dataset_configs.train_path:
        path = dataset_configs.train_path
      else:
        # Add path to your training data here:
        path = ''
    elif split == 'valid':
      if dataset_configs.eval_path:
        path = dataset_configs.eval_path
      else:
        # Add path to your validation data here:
        path = ''
    elif split == 'test':
      if dataset_configs.test_path:
        path = dataset_configs.test_path
      else:
        # Add path to your test data here:
        path = ''
    else:
      raise ValueError('Invalid split value.')

    if not path:
      raise ValueError('No path provide. Please modify the path variable to '
                       'hardcode the %s path.' %split)
    dataset = create_coverage_based_dataset(filenames=path)

    # Creating a Dataset that includes only 1/num_shards of data so the data is
    # splitted between different hosts.
    num_hosts, host_id = jax.process_count(), jax.process_index()
    dataset = dataset.shard(num_shards=num_hosts, index=host_id)

    dataset = dataset.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

  # Use different seed for each host.
  if shuffle_seed is None:
    local_seed = None
  else:
    data_seed = 0
    local_seed = data_seed + jax.process_index()

  train_dataset = build_dataset(
      dataset_fn=build_pileup_window_dataset,
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
      dataset_fn=build_pileup_window_dataset,
      split='valid',
      batch_size=eval_batch_size,
      strategy=None)

  test_dataset = build_dataset(
      dataset_fn=build_pileup_window_dataset,
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

  num_classes = 3
  image_size = 256
  input_shape = [-1, image_size, image_size, 7]

  meta_data = {
      'num_classes': num_classes,
      'input_shape': input_shape,
      'num_train_examples': 31000 * 24,
      'num_eval_examples': 31000 * 6,
      'num_test_examples': 31000,
      'input_dtype': getattr(jnp, dtype_str),
      'target_is_onehot': True,
  }

  return dataset_utils.Dataset(train_iter, valid_iter, test_iter, meta_data)
