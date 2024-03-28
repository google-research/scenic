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

"""Data generators for ModelNet40 dataset.

Input data format:
Point Cloud - a tensor of shape [seq_length, 3] containing (x, y, z) coordinates
Label - a class label for the point cloud object
"""

import dataclasses
import functools
from typing import Optional

from absl import logging
from flax import jax_utils
import jax.numpy as jnp
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclasses.dataclass
class ModelNet40DatasetConfig:
  """Dataset config."""

  dataset_path: str = ''

  num_train_pointclouds: int = 9843
  num_eval_pointclouds: int = 2468
  seq_length: int = 1024
  dataset_name: str = 'modelnet40_classification'
  version: str = '1.0.0'
  num_classes: int = 40


def get_dataset_config(seq_length):
  """Returns dataset config."""
  if seq_length == 1024:
    return ModelNet40DatasetConfig(
        seq_length=1024)
  elif seq_length == 2048:
    return ModelNet40DatasetConfig(
        num_train_pointclouds=9840,
        seq_length=2048,
        dataset_name='modelnet40_h5')
  elif seq_length == 4096:
    return ModelNet40DatasetConfig(
        seq_length=4096,
        version='1.1.0')
  elif seq_length == 8192:
    return ModelNet40DatasetConfig(
        seq_length=8192,
        version='1.2.0')
  return ModelNet40DatasetConfig()


def modelnet40_load_split(batch_size,
                          dataset_config,
                          train,
                          onehot_labels=True,
                          dtype=tf.float32,
                          prefetch_buffer_size=10,
                          shuffle_seed=None):
  """Creates a split from the ModelNet40 dataset using TensorFlow Datasets.

  For the training set, we drop the last partial batch. This is fine to do
  because we additionally shuffle the data randomly each epoch, thus the trainer
  will see all data in expectation. For the validation set, we pad the final
  batch to the desired batch size.

  Args:
    batch_size: int; The batch size returned by the data pipeline.
    dataset_config: dataset configuration.
    train: bool; Whether to load the train or evaluation split.
    onehot_labels: Whether to transform the labels to one hot.
    dtype: TF data type; Data type of the image.
    prefetch_buffer_size: int; Buffer size for the TFDS prefetch.
    shuffle_seed: The seed to use when shuffling the train split.

  Returns:
    A `tf.data.Dataset`.
  """

  def random_noise(pointcloud, std=0.02):
    assert len(pointcloud.shape) == 2
    noise = tf.random.normal(tf.shape(pointcloud), mean=0, stddev=std/2)
    noisy_pointcloud = pointcloud + tf.clip_by_value(
        noise, clip_value_min=-std, clip_value_max=std)
    return noisy_pointcloud

  def decode_example(example):
    pointcloud = tf.cast(example['pc'], dtype=dtype)
    if train:
      pointcloud = random_noise(pointcloud)
      pointcloud = tf.random.shuffle(pointcloud)

    label = example['label']
    label = tf.one_hot(label,
                       dataset_config.num_classes) if onehot_labels else label
    return {'inputs': pointcloud, 'label': label}

  split = 'train' if train else 'test'
  dataset_builder = dataset_utils.get_dataset_tfds(
      dataset_config.dataset_name,
      split=split,
      data_dir=dataset_config.dataset_path,
      skip_decode=('pc',))
  # Download dataset:
  dataset_builder.download_and_prepare()

  ds = dataset_builder.as_dataset(
      split=split, decoders={
          'pc': tfds.decode.SkipDecoding(),
      })
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  ds = ds.cache()

  if train:
    ds = ds.repeat()
    ds = ds.shuffle(16 * batch_size, seed=shuffle_seed)

  # decode_example should be applied after caching as it also does augmentation
  ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size, drop_remainder=train)

  ds = ds.prefetch(prefetch_buffer_size)
  return ds


@datasets.add_dataset('modelnet40')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                prefetch_buffer_size=2,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None,
                seq_length=1024):
  """Returns generators for the ModelNet40 train, validation, and test sets.

  Args:
    batch_size: int; Determines the train batch size.
    eval_batch_size: int; Determines the evaluation batch size.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image (e.g. 'float32').
    shuffle_seed: int; Seed for shuffling the training data.
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    prefetch_buffer_size: int; Buffer size for the device prefetch.
    dataset_configs: dict; Dataset specific configurations.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address.
    seq_length: maximum sequence length.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """
  modelnet40config = get_dataset_config(seq_length)
  dataset_configs = dataset_configs or {}
  del rng
  data_augmentations = dataset_configs.get('data_augmentations', ['default'])
  # TODO(dehghani): add mixup data augmentation.
  for da in data_augmentations:
    if da not in ['default']:
      raise ValueError(f'Data augmentation {data_augmentations} is not '
                       f'(yet) supported in the ModelNet40 dataset.')
  dtype = getattr(tf, dtype_str)
  onehot_labels = dataset_configs.get('onehot_labels', False)

  logging.info('Loading train split of the ModelNet40 dataset.')
  train_ds = modelnet40_load_split(
      batch_size,
      modelnet40config,
      train=True,
      onehot_labels=onehot_labels,
      dtype=dtype,
      shuffle_seed=shuffle_seed)

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you '
                       'want to run with dataset service.')
    logging.info('Using the tf.data service at %s', dataset_service_address)
    train_ds = dataset_utils.distribute(train_ds, dataset_service_address)

  logging.info('Loading validation split of the ModelNet40 dataset.')
  eval_ds = modelnet40_load_split(
      eval_batch_size,
      modelnet40config, train=False, onehot_labels=onehot_labels, dtype=dtype)

  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size)
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch, train=False, batch_size=eval_batch_size)
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)

  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(shard_batches, train_iter)
  train_iter = jax_utils.prefetch_to_device(train_iter, prefetch_buffer_size)

  eval_iter = iter(eval_ds)
  eval_iter = map(dataset_utils.tf_to_numpy, eval_iter)
  eval_iter = map(maybe_pad_batches_eval, eval_iter)
  eval_iter = map(shard_batches, eval_iter)
  eval_iter = jax_utils.prefetch_to_device(eval_iter, prefetch_buffer_size)

  input_shape = (-1, seq_length, 3)

  meta_data = {
      'num_classes': modelnet40config.num_classes,
      'input_shape': input_shape,
      'num_train_examples': modelnet40config.num_train_pointclouds,
      'num_eval_examples': modelnet40config.num_eval_pointclouds,
      'input_dtype': getattr(jnp, dtype_str),
      'target_is_onehot': onehot_labels,
  }
  return dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)

