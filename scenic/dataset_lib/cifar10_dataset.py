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

"""Data generators for the CIFAR10 dataset."""

import functools
from typing import Optional

from absl import logging
import jax.numpy as jnp
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import tensorflow as tf

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3

# Computed from the training set by taking the per-channel mean/std-dev
# over sample, height and width axes of all training samples.
MEAN_RGB = [0.4914 * 255, 0.4822 * 255, 0.4465 * 255]
STDDEV_RGB = [0.2470 * 255, 0.2435 * 255, 0.2616 * 255]


def preprocess_example(example, dtype=tf.float32):
  """Preprocesses the given example.

  Args:
    example: dict; Example that has an 'image' and a 'label'.
    dtype: Tensorflow data type; Data type of the image.

  Returns:
    A preprocessed example.
  """
  image = tf.cast(example['image'], dtype=dtype)
  if dtype not in [tf.int32, tf.int64, tf.uint32, tf.uint64]:
    mean_rgb = tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=dtype)
    std_rgb = tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=dtype)
    image = (image - mean_rgb) / std_rgb
  return {'inputs': image, 'label': example['label']}


def augment_example(example, dtype=tf.float32, data_augmentations=None):
  """Apply data augmentation on the given training example.

  Args:
    example: dict; Example that has an 'image' and a 'label'.
    dtype: Tensorflow data type; Data type of the image.
    data_augmentations: list(str); Types of data augmentation applied on
      training data.

  Returns:
    An augmented training example.
  """
  image = tf.cast(example['inputs'], dtype=dtype)
  if data_augmentations is not None:
    if 'cifar_default' in data_augmentations:
      image = dataset_utils.augment_random_crop_flip(
          image, HEIGHT, WIDTH, NUM_CHANNELS, crop_padding=4, flip=True)
    image = tf.cast(image, dtype=dtype)
  return {'inputs': image, 'label': example['label']}


@datasets.add_dataset('cifar10')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None):
  """Returns generators for the CIFAR10 train, validation, and test set.

  Args:
    batch_size: int; Determines the train batch size.
    eval_batch_size: int; Determines the evaluation batch size.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image (e.g. 'float32').
    shuffle_seed: int; Seed for shuffling the training data.
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    dataset_configs: dict; Dataset specific configurations.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """
  del rng
  dataset_configs = dataset_configs or {}
  data_augmentations = dataset_configs.get('data_augmentations', [])
  # alwayse include the default data augmentation
  data_augmentations.append('cifar_default')
  for da in data_augmentations:
    if da not in ['mixup', 'cifar_default']:
      raise ValueError(f'Data augmentation type {da} is not yet supported '
                       f'in the CIFAR dataset.')

  dtype = getattr(tf, dtype_str)
  target_is_onehot = False
  preprocess_ex = functools.partial(preprocess_example, dtype=dtype)

  logging.info('Loading train split of the CIFAR10 dataset.')
  augment_ex = functools.partial(
      augment_example, dtype=dtype, data_augmentations=data_augmentations)
  train_ds, train_ds_info = dataset_utils.load_split_from_tfds(
      'cifar10',
      batch_size,
      split='train',
      preprocess_example=preprocess_ex,
      augment_train_example=augment_ex,
      shuffle_seed=shuffle_seed)

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you '
                       'want to run with dataset service.')
    logging.info('Using the tf.data service at %s', dataset_service_address)
    train_ds = dataset_utils.distribute(train_ds, dataset_service_address)

  logging.info('Loading test split of the CIFAR10 dataset.')
  eval_ds, _ = dataset_utils.load_split_from_tfds(
      'cifar10',
      eval_batch_size,
      split='test',
      preprocess_example=preprocess_ex)

  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size)
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch, train=False, batch_size=eval_batch_size)
  num_classes = train_ds_info.features['label'].num_classes
  target_to_one_hot_batches = functools.partial(
      dataset_utils.target_to_one_hot, num_classes=num_classes)
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
  mixup_batches = functools.partial(dataset_utils.mixup, alpha=1.0)

  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  if 'mixup' in data_augmentations:
    train_iter = map(target_to_one_hot_batches, train_iter)
    train_iter = map(mixup_batches, train_iter)
    target_is_onehot = True
  train_iter = map(shard_batches, train_iter)

  # Note: samples will be dropped if the number of test samples
  # (EVAL_IMAGES=10000) is not divisible by the evaluation batch size
  eval_iter = iter(eval_ds)
  eval_iter = map(dataset_utils.tf_to_numpy, eval_iter)
  eval_iter = map(maybe_pad_batches_eval, eval_iter)
  if target_is_onehot:
    eval_iter = map(target_to_one_hot_batches, eval_iter)
  eval_iter = map(shard_batches, eval_iter)

  input_shape = (-1, HEIGHT, WIDTH, NUM_CHANNELS)
  meta_data = {
      'num_classes': num_classes,
      'input_shape': input_shape,
      'num_train_examples': dataset_utils.get_num_examples('cifar10', 'train'),
      'num_eval_examples': dataset_utils.get_num_examples('cifar10', 'test'),
      'input_dtype': getattr(jnp, dtype_str),
      'target_is_onehot': target_is_onehot,
  }
  return dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)
