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

"""Data generators for the SVHN dataset.

The Street View House Numbers (SVHN) Dataset is an image digit recognition
  dataset of over 600,000 color digit images coming from real world data.
  Split size:
    - Training set: 73,257 images
    - Testing set: 26,032 images
    - Extra training set: 531,131 images
  Following the common setup on SVHN, we only use the official training and
  testing data. Images are cropped to 32x32.

  URL: http://ufldl.stanford.edu/housenumbers/
"""

import functools
from typing import Optional

from absl import logging
import jax.numpy as jnp
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import tensorflow as tf


def preprocess_example(example, dtype=tf.float32):
  """Preprocesses the given example.

  Args:
    example: dict; Example that has an 'image' and a 'label'.
    dtype: Tensorflow data type; Data type of the image.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = tf.cast(example['image'], dtype=dtype)
  if dtype not in [tf.int32, tf.int64, tf.uint32, tf.uint64]:
    image /= tf.constant(255.0, shape=[1, 1, 1], dtype=dtype)
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
    if 'random_crop_flip' in data_augmentations:
      image = dataset_utils.augment_random_crop_flip(
          image, crop_padding=4, flip=True)
    image = tf.cast(image, dtype=dtype)
  return {'inputs': image, 'label': example['label']}


@datasets.add_dataset('svhn')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None):
  """Returns generators for the SVHN train, validation, and test set.

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
  for da in data_augmentations:
    if da not in ['random_crop_flip']:
      raise ValueError(f'Data augmentation type {da} is not yet supported '
                       f'in the SVHN dataset.')

  dtype = getattr(tf, dtype_str)
  preprocess_ex = functools.partial(preprocess_example, dtype=dtype)

  logging.info('Loading train split of the SVHN dataset.')
  augment_ex = functools.partial(
      augment_example, dtype=dtype, data_augmentations=data_augmentations)
  train_ds, train_ds_info = dataset_utils.load_split_from_tfds(
      'svhn_cropped:3.*.*',
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

  logging.info('Loading test split of the SVHN dataset.')
  eval_ds, _ = dataset_utils.load_split_from_tfds(
      'svhn_cropped:3.*.*',
      eval_batch_size,
      split='test',
      preprocess_example=preprocess_ex)

  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size)
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch, train=False, batch_size=eval_batch_size)

  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(shard_batches, train_iter)

  eval_iter = iter(eval_ds)
  eval_iter = map(dataset_utils.tf_to_numpy, eval_iter)
  eval_iter = map(maybe_pad_batches_eval, eval_iter)
  eval_iter = map(shard_batches, eval_iter)

  input_shape = (-1, 32, 32, 3)
  meta_data = {
      'num_classes':
          train_ds_info.features['label'].num_classes,
      'input_shape':
          input_shape,
      'num_train_examples':
          dataset_utils.get_num_examples('svhn_cropped:3.*.*', 'train'),
      'num_eval_examples':
          dataset_utils.get_num_examples('svhn_cropped:3.*.*', 'test'),
      'input_dtype':
          getattr(jnp, dtype_str),
      'target_is_onehot':
          False,
  }
  return dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)
