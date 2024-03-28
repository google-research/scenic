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

"""Data generators for ShapeNet dataset."""

import collections
import dataclasses
import functools
from typing import Optional

from absl import logging
from flax import jax_utils
import jax.numpy as jnp
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import tensorflow as tf

# IOU Utilities
class_choices = [
    'airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife',
    'lamp', 'laptop', 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard',
    'table'
]
seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


@dataclasses.dataclass
class ShapeNetDatasetConfig:
  """Dataset config."""
  dataset_path: str = ''
  num_train_pointclouds: int = 12137
  num_validation_pointclouds: int = 1870
  num_test_pointclouds: int = 2874
  seq_length: int = 2048
  dataset_name: str = 'shapenet'
  version: str = '1.0.0'
  num_classes: int = 50
  num_category_classes: int = 16


def get_dataset_config():
  """Returns dataset config."""
  return ShapeNetDatasetConfig()


def get_class_proportions(dataset, num_class):
  """Computes class proportions in case of label imbalance."""
  labels = []
  for entry in iter(dataset):
    labels.extend(tf.squeeze(entry['label']).numpy())
  freq = collections.Counter(labels)
  proportions = [freq[i] for i in range(num_class)]
  proportions = np.maximum(proportions / np.sum(proportions), 1e-8)
  return proportions


def shapenet_load_split(batch_size,
                        dataset_config,
                        split='train',
                        dtype=tf.float32,
                        prefetch_buffer_size=10,
                        shuffle_seed=None):
  """Creates a split from the ShapeNet dataset using TensorFlow Datasets.

  For the training set, we drop the last partial batch. This is fine to do
  because we additionally shuffle the data randomly each epoch, thus the trainer
  will see all data in expectation. For the validation set, we pad the final
  batch to the desired batch size.

  Args:
    batch_size: int; The batch size returned by the data pipeline.
    dataset_config: dataset configuration.
    split: str; Whether to load the train or evaluation split.
    dtype: TF data type; Data type of the image.
    prefetch_buffer_size: int; Buffer size for the TFDS prefetch.
    shuffle_seed: The seed to use when shuffling the train split.

  Returns:
    A `tf.data.Dataset`.
  """

  is_train = (split == 'train')

  def random_noise(pointcloud, std=0.02):
    assert len(pointcloud.shape) == 2
    noise = tf.random.normal(tf.shape(pointcloud), mean=0, stddev=std/2)
    noisy_pointcloud = pointcloud + tf.clip_by_value(
        noise, clip_value_min=-std, clip_value_max=std)
    return noisy_pointcloud

  def decode_example(example):
    pointcloud = tf.cast(example['pc'], dtype=dtype)
    label = tf.squeeze(example['label'])

    if is_train:
      pointcloud = random_noise(pointcloud)

      # shuffle points
      indices = tf.range(start=0, limit=tf.shape(pointcloud)[0], dtype=tf.int32)
      shuffled_indices = tf.random.shuffle(indices)
      pointcloud = tf.gather(pointcloud, shuffled_indices)
      label = tf.gather(label, shuffled_indices)

    class_label = tf.one_hot(example['class_label'],
                             dataset_config.num_category_classes)
    return {
        'inputs': pointcloud,
        'label': label,
        'class_label': class_label,
        'confusion_matrix_mask': example['confusion_matrix_mask']
    }

  ds = dataset_utils.get_dataset_tfds(
      dataset_config.dataset_name,
      split,
      data_dir=dataset_config.dataset_path,
      skip_decode=('pc',))

  class_proportions = get_class_proportions(ds, dataset_config.num_classes)

  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  ds = ds.cache()

  if is_train:
    ds = ds.repeat()
    ds = ds.shuffle(16 * batch_size, seed=shuffle_seed)

  # decode_example should be applied after caching as it also does augmentation
  ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size, drop_remainder=is_train)

  if not is_train:
    ds = ds.repeat()

  ds = ds.prefetch(prefetch_buffer_size)
  return ds, class_proportions


@datasets.add_dataset('shapenet')
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
                seq_length=2048):
  """Returns generators for the ShapeNet train, validation, and test sets.

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
  shapenet_config = get_dataset_config()
  dataset_configs = dataset_configs or {}
  del rng
  data_augmentations = dataset_configs.get('data_augmentations', ['default'])
  # TODO(dehghani): add mixup data augmentation.
  for da in data_augmentations:
    if da not in ['default']:
      raise ValueError(f'Data augmentation {data_augmentations} is not '
                       f'(yet) supported in the ShapeNet dataset.')
  dtype = getattr(tf, dtype_str)
  onehot_labels = False  # dataset_configs.get('onehot_labels', False)

  logging.info('Loading train split of the ShapeNet dataset.')
  train_ds, _ = shapenet_load_split(
      batch_size,
      shapenet_config,
      split='train',
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

  logging.info('Loading validation split of the ShapeNet dataset.')
  val_ds, _ = shapenet_load_split(
      eval_batch_size,
      shapenet_config,
      split='validation',
      dtype=dtype)

  logging.info('Loading test split of the ShapeNet dataset.')
  test_ds, _ = shapenet_load_split(
      eval_batch_size,
      shapenet_config,
      split='test',
      dtype=dtype)

  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch,
      train=True,
      batch_size=batch_size,
      pixel_level=True)
  maybe_pad_batches_val = functools.partial(
      dataset_utils.maybe_pad_batch,
      train=False,
      batch_size=eval_batch_size,
      pixel_level=True)
  maybe_pad_batches_test = functools.partial(
      dataset_utils.maybe_pad_batch,
      train=False,
      batch_size=eval_batch_size,
      pixel_level=True)
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)

  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(shard_batches, train_iter)
  train_iter = jax_utils.prefetch_to_device(train_iter, prefetch_buffer_size)

  val_iter = iter(val_ds)
  val_iter = map(dataset_utils.tf_to_numpy, val_iter)
  val_iter = map(maybe_pad_batches_val, val_iter)
  val_iter = map(shard_batches, val_iter)
  val_iter = jax_utils.prefetch_to_device(val_iter, prefetch_buffer_size)

  test_iter = iter(test_ds)
  test_iter = map(dataset_utils.tf_to_numpy, test_iter)
  test_iter = map(maybe_pad_batches_test, test_iter)
  test_iter = map(shard_batches, test_iter)
  test_iter = jax_utils.prefetch_to_device(test_iter, prefetch_buffer_size)

  input_shape = (-1, seq_length, 3)
  label_shape = (-1, shapenet_config.num_category_classes)

  meta_data = {
      'num_classes': shapenet_config.num_classes,
      'input_shape': input_shape,
      'label_shape': label_shape,
      'num_train_examples': shapenet_config.num_train_pointclouds,
      'num_eval_examples': shapenet_config.num_validation_pointclouds,
      'num_test_examples': shapenet_config.num_test_pointclouds,
      'input_dtype': getattr(jnp, dtype_str),
      'label_dtype': getattr(jnp, 'int64'),
      'target_is_onehot': onehot_labels,
  }
  return dataset_utils.Dataset(train_iter, val_iter, test_iter, meta_data)

