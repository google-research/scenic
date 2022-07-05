# Copyright 2022 The Scenic Authors.
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

"""Data generators for the BAIR Robot dataset."""

import functools
from typing import Optional

from absl import logging
from dmvr import processors
from flax import jax_utils
import jax.numpy as jnp
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import tensorflow as tf


def preprocess_example(example,
                       dtype=tf.float32,
                       num_frames=30,
                       stride=1,
                       zero_centering=True,
                       eval_beginning_frames=False,
                       train=True):
  """Preprocesses the given image.

  Args:
    example: dict; Example that has an 'image' and a 'label'.
    dtype: Tensorflow data type; Data type of the image.
    num_frames: Number of frames per subclip.
    stride: Temporal stride to sample frames.
    zero_centering: If True, frames are normalized to values in [-1, 1].
      If False, values in [0, 1].
    eval_beginning_frames: If True, returns the first clip instead of the center
       clip for evaluation.
    train: Whether in training mode.

  Returns:
    A preprocessed image `Tensor`.
  """
  frames = example['image_main']
  if train:
    frames = processors.sample_sequence(frames, num_frames, True, stride)
    frames = processors.random_flip_left_right(frames)
  else:
    if eval_beginning_frames:
      frames = frames[:num_frames * stride:stride]
    else:
      frames = processors.sample_sequence(frames, num_frames, False, stride)
  frames = processors.normalize_image(frames, zero_centering, dtype)
  return {'inputs': frames}


@datasets.add_dataset('bair')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None):
  """Returns generators for the BAIR train, validation, and test set.

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
  dtype = getattr(tf, dtype_str)
  dataset_configs = dataset_configs or {}
  num_frames = dataset_configs.get('num_frames', 30)
  stride = dataset_configs.get('stride', 1)
  zero_centering = dataset_configs.get('zero_centering', True)
  eval_beginning_frames = dataset_configs.get('eval_beginning_frames', False)
  preprocess_ex_train = functools.partial(
      preprocess_example,
      dtype=dtype,
      num_frames=num_frames,
      stride=stride,
      zero_centering=zero_centering,
      train=True)
  preprocess_ex_eval = functools.partial(
      preprocess_example,
      dtype=dtype,
      num_frames=num_frames,
      stride=stride,
      zero_centering=zero_centering,
      eval_beginning_frames=eval_beginning_frames,
      train=False)

  logging.info('Loading train split of the BAIR dataset.')
  train_ds, _ = dataset_utils.load_split_from_tfds(
      'bair_robot_pushing_small',
      batch_size,
      split='train',
      preprocess_example=preprocess_ex_train,
      shuffle_seed=shuffle_seed)

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you '
                       'want to run with dataset service.')
    logging.info('Using the tf.data service at %s', dataset_service_address)
    train_ds = dataset_utils.distribute(train_ds, dataset_service_address)

  logging.info('Loading test split of the MNIST dataset.')
  eval_ds, _ = dataset_utils.load_split_from_tfds(
      'bair_robot_pushing_small',
      eval_batch_size,
      split='test',
      preprocess_example=preprocess_ex_eval)

  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size)
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch, train=False, batch_size=eval_batch_size)
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)

  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(shard_batches, train_iter)
  if dataset_configs.get('prefetch_to_device'):
    # Async bind batch to device which speeds up training.
    train_iter = jax_utils.prefetch_to_device(
        train_iter, dataset_configs.get('prefetch_to_device'))

  eval_iter = iter(eval_ds)
  eval_iter = map(dataset_utils.tf_to_numpy, eval_iter)
  eval_iter = map(maybe_pad_batches_eval, eval_iter)
  eval_iter = map(shard_batches, eval_iter)

  input_shape = (-1, num_frames, 64, 64, 3)
  meta_data = {
      'num_classes':
          None,
      'input_shape':
          input_shape,
      'num_train_examples':
          dataset_utils.get_num_examples('bair_robot_pushing_small', 'train'),
      'num_eval_examples':
          dataset_utils.get_num_examples('bair_robot_pushing_small', 'test'),
      'input_dtype':
          getattr(jnp, dtype_str),
      'target_is_onehot':
          False,
  }
  return dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)
