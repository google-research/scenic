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

"""Dataset for long tail aberrations with baseline-formatted examples."""

import functools
from typing import Optional

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.projects.tasseo import dataset_utils as ts_dataset_utils
import tensorflow as tf


NUM_CLASSES = 2
CLASS_NAMES = ('normal', 'abnormal')
NUM_CHANNELS = 1
NUM_FOLDS = 10

DATASET_KFOLD_PATTERN = (
    '/long_tail_datasets_plus_all_healthy/%(pattern_pathname)s'
    '/%(pattern_pathname)s_kfold-%(fold_num)02d-of-%(num_folds)02d')


FOLD_METADATA = {
    'inv_16': {
        0: {'num_test': 363, 'num_train': 3105},
        1: {'num_test': 361, 'num_train': 3107},
        2: {'num_test': 336, 'num_train': 3132},
        3: {'num_test': 328, 'num_train': 3140},
        4: {'num_test': 343, 'num_train': 3125},
        5: {'num_test': 330, 'num_train': 3138},
        6: {'num_test': 341, 'num_train': 3127},
        7: {'num_test': 369, 'num_train': 3099},
        8: {'num_test': 370, 'num_train': 3098},
        9: {'num_test': 327, 'num_train': 3141},
    },
    'inv_3_q21q2': {
        0: {'num_test': 364, 'num_train': 3128},
        1: {'num_test': 346, 'num_train': 3146},
        2: {'num_test': 322, 'num_train': 3170},
        3: {'num_test': 334, 'num_train': 3158},
        4: {'num_test': 349, 'num_train': 3143},
        5: {'num_test': 320, 'num_train': 3172},
        6: {'num_test': 329, 'num_train': 3163},
        7: {'num_test': 389, 'num_train': 3103},
        8: {'num_test': 354, 'num_train': 3138},
        9: {'num_test': 385, 'num_train': 3107},
    },
    't_11_19': {
        0: {'num_test': 352, 'num_train': 3192},
        1: {'num_test': 375, 'num_train': 3169},
        2: {'num_test': 323, 'num_train': 3221},
        3: {'num_test': 380, 'num_train': 3164},
        4: {'num_test': 330, 'num_train': 3214},
        5: {'num_test': 328, 'num_train': 3216},
        6: {'num_test': 393, 'num_train': 3151},
        7: {'num_test': 324, 'num_train': 3220},
        8: {'num_test': 346, 'num_train': 3198},
        9: {'num_test': 393, 'num_train': 3151},
    },
    't_9_11': {
        0: {'num_test': 349, 'num_train': 3203},
        1: {'num_test': 344, 'num_train': 3208},
        2: {'num_test': 324, 'num_train': 3228},
        3: {'num_test': 356, 'num_train': 3196},
        4: {'num_test': 392, 'num_train': 3160},
        5: {'num_test': 353, 'num_train': 3199},
        6: {'num_test': 360, 'num_train': 3192},
        7: {'num_test': 361, 'num_train': 3191},
        8: {'num_test': 362, 'num_train': 3190},
        9: {'num_test': 351, 'num_train': 3201},
    },
}


@datasets.add_dataset('longtail_baseline')
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
  pattern_pathname = dataset_configs.pattern_pathname
  def get_fold_filepath_pattern(fold_num):
    return DATASET_KFOLD_PATTERN % {
        'pattern_pathname': pattern_pathname,
        'fold_num': fold_num,
        'num_folds': NUM_FOLDS,
    }

  fold_filepath_patterns = [
      get_fold_filepath_pattern(i) for i in range(NUM_FOLDS)
  ]
  eval_fold_nums = [dataset_configs.test_fold_num]
  train_fold_nums = [i for i in range(NUM_FOLDS) if i not in eval_fold_nums]
  kfold_dataset_prefixes = {
      'train': [fold_filepath_patterns[i] for i in train_fold_nums],
      'eval': [fold_filepath_patterns[i] for i in eval_fold_nums],
  }

  # For debugging and validation.
  for fold_num, filepath_prefix in enumerate(fold_filepath_patterns):
    fold_filepaths = tf.io.matching_files(filepath_prefix + '*')
    logging.info('Found %d files matching "%s" for fold %d',
                 len(fold_filepaths), filepath_prefix, fold_num)
  logging.info('train folds: %r', train_fold_nums)
  logging.info('eval folds: %r', eval_fold_nums)

  def build_baseline_dataset_kfold(split='train', shuffle=True):
    """dataset_fn called by data.build_dataset(**kwargs)."""
    parallel_reads = 4 if shuffle else 1

    dataset_prefixes = None
    if split == 'train':
      dataset_prefixes = kfold_dataset_prefixes['train']
    else:
      dataset_prefixes = kfold_dataset_prefixes['eval']

    ds = None
    for dataset_prefix in dataset_prefixes:
      fold_ds = ts_dataset_utils.load_data(
          dataset_prefix,
          is_train=(split == 'train'),
          parallel_reads=parallel_reads)
      if ds is None:
        ds = fold_ds
      else:
        ds = ds.concatenate(fold_ds)
    if ds is None:  # Avoids type exception due too possible None.map.
      raise ValueError('No folds found for %s dataset matching prefixes: %r'
                       % (split, dataset_prefixes))
    # pylint: disable=g-long-lambda
    ds = ds.map(
        lambda x: ts_dataset_utils.preprocess(
            x, 'label', dataset_configs.chrm_image_shape,
            class_names=CLASS_NAMES))
    # pylint: enable=g-long-lambda
    return ds

  # use different seed for each host
  if shuffle_seed is None:
    local_seed = None
  else:
    data_seed = 0
    local_seed = data_seed + jax.process_index()

  train_dataset = ts_dataset_utils.build_dataset(
      dataset_fn=build_baseline_dataset_kfold,
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

  eval_dataset = ts_dataset_utils.build_dataset(
      dataset_fn=build_baseline_dataset_kfold,
      split='valid',
      batch_size=eval_batch_size,
      strategy=None)

  test_dataset = ts_dataset_utils.build_dataset(
      dataset_fn=build_baseline_dataset_kfold,
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
  fold_metadata = (
      FOLD_METADATA[dataset_configs.pattern_pathname][
          dataset_configs.test_fold_num])
  num_train_examples = fold_metadata['num_train']
  num_test_examples = fold_metadata['num_test']
  meta_data = {
      'num_classes':
          NUM_CLASSES,
      'input_shape':
          input_shape,
      'num_train_examples':
          num_train_examples,
      'num_eval_examples':
          num_test_examples,
      'input_dtype':
          getattr(jnp, dtype_str),
      'target_is_onehot':
          True,
  }

  return dataset_utils.Dataset(train_iter, valid_iter, test_iter, meta_data)

