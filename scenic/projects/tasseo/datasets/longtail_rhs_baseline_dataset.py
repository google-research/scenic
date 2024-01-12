# Copyright 2023 The Scenic Authors.
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

"""Dataset for long tail rhs aberrations with baseline-formatted examples."""

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
    't_11_19': {
        0: {'num_test': 335, 'num_train': 3216},
        1: {'num_test': 347, 'num_train': 3204},
        2: {'num_test': 363, 'num_train': 3188},
        3: {'num_test': 331, 'num_train': 3220},
        4: {'num_test': 361, 'num_train': 3190},
        5: {'num_test': 390, 'num_train': 3161},
        6: {'num_test': 338, 'num_train': 3213},
        7: {'num_test': 345, 'num_train': 3206},
        8: {'num_test': 368, 'num_train': 3183},
        9: {'num_test': 373, 'num_train': 3178}},
    't_9_11': {
        0: {'num_test': 344, 'num_train': 3283},
        1: {'num_test': 356, 'num_train': 3271},
        2: {'num_test': 323, 'num_train': 3304},
        3: {'num_test': 388, 'num_train': 3239},
        4: {'num_test': 350, 'num_train': 3277},
        5: {'num_test': 345, 'num_train': 3282},
        6: {'num_test': 421, 'num_train': 3206},
        7: {'num_test': 331, 'num_train': 3296},
        8: {'num_test': 388, 'num_train': 3239},
        9: {'num_test': 381, 'num_train': 3246}
    }
}


@datasets.add_dataset('longtail_rhs_baseline')
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

