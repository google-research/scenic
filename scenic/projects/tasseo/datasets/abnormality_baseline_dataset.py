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

"""Dataset for abnormality tasks with baseline-formatted examples."""

import functools
from typing import Optional

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.projects.tasseo import dataset_utils as ts_dataset_utils


NUM_TRAIN_EXAMPLES = {
    'del5_simple': 7145,
    'del5_net': 7475,
    't922_chrm22': 4236,
    't922_chrm9': 4057,
}
NUM_VALIDATION_EXAMPLES = {
    'del5_simple': 887,
    'del5_net': 932,
    't922_chrm22': 484,
    't922_chrm9': 460,
}
NUM_TEST_EXAMPLES = {
    'del5_simple': 828,
    'del5_net': 854,
    't922_chrm22': 0,
    't922_chrm9': 0,
}
NUM_CLASSES = 2
CLASS_NAMES = ('normal', 'abnormal')
NUM_CHANNELS = 1


@datasets.add_dataset('abnormality_baseline')
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
    # Add path to your training data here:
    path = ''

  def build_baseline_dataset(split='train', shuffle=True):
    """dataset_fn called by data.build_dataset(**kwargs)."""
    parallel_reads = 4 if shuffle else 1

    if split == 'train':
      dataset_prefix = dataset_prefixes['train'] % (
          dataset_configs.num_abnormal, dataset_configs.replica)
    else:
      dataset_prefix = dataset_prefixes[split]
    ds = ts_dataset_utils.load_data(
        dataset_prefix,
        is_train=(split == 'train'),
        parallel_reads=parallel_reads)
    ds = ds.map(
        lambda x: ts_dataset_utils.preprocess(  # pylint:disable=g-long-lambda
            x, 'label', dataset_configs.chrm_image_shape,
            class_names=CLASS_NAMES))
    return ds

  # use different seed for each host
  if shuffle_seed is None:
    local_seed = None
  else:
    data_seed = 0
    local_seed = data_seed + jax.process_index()

  train_dataset = ts_dataset_utils.build_dataset(
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

  eval_dataset = ts_dataset_utils.build_dataset(
      dataset_fn=build_baseline_dataset,
      split='valid',
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

  input_shape = [-1] + list(dataset_configs.chrm_image_shape) + [NUM_CHANNELS]
  meta_data = {
      'num_classes':
          NUM_CLASSES,
      'input_shape':
          input_shape,
      'num_train_examples':
          int(NUM_TRAIN_EXAMPLES[dataset_configs.abnormality]),
      'num_eval_examples':
          NUM_VALIDATION_EXAMPLES[dataset_configs.abnormality],
      'input_dtype':
          getattr(jnp, dtype_str),
      'target_is_onehot':
          True,
  }

  return dataset_utils.Dataset(train_iter, valid_iter, None, meta_data)

