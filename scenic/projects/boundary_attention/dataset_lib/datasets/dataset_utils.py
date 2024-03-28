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

"""Dataset library."""

import functools

from absl import logging
from flax import jax_utils
from scenic.dataset_lib import dataset_utils as scenic_dataset_utils
import tensorflow_datasets as tfds


def get_dataset(
    *,
    dataset_module,
    dataset_config,
    batch_size=1,
    eval_batch_size=1,
    num_shards=1,
    shuffle_buffer_size=10,
    shuffle_seed=0,
    dataset_shuffle=0,
    prefetch_buffer_size=2,
    dataset_service_address=None,
    ):
  """Returns a dataset.

  Args:
    dataset_module: A dataset module.
    dataset_config: A ml_collections dataset config.
    batch_size: Training batch size.
    eval_batch_size: Evaluation batch size.
    num_shards: Number of shards (devices).
    shuffle_buffer_size: Size of shuffle buffer.
    shuffle_seed: Seed for shuffling.
    dataset_shuffle: Whether to shuffle the dataset.
    prefetch_buffer_size: Size of the buffer for dataset preprocessing.
    dataset_service_address: Address of the dataset service.

  Returns:
    A dataset object containing train_iter, val_iter, and metadata.
  """

  shard_batches = functools.partial(scenic_dataset_utils.shard,
                                    n_devices=num_shards)

  # train setup
  train_dataset, eval_dataset, metadata = dataset_module.get_dataset_and_count(
      dataset_config=dataset_config,
      batch_size=batch_size,
      eval_batch_size=eval_batch_size,
      shuffle_buffer_size=shuffle_buffer_size,
      dataset_shuffle=dataset_shuffle)

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you '
                       'want to run with dataset service.')
    logging.info('Using the tf.data service at %s', dataset_service_address)
    train_dataset = scenic_dataset_utils.distribute(train_dataset,
                                                    dataset_service_address)

  train_dataset = tfds.as_numpy(train_dataset)
  eval_dataset = tfds.as_numpy(eval_dataset)

  train_iter = iter(train_dataset)
  train_iter = map(shard_batches, train_iter)
  train_iter = jax_utils.prefetch_to_device(train_iter, prefetch_buffer_size)

  eval_iter = iter(eval_dataset)
  eval_iter = map(shard_batches, eval_iter)
  eval_iter = jax_utils.prefetch_to_device(eval_iter, prefetch_buffer_size)

  return scenic_dataset_utils.Dataset(train_iter, eval_iter, None, metadata)
