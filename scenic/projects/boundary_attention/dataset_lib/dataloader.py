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

"""Dataset loader."""
import functools
from typing import Optional

from absl import logging
import jax
import jax.numpy as jnp
import ml_collections
from scenic.projects.boundary_attention.dataset_lib.datasets import dataset_utils
from scenic.projects.boundary_attention.dataset_lib.datasets import kaleidoshapes_dataset
from scenic.projects.boundary_attention.dataset_lib.datasets import pickled_dataset


_ALL_DATASETS = {
    'circle_triangles': pickled_dataset,
    'kaleidoshapes': kaleidoshapes_dataset,
}


def get_dataset_by_name(name):
  try:
    return functools.partial(
        dataset_utils.get_dataset, dataset_module=_ALL_DATASETS[name]
    )
  except:
    raise NotImplementedError('Cannot find dataset: %s, %s' % (name)) from None


def get_dataloader(config: ml_collections.ConfigDict,
                   data_rng: jnp.ndarray,
                   *,
                   dataset_service_address: Optional[str] = None,
                   dataset_shuffle=0):
  """Given a config, returns a dataset dataloader."""
  del data_rng

  dataset_config = config.dataset

  device_count = jax.device_count()
  logging.info('device_count: %d', device_count)
  logging.info('jax.local_device_count(): %d', jax.local_device_count())
  logging.info('num_hosts : %d', jax.process_count())
  logging.info('host_id : %d', jax.process_index())
  logging.info('dataset_shuffle: %d', dataset_shuffle)

  dataset_builder = get_dataset_by_name(name=dataset_config.name)

  batch_size = config.batch_size * device_count
  eval_batch_size = config.eval_batch_size * device_count

  local_batch_size = batch_size // jax.process_count()
  eval_local_batch_size = eval_batch_size // jax.process_count()
  device_batch_size = batch_size // device_count
  logging.info('local_batch_size : %d', local_batch_size)
  logging.info('device_batch_size : %d', device_batch_size)

  shuffle_seed = config.get('shuffle_seed', None)
  if dataset_service_address and shuffle_seed is not None:
    raise ValueError('Using dataset service with a random seed causes each '
                     'worker to produce exactly the same data. Add '
                     'config.shuffle_seed = None to your config if you want '
                     'to run with dataset service.')

  dataset = dataset_builder(
      dataset_config=dataset_config,
      batch_size=local_batch_size,
      eval_batch_size=eval_local_batch_size,
      num_shards=jax.local_device_count(),
      shuffle_seed=shuffle_seed,
      dataset_shuffle=dataset_shuffle,
      dataset_service_address=dataset_service_address,
  )

  return dataset
