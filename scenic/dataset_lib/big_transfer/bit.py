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

"""Scenic wrapper around BigTransfer dataset loaders."""

import functools
from typing import Optional
from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.dataset_lib.big_transfer import builder
# Unused imports are to register preprocessing ops:
from scenic.dataset_lib.big_transfer.preprocessing import vtab_ops  # pylint: disable=unused-import


def tf_to_numpy(batch):
  """Convert a input batch from tf Tensors to numpy arrays.

  Args:
    batch: dict; A dictionary tha has items in a batch: image and labels.

  Returns:
    Numpy arrays of the given tf Tensors.
  """
  # Transforms x into read-only numpy array without copy if possible, see:
  # https://github.com/tensorflow/tensorflow/issues/33254#issuecomment-542379165
  convert_data = lambda x: np.asarray(memoryview(x))
  return jax.tree_util.tree_map(convert_data, batch)


@datasets.add_dataset('bit')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None,
                devices: Optional[np.ndarray] = None):
  """Returns generators for train and validation sets for a specified dataset.

  Args:
    batch_size: int; Determines the train batch size.
    eval_batch_size: int; Determines the evaluation batch size.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image (e.g. 'float32').
    shuffle_seed: int; Seed for shuffling the training data. Not used.
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    dataset_configs: dict; Dataset specific configurations.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address.
    devices: Numpy array of Jax devices with mesh_shape which is used for
      sharding the data. Optional, and required for jit-based pipelines. Should
      not be used for pmap-based data parallelism.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """
  del rng
  assert dataset_configs is not None
  logging.info('Loading train split of the %s'
               'from bit dataset.', dataset_configs.dataset)
  target_is_onehot = 'onehot' in dataset_configs.pp_train

  def pp_fn(x, how):
    pp = builder.get_preprocess_fn(how)
    example = pp(x)
    # to scenic format
    return {'inputs': example['image'], 'label': example['labels']}

  # E.g. for testing with TAP.
  shuffle_buffer_size = (1000 if num_shards == 1 else
                         dataset_configs.shuffle_buffer_size)

  # Whether to cache training data. None: no caching. 'loaded':
  # cache right after loading a datapoint. 'batched': cache whole batches.
  cache = dataset_configs.get('cache', 'loaded')
  skip_decode = dataset_configs.get('skip_decode', ('image',))
  if isinstance(skip_decode, ml_collections.ConfigDict):
    skip_decode = skip_decode.to_dict()
  train_ds = dataset_utils.get_data(
      dataset=dataset_configs.dataset,
      split=dataset_configs.train_split,
      data_dir=dataset_configs.get('dataset_dir'),
      batch_size=batch_size,
      preprocess_fn=functools.partial(pp_fn, how=dataset_configs.pp_train),
      shuffle_buffer_size=shuffle_buffer_size,
      prefetch=dataset_configs.get('prefetch_to_host', 2),
      cache=cache,
      ignore_errors=True,
      skip_decode=skip_decode)

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you '
                       'want to run with dataset service.')
    logging.info('Using the tf.data service at %s', dataset_service_address)
    assert shuffle_buffer_size is not None
    train_ds = dataset_utils.distribute(train_ds, dataset_service_address)

  n_train_ex = dataset_utils.get_num_examples(
      dataset_configs.dataset,
      dataset_configs.train_split,
      data_dir=dataset_configs.get('dataset_dir'))

  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size)
  if devices is None:
    shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
    prefetch_fn = jax_utils.prefetch_to_device
  else:
    shard_batches = functools.partial(dataset_utils.shard_jit,
                                      global_devices=devices)
    prefetch_fn = dataset_utils.prefetch_iterator

  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(shard_batches, train_iter)
  if dataset_configs.get('prefetch_to_device'):
    train_iter = prefetch_fn(train_iter, dataset_configs.prefetch_to_device)

  logging.info('Loading validation split of the %s'
               'from bit dataset.', dataset_configs.dataset)
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch, train=False, batch_size=eval_batch_size)

  def _get_eval_iter(dataset, split, pp_eval, dataset_dir):
    val_ds = dataset_utils.get_data(
        dataset=dataset,
        split=split,
        data_dir=dataset_dir,
        batch_size=eval_batch_size,
        preprocess_fn=functools.partial(pp_fn, how=pp_eval),
        cache='batched',
        repeat_after_batching=True,
        drop_remainder=False,
        skip_decode=skip_decode)

    valid_iter = iter(val_ds)
    valid_iter = map(tf_to_numpy, valid_iter)
    valid_iter = map(maybe_pad_batches_eval, valid_iter)
    valid_iter = map(shard_batches, valid_iter)
    if dataset_configs.prefetch_to_device:
      valid_iter = prefetch_fn(valid_iter, dataset_configs.prefetch_to_device)

    return valid_iter

  def _get_num_eval_examples(dataset, split, data_dir):
    return dataset_utils.get_num_examples(dataset, split, data_dir)

  if isinstance(dataset_configs.val_split, str):
    valid_iter = _get_eval_iter(dataset_configs.dataset,
                                dataset_configs.val_split,
                                dataset_configs.pp_eval,
                                dataset_configs.get('dataset_dir'))
    n_eval_ex = _get_num_eval_examples(
        dataset_configs.dataset,
        dataset_configs.val_split,
        data_dir=dataset_configs.get('dataset_dir'))
  else:
    valid_iter, n_eval_ex = {}, {}
    for eval_spec in dataset_configs.val_split:
      if len(eval_spec) == 4:
        name, dataset, split, pp_eval = eval_spec
        dataset_dir = dataset_configs.get('dataset_dir')
      elif len(eval_spec) == 5:
        name, dataset, split, pp_eval, dataset_dir = eval_spec
      else:
        raise ValueError(f'Unknown eval_spec {eval_spec}')
      valid_iter[name] = _get_eval_iter(dataset, split, pp_eval, dataset_dir)
      n_eval_ex[name] = _get_num_eval_examples(
          dataset, split, data_dir=dataset_dir)

  input_shape = (-1,) + tuple(train_ds.element_spec['inputs'].shape[1:])

  num_classes = dataset_configs.get('num_classes')
  if num_classes is None:
    logging.warning('For the BiT datasets, if the task is classification, '
                    '`num_classes` should be specified in the config.')

  meta_data = {
      'num_classes': num_classes,
      'input_shape': input_shape,
      'num_train_examples': n_train_ex,
      'num_eval_examples': n_eval_ex,
      'input_dtype': getattr(jnp, dtype_str),
      'target_is_onehot': target_is_onehot,
  }
  if dataset_configs.get('extra_meta_data'):
    for k, v in dataset_configs.extra_meta_data.items():
      meta_data[k] = v
  return dataset_utils.Dataset(train_iter, valid_iter, None, meta_data)
