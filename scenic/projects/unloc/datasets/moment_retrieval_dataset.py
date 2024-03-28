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

"""Registers moment retrieval datasets."""

import functools
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple, Type

from absl import logging
from dmvr import tokenizers as dmvr_tokenizers
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.dataset_lib import video_ops
from scenic.projects.unloc.datasets import dataset_factory
from scenic.projects.unloc.datasets import dataset_utils as dataset_utils_unloc
import tensorflow as tf

PRNGKey = jnp.ndarray


def load_split_from_dmvr(
    dataset_cls: Type[dataset_factory.MomentRetrievalDatasetFactory],
    batch_size: int,
    dataset_config: ml_collections.ConfigDict,
    tokenizers: Mapping[str, dmvr_tokenizers.TextTokenizer],
    subset: str = 'train',
    keep_key: bool = False,
) -> Tuple[tf.data.Dataset, int]:
  """Creates a moment retrieval dataset from file paths.

  Args:
    dataset_cls: Dataset class.
    batch_size: Batch size.
    dataset_config: A dataset config.
    tokenizers: Mapping from tokenizer name to TextTokenizer instances.
    subset: 'train', 'validation' or 'test'.
    keep_key: If true, also return the key for each example.

  Returns:
    ds: A `tf.data.Dataset` object.
    int, Number of examples in the subset.
  """

  ds_factory = dataset_cls(
      base_dir=dataset_config.base_dir,
      tables=dataset_config.tables,
      examples_per_subset=dataset_config.examples_per_subset,
      subset=subset,
      num_groups=jax.process_count(),
      group_index=jax.process_index())
  ds_factory = ds_factory.configure(
      dataset_config, tokenizers, is_training=(subset == 'train'))

  # Only applies to `rgb` modality.
  if (
      subset == 'train'
      and 'rgb' in dataset_config.modality_configs
      and dataset_config.modality_configs['rgb'].get('augmentation_params')
  ):
    ds_factory = video_ops.additional_augmentations(
        ds_factory,
        dataset_config.modality_configs['rgb'].augmentation_params,
        dataset_config.modality_configs['rgb'].crop_size,
        dataset_config.num_frames,
        dataset_config.modality_configs['rgb'].get('zero_centering', True),
        rgb_feature_name='rgb')

  logging.info('Preprocessing graph: %s',
               ds_factory.preprocessor_builder.get_summary())
  logging.info('Postprocessing graph: %s',
               ds_factory.postprocessor_builder.get_summary())

  ds = ds_factory.make_dataset(
      batch_size=batch_size,
      shuffle=(subset == 'train'),
      drop_remainder=(subset == 'train'),
      keep_key=(subset != 'train' and keep_key))

  if subset != 'train':
    ds = ds.repeat(None)

  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  return ds, ds_factory.get_num_examples()


def map_keys(
    batch: Dict[str, Any],
    tokenizers: Mapping[str, dmvr_tokenizers.TextTokenizer],
    config: ml_collections.ConfigDict,
) -> Dict[str, Any]:
  """Changes key names for 'inputs'."""
  batch['inputs'] = {}
  for name, modality_config in config.modality_configs.items():
    if modality_config.type == 'text':
      batch['inputs'][name] = {}
      input_mask = batch[name] != tokenizers[
          modality_config.tokenizer_type].pad_token
      batch['inputs'][name]['input_mask'] = input_mask.astype(np.int32)
      batch['inputs'][name]['input_type_ids'] = np.zeros_like(
          batch[name], dtype=np.int32)
      batch['inputs'][name]['input_word_ids'] = batch.pop(name)
    elif modality_config.type == 'rgb':
      batch['inputs']['rgb'] = batch.pop(name)
    elif modality_config.type == 'flow':
      batch['inputs']['flow'] = batch.pop(name)
    elif modality_config.type == 'embedding':
      batch['inputs'][name] = batch.pop(name)

  batch['inputs']['input_mask'] = batch.pop('input_mask')
  if 'caption_mask' in batch:
    batch['inputs']['caption_mask'] = batch.pop('caption_mask')
  displacement_normalizer = config.get('displacement_normalizer', 'duration')
  if displacement_normalizer == 'duration':
    batch['displacements'] = batch['displacements'] / batch[
        'total_frames'][:, None, None, None]
  elif displacement_normalizer == 'sampled_span':
    batch['displacements'] = batch['displacements'] / (
        config.num_frames * config.get('stride', 1))
  elif displacement_normalizer == 'none':
    batch['displacements'] = batch['displacements'].astype(np.float32)
  return batch


def init_tokenizers(
    modality_configs: ml_collections.ConfigDict,
) -> Dict[str, dmvr_tokenizers.TextTokenizer]:
  """Initializes text tokenizers."""
  tokenizers = {}
  for _, config in modality_configs.items():
    if config.type != 'text':
      continue
    tokenizer_type = config.get('tokenizer_type', 'clip')
    if tokenizer_type not in tokenizers:
      tokenizers[tokenizer_type] = dataset_utils_unloc.init_tokenizer(config)
  return tokenizers


def create_dataset_iterator(
    dataset_configs: ml_collections.ConfigDict,
    subset: str,
    batch_size: int,
    num_shards: int,
    dataset_cls: Type[dataset_factory.MomentRetrievalDatasetFactory],
    tokenizers: Mapping[str, dmvr_tokenizers.TextTokenizer],
    keep_key: bool = False,
) -> Tuple[Iterator[Dict[str, jnp.ndarray]], int]:
  """Creates a moment retrieval dataset iterator."""

  is_training = subset == 'train'
  logging.info('Loading split %s', subset)

  dataset, num_examples = load_split_from_dmvr(
      dataset_cls=dataset_cls,
      batch_size=batch_size,
      dataset_config=dataset_configs,
      tokenizers=tokenizers,
      subset=subset,
      keep_key=keep_key,
  )
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
  current_iter = iter(dataset)
  current_iter = map(dataset_utils.tf_to_numpy, current_iter)
  current_iter = map(
      functools.partial(
          map_keys, tokenizers=tokenizers, config=dataset_configs),
      current_iter)
  current_iter = map(
      functools.partial(
          dataset_utils.maybe_pad_batch,
          train=is_training,
          batch_size=batch_size,
          inputs_key=None), current_iter)
  current_iter = map(shard_batches, current_iter)

  if is_training and dataset_configs.get('prefetch_to_device'):
    # Async bind batch to device which speeds up training.
    current_iter = jax_utils.prefetch_to_device(
        current_iter, dataset_configs.get('prefetch_to_device'))

  return current_iter, num_examples


@datasets.add_dataset('moment_retrieval_dataset')
def get_dataset(*,
                batch_size: int,
                eval_batch_size: int,
                num_shards: int,
                dtype_str: str = 'float32',
                shuffle_seed: int = 0,
                rng: Optional[PRNGKey] = None,
                dataset_configs: Optional[ml_collections.ConfigDict] = None,
                dataset_service_address: Optional[str] = None):
  """Returns a generator for the moment retrieval dataset."""
  del rng, shuffle_seed, dataset_service_address
  if dataset_configs is None:
    raise ValueError(
        'dataset_configs must be set for moment retrieval dataset.')
  if dataset_configs.get('base_dir') is None:
    raise ValueError('base_dir must be specified for moment retrieval dataset')
  if not dataset_configs.get('tables'):
    raise ValueError(
        'tables mapping must be specified for moment retrieval dataset')

  tokenizers = init_tokenizers(dataset_configs.modality_configs)

  train_iter, num_train_examples = create_dataset_iterator(
      dataset_configs,
      'train',
      batch_size,
      num_shards,
      dataset_cls=dataset_factory.MomentRetrievalDatasetFactory,
      tokenizers=tokenizers,
  )
  eval_iter, num_eval_examples = create_dataset_iterator(
      dataset_configs,
      'validation',
      eval_batch_size,
      num_shards,
      dataset_cls=dataset_factory.MomentRetrievalDatasetFactory,
      tokenizers=tokenizers,
  )
  test_batch_size = dataset_configs.get('test_batch_size', eval_batch_size)
  test_iter, num_test_examples = create_dataset_iterator(
      dataset_configs,
      'test',
      test_batch_size,
      num_shards,
      dataset_cls=dataset_factory.MomentRetrievalDatasetFactory,
      tokenizers=tokenizers,
  )
  feature_pyramid_levels = dataset_configs.get(
      'feature_pyramid_config.feature_pyramid_levels', None)
  feature_pyramid_downsample_stride = dataset_configs.get(
      'feature_pyramid_config.feature_pyramid_downsample_stride', 2)
  if feature_pyramid_levels is None:
    total_frames = dataset_configs.num_frames
  else:
    total_frames = sum([
        dataset_configs.num_frames // (feature_pyramid_downsample_stride**idx)
        for idx in range(len(feature_pyramid_levels))
    ])
  meta_data = {
      'num_classes': 1,
      'num_train_examples': num_train_examples,
      'num_eval_examples': num_eval_examples,
      'num_test_examples': num_test_examples,
      'input_shape': {
          'input_mask': (-1, total_frames),
          'caption_mask': (-1, dataset_configs.train_max_num_captions),
      },
      'input_dtype': {'input_mask': jnp.int32, 'caption_mask': jnp.int32},
      'target_is_onehot': True,
  }
  for modality_name, modality_config in dataset_configs.modality_configs.items(
  ):
    if modality_config.type == 'rgb':
      meta_data['input_shape']['rgb'] = (-1, dataset_configs.num_frames,
                                         modality_config.crop_size,
                                         modality_config.crop_size, 3)
      meta_data['input_dtype']['rgb'] = getattr(jnp, dtype_str)
    if modality_config.type == 'flow':
      meta_data['input_shape']['flow'] = (
          -1,
          dataset_configs.num_frames,
          modality_config.crop_size,
          modality_config.crop_size,
          2,
      )
      meta_data['input_dtype']['flow'] = getattr(jnp, dtype_str)
    elif modality_config.type == 'text':
      meta_data['input_shape'][modality_name] = {
          'input_word_ids': (
              -1,
              dataset_configs.train_max_num_captions,
              modality_config.max_num_tokens,
          ),
          'input_type_ids': (
              -1,
              dataset_configs.train_max_num_captions,
              modality_config.max_num_tokens,
          ),
          'input_mask': (
              -1,
              dataset_configs.train_max_num_captions,
              modality_config.max_num_tokens,
          ),
      }
      meta_data['input_dtype'][modality_name] = {
          'input_word_ids': jnp.int32,
          'input_type_ids': jnp.int32,
          'input_mask': jnp.int32,
      }
    elif modality_config.type == 'embedding':
      meta_data['input_shape'][modality_name] = (
          -1, dataset_configs.num_frames, modality_config.feature_dimension)
      meta_data['input_dtype'][modality_name] = getattr(jnp, dtype_str)

  logging.info('Dataset metadata:\n%s', meta_data)

  return dataset_utils.Dataset(train_iter, eval_iter, test_iter, meta_data)
