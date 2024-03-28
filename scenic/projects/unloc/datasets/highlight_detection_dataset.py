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

"""Registers highlight detection datasets."""

from typing import Optional

from absl import logging
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.projects.unloc.datasets import dataset_factory
from scenic.projects.unloc.datasets import moment_retrieval_dataset

PRNGKey = jnp.ndarray


@datasets.add_dataset('highlight_detection_dataset')
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

  tokenizers = moment_retrieval_dataset.init_tokenizers(
      dataset_configs.modality_configs
  )

  train_iter, num_train_examples = (
      moment_retrieval_dataset.create_dataset_iterator(
          dataset_configs,
          'train',
          batch_size,
          num_shards,
          dataset_cls=dataset_factory.HighlightDetectionDatasetFactory,
          tokenizers=tokenizers,
      )
  )
  eval_iter, num_eval_examples = (
      moment_retrieval_dataset.create_dataset_iterator(
          dataset_configs,
          'validation',
          eval_batch_size,
          num_shards,
          dataset_cls=dataset_factory.HighlightDetectionDatasetFactory,
          tokenizers=tokenizers,
      )
  )
  test_batch_size = dataset_configs.get('test_batch_size', eval_batch_size)
  test_iter, num_test_examples = (
      moment_retrieval_dataset.create_dataset_iterator(
          dataset_configs,
          'test',
          test_batch_size,
          num_shards,
          dataset_cls=dataset_factory.HighlightDetectionDatasetFactory,
          tokenizers=tokenizers,
      )
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
      },
      'input_dtype': {
          'input_mask': jnp.int32,
          'caption_mask': jnp.int32
      },
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
          'input_word_ids': (-1, modality_config.max_num_tokens),
          'input_type_ids': (-1, modality_config.max_num_tokens),
          'input_mask': (-1, modality_config.max_num_tokens),
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
