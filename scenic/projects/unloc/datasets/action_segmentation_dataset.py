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

"""Registers action segmentation datasets."""

from typing import Any, Dict, Optional

from absl import logging
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.projects.unloc.datasets import dataset_factory
from scenic.projects.unloc.datasets import dataset_utils as unloc_dataset_utils
from scenic.projects.unloc.datasets import temporal_localization_dataset


PRNGKey = jnp.ndarray


def map_keys(batch: Dict[str, Any],
             config: ml_collections.ConfigDict) -> Dict[str, Any]:
  """Changes key names for 'inputs'."""
  batch['inputs'] = {}
  for modality in config.modality_configs.keys():
    batch['inputs'][modality] = batch.pop(modality)
  batch['inputs']['input_mask'] = batch.pop('input_mask')
  return batch


@datasets.add_dataset('action_segmentation_dataset')
def get_dataset(*,
                batch_size: int,
                eval_batch_size: int,
                num_shards: int,
                dtype_str: str = 'float32',
                shuffle_seed: int = 0,
                rng: Optional[PRNGKey] = None,
                dataset_configs: Optional[ml_collections.ConfigDict] = None,
                dataset_service_address: Optional[str] = None):
  """Returns a generator for the action segmentation dataset."""
  del rng, shuffle_seed, dataset_service_address
  if dataset_configs is None:
    raise ValueError(
        'dataset_configs must be set for action segmentation dataset.')
  if dataset_configs.get('base_dir') is None:
    raise ValueError(
        'base_dir must be specified for action segmentation dataset')
  if not dataset_configs.get('tables'):
    raise ValueError(
        'tables mapping must be specified for action segmentation dataset')

  class_name_ids = None
  tokenizer = None
  class_name_embeddings = None
  num_prompts = 1
  if dataset_configs.get('class_name_csv') is not None:
    class_names = unloc_dataset_utils.read_strings_from_csv(
        dataset_configs.class_name_csv)
    if len(class_names) != dataset_configs.num_classes:
      raise ValueError(
          'Number of class names does not match "dataset_configs.num_classes".')
    tokenizer = unloc_dataset_utils.init_tokenizer(
        dataset_configs.tokenizer_config)
    if dataset_configs.get('prompt_csv') is not None:
      prompts = unloc_dataset_utils.read_strings_from_csv(
          dataset_configs.prompt_csv)
      num_prompts = len(prompts)
      augmented_class_names = []
      for name in class_names:
        augmented_class_names.extend(prompt.format(name) for prompt in prompts)
    else:
      augmented_class_names = class_names
    class_name_ids = unloc_dataset_utils.tokenize_class_names(
        tokenizer, dataset_configs.tokenizer_config, augmented_class_names)

  if dataset_configs.get('class_name_embedding_npy') is not None:
    class_name_embeddings = unloc_dataset_utils.read_string_embeddings(
        dataset_configs.class_name_embedding_npy)
    num_prompts = class_name_embeddings.shape[1]
    assert class_name_embeddings.shape[0] == dataset_configs.num_classes

  (train_iter, num_train_examples) = (
      temporal_localization_dataset.create_dataset_iterator(
          dataset_configs,
          'train',
          batch_size,
          num_shards,
          dataset_factory.ActionSegmentationDatasetFactory,
          map_key_fn=map_keys,
          class_name_ids=class_name_ids,
          tokenizer=tokenizer,
          num_prompts=num_prompts,
          class_name_embeddings=class_name_embeddings,
      )
  )
  (eval_iter, num_eval_examples) = (
      temporal_localization_dataset.create_dataset_iterator(
          dataset_configs,
          'validation',
          eval_batch_size,
          num_shards,
          dataset_factory.ActionSegmentationDatasetFactory,
          map_key_fn=map_keys,
          class_name_ids=class_name_ids,
          tokenizer=tokenizer,
          num_prompts=num_prompts,
          class_name_embeddings=class_name_embeddings,
      )
  )
  test_batch_size = dataset_configs.get('test_batch_size', eval_batch_size)
  (test_iter, num_test_examples) = (
      temporal_localization_dataset.create_dataset_iterator(
          dataset_configs,
          'test',
          test_batch_size,
          num_shards,
          dataset_factory.ActionSegmentationDatasetFactory,
          map_key_fn=map_keys,
          class_name_ids=class_name_ids,
          tokenizer=tokenizer,
          num_prompts=num_prompts,
          class_name_embeddings=class_name_embeddings,
      )
  )
  meta_data = {
      'num_classes': dataset_configs.num_classes,
      'num_train_examples': num_train_examples,
      'num_eval_examples': num_eval_examples,
      'num_test_examples': num_test_examples,
      'input_shape': {
          'input_mask': (-1, dataset_configs.num_frames)
      },
      'input_dtype': {
          'input_mask': jnp.int32
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
    if modality_config.type == 'embedding':
      meta_data['input_shape'][modality_name] = (
          -1, dataset_configs.num_frames, modality_config.feature_dimension)
      meta_data['input_dtype'][modality_name] = getattr(jnp, dtype_str)

  if dataset_configs.get('class_name_csv') is not None:
    meta_data['input_shape']['class_names'] = {
        'input_word_ids': (-1, dataset_configs.num_classes,
                           dataset_configs.tokenizer_config.max_num_tokens),
        'input_type_ids': (-1, dataset_configs.num_classes,
                           dataset_configs.tokenizer_config.max_num_tokens),
        'input_mask': (-1, dataset_configs.num_classes,
                       dataset_configs.tokenizer_config.max_num_tokens),
    }
    meta_data['input_dtype']['class_names'] = {
        'input_word_ids': jnp.int32,
        'input_type_ids': jnp.int32,
        'input_mask': jnp.int32,
    }
  if dataset_configs.get('class_name_embedding_npy') is not None:
    meta_data['input_shape']['class_names'] = (-1, dataset_configs.num_classes,
                                               class_name_embeddings.shape[-1])  # pytype: disable=attribute-error
    meta_data['input_dtype']['class_names'] = getattr(jnp, dtype_str)

  logging.info('Dataset metadata:\n%s', meta_data)

  return dataset_utils.Dataset(train_iter, eval_iter, test_iter, meta_data)
