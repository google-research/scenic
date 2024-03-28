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

"""Dataset and Loader for Wikipedia Image-Text (WIT) dataset for retrieval training.

Only prepare <image, caption> paired with knowledge (contextualalized passages)
"""
import functools
from typing import Optional

from absl import logging
import jax
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.dataset_lib.big_transfer import builder
from scenic.dataset_lib.big_transfer import registry
from scenic.dataset_lib import web_image_text_dataset
from scenic.projects.knowledge_visual_language.data import data_utils
import tensorflow as tf

SPAN_MAX_LENGTH = 5
OUTPUT_MAX_LENGTH = 36
KNOWLEDGE_MAX_LENGTH = 320
IMAGE_SIZE = 224


@registry.Registry.register('preprocess_ops.get_table_knowledge', 'function')
def get_table_knowledge():
  """Concat title passage and document together to form knowledge."""

  def get_table_knowledge_fn(data):
    """Prepare Knowledge by concating hierarchy, passage and first-paragraph."""

    knowledges = [
        data['hierarchical_section_title'],
        data['context_section_description'],
        data['context_page_description'],
        data['caption_reference_description_canonicalized'],
        data['caption_alt_text_description_canonicalized'],
        tf.strings.regex_replace(
            data['caption_attribution_description_canonicalized'],
            '^english ',
            '',
        ),
    ]
    data['knowledge'] = tf.strings.join(knowledges, separator=' <extra_id_99> ')
    # data['raw_image'] = data['image']
    return data

  return get_table_knowledge_fn


def get_default_dataset_config():
  """Gets default configs for wit_internal (en) dataset."""
  dataset_configs = ml_collections.ConfigDict()
  # Add path to your data here:
  dataset_configs.dataset = ''
  dataset_configs.train_split = 'train'
  dataset_configs.output_max_num_tokens = OUTPUT_MAX_LENGTH
  dataset_configs.knowledge_max_num_tokens = OUTPUT_MAX_LENGTH
  dataset_configs.image_size = IMAGE_SIZE
  dataset_configs.pp_train = (
      f'get_table_knowledge|decode|resize(resize_size={IMAGE_SIZE})|value_range(-1,1)|t5_tokenize(max_num_tokens={KNOWLEDGE_MAX_LENGTH},'
      ' inkey="knowledge", outkey="knowledge_tokens",'
      f' prompt="{data_utils.KNOWLEDGE_PREFIX}")|keep("image",'
      ' "knowledge_tokens", "canonical_doc_id")'
  )
  dataset_configs.vocab_size = data_utils.VOCAB_SIZE_T5
  dataset_configs.prefetch_to_device = 2
  return dataset_configs


@datasets.add_dataset('wit_table')
def get_dataset(
    *,
    batch_size,
    eval_batch_size,
    num_shards,
    dtype_str='float32',
    shuffle_seed=None,
    rng=None,
    dataset_configs=None,
    dataset_service_address: Optional[str] = None,
):
  """Returns generators for the CC12M train, validation and test sets.

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

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
    a test_iter, and a dict of meta_data.
  """
  del batch_size
  default_dataset_config = get_default_dataset_config()
  if dataset_configs:
    default_dataset_config.update(dataset_configs)

  dataset_configs = default_dataset_config

  del rng
  assert dataset_configs is not None
  logging.info('Loading train split of the %s', dataset_configs.dataset)

  def pp_fn(x, how):
    pp = builder.get_preprocess_fn(how, remove_tpu_dtypes=False)
    example = pp(x)
    example['image'] = tf.cast(example['image'], dtype=dtype_str)
    return example

  # E.g. for testing with TAP.
  shuffle_buffer_size = None

  train_ds = data_utils.get_data(
      dataset=dataset_configs.dataset,
      split=dataset_configs.train_split,
      batch_size=eval_batch_size,
      preprocess_fn=functools.partial(pp_fn, how=dataset_configs.pp_train),
      shuffle_buffer_size=None,
      shuffle_files=False,
      prefetch=dataset_configs.get('prefetch_to_host', 2),
      cache='loaded',
      ignore_errors=True,
      drop_remainder=True,
  )

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError(
          'Using dataset service with a random seed causes each '
          'worker to produce exactly the same data. Add '
          'config.shuffle_seed = None to your config if you '
          'want to run with dataset service.'
      )
    logging.info('Using the tf.data service at %s', dataset_service_address)
    assert shuffle_buffer_size is not None
    train_ds = dataset_utils.distribute(train_ds, dataset_service_address)

  n_train_ex = dataset_utils.get_num_examples(
      dataset_configs.dataset,
      dataset_configs.train_split,
      data_dir=dataset_configs.get('dataset_dir'),
  )

  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch,
      inputs_key='image',
      train=True,
      batch_size=eval_batch_size,
  )
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)

  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  if num_shards > 0:
    train_iter = map(shard_batches, train_iter)

  meta_data = {
      'num_train_examples': n_train_ex,
      'example_per_shard': int(n_train_ex // jax.process_count()),
      'batch_size': eval_batch_size,
  }

  image_shape = (dataset_configs.image_size, dataset_configs.image_size, 3)
  knowledge_shape = (KNOWLEDGE_MAX_LENGTH + data_utils.PROMPT_LENGTH,)

  meta_data['image_spec'] = (image_shape, getattr(jnp, dtype_str))
  meta_data['knowledge_spec'] = (knowledge_shape, jnp.int16)
  return dataset_utils.Dataset(train_iter, None, None, meta_data)
