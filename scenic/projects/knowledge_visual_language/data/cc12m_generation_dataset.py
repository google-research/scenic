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

"""Dataset and Loader for CC12M dataset for caption generation pre-training.

Key Changes: 1) use T5 tokenizer; 2) Use text prefix as enoder input;
3) add autoregressive output.

TODO(ziniu): probably consider image masking?
"""
import functools
from typing import Optional

from absl import logging
from flax import jax_utils
# import jax
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.dataset_lib.big_transfer import bit
from scenic.dataset_lib.big_transfer import builder
from scenic.dataset_lib import web_image_text_dataset
from scenic.projects.knowledge_visual_language.data import data_utils

FILTER_LENGTH = 16
PREFIX_MAX_LENGTH = 12
OUTPUT_MAX_LENGTH = 48
IMAGE_SIZE = 224


def get_default_dataset_config(runlocal=False, additional_valid_dataset=True):
  """Gets default configs for CC12M dataset."""
  dataset_configs = ml_collections.ConfigDict()
  # Add path to your data here:
  dataset_configs.dataset_dir = ''
  dataset_configs.train_split = 'full[10000:]'
  MAX_LENGTH = OUTPUT_MAX_LENGTH  # pylint: disable=invalid-name
  pp_common = (
      f'|t5_tokenize(max_num_tokens={MAX_LENGTH}, inkey="texts",'
      f' prompt="{data_utils.CAPTION_PREFIX}")|keep("image", "tokens")'
  )
  dataset_configs.max_num_tokens = MAX_LENGTH
  dataset_configs.image_size = IMAGE_SIZE
  dataset_configs.pp_train = (
      f'decode|resize(resize_size={IMAGE_SIZE})|value_range(-1,1)' + pp_common
  )
  dataset_configs.shuffle_buffer_size = 250000 if not runlocal else 50
  pp_common_eval = f'decode|resize(resize_size={IMAGE_SIZE})|value_range(-1,1)'
  pp_argus_eval = pp_common_eval + pp_common
  sub = '[:4]' if runlocal else ''
  if additional_valid_dataset:
    pp_coco_eval = (
        pp_common_eval
        + '|coco_captions(inkey="captions", outkey="texts")'
        + pp_common
    )
    # pp_imagenet_eval = pp_common_eval + (
    #     '|clip_i1k_label_names(inkey="label", outkey="texts")') + pp_common
    dataset_configs.val_split = [
        (
            'val',
            dataset_configs.dataset,
            ['full[:10000]', f'full{sub}'][runlocal],
            pp_argus_eval,
        ),
        ('coco', 'coco_captions', f'val{sub}', pp_coco_eval),
        # ('imagenet', 'imagenet2012', f'validation{sub}', pp_imagenet_eval),
    ]
  else:
    dataset_configs.val_split = f'full{sub}' if runlocal else 'full[:50000]'
    dataset_configs.pp_eval = pp_argus_eval

  dataset_configs.val_cache = 'loaded'  # Unfortunately, "batched" gets us OOM.
  dataset_configs.vocab_size = data_utils.VOCAB_SIZE_T5
  dataset_configs.prefetch_to_device = 2
  return dataset_configs


@datasets.add_dataset('cc12m_generation')
def get_dataset(
    *,
    batch_size,
    eval_batch_size,
    num_shards,
    dtype_str='float32',
    shuffle_seed=0,
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
  default_dataset_config = get_default_dataset_config(
      runlocal=False, additional_valid_dataset=True
  )
  if dataset_configs:
    default_dataset_config.update(dataset_configs)

  dataset_configs = default_dataset_config

  del rng
  assert dataset_configs is not None
  logging.info(
      'Loading train split of the %sfrom cc12m dataset.',
      dataset_configs.dataset,
  )

  def pp_fn(x, how):
    pp = builder.get_preprocess_fn(how)
    example = pp(x)
    return {'image': example['image'], 'tokens': example['tokens']}

  # E.g. for testing with TAP.
  shuffle_buffer_size = (
      1000 if num_shards == 1 else dataset_configs.shuffle_buffer_size
  )

  train_ds = data_utils.get_data(
      dataset=dataset_configs.dataset,
      split=dataset_configs.train_split,
      data_dir=dataset_configs.get('dataset_dir'),
      batch_size=batch_size,
      filter_fn=functools.partial(
          data_utils.filter_text_length, filter_len=FILTER_LENGTH
      ),
      preprocess_fn=functools.partial(pp_fn, how=dataset_configs.pp_train),
      shuffle_buffer_size=shuffle_buffer_size,
      prefetch=dataset_configs.get('prefetch_to_host', 2),
      cache='loaded',
      ignore_errors=True,
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
      inputs_key='encoder_input_image',
      train=True,
      batch_size=batch_size,
  )
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
  map_generation_split_batches = functools.partial(
      data_utils.map_generation_split, prefix_h=PREFIX_MAX_LENGTH
  )

  train_iter = iter(train_ds)
  train_iter = map(map_generation_split_batches, train_iter)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  if num_shards > 0:
    train_iter = map(shard_batches, train_iter)
    if dataset_configs.prefetch_to_device:
      train_iter = jax_utils.prefetch_to_device(
          train_iter, dataset_configs.prefetch_to_device
      )

  logging.info(
      'Loading validation split of the %sfrom cc12m dataset.',
      dataset_configs.dataset,
  )
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch,
      inputs_key='encoder_input_image',
      train=False,
      batch_size=eval_batch_size,
  )

  def _get_eval_iter(dataset, split, pp_eval):
    val_ds = data_utils.get_data(
        dataset=dataset,
        split=split,
        data_dir=dataset_configs.get('dataset_dir'),
        batch_size=eval_batch_size,
        filter_fn=functools.partial(
            data_utils.filter_text_length, filter_len=FILTER_LENGTH
        ),
        preprocess_fn=functools.partial(pp_fn, how=pp_eval),
        cache='batched',
        repeat_after_batching=True,
        drop_remainder=False,
    )

    valid_iter = iter(val_ds)
    valid_iter = map(map_generation_split_batches, valid_iter)
    valid_iter = map(bit.tf_to_numpy, valid_iter)
    valid_iter = map(maybe_pad_batches_eval, valid_iter)
    if num_shards > 0:
      valid_iter = map(shard_batches, valid_iter)
      if dataset_configs.prefetch_to_device:
        valid_iter = jax_utils.prefetch_to_device(
            valid_iter, dataset_configs.prefetch_to_device
        )

    return valid_iter

  def _get_num_eval_examples(dataset, split, data_dir):
    return dataset_utils.get_num_examples(dataset, split, data_dir)

  if isinstance(dataset_configs.val_split, str):
    valid_iter = _get_eval_iter(
        dataset_configs.dataset,
        dataset_configs.val_split,
        dataset_configs.pp_eval,
    )
    n_eval_ex = _get_num_eval_examples(
        dataset_configs.dataset,
        dataset_configs.val_split,
        data_dir=dataset_configs.get('dataset_dir'),
    )
  else:
    valid_iter, n_eval_ex = {}, {}
    for eval_spec in dataset_configs.val_split:
      name, dataset, split, pp_eval = eval_spec
      valid_iter[name] = _get_eval_iter(dataset, split, pp_eval)
      n_eval_ex[name] = _get_num_eval_examples(
          dataset, split, data_dir=dataset_configs.get('dataset_dir')
      )

  meta_data = {'num_train_examples': n_train_ex, 'num_eval_examples': n_eval_ex}

  if dataset_configs.get('extra_meta_data'):
    for k, v in dataset_configs.extra_meta_data.items():
      meta_data[k] = v

  image_shape = (-1, dataset_configs.image_size, dataset_configs.image_size, 3)
  predix_shape = (-1, PREFIX_MAX_LENGTH)
  input_shape = (-1, OUTPUT_MAX_LENGTH)

  meta_data['encoder_input_image_spec'] = (image_shape, getattr(jnp, dtype_str))
  meta_data['encoder_input_tokens_spec'] = (predix_shape, jnp.int16)
  meta_data['decoder_input_tokens_spec'] = (input_shape, jnp.int16)
  meta_data['decoder_target_tokens_spec'] = (input_shape, jnp.int16)
  return dataset_utils.Dataset(train_iter, valid_iter, None, meta_data)
