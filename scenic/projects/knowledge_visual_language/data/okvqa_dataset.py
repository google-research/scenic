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

"""Dataset and Loader for OK-VQA dataset."""

import functools
from typing import Optional

from absl import logging
from flax import jax_utils
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.dataset_lib.big_transfer import bit
from scenic.dataset_lib.big_transfer import builder
from scenic.dataset_lib.big_transfer import registry
from scenic.dataset_lib import web_image_text_dataset
from scenic.projects.knowledge_visual_language.data import data_utils
import tensorflow as tf

# import jax
OUTPUT_MAX_LENGTH = 64
IMAGE_SIZE = 224
QUESTION_LENGTH = 64
ANSWER_LENGTH = 32
KNOWLEDGE_MAX_LENGTH = 320


@registry.Registry.register('preprocess_ops.get_qa_pair', 'function')
def get_qa_pair():
  """Concat title passage and document together to form knowledge."""

  def get_qa_pair_fn(data):
    """Prepare Knowledge by concating hierarchy, passage and first-paragraph."""
    data['question'] = data['question/answers']['question_text']
    data['answers'] = tf.reshape(data['question/answers']['answers'], (-1,))
    data['answer'] = data['answers'][0]
    data['top_answers'] = tf.strings.reduce_join(
        data['answers'], separator=', '
    )
    return data

  return get_qa_pair_fn


def map_vqa_split(batch):
  """Split answer into decoder_input and decoder_output."""

  full_tokens = batch.pop('answer')
  batch['decoder_input_tokens'] = full_tokens[..., :-1]
  batch['decoder_target_tokens'] = full_tokens[..., 1:]
  return batch


def get_default_dataset_config(runlocal=False):
  """Gets default configs for CC12M dataset."""
  dataset_configs = ml_collections.ConfigDict()
  dataset_configs.dataset = 'okvqa'
  # Add path to your data here:
  dataset_configs.dataset_dir = ''
  dataset_configs.train_split = 'train'
  dataset_configs.question_max_num_tokens = QUESTION_LENGTH
  dataset_configs.answer_max_num_tokens = ANSWER_LENGTH
  dataset_configs.image_size = IMAGE_SIZE
  dataset_configs.pp_train = (
      f'decode|resize(resize_size={IMAGE_SIZE})|value_range(-1,1)|get_qa_pair|t5_tokenize(max_num_tokens={KNOWLEDGE_MAX_LENGTH},inkey="top_answers",'
      ' outkey="retr_texts",'
      f' prompt="{data_utils.KNOWLEDGE_PREFIX}")|t5_tokenize(max_num_tokens={ANSWER_LENGTH},'
      f' inkey="answer",outkey="answer")|multi_t5_tokenize(max_num_tokens={ANSWER_LENGTH},inkey="answers",outkey="answers")|t5_tokenize(max_num_tokens={QUESTION_LENGTH},'
      ' inkey="question", outkey="question",'
      f' prompt="{data_utils.VQA_PREFIX}")|keep("image", "question", "answer",'
      ' "retr_texts", "answers")'
  )

  dataset_configs.val_split = [(
      'val',
      dataset_configs.dataset,
      'validation',
      dataset_configs.pp_train,
  )]

  dataset_configs.shuffle_buffer_size = 10000 if not runlocal else 50
  dataset_configs.val_cache = 'loaded'  # Unfortunately, "batched" gets us OOM.
  dataset_configs.vocab_size = data_utils.VOCAB_SIZE_T5
  dataset_configs.prefetch_to_device = 2
  return dataset_configs


@datasets.add_dataset('okvqa')
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
  """Returns generators for the OKVQA train and validation sets.

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
  default_dataset_config = get_default_dataset_config(runlocal=False)
  if dataset_configs:
    default_dataset_config.update(dataset_configs)

  dataset_configs = default_dataset_config

  del rng
  assert dataset_configs is not None
  logging.info('Loading train split of the %s', dataset_configs.dataset)

  def pp_fn(x, how):
    pp = builder.get_preprocess_fn(how, remove_tpu_dtypes=False)
    example = pp(x)
    return {
        'encoder_input_image': example['image'],
        'encoder_input_tokens': example['question'],
        'answer': example['answer'],
        'answers': example['answers'],
        'retr_texts': example['retr_texts'],
    }

  # E.g. for testing with TAP.
  shuffle_buffer_size = (
      1000 if num_shards == 1 else dataset_configs.shuffle_buffer_size
  )

  train_ds = data_utils.get_data(
      dataset=dataset_configs.dataset,
      split=dataset_configs.train_split,
      data_dir=dataset_configs.get('dataset_dir'),
      batch_size=batch_size,
      preprocess_fn=functools.partial(pp_fn, how=dataset_configs.pp_train),
      shuffle_buffer_size=shuffle_buffer_size,
      prefetch=dataset_configs.get('prefetch_to_host', 2),
      cache=dataset_configs.val_cache,
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

  train_iter = iter(train_ds)
  train_iter = map(map_vqa_split, train_iter)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(data_utils.sample_retr_image, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  if num_shards > 0:
    train_iter = map(shard_batches, train_iter)
    if dataset_configs.prefetch_to_device:
      train_iter = jax_utils.prefetch_to_device(
          train_iter, dataset_configs.prefetch_to_device
      )

  logging.info('Loading validation split of the %s', dataset_configs.dataset)
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
        preprocess_fn=functools.partial(pp_fn, how=pp_eval),
        cache='batched',
        repeat_after_batching=True,
        drop_remainder=False,
    )

    valid_iter = iter(val_ds)
    valid_iter = map(map_vqa_split, valid_iter)
    valid_iter = map(bit.tf_to_numpy, valid_iter)
    valid_iter = map(data_utils.sample_retr_image, valid_iter)
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
  predix_shape = (-1, QUESTION_LENGTH)
  input_shape = (-1, ANSWER_LENGTH)
  retr_texts_shape = (-1, KNOWLEDGE_MAX_LENGTH + data_utils.PROMPT_LENGTH)
  retr_image_shape = (
      -1,
      dataset_configs.image_size,
      dataset_configs.image_size,
      3,
  )
  meta_data['encoder_input_image_spec'] = (image_shape, getattr(jnp, dtype_str))
  meta_data['encoder_input_tokens_spec'] = (predix_shape, jnp.int16)
  meta_data['decoder_input_tokens_spec'] = (input_shape, jnp.int16)
  meta_data['decoder_target_tokens_spec'] = (input_shape, jnp.int16)
  meta_data['retr_texts_spec'] = (retr_texts_shape, jnp.int16)
  meta_data['retr_images_spec'] = (retr_image_shape, getattr(jnp, dtype_str))
  return dataset_utils.Dataset(train_iter, valid_iter, None, meta_data)
