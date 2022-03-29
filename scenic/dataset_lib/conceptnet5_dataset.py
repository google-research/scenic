# Copyright 2022 The Scenic Authors.
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

"""Data generators for Conceptnet5 dataset."""
import functools
from typing import Optional

from absl import logging
import big_vision.pp.ops_text as bv_text_ops

from flax import jax_utils
import jax
import jax.numpy as jnp

from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.dataset_lib.big_transfer import builder
from scenic.dataset_lib.big_transfer.preprocessing import utils
from scenic.dataset_lib.big_transfer.registry import Registry

import tensorflow as tf


PAD_TOKEN = 0


@Registry.register('preprocess_ops.tokenize', 'function')
@utils.InKeyOutKey(indefault=None, outdefault='labels')
def get_pp_tokenize(max_len,
                    eos,
                    model=bv_text_ops.KNOWN_TOKENIZERS['c4_en'],
                    lower=True,
                    sample_if_multi=True,
                    pad_value=PAD_TOKEN):
  """Tokenizes a text.

  Refer to Argus implementation for full details
  third_party/py/big_vision/pp/proj/argus/pp_ops.py

  Args:
    max_len: maximum length of the tokenized text.
    eos: Whether to add an "</s>" (end of sentence) token and whether to keep it
      when the sequence is longer than `max_len - 1`. See examples above for
      details. Valid values: "none", "yes", "sticky".
    model: a path to the pretrained sentencepiece model.
    lower: lowercase the text before tokenizing.
    sample_if_multi: If there's more than one, randomly pick one if this is
      True, or only take the first one if this is False.
    pad_value: which token to pad the sequence with. The default is `PAD_TOKEN`.
      Note that there is no guarantee to have any padding at the end of the
      sentence, if the sentence is longer than `max_len`.

  Returns:
    an op that outputs tokenized text.
  """

  if eos not in ('yes', 'none', 'sticky'):
    raise ValueError(f"Invalid value for eos: '{eos}'.")

  tokenizer = bv_text_ops.create_tokenizer(model, add_eos=eos != 'none')

  def _pp_tokenize(labels):
    labels = tf.reshape(labels, (-1,))
    # Append an empty string so we gracefully handle empty cases.
    labels = tf.concat([labels, ['']], axis=0)

    if sample_if_multi:
      num_texts = tf.maximum(tf.shape(labels)[0] - 1, 1)  # Don't sample "".
      txt = labels[tf.random.uniform([], 0, num_texts, dtype=tf.int32)]
    else:
      txt = labels[0]  # Always works, since we append "" earlier on.

    if lower:
      txt = tf.strings.lower(txt)

    return bv_text_ops.tokenize(
        txt, tokenizer, max_len, pad_value=pad_value, force_eos=eos == 'sticky')

  return _pp_tokenize


def preprocess_example(example, how):
  """Preprocesses the given example.

  Args:
    example: dict; Example that has an 'input' and a 'label'.
    how: string listing the preprocessing function

  Returns:
    A preprocessed example.
  """
  pp = builder.get_preprocess_fn(how)
  logging.info('Value of pre-example: %s', example)
  example = pp(example)
  logging.info('Value of example: %s', example)
  # To conceptnet5 format.
  return {'sub': example['arg1'], 'rel': example['rel'],
          'pred': example['arg2']}


@datasets.add_dataset('conceptnet5_dataset')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None):
  """Returns a generator for additional training Two-Tower on conceptnet5.

  Args:
    batch_size: int; Determines the train batch size.
    eval_batch_size: int; Not used.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image (e.g. 'float32').
    shuffle_seed: int; Seed for shuffling the training data.
    rng: Not used.
    dataset_configs: dict; Dataset specific configurations.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address.

  Returns:
    A dataset_utils.Dataset() which includes train_iter and dict of meta_data.
  """
  del rng
  assert dataset_configs is not None
  logging.info('Loading train split of the %s'
               'from bit dataset.', dataset_configs.dataset)
  target_is_onehot = 'onehot' in dataset_configs.pp_train

  train_ds = dataset_utils.get_data(
      dataset=dataset_configs.dataset,
      split=dataset_configs.train_split,
      batch_size=batch_size,
      preprocess_fn=functools.partial(
          preprocess_example, how=dataset_configs.pp_train),
      shuffle_buffer_size=dataset_configs.shuffle_buffer_size,
      prefetch=dataset_configs.get('prefetch_to_host', 2),
      drop_remainder=True,
      cache=False,
      ignore_errors=True)

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you '
                       'want to run with dataset service.')
    logging.info('Using the tf.data service at %s', dataset_service_address)
    assert dataset_configs.shuffle_buffer_size is not None
    train_ds = dataset_utils.distribute(train_ds, dataset_service_address)

  n_train_ex = dataset_utils.get_num_examples(dataset_configs.dataset,
                                              dataset_configs.train_split)

  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size,
      inputs_key='sub')
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)

  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(shard_batches, train_iter)
  if dataset_configs.prefetch_to_device:
    train_iter = jax_utils.prefetch_to_device(
        train_iter, dataset_configs.prefetch_to_device)

  logging.info('Loading validation split of the %s'
               'from bit dataset.', dataset_configs.dataset)
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch, train=False, batch_size=eval_batch_size,
      inputs_key='sub')
  def _get_eval_iter(dataset, split, pp_eval):
    val_ds = dataset_utils.get_data(
        dataset=dataset,
        split=split,
        data_dir=dataset_configs.get('dataset_dir'),
        batch_size=eval_batch_size,
        preprocess_fn=functools.partial(preprocess_example, how=pp_eval),
        cache='batched',
        repeat_after_batching=True,
        drop_remainder=False)

    valid_iter = iter(val_ds)
    valid_iter = map(dataset_utils.tf_to_numpy, valid_iter)
    valid_iter = map(maybe_pad_batches_eval, valid_iter)
    valid_iter = map(shard_batches, valid_iter)
    if dataset_configs.prefetch_to_device:
      valid_iter = jax_utils.prefetch_to_device(
          valid_iter, dataset_configs.prefetch_to_device)

    return valid_iter

  def _get_num_eval_examples(dataset, split):
    return dataset_utils.get_num_examples(dataset, split)

  if isinstance(dataset_configs.val_split, str):
    valid_iter = _get_eval_iter(dataset_configs.dataset,
                                dataset_configs.val_split,
                                dataset_configs.pp_eval)
    n_eval_ex = _get_num_eval_examples(dataset_configs.dataset,
                                       dataset_configs.val_split)

  logging.info('train_ds: %s', jax.tree_map(jnp.shape, train_ds))
  input_shape = (-1,) + tuple(train_ds.element_spec['sub'].shape[1:])

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
