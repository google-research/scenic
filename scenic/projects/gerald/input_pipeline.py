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

"""Data generators used for the GER input pipeline."""

import functools
from typing import Optional, Sequence, Union
from absl import logging

from dmvr import tokenizers
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib.big_transfer.preprocessing import ops as pp_ops
from scenic.projects.t5 import tokenizer as t5_tokenizer
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text

PRNGKey = jnp.ndarray


def make_inception_crop(features, crop_size, range_low=8):
  rescaled_image = pp_ops.get_inception_crop(
      crop_size, range_low)({'image': features['inputs']})['image']
  features['inputs'] = rescaled_image
  return features


def make_central_crop(features, crop_size):
  """Preprocessing and data-augmentation functions for eval."""
  resized_image = pp_ops.get_resize_small(
      crop_size)({'image': features['inputs']})['image']
  resized_image = pp_ops.get_central_crop(crop_size)(
      {'image': resized_image})['image']
  features['inputs'] = resized_image
  return features


def decode_annotations(
    example, tokenizer, max_context_tokens=0,
    questionid2id=None, question_key='question/id',
    wikipedia_entity_id2id=None, entity_key='entity/id'):
  """Given an instance and raw labels, creates <inputs, label> pair."""
  image = tf.cast(example['image'], tf.float32)

  assert entity_key in example
  assert wikipedia_entity_id2id is not None
  entity_token = tf.cast(wikipedia_entity_id2id.string_tensor_to_indices(
      example[entity_key]), dtype=tf.int32)
  target = {'entity/id': entity_token}

  if question_key in example and questionid2id is not None:
    target['image/id'] = tf.cast(questionid2id.string_tensor_to_indices(
        example['question/id']), dtype=tf.int32)[0][0]
  else:
    target['image/id'] = tf.cast(
        example['image/id'], dtype=tf.int32) if (
            'image/id' in example) else tf.constant(0, dtype=tf.int32)
  output = {
      'inputs': image,
      'label': target,
  }
  context_field = None
  if 'context' in example:
    context_field = example['context']
  if context_field is not None and max_context_tokens > 0:
    context_tokens = tokenizer.string_tensor_to_indices(
        tf.strings.lower(context_field), prepend_bos=False,
        append_eos=False, max_num_tokens=max_context_tokens,
    )[0, :max_context_tokens]
    output['context'] = context_tokens
  return output


def convert_oven_format(x, question_key='question/id', entity_key='entity/id'):
  """Converting OVEN format."""
  out = {
      'image': tf.io.decode_image(x['image']['encoded'], channels=3,
                                  expand_animations=False),
      'context': tf.reshape(tf.convert_to_tensor(
          x['question']['raw'], dtype=tf.string), (-1,)),
      question_key: tf.reshape(tf.convert_to_tensor(
          x['question']['id'], dtype=tf.string), (-1,)),
  }
  out[entity_key] = tf.reshape(tf.convert_to_tensor(x['answer']['id'],
                                                    dtype=tf.string), (-1,))
  return out


def convert_oven_entities_format(x, entity_key='entity/id'):
  """Converting OVEN entities format."""
  out = {
      'image': tf.image.decode_jpeg(x['wikipedia_image'], channels=3),
  }
  out[entity_key] = tf.reshape(tf.convert_to_tensor(x['wikidata_id'],
                                                    dtype=tf.string), (-1,))
  return out


def load_split(
    batch_size,
    *,
    train,
    dataset,
    preprocess_fn,
    decode_fn,
    tokenizer,
    split,
    data_dir,
    cache=False,
    max_size=224,
    max_context_tokens=40,
    shuffle_buffer_size=1000,
    shuffle_seed=0,
    private_threadpool_size=48,
    ):
  """Loads OVEN or entity-based pretraining dataset using TensorFlow Datasets.

  Args:
    batch_size: int; The batch size returned by the data pipeline.
    train: bool; Whether to load the train or evaluation split.
    dataset: str
    preprocess_fn: function; A function that given an example, train flag,
      and dtype returns the preprocessed the example. Note that the
      preprocessing is done BEFORE caching to re-use them.
    decode_fn: A function that given an example decodes the image, converts
      it to float32, mean-subtracts it, and pulls out the relevant parts from
      the tfds features.
    tokenizer: The text tokenizer (used to tokenize the input question).
    split: str.
    data_dir: str.
    cache: bool; whether to use the ds.cache or nor.
    max_size: int; Maximum image size.
    max_context_tokens: int;
    shuffle_buffer_size: int; Size of the shuffle buffer.
    shuffle_seed: int; Seed for shuffling the training data.
    private_threadpool_size: for dataloading.

  Returns:
    A `tf.data.Dataset`, and dataset info.
  """
  options = tf.data.Options()
  if private_threadpool_size > 0:
    options.threading.private_threadpool_size = private_threadpool_size
  # Loads OVEN dataset.
  if dataset == 'oven':
    ds = dataset_utils.get_dataset_tfds(
        dataset='oven',
        split=split,
        shuffle_files=False,
        data_dir=data_dir,
        skip_decode={'image': {'encoded': tfds.decode.SkipDecoding()}},)
    num_example = dataset_utils.get_num_examples('oven', split, data_dir)
    logging.info('%d files in %s oven', num_example, split)
    ds = ds.map(
        functools.partial(convert_oven_format),
        num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.with_options(options)
    if train:
      ds = ds.repeat()
      ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)

  # Loads entity-based pretraining dataset.
  else:
    assert dataset.startswith('oven_entities')
    ds = dataset_utils.get_dataset_tfds(
        dataset=dataset,
        split=split,
        shuffle_files=False,
        data_dir=data_dir,
        skip_decode={'wikipedia_image': tfds.decode.SkipDecoding()})
    num_example = dataset_utils.get_num_examples(dataset, split, data_dir)
    logging.info('%d files in %s %s', num_example, split, dataset)
    ds = ds.map(functools.partial(convert_oven_entities_format,),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.with_options(options)
    if train:
      ds = ds.repeat()
      ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)

  ds = ds.map(
      functools.partial(decode_fn, tokenizer=tokenizer),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if cache:
    ds = ds.cache()

  padded_shapes = {
      'inputs': [max_size, max_size, 3],
      'label': {
          'image/id': [],
          'entity/id': [1, 1],
      },
  }
  if max_context_tokens:
    padded_shapes['context'] = max_context_tokens

  ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if dataset.startswith('oven_entities') and max_context_tokens > 0:
    ds = ds.map(AddContextFn(tokenizer, max_context_tokens),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.padded_batch(batch_size, padded_shapes=padded_shapes,
                       drop_remainder=train)
  if not train:
    ds = ds.repeat()

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds, num_example


class AddContextFn():
  """Add a random question prompt."""

  def __init__(self, tokenizer, max_context_tokens):
    self.prompts = [
        'what is the main object?',
        'what is shown in the photo?',
        'which category of item is shown in the image?',
        'what item is presented in the image?',
        'what object is presented in the image?',
        'what is the main content of this image?',
    ]
    self.tokenized_prompts = None
    self.tokenizer = tokenizer
    self.max_context_tokens = max_context_tokens

  def __call__(self, features):
    ind = tf.random.uniform([], 0, len(self.prompts), dtype=tf.int32)
    context = tf.reshape(tf.convert_to_tensor(
        tf.convert_to_tensor(self.prompts)[ind], dtype=tf.string), (-1,))

    context_tokens = self.tokenizer.string_tensor_to_indices(
        tf.strings.lower(context), prepend_bos=False,
        append_eos=False, max_num_tokens=self.max_context_tokens,
    )[0, :self.max_context_tokens]

    features['context'] = context_tokens
    return features


def dataset_builder(*,
                    batch_size,
                    eval_batch_size,
                    num_shards,
                    dtype_str='float32',
                    shuffle_seed=0,
                    rng=None,
                    dataset_configs=None,):
  """Returns generators for pretraining dataset or OVEN sets.

  Args:
    batch_size: int; Determines the train batch size.
    eval_batch_size: int; Determines the evaluation batch size.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image. Only 'float32' is currently supported.
    shuffle_seed: int; Seed for shuffling the training data.
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    dataset_configs: dict; Dataset specific configurations. Must be empty.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
    a test_iter, and a dict of meta_data.
  """
  del rng
  del dtype_str

  dataset_configs = dataset_configs or {}
  data_dir = dataset_configs.get('data_dir')

  train_datasets = dataset_configs.get('train_datasets', '')
  if not isinstance(train_datasets, tuple):
    train_datasets = (train_datasets,)
  eval_datasets = dataset_configs.get('eval_datasets')
  assert len(eval_datasets) == 2  # seen and unseen
  crop_size = dataset_configs.get('crop_size', 224)
  max_context_tokens = dataset_configs.get('max_context_tokens', 0)
  tokenizer_type = dataset_configs.get('tokenizer_type', 'bert')

  train_preprocess_fn = functools.partial(
      make_inception_crop, crop_size=crop_size)
  eval_preprocess_fn = functools.partial(make_central_crop, crop_size=crop_size)

  # Tokenizer init
  if tokenizer_type == 'clip':
    tokenizer = tokenizers.ClipTokenizer()
  elif tokenizer_type == 't5':
    tokenizer = t5_tokenizer.build_dmvr_sp_model()
  else:
    assert tokenizer_type == 'bert'
    tokenizer = tokenizers.BertTokenizer()
  tokenizer.initialize()

  #####################################
  # We load the training set mixture. #
  #####################################

  # Entity id (str) to id (int)
  wikid2id_path = dataset_configs.get('wikid2id_path', None)
  wikid2id = None
  if wikid2id_path:
    wikid2id = Stringid2IntIdClass(wikid2id_path)
  train_ds, num_train_examples = [], 0
  dataset_sample_weights = dataset_configs.get('dataset_sample_weights', None)
  decode_fn = functools.partial(
      decode_annotations,
      max_context_tokens=max_context_tokens,
      wikipedia_entity_id2id=wikid2id,
  )
  for train_dataset_i in train_datasets:
    train_dataset_i, split_i = train_dataset_i.split('-')
    train_ds_i, num_train_examples_i = load_split(
        batch_size, train=True,
        dataset=train_dataset_i,
        preprocess_fn=train_preprocess_fn,
        split=split_i,
        decode_fn=decode_fn,
        tokenizer=tokenizer,
        max_context_tokens=max_context_tokens,
        shuffle_buffer_size=dataset_configs.get('shuffle_buffer_size', 1000),
        max_size=crop_size,
        shuffle_seed=shuffle_seed,
        data_dir=data_dir,
        private_threadpool_size=dataset_configs.get(
            'private_threadpool_size', 48),
    )
    num_train_examples += num_train_examples_i
    train_ds.append(train_ds_i)
  train_ds = tf.data.Dataset.sample_from_datasets(train_ds,
                                                  dataset_sample_weights)

  ###########################################
  # We load the eval sets(seen and unseen). #
  ###########################################

  wikid2id_path_eval = dataset_configs.get('wikid2id_path_eval', wikid2id_path)
  if wikid2id_path_eval != wikid2id_path:
    wikid2id = Stringid2IntIdClass(wikid2id_path)
  # Question id (str) to id (int) (used to get val/test data id)
  questionid2id_path = dataset_configs.get('questionid2id_path', None)
  questionid2id = None
  if questionid2id_path:
    questionid2id = Stringid2IntIdClass(questionid2id_path)
  eval_decode_fn = functools.partial(
      decode_annotations,
      max_context_tokens=max_context_tokens,
      questionid2id=questionid2id,
      wikipedia_entity_id2id=wikid2id,
  )
  # OVEN seen entities
  eval_seen_ds, num_eval_seen_examples = load_split(
      eval_batch_size, train=False,
      dataset='oven',
      preprocess_fn=eval_preprocess_fn,
      split=eval_datasets[0],
      decode_fn=eval_decode_fn,
      tokenizer=tokenizer,
      max_context_tokens=max_context_tokens,
      max_size=crop_size,
      data_dir=dataset_configs.get('oven_data_dir', data_dir),
      )
  # OVEN unseen entities
  eval_unseen_ds, num_eval_unseen_examples = load_split(
      eval_batch_size,
      dataset='oven',
      split=eval_datasets[1],
      train=False,
      preprocess_fn=eval_preprocess_fn,
      max_size=crop_size,
      decode_fn=eval_decode_fn,
      max_context_tokens=max_context_tokens,
      tokenizer=tokenizer,
      data_dir=dataset_configs.get('oven_data_dir', data_dir),
      )

  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size)
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch, train=False, batch_size=eval_batch_size)
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)

  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(shard_batches, train_iter)
  if dataset_configs.get('prefetch_to_device'):
    # Async bind batch to device which speeds up training.
    train_iter = jax_utils.prefetch_to_device(
        train_iter, dataset_configs.get('prefetch_to_device'))

  eval_seen_iter = iter(eval_seen_ds)
  eval_seen_iter = map(dataset_utils.tf_to_numpy, eval_seen_iter)
  eval_seen_iter = map(maybe_pad_batches_eval, eval_seen_iter)
  eval_seen_iter = map(shard_batches, eval_seen_iter)

  eval_unseen_iter = iter(eval_unseen_ds)
  eval_unseen_iter = map(dataset_utils.tf_to_numpy, eval_unseen_iter)
  eval_unseen_iter = map(maybe_pad_batches_eval, eval_unseen_iter)
  eval_unseen_iter = map(shard_batches, eval_unseen_iter)

  meta_data = {
      'num_train_examples': num_train_examples,
      'num_eval_seen_examples': num_eval_seen_examples,
      'num_eval_unseen_examples': num_eval_unseen_examples,
      'input_dtype': jnp.float32,
      'input_shape': [-1, crop_size, crop_size, 3],
  }
  return dataset_utils.Dataset(train_iter, eval_seen_iter, eval_unseen_iter,
                               meta_data)


def get_dataset(
    config: ml_collections.ConfigDict,
    data_rng: PRNGKey,
    *,
    dataset_name: Optional[str] = None,
    dataset_configs: Optional[ml_collections.ConfigDict] = None
) -> dataset_utils.Dataset:
  """Creates dataset.

  Args:
    config: The configuration of the experiment.
    data_rng: Random number generator key to use for the dataset.
    dataset_name: Name of dataset to load, if not reading from the config.
    dataset_configs: Configuration of the dataset, if not reading directly from
      the config.

  Returns:
    A dataset_utils.Dataset object.
  """
  device_count = jax.device_count()
  logging.info('device_count: %d', device_count)
  logging.info('num_hosts : %d', jax.process_count())
  logging.info('host_id : %d', jax.process_index())
  del dataset_name  # We get dataset name from dataset_configs.data_path

  batch_size = config.batch_size
  if batch_size % device_count > 0:
    raise ValueError(f'Batch size ({batch_size}) must be divisible by the '
                     f'number of devices ({device_count})')

  eval_batch_size = config.get('eval_batch_size', batch_size)
  if eval_batch_size % device_count > 0:
    raise ValueError(f'Eval batch size ({eval_batch_size}) must be divisible '
                     f'by the number of devices ({device_count})')

  local_batch_size = batch_size // jax.process_count()
  eval_local_batch_size = eval_batch_size // jax.process_count()
  device_batch_size = batch_size // device_count
  logging.info('local_batch_size : %d', local_batch_size)
  logging.info('device_batch_size : %d', device_batch_size)

  shuffle_seed = config.get('shuffle_seed', None)

  dataset_configs = dataset_configs or config.get('dataset_configs')
  dataset = dataset_builder(
      batch_size=local_batch_size,
      eval_batch_size=eval_local_batch_size,
      num_shards=jax.local_device_count(),
      dtype_str=config.data_dtype_str,
      rng=data_rng,
      shuffle_seed=shuffle_seed,
      dataset_configs=dataset_configs,)

  return dataset


class Stringid2IntIdClass():
  """Helper to go from question id (str) to id (int)."""

  def __init__(self, vocabulary_path: str):
    """Initializes the `Questionid2IdClass`."""
    # Parse the vocabulary.
    idx2word = {}
    self._vocabulary_path = vocabulary_path
    with tf.io.gfile.GFile(vocabulary_path) as f:
      for idx, line in enumerate(f):
        word = line.strip().replace('"', '')
        idx2word[idx] = word

    # Validate.
    if len(idx2word) != len(set(idx2word.values())):
      raise ValueError('Words in vocabulary are not unique.')

    self._idx2word = idx2word
    self._word2idx = {v: k for k, v in idx2word.items()}

    self._vocab_size = len(idx2word)
    ids_tensor = tf.constant([i for _, i in self._word2idx.items()],
                             dtype=tf.int32)
    words_tensor = tf.constant([w for w, _ in self._word2idx.items()],
                               dtype=tf.string)
    self._tf_word2idx = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(words_tensor, ids_tensor), -1)
    self._tf_whitespace_tokenizer = tensorflow_text.WhitespaceTokenizer()
    logging.info('String ID --> Int ID initialized from file %s with %d items.',
                 vocabulary_path, self._vocab_size)

  def string_tensor_to_indices(
      self, string_tensor: Union[tf.Tensor, Sequence[str]],) -> tf.Tensor:
    tokenized = self._tf_whitespace_tokenizer.tokenize(string_tensor)
    tokenized = self._tf_word2idx.lookup(tokenized)
    max_num_tokens = 1
    shape = None if max_num_tokens is None else [None, max_num_tokens]
    tokenized = tokenized.to_tensor(default_value=-1, shape=shape)
    return tokenized

  def indices_to_string(self, idx: int) -> str:
    return self._idx2word[idx]
