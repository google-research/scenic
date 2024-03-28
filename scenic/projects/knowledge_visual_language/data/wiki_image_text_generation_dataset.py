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
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
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


@registry.Registry.register('preprocess_ops.get_wit_knowledge', 'function')
def get_wit_knowledge():
  """Concat title passage and document together to form knowledge."""

  def get_wit_knowledge_fn(data):
    """Prepare Knowledge by concating hierarchy, passage and first-paragraph."""

    knowledges = [
        data['hierarchical_section_title'],
        data['context_section_description'],
        data['context_page_description'],
    ]
    data['knowledge'] = tf.strings.join(knowledges, separator=' <extra_id_99> ')

    caption = data['caption_reference_description_canonicalized']
    if tf.strings.length(caption) < 1:
      caption = data['caption_alt_text_description_canonicalized']
    if tf.strings.length(caption) < 1:
      caption = tf.strings.regex_replace(
          data['caption_attribution_description_canonicalized'], '^english ', ''
      )
    data['caption'] = caption
    return data

  return get_wit_knowledge_fn


def get_default_dataset_config(runlocal=False):
  """Gets default configs for wit_internal (en) dataset."""
  dataset_configs = ml_collections.ConfigDict()
  # Add path to your data here:
  dataset_configs.dataset = ''
  dataset_configs.train_split = 'train[1000:]'
  dataset_configs.output_max_num_tokens = OUTPUT_MAX_LENGTH
  dataset_configs.knowledge_max_num_tokens = OUTPUT_MAX_LENGTH
  dataset_configs.image_size = IMAGE_SIZE
  dataset_configs.pp_train = (
      f'decode|resize(resize_size={IMAGE_SIZE})|value_range(-1,1)|get_wit_knowledge|t5_tokenize(max_num_tokens={KNOWLEDGE_MAX_LENGTH},'
      ' inkey="knowledge", outkey="retr_texts",'
      f' prompt="{data_utils.KNOWLEDGE_PREFIX}")|t5_tokenize(max_num_tokens={OUTPUT_MAX_LENGTH * 3},'
      ' inkey="caption", outkey="caption_tokens",'
      f' prompt="{data_utils.CAPTION_PREFIX}")|keep("image", "caption_tokens",'
      ' "retr_texts")'
  )

  dataset_configs.val_split = [(
      'val',
      dataset_configs.dataset,
      'train[:1000]',
      dataset_configs.pp_train,
  )]

  dataset_configs.shuffle_buffer_size = 250000 if not runlocal else 50
  dataset_configs.val_cache = 'loaded'  # Unfortunately, "batched" gets us OOM.
  dataset_configs.vocab_size = data_utils.VOCAB_SIZE_T5
  dataset_configs.prefetch_to_device = 2
  return dataset_configs


def inception_crop(image, resize_size=224, area_min=20, area_max=80):
  """Random crop input image."""
  begin, size, _ = tf.image.sample_distorted_bounding_box(
      tf.shape(image),
      tf.zeros([0, 0, 4], tf.float32),
      area_range=(area_min / 100, area_max / 100),
      min_object_covered=0,  # Don't enforce a minimum area.
      use_image_if_no_bounding_boxes=True,
  )
  crop = tf.slice(image, begin, size)
  # Unfortunately, the above operation loses the depth-dimension. So we need
  # to restore it the manual way.
  crop.set_shape([None, None, image.shape[-1]])
  if resize_size:
    crop = tf.cast(
        tf.image.resize(crop, [resize_size, resize_size]), image.dtype
    )
  return crop


# def sample_retr_image(batch, random_ratio=0.):
#   """Sample image from similar sample by tfidf."""

#   rds = np.random.random(size=len(batch['encoder_input_image']))
#   passages = [p.decode('utf-8')[:256] for p in batch.pop('knowledge')]
#   passages_tfidf = TfidfVectorizer().fit_transform(passages)
#   sim = cosine_similarity(passages_tfidf)
#   np.fill_diagonal(sim, 0)
#   rd_imgs = batch['encoder_input_image'][np.argmax(sim, axis=1)]
#   del passages_tfidf, passages

#   batch['retr_images'] = []
#   for rd_img, rd, img in zip(rd_imgs, rds, batch['encoder_input_image']):
#     if rd < random_ratio:
#       batch['retr_images'] += [rd_img]
#     else:
#       batch['retr_images'] += [inception_crop(img, area_min=5, area_max=60)]
#   batch['retr_images'] = np.expand_dims(batch['retr_images'], axis=1)

#   crops = []
#   for img in batch['encoder_input_image']:
#     crop = inception_crop(img, area_min=40, area_max=100)
#     crops += [crop]
#   batch['encoder_input_image'] = np.stack(crops, axis=0)
#   return batch


def sample_retr_image(batch):
  """Sample image from similar sample by tfidf."""

  crops = []
  for img in batch['encoder_input_image']:
    crops += [inception_crop(img, area_min=5, area_max=60)]
  batch['retr_images'] = np.stack(crops, axis=0)
  return batch


@datasets.add_dataset('wiki_image_text_generation')
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
    example['image'] = tf.cast(example['image'], dtype=dtype_str)
    return example

  # E.g. for testing with TAP.
  shuffle_buffer_size = (
      1000 if num_shards == 1 else dataset_configs.shuffle_buffer_size
  )

  train_ds = data_utils.get_data(
      dataset=dataset_configs.dataset,
      split=dataset_configs.train_split,
      batch_size=batch_size,
      preprocess_fn=functools.partial(pp_fn, how=dataset_configs.pp_train),
      filter_fn=functools.partial(data_utils.filter_text_length, filter_len=4),
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
      data_utils.map_generation_split,
      span_len=SPAN_MAX_LENGTH,
      output_max_len=OUTPUT_MAX_LENGTH,
      split_key='caption_tokens',
      add_retr=False,
  )

  train_iter = iter(train_ds)
  train_iter = map(map_generation_split_batches, train_iter)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(sample_retr_image, train_iter)
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
        filter_fn=functools.partial(
            data_utils.filter_text_length, filter_len=4
        ),
        cache=dataset_configs.val_cache,
        repeat_after_batching=True,
        drop_remainder=False,
    )

    valid_iter = iter(val_ds)
    valid_iter = map(map_generation_split_batches, valid_iter)
    valid_iter = map(dataset_utils.tf_to_numpy, valid_iter)
    valid_iter = map(sample_retr_image, valid_iter)
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

  # builder, split, host_id, host_count

  # splitname, host_start, host_end = dataset_utils._get_data_range(
  #     dataset_configs.dataset,
  #     dataset_configs.train_split,
  #     data_dir=dataset_configs.get('dataset_dir'))

  meta_data = {
      'num_train_examples': n_train_ex,
      'example_per_shard': int(n_train_ex // jax.process_count()),
      'num_eval_examples': n_eval_ex,
  }

  if dataset_configs.get('extra_meta_data'):
    for k, v in dataset_configs.extra_meta_data.items():
      meta_data[k] = v

  image_shape = (-1, dataset_configs.image_size, dataset_configs.image_size, 3)
  predix_shape = (-1, data_utils.PROMPT_LENGTH + SPAN_MAX_LENGTH + 1)
  input_shape = (-1, OUTPUT_MAX_LENGTH)
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
  meta_data['retr_texts_spec'] = (retr_texts_shape, jnp.int16)
  meta_data['retr_images_spec'] = (retr_image_shape, getattr(jnp, dtype_str))
  meta_data['decoder_target_tokens_spec'] = (input_shape, jnp.int16)
  return dataset_utils.Dataset(train_iter, valid_iter, None, meta_data)
