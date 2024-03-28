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

"""Util functions for preparing dataset wrapper in scenic."""
import functools
from big_vision.datasets.imagenet import class_names as imagenet_class_names
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib.big_transfer import registry
from scenic.dataset_lib.big_transfer.preprocessing import utils
from scenic.projects.t5 import tokenizer as t5_tokenizer
import tensorflow as tf


# import numpy as np

CAPTION_PREFIX = 'Please describe this image:'
VQA_PREFIX = 'Please based on this image to answer the question:'
KNOWLEDGE_PREFIX = 'Please summarize this knowledge:'

PROMPT_LENGTH = 6
VOCAB_SIZE_T5 = 32128
MASK_TOKEN_ID = 32099
BOS_ID = 32001
EOS_ID = 1
SEP_ID = 32000


@registry.Registry.register('preprocess_ops.clip_i1k_label_names', 'function')
@utils.InKeyOutKey(indefault='label', outdefault='texts')
def get_pp_clip_i1k_label_names():
  """Convert i1k label numbers to strings, using CLIP's class names."""

  def _pp_imagenet_labels(label):
    return tf.reshape(
        tf.gather(imagenet_class_names.CLIP_IMAGENET_CLASS_NAMES, label), (-1,)
    )

  return _pp_imagenet_labels


@registry.Registry.register('preprocess_ops.coco_captions', 'function')
@utils.InKeyOutKey(indefault='captions', outdefault='texts')
def get_coco_captions():
  """Extracts coco's captions from nested dict."""

  def _pp_coco_captions(captions, sample=False):
    t = captions['text']
    if sample:
      ts = tf.concat([t, ['']], axis=0)
      num_texts = tf.reduce_max([tf.shape(ts)[0] - 1, 1])
      idx = tf.random.uniform([], 0, num_texts, dtype=tf.int16)
    else:
      idx = tf.argmax(tf.strings.length(t))
    return tf.reshape(tf.strings.lower(t[idx]), (-1,))

  return _pp_coco_captions


@registry.Registry.register('preprocess_ops.t5_tokenize', 'function')
@utils.InKeyOutKey(indefault='texts', outdefault='tokens')
def get_t5_tokenize(max_num_tokens, append_eos=True, prompt=None):
  """Tokenizes a text using T5 Tokenizer."""

  tokenizer = t5_tokenizer.build_dmvr_sp_model()
  tokenizer.initialize()
  if prompt is None:
    prompt = [BOS_ID]
  else:
    prompt = tokenizer.string_to_indices(prompt, max_num_tokens=None)
    prompt = tf.concat([[BOS_ID], prompt], axis=-1)

  def _t5_tokenize(texts):
    if texts.shape.ndims == 0:
      texts = tf.reshape(texts, (-1,))
    tokens = tokenizer.string_tensor_to_indices(
        string_tensor=texts,
        max_num_tokens=max_num_tokens,
        append_eos=append_eos,
    )[0]
    return tf.cast(tf.concat([prompt, tokens], axis=-1), tf.int16)

  return _t5_tokenize


@registry.Registry.register('preprocess_ops.list_t5_tokenize', 'function')
@utils.InKeyOutKey(indefault='texts', outdefault='tokens')
def get_list_t5_tokenize(max_num_tokens, prompt=None):
  """Tokenizes a text using T5 Tokenizer."""

  tokenizer = t5_tokenizer.build_dmvr_sp_model()
  tokenizer.initialize()
  if prompt is None:
    prompt = [BOS_ID]
  else:
    prompt = tokenizer.string_to_indices(prompt, max_num_tokens=None)
    prompt = tf.concat([[BOS_ID], prompt], axis=-1)

  def add_prompt(tokens):
    return tf.concat([prompt, tokens], axis=-1)

  def _list_t5_tokenize(texts):
    if texts.shape.ndims == 0:
      texts = tf.reshape(texts, (-1,))
    token_list = tokenizer.string_tensor_to_indices(
        string_tensor=texts,
        max_num_tokens=max_num_tokens,
        append_eos=True,
    )
    token_list = tf.stack(tf.map_fn(add_prompt, token_list), axis=0)
    return tf.cast(token_list, tf.int16)

  return _list_t5_tokenize


@registry.Registry.register('preprocess_ops.multi_t5_tokenize', 'function')
@utils.InKeyOutKey(indefault='texts', outdefault='tokens')
def get_multi_t5_tokenize(max_num_tokens, append_eos=True):
  """Tokenizes a text using T5 Tokenizer."""

  tokenizer = t5_tokenizer.build_dmvr_sp_model()
  tokenizer.initialize()
  max_answers = 10

  def _multi_t5_tokenize(texts):
    parse = functools.partial(
        tokenizer.string_tensor_to_indices,
        max_num_tokens=max_num_tokens,
        append_eos=append_eos,
    )
    # if texts.shape.ndims == 1:
    #   tokens = parse(string_tensor=texts)
    # else:
    #   tokens = tf.map_fn(parse, texts)
    tokens = parse(string_tensor=texts)[:max_answers]
    return tf.cast(tokens, tf.int16)

  return _multi_t5_tokenize


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


def sample_retr_image(batch):
  """Sample image from similar sample by tfidf."""

  crops = []
  for img in batch['encoder_input_image']:
    crops += [inception_crop(img, area_min=10, area_max=80)]
  batch['retr_images'] = np.stack(crops, axis=0)
  return batch


def map_generation_split(
    batch, span_len, output_max_len, split_key='tokens', add_retr=False
):
  """Split tokens into prefix, decoder_input and decoder_output."""
  full_tokens = batch.pop(split_key)
  full_masks = tf.greater(full_tokens, 0)
  min_length = tf.reduce_max([
      tf.reduce_min(tf.reduce_sum(tf.cast(full_masks, tf.int16), axis=1)),
      PROMPT_LENGTH + 4,
  ]).numpy()
  max_length = PROMPT_LENGTH + span_len + 1
  bsz = full_tokens.shape[0]
  idx = tf.experimental.numpy.random.randint(
      low=PROMPT_LENGTH, high=tf.reduce_min([min_length, max_length]).numpy()
  ).numpy()
  input_tokens = [
      full_tokens[..., :idx],
      tf.ones([bsz, 1], dtype=tf.int16) * MASK_TOKEN_ID,
      tf.zeros([bsz, max_length - idx - 1], dtype=tf.int16),
  ]
  output_tokens = [
      tf.ones([bsz, 1], dtype=tf.int16) * BOS_ID,
      full_tokens[..., idx : idx + output_max_len],
  ]
  batch['encoder_input_tokens'] = tf.concat(input_tokens, axis=1)
  batch['encoder_input_image'] = batch.pop('image')
  output_tokens = tf.concat(output_tokens, axis=1)
  batch['decoder_input_tokens'] = output_tokens[..., :-1]
  batch['decoder_target_tokens'] = output_tokens[..., 1:]
  if add_retr:
    if 'retr_texts' in batch:
      batch['retr_texts'] = tf.expand_dims(batch['retr_texts'], axis=1)
    else:
      batch['retr_texts'] = tf.expand_dims(
          batch['decoder_input_tokens'], axis=1
      )
  return batch


def get_data(
    dataset,
    split,
    batch_size,
    filter_fn=None,
    preprocess_fn=lambda x: x,
    repeats=None,
    shuffle_buffer_size=None,
    prefetch=2,
    cache='loaded',
    repeat_after_batching=False,
    drop_remainder=True,
    data_dir=None,
    ignore_errors=False,
    shuffle_files=True,
    dataset_service_address=None,
):
  """API kept for backwards compatibility."""
  dataset = dataset_utils.get_dataset_tfds(
      dataset=dataset,
      split=split,
      shuffle_files=shuffle_files,
      data_dir=data_dir,
  )
  if 'train' not in split:
    dataset_service_address = None
  if filter_fn:
    dataset = dataset.filter(filter_fn)
  return dataset_utils.make_pipeline(
      data=dataset,
      preprocess_fn=preprocess_fn,
      batch_size=batch_size,
      drop_remainder=drop_remainder,
      cache=cache,
      repeats=repeats,
      prefetch=prefetch,
      shuffle_buffer_size=shuffle_buffer_size,
      repeat_after_batching=repeat_after_batching,
      ignore_errors=ignore_errors,
      dataset_service_address=dataset_service_address,
  )


def filter_text_length(d, filter_len):
  if 'texts' in d:
    return tf.strings.length(d['texts'][0]) > filter_len
  elif 'caption' in d:
    return tf.strings.length(d['caption']) > filter_len
  elif 'alt_texts' in d:
    return tf.strings.length(d['alt_texts'][0]) > filter_len
  return True


def _get_bytes_feature(example, name):
  return example.features.feature[name].bytes_list.value[0]


def _get_integer_list_feature(example, name):
  return list(example.features.feature[name].int64_list.value)


def _extract_wit_features(example):
  return [
      _get_integer_list_feature(example, 'knowledge'),
      _get_integer_list_feature(example, 'caption'),
      _get_bytes_feature(example, 'image'),
  ]


