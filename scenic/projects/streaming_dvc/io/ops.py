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

"""Preprocessing operations."""

import dataclasses
from typing import Optional, Sequence, Tuple, Union

from clu import preprocess_spec
from dmvr import tokenizers

from scenic.projects.baselines.centernet import transforms
from scenic.projects.t5 import tokenizer as t5_tokenizer
import tensorflow as tf


SP_MODEL_PATH = 'gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model'
SP_VOCAB_SIZE = 32128
TOKENIZER = Union[
    tokenizers.BertTokenizer,
    t5_tokenizer.SentencePieceTokenizer,
    ]


def get_tokenizer(tokenizer_weight_path) -> TOKENIZER:
  if tokenizer_weight_path == 't5':
    tokenizer = t5_tokenizer.build_dmvr_sp_model()
  else:
    tokenizer = tokenizers.BertTokenizer(tokenizer_weight_path)
  tokenizer.initialize()
  return tokenizer


@dataclasses.dataclass(frozen=True)
class InitPaddingMask:
  """Create a `padding_mask` of `ones` to match the current unpadded image."""

  def __call__(self, features):
    with tf.name_scope(type(self).__name__):
      h, w = transforms.get_hw(features, dtype=tf.int32)
      # padding_mask is initialized as ones. It will later be padded with zeros.
      features_new = features.copy()
      features_new['padding_mask'] = tf.ones((h, w), dtype=tf.float32)
      return features_new


@dataclasses.dataclass(frozen=True)
class FixedSizeCrop:
  """Crop a random sized region from the image as done in DETR.

  Assumes the features dictionary contains "inputs", "label" and "padding_mask".
  """
  crop_size: int

  def __call__(self, features):
    with tf.name_scope(type(self).__name__):
      h, w = transforms.get_hw(features, dtype=tf.int32)
      wcrop = tf.cast(tf.minimum(w, self.crop_size), tf.int32)
      hcrop = tf.cast(tf.minimum(h, self.crop_size), tf.int32)
      i = tf.random.uniform([], 0, h - hcrop + 1, dtype=tf.int32)
      j = tf.random.uniform([], 0, w - wcrop + 1, dtype=tf.int32)
      region = (i, j, hcrop, wcrop)
      features_new = features.copy()
      return transforms.crop(features_new, region)


@dataclasses.dataclass(frozen=True)
class CenterCrop:
  """Crop the center region from the image as done in DETR."""

  crop_size: int

  def __call__(self, features):
    with tf.name_scope(type(self).__name__):
      h, w = transforms.get_hw(features, dtype=tf.int32)
      wcrop = tf.cast(tf.minimum(w, self.crop_size), tf.int32)
      hcrop = tf.cast(tf.minimum(h, self.crop_size), tf.int32)
      i = (h - hcrop) // 2
      j = (w - wcrop) // 2
      region = (i, j, hcrop, wcrop)
      features_new = features.copy()
      return transforms.crop(features_new, region)


@dataclasses.dataclass(frozen=True)
class RandomRatioResize:
  """EfficientNet data augmentation. First resize than crop a fixed size."""

  min_scale: float
  max_scale: float
  target_size: int

  def __call__(self, features):
    with tf.name_scope(type(self).__name__):
      ratio = tf.random.uniform(
          [], self.min_scale, self.max_scale, dtype=tf.float32)
      size = tf.cast(tf.cast(self.target_size, tf.float32) * ratio, tf.int32)
      features_new = features.copy()
      return transforms.resize(features_new, size, max_size=size)


@dataclasses.dataclass(frozen=True)
class ResizeShorter:
  """Resize the shorter side to a fixed size."""

  target_size: int

  def __call__(self, features):
    with tf.name_scope(type(self).__name__):
      features_new = features.copy()
      return transforms.resize(features_new, self.target_size, max_size=None)


@dataclasses.dataclass
class DecodeActivityNetParagraphCaptionAnnotations:
  """Decode ActivityNet-Paragraph caption annotations."""

  tokenizer_weight_path: str
  num_captions_per_sample: int = 2
  max_text_tokens: int = 1024
  # concat_captions is set to 'concat_all' for training, 'concat_twosplit'
  # for evaluation when we have more than one annotated paragraph.
  concat_captions: str = 'concat_twosplit'
  # split field (in the input features) is used for evaluation, to determine
  # which subsets of sentence captions should be concatenated together.
  split_field: str = 'split'
  additional_keys: Tuple[str, ...] = ()
  _tokenizer: TOKENIZER = dataclasses.field(init=False)

  def __post_init__(self):
    self._tokenizer = get_tokenizer(self.tokenizer_weight_path)

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    image = tf.cast(features['image'], tf.float32)
    text_features = features['captions']['text']
    if self.concat_captions == 'concat_all':  # ActivityNet-Para train
      text_features = tf.strings.reduce_join(text_features)
      text_features = text_features[None]
    elif self.concat_captions == 'concat_twosplit':  # ActivityNet-Para eval
      assert self.num_captions_per_sample == 2, 'eval setting has 2 paragraphs.'
      split_features = features[self.split_field]
      text_features_para1 = tf.strings.reduce_join(
          text_features[split_features == 1])[None]
      # handle cases where the second paragraph split is missing (~1% of videos)
      text_features = tf.cond(
          tf.reduce_any(split_features == 2),
          lambda: tf.concat([   # pylint: disable=g-long-lambda
              text_features_para1,
              tf.strings.reduce_join(
                  text_features[split_features == 2])[None]
              ], axis=0),
          lambda: text_features_para1)
    else:
      raise ValueError(
          f'Unrecognized concat_captions value: "{self.concat_captions}"')
    text_tokens = self._tokenizer.string_tensor_to_indices(
        text_features,
        prepend_bos=True,
        append_eos=True,
        max_num_tokens=self.max_text_tokens,
    )[:, : self.max_text_tokens]
    target = {
        'text_tokens': text_tokens,
    }

    if self.concat_captions == 'concat_twosplit':
      # Needed for cases where only one paragraph is available during eval.
      pad_n = self.num_captions_per_sample
      target['text_tokens'] = tf.pad(
          target['text_tokens'],
          [[0, pad_n - tf.shape(text_tokens)[0]], [0, 0]])

    # Miscellaneous metadata, kept from COCO captions decoder.
    target['orig_size'] = tf.cast(tf.shape(image)[-3:-1], dtype=tf.int32)
    target['size'] = tf.identity(target['orig_size'])
    target['image/id'] = (
        features['image/id']
        if 'image/id' in features
        else tf.constant(0, dtype=tf.int32)
    )
    for key in self.additional_keys:
      target[key] = features[key]

    output = {
        'inputs': image,
        'label': target,
    }

    return output


@dataclasses.dataclass
class DecodeCocoCaptionAnnotations:
  """Decode Coco-Caption annotations."""

  tokenizer_weight_path: str
  num_captions_per_sample: int = 5
  max_text_tokens: int = 40
  concat_captions: bool = False
  max_context_tokens: int = -1
  append_context_eos: bool = False
  add_string_keys: Tuple[str, ...] = ()
  context_prefix: str = ''
  context_suffix: str = ''
  pad_text_tokens: bool = False
  _tokenizer: TOKENIZER = dataclasses.field(init=False)

  def __post_init__(self):
    self._tokenizer = get_tokenizer(self.tokenizer_weight_path)

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    image = tf.cast(features['image'], tf.float32)
    text_features = features['captions']['text']
    if self.concat_captions:
      text_features = tf.strings.reduce_join(text_features)
      text_features = text_features[None]
    text_tokens = self._tokenizer.string_tensor_to_indices(
        text_features,
        prepend_bos=True,
        append_eos=True,
        max_num_tokens=self.max_text_tokens,
    )[:, : self.max_text_tokens]
    inds = None
    if not self.concat_captions:
      # Randomly select num_captions_per_sample
      inds = tf.random.shuffle(tf.range(tf.shape(text_tokens)[0]))[
          : self.num_captions_per_sample
      ]
      text_tokens = tf.gather(text_tokens, inds)
    target = {
        'text_tokens': text_tokens,
    }

    if 'context' in features:
      assert not self.concat_captions, 'paragraph QA is not supported!'
      max_context_tokens = self.max_context_tokens if (
          self.max_context_tokens > 0) else self.max_text_tokens
      context = features['context']
      if self.context_prefix:
        context = tf.constant(
            self.context_prefix, dtype=tf.string)[None] + context
      if self.context_suffix:
        context = context + tf.constant(
            self.context_suffix, dtype=tf.string)[None]
      context_tokens = self._tokenizer.string_tensor_to_indices(
          context,
          prepend_bos=False,
          append_eos=self.append_context_eos,
          max_num_tokens=max_context_tokens,
      )[:, : max_context_tokens]
      context_tokens = tf.gather(context_tokens, inds)
      target['context_tokens'] = context_tokens

    if self.pad_text_tokens:
      # This is needed for datasets with variable number of captions. E.g,
      # NextQA test set with multiple GT answers. We don't need this for COCO
      # Or MSRVTT as they all have the same number of caption annotations.
      pad_n = self.num_captions_per_sample
      target['text_tokens'] = tf.pad(
          target['text_tokens'],
          [[0, pad_n - tf.shape(text_tokens)[0]], [0, 0]])
      if 'context_tokens' in target:
        target['context_tokens'] = tf.pad(
            target['context_tokens'],
            [[0, pad_n - tf.shape(target['context_tokens'])[0]], [0, 0]])

    target['orig_size'] = tf.cast(tf.shape(image)[-3:-1], dtype=tf.int32)
    target['size'] = tf.identity(target['orig_size'])
    target['image/id'] = (
        features['image/id']
        if 'image/id' in features
        else tf.constant(0, dtype=tf.int32)
    )
    for key in self.add_string_keys:
      target[key] = features[key]

    output = {
        'inputs': image,
        'label': target,
    }
    if preprocess_spec.SEED_KEY in features:
      output[preprocess_spec.SEED_KEY] = features[preprocess_spec.SEED_KEY]
    return output


@dataclasses.dataclass(frozen=True)
class ParseCustomExample:
  """Converts custom example into the standard format."""
  image_key: str
  caption_key: str
  context_key: str = ''

  def __call__(
      self, features: preprocess_spec.Features) -> preprocess_spec.Features:
    with tf.name_scope(type(self).__name__):
      features_new = {
          'captions': {'text': tf.sparse.to_dense(features[self.caption_key])},
          'image': tf.image.decode_jpeg(features[self.image_key], channels=3),
      }
      if self.context_key:
        features_new['context'] = tf.sparse.to_dense(
            features[self.context_key])
      return features_new




@dataclasses.dataclass(frozen=True)
class PadImages:
  """Pad images and "padding_mask" to a fixed size."""

  pad_h: int
  pad_w: Optional[int] = None
  pad_c: int = 3

  def __call__(
      self, features: preprocess_spec.Features) -> preprocess_spec.Features:

    features_new = features.copy()
    pad_w = self.pad_w or self.pad_h

    h = tf.shape(features['inputs'])[0]
    w = tf.shape(features['inputs'])[1]
    c = tf.shape(features['inputs'])[2]

    features_new['inputs'] = tf.pad(
        features['inputs'],
        [[0, self.pad_h - h], [0, pad_w - w], [0, self.pad_c - c]],
        mode='CONSTANT',
        constant_values=0)
    if 'padding_mask' in features:
      features_new['padding_mask'] = tf.pad(
          features['padding_mask'],
          [[0, self.pad_h - h], [0, pad_w - w]],
          mode='CONSTANT',
          constant_values=0)
    return features_new


@dataclasses.dataclass(frozen=True)
class VideoResizeCentralCrop:
  """Resizes video to the target size and takes a central crop of it."""

  crop_size: int

  def __call__(
      self, features: preprocess_spec.Features) -> preprocess_spec.Features:

    features_new = features.copy()
    frames = features['inputs']
    original_size = tf.shape(frames)[1:3]
    h, w = transforms.get_size_with_aspect_ratio(
        original_size, self.crop_size)
    rescaled_frames = tf.image.resize(frames, (h, w))
    wcrop = tf.cast(tf.minimum(w, self.crop_size), tf.int32)
    hcrop = tf.cast(tf.minimum(h, self.crop_size), tf.int32)
    i = (h - hcrop) // 2
    j = (w - wcrop) // 2
    cropped_frames = rescaled_frames[:, i: i + hcrop, j: j + wcrop]

    features_new['inputs'] = cropped_frames
    features['label']['size'] = tf.stack((hcrop, wcrop))
    return features_new


@dataclasses.dataclass(frozen=True)
class DecodeAndSubsampleVideo:
  """Decodes and subsamples frames from video, returning in common format.

  This common format is A dictionary of features containing
  {image, captions/text, image/id}.

  Attributes:
    num_frames: The number of frames to decode. These are randomly sampled if
      training. Otherwise, uniformly sampled through the video. If
      num_frames < 0, all the frames are returned.
    is_train: If training.
    caption_field: Which field in the sstable is used to store the caption.
    context_field: Field in the sstable for questions in QA tasks. Empty means
      no such field.
    max_frames: this is used when we want to use all-frames in a video
      (num_frames <=0), to limit the max number of frames in a batch.
    add_media_id: bool; if true, add revertiable media_id and frame indexes.
    additional_keys: tuple; string keys to decode
    additional_keys_decode_bytes: tuple of bool; if true, decodes the key
      strings added by add_string_keys into byte arrays. Only relevant when
      add_string_keys is provided.
  """

  num_frames: int
  is_train: bool
  caption_field: str = 'caption/string'
  context_field: str = ''
  max_frames: int = -1
  additional_keys: Tuple[str, ...] = ()
  additional_keys_decode_bytes: Tuple[bool, ...] = ()

  def __call__(
      self, features: preprocess_spec.Features) -> preprocess_spec.Features:

    frames = features['image/encoded']
    if self.num_frames > 0:
      max_frames = tf.shape(frames)[0]
      if self.is_train:
        inds = tf.sort(tf.random.shuffle(tf.range(max_frames))[
            :self.num_frames])
      else:
        stride = tf.maximum(max_frames // self.num_frames, 1)
        inds = tf.range(tf.minimum(self.num_frames, max_frames)) * stride
    else:
      inds = tf.range(tf.shape(frames)[0])

    frames = tf.gather(frames, inds)
    frames = tf.map_fn(
        lambda x: tf.image.decode_jpeg(x, channels=3),
        frames, back_prop=False, dtype=tf.uint8)
    if self.num_frames > 0:
      frames = tf.pad(
          frames,
          [[0, self.num_frames - tf.shape(frames)[0]], [0, 0], [0, 0], [0, 0]])
    if self.max_frames > 0:
      frames = frames[:self.max_frames]
      frames = tf.pad(
          frames,
          [[0, self.max_frames - tf.shape(frames)[0]], [0, 0], [0, 0], [0, 0]])
    video_id = tf.zeros((), tf.int32)
    # Return features dictionary in the standard format.
    features_new = {
        'image': frames,
        'captions': {
            'text': tf.sparse.to_dense(features[self.caption_field])},
        'image/id': video_id,
    }
    if self.context_field:
      features_new['context'] = tf.sparse.to_dense(features[self.context_field])
    for k, decode_bytes in zip(
        self.additional_keys, self.additional_keys_decode_bytes):
      k_str = tf.sparse.to_dense(features[k])
      if decode_bytes:
        features_new[k] = tf.io.decode_raw(k_str,
                                           out_type=tf.uint8,
                                           fixed_length=32)[0]
      else:
        features_new[k] = k_str

    if preprocess_spec.SEED_KEY in features:
      features_new[preprocess_spec.SEED_KEY] = features[
          preprocess_spec.SEED_KEY]

    return features_new


@dataclasses.dataclass(frozen=True)
class Drop():
  """Drops the given keys."""

  keys: Sequence[str]
  ignore_missing_features: bool = False

  def __call__(
      self, features: preprocess_spec.Features) -> preprocess_spec.Features:
    if not self.ignore_missing_features:
      for k in self.keys:
        if k not in features:
          raise ValueError(
              f"Could not drop features '{k}'. Available features:"
              f" {list(features)}"
          )
    return {k: v for k, v in features.items() if k not in self.keys}
