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
from typing import Optional, Any, Sequence

from clu import preprocess_spec
from scenic.projects.pixel_llm import tokenizers as pixel_llm_tokenizers
from scenic.projects.pixel_llm.io import transforms
import tensorflow as tf

TOKENIZER = pixel_llm_tokenizers.TOKENIZER
get_tokenizer = pixel_llm_tokenizers.get_tokenizer
INF = 2**16 - 1

PADDING_QUERY = ''


@dataclasses.dataclass(frozen=True)
class Drop:
  """Drops the given keys."""

  keys: Sequence[str]
  ignore_missing_features: bool = False

  def __call__(self, features):
    if not self.ignore_missing_features:
      for k in self.keys:
        if k not in features:
          raise ValueError(
              f"Could not drop features '{k}'. Available features:"
              f" {list(features)}"
          )
    return {k: v for k, v in features.items() if k not in self.keys}


@dataclasses.dataclass(frozen=True)
class DropNested:
  """Drops the nested given keys."""

  parent_key: str
  keys: Sequence[str]
  ignore_missing_features: bool = False

  def __call__(self, features):
    child_features = features[self.parent_key]
    if not self.ignore_missing_features:
      for k in self.keys:
        if k not in child_features:
          raise ValueError(
              f"Could not drop features '{k}'. Available features:"
              f' {list(child_features)}'
          )
    new_child_features = {
        k: v for k, v in child_features.items() if k not in self.keys
    }
    features[self.parent_key] = new_child_features

    return features


def point_to_coord(point, image_size):
  """Converts normalized point coordinates to integer.

  Args:
    point: Tensor of shape [..., 2]
    image_size: Tensor or list/tuple of two elements representing (height,
      width)

  Returns:
      A tensor of the same shape as input, but in integer coordinates
  """
  height, width = image_size[0], image_size[1]
  point_int = tf.round(point * [width, height])
  return point_int


def bbox_to_coord(bbox, image_size):
  """Converts normalized bounding box coordinates (xyxy) to integer.

  Args:
    bbox: Tensor of shape [..., 4]
    image_size: Tensor or list/tuple of two elements representing (height,
      width)

  Returns:
      A tensor of the same shape as input, but in integer coordinates
  """
  height, width = image_size[0], image_size[1]
  bbox_int = tf.round(bbox * [width, height, width, height])
  return bbox_int


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
      # TODO(aarnab, zhouxy): Should use stateless rng and provide seeds
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
      # TODO(aarnab, zhouxy): Should use stateless rng and provide seeds.
      ratio = tf.random.uniform(
          [], self.min_scale, self.max_scale, dtype=tf.float32)
      size = tf.cast(tf.cast(self.target_size, tf.float32) * ratio, tf.int32)
      features_new = features.copy()
      return transforms.resize(features_new, size, max_size=size)


@dataclasses.dataclass(frozen=True)
class ResizeShorter:
  """Resize the shorter side to a fixed size."""

  target_size: int
  max_size: Optional[int] = None

  def __call__(self, features):
    with tf.name_scope(type(self).__name__):
      features_new = features.copy()
      return transforms.resize(
          features_new, self.target_size, max_size=self.max_size)


@dataclasses.dataclass(frozen=True)
class RandomHorizontalFlip:
  """Horizontally flip image and boxes [cxcywh format] with probability `p`."""

  p: float = 0.5

  def __call__(self, features):
    rnd = tf.random.uniform([], minval=0.0, maxval=1.0, dtype=tf.float32)
    if rnd < self.p:
      return transforms.hflip(
          transforms.identity(features)
      )  # Identity helps avoid autograph errors.
    else:
      return transforms.identity(features)


def decode_boxes(bbox, size):
  """Convert yxyx [0, 1] normalized boxes to xyxy unnormalized format."""
  y0, x0, y1, x1 = tf.split(bbox, 4, axis=-1)
  h = tf.cast(size[0], tf.float32)
  w = tf.cast(size[1], tf.float32)

  y0 = tf.clip_by_value(y0 * h, 0.0, h)
  x0 = tf.clip_by_value(x0 * w, 0.0, w)
  y1 = tf.clip_by_value(y1 * h, 0.0, h)
  x1 = tf.clip_by_value(x1 * w, 0.0, w)

  bbox = tf.concat([x0, y0, x1, y1], axis=-1)
  return bbox


@dataclasses.dataclass
class DecodeCocoCaptionAnnotations:
  """Decode Coco-Caption annotations."""

  tokenizer_weight_path: str
  num_captions_per_sample: int = 5
  max_text_tokens: int = 40
  max_context_tokens: int = -1
  append_context_eos: bool = False
  context_prefix: str = ''
  context_suffix: str = ''
  class_id_offset: int = 1
  _tokenizer: TOKENIZER = dataclasses.field(init=False)

  def __post_init__(self):
    self._tokenizer = get_tokenizer(self.tokenizer_weight_path)
    self._tokenizer.initialize()

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    image = tf.cast(features['image'], tf.float32)
    text_features = features['captions']['text']
    text_tokens = self._tokenizer.string_tensor_to_indices(
        text_features,
        prepend_bos=True,
        append_eos=True,
        max_num_tokens=self.max_text_tokens,
    )[:, : self.max_text_tokens]
    # Randomly select num_captions_per_sample
    inds = tf.random.shuffle(tf.range(tf.shape(text_tokens)[0]))[
        : self.num_captions_per_sample
    ]
    text_tokens = tf.gather(text_tokens, inds)
    target = {
        'text_tokens': text_tokens,
    }

    if 'context' in features:
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

    target['orig_size'] = tf.cast(tf.shape(image)[-3:-1], dtype=tf.int32)
    target['size'] = tf.identity(target['orig_size'])
    target['image/id'] = (
        tf.cast(features['image/id'], dtype=tf.int64)
        if 'image/id' in features
        else tf.constant(0, dtype=tf.int64)
    )
    target['labels'] = (
        tf.zeros(tf.shape(text_tokens)[0], dtype=tf.int64)
        + self.class_id_offset
    )

    output = {
        'inputs': image,
        'label': target,
    }
    return output


@dataclasses.dataclass
class AddPromptTokens:
  """Adds additional prompt tokens which can be used as context for decoding."""

  tokenizer_weight_path: str
  num_captions_per_sample: int = 5
  prompt: list[str] = dataclasses.field(default_factory=lambda: ['a photo of '])
  max_context_tokens: int = 8
  append_eos: bool = False
  # randomly samples one prompt if many are given
  prompt_sampling_strategy: str = 'uniform'

  _tokenizer: TOKENIZER = dataclasses.field(init=False)
  _prompt_tensor: Any = dataclasses.field(init=False)

  def __post_init__(self):
    self._tokenizer = get_tokenizer(self.tokenizer_weight_path)
    self._tokenizer.initialize()
    if len(self.prompt) > 1:
      assert (
          self.prompt_sampling_strategy == 'uniform'
      ), 'No other sampling strategy implemented'
    self._prompt_tensor = tf.constant(
        self.prompt,
        dtype=tf.string,
    )

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    if self.prompt:
      # uniformly sample a single caption
      inds = tf.random.uniform(
          shape=(self.num_captions_per_sample,),
          minval=0,
          maxval=tf.shape(self._prompt_tensor)[-1],
          dtype=tf.int32,
      )
      selected_text = tf.gather(self._prompt_tensor, inds)
      text_tokens = self._tokenizer.string_tensor_to_indices(
          selected_text,
          prepend_bos=False,
          append_eos=self.append_eos,
          max_num_tokens=self.max_context_tokens,
      )
      features['label']['context_tokens'] = text_tokens
    return features


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
    features_new['label']['padded_size'] = tf.stack([self.pad_h, self.pad_w])
    if 'padding_mask' in features:
      features_new['padding_mask'] = tf.pad(
          features['padding_mask'],
          [[0, self.pad_h - h], [0, pad_w - w]],
          mode='CONSTANT',
          constant_values=0)
    return features_new


@dataclasses.dataclass(frozen=True)
class PadMasks:
  """Pad images and "padding_mask" to a fixed size."""

  max_masks: int
  pad_h: int
  pad_w: Optional[int] = None
  pad_c: int = 1

  def __call__(
      self, features: preprocess_spec.Features) -> preprocess_spec.Features:

    features_new = features.copy()
    pad_w = self.pad_w or self.pad_h

    masks = features['label']['masks'][:self.max_masks]
    num_masks = tf.shape(masks)[0]
    h = tf.shape(masks)[1]
    w = tf.shape(masks)[2]
    c = tf.shape(masks)[3]

    features_new['label']['masks'] = tf.pad(
        masks,
        [
            [0, self.max_masks - num_masks],
            [0, self.pad_h - h],
            [0, pad_w - w],
            [0, self.pad_c - c],
        ],
        mode='CONSTANT',
        constant_values=0,
    )
    return features_new


@dataclasses.dataclass(frozen=True)
class PadDetectionAnnotations:
  """Pad detection annotations to a fixed size."""

  max_boxes: int

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    features_new = features.copy()
    # force padding boxes and labels
    if 'boxes' not in features['label']:
      features_new['label']['boxes'] = tf.zeros((self.max_boxes, 4))
    if 'labels' not in features['label']:
      features_new['label']['labels'] = tf.zeros(
          (self.max_boxes,), dtype=tf.int64
      )

    for key in [
        'boxes',
        'text_tokens',
        'context_tokens',
        'area',
        'objects/id',
        'is_crowd',
        'labels',
        'refexp_ids',
    ]:
      if key not in features['label']:
        continue
      item = features['label'][key][: self.max_boxes]
      num_item = tf.shape(item)[0]
      features_new['label'][key] = tf.pad(
          item,
          [[0, self.max_boxes - num_item]]
          + [[0, 0]] * (len(tf.shape(item)) - 1),
          mode='CONSTANT',
          constant_values=0,
      )

    return features_new


@dataclasses.dataclass(frozen=True)
class PadLocoAnnotations:
  """Pad localized narrative to a fixed size."""

  num_prompts: int
  num_points: int

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    features_new = features.copy()
    max_cap_len = tf.shape(features['label']['text_tokens'])[-1]
    # force padding boxes and labels
    if 'points' not in features['label']:
      features_new['label']['points'] = tf.zeros(
          (self.num_prompts, max_cap_len, self.num_points, 2)
      )

    for key in ['points', 'prompt_points', 'token_phrase_idx']:
      if key not in features['label']:
        continue
      item = features['label'][key][: self.num_prompts]
      num_item = tf.shape(item)[0]
      features_new['label'][key] = tf.pad(
          item,
          [[0, self.num_prompts - num_item]]
          + [[0, 0]] * (len(tf.shape(item)) - 1),
          mode='CONSTANT',
          constant_values=0,
      )

    return features_new


@dataclasses.dataclass(frozen=True)
class PadCaptionAnnotations:
  """Pad detection annotations to a fixed size."""

  max_captions: int

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    features_new = features.copy()
    # force padding boxes and labels
    if 'labels' not in features['label']:
      features_new['label']['labels'] = tf.zeros(
          (self.max_captions,), dtype=tf.int64
      )

    for key in [
        'text_tokens',
        'context_tokens',
        'labels',
        'refexp_ids',
    ]:
      if key not in features['label']:
        continue
      item = features['label'][key][: self.max_captions]
      num_item = tf.shape(item)[0]
      features_new['label'][key] = tf.pad(
          item,
          [[0, self.max_captions - num_item]]
          + [[0, 0]] * (len(tf.shape(item)) - 1),
          mode='CONSTANT',
          constant_values=0,
      )

    return features_new


def build_solid_grid(points_per_side, with_offset=True):
  """Generates a 2D grid of points evenly spaced in [0, 1] x [0, 1]."""
  if points_per_side < 1:
    points = tf.stack(
        [tf.zeros((2,), dtype=tf.float32), tf.ones((2,), dtype=tf.float32)],
        axis=0,
    )
    return points
  if with_offset:
    offset = 1.0 / (2 * points_per_side)
  else:
    offset = 0.0
  points_one_side = tf.linspace(offset, 1 - offset, points_per_side)
  points_x = tf.tile(points_one_side[None, :], (points_per_side, 1))
  points_y = tf.tile(points_one_side[:, None], (1, points_per_side))
  points = tf.stack([points_x, points_y], axis=-1)
  points = tf.reshape(points, (-1, 2))
  return points  # (points_per_side ** 2, 2)


def boxes_to_points(boxes, grids):
  """Sample points from boxes."""
  x0, y0, x1, y1 = tf.split(boxes, 4, axis=-1)
  # [..., 4]
  boxes_xywh = tf.concat([x0, y0, x1-x0, y1-y0], axis=-1)
  # [..., 1, 4]
  boxes_xywh = tf.expand_dims(boxes_xywh, axis=-2)

  # [..., num_points, 2]
  points = boxes_xywh[...,:2] + grids * boxes_xywh[..., 2:]

  return points


@dataclasses.dataclass
class DecodeLocalizedNarrativesAnnotations:
  """Given an instance and raw labels, creates <inputs, label> pair.

  We sample the centers/traces/boxes and corresponding captions in following
  steps:
    1. Compute the relative position of each utterance
    2. Compute the relative position of each token
    3. Get token to utterance mapping
  """

  tokenizer_weight_path: str
  num_captions_per_sample: int = 1
  max_text_tokens: int = 128
  num_points_per_token: int = 2
  class_id_offset: int = 1
  box_points_per_side: int = 0
  with_image_id: bool = False
  _tokenizer: TOKENIZER = dataclasses.field(init=False)

  def __post_init__(self):
    self._tokenizer = get_tokenizer(self.tokenizer_weight_path)
    self._tokenizer.initialize()

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    captions = tf.sparse.to_dense(features['caption/string'])
    # BC
    if captions.ndim != 1:
      captions = captions[:, 0]

    utterance = tf.sparse.to_dense(features['caption/utterance'])
    center = tf.sparse.to_dense(features['caption/center'])
    bbox = tf.sparse.to_dense(features['caption/bbox'])

    inds = tf.random.shuffle(tf.range(tf.shape(captions)[0]))[
        : self.num_captions_per_sample
    ]

    captions = tf.gather(captions, inds)
    utterance = tf.gather(utterance, inds)
    center = tf.gather(center, inds)
    bbox = tf.gather(bbox, inds)

    image = tf.cast(image, tf.float32)

    # [num_caption, num_utterance]
    utterance = tf.strings.strip(utterance)

    num_caption = tf.shape(captions)[0]
    num_utterance = tf.shape(utterance)[1]

    # [num_caption, num_utterance, 2]
    center = tf.reshape(center, [num_caption, num_utterance, 2])
    # [num_caption, num_utterance, 4]
    bbox = tf.reshape(bbox, [num_caption, num_utterance, 4])

    text_tokens = self._tokenizer.string_tensor_to_indices(
        captions,
        prepend_bos=True,
        append_eos=True,
        max_num_tokens=self.max_text_tokens,
    )[:, :self.max_text_tokens]

    # ==== Start: 1.Compute utterance pos ====
    # [num_caption, num_utterance]
    valid_utt_mask = tf.cast(tf.strings.length(utterance) > 0, tf.int32)
    # [num_caption, num_utterance]
    vaid_center_mask = tf.cast(
        tf.reduce_all(center > 1e-5, axis=-1), tf.int32)
    valid_utt_mask = valid_utt_mask * vaid_center_mask
    # [num_caption, num_utterance], [0-1] pos of utterance
    sampled_utt_pos = tf.cumsum(valid_utt_mask, axis=-1) / tf.reduce_sum(
        valid_utt_mask, axis=-1, keepdims=True)
    sampled_utt_pos = tf.where(
        valid_utt_mask > 0, sampled_utt_pos, -tf.ones_like(sampled_utt_pos))
    # ==== End: 1.Compute utterance pos ====

    # ==== Start: 2.Compute token pos ====
    # [num_caption]
    sampled_utt_start_pos = tf.reduce_min(
        tf.where(
            sampled_utt_pos > 0,
            sampled_utt_pos,
            tf.ones_like(sampled_utt_pos) * INF,
        ),
        axis=-1,
    )
    sampled_utt_end_pos = tf.reduce_max(
        tf.where(
            sampled_utt_pos > 0,
            sampled_utt_pos,
            -tf.ones_like(sampled_utt_pos) * INF,
        ),
        axis=-1,
    )

    # [num_caption, max_token_length], [0-1] pos of token
    valid_token_mask = tf.cast(
        (text_tokens != self._tokenizer.pad_token)
        & (text_tokens != self._tokenizer.bos_token)
        & (text_tokens != self._tokenizer.eos_token),
        tf.int32,
    )
    token_pos = tf.cumsum(valid_token_mask, axis=-1) / tf.reduce_sum(
        valid_token_mask, axis=-1, keepdims=True
    )
    # rescale toekn pos
    # [num_caption, max_token_length]
    token_pos = sampled_utt_start_pos[:, None] + token_pos * (
        sampled_utt_end_pos[:, None] - sampled_utt_start_pos[:, None]
    )
    # ==== End: 2.Compute token pos ====

    # ==== Start: 3.Get token to utterance mapping ====
    # compute utt indices
    # [num_caption, max_token_length, num_utterance]
    token_utt_dist = (token_pos[:, :, None] - sampled_utt_pos[:, None, :]) ** 2
    # [num_caption, max_token_length]
    token2utt = tf.argmin(token_utt_dist, axis=-1)
    # ==== End: 3.Get token to utterance mapping ====

    # ==== post processing ====

    # [num_caption, max_text_tokens, 2]
    sampled_center = tf.gather(center, token2utt, batch_dims=1)
    # [num_caption, max_text_tokens, 4]
    sampled_bbox = tf.gather(bbox, token2utt, batch_dims=1)

    sampled_center *= tf.cast(valid_token_mask[..., None], tf.float32)
    sampled_bbox *= tf.cast(valid_token_mask[..., None], tf.float32)

    sampled_center = tf.clip_by_value(sampled_center, 0.0, 1.0)
    sampled_bbox = tf.clip_by_value(sampled_bbox, 0.0, 1.0)

    sampled_center = point_to_coord(
        sampled_center, tf.cast(tf.shape(image)[-3:-1], dtype=tf.int32)
    )
    sampled_bbox = bbox_to_coord(
        sampled_bbox, tf.cast(tf.shape(image)[-3:-1], dtype=tf.int32)
    )

    if self.box_points_per_side >= 0:
      # [num_captions_per_sample, max_text_tokens, num_box_points, 2]
      sampled_point = boxes_to_points(
          sampled_bbox, build_solid_grid(self.box_points_per_side)
      )
    else:
      sampled_point = sampled_center

    target = {
        'text_tokens': text_tokens,
        'points': sampled_point,
        'labels': (
            tf.zeros(tf.shape(text_tokens)[0], dtype=tf.int64)
            + self.class_id_offset
        ),
    }

    target['orig_size'] = tf.cast(tf.shape(image)[-3:-1], dtype=tf.int32)
    target['size'] = tf.identity(target['orig_size'])
    if 'image/id' in features and self.with_image_id:
      target['image/id'] = tf.io.decode_raw(
          tf.sparse.to_dense(features['image/id'])[0], out_type=tf.uint8,
          fixed_length=32
      )
      target['image/id'] = tf.cast(target['image/id'], tf.int64)
    else:
      target['image/id'] = tf.constant(0, dtype=tf.int64)
    # NOTE(jiaruixu): use padding int64 for compability with other datasets
    # target['image/id'] = tf.constant(0, dtype=tf.int64)

    return {
        'inputs': image,
        'label': target,
    }


@dataclasses.dataclass(frozen=True)
class ParseRefCoco:
  """Converts TF RefCoco annotations into the standard format."""

  refexp_field: str = 'raw'

  def __call__(
      self, data: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    example = {}
    example['image'] = tf.io.decode_jpeg(data['image'])
    example['image/id'] = data['image/id']

    example['objects'] = {}
    example['objects']['bbox'] = tf.reshape(
        data['objects/bbox'].values, [-1, 4]
    )
    example['objects']['id'] = data['objects/id'].values
    example['objects']['area'] = data['objects/area'].values
    example['objects']['label'] = data['objects/label'].values
    # example['objects']['gt_box_index'] = data['objects/gt_box_index'].values

    example['objects']['refexp'] = {}
    for field in ['refexp_id', self.refexp_field]:
      row_lengths_name = f'objects/refexp/{field}/ragged_row_lengths_0'
      flat_values_name = f'objects/refexp/{field}/ragged_flat_values'
      refexp_field_row_lengths = data[row_lengths_name].values
      refexp_field_flat_values = data[flat_values_name].values
      example['objects']['refexp'][field] = tf.RaggedTensor.from_row_lengths(
          values=refexp_field_flat_values, row_lengths=refexp_field_row_lengths
      )

    if 'objects/mask' in data:
      segmentation = data['objects/mask'].values
      height, width, _ = tf.unstack(tf.shape(example['image']))
      if tf.shape(segmentation)[0] > 0:
        segmentation = tf.map_fn(
            tf.image.decode_jpeg, segmentation, back_prop=False, dtype=tf.uint8
        )
      else:
        segmentation = tf.zeros((0,), dtype=tf.uint8)
      example['objects']['mask'] = tf.reshape(
          segmentation, [-1, height, width, 1]
      )
    return example  # pytype: disable=bad-return-type


@dataclasses.dataclass
class DecodeRefCocoAnnotations:
  """Decode RefCoco annotations."""

  tokenizer_weight_path: str
  num_captions_per_sample: int = -1
  max_text_tokens: int = 40
  caption_prefix: str = ''
  caption_suffix: str = ''
  use_text_as_context: bool = False
  max_context_tokens: int = -1
  append_context_eos: bool = False
  context_prefix: str = ''
  context_suffix: str = ''
  class_id_offset: int = 1
  refexp_field: str = 'raw'
  _tokenizer: TOKENIZER = dataclasses.field(init=False)

  def __post_init__(self):
    self._tokenizer = get_tokenizer(self.tokenizer_weight_path)
    self._tokenizer.initialize()

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    image = tf.cast(features['image'], tf.float32)
    # [num_boxes, 4]
    boxes = decode_boxes(features['objects']['bbox'], tf.shape(image)[0:2])
    masks = None
    if 'mask' in features['objects']:
      masks = features['objects']['mask']

    # [num_texts]
    text_features = features['objects']['refexp'][self.refexp_field].flat_values  # pytype: disable=attribute-error  # allow-recursive-types
    refexp_ids = features['objects']['refexp']['refexp_id'].flat_values  # pytype: disable=attribute-error  # allow-recursive-types
    row_ids = features['objects']['refexp']['refexp_id'].value_rowids()  # pytype: disable=attribute-error  # allow-recursive-types
    # [num_texts, 4]
    boxes = tf.gather(boxes, row_ids)
    if masks is not None:
      masks = tf.gather(masks, row_ids)

    if self.num_captions_per_sample > 0:
      inds = tf.random.shuffle(tf.range(tf.shape(text_features)[0]))[
          : self.num_captions_per_sample
      ]
      text_features = tf.gather(text_features, inds)
      boxes = tf.gather(boxes, inds)
      refexp_ids = tf.gather(refexp_ids, inds)
      if masks is not None:
        masks = tf.gather(masks, inds)

    if self.caption_prefix:
      text_features = (
          tf.constant(self.caption_prefix, dtype=tf.string)[None]
          + text_features
      )
    if self.caption_suffix:
      text_features = (
          text_features
          + tf.constant(self.caption_suffix, dtype=tf.string)[None]
      )

    text_tokens = self._tokenizer.string_tensor_to_indices(
        text_features,
        prepend_bos=True,
        append_eos=True,
        max_num_tokens=self.max_text_tokens,
    )[:, : self.max_text_tokens]

    target = {
        'boxes': boxes,
        'text_tokens': text_tokens,
        'labels': (
            tf.zeros(tf.shape(boxes)[0], dtype=tf.int64) + self.class_id_offset
        ),
        'refexp_ids': refexp_ids,
    }

    if self.use_text_as_context:
      max_context_tokens = self.max_context_tokens if (
          self.max_context_tokens > 0) else self.max_text_tokens
      context = text_features
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
      target['context_tokens'] = context_tokens

    if masks is not None:
      target['masks'] = masks

    target['orig_size'] = tf.cast(tf.shape(image)[-3:-1], dtype=tf.int32)
    target['size'] = tf.identity(target['orig_size'])
    target['image/id'] = features['image/id']

    output = {
        'inputs': image,
        'label': target,
    }

    return output


@dataclasses.dataclass(frozen=True)
class AddPromptBoxes:
  """Add prompt boxes."""
  num_prompts: int = -1
  zero_boxes: bool = False
  use_gt_prompt: bool = False

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    image_size = features['label']['size']
    if self.num_prompts < 0:
      num_prompts = tf.shape(features['label']['text_tokens'])[0]
    else:
      # use during inference
      num_prompts = self.num_prompts
    if self.use_gt_prompt:
      # location conditioned captioning
      prompt_boxes = features['label']['boxes']
    else:
      h, w = tf.split(tf.cast(image_size, tf.float32), 2, axis=-1)
      prompt_boxes = tf.concat(
          [tf.zeros((2,), dtype=tf.float32), w, h], axis=-1
      )
      if self.zero_boxes:
        prompt_boxes = tf.zeros_like(prompt_boxes)
      prompt_boxes = tf.tile(prompt_boxes[None], [num_prompts, 1])
    features['label']['prompt_boxes'] = prompt_boxes

    return features


@dataclasses.dataclass(frozen=True)
class AddPromptPoints:
  """Add prompt boxes."""
  num_prompts: int = -1
  num_points_per_prompt: int = 4

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    if self.num_prompts < 0:
      num_prompts = tf.shape(features['label']['text_tokens'])[0]
    else:
      # use during inference
      num_prompts = self.num_prompts

    features['label']['prompt_points'] = tf.zeros(
        (num_prompts, self.num_points_per_prompt, 2)
    )

    return features


@dataclasses.dataclass(frozen=True)
class AddTaskMask:
  """Add task mask."""

  tasks: Sequence[str]

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    cap_loss_valid_mask = 0
    proposal_loss_valid_mask = 0
    objcap_loss_valid_mask = 0
    point_loss_valid_mask = 0
    for task in self.tasks:
      if task == 'caption':
        cap_loss_valid_mask = 1
      elif task == 'detection':
        proposal_loss_valid_mask = 1
        objcap_loss_valid_mask = 1
      elif task == 'point':
        point_loss_valid_mask = 1
      elif task == 'proposal':
        proposal_loss_valid_mask = 1
      elif task == 'object_caption':
        objcap_loss_valid_mask = 1
      else:
        raise ValueError(f'Unsupported task: {task}')
    features['label']['cap_loss_valid_mask'] = cap_loss_valid_mask
    features['label']['proposal_loss_valid_mask'] = proposal_loss_valid_mask
    features['label']['objcap_loss_valid_mask'] = objcap_loss_valid_mask
    features['label']['point_loss_valid_mask'] = point_loss_valid_mask

    return features


def split_string(input_str):
  """Split string."""
  # Tokenize the string into words
  words = tf.strings.split([input_str], sep=' ').values

  # Find the total number of words
  num_words = tf.shape(words)[0]

  # Determine the split point,
  # ensuring the second part's length is greater than half
  split_point = tf.random.uniform(
      (), minval=0, maxval=num_words // 2, dtype=tf.int32
  )

  # Split the words into two parts based on the split point
  first_part = words[:split_point]
  second_part = words[split_point:]

  # Join the tokenized words back into strings
  first_str = tf.strings.reduce_join(first_part, separator=' ')
  second_str = tf.strings.reduce_join(second_part, separator=' ')

  return first_str, second_str


@dataclasses.dataclass(frozen=True)
class SplitText:
  """Split context."""

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    features_new = features.copy()
    text_features = features['captions']['text']
    assert 'context' not in features
    left_text, right_text = split_string(text_features[0])
    features_new['context'] = left_text[None]
    features_new['captions']['text'] = right_text[None]

    return features_new


@dataclasses.dataclass(frozen=True)
class ParseVg:
  """Parse VG annotations."""

  def __call__(
      self, data: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    example = {}
    example['image'] = tf.io.decode_jpeg(data['image'])
    example['image/id'] = data['img_id']

    example['objects'] = {}
    example['objects']['bbox'] = tf.reshape(
        data['regions/bbox'].values, [-1, 4]
    )
    example['objects']['phrase'] = data['regions/phrase'].values

    return example


@dataclasses.dataclass
class DecodeVgAnnotations:
  """Decode VG annotations."""

  tokenizer_weight_path: str
  max_text_tokens: int = 40
  num_captions_per_sample: int = -1
  caption_prefix: str = ''
  caption_suffix: str = ''
  class_id_offset: int = 1
  _tokenizer: TOKENIZER = dataclasses.field(init=False)

  def __post_init__(self):
    self._tokenizer = get_tokenizer(self.tokenizer_weight_path)
    self._tokenizer.initialize()

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    image = tf.cast(features['image'], tf.float32)
    # [num_boxes, 4]
    boxes = decode_boxes(features['objects']['bbox'], tf.shape(image)[0:2])

    # [num_boxes]
    text_features = features['objects']['phrase']

    if self.caption_prefix:
      text_features = (
          tf.constant(self.caption_prefix, dtype=tf.string)[None]
          + text_features
      )
    if self.caption_suffix:
      text_features = (
          text_features
          + tf.constant(self.caption_suffix, dtype=tf.string)[None]
      )

    text_tokens = self._tokenizer.string_tensor_to_indices(
        text_features,
        prepend_bos=True,
        append_eos=True,
        max_num_tokens=self.max_text_tokens,
    )[:, : self.max_text_tokens]

    if self.num_captions_per_sample > 0:
      inds = tf.random.shuffle(tf.range(tf.shape(text_tokens)[0]))[
          : self.num_captions_per_sample
      ]
      text_tokens = tf.gather(text_tokens, inds)
      boxes = tf.gather(boxes, inds)

    target = {
        'boxes': boxes,
        'text_tokens': text_tokens,
        'labels': (
            tf.zeros(tf.shape(boxes)[0], dtype=tf.int64) + self.class_id_offset
        ),
    }

    target['orig_size'] = tf.cast(tf.shape(image)[-3:-1], dtype=tf.int32)
    target['size'] = tf.identity(target['orig_size'])
    target['image/id'] = features['image/id']

    output = {
        'inputs': image,
        'label': target,
    }

    return output


@dataclasses.dataclass(frozen=True)
class ParseLlava:
  """Parse LLaVA annotations."""

  def __call__(self, data: preprocess_spec.Features):
    example = {}
    example['image'] = tf.io.decode_jpeg(data['image/encoded'], channels=3)
    # example['image/id'] = data['image/id']

    example['context'] = data['conversations/human'].values
    example['context'] = tf.strings.regex_replace(
        example['context'], '<image>\n', ''
    )
    example['context'] = tf.strings.regex_replace(
        example['context'], '\n<image>', ''
    )
    example['captions'] = {}
    example['captions']['text'] = data['conversations/agent'].values

    return example
