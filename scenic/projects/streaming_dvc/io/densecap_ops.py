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

"""Preprocessing operations for dense video captioning."""

import dataclasses
from typing import Union

from clu import preprocess_spec
from dmvr import tokenizers

from scenic.projects.t5 import tokenizer as t5_tokenizer
from scenic.projects.vid2seq import data_utils as vid2seq_data_utils
import tensorflow as tf

BERT_VOCAB_SIZE = 30522
SP_VOCAB_SIZE = 32128
TOKENIZER = Union[tokenizers.BertTokenizer, t5_tokenizer.SentencePieceTokenizer]


def remove_padding_and_concat_and_pad_tokens(
    tokens, bos_id, eos_id, max_text_tokens, padding_id=0):
  """Remove padding and concat and pad tokens.

    Removing padding tokens at the end of each caption, concat them into
    a single paragraph caption, and add paddings to the paragraph captions.

  Example: bos=101, eos=102, padding_id=0, max_text_tokens=8, tokens being
    [[101, 1, 2, 102, 0, 0, 0, 0],
     [101, 3, 102, 0, 0, 0, 0, 0]]
  The output should be:
    [[101, 1, 2, 3, 102, 0, 0, 0]]

  Args:
    tokens: (num_captions, max_single_text_tokens). The tokens should all have
      bos, but no eos.
    bos_id: int
    eos_id: int
    max_text_tokens: int
    padding_id: int
  Returns:
    merged_token: (1, max_text_tokens)
  """
  merged_token = tf.reshape(
      tokens, [-1])  # (num_captions * max_single_text_tokens,)
  merged_token = tf.boolean_mask(
      merged_token,
      tf.not_equal(merged_token, padding_id))  # (num_remaining_tokens,)
  merged_token = tf.boolean_mask(
      merged_token, tf.not_equal(merged_token, bos_id))[:max_text_tokens - 2]
  # (<self.max_text_tokens - 2,) so that we can add EOS and BOS.

  merged_token = tf.concat(
      [[bos_id], merged_token, [eos_id]], axis=0)  # (<self.max_text_tokens,)
  merged_token = tf.pad(
      merged_token, [[0, max_text_tokens - tf.shape(merged_token)[0]]],
      constant_values=padding_id,
  )  # (self.max_text_tokens,)
  return merged_token[None]  # (1, self.max_text_tokens,)


def create_caption_and_time_tokens(
    text_features, start, end, duration, tokenizer,
    num_bins, original_vocab_size, max_single_text_tokens,
    min_caption_tokens=-1, min_segment_duration=-1):
  """Tokenize strings and create time tokens."""
  timestamp_token = vid2seq_data_utils.timestampify(
      start=start,
      end=end,
      duration=duration,
      abs_time_token=False,
      num_bins=num_bins,
      vocabulary_size=original_vocab_size,
      time_format='se')  # (num_captions, 2)
  caption_token = tokenizer.string_tensor_to_indices(
      text_features,
      prepend_bos=True,
      append_eos=False,
      max_num_tokens=max_single_text_tokens,
  )[:, :max_single_text_tokens]  # (num_captions, max_single_text_tokens)
  # Remove segments with very few words. These are likely meaningless words,
  # for example, welcome, OK, etc.
  if min_caption_tokens > 0:
    keep = tf.reduce_sum(
        tf.cast(caption_token > 0, tf.int32),
        axis=1) > min_caption_tokens
    caption_token = tf.boolean_mask(caption_token, keep)
    timestamp_token = tf.boolean_mask(timestamp_token, keep)
  # Remove text with too small duration.
  if min_segment_duration > 0:
    segment_duration = timestamp_token[:, 1] - timestamp_token[:, 0]
    keep = segment_duration >= min_segment_duration
    caption_token = tf.boolean_mask(caption_token, keep)
    timestamp_token = tf.boolean_mask(timestamp_token, keep)
  return caption_token, timestamp_token


def crop_and_remove_segments(
    segment_start, segment_end, captions, video_start, video_end):
  """Crop segments and remove out-of-crop ones."""
  video_length = video_end - video_start
  cropped_start = tf.maximum(segment_start - video_start, 0)
  cropped_end = tf.minimum(segment_end - video_start, video_length)
  valid = tf.logical_and(
      tf.greater_equal(cropped_end, 0),
      tf.less_equal(cropped_start, video_length))
  keep = tf.where(valid)[:, 0]
  new_start = tf.gather(cropped_start, keep)
  new_end = tf.gather(cropped_end, keep)
  new_captions = tf.gather(captions, keep)
  return new_start, new_end, new_captions


@dataclasses.dataclass(frozen=True)
class DecodeAndSubsampleDensecapVideo:
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
    localization_keys: dict of strings to their mapped names.
    media_id_key: str
    recompute_duration: bool; make duration align with sampled frames.
    zero_pad_frames: bool; if True, pad 0 when the videos are shorter than
      num_frames; if False, do linespace in between to fill num_frames.
    min_crop_ratio: float; avoid very aggressive cropping.
    only_crop_at_end: bool;
    crop_prob: float; Probability to apply crop augmentation.
    with_clip_embeddings: bool; If True, we load pre-computed image features.
  """

  num_frames: int
  is_train: bool
  caption_field: str = 'caption/string'
  context_field: str = ''
  localization_keys: dict[str, str] = dataclasses.field(
      default_factory=lambda: {  # pylint:disable=g-long-lambda
          'video/timestamps/start': 'video/timestamps/start',
          'video/timestamps/end': 'video/timestamps/end',
          'video/duration': 'video/duration'})
  media_id_key: str = 'media_id'
  recompute_duration: bool = False
  zero_pad_frames: bool = True
  min_crop_ratio: float = 1.0
  only_crop_at_end: bool = False
  crop_prob: float = 0.0
  with_clip_embeddings: bool = False

  def __call__(
      self, features: preprocess_spec.Features) -> preprocess_spec.Features:

    frames = features['image/encoded']
    max_frames = tf.shape(frames)[0]
    if self.is_train:
      apply_crop = tf.cast(
          tf.random.uniform([]) < self.crop_prob, tf.float32)
      # crop_ratio == 1.0 means not crop.
      crop_ratio = self.min_crop_ratio * apply_crop + 1. * (1. - apply_crop)
      min_crop_frames = tf.maximum(
          tf.cast(tf.cast(max_frames, tf.float32) * crop_ratio,
                  tf.int32), self.num_frames)
      st = tf.random.uniform(
          [], maxval=tf.maximum(max_frames - min_crop_frames, 1),
          dtype=tf.int32) if not self.only_crop_at_end else 0
      ed = tf.random.uniform(
          [], minval=tf.minimum(st + min_crop_frames, max_frames - 1),
          maxval=max_frames, dtype=tf.int32)
      if self.zero_pad_frames:
        stride = tf.maximum((ed - st + 1) // self.num_frames, 1)
        inds = tf.range(tf.minimum(self.num_frames, ed - st + 1)) * stride + st
      else:
        inds = tf.cast(tf.linspace(st, ed, self.num_frames), tf.int32)
    else:
      if self.zero_pad_frames:
        stride = tf.maximum(max_frames // self.num_frames, 1)
        inds = tf.range(tf.minimum(self.num_frames, max_frames)) * stride
      else:
        inds = tf.cast(
            tf.linspace(0, max_frames - 1, self.num_frames), tf.int32)
    frames = tf.gather(frames, inds)
    frames = tf.map_fn(
        lambda x: tf.image.decode_jpeg(x, channels=3),
        frames, back_prop=False, dtype=tf.uint8)

    if self.zero_pad_frames:
      frames = tf.pad(
          frames,
          [[0, self.num_frames - tf.shape(frames)[0]], [0, 0], [0, 0], [0, 0]])

    features_new = {
        'image': frames,
        'captions': {
            'text': tf.sparse.to_dense(features[self.caption_field])},
        'image/id': tf.zeros((), tf.int32),
    }
    features_new['media_id'] = tf.io.decode_raw(
        tf.sparse.to_dense(features[self.media_id_key]), out_type=tf.uint8)[0]

    if self.context_field:
      features_new['context'] = tf.sparse.to_dense(features[self.context_field])

    for k, v in self.localization_keys.items():
      features_new[v] = tf.sparse.to_dense(features[k])

    if 'video/duration' not in features_new:  # This is only needed for YTT
      features_new['video/duration'] = (
          features['image/timestamp'][-1] - features['image/timestamp'][0]
          )[None]
    features_new['video/original_duration'] = features_new['video/duration']

    if self.recompute_duration or self.is_train:
      # filter-out out-of-crop segments
      image_timestamp = features['image/timestamp']
      sampled_timestamp = tf.gather(image_timestamp, inds)
      video_start = sampled_timestamp[0]
      video_end = sampled_timestamp[-1]
      new_duration = video_end - video_start
      segment_start_original = features_new['video/timestamps/start']
      segment_end_original = features_new['video/timestamps/end']
      valid = tf.logical_and(
          tf.greater_equal(segment_end_original, video_start),
          tf.less_equal(segment_start_original, video_end))
      keep = tf.where(valid)[:, 0]
      segment_start = tf.maximum(segment_start_original - video_start, 0)
      segment_end = tf.minimum(
          segment_end_original - video_start, new_duration)
      features_new['video/timestamps/start'] = tf.gather(segment_start, keep)
      features_new['video/timestamps/end'] = tf.gather(segment_end, keep)
      features_new['captions']['text'] = tf.gather(
          features_new['captions']['text'], keep)
      features_new['video/duration'] = new_duration[None]

    if self.with_clip_embeddings:
      clip_feature_key = 'image/clip_embeddings'
      clip_embeddings = features[clip_feature_key]
      sampled_clip_embeddings = tf.gather(clip_embeddings, inds)
      features_new[clip_feature_key] = sampled_clip_embeddings

    if preprocess_spec.SEED_KEY in features:
      features_new[preprocess_spec.SEED_KEY] = features[
          preprocess_spec.SEED_KEY]
    return features_new


@dataclasses.dataclass
class DecodeActivityNetDenseCaptionAnnotations:
  """Decode ActivityNet-Dense caption annotations."""

  tokenizer_weight_path: str
  max_text_tokens: int = 1024
  max_single_text_tokens: int = 128
  num_bins: int = 100
  with_clip_embeddings: bool = False
  min_caption_tokens: int = -1
  min_segment_duration: bool = False
  _tokenizer: TOKENIZER = dataclasses.field(init=False)
  _original_vocab_size: int = -1
  _bos_id: int = 0
  _eos_id: int = 1

  def __post_init__(self):
    if self.tokenizer_weight_path == 't5':
      self._tokenizer = t5_tokenizer.build_dmvr_sp_model()
      self._bos_id = 0
      self._eos_id = 1
      # NOTE: We can't use self._tokenizer.vocab_size here.
      # self._tokenizer.vocab_size of the T5 tokenizer is 32100, instead of
      # model weight shape 32128.
      self._original_vocab_size = SP_VOCAB_SIZE
    else:
      self._tokenizer = tokenizers.BertTokenizer(self.tokenizer_weight_path)
      self._bos_id = 101
      self._eos_id = 102
      self._original_vocab_size = BERT_VOCAB_SIZE
    self._tokenizer.initialize()

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    image = tf.cast(features['image'], tf.float32)
    text_features = features['captions']['text']  # (num_captions,)

    start = features['video/timestamps/start']
    end = features['video/timestamps/end']
    duration = features['video/duration']
    caption_token, timestamp_token = create_caption_and_time_tokens(
        text_features, start, end, duration, self._tokenizer,
        num_bins=self.num_bins,
        original_vocab_size=self._original_vocab_size,
        max_single_text_tokens=self.max_single_text_tokens,
        min_caption_tokens=self.min_caption_tokens,
        min_segment_duration=self.min_segment_duration)
    merged_token = vid2seq_data_utils.merge_cap_time_tokens(
        caption_token, timestamp_token, order='ld',
    )  # (num_captions, max_single_text_tokens)
    text_tokens = remove_padding_and_concat_and_pad_tokens(
        merged_token, self._bos_id, self._eos_id,
        self.max_text_tokens)  # (1, self.max_text_tokens)
    target = {'text_tokens': text_tokens}

    target['orig_size'] = tf.cast(tf.shape(image)[-3:-1], dtype=tf.int32)
    target['size'] = tf.identity(target['orig_size'])
    target['image/id'] = features['image/id']
    target['media_id'] = features['media_id']
    target['duration'] = duration  # Used for recovering the original timestamp.
    target['original_duration'] = features['video/original_duration']

    output = {
        'inputs': image,
        'label': target,
    }

    if self.with_clip_embeddings:
      output['image_features'] = features['image/clip_embeddings']

    if preprocess_spec.SEED_KEY in features:
      output[preprocess_spec.SEED_KEY] = features[preprocess_spec.SEED_KEY]
    return output


@dataclasses.dataclass
class DecodeActivityNetDenseCaptionAnnotationsDenseOutputsAugContext:
  """Decode dense with intermediate supervision whose locations are given."""

  is_train: bool
  num_dense_outputs: int
  tokenizer_weight_path: str
  max_text_tokens: int = 1024
  max_single_text_tokens: int = 128
  num_bins: int = 100
  early_segments_as_context: bool = False
  normalize_early_timestamps: bool = False
  context_mask_ratio: float = 0.0
  no_timestamp_in_context: bool = False
  dynamic_location: bool = False
  only_use_augmented_context: bool = False
  continuous_random_mask: bool = False
  with_clip_embeddings: bool = False
  _tokenizer: TOKENIZER = dataclasses.field(init=False)
  _original_vocab_size: int = -1
  _bos_id: int = 0
  _eos_id: int = 1

  def __post_init__(self):
    if self.tokenizer_weight_path == 't5':
      self._tokenizer = t5_tokenizer.build_dmvr_sp_model()
      self._bos_id = 0
      self._eos_id = 1
      # NOTE: We can't use self._tokenizer.vocab_size here.
      # self._tokenizer.vocab_size of the T5 tokenizer is 32100, instead of
      # model weight shape 32128.
      self._original_vocab_size = SP_VOCAB_SIZE
    else:
      self._tokenizer = tokenizers.BertTokenizer(self.tokenizer_weight_path)
      self._bos_id = 101
      self._eos_id = 102
      self._original_vocab_size = BERT_VOCAB_SIZE
    self._tokenizer.initialize()

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    image = tf.cast(features['image'], tf.float32)
    text_features = features['captions']['text']  # (num_captions,)

    start = features['video/timestamps/start']
    end = features['video/timestamps/end']
    duration = features['video/duration']
    timestamp_token = vid2seq_data_utils.timestampify(
        start=start,
        end=end,
        duration=duration,
        abs_time_token=False,
        num_bins=self.num_bins,
        vocabulary_size=self._original_vocab_size,
        time_format='se')
    # (num_captions, 2). Value in [0, num_bins] with respect to the whole video.

    caption_token = self._tokenizer.string_tensor_to_indices(
        text_features,
        prepend_bos=True,
        append_eos=False,
        max_num_tokens=self.max_single_text_tokens,
    )[:, :self.max_single_text_tokens]  # (num_captions, max_single_text_tokens)
    merged_token = vid2seq_data_utils.merge_cap_time_tokens(
        caption_token, timestamp_token, order='ld',
    )  # (num_captions, max_single_text_tokens), value might change below

    num_frames = tf.shape(image)[0]
    tf.debugging.assert_equal(
        num_frames, self.num_bins,
        f'num frames {num_frames} should be {self.num_bins}.')
    checkpoint_size = (self.num_bins - 1) // self.num_dense_outputs + 1
    dense_output_tokens = []
    context_tokens = []
    checkpoint_inds = []

    # We still uniformly sample checkpoint locations, as they are supposed to
    # be unknown at testing. Now this number of checkpoints can be different
    # from testing though.
    for i in range(self.num_dense_outputs):
      end_idx = timestamp_token[:, 1] - self._original_vocab_size
      if self.dynamic_location:
        checkpoint_end_time = tf.random.uniform(
            (), minval=1, maxval=num_frames + 1, dtype=tf.int32)
        if i == self.num_dense_outputs - 1:
          # We want at least one checkpoint to see all the segments, as many
          # segments in the annotation ends at the last frame.
          checkpoint_end_time = num_frames
        checkpoint_start_time = 0
      else:
        checkpoint_end_time = (i + 1) * checkpoint_size
        checkpoint_start_time = i * checkpoint_size

      if self.normalize_early_timestamps:
        normalized_timestamp_token = self._original_vocab_size + ((
            timestamp_token - self._original_vocab_size) * (
                self.num_bins)) // checkpoint_end_time
        normalized_timestamp_token = tf.minimum(
            normalized_timestamp_token,
            self._original_vocab_size + self.num_bins - 1)  # Avoid nan
        merged_token = vid2seq_data_utils.merge_cap_time_tokens(
            caption_token, normalized_timestamp_token, order='ld',
        )  # (num_captions, max_single_text_tokens)

      # Following shapes are (num_captions,)
      finished_before_checkpoint_ends = end_idx < checkpoint_end_time
      finished_within_checkpoint_range = tf.logical_and(
          finished_before_checkpoint_ends, end_idx >= checkpoint_start_time)
      finished_before_checkpoint_starts = end_idx < checkpoint_start_time

      context_source = merged_token if (
          not self.no_timestamp_in_context) else caption_token

      # The original context and their supervision.
      full_context = tf.boolean_mask(
          context_source, finished_before_checkpoint_starts)
      full_context = remove_padding_and_concat_and_pad_tokens(
          full_context, self._bos_id, self._eos_id, self.max_text_tokens)

      text_tokens_with_full_context = tf.boolean_mask(
          merged_token, finished_within_checkpoint_range)
      text_tokens_with_full_context = remove_padding_and_concat_and_pad_tokens(
          text_tokens_with_full_context,
          self._bos_id, self._eos_id, self.max_text_tokens)

      # Augmented context and their supervision, by masking out valid contexts.
      if self.continuous_random_mask:
        max_valid_segment_end_time = tf.reduce_max(tf.boolean_mask(
            end_idx, finished_before_checkpoint_ends))
        maxval = max_valid_segment_end_time
        maxval = tf.maximum(2, maxval)
        checkpoint_start_time = tf.random.uniform(
            (), minval=1, maxval=maxval, dtype=tf.int32)
        random_mask = end_idx < checkpoint_start_time
      else:
        random_mask = tf.cast(tf.random.uniform(
            (tf.shape(end_idx)[0],), minval=0, maxval=1,
            dtype=tf.float32) < self.context_mask_ratio, tf.bool)

      if self.dynamic_location:
        # When dynamic location is on, the "past range" is 0-0, and we don't
        # have any context by default. So changing it to masking any valid
        # "current" segments.
        augmented_context_mask = tf.logical_and(
            finished_before_checkpoint_ends, random_mask)
      else:
        augmented_context_mask = tf.logical_and(
            finished_before_checkpoint_starts, random_mask)
      augmented_context = tf.boolean_mask(
          context_source, augmented_context_mask)
      augmented_context = remove_padding_and_concat_and_pad_tokens(
          augmented_context, self._bos_id, self._eos_id, self.max_text_tokens)

      augmented_text_token_mask = tf.logical_and(
          finished_before_checkpoint_ends,
          tf.logical_not(augmented_context_mask))
      text_tokens_with_augmented_context = tf.boolean_mask(
          merged_token, augmented_text_token_mask)
      text_tokens_with_augmented_context = (
          remove_padding_and_concat_and_pad_tokens(
              text_tokens_with_augmented_context,
              self._bos_id, self._eos_id, self.max_text_tokens))

      # Get the actual frame index. This is in range [0, num_frames).
      checkpoint_ind = tf.broadcast_to(
          tf.minimum(checkpoint_end_time, num_frames) - 1, (1,))
      # We train on both the original contexts and the augmented ones.
      if self.only_use_augmented_context:
        all_context = augmented_context
        all_text_tokens = text_tokens_with_augmented_context
      else:
        all_context = tf.concat([full_context, augmented_context], axis=0)
        all_text_tokens = tf.concat(
            [text_tokens_with_full_context, text_tokens_with_augmented_context],
            axis=0)
        # 2 means the location for both original and augmented target
        checkpoint_ind = tf.broadcast_to(checkpoint_ind, (2,))

      context_tokens.append(all_context)
      dense_output_tokens.append(all_text_tokens)
      checkpoint_inds.append(checkpoint_ind)

    dense_output_tokens = tf.concat(
        dense_output_tokens, axis=0)  # (num_dense_outputs * 2, max_text_tokens)
    target = {
        'text_tokens': dense_output_tokens,
    }
    if self.is_train:
      checkpoint_inds = tf.concat(
          checkpoint_inds, axis=0)  # (num_dense_outputs * 2,)
      target['checkpoint_inds'] = checkpoint_inds
      context_tokens = tf.concat(
          context_tokens, axis=0)  # (num_dense_outputs * 2, max_text_tokens)
      target['context_tokens'] = context_tokens

    target['orig_size'] = tf.cast(tf.shape(image)[-3:-1], dtype=tf.int32)
    target['size'] = tf.identity(target['orig_size'])
    target['image/id'] = features['image/id']
    target['media_id'] = features['media_id']
    target['duration'] = duration  # Used for recovering the original timestamp.
    target['original_duration'] = features['video/original_duration']

    output = {
        'inputs': image,
        'label': target,
    }
    if self.with_clip_embeddings:
      output['image_features'] = features['image/clip_embeddings']

    if preprocess_spec.SEED_KEY in features:
      output[preprocess_spec.SEED_KEY] = features[preprocess_spec.SEED_KEY]
    return output
