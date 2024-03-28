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

"""Utils for dense video captioning data loading."""

import functools
from typing import Any, Mapping, Optional, Union

from dmvr import builders
from dmvr import processors
from dmvr import tokenizers
import gin
from t5.data.preprocessors import DenoiseInputsFn
from t5.data.preprocessors import DenoiseNoiseMaskFn
from t5.data.preprocessors import DenoiseTargetsFn
from t5.data.preprocessors import random_spans_noise_mask
import tensorflow as tf


PyTree = Union[Mapping[str, Mapping], Any]
FeatureType = Mapping[str, tf.Tensor]


def timestampify(start: tf.Tensor,
                 end: tf.Tensor,
                 duration: tf.Tensor,
                 abs_time_token: bool,
                 num_bins: int,
                 vocabulary_size: int,
                 time_format: str,
                 t: int = 1000000):
  """Tokenizes timestamps.

  Args:
    start: Start times of the events.
    end: End times of the events.
    duration: Duration of the video.
    abs_time_token: Whether to use absolute (vs relative) time tokens.
    num_bins: Number of quantization bins for time tokens.
    vocabulary_size: Number of text tokens.
    time_format: st for start-end or cd for center-duration.
    t: FPS * 1000000

  Returns:
    Tensor of start+end time tokens for each event.
  """

  if time_format == 'cd':
    timestamp = tf.stack([(start + end) // 2, end - start], axis=1)
  else:
    timestamp = tf.stack([start, end], axis=1)
  timestamp = tf.minimum(timestamp, duration)

  if not abs_time_token:  # relative time token
    max_offset = tf.cast(num_bins - 1, tf.float64)
    rel_timestamp = tf.math.divide(timestamp, duration)
    timestamp_token = tf.math.add(tf.math.multiply(rel_timestamp, max_offset),
                                  vocabulary_size)
    timestamp_token = tf.cast(timestamp_token, tf.int32)

  else:  # absolute time token
    timestamp_token = tf.math.add(
        tf.cast(timestamp / t, tf.int32), vocabulary_size)

  return timestamp_token


def merge_cap_time_tokens(caption_tokens, timestamp_token, order):
  """Merge tensors of time and text tokens into a single tensor.

  Args:
    caption_tokens: Tensor of text tokens for each event.
    timestamp_token: Tensor of time tokens for each event.
    order: ld for time tokens first, dl for text tokens first.

  Returns:
    Tensor of text and time tokens for each event.
  """

  if order == 'ld':
    seq = tf.concat(
        [
            caption_tokens[:, :1],  # BOS
            timestamp_token,  # timestamp
            caption_tokens[:, 1:-2]
        ],  # caption
        axis=1)
  else:
    seq = tf.concat(
        [
            caption_tokens[:, :-2],  # caption
            timestamp_token
        ],  # timestamp
        axis=1)
  return seq


@gin.configurable()
def sentinel_id(vocabulary_size, return_value=None):
  """T5-style preprocessing."""
  if return_value is not None:
    return return_value
  return vocabulary_size - 1


@gin.configurable()
def noise_span_to_unique_sentinel(tokens, noise_mask, vocabulary_size, seeds):
  """T5-style preprocessing."""
  del seeds

  prev_token_is_noise = tf.pad(noise_mask[:-1], [[1, 0]])

  first_noise_tokens = tf.logical_and(
      noise_mask, tf.logical_not(prev_token_is_noise))
  subsequent_noise_tokens = tf.logical_and(noise_mask, prev_token_is_noise)

  sentinel = sentinel_id(vocabulary_size) + 1 - tf.cumsum(
      tf.cast(first_noise_tokens, tokens.dtype))

  tokens = tf.where(first_noise_tokens, sentinel, tokens)
  return tf.boolean_mask(tokens, tf.logical_not(subsequent_noise_tokens))


@gin.configurable()
def nonnoise_span_to_unique_sentinel(tokens, noise_mask, vocabulary_size,
                                     seeds):
  """T5-style preprocessing."""
  return noise_span_to_unique_sentinel(tokens, tf.logical_not(noise_mask),
                                       vocabulary_size, seeds)


def single_example_denoise(tokens: tf.Tensor, vocabulary_size: int,
                           noise_density: float,
                           noise_mask_fn: DenoiseNoiseMaskFn,
                           inputs_fn: DenoiseInputsFn,
                           targets_fn: DenoiseTargetsFn) -> FeatureType:
  """T5-style preprocessing."""
  seed = tf.random.uniform([2], minval=0, maxval=2**16, dtype=tf.dtypes.int32,
                           seed=None, name=None)
  seeds = tf.unstack(tf.random.experimental.stateless_split(seed, 6))
  noise_mask = noise_mask_fn(tf.size(tokens), noise_density, seeds=seeds[:2])
  inputs = inputs_fn(tokens, noise_mask, vocabulary_size, seeds=seeds[2:4])
  targets = targets_fn(tokens, noise_mask, vocabulary_size, seeds=seeds[4:6])
  return {
      'inputs': inputs,
      'outputs': targets,
  }


def add_text_with_timestamps(
    parser_builder: builders.BaseParserBuilder,
    preprocessor_builder: builders.PreprocessorBuilder,
    tokenizer: tokenizers.TextTokenizer,
    input_feature_name: str = 'caption/string',
    input_timestamp_start_name: str = 'caption/timestamp/start',
    input_timestamp_end_name: str = 'caption/timestamp/end',
    input_duration_name: str = 'video/duration',
    output_raw_timestamp_name: str = 'timestamp',
    output_raw_duration_name: str = 'duration',
    max_events: int = 50,
    vocabulary_size: int = 32128,
    num_bins: int = 100,
    output_raw_string_name: str = builders.TEXT_FEATURE_NAME,
    output_feature_name: str = builders.TEXT_INDICES_FEATURE_NAME,
    prepend_bos: bool = True,
    append_eos: bool = True,
    keep_raw_string: bool = False,
    max_num_tokens: Optional[int] = 128,
    abs_time_token: bool = False,
    time_format: str = 'se',
    order: str = 'ld',
    notime: bool = False,
    asr_input: bool = True,
    max_num_input_words: int = 512,
    asr_raw_string_name: str = 'ASR/string',
    asr_timestamp_name: str = 'ASR/timestamps',
    corrupt: float = 0.,
    span_len: float = 3.0,
    tmp_only: bool = False,
    asr_notime: bool = False,
    t: int = 1000000):
  """Prepares data for Vid2Seq model.

  Args:
    parser_builder: DMVR Parser builder.
    preprocessor_builder: DMVR Preprocessor builder.
    tokenizer: Text tokenizer.
    input_feature_name: Field name for the captions.
    input_timestamp_start_name: Field name for the start timestamps.
    input_timestamp_end_name: Field name for the end timestamps.
    input_duration_name: Field name for video duration.
    output_raw_timestamp_name: Output key for the timestamps.
    output_raw_duration_name: Output key for the duration.
    max_events: Maximum number of events to consider.
    vocabulary_size: Number of tokens of the text tokenizer.
    num_bins: Number of quantization bins for time tokens.
    output_raw_string_name: Output key for the captions.
    output_feature_name: Output key for the caption tokens.
    prepend_bos: Whether to put BOS at the start of the sequences.
    append_eos: Whether to add EOS at the end of the sequences.
    keep_raw_string: Whether to keep raw string in batch.
    max_num_tokens: Maximum number of tokens for sequences.
    abs_time_token: Whether to use absolute (vs relative) time tokens.
    time_format: st for start-end or cd for center-duration.
    order: ld for time tokens first, dl for text tokens first.
    notime: Whether to use time tokens.
    asr_input: Whether to use ASR as input.
    max_num_input_words: Maximum number of tokens in ASR input.
    asr_raw_string_name: Field name for input ASR text.
    asr_timestamp_name: Field name for input ASR timestamps.
    corrupt: Ratio of corruption for T5-style corrupted sequence.
    span_len: Average length of corrupted spans for T5-style corrupted sequence.
    tmp_only: Localization only mode.
    asr_notime: Whether to use time tokens for ASR input.
    t: FPS * 1000000

  Returns:
    Nothing, modifies DMVR builder inplace.
  """

  # Parse text indices.
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.VarLenFeature(dtype=tf.string),
        output_name=output_raw_string_name,
        is_context=True)
  elif isinstance(parser_builder, builders.ExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.VarLenFeature(dtype=tf.string),
        output_name=output_raw_string_name)

  # Parser builder for timestamps and video duration.
  parser_builder.parse_feature(
      feature_name=input_timestamp_start_name,
      feature_type=tf.io.VarLenFeature(dtype=tf.int64),
      output_name=output_raw_timestamp_name + '_start',
      is_context=True)
  parser_builder.parse_feature(
      feature_name=input_timestamp_end_name,
      feature_type=tf.io.VarLenFeature(dtype=tf.int64),
      output_name=output_raw_timestamp_name + '_end',
      is_context=True)
  parser_builder.parse_feature(
      feature_name=input_duration_name,
      feature_type=tf.io.VarLenFeature(dtype=tf.int64),
      output_name=output_raw_duration_name,
      is_context=True)

  # Tokenize the text captions individually.
  preprocessor_builder.add_fn(
      fn=lambda x: processors.tokenize(  # pylint: disable=g-long-lambda
          x, tokenizer, output_raw_string_name, output_feature_name,
          prepend_bos, append_eos, max_num_tokens, keep_raw_string),
      fn_name=f'{output_feature_name}_tokenization')

  if asr_input:
    # Parse ASR text indices.
    if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
      parser_builder.parse_feature(
          feature_name=asr_raw_string_name,
          feature_type=tf.io.VarLenFeature(dtype=tf.string),
          output_name='asr_string',
          is_context=True)
    elif isinstance(parser_builder, builders.ExampleParserBuilder):
      parser_builder.parse_feature(
          feature_name=asr_raw_string_name,
          feature_type=tf.io.VarLenFeature(dtype=tf.string),
          output_name='asr_string')

    # Parser builder for ASR timestamps.
    parser_builder.parse_feature(
        feature_name=asr_timestamp_name + '/start',
        feature_type=tf.io.VarLenFeature(dtype=tf.int64),
        output_name='asr_start',
        is_context=True)
    parser_builder.parse_feature(
        feature_name=asr_timestamp_name + '/end',
        feature_type=tf.io.VarLenFeature(dtype=tf.int64),
        output_name='asr_end',
        is_context=True)

    # Tokenize the ASR sentences individually.
    preprocessor_builder.add_fn(
        fn=lambda x: processors.tokenize(  # pylint: disable=g-long-lambda
            x, tokenizer, 'asr_string', 'asr_indices',
            prepend_bos, append_eos, max_num_input_words, False),
        fn_name='asr_indices_tokenization')

  def add_timestamp(batch):
    # Tokenize timestamp and add these tokens to each caption.

    # Load batch.
    duration = batch[output_raw_duration_name]
    start = batch[output_raw_timestamp_name + '_start']
    end = batch[output_raw_timestamp_name + '_end']
    caption_tokens = batch[output_feature_name]

    # Truncate events.
    start = start[:max_events]
    end = end[:max_events]
    caption_tokens = caption_tokens[:max_events]

    # Tokenize timestamps.
    timestamp_token = timestampify(
        start=start,
        end=end,
        duration=duration,
        abs_time_token=abs_time_token,
        num_bins=num_bins,
        vocabulary_size=vocabulary_size,
        time_format=time_format,
        t=t)

    # Merge caption and time tokens.
    if (not notime) and (not tmp_only):
      seq = merge_cap_time_tokens(caption_tokens, timestamp_token, order)
    elif notime:  # only consider caption tokens
      seq = caption_tokens
    elif tmp_only:  # only consider time tokens
      seq = timestamp_token
      seq -= 32126

    # Prepare timestamp for ASR.
    if asr_input:
      asr_start = batch['asr_start']
      asr_end = batch['asr_end']
      asr_tokens = batch['asr_indices']
      asr_stamp_token = timestampify(start=asr_start,
                                     end=asr_end,
                                     duration=duration,
                                     abs_time_token=abs_time_token,
                                     num_bins=num_bins,
                                     vocabulary_size=vocabulary_size,
                                     time_format=time_format)
      if asr_notime:
        batch['asr_indices'] = asr_tokens
      else:
        batch['asr_indices'] = merge_cap_time_tokens(asr_tokens,
                                                     asr_stamp_token, order)

      del batch['asr_start']
      del batch['asr_end']

    batch[output_feature_name] = seq  # [n_events, max_num_words]

    # Pad caption, start, end, split to max_events for data collation.
    if keep_raw_string:  # eval
      n_events = tf.shape(input=start)[0]
      padding_pattern = [
          [0, tf.maximum(0, max_events - n_events)],
      ]
      if 'split' in batch:
        batch['split'] = tf.pad(
            tensor=batch['split'], paddings=padding_pattern, constant_values=-1)
      batch[output_raw_string_name] = tf.pad(
          tensor=batch[output_raw_string_name],
          paddings=padding_pattern,
          constant_values='')
      batch[output_raw_timestamp_name + '_start'] = tf.pad(
          tensor=start, paddings=padding_pattern, constant_values=-1)
      batch[output_raw_timestamp_name + '_end'] = tf.pad(
          tensor=end, paddings=padding_pattern, constant_values=-1)
    else:
      del batch[output_raw_timestamp_name + '_start']
      del batch[output_raw_timestamp_name + '_end']

    return batch

  # Add timestamp tokens.
  preprocessor_builder.add_fn(
      fn=add_timestamp,
      fn_name=f'{output_feature_name}_timify')

  # Reshape to concatenate all the dense captions.
  preprocessor_builder.add_fn(
      fn=lambda x: tf.reshape(x, [-1]),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_reshape')

  # Get the boolean mask and remove padded tokens for each caption.
  preprocessor_builder.add_fn(
      fn=lambda x: tf.boolean_mask(x, tf.not_equal(x, 0)),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_unpad')

  # Get the boolean mask and remove EOS tokens for each caption.
  preprocessor_builder.add_fn(
      fn=lambda x: tf.boolean_mask(x, tf.not_equal(x, 1)),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_uneos')

  def corruption(batch):
    tokens = batch[output_feature_name]
    x = single_example_denoise(
        tokens=tokens,
        vocabulary_size=vocabulary_size,
        noise_density=corrupt,
        noise_mask_fn=functools.partial(
            random_spans_noise_mask,
            mean_noise_span_length=span_len),
        inputs_fn=noise_span_to_unique_sentinel,
        targets_fn=nonnoise_span_to_unique_sentinel)
    # no BOS in encoder and no EOS anywhere here
    batch[output_feature_name +
          '_corrupt_outputs'] = processors.sample_or_pad_non_sorted_sequence(
              tf.concat([[0], x['outputs'][:max_num_tokens - 1]], 0),
              max_num_tokens,
              pad_value=0,
              random=False)  # max_num_tokens
    batch[output_feature_name +
          '_corrupt_inputs'] = processors.sample_or_pad_non_sorted_sequence(
              x['inputs'][:max_num_tokens - 1], max_num_tokens - 1, pad_value=0,
              random=False)  # max_num_tokens - 1
    return batch

  preprocessor_builder.add_fn(
      fn=corruption, fn_name=f'{output_feature_name}_corrupt')

  # Readd BOS=PAD and EOS token, truncate+pad for the full sequence.
  preprocessor_builder.add_fn(
      fn=lambda x: processors.sample_or_pad_non_sorted_sequence(  # pylint: disable=g-long-lambda
          tf.concat([[0], x, [1]], 0),
          max_num_tokens,
          pad_value=0,
          random=False),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_repad')

  if asr_input:
    # Reshape to concatenate all the dense captions
    preprocessor_builder.add_fn(
        fn=lambda x: tf.reshape(x, [-1]),
        feature_name='asr_indices',
        fn_name='asr_indices_reshape')

    # Get the boolean mask and remove padded tokens for each caption
    preprocessor_builder.add_fn(
        fn=lambda x: tf.boolean_mask(x, tf.not_equal(x, 0)),
        feature_name='asr_indices',
        fn_name='asr_indices_unpad')

    # Get the boolean mask and remove EOS tokens for each caption
    preprocessor_builder.add_fn(
        fn=lambda x: tf.boolean_mask(x, tf.not_equal(x, 1)),
        feature_name='asr_indices',
        fn_name='asr_indices_uneos')

    # Readd BOS=PAD and EOS token, truncate+pad for the full sequence
    preprocessor_builder.add_fn(
        fn=lambda x: processors.sample_or_pad_non_sorted_sequence(  # pylint: disable=g-long-lambda
            tf.concat([x, [1]], 0),
            max_num_input_words,
            pad_value=0,
            random=False),
        feature_name='asr_indices',
        fn_name='asr_indices_repad')


def random_apply(func, gunc, p, x):
  """Randomly apply function func to x with probability p otherwise function gunc."""
  return tf.cond(
      tf.less(
          tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
          tf.cast(p, tf.float32)), lambda: func(x), lambda: gunc(x))


def sample_equal_sequence(
    batch,
    num_steps: int,
    is_training: bool,
    output_feature_lists_name: str,
    output_raw_timestamp_name: str,
    output_raw_duration_name: str,
    output_raw_string_name: str,
    p: float,
    t: int,
    preserve: bool,
    asr_input: bool,
    max_segments: int) -> tf.Tensor:
  """Samples at equal distance num_steps features + pad + random temporal crop."""

  if is_training and p > 0:
    # Random temporal cropping with proba p, preservering or not events.
    def random_crop(batch, preserve=True):

      # Load batch.
      sequence = batch[output_feature_lists_name]
      duration = batch[output_raw_duration_name]
      captions = batch[output_raw_string_name]
      start = batch[output_raw_timestamp_name + '_start']
      end = batch[output_raw_timestamp_name + '_end']

      # Sample start offset.
      sequence_length = tf.shape(input=sequence)[0]
      if preserve:
        max_offset = tf.cast(tf.math.reduce_min(end / t), dtype=tf.int32) - 1
      else:
        max_offset = sequence_length
      max_offset = tf.minimum(max_offset, sequence_length)
      max_offset = tf.maximum(max_offset, 1)
      offset_start = tf.random.uniform((),
                                       maxval=max_offset,
                                       dtype=tf.int32)

      # Modify captions/start/end given the sampled start.
      start = tf.cast(start, tf.int32) - offset_start * t
      end = tf.cast(end, tf.int32) - offset_start * t
      idx_to_keep = tf.where(end > 0)[:, 0]
      captions = tf.gather(captions, idx_to_keep, axis=0)
      start = tf.gather(start, idx_to_keep, axis=0)
      start = tf.maximum(start, 0)
      end = tf.gather(end, idx_to_keep, axis=0)

      # Sample end offset.
      if preserve:
        min_offset = tf.cast(tf.math.reduce_max(start / t), dtype=tf.int32) + 1
      else:
        min_offset = offset_start
      maxval = sequence_length
      if max_segments:  # only consider a given maximum number of segments
        maxval = tf.math.reduce_max(end[:max_segments] // t)
        maxval = tf.minimum(maxval, sequence_length)
        maxval = tf.maximum(maxval, min_offset + 1)
      min_offset = tf.minimum(min_offset, maxval - 1)
      offset_end = tf.random.uniform((),
                                     minval=min_offset,
                                     maxval=maxval,
                                     dtype=tf.int32)

      # Modify captions/start/end given the sampled end.
      idx_to_keep = tf.where(start < offset_end * t)[:, 0]
      captions = tf.gather(captions, idx_to_keep, axis=0)
      start = tf.gather(start, idx_to_keep, axis=0)
      end = tf.gather(end, idx_to_keep, axis=0)
      end = tf.minimum(end, offset_end * t)

      # Modify sequence and duration given the sampled offsets.
      sequence = sequence[offset_start: offset_end + 1]
      duration = (offset_end - offset_start + 1) * t

      # Correct dimensions and types.
      duration = tf.cast(duration, tf.int64)[None]
      start = tf.cast(start, tf.int64)
      end = tf.cast(end, tf.int64)

      if asr_input:
        # Modify ASR given the sampled start.
        asr = batch['asr_string']
        asr_st = batch['asr_start']
        asr_ed = batch['asr_end']
        asr_st = tf.cast(asr_st, tf.int32) - offset_start * t
        asr_ed = tf.cast(asr_ed, tf.int32) - offset_start * t
        asr_idx = tf.where(asr_ed > 0)[:, 0]
        asr = tf.gather(asr, asr_idx, axis=0)
        asr_st = tf.gather(asr_st, asr_idx, axis=0)
        asr_ed = tf.gather(asr_ed, asr_idx, axis=0)
        # Modify ASR given the sampled end.
        asr_idx = tf.where(asr_st < offset_end * t)[:, 0]
        asr = tf.gather(asr, asr_idx, axis=0)
        asr_st = tf.gather(asr_st, asr_idx, axis=0)
        asr_ed = tf.gather(asr_ed, asr_idx, axis=0)
        asr_st = tf.cast(asr_st, tf.int64)
        asr_ed = tf.cast(asr_ed, tf.int64)
        return sequence, duration, captions, start, end, asr, asr_st, asr_ed

      return sequence, duration, captions, start, end

    def no_crop(batch):
      if asr_input:
        return batch[output_feature_lists_name], batch[
            output_raw_duration_name], batch[output_raw_string_name], batch[
                output_raw_timestamp_name +
                '_start'], batch[output_raw_timestamp_name + '_end'], batch[
                    'asr_string'], batch['asr_start'], batch['asr_end']
      return batch[output_feature_lists_name], batch[
          output_raw_duration_name], batch[output_raw_string_name], batch[
              output_raw_timestamp_name +
              '_start'], batch[output_raw_timestamp_name + '_end']

    output = random_apply(
        func=functools.partial(random_crop, preserve=preserve),
        gunc=no_crop,
        p=p,
        x=batch)

    # Update batch
    batch[output_feature_lists_name] = output[0]
    batch[output_raw_duration_name] = output[1]
    batch[output_raw_string_name] = output[2]
    batch[output_raw_timestamp_name + '_start'] = output[3]
    batch[output_raw_timestamp_name + '_end'] = output[4]

    if asr_input:
      batch['asr_string'] = output[5]
      batch['asr_start'] = output[6]
      batch['asr_end'] = output[7]

  sequence = batch[output_feature_lists_name]
  sequence_length = tf.shape(input=sequence)[0]

  # Pad or sample
  output = tf.cond(
      sequence_length < num_steps,
      lambda: processors.sample_or_pad_non_sorted_sequence(  # pylint: disable=g-long-lambda
          sequence, num_steps, 0, False),
      lambda: processors.sample_linspace_sequence(sequence, num_steps, 1, 1)
      )

  batch[output_feature_lists_name] = output
  return batch


def add_embeddings(
    parser_builder: builders.BaseParserBuilder,
    sampler_builder: builders.SamplerBuilder,
    input_feature_lists_name: str,
    output_feature_lists_name: str,
    num_frames: int,
    features_dim: int,
    sync_random_state: bool,
    output_raw_timestamp_name: str,
    output_raw_duration_name: str,
    is_training: bool,
    output_raw_string_name: str,
    p: float,
    output_feature_name: str = builders.TEXT_INDICES_FEATURE_NAME,
    t: int = 1000000,  # 1FPS
    preserve: bool = True,
    asr_input: bool = False,
    max_segments: int = 0):
  """Add visual features [num_frames, 768]."""

  if not isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    raise ValueError('add_embeddings only supports tf.SequenceExample.')

  parser_builder.parse_feature(
      feature_name=input_feature_lists_name,
      feature_type=tf.io.FixedLenSequenceFeature([features_dim],
                                                 dtype=tf.float32),
      output_name=output_feature_lists_name)

  # Moved from the decoder builder as these are used for temporal cropping.
  sampler_builder.add_fn(
      fn=tf.sparse.to_dense,
      feature_name=output_raw_string_name,
      fn_name=f'{output_feature_name}_sparse_to_dense')
  if output_raw_timestamp_name:
    sampler_builder.add_fn(
        fn=tf.sparse.to_dense,
        feature_name=output_raw_timestamp_name + '_start',
        fn_name=f'{output_raw_timestamp_name}_start_sparse_to_dense')
    sampler_builder.add_fn(
        fn=tf.sparse.to_dense,
        feature_name=output_raw_timestamp_name + '_end',
        fn_name=f'{output_raw_timestamp_name}_end_sparse_to_dense')
    sampler_builder.add_fn(
        fn=tf.sparse.to_dense,
        feature_name=output_raw_duration_name,
        fn_name=f'{output_raw_duration_name}_sparse_to_dense')

  if asr_input:
    sampler_builder.add_fn(
        fn=tf.sparse.to_dense,
        feature_name='asr_string',
        fn_name='asr_string_sparse_to_dense')
    if output_raw_timestamp_name:
      sampler_builder.add_fn(
          fn=tf.sparse.to_dense,
          feature_name='asr_start',
          fn_name='asr_start_sparse_to_dense')
      sampler_builder.add_fn(
          fn=tf.sparse.to_dense,
          feature_name='asr_end',
          fn_name='asr_end_sparse_to_dense')

  if output_raw_timestamp_name:
    sampler_builder.add_fn(
        fn=lambda x, s=None: sample_equal_sequence(  # pylint: disable=g-long-lambda
            x,
            num_frames,
            is_training=is_training,
            output_feature_lists_name=output_feature_lists_name,
            output_raw_timestamp_name=output_raw_timestamp_name,
            output_raw_duration_name=output_raw_duration_name,
            output_raw_string_name=output_raw_string_name,
            p=p,
            t=t,
            preserve=preserve,
            asr_input=asr_input,
            max_segments=max_segments),
        fn_name=f'{output_feature_lists_name}_sample',
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state)
  else:
    sampler_builder.add_fn(
        fn=functools.partial(
            processors.sample_or_pad_non_sorted_sequence,  # pylint: disable=g-long-lambda
            max_num_steps=num_frames, pad_value=0, random=False),
        feature_name=output_feature_lists_name,
        fn_name='pad_features'
    )

