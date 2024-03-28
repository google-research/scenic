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

"""Contains dataset utility functions."""

import csv
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from absl import logging
from dmvr import builders
from dmvr import processors
from dmvr import tokenizers as dmvr_tokenizers
import ml_collections
import numpy as np
from scenic.projects.t5 import tokenizer as t5_tokenizer
import tensorflow as tf


def sample_or_pad_sequence(
    sequence: tf.Tensor,
    max_num_steps: int,
    pad_value: Any,
    random: bool,
    stride: int = 1,
    seed: Optional[int] = None,
    state: Optional[builders.ProcessorState] = None) -> tf.Tensor:
  """Samples or pads (with `pad_value`) elements from the input sequence.

  This function is adapted from sample_sequence() from dmvr/processors.py.

  The input sequence can be multidimensional, but the sampling or pads will
  only happen in the first dimension. processors.sample_sequence() performs
  padding by repeating the input sequence, which may not be ideal some
  applications, such as localization. This function pads the sequence by a
  constant.

  Args:
    sequence: Any tensor where the first dimension is timesteps.
    max_num_steps: Maximum number of steps to be kept from the input. If the
      input contains more, it's sampled, if less, it's padded.
    pad_value: Value to be used when padding. Same type as `sequence`.
    random: A boolean indicating whether to randomly sample from input. If
      False, the central `max_num_steps` elements will be sampled.
    stride: Temporal stride.
    seed: A deterministic seed to use when sampling.
    state:  A mutable dictionary where keys are strings. The dictionary might
      contain 'sample_sequence_random_offset' as key with metadata useful for
      sampling. It will be modified with added metadata if needed. This can be
      used to keep consistency between sampling of different sequences. Note
      that a runtime error will be raised in case state is provided but the
      sequences that one tries to sync are of different lengths.

  Returns:
    A single tensor with first dimension `max_num_steps` with the sampled
    elements.

  Raises:
    tf.errors.InvalidArgumentError: if state is provided but the sequences that
      one tries to sync are of different lengths.
  """
  sequence_length = tf.shape(input=sequence)[0]
  requested_length = (max_num_steps - 1) * stride + 1
  padding_pattern = [
      [0, tf.maximum(0, requested_length - sequence_length)],
  ]
  num_dim = len(tf.shape(input=sequence))
  if num_dim > 1:
    padding_pattern.append([0, 0] * (num_dim - 1))
  padded_sequence = tf.pad(
      tensor=sequence, paddings=padding_pattern, constant_values=pad_value)

  if random:
    if state and 'sample_sequence_random_offset' in state:
      # Read offset from state to ensure consistent offsets for different
      # modalities.
      offset = state['sample_sequence_random_offset']
    else:
      offset_max = tf.maximum(1, sequence_length - (max_num_steps - 1) * stride)
      offset = tf.random.uniform(
          shape=(), minval=0, maxval=offset_max, dtype=tf.int32, seed=seed)
      if state is not None:
        state['sample_sequence_random_offset'] = offset
  else:
    offset = tf.maximum(0, sequence_length - (max_num_steps - 1) * stride) // 2
  return tf.gather(padded_sequence,
                   tf.range(offset, offset + requested_length, stride))


def add_fixed_len_context_feature(
    parser_builder: builders.BaseParserBuilder,
    input_context_feature_name: str,
    output_context_feature_name: str,
    dtype: tf.dtypes.DType,
    feature_dim: int = 0,
):
  """Adds functions to process fixed length context features.

  The input proto is expected to be tf.SequenceExample and its structure
  follows:

  context {
    feature: {
      key: input_context_feature_name
      value: {
        int64_list: {
          value: 0
        }
      }
    }
  }
  or
  context {
    feature: {
      key: input_context_feature_name
      value: {
        bytes_list: {
          value: ""
        }
      }
    }
  }
  or
  context {
    feature: {
      key: input_context_feature_name
      value: {
        float_list: {
          value: 0.0
        }
      }
    }
  }

  Common fixed length context features are frame rate, media id, embeddings,
  etc.

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    input_context_feature_name: Name of the context feature in the input
      tf.train.SequenceExample`.
    output_context_feature_name: Name of the context feature in the output
      features dictionary.
    dtype: Value type, tf.string, tf.float32, tf.int64
    feature_dim: Feature dimension. If it is a scalar, feature_dim = 0.
  """
  if not isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    raise ValueError('add_context_feature only supports tf.SequenceExample.')

  parser_builder.parse_feature(
      feature_name=input_context_feature_name,
      feature_type=tf.io.FixedLenFeature(
          shape=([] if feature_dim == 0 else feature_dim), dtype=dtype),
      output_name=output_context_feature_name,
      is_context=True)


def add_pad_context_feature(parser_builder: builders.BaseParserBuilder,
                            decoder_builder: builders.DecoderBuilder,
                            preprocessor_builder: builders.PreprocessorBuilder,
                            input_context_feature_name: str,
                            output_context_feature_name: str,
                            dtype: tf.dtypes.DType,
                            max_feature_length: int,
                            pad_value: Any,
                            is_training: bool = False,
                            sync_random_state: bool = True):
  """Adds functions to add all context features.

  The input proto is expected to be tf.SequenceExample and its structure
  follows:

  context {
    feature: {
      key: input_context_feature_name
      value: {
        int64_list: {
          value: 0
          value: 0
          ...
        }
      }
    }
  }
  or
  context {
    feature: {
      key: input_context_feature_name
      value: {
        bytes_list: {
          value: ""
          value: ""
          ...
        }
      }
    }
  }
  or
  context {
    feature: {
      key: input_context_feature_name
      value: {
        float_list: {
          value: 0.0
          value: 0.0
          ...
        }
      }
    }
  }

  Common variable length context features are labels, start and end times, etc.

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    decoder_builder: An instance of a `builders.DecoderBuilder`.
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    input_context_feature_name: Name of the context feature in the input
      tf.train.SequenceExample`.
    output_context_feature_name: Name of the context feature in the output
      features dictionary.
    dtype: Value type, tf.string, tf.float32, tf.int32
    max_feature_length: The number of returned features. If the actual feature
      length is less than this number, padding will be used.
    pad_value: Padding value.
    is_training: Whether or not it is in training.
    sync_random_state: Whether to use stateful option to keep random operations
      in sync.
  """
  if not isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    raise ValueError(
        'add_pad_context_feature only supports tf.SequenceExample.')

  parser_builder.parse_feature(
      feature_name=input_context_feature_name,
      feature_type=tf.io.VarLenFeature(dtype=dtype),
      output_name=output_context_feature_name,
      is_context=True)

  decoder_builder.add_fn(
      fn=tf.sparse.to_dense,
      feature_name=output_context_feature_name,
      fn_name=f'{output_context_feature_name}_sparse_to_dense')

  preprocessor_builder.add_fn(
      # pylint: disable=g-long-lambda
      # Matches the same sampling method used in DMVR add_text().
      lambda x, s=None: processors.sample_or_pad_non_sorted_sequence(
          x,
          max_num_steps=max_feature_length,
          pad_value=pad_value,
          random=is_training,
          state=s),
      # pylint: enable=g-long-lambda
      feature_name=output_context_feature_name,
      fn_name=f'{output_context_feature_name}_add_pad_context_feature',
      stateful=sync_random_state)


def add_input_mask(
    preprocessor_builder: builders.PreprocessorBuilder,
    total_length_name: str,
    output_feature_name: str,
    num_frames: int,
    stride: int,
    sampling_strategy: str = 'random',
    feature_pyramid_levels: Optional[Sequence[int]] = None,
    feature_pyramid_downsample_stride: int = 2,
):
  """Adds a function to create input mask.

  The paddings will be assigned zeros. This input mask will be used as attention
  mask in transformers or used in loss/metric computation.

  Args:
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    total_length_name: Name of the context feature that stores the total number
      of frames.
    output_feature_name: Name of the output feature in the output features
      dictionary.
    num_frames: Number of frames in the output feature.
    stride: Temporal stride used to sample frames.
    sampling_strategy: `random` or `linspace`.
    feature_pyramid_levels: A list of layers from which we build feature
      pyramid.
    feature_pyramid_downsample_stride: The stride used to downsample the
      features in the pyramid.
  """

  assert sampling_strategy in {'random', 'linspace'}

  def _create_mask_linspace_sampling(
      feature_dict: builders.FeaturesDict) -> tf.Tensor:
    del feature_dict
    return tf.ones((num_frames,), dtype=tf.int32)

  def _create_fpn_mask_linspace_sampling(
      feature_dict: builders.FeaturesDict) -> tf.Tensor:
    del feature_dict
    total_frames = 0
    for level in range(len(feature_pyramid_levels)):
      cur_downsample_stride = feature_pyramid_downsample_stride**level
      cur_num_frames = num_frames // cur_downsample_stride
      total_frames += cur_num_frames
    return tf.ones((total_frames,), dtype=tf.int32)

  def _create_mask_random_sampling(
      feature_dict: builders.FeaturesDict) -> tf.Tensor:
    total_length = tf.cast(feature_dict[total_length_name], tf.int32)
    indices = tf.range(0, num_frames * stride, stride, dtype=tf.int32)
    return tf.cast(indices < total_length, tf.int32)

  def _create_fpn_mask_random_sampling(
      feature_dict: builders.FeaturesDict) -> tf.Tensor:
    mask = []
    total_length = tf.cast(feature_dict[total_length_name], tf.int32)

    for level in range(len(feature_pyramid_levels)):
      cur_downsample_stride = feature_pyramid_downsample_stride**level
      cur_num_frames = num_frames // cur_downsample_stride
      cur_stride = stride * cur_downsample_stride
      indices = tf.range(
          0, cur_num_frames * cur_stride, cur_stride, dtype=tf.int32)
      mask.append(tf.cast(indices < total_length, tf.int32))

    return tf.concat(mask, axis=0)

  if sampling_strategy == 'random':
    if feature_pyramid_levels is None:
      create_mask_fn = _create_mask_random_sampling
    else:
      create_mask_fn = _create_fpn_mask_random_sampling
  else:  # 'linspace'
    if feature_pyramid_levels is None:
      create_mask_fn = _create_mask_linspace_sampling
    else:
      create_mask_fn = _create_fpn_mask_linspace_sampling

  def _add_mask(feature_dict: builders.FeaturesDict) -> builders.FeaturesDict:
    feature_dict[output_feature_name] = create_mask_fn(feature_dict)
    return feature_dict

  preprocessor_builder.add_fn(_add_mask)


def add_caption_mask(
    preprocessor_builder: builders.PreprocessorBuilder,
    input_feature_name: str,
    padding_value: Union[int, float, str],
    output_feature_name: str,
):
  """Adds a function to create a mask for input captions.

  The paddings will be assigned zeros. This mask will be used in computing the
  loss.

  Args:
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    input_feature_name: Name of the context feature that has the same length as
      captions.
    padding_value: Padding value for the input feature.
    output_feature_name: Name of the output feature in the output features
      dictionary to store the mask.
  """

  def _add_mask(feature_dict: builders.FeaturesDict) -> builders.FeaturesDict:
    feature_dict[output_feature_name] = tf.cast(
        feature_dict[input_feature_name] != padding_value, tf.int32)
    return feature_dict

  preprocessor_builder.add_fn(_add_mask)


def _get_random_offset(state: Dict[str, Any], total_length: tf.Tensor,
                       num_frames: int, stride: int,
                       is_training: bool) -> tf.Tensor:
  """Generates a random offset for sampling the sequence."""

  if not is_training:
    return tf.maximum(
        0, tf.cast((total_length - num_frames * stride) // 2, tf.int32))
  if state and 'sample_offset_proportion' in state:
    # 'sample_offset_proportion' is the same key used in add_images.
    offset = state['sample_offset_proportion'] * total_length
    offset = tf.cast(tf.math.round(offset), tf.int32)
  else:
    offset = processors._get_random_sampling_offset(  # pylint:disable=protected-access
        sequence=tf.ones((total_length,)),
        num_steps=num_frames,
        stride=stride)
    if state is not None:
      # Update state.
      sample_offset_proportion = (
          tf.cast(offset, tf.float32) / tf.cast(total_length, tf.float32))
      state['sample_offset_proportion'] = sample_offset_proportion
  return offset


def add_action_segmentation_labels(
    preprocessor_builder: builders.PreprocessorBuilder,
    segment_start_index_name: str,
    segment_end_index_name: str,
    segment_label_index_name: str,
    total_length_name: str,
    output_label_name: str,
    max_num_segments: int,
    num_frames: int,
    num_classes: int,
    is_training: bool = True,
    sync_random_state: bool = True,
):
  """Adds functions to action segmentation labels.

  The output labels are onehot or all-zero vectors of shape (num_frames,
  num_classes).

  Args:
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    segment_start_index_name: Name of the context feature that stores segment
      start indices.
    segment_end_index_name: Name of the context feature that stores segment
      end indices.
    segment_label_index_name: Name of the context feature that stores segment
      label indices.
    total_length_name: Name of the context feature that stores the total number
      of frames.
    output_label_name: Name of the feature that stores the frame labels in the
      output feature dictionary.
    max_num_segments: Max number of segments in the whole dataset.
    num_frames: Number of frames in the output feature.
    num_classes: Number of classes. Set it to a negative value when the task is
      moment retrieval.
    is_training: Whether or not it is in training.
    sync_random_state: Whether to use stateful option to keep random operations
      in sync.
  """

  def _add_labels(state: Dict[str, Any],
                  feature_dict: builders.FeaturesDict) -> builders.FeaturesDict:
    segment_start_indices = tf.cast(feature_dict[segment_start_index_name],
                                    tf.int32)
    segment_end_indices = tf.cast(feature_dict[segment_end_index_name],
                                  tf.int32)
    segment_label_indices = tf.cast(feature_dict[segment_label_index_name],
                                    tf.int32)
    frame_label_indices = tf.fill([num_frames], -1)
    total_length = feature_dict[total_length_name]
    offset = _get_random_offset(state, total_length, num_frames, 1, is_training)
    frame_indices = tf.range(0, num_frames, dtype=tf.int32) + offset

    for i in range(max_num_segments):
      segment_start_index = segment_start_indices[i]
      segment_end_index = segment_end_indices[i]
      in_segment_mask = tf.logical_and(
          tf.math.greater_equal(frame_indices, segment_start_index),
          tf.math.less_equal(frame_indices, segment_end_index))
      frame_label_indices = tf.where(
          in_segment_mask, x=segment_label_indices[i], y=frame_label_indices)

    multihot_target = tf.one_hot(frame_label_indices, depth=num_classes)
    multihot_target = tf.cast(multihot_target, tf.int32)
    feature_dict[output_label_name] = multihot_target

    return feature_dict

  preprocessor_builder.add_fn(
      lambda x, s=None: _add_labels(s, x), stateful=sync_random_state)


def _add_labels_from_one_pyramid_level_moment_retrieval(
    segment_start_indices: tf.Tensor,
    segment_end_indices: tf.Tensor,
    frame_indices: tf.Tensor,
    max_num_segments: int,
    box_jitter_ratio: float = 0.0,
    radius: Optional[float] = None,
    regression_range: Tuple[float, float] = (0.0, float('inf')),
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Adds frame labels and displacements for one pyramid level.

  Args:
    segment_start_indices: A int tensor of shape (max_num_segments,) containing
      the caption start indices in the normalized coordinate system.
    segment_end_indices: A int tensor of shape (max_num_segments,) containing
      the caption end indices in the normalized coordinate system.
    frame_indices: Frame indices in the normalized coordinate system.
    max_num_segments: Max number of captions in this dataset. We pad the
      examples that have fewer captions.
    box_jitter_ratio: The ratio of segment length used to jitter segment
      start/end times.
    radius: If set, a frame is marked as positive only if it is within `radius`
      distance of the segment center.
    regression_range: If set, a frame is marked as positive only if its distance
      to the start/end indices are within this range.

  Returns:
    frame_labels: A 3D binary tensor of shape (max_captions, num_frames, 1)
      indicating the frame labels of each caption.
    displacements: A 3D float tensor of shape (max_captions, num_frames, 2)
      indicating the distances to the caption start/end time for each frame.
  """

  displacements = []
  frame_labels = []

  for i in range(max_num_segments):
    segment_start_index = segment_start_indices[i]
    segment_end_index = segment_end_indices[i]
    distortion = tf.random.uniform(
        shape=[2],
        minval=-box_jitter_ratio,
        maxval=box_jitter_ratio,
        dtype=tf.float32,
    )
    segment_duration = segment_end_index - segment_start_index
    segment_start_index += segment_duration * distortion[0]
    segment_end_index += segment_duration * distortion[1]
    displacements_to_start = frame_indices - segment_start_index
    displacements_to_end = segment_end_index - frame_indices
    displacements.append(
        tf.stack([displacements_to_start, displacements_to_end], axis=1))
    if regression_range is not None:
      distances = tf.stack([
          segment_end_index - frame_indices, frame_indices - segment_start_index
      ],
                           axis=-1)
    max_distances = tf.cast(tf.reduce_max(distances, axis=-1), tf.float32)
    in_range_mask = tf.logical_and(
        tf.math.greater_equal(max_distances, regression_range[0]),
        tf.math.less(max_distances, regression_range[1]))
    if radius is not None:
      segment_center_index = (segment_start_index + segment_end_index) * 0.5
      segment_start_index = tf.maximum(segment_start_index,
                                       segment_center_index - radius)
      segment_end_index = tf.minimum(segment_end_index,
                                     segment_center_index + radius)
    in_segment_mask = tf.logical_and(
        tf.math.greater_equal(frame_indices, segment_start_index),
        tf.math.less_equal(frame_indices, segment_end_index))
    frame_labels.append(
        tf.cast(tf.logical_and(in_segment_mask, in_range_mask), tf.int32))

  displacements = tf.stack(displacements, axis=0)
  frame_labels = tf.stack(frame_labels, axis=0)[..., None]
  return frame_labels, displacements


def _add_labels_from_one_pyramid_level_tal(
    segment_start_indices: tf.Tensor,
    segment_end_indices: tf.Tensor,
    segment_label_indices: tf.Tensor,
    frame_indices: tf.Tensor,
    total_length: tf.Tensor,
    num_frames: int,
    num_classes: int,
    max_num_segments: int,
    box_jitter_ratio: float = 0.0,
    radius: Optional[float] = None,
    regression_range: Tuple[float, float] = (0.0, float('inf')),
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Adds frame labels and displacements for one pyramid level."""

  multihot_target = tf.zeros((num_frames, num_classes), dtype=tf.float32)
  # The first row will NOT be used and serves as a placeholder when
  # segment_label = -1.
  displacements_to_start = tf.fill([num_classes + 1, num_frames],
                                   total_length + 1.0)
  # The first row will NOT be used and serves as a placeholder when
  # segment_label = -1.
  displacements_to_end = tf.fill([num_classes + 1, num_frames],
                                 total_length + 1.0)

  for i in range(max_num_segments):
    frame_label_indices = tf.fill([num_frames], -1)
    segment_start_index = segment_start_indices[i]
    segment_end_index = segment_end_indices[i]
    distortion = tf.random.uniform(
        shape=[2],
        minval=-box_jitter_ratio,
        maxval=box_jitter_ratio,
        dtype=tf.float32,
    )
    segment_duration = segment_end_index - segment_start_index
    segment_start_index += segment_duration * distortion[0]
    segment_end_index += segment_duration * distortion[1]
    cur_segment_label_index = segment_label_indices[i] + 1
    # The distances to the start should only be considered for the frames
    # after the start.
    cur_displacements_to_start = tf.where(
        frame_indices - segment_start_index < 0, total_length + 1,
        frame_indices - segment_start_index)
    displacements_to_start = tf.tensor_scatter_nd_min(
        displacements_to_start, [[cur_segment_label_index]],
        cur_displacements_to_start[None, :])
    # The distances to the end should only be considered for the frames
    # before the end.
    cur_displacements_to_end = tf.where(segment_end_index - frame_indices < 0,
                                        total_length + 1,
                                        segment_end_index - frame_indices)
    displacements_to_end = tf.tensor_scatter_nd_min(
        displacements_to_end, [[cur_segment_label_index]],
        cur_displacements_to_end[None, :])
    if regression_range is not None:
      distances = tf.stack([
          segment_end_index - frame_indices, frame_indices - segment_start_index
      ],
                           axis=-1)
    max_distances = tf.cast(tf.reduce_max(distances, axis=-1), tf.float32)
    in_range_mask = tf.logical_and(
        tf.math.greater_equal(max_distances, regression_range[0]),
        tf.math.less(max_distances, regression_range[1]))
    if radius is not None:
      segment_center_index = (segment_start_index + segment_end_index) * 0.5
      segment_start_index = tf.maximum(segment_start_index,
                                       segment_center_index - radius)
      segment_end_index = tf.minimum(segment_end_index,
                                     segment_center_index + radius)
    in_segment_mask = tf.logical_and(
        tf.math.greater_equal(frame_indices, segment_start_index),
        tf.math.less_equal(frame_indices, segment_end_index))
    frame_label_indices = tf.where(
        tf.logical_and(in_segment_mask, in_range_mask),
        x=segment_label_indices[i],
        y=frame_label_indices)
    multihot_target += tf.one_hot(frame_label_indices, depth=num_classes)

  # in case of duplicate labels.
  multihot_target = tf.clip_by_value(multihot_target, 0.0, 1.0)
  multihot_target = tf.cast(multihot_target, tf.int32)
  # Removes the first row.
  displacements_to_start = tf.transpose(displacements_to_start[1:])
  # Removes the first row.
  displacements_to_end = tf.transpose(displacements_to_end[1:])
  displacements = tf.stack([displacements_to_start, displacements_to_end],
                           axis=2)
  return multihot_target, displacements


def add_frame_labels_and_displacements(
    preprocessor_builder: builders.PreprocessorBuilder,
    segment_start_index_name: str,
    segment_end_index_name: str,
    total_length_name: str,
    output_label_name: str,
    output_displacement_name: str,
    max_num_segments: int,
    num_frames: int,
    stride: int,
    num_classes: int,
    sampling_strategy: str = 'random',
    segment_label_index_name: Optional[str] = None,
    radius: Optional[float] = None,
    feature_pyramid_levels: Optional[Sequence[int]] = None,
    feature_pyramid_downsample_stride: int = 2,
    regression_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    normalize_displacements_by_downsample_stride: bool = False,
    min_displacements_across_class: bool = False,
    box_jitter_ratio: float = 0.0,
    is_training: bool = True,
    sync_random_state: bool = True,
):
  """Adds functions to create per-frame labels and start/end time displacements.

  For every frame, we output a multihot vector indicating whether the current
  frame is within predefined segments or not. If radius is defined, the frames
  within the radius of the center of a segment are considered positive. For
  every frame, we also output the distances between the current frame and the
  closest segment's start and end times.

  For temporal localization (num_classes > 0),
  the output labels are multihot vectors of shape (num_frames, num_classes) and
  the output displacements are of shape (num_frames, num_classes, 2) if
  min_displacements_across_class = False, otherwise (num_frames, 2).

  For moment retrieval (num_classes < 0),
  the output labels are multihot vectors of shape (max_num_captions, num_frames,
  1) and the output displacements are of shape (max_num_captions, num_frames,
  2).

  Args:
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    segment_start_index_name: Name of the context feature that stores segment
      start indices.
    segment_end_index_name: Name of the context feature that stores segment
      start indices.
    total_length_name: Name of the context feature that stores the total number
      of frames.
    output_label_name: Name of the feature that stores the frame labels in the
      output feature dictionary.
    output_displacement_name: Name of the feature that stores the displacements
      to the start/end times in the output feature dictionary.
    max_num_segments: Max number of segments in the whole dataset.
    num_frames: Number of frames in the output feature.
    stride: Temporal stride used to sample frames.
    num_classes: Number of classes. Set it to a negative value when the task is
      moment retrieval.
    sampling_strategy: 'linspace' or 'random'. Under `random` strategy, a set of
      consecutive frames is selected with a random start index when
      is_training=True and the center clip is selected when is_training=False.
      Under `linspace` strategy, tf.linspace() is used to generate a set of
      frame indices in both training and test.
    segment_label_index_name: Name of the context feature that stores segment
      label indices. If None, we output binary labels.
    radius: Radius used to determine whether a frame is within a positive
      segment. If None, original segments are used.
    feature_pyramid_levels: A list of layers from which we build feature
      pyramid.
    feature_pyramid_downsample_stride: The stride used to downsample the
      features in the pyramid.
    regression_ranges: The output regression ranges for each pyramid level.
    normalize_displacements_by_downsample_stride: Whether or not to normalize
      the displacements by downsample stride.
    min_displacements_across_class: If True, we assume there is no overlapping
      segments in the video and return the minimum displacements across all
      classes. This flag is only applicable to temporal localization.
    box_jitter_ratio: The ratio of segment length used to jitter segment
      start/end times.
    is_training: Whether or not it is in training.
    sync_random_state: Whether to use stateful option to keep random operations
      in sync.
  """

  if segment_label_index_name is None and num_classes >= 0:
    raise ValueError(
        'num_classes must be negative if segment_label_index_name is not set.')
  assert sampling_strategy in {'linspace', 'random'}

  if regression_ranges is None:
    regression_ranges = [(0.0, float('inf'))]

  def _add_labels(state: Dict[str, Any],
                  feature_dict: builders.FeaturesDict) -> builders.FeaturesDict:
    segment_start_indices = tf.cast(feature_dict[segment_start_index_name],
                                    tf.float32)
    segment_end_indices = tf.cast(feature_dict[segment_end_index_name],
                                  tf.float32)
    total_length = tf.cast(feature_dict[total_length_name], tf.float32)
    if sampling_strategy == 'linspace':
      segment_start_indices *= (num_frames - 1) / (total_length - 1)
      segment_end_indices *= (num_frames - 1) / (total_length - 1)
    segment_label_indices = (
        tf.fill([max_num_segments], 0) if segment_label_index_name is None else
        tf.cast(feature_dict[segment_label_index_name], tf.int32))

    multihot_target = []
    displacements = []
    if sampling_strategy == 'random':
      offset = _get_random_offset(state, total_length, num_frames, stride,
                                  is_training)
      offset = tf.cast(offset, tf.float32)
    else:  # 'linspace'
      offset = 0.0
    if feature_pyramid_levels is None:
      levels = 1
    else:
      levels = len(feature_pyramid_levels)

    linspace_frame_indices = tf.cast(tf.range(0, num_frames), tf.float32)

    for level_idx in range(levels):
      cur_downsample_stride = feature_pyramid_downsample_stride**level_idx
      cur_num_frames = num_frames // cur_downsample_stride
      if sampling_strategy == 'random':
        cur_stride = stride * cur_downsample_stride
        frame_indices = tf.range(
            0, num_frames * stride, cur_stride, dtype=tf.float32)
      else:  # 'linspace'
        frame_indices = tf.gather(
            linspace_frame_indices,
            tf.range(0, num_frames, delta=cur_downsample_stride))
      frame_indices += offset
      cur_radius = None if radius is None else radius * cur_downsample_stride
      if num_classes > 0:
        cur_multihot_target, cur_displacements = (
            _add_labels_from_one_pyramid_level_tal(
                segment_start_indices,
                segment_end_indices,
                segment_label_indices,
                frame_indices,
                total_length,
                cur_num_frames,
                num_classes,
                max_num_segments,
                box_jitter_ratio,
                cur_radius,
                regression_ranges[level_idx],
            )
        )
        if min_displacements_across_class:
          cur_displacements = tf.reduce_min(cur_displacements, axis=1)
      else:
        cur_multihot_target, cur_displacements = (
            _add_labels_from_one_pyramid_level_moment_retrieval(
                segment_start_indices,
                segment_end_indices,
                frame_indices,
                max_num_segments,
                box_jitter_ratio,
                cur_radius,
                regression_ranges[level_idx],
            )
        )
      multihot_target.append(cur_multihot_target)
      if normalize_displacements_by_downsample_stride:
        cur_displacements = cur_displacements / cur_downsample_stride
      displacements.append(cur_displacements)

    concat_axis = 0 if num_classes > 0 else 1
    multihot_target = tf.concat(multihot_target, axis=concat_axis)
    displacements = tf.concat(displacements, axis=concat_axis)
    feature_dict[output_label_name] = multihot_target
    feature_dict[output_displacement_name] = displacements

    return feature_dict

  preprocessor_builder.add_fn(
      lambda x, s=None: _add_labels(s, x), stateful=sync_random_state)


def add_background_labels(
    postprocessor_builder: builders.PostprocessorBuilder,
    input_feature_name: str,
):
  """Adds background labels."""

  def _add_background(
      feature_dict: builders.FeaturesDict,
  ) -> builders.FeaturesDict:
    multihot_labels = feature_dict[input_feature_name]
    background_labels = 1 - (
        tf.cast(
            tf.reduce_sum(multihot_labels, axis=-1, keepdims=True) > 0, tf.int32
        )
    )
    feature_dict[input_feature_name] = tf.concat(
        [multihot_labels, background_labels], axis=-1
    )
    return feature_dict

  postprocessor_builder.add_fn(_add_background)


def squeeze_features(
    postprocessor_builder: builders.PostprocessorBuilder,
    input_feature_name: str,
    axis: int,
):
  """Adds a postprocessor to squeeze features."""

  def _squeeze_feature(
      feature_dict: builders.FeaturesDict) -> builders.FeaturesDict:
    feature = feature_dict[input_feature_name]
    feature_dict[input_feature_name] = tf.squeeze(feature, axis=axis)
    return feature_dict

  postprocessor_builder.add_fn(_squeeze_feature)


def expand_features(
    postprocessor_builder: builders.PostprocessorBuilder,
    input_feature_name: str,
    axis: int,
):
  """Adds a postprocessor to expand features."""

  def _expand_feature(
      feature_dict: builders.FeaturesDict) -> builders.FeaturesDict:
    feature = feature_dict[input_feature_name]
    feature_dict[input_feature_name] = tf.expand_dims(feature, axis=axis)
    return feature_dict

  postprocessor_builder.add_fn(_expand_feature)


def linspace_sampling(x: tf.Tensor,
                      num_frames: int,
                      floor: bool = True,
                      state=None) -> tf.Tensor:
  """Applies linspace sampling."""

  del state
  if floor:
    indices = tf.cast(
        tf.math.floor(tf.linspace(0,
                                  tf.shape(x)[0] - 1, num_frames)), tf.int32)
  else:
    indices = tf.cast(
        tf.math.ceil(tf.linspace(0,
                                 tf.shape(x)[0] - 1, num_frames)), tf.int32)
  return tf.gather(x, indices)


def interpolate_embeddings(
    preprocessor_builder: builders.PreprocessorBuilder,
    num_frames: int,
    output_feature_name: str,
    total_length_feature_name: str,
):
  """Adds a function to interpolate embeddings.

  Args:
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    num_frames: Number of frames in the input sequence.
    output_feature_name: Name of the output embedding feature. The corresponding
      input feature names are, `output_feature_name`_floor and
      `output_feature_name`_ceil.
    total_length_feature_name: The feature name storing the total length of the
      video.
  """

  def _interpolate(
      feature_dict: builders.FeaturesDict) -> builders.FeaturesDict:
    embeddings_floor = feature_dict[f'{output_feature_name}_floor']
    embeddings_ceil = feature_dict[f'{output_feature_name}_ceil']
    indices = tf.linspace(
        0.0, tf.cast(feature_dict[total_length_feature_name] - 1, tf.float32),
        num_frames)
    weights_ceil = tf.cast(indices - tf.math.floor(indices), tf.float32)
    weights_floor = tf.cast(1.0 - weights_ceil, tf.float32)
    feature_dict[output_feature_name] = (
        embeddings_floor * weights_floor[:, None] +
        embeddings_ceil * weights_ceil[:, None])
    return feature_dict

  preprocessor_builder.add_fn(_interpolate)


def add_embeddings(
    parser_builder: builders.BaseParserBuilder,
    sampler_builder: builders.SamplerBuilder,
    input_feature_lists_name: str,
    output_feature_lists_name: str,
    num_frames: int,
    stride: int,
    feature_dim: int,
    sampling_strategy: str = 'linspace',
    is_training: bool = True,
    sync_random_state: bool = True,
):
  """Adds functions to process float feature lists.

  The input proto is expected to be tf.SequenceExample and its structure
  follows:

  feature_lists {
    feature_list {
      key: "input_feature_lists_name"
      value: {
        float_list: 0.0
        float_list: 0.0
        ...
      }
    }
  }

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    sampler_builder: An instance of a `builders.SamplerBuilder`.
    input_feature_lists_name: Name of the feature lists in the input
      `tf.train.SequenceExample`.
    output_feature_lists_name: Name of the feature lists in the output features
      dictionary.
    num_frames: Number of frames to sample.
    stride: Sampling stride.
    feature_dim: The dimension of the feature.
    sampling_strategy: Sampling strategy. Currently, we support 'linspace',
      'linspace_floor', 'linspace_ceil' and 'random'. When using 'linspace', we
      sample `num_frames` frames evenly from the input feature lists and when
      using 'random' we randomly sample N(=num_frames) consecutive frames during
      training and during testing the center of the sampled frames aligns with
      the input feature lists.
    is_training: Whether or not in training mode. This option only affects the
      behavior of `random` strategy.
    sync_random_state: Whether to use stateful option to keep random operations
      in sync.
  """
  if not isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    raise ValueError('add_embeddings only supports tf.SequenceExample.')

  parser_builder.parse_feature(
      feature_name=input_feature_lists_name,
      feature_type=tf.io.FixedLenSequenceFeature([feature_dim],
                                                 dtype=tf.float32),
      output_name=output_feature_lists_name)

  if sampling_strategy.startswith('linspace'):
    if sampling_strategy == 'linspace' or sampling_strategy == 'linspace_floor':
      floor = True
    else:
      floor = False
    sampler_builder.add_fn(
        # pylint: disable=g-long-lambda
        lambda x: linspace_sampling(x, num_frames=num_frames, floor=floor),
        # pylint: enable=g-long-lambda
        feature_name=output_feature_lists_name,
        fn_name=f'{output_feature_lists_name}_sample_feature_lists')
  elif sampling_strategy == 'random':
    sampler_builder.add_fn(
        # pylint: disable=g-long-lambda
        # The same key `sample_offset_proportion` used in sample_sequence
        # matches the one used in add_frame_labels_and_displacements().
        lambda x, s=None: processors.sample_sequence(
            x, num_frames, random=is_training, stride=stride, state=s),
        # pylint: enable=g-long-lambda
        feature_name=output_feature_lists_name,
        fn_name=f'{output_feature_lists_name}_sample_feature_lists',
        stateful=sync_random_state)
  else:
    raise ValueError(f'Sampling strategy {sampling_strategy} not supported.')


def read_strings_from_csv(input_csv: str) -> List[str]:
  """Reads strings from a CSV file."""
  strings = []
  with tf.io.gfile.GFile(input_csv, 'r') as fid:
    reader = csv.reader(fid, delimiter=',')
    for row in reader:
      strings.append(row[0])
  return strings


def read_string_embeddings(input_file: str) -> np.ndarray:
  """Reads string embeddings (query, class names) from a npy file."""
  with tf.io.gfile.GFile(input_file, 'rb') as f:
    embeddings = np.load(f)
  return embeddings


def init_tokenizer(
    config: ml_collections.ConfigDict,
) -> dmvr_tokenizers.TextTokenizer:
  """Initializes a text tokenizer."""
  if config.tokenizer_type == 't5':
    logging.info('Initialize T5 tokenizer.')
    tokenizer = t5_tokenizer.SentencePieceTokenizer(
        config.get('vocabulary_path', t5_tokenizer.SP_MODEL_PATH),
        bos_id=config.get('cls_token_id', 0),
    )
    tokenizer.initialize()
  elif config.tokenizer_type == 'clip':
    logging.info('Initialize CLIP tokenizer.')
    tokenizer = dmvr_tokenizers.ClipTokenizer(config.get('vocabulary_path'))
    tokenizer.initialize()
  else:
    raise ValueError(f'Unknown tokenizer_type: {config.tokenizer_type}.')
  return tokenizer


def tokenize_class_names(
    tokenizer: dmvr_tokenizers.TextTokenizer,
    config: ml_collections.ConfigDict,
    class_names: Sequence[str],
) -> np.ndarray:
  """Tokenizes class names."""
  class_name_ids = tokenizer.string_tensor_to_indices(
      class_names,
      prepend_bos=config.get('prepend_bos', True),
      append_eos=config.get('append_eos', True),
      max_num_tokens=config.max_num_tokens,
  )
  return class_name_ids.numpy()


def add_class_names(
    batch: Dict[str, np.ndarray],
    class_name_ids: np.ndarray,
    tokenizer: dmvr_tokenizers.TextTokenizer,
    exec_mode: str = 'train',
    num_prompts: int = 1,
) -> Dict[str, np.ndarray]:
  """Adds class name ids to the batch.

  Args:
    batch: A batch of input data.
    class_name_ids: A 2D array of tokenized class name ids. It has a shape of
      (num_prompts * num_classes, max_num_tokens).
    tokenizer: A text tokenizer.
    exec_mode: 'train', 'validation', or 'test'.
    num_prompts: Number of prompts.

  Returns:
    A new batch of input data with 'class_names'. batch['inputs']['class_names']
    has three fields, 'input_word_ids', 'input_type_ids', and 'input_mask' where
    'input_word_ids' is an int32 tensor of shape (batch_size,
    num_classes * num_prompts, max_num_tokens), 'input_type_ids' is an int32
    tensor of shape (batch_size, num_classes * num_prompts), and 'input_mask' is
    a binary tensor of shape (batch_size, num_classes * num_prompts).
  """
  assert num_prompts >= 1
  assert class_name_ids.shape[0] % num_prompts == 0
  num_classes = class_name_ids.shape[0] // num_prompts
  if exec_mode != 'test':
    class_name_ids = class_name_ids.reshape(
        (num_classes, num_prompts, class_name_ids.shape[-1])
    )
  if exec_mode == 'train':
    prompt_index = np.random.randint(num_prompts, size=num_classes)
    class_name_ids = class_name_ids[np.arange(num_classes), prompt_index]
  elif exec_mode == 'validation':
    prompt_index = num_prompts // 2
    class_name_ids = class_name_ids[:, prompt_index]
  class_name_ids = np.tile(
      np.expand_dims(class_name_ids, axis=0), [batch['label'].shape[0], 1, 1]
  )
  batch['inputs']['class_names'] = {
      'input_word_ids': class_name_ids,
      'input_type_ids': np.zeros_like(class_name_ids, dtype=np.int32),
      'input_mask': class_name_ids != tokenizer.pad_token,
  }
  return batch


def add_class_name_embeddings(
    batch: Dict[str, np.ndarray],
    class_name_embeddings: np.ndarray,
    exec_mode: str = 'train',
) -> Dict[str, np.ndarray]:
  """Adds class name embeddings to the batch.

  During training, we randomly select a prompt template. During validation, we
  always pick the middle prompt and in testing we use all prompts.

  Args:
    batch: A batch of input data.
    class_name_embeddings: A 3D array of class name embeddings. It has a shape
      of (num_classes, num_prompts, embedding_size).
    exec_mode: 'train', 'validation', or 'test'.

  Returns:
    A new batch of input data with a new field 'class_names', which stores the
    class name embeddings. The new field has a shape of (batch_size,
    num_classes, hidden_size) when exec_mode = 'train' or 'validation' and a
    shape of (batch_size, num_classes * num_prompts, hidden_size) when exec_mode
    = 'test'.
  """
  num_classes = class_name_embeddings.shape[0]
  num_prompts = class_name_embeddings.shape[1]
  if exec_mode == 'train':
    prompt_index = np.random.randint(num_prompts, size=num_classes)
    class_name_embeddings = class_name_embeddings[
        np.arange(num_classes), prompt_index
    ]
  elif exec_mode == 'validation':
    prompt_index = num_prompts // 2
    class_name_embeddings = class_name_embeddings[:, prompt_index]
  elif exec_mode == 'test':
    class_name_embeddings = class_name_embeddings.reshape(
        (num_classes * num_prompts, -1)
    )
  else:
    raise ValueError(f'Unknown exec_mode: {exec_mode}.')
  class_name_embeddings = np.tile(
      np.expand_dims(class_name_embeddings, axis=0),
      [batch['label'].shape[0], 1, 1],
  )
  batch['inputs']['class_names'] = np.asarray(class_name_embeddings)
  return batch


def add_image(
    parser_builder: builders.BaseParserBuilder,
    sampler_builder: builders.SamplerBuilder,
    decoder_builder: builders.DecoderBuilder,
    preprocessor_builder: builders.PreprocessorBuilder,
    postprocessor_builder: builders.PostprocessorBuilder,
    input_feature_name: str = 'image/encoded',
    output_feature_name: str = builders.IMAGE_FEATURE_NAME,
    is_training: bool = True,
    # Video related parameters.
    num_frames: int = 32,
    stride: int = 1,
    num_test_clips: int = 1,
    resize_method: str = tf.image.ResizeMethod.BILINEAR,
    target_size: Union[int, tuple[int, int]] = 224,
    zero_centering_image: bool = False,
    sync_random_state: bool = True,
    is_rgb: Optional[bool] = True,
    is_flow: bool = False,
    random_flip: bool = True,
    normalization_mean: Union[tf.Tensor, float] = 0,
    normalization_std: Union[tf.Tensor, float] = 1,
) -> None:
  """Adds functions to process image feature to builders.

  This function is branched from dmvr/modalities.py. The
  difference is the resizing method. In this method, we resize the image to a
  target size without keeping the aspect ratio while in the original one, the
  images are first resized based on the shorter side and then cropped to the
  target size. The same resizing method is used during Polymath's training.

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    sampler_builder: An instance of a `builders.SamplerBuilder`.
    decoder_builder: An instance of a `builders.DecoderBuilder`.
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    postprocessor_builder: An instance of a `builders.PostprocessorBuilder`.
    input_feature_name: Name of the feature in the input `tf.train.Example` or
      `tf.train.SequenceExample`. Exposing this as an argument allows using this
      function for different image features within a single dataset.
    output_feature_name: Name of the feature in the output features dictionary.
      Exposing this as an argument allows using this function for different
      image features within a single dataset.
    is_training: Whether in training mode. If `True`, random sample, crop and
      left right flip is used.
    num_frames: Number of frames per subclip. For single images, use 1.
    stride: Temporal stride to sample frames.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggregated in the batch dimension.
    resize_method: A resizing method.
    target_size: The final output size of the image.
    zero_centering_image: If `True`, frames are normalized to values in [-1, 1].
      If `False`, values in [0, 1].
    sync_random_state: Whether to use stateful option to keep random operations
      in sync between different modalities. All modalities having this option
      `True` will use the same outcome in random operations such as sampling and
      cropping.
    is_rgb: If `True`, the number of channels in the JPEG is 3, if False, 1. If
      is_flow is `True`, `is_rgb` should be set to `None` (see below).
    is_flow: If `True`, the image is assumed to contain flow and will be
      processed as such. Note that the number of channels in the JPEG for flow
      is 3, but only two channels will be output corresponding to the valid
      horizontal and vertical displacement.
    random_flip: If `True`, a random horizontal flip is applied to the input
      image. This augmentation may not be used if the label set contains
      direction related classes, such as `pointing left`, `pointing right`, etc.
    normalization_mean: value to subtract from the input image to normalize it.
    normalization_std: value to divide by from the input image to normalize it.
  """

  # Validate parameters.
  if is_flow and is_rgb is not None:
    raise ValueError('`is_rgb` should be `None` when requesting flow.')

  if is_flow and not zero_centering_image:
    raise ValueError(
        'Flow contains displacement values that can be negative, '
        'but `zero_centering_image` was set to `False`.'
    )

  if is_training and num_test_clips != 1:
    logging.info(
        '`num_test_clips` %d is ignored since `is_training` is true.',
        num_test_clips,
    )

  # Parse frames or single image.
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.FixedLenSequenceFeature((), dtype=tf.string),
        output_name=output_feature_name,
    )
  elif isinstance(parser_builder, builders.ExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.FixedLenFeature((), dtype=tf.string),
        output_name=output_feature_name,
    )
    # Expand dimensions so single images have the same structure as videos.
    sampler_builder.add_fn(
        fn=lambda x: tf.expand_dims(x, axis=0),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_expand_dims',
    )
  else:
    raise ValueError('`parser_builder` has an unexpected type.')

  # Temporal sampler.
  if is_training:
    # Sample random clip.
    sampler_builder.add_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x, s=None: processors.sample_sequence(
            x, num_frames, True, stride, state=s
        ),
        # pylint: enable=g-long-lambda
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_random_sample',
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state,
    )
  else:
    if num_test_clips > 1:
      # Sample linspace clips.
      sampler_builder.add_fn(
          # pylint: disable=g-long-lambda
          fn=lambda x: processors.sample_linspace_sequence(
              x, num_test_clips, num_frames, stride
          ),
          # pylint: enable=g-long-lambda
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_linspace_sample',
      )
    else:
      # Sample middle clip.
      sampler_builder.add_fn(
          fn=lambda x: processors.sample_sequence(x, num_frames, False, stride),
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_middle_sample',
      )

  # Decode JPEG string to `tf.uint8`.
  # Note that for flow, 3 channels are stored in the JPEG: the first two
  # corresponds to horizontal and vertical displacement, respectively.
  # The last channel contains zeros and is dropped later in the preprocessing.
  # Hence, the output number of channels for flow is 2.
  num_raw_channels = 3 if (is_rgb or is_flow) else 1
  decoder_builder.add_fn(
      fn=lambda x: processors.decode_jpeg(x, channels=num_raw_channels),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_decode_jpeg',
  )

  if is_flow:
    # Cast the flow to `tf.float32`, normalizing between [-1.0, 1.0].
    preprocessor_builder.add_fn(
        fn=lambda x: processors.normalize_image(x, zero_centering_image=True),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_normalize',
    )

  # Resize images (resize happens only if necessary to save compute).
  if isinstance(target_size, int):
    target_size = (target_size, target_size)
  preprocessor_builder.add_fn(
      fn=lambda x: tf.image.resize(x, target_size, method=resize_method),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_resize',
  )

  if is_training and random_flip:
    preprocessor_builder.add_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x, s=None: processors.random_flip_left_right(
            x, state=s, is_flow=is_flow
        ),
        # pylint: enable=g-long-lambda
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_random_flip',
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state,
    )

  if is_flow:
    # Keep only two channels for the flow: horizontal and vertical displacement.
    preprocessor_builder.add_fn(
        fn=lambda x: x[:, :, :, :2],
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_extract_flow_channels',
    )

    # Clip the flow to stay between [-1.0 and 1.0]
    preprocessor_builder.add_fn(
        fn=lambda x: tf.clip_by_value(x, -1.0, 1.0),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_clip_flow',
    )
  else:
    # Cast the frames to `tf.float32`, normalizing according to
    # `zero_centering_image`.
    preprocessor_builder.add_fn(
        fn=lambda x: processors.normalize_image(x, zero_centering_image),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_normalize',
    )

  preprocessor_builder.add_fn(
      fn=lambda x: x - normalization_mean,
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_subtract_given_mean',
  )

  preprocessor_builder.add_fn(
      fn=lambda x: x / normalization_std,
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_divide_by_given_std',
  )

  if num_test_clips > 1 and not is_training:
    # In this case, multiple clips are merged together in batch dimension which
    # will be `B * num_test_clips`.
    postprocessor_builder.add_fn(
        fn=lambda x: tf.reshape(  # pylint: disable=g-long-lambda
            x, (-1, num_frames, x.shape[2], x.shape[3], x.shape[4])
        ),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_reshape',
    )
