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

"""Contains dataset factory module for temporal localization use case."""

import os
from typing import Dict, List, Mapping, Optional, Union

from dmvr import modalities
from dmvr import tokenizers as dmvr_tokenizers
from dmvr import video_dataset
from mediapipe.util.sequence import media_sequence as ms
import ml_collections
from scenic.projects.unloc.datasets import dataset_utils as unloc_dataset_utils
from scenic.projects.vivit.data import video_tfrecord_dataset
import tensorflow as tf


class TemporalLocalizationDatasetFactory(video_dataset.BaseVideoDatasetFactory):
  """Reader for temporal localization datasets.

  We assume the examples are in MediaSequence format, frames and embeddings are
  already extracted.
  """

  def __init__(
      self,
      base_dir: str,
      tables: Dict[str, Union[str, List[str]]],
      examples_per_subset: Dict[str, int],
      subset: str = 'train',
      prop_data: float = 1.0,
      prop_seed: Optional[int] = None,
      num_groups: Optional[int] = None,
      group_index: Optional[int] = None,
  ):
    """Initializes the instance of TemporalLocalizationDatasetFactory.

    Initializes a data-loader using DeepMind Video Reader (DMVR) pre-processing
    (https://github.com/deepmind/dmvr).
    TFRecords are assumed to consist of tf.SequenceExample protocol buffers in
    the MediaSequence
    (https://github.com/google/mediapipe/tree/master/mediapipe/util/sequence)
    format.

    Args:
      base_dir: The base directory of the TFRecordss.
      tables: A dictionary mapping the subset name (train, val or test) to the
        relative path of the TFRecords containing them. Follows DMVR convention.
        The values of the dictionary can either be a string or a list. If it is
        a string, it specifies all the shards in the TFRecords. Example -
        "/path/to/tfrecord@10". If passing a list, each entry is a shard of the
        TFRecords. Example - "[/path/to/tfrecord_shard_1_of_10, ...,
        /path/to/sstabble_shard_10_of_10]." The latter scenario is useful for
        debugging.
      examples_per_subset:  A dictionary mapping the subset name (train, val or
        test) to the number of examples in the dataset for that subset.
      subset: The subset of the dataset to load. Must be a key of "tables"
      prop_data: The proportion of the data to load. If less than 1.0, this
        proportion of the total TFRecords shards are read.
      prop_seed: Whether to shuffle the shards (with the given seed) before
        choosing the data used (given the proportion).
      num_groups: If specified will reshard the data according to `num_groups`.
        A `group_index` should be specified if using `num_groups`.
      group_index: Index of the shard to return after resharding. `num_groups`
        should be specified if using `group_index`. This is useful in
        distributed setting where one wants to ensure that different data is
        read by different workers.
    """
    if (subset not in tables) or (subset not in examples_per_subset):
      raise ValueError(
          f'Invalid subset {subset!r}. '
          f'The available subsets are: {set(tables)!r}'
      )
    self._base_dir = base_dir
    self._subset = subset
    self._num_examples = examples_per_subset[subset]

    data_relative_path = tables[subset]
    if isinstance(data_relative_path, list):
      shards = [os.path.join(self._base_dir, x) for x in data_relative_path]
    else:
      data_path = os.path.join(self._base_dir, data_relative_path)
      shards = video_tfrecord_dataset.get_sharded_files(
          data_path=data_path,
          fraction_data=prop_data,
          num_groups=num_groups,
          group_index=group_index,
      )
    super().__init__(shards=shards)

  def _build(
      self,
      config: ml_collections.ConfigDict,
      is_training: bool = True,
  ):
    """Default build for this dataset.

    Args:
      config: A dataset config.
      is_training: Whether or not in training mode.
    """
    modality_types = set()
    for modality_name, modality_config in config.modality_configs.items():
      feature_type = modality_config.type
      modality_types.add(feature_type)
      if feature_type == 'embedding':
        if (
            modality_config.get('interpolate_embeddings', False)
            and config.get('sampling_strategy', 'linspace') == 'linspace'
        ):
          unloc_dataset_utils.add_embeddings(
              parser_builder=self.parser_builder,
              sampler_builder=self.sampler_builder,
              input_feature_lists_name=modality_config.input_feature_name,
              output_feature_lists_name=f'{modality_name}_floor',
              is_training=is_training,
              num_frames=config.num_frames,
              feature_dim=modality_config.feature_dimension,
              sampling_strategy='linspace_floor',
              stride=config.get('stride', 1),
              sync_random_state=True,
          )
          unloc_dataset_utils.add_embeddings(
              parser_builder=self.parser_builder,
              sampler_builder=self.sampler_builder,
              input_feature_lists_name=modality_config.input_feature_name,
              output_feature_lists_name=f'{modality_name}_ceil',
              is_training=is_training,
              num_frames=config.num_frames,
              feature_dim=modality_config.feature_dimension,
              sampling_strategy='linspace_ceil',
              stride=config.get('stride', 1),
              sync_random_state=True,
          )
          unloc_dataset_utils.interpolate_embeddings(
              self.preprocessor_builder,
              num_frames=config.num_frames,
              output_feature_name=modality_name,
              total_length_feature_name='total_frames',
          )
        else:
          unloc_dataset_utils.add_embeddings(
              parser_builder=self.parser_builder,
              sampler_builder=self.sampler_builder,
              input_feature_lists_name=modality_config.input_feature_name,
              output_feature_lists_name=modality_name,
              is_training=is_training,
              num_frames=config.num_frames,
              feature_dim=modality_config.feature_dimension,
              sampling_strategy=config.get('sampling_strategy', 'linspace'),
              stride=config.get('stride', 1),
              sync_random_state=True,
          )
      elif feature_type == 'video_embedding':
        unloc_dataset_utils.add_embeddings(
            parser_builder=self.parser_builder,
            sampler_builder=self.sampler_builder,
            input_feature_lists_name=modality_config.input_feature_name,
            output_feature_lists_name=modality_name,
            is_training=False,
            num_frames=1,
            feature_dim=modality_config.feature_dimension,
            sampling_strategy='random',
            stride=1,
            sync_random_state=False,
        )
      elif feature_type == 'rgb':
        if modality_config.get('resize_keep_aspect_ratio', True):
          modalities.add_image(
              parser_builder=self.parser_builder,
              sampler_builder=self.sampler_builder,
              decoder_builder=self.decoder_builder,
              preprocessor_builder=self.preprocessor_builder,
              postprocessor_builder=self.postprocessor_builder,
              input_feature_name='image/encoded',
              output_feature_name='rgb',
              is_training=is_training,
              random_flip=modality_config.get('random_flip', True),
              num_frames=config.num_frames,
              stride=config.get('stride', 1),
              num_test_clips=1,
              min_resize=modality_config.min_resize,
              crop_size=modality_config.crop_size,
              zero_centering_image=modality_config.get('zero_centering', True),
              normalization_mean=modality_config.get('normalization_mean', 0.0),
              normalization_std=modality_config.get('normalization_std', 1.0),
              is_rgb=True,
          )
        else:
          unloc_dataset_utils.add_image(
              parser_builder=self.parser_builder,
              sampler_builder=self.sampler_builder,
              decoder_builder=self.decoder_builder,
              preprocessor_builder=self.preprocessor_builder,
              postprocessor_builder=self.postprocessor_builder,
              input_feature_name=modality_config.get(
                  'input_feature_name', 'image/encoded'
              ),
              output_feature_name='rgb',
              is_training=is_training,
              random_flip=modality_config.get('random_flip', True),
              num_frames=config.num_frames,
              stride=config.get('stride', 1),
              num_test_clips=1,
              target_size=modality_config.crop_size,
              zero_centering_image=modality_config.get('zero_centering', True),
              normalization_mean=modality_config.get('normalization_mean', 0.0),
              normalization_std=modality_config.get('normalization_std', 1.0),
              is_rgb=True,
          )
      elif feature_type == 'flow':
        modalities.add_image(
            parser_builder=self.parser_builder,
            sampler_builder=self.sampler_builder,
            decoder_builder=self.decoder_builder,
            preprocessor_builder=self.preprocessor_builder,
            postprocessor_builder=self.postprocessor_builder,
            input_feature_name=modality_config.get(
                'input_feature_name', 'forward_flow/encoded'
            ),
            output_feature_name='flow',
            is_training=is_training,
            random_flip=modality_config.get('random_flip', True),
            num_frames=config.num_frames,
            stride=config.get('stride', 1),
            num_test_clips=1,
            min_resize=modality_config.min_resize,
            crop_size=modality_config.crop_size,
            zero_centering_image=True,
            sync_random_state=False,
            is_rgb=None,
            is_flow=True,
        )
      else:
        raise NotImplementedError(f'{feature_type} is not supported.')
    unloc_dataset_utils.add_pad_context_feature(
        self.parser_builder,
        self.decoder_builder,
        self.preprocessor_builder,
        input_context_feature_name=config.get(
            'segment_start_index_key', ms.SEGMENT_START_INDEX_KEY
        ),
        output_context_feature_name='segment_start_index',
        pad_value=-config.get('max_frame_index', 1000),
        dtype=tf.int64,
        max_feature_length=config.max_num_segments,
    )
    unloc_dataset_utils.add_pad_context_feature(
        self.parser_builder,
        self.decoder_builder,
        self.preprocessor_builder,
        input_context_feature_name=config.get(
            'segment_end_index_key', ms.SEGMENT_END_INDEX_KEY
        ),
        output_context_feature_name='segment_end_index',
        pad_value=-config.get('max_frame_index', 1000),
        dtype=tf.int64,
        max_feature_length=config.max_num_segments,
    )
    unloc_dataset_utils.add_pad_context_feature(
        self.parser_builder,
        self.decoder_builder,
        self.preprocessor_builder,
        input_context_feature_name=config.get(
            'segment_start_timestamp_key', ms.SEGMENT_START_TIMESTAMP_KEY
        ),
        output_context_feature_name='segment_start_timestamp',
        pad_value=-1,
        dtype=tf.int64,
        max_feature_length=config.max_num_segments,
    )
    unloc_dataset_utils.add_pad_context_feature(
        self.parser_builder,
        self.decoder_builder,
        self.preprocessor_builder,
        input_context_feature_name=config.get(
            'segment_end_timestamp_key', ms.SEGMENT_END_TIMESTAMP_KEY
        ),
        output_context_feature_name='segment_end_timestamp',
        pad_value=-1,
        dtype=tf.int64,
        max_feature_length=config.max_num_segments,
    )
    unloc_dataset_utils.add_pad_context_feature(
        self.parser_builder,
        self.decoder_builder,
        self.preprocessor_builder,
        input_context_feature_name=ms.SEGMENT_LABEL_INDEX_KEY,
        output_context_feature_name='segment_label_index',
        pad_value=-1,
        dtype=tf.int64,
        max_feature_length=config.max_num_segments,
    )
    unloc_dataset_utils.add_fixed_len_context_feature(
        self.parser_builder,
        input_context_feature_name='clip/frames',
        output_context_feature_name='total_frames',
        feature_dim=0,
        dtype=tf.int64,
    )
    unloc_dataset_utils.add_frame_labels_and_displacements(
        self.preprocessor_builder,
        segment_start_index_name='segment_start_index',
        segment_end_index_name='segment_end_index',
        segment_label_index_name='segment_label_index',
        total_length_name='total_frames',
        output_label_name='label',
        output_displacement_name='displacements',
        max_num_segments=config.max_num_segments,
        num_frames=config.num_frames,
        stride=config.get('stride', 1),
        num_classes=config.num_classes,
        sampling_strategy=config.get('sampling_strategy', 'linspace'),
        radius=config.get('radius', None),
        feature_pyramid_levels=config.get(
            'feature_pyramid_config.feature_pyramid_levels', None
        ),
        feature_pyramid_downsample_stride=config.get(
            'feature_pyramid_config.feature_pyramid_downsample_stride', 2
        ),
        regression_ranges=config.get(
            'feature_pyramid_config.regression_ranges', None
        ),
        normalize_displacements_by_downsample_stride=config.get(
            'feature_pyramid_config.normalize_displacements_by_downsample_stride',
            False,
        ),
        min_displacements_across_class=config.get(
            'min_displacements_across_class', False
        ),
        box_jitter_ratio=config.get('box_jitter_ratio', 0.0),
        is_training=is_training,
    )
    unloc_dataset_utils.add_input_mask(
        self.preprocessor_builder,
        total_length_name='total_frames',
        output_feature_name='input_mask',
        num_frames=config.num_frames,
        stride=config.get('stride', 1),
        sampling_strategy=config.get('sampling_strategy', 'linspace'),
        feature_pyramid_levels=config.get(
            'feature_pyramid_config.feature_pyramid_levels', None
        ),
        feature_pyramid_downsample_stride=config.get(
            'feature_pyramid_config.feature_pyramid_downsample_stride', 2
        ),
    )

    include_video_id = config.get('include_video_id', False)
    is_video_id_int = config.get('is_video_id_int', False)
    vid_input_feature_name = config.get('vid_input_feature_name',
                                        'clip/media_id')
    if include_video_id and self._subset == 'test':
      unloc_dataset_utils.add_fixed_len_context_feature(
          self.parser_builder,
          input_context_feature_name=vid_input_feature_name,
          output_context_feature_name='vid',
          feature_dim=0,
          dtype=tf.int64 if is_video_id_int else tf.string,
      )
    sampling_strategy = config.get('sampling_strategy', 'linspace')
    if sampling_strategy == 'linspace':
      # TODO(xxman): add random jittering to indices.
      def linspace_sampling(x, state=None):
        del state
        indices = tf.cast(
            tf.linspace(0,
                        tf.shape(x)[0] - 1, config.num_frames), tf.int32)
        return tf.gather(x, indices)

      for modality_type in modality_types:
        if modality_type in {'rgb', 'flow', 'spectrogram'}:
          if is_training:
            replace_sampling_fn_name = f'{modality_type}_random_sample'
          else:
            replace_sampling_fn_name = f'{modality_type}_middle_sample'
          self.sampler_builder.replace_fn(
              fn_name=replace_sampling_fn_name, fn=linspace_sampling
          )

  def get_num_examples(self) -> int:
    """Returns the number of examples in the TFRecordss."""
    return self._num_examples


class MomentRetrievalDatasetFactory(video_dataset.BaseVideoDatasetFactory):
  """Reader for moment retrieval datasets.

  In moment retrieval dataset, one query is associated with one segment while in
  temporal localization dataset each video could have multiple segments
  belonging to the same class. In training, we randomly select a segment within
  a sequence example. In testing, we assume each sequence example only contain
  one segment.
  """

  def __init__(
      self,
      base_dir: str,
      tables: Dict[str, Union[str, List[str]]],
      examples_per_subset: Dict[str, int],
      subset: str = 'train',
      prop_data: float = 1.0,
      prop_seed: Optional[int] = None,
      num_groups: Optional[int] = None,
      group_index: Optional[int] = None,
  ):
    """Initializes the instance of MomentRetrievalDatasetFactory.

    Initializes a data-loader using DeepMind Video Reader (DMVR) pre-processing
    (https://github.com/deepmind/dmvr).
    TFRecords are assumed to consist of tf.SequenceExample protocol buffers in
    the MediaSequence
    (https://github.com/google/mediapipe/tree/master/mediapipe/util/sequence)
    format.

    Args:
      base_dir: The base directory of the TFRecordss.
      tables: A dictionary mapping the subset name (train, val or test) to the
        relative path of the TFRecords containing them. Follows DMVR convention.
        The values of the dictionary can either be a string or a list. If it is
        a string, it specifies all the shards in the TFRecords. Example -
        "/path/to/tfrecord@10". If passing a list, each entry is a shard of the
        TFRecords. Example - "[/path/to/tfrecord_shard_1_of_10, ...,
        /path/to/sstabble_shard_10_of_10]." The latter scenario is useful for
        debugging.
      examples_per_subset:  A dictionary mapping the subset name (train, val or
        test) to the number of examples in the dataset for that subset.
      subset: The subset of the dataset to load. Must be a key of "tables"
      prop_data: The proportion of the data to load. If less than 1.0, this
        proportion of the total TFRecords shards are read.
      prop_seed: Whether to shuffle the shards (with the given seed) before
        choosing the data used (given the proportion).
      num_groups: If specified will reshard the data according to `num_groups`.
        A `group_index` should be specified if using `num_groups`.
      group_index: Index of the shard to return after resharding. `num_groups`
        should be specified if using `group_index`. This is useful in
        distributed setting where one wants to ensure that different data is
        read by different workers.
    """

    if (subset not in tables) or (subset not in examples_per_subset):
      raise ValueError(
          f'Invalid subset {subset!r}. '
          f'The available subsets are: {set(tables)!r}'
      )
    self._base_dir = base_dir
    self._subset = subset
    self._num_examples = examples_per_subset[subset]

    data_relative_path = tables[subset]
    if isinstance(data_relative_path, list):
      shards = [os.path.join(self._base_dir, x) for x in data_relative_path]
    else:
      data_path = os.path.join(self._base_dir, data_relative_path)
      shards = video_tfrecord_dataset.get_sharded_files(
          data_path=data_path,
          fraction_data=prop_data,
          num_groups=num_groups,
          group_index=group_index,
      )
    super().__init__(shards=shards)

  def _add_tokenized_text(
      self,
      tokenizers: Mapping[str, dmvr_tokenizers.TextTokenizer],
      modality_name: str,
      config: ml_collections.ConfigDict,
      max_num_captions: int,
      is_training: bool,
  ):
    """Adds tokenized text to the input pipeline."""

    modalities.add_text(
        parser_builder=self.parser_builder,
        decoder_builder=self.decoder_builder,
        preprocessor_builder=self.preprocessor_builder,
        input_feature_name=config.input_feature_name,
        output_feature_name=modality_name,
        output_raw_string_name=f'{modality_name}_raw_string',
        max_num_captions=max_num_captions,
        max_num_tokens=config.max_num_tokens,
        prepend_bos=config.get('prepend_bos', True),
        append_eos=config.get('append_eos', True),
        tokenizer=tokenizers[config.get('tokenizer_type', 'clip')],
        keep_raw_string=config.get('keep_raw_string', False),
        is_training=is_training,
        sync_random_state=True,
    )

  def _build(self,
             config: ml_collections.ConfigDict,
             tokenizers: Mapping[str, dmvr_tokenizers.TextTokenizer],
             is_training: bool = True):
    """Default build for this dataset.

    In the training split, we assume that a sequence example could have multiple
    segments but in the validation split a sequence example only has one
    segment. In training, a random segment is selected in each iteration and the
    same segment caption is selected by setting `sync_random_state = True`.

    Args:
      config: A dataset config.
      tokenizers: Mapping from tokenizer names to an instance of TextTokenizer.
      is_training: Whether or not in training mode.
    """

    unloc_dataset_utils.add_fixed_len_context_feature(
        self.parser_builder,
        input_context_feature_name='clip/frames',
        output_context_feature_name='total_frames',
        feature_dim=0,
        dtype=tf.int64,
    )
    modality_types = set()
    for modality_name, modality_config in config.modality_configs.items():
      feature_type = modality_config.type
      modality_types.add(feature_type)
      if feature_type == 'embedding':
        if (
            modality_config.get('interpolate_embeddings', False)
            and config.get('sampling_strategy', 'linspace') == 'linspace'
        ):
          unloc_dataset_utils.add_embeddings(
              parser_builder=self.parser_builder,
              sampler_builder=self.sampler_builder,
              input_feature_lists_name=modality_config.input_feature_name,
              output_feature_lists_name=f'{modality_name}_floor',
              is_training=is_training,
              num_frames=config.num_frames,
              feature_dim=modality_config.feature_dimension,
              sampling_strategy='linspace_floor',
              stride=config.get('stride', 1),
              sync_random_state=True,
          )
          unloc_dataset_utils.add_embeddings(
              parser_builder=self.parser_builder,
              sampler_builder=self.sampler_builder,
              input_feature_lists_name=modality_config.input_feature_name,
              output_feature_lists_name=f'{modality_name}_ceil',
              is_training=is_training,
              num_frames=config.num_frames,
              feature_dim=modality_config.feature_dimension,
              sampling_strategy='linspace_ceil',
              stride=config.get('stride', 1),
              sync_random_state=True,
          )
          unloc_dataset_utils.interpolate_embeddings(
              self.preprocessor_builder,
              num_frames=config.num_frames,
              output_feature_name=modality_name,
              total_length_feature_name='total_frames',
          )
        else:
          unloc_dataset_utils.add_embeddings(
              parser_builder=self.parser_builder,
              sampler_builder=self.sampler_builder,
              input_feature_lists_name=modality_config.input_feature_name,
              output_feature_lists_name=modality_name,
              is_training=is_training,
              num_frames=config.num_frames,
              feature_dim=modality_config.feature_dimension,
              sampling_strategy=config.get('sampling_strategy', 'linspace'),
              stride=config.get('stride', 1),
              sync_random_state=True,
          )
      elif feature_type == 'rgb':
        if modality_config.get('resize_keep_aspect_ratio', True):
          modalities.add_image(
              parser_builder=self.parser_builder,
              sampler_builder=self.sampler_builder,
              decoder_builder=self.decoder_builder,
              preprocessor_builder=self.preprocessor_builder,
              postprocessor_builder=self.postprocessor_builder,
              input_feature_name='image/encoded',
              output_feature_name='rgb',
              is_training=is_training,
              random_flip=modality_config.get('random_flip', True),
              num_frames=config.num_frames,
              stride=config.get('stride', 1),
              num_test_clips=1,
              min_resize=modality_config.min_resize,
              crop_size=modality_config.crop_size,
              zero_centering_image=modality_config.get('zero_centering', True),
              normalization_mean=modality_config.get('normalization_mean', 0.0),
              normalization_std=modality_config.get('normalization_std', 1.0),
              is_rgb=True,
          )
        else:
          unloc_dataset_utils.add_image(
              parser_builder=self.parser_builder,
              sampler_builder=self.sampler_builder,
              decoder_builder=self.decoder_builder,
              preprocessor_builder=self.preprocessor_builder,
              postprocessor_builder=self.postprocessor_builder,
              input_feature_name=modality_config.get(
                  'input_feature_name', 'image/encoded'
              ),
              output_feature_name='rgb',
              is_training=is_training,
              random_flip=modality_config.get('random_flip', True),
              num_frames=config.num_frames,
              stride=config.get('stride', 1),
              num_test_clips=1,
              target_size=modality_config.crop_size,
              zero_centering_image=modality_config.get('zero_centering', True),
              normalization_mean=modality_config.get('normalization_mean', 0.0),
              normalization_std=modality_config.get('normalization_std', 1.0),
              is_rgb=True,
          )
      elif feature_type == 'flow':
        modalities.add_image(
            parser_builder=self.parser_builder,
            sampler_builder=self.sampler_builder,
            decoder_builder=self.decoder_builder,
            preprocessor_builder=self.preprocessor_builder,
            postprocessor_builder=self.postprocessor_builder,
            input_feature_name=modality_config.get(
                'input_feature_name', 'forward_flow/encoded'
            ),
            output_feature_name='flow',
            is_training=is_training,
            random_flip=modality_config.get('random_flip', True),
            num_frames=config.num_frames,
            stride=config.get('stride', 1),
            num_test_clips=1,
            min_resize=modality_config.min_resize,
            crop_size=modality_config.crop_size,
            zero_centering_image=True,
            sync_random_state=False,
            is_rgb=None,
            is_flow=True,
        )
      elif modality_config.type == 'text':
        self._add_tokenized_text(
            tokenizers,
            modality_name,
            modality_config,
            config.get('train_max_num_captions', 1)
            if self._subset == 'train'
            else config.get('eval_max_num_captions', 1),
            is_training,
        )
      else:
        raise NotImplementedError(f'{feature_type} is not supported.')
    num_captions = (
        config.get('eval_max_num_captions', 1)
        if self._subset != 'train'
        else config.get('train_max_num_captions', 1)
    )
    unloc_dataset_utils.add_pad_context_feature(
        self.parser_builder,
        self.decoder_builder,
        self.preprocessor_builder,
        input_context_feature_name=config.get(
            'segment_start_index_key', ms.SEGMENT_START_INDEX_KEY
        ),
        output_context_feature_name='segment_start_index',
        pad_value=-config.get('max_frame_index', 1000),
        dtype=tf.int64,
        max_feature_length=num_captions,
        is_training=is_training,
    )
    unloc_dataset_utils.add_pad_context_feature(
        self.parser_builder,
        self.decoder_builder,
        self.preprocessor_builder,
        input_context_feature_name=config.get(
            'segment_end_index_key', ms.SEGMENT_END_INDEX_KEY
        ),
        output_context_feature_name='segment_end_index',
        pad_value=-config.get('max_frame_index', 1000),
        dtype=tf.int64,
        max_feature_length=num_captions,
        is_training=is_training,
    )
    if self._subset == 'test':
      unloc_dataset_utils.add_pad_context_feature(
          self.parser_builder,
          self.decoder_builder,
          self.preprocessor_builder,
          input_context_feature_name=config.get(
              'segment_start_timestamp_key', ms.SEGMENT_START_TIMESTAMP_KEY
          ),
          output_context_feature_name='segment_start_timestamp',
          pad_value=-int(1e6),
          dtype=tf.int64,
          max_feature_length=num_captions,
          is_training=False,
      )
      unloc_dataset_utils.add_pad_context_feature(
          self.parser_builder,
          self.decoder_builder,
          self.preprocessor_builder,
          input_context_feature_name=config.get(
              'segment_end_timestamp_key', ms.SEGMENT_END_TIMESTAMP_KEY
          ),
          output_context_feature_name='segment_end_timestamp',
          pad_value=-int(1e6),
          dtype=tf.int64,
          max_feature_length=num_captions,
          is_training=False,
      )
    unloc_dataset_utils.add_frame_labels_and_displacements(
        self.preprocessor_builder,
        segment_start_index_name='segment_start_index',
        segment_end_index_name='segment_end_index',
        segment_label_index_name=None,
        total_length_name='total_frames',
        output_label_name='label',
        output_displacement_name='displacements',
        max_num_segments=num_captions,
        num_frames=config.num_frames,
        stride=config.get('stride', 1),
        num_classes=-1,
        sampling_strategy=config.get('sampling_strategy', 'linspace'),
        radius=config.get('radius', None),
        feature_pyramid_levels=config.get(
            'feature_pyramid_config.feature_pyramid_levels', None
        ),
        feature_pyramid_downsample_stride=config.get(
            'feature_pyramid_config.feature_pyramid_downsample_stride', 2
        ),
        regression_ranges=config.get(
            'feature_pyramid_config.regression_ranges', None
        ),
        normalize_displacements_by_downsample_stride=config.get(
            'feature_pyramid_config.normalize_displacements_by_downsample_stride',
            False,
        ),
        box_jitter_ratio=config.get('box_jitter_ratio', 0.0),
        is_training=is_training,
    )
    unloc_dataset_utils.add_input_mask(
        self.preprocessor_builder,
        total_length_name='total_frames',
        output_feature_name='input_mask',
        num_frames=config.num_frames,
        stride=config.get('stride', 1),
        sampling_strategy=config.get('sampling_strategy', 'linspace'),
        feature_pyramid_levels=config.get(
            'feature_pyramid_config.feature_pyramid_levels', None
        ),
        feature_pyramid_downsample_stride=config.get(
            'feature_pyramid_config.feature_pyramid_downsample_stride', 2
        ),
    )
    unloc_dataset_utils.add_caption_mask(
        self.preprocessor_builder,
        input_feature_name='segment_start_index',
        padding_value=-config.get('max_frame_index', 1000),
        output_feature_name='caption_mask',
    )
    include_video_id = config.get('include_video_id', False)
    is_video_id_int = config.get('is_video_id_int', False)
    vid_input_feature_name = config.get('vid_input_feature_name',
                                        'clip/media_id')
    if include_video_id and self._subset == 'test':
      unloc_dataset_utils.add_fixed_len_context_feature(
          self.parser_builder,
          input_context_feature_name=vid_input_feature_name,
          output_context_feature_name='vid',
          feature_dim=0,
          dtype=tf.int64 if is_video_id_int else tf.string,
      )
    sampling_strategy = config.get('sampling_strategy', 'linspace')
    if sampling_strategy == 'linspace':
      # TODO(xxman): add random jittering to indices.
      def linspace_sampling(x, state=None):
        del state
        indices = tf.cast(
            tf.linspace(0, tf.shape(x)[0] - 1, config.num_frames), tf.int32
        )
        return tf.gather(x, indices)

      for modality_type in modality_types:
        if modality_type in {'rgb', 'flow', 'spectrogram'}:
          if is_training:
            replace_sampling_fn_name = f'{modality_type}_random_sample'
          else:
            replace_sampling_fn_name = f'{modality_type}_middle_sample'
          self.sampler_builder.replace_fn(
              fn_name=replace_sampling_fn_name, fn=linspace_sampling
          )

  def get_num_examples(self) -> int:
    """Returns the number of examples in the TFRecordss."""
    return self._num_examples


class HighlightDetectionDatasetFactory(MomentRetrievalDatasetFactory):
  """Reader for highlight detection datasets.

  Each video could have multiple highlight segments and those segments can
  associate with a text field, such as `query` or `video_title`.
  """

  def _build(
      self,
      config: ml_collections.ConfigDict,
      tokenizers: Mapping[str, dmvr_tokenizers.TextTokenizer],
      is_training: bool = True,
  ):
    """Default build for this dataset.

    Args:
      config: A dataset config.
      tokenizers: Mapping from tokenizer names to an instance of TextTokenizer.
      is_training: Whether or not in training mode.
    """
    modality_types = set()
    for modality_name, modality_config in config.modality_configs.items():
      feature_type = modality_config.type
      modality_types.add(feature_type)
      if feature_type == 'embedding':
        if (
            modality_config.get('interpolate_embeddings', False)
            and config.get('sampling_strategy', 'linspace') == 'linspace'
        ):
          unloc_dataset_utils.add_embeddings(
              parser_builder=self.parser_builder,
              sampler_builder=self.sampler_builder,
              input_feature_lists_name=modality_config.input_feature_name,
              output_feature_lists_name=f'{modality_name}_floor',
              is_training=is_training,
              num_frames=config.num_frames,
              feature_dim=modality_config.feature_dimension,
              sampling_strategy='linspace_floor',
              stride=config.get('stride', 1),
              sync_random_state=True,
          )
          unloc_dataset_utils.add_embeddings(
              parser_builder=self.parser_builder,
              sampler_builder=self.sampler_builder,
              input_feature_lists_name=modality_config.input_feature_name,
              output_feature_lists_name=f'{modality_name}_ceil',
              is_training=is_training,
              num_frames=config.num_frames,
              feature_dim=modality_config.feature_dimension,
              sampling_strategy='linspace_ceil',
              stride=config.get('stride', 1),
              sync_random_state=True,
          )
          unloc_dataset_utils.interpolate_embeddings(
              self.preprocessor_builder,
              num_frames=config.num_frames,
              output_feature_name=modality_name,
              total_length_feature_name='total_frames',
          )
        else:
          unloc_dataset_utils.add_embeddings(
              parser_builder=self.parser_builder,
              sampler_builder=self.sampler_builder,
              input_feature_lists_name=modality_config.input_feature_name,
              output_feature_lists_name=modality_name,
              is_training=is_training,
              num_frames=config.num_frames,
              feature_dim=modality_config.feature_dimension,
              sampling_strategy=config.get('sampling_strategy', 'linspace'),
              stride=config.get('stride', 1),
              sync_random_state=True,
          )
      elif feature_type == 'video_embedding':
        unloc_dataset_utils.add_embeddings(
            parser_builder=self.parser_builder,
            sampler_builder=self.sampler_builder,
            input_feature_lists_name=modality_config.input_feature_name,
            output_feature_lists_name=modality_name,
            is_training=False,
            num_frames=1,
            feature_dim=modality_config.feature_dimension,
            sampling_strategy='random',
            stride=1,
            sync_random_state=False,
        )
      elif feature_type == 'rgb':
        if modality_config.get('resize_keep_aspect_ratio', True):
          modalities.add_image(
              parser_builder=self.parser_builder,
              sampler_builder=self.sampler_builder,
              decoder_builder=self.decoder_builder,
              preprocessor_builder=self.preprocessor_builder,
              postprocessor_builder=self.postprocessor_builder,
              input_feature_name='image/encoded',
              output_feature_name='rgb',
              is_training=is_training,
              random_flip=modality_config.get('random_flip', True),
              num_frames=config.num_frames,
              stride=config.get('stride', 1),
              num_test_clips=1,
              min_resize=modality_config.min_resize,
              crop_size=modality_config.crop_size,
              zero_centering_image=modality_config.get('zero_centering', True),
              normalization_mean=modality_config.get('normalization_mean', 0.0),
              normalization_std=modality_config.get('normalization_std', 1.0),
              is_rgb=True,
          )
        else:
          unloc_dataset_utils.add_image(
              parser_builder=self.parser_builder,
              sampler_builder=self.sampler_builder,
              decoder_builder=self.decoder_builder,
              preprocessor_builder=self.preprocessor_builder,
              postprocessor_builder=self.postprocessor_builder,
              input_feature_name=modality_config.get(
                  'input_feature_name', 'image/encoded'
              ),
              output_feature_name='rgb',
              is_training=is_training,
              random_flip=modality_config.get('random_flip', True),
              num_frames=config.num_frames,
              stride=config.get('stride', 1),
              num_test_clips=1,
              target_size=modality_config.crop_size,
              zero_centering_image=modality_config.get('zero_centering', True),
              normalization_mean=modality_config.get('normalization_mean', 0.0),
              normalization_std=modality_config.get('normalization_std', 1.0),
              is_rgb=True,
          )
      elif feature_type == 'flow':
        modalities.add_image(
            parser_builder=self.parser_builder,
            sampler_builder=self.sampler_builder,
            decoder_builder=self.decoder_builder,
            preprocessor_builder=self.preprocessor_builder,
            postprocessor_builder=self.postprocessor_builder,
            input_feature_name=modality_config.get(
                'input_feature_name', 'forward_flow/encoded'
            ),
            output_feature_name='flow',
            is_training=is_training,
            random_flip=modality_config.get('random_flip', True),
            num_frames=config.num_frames,
            stride=config.get('stride', 1),
            num_test_clips=1,
            min_resize=modality_config.min_resize,
            crop_size=modality_config.crop_size,
            zero_centering_image=True,
            sync_random_state=False,
            is_rgb=None,
            is_flow=True,
        )
      elif feature_type == 'text':  # E.g., video title
        modalities.add_text(
            parser_builder=self.parser_builder,
            decoder_builder=self.decoder_builder,
            preprocessor_builder=self.preprocessor_builder,
            input_feature_name=modality_config.input_feature_name,
            output_feature_name=modality_name,
            output_raw_string_name=f'{modality_name}_raw_string',
            max_num_captions=1,
            max_num_tokens=modality_config.max_num_tokens,
            prepend_bos=modality_config.get('prepend_bos', True),
            append_eos=modality_config.get('append_eos', True),
            tokenizer=tokenizers[modality_config.get('tokenizer_type', 'clip')],
            keep_raw_string=modality_config.get('keep_raw_string', False),
            is_training=is_training,
        )
        # The feature has a shape of [batch, 1, max_num_words]. We squeeze it to
        # [batch, max_num_words].
        unloc_dataset_utils.squeeze_features(
            postprocessor_builder=self.postprocessor_builder,
            input_feature_name=modality_name,
            axis=1,
        )
      else:
        raise NotImplementedError(f'{feature_type} is not supported.')
    unloc_dataset_utils.add_pad_context_feature(
        self.parser_builder,
        self.decoder_builder,
        self.preprocessor_builder,
        input_context_feature_name=config.get(
            'segment_start_index_key', ms.SEGMENT_START_INDEX_KEY
        ),
        output_context_feature_name='segment_start_index',
        pad_value=-config.get('max_frame_index', 1000),
        dtype=tf.int64,
        max_feature_length=config.max_num_segments,
    )
    unloc_dataset_utils.add_pad_context_feature(
        self.parser_builder,
        self.decoder_builder,
        self.preprocessor_builder,
        input_context_feature_name=config.get(
            'segment_end_index_key', ms.SEGMENT_END_INDEX_KEY
        ),
        output_context_feature_name='segment_end_index',
        pad_value=-config.get('max_frame_index', 1000),
        dtype=tf.int64,
        max_feature_length=config.max_num_segments,
    )
    unloc_dataset_utils.add_pad_context_feature(
        self.parser_builder,
        self.decoder_builder,
        self.preprocessor_builder,
        input_context_feature_name=config.get(
            'segment_start_timestamp_key', ms.SEGMENT_START_TIMESTAMP_KEY
        ),
        output_context_feature_name='segment_start_timestamp',
        pad_value=-1,
        dtype=tf.int64,
        max_feature_length=config.max_num_segments,
    )
    unloc_dataset_utils.add_pad_context_feature(
        self.parser_builder,
        self.decoder_builder,
        self.preprocessor_builder,
        input_context_feature_name=config.get(
            'segment_end_timestamp_key', ms.SEGMENT_END_TIMESTAMP_KEY
        ),
        output_context_feature_name='segment_end_timestamp',
        pad_value=-1,
        dtype=tf.int64,
        max_feature_length=config.max_num_segments,
    )
    unloc_dataset_utils.add_pad_context_feature(
        self.parser_builder,
        self.decoder_builder,
        self.preprocessor_builder,
        input_context_feature_name=ms.SEGMENT_LABEL_INDEX_KEY,
        output_context_feature_name='segment_label_index',
        pad_value=-1,
        dtype=tf.int64,
        max_feature_length=config.max_num_segments,
    )
    unloc_dataset_utils.add_fixed_len_context_feature(
        self.parser_builder,
        input_context_feature_name='clip/frames',
        output_context_feature_name='total_frames',
        feature_dim=0,
        dtype=tf.int64,
    )
    unloc_dataset_utils.add_frame_labels_and_displacements(
        self.preprocessor_builder,
        segment_start_index_name='segment_start_index',
        segment_end_index_name='segment_end_index',
        segment_label_index_name='segment_label_index',
        total_length_name='total_frames',
        output_label_name='label',
        output_displacement_name='displacements',
        max_num_segments=config.max_num_segments,
        num_frames=config.num_frames,
        stride=config.get('stride', 1),
        num_classes=1,
        sampling_strategy=config.get('sampling_strategy', 'linspace'),
        radius=config.get('radius', None),
        feature_pyramid_levels=config.get(
            'feature_pyramid_config.feature_pyramid_levels', None
        ),
        feature_pyramid_downsample_stride=config.get(
            'feature_pyramid_config.feature_pyramid_downsample_stride', 2
        ),
        regression_ranges=config.get(
            'feature_pyramid_config.regression_ranges', None
        ),
        normalize_displacements_by_downsample_stride=config.get(
            'feature_pyramid_config.normalize_displacements_by_downsample_stride',
            False,
        ),
        min_displacements_across_class=config.get(
            'min_displacements_across_class', False
        ),
        box_jitter_ratio=config.get('box_jitter_ratio', 0.0),
        is_training=is_training,
    )
    if config.get('add_background', False):
      unloc_dataset_utils.add_background_labels(
          self.postprocessor_builder, input_feature_name='label'
      )
    unloc_dataset_utils.add_input_mask(
        self.preprocessor_builder,
        total_length_name='total_frames',
        output_feature_name='input_mask',
        num_frames=config.num_frames,
        stride=config.get('stride', 1),
        sampling_strategy=config.get('sampling_strategy', 'linspace'),
        feature_pyramid_levels=config.get(
            'feature_pyramid_config.feature_pyramid_levels', None
        ),
        feature_pyramid_downsample_stride=config.get(
            'feature_pyramid_config.feature_pyramid_downsample_stride', 2
        ),
    )

    include_video_id = config.get('include_video_id', False)
    is_video_id_int = config.get('is_video_id_int', False)
    vid_input_feature_name = config.get(
        'vid_input_feature_name', 'clip/media_id'
    )
    if include_video_id and self._subset == 'test':
      unloc_dataset_utils.add_fixed_len_context_feature(
          self.parser_builder,
          input_context_feature_name=vid_input_feature_name,
          output_context_feature_name='vid',
          feature_dim=0,
          dtype=tf.int64 if is_video_id_int else tf.string,
      )
    sampling_strategy = config.get('sampling_strategy', 'linspace')
    if sampling_strategy == 'linspace':
      # TODO(xxman): add random jittering to indices.
      def linspace_sampling(x, state=None):
        del state
        indices = tf.cast(
            tf.linspace(0, tf.shape(x)[0] - 1, config.num_frames), tf.int32
        )
        return tf.gather(x, indices)

      for modality_type in modality_types:
        if modality_type in {'rgb', 'flow', 'spectrogram'}:
          if is_training:
            replace_sampling_fn_name = f'{modality_type}_random_sample'
          else:
            replace_sampling_fn_name = f'{modality_type}_middle_sample'
          self.sampler_builder.replace_fn(
              fn_name=replace_sampling_fn_name, fn=linspace_sampling
          )


class ActionSegmentationDatasetFactory(TemporalLocalizationDatasetFactory):
  """Reader for action segmentation datasets.

  We assume the examples are in MediaSequence format, frames and embeddings are
  already extracted.
  """

  def _build(self, config: ml_collections.ConfigDict, is_training: bool = True):
    """Default build for this dataset.

    Args:
      config: A dataset config.
      is_training: Whether or not in training mode.
    """

    for modality_name, modality_config in config.modality_configs.items():
      feature_type = modality_config.type
      if feature_type == 'embedding':
        unloc_dataset_utils.add_embeddings(
            parser_builder=self.parser_builder,
            sampler_builder=self.sampler_builder,
            input_feature_lists_name=modality_config.input_feature_name,
            output_feature_lists_name=modality_name,
            is_training=is_training,
            num_frames=config.num_frames,
            feature_dim=modality_config.feature_dimension,
            sampling_strategy='random',
            stride=1,
            sync_random_state=True,
        )
      elif feature_type == 'rgb':
        if modality_config.get('resize_keep_aspect_ratio', True):
          modalities.add_image(
              parser_builder=self.parser_builder,
              sampler_builder=self.sampler_builder,
              decoder_builder=self.decoder_builder,
              preprocessor_builder=self.preprocessor_builder,
              postprocessor_builder=self.postprocessor_builder,
              input_feature_name='image/encoded',
              output_feature_name='rgb',
              is_training=is_training,
              random_flip=modality_config.get('random_flip', True),
              num_frames=config.num_frames,
              stride=1,
              num_test_clips=1,
              min_resize=modality_config.min_resize,
              crop_size=modality_config.crop_size,
              zero_centering_image=modality_config.get('zero_centering', True),
              normalization_mean=modality_config.get('normalization_mean', 0.0),
              normalization_std=modality_config.get('normalization_std', 1.0),
              is_rgb=True,
          )
        else:
          unloc_dataset_utils.add_image(
              parser_builder=self.parser_builder,
              sampler_builder=self.sampler_builder,
              decoder_builder=self.decoder_builder,
              preprocessor_builder=self.preprocessor_builder,
              postprocessor_builder=self.postprocessor_builder,
              input_feature_name='image/encoded',
              output_feature_name='rgb',
              is_training=is_training,
              random_flip=modality_config.get('random_flip', True),
              num_frames=config.num_frames,
              stride=1,
              num_test_clips=1,
              target_size=modality_config.crop_size,
              zero_centering_image=modality_config.get('zero_centering', True),
              normalization_mean=modality_config.get('normalization_mean', 0.0),
              normalization_std=modality_config.get('normalization_std', 1.0),
              is_rgb=True)
      elif feature_type == 'flow':
        modalities.add_image(
            parser_builder=self.parser_builder,
            sampler_builder=self.sampler_builder,
            decoder_builder=self.decoder_builder,
            preprocessor_builder=self.preprocessor_builder,
            postprocessor_builder=self.postprocessor_builder,
            input_feature_name=modality_config.get(
                'input_feature_name', 'forward_flow/encoded'
            ),
            output_feature_name='flow',
            is_training=is_training,
            random_flip=modality_config.get('random_flip', True),
            num_frames=config.num_frames,
            stride=config.get('stride', 1),
            num_test_clips=1,
            min_resize=modality_config.min_resize,
            crop_size=modality_config.crop_size,
            zero_centering_image=modality_config.get('zero_centering', True),
            normalization_mean=modality_config.get('normalization_mean', 0.0),
            normalization_std=modality_config.get('normalization_std', 1.0),
            sync_random_state=False,
            is_rgb=False,
            is_flow=True,
        )
      else:
        raise NotImplementedError(f'{feature_type} is not supported.')
    unloc_dataset_utils.add_pad_context_feature(
        self.parser_builder,
        self.decoder_builder,
        self.preprocessor_builder,
        input_context_feature_name=config.get(
            'segment_start_index_key', ms.SEGMENT_START_INDEX_KEY
        ),
        output_context_feature_name='segment_start_index',
        pad_value=-config.get('max_frame_index', 1000),
        dtype=tf.int64,
        max_feature_length=config.max_num_segments,
    )
    unloc_dataset_utils.add_pad_context_feature(
        self.parser_builder,
        self.decoder_builder,
        self.preprocessor_builder,
        input_context_feature_name=config.get(
            'segment_end_index_key', ms.SEGMENT_END_INDEX_KEY
        ),
        output_context_feature_name='segment_end_index',
        pad_value=-config.get('max_frame_index', 1000),
        dtype=tf.int64,
        max_feature_length=config.max_num_segments,
    )
    unloc_dataset_utils.add_pad_context_feature(
        self.parser_builder,
        self.decoder_builder,
        self.preprocessor_builder,
        input_context_feature_name=config.get(
            'segment_start_timestamp_key', ms.SEGMENT_START_TIMESTAMP_KEY
        ),
        output_context_feature_name='segment_start_timestamp',
        pad_value=-1,
        dtype=tf.int64,
        max_feature_length=config.max_num_segments,
    )
    unloc_dataset_utils.add_pad_context_feature(
        self.parser_builder,
        self.decoder_builder,
        self.preprocessor_builder,
        input_context_feature_name=config.get(
            'segment_end_timestamp_key', ms.SEGMENT_END_TIMESTAMP_KEY
        ),
        output_context_feature_name='segment_end_timestamp',
        pad_value=-1,
        dtype=tf.int64,
        max_feature_length=config.max_num_segments,
    )
    unloc_dataset_utils.add_pad_context_feature(
        self.parser_builder,
        self.decoder_builder,
        self.preprocessor_builder,
        input_context_feature_name=ms.SEGMENT_LABEL_INDEX_KEY,
        output_context_feature_name='segment_label_index',
        pad_value=-1,
        dtype=tf.int64,
        max_feature_length=config.max_num_segments,
    )
    unloc_dataset_utils.add_fixed_len_context_feature(
        self.parser_builder,
        input_context_feature_name='clip/frames',
        output_context_feature_name='total_frames',
        feature_dim=0,
        dtype=tf.int64,
    )
    unloc_dataset_utils.add_action_segmentation_labels(
        self.preprocessor_builder,
        segment_start_index_name='segment_start_index',
        segment_end_index_name='segment_end_index',
        segment_label_index_name='segment_label_index',
        total_length_name='total_frames',
        output_label_name='label',
        max_num_segments=config.max_num_segments,
        num_frames=config.num_frames,
        num_classes=config.num_classes,
        is_training=is_training,
    )
    unloc_dataset_utils.add_input_mask(
        self.preprocessor_builder,
        total_length_name='total_frames',
        output_feature_name='input_mask',
        num_frames=config.num_frames,
        stride=1,
        sampling_strategy='random',
        feature_pyramid_levels=None,
        feature_pyramid_downsample_stride=2,
    )

    include_video_id = config.get('include_video_id', False)
    is_video_id_int = config.get('is_video_id_int', False)
    vid_input_feature_name = config.get('vid_input_feature_name',
                                        'clip/media_id')
    if include_video_id and self._subset == 'test':
      unloc_dataset_utils.add_fixed_len_context_feature(
          self.parser_builder,
          input_context_feature_name=vid_input_feature_name,
          output_context_feature_name='vid',
          feature_dim=0,
          dtype=tf.int64 if is_video_id_int else tf.string,
      )
