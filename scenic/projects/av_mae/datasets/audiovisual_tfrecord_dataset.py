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

"""Data-loader to read from TFRecords using the MediaSequence format.

Forked from scenic/projects/vivit/data/video_tfrecord_dataset.py
"""

import functools
from typing import Any, Dict, Iterator, List, Optional, Text, Tuple, Union

from absl import logging
from dmvr import builders
from dmvr import modalities as data_utils
from dmvr import processors
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.dataset_lib import video_ops
import scenic.projects.av_mae.datasets.dataset_utils as dataset_util_avmae
from scenic.projects.vivit.data import video_tfrecord_dataset
import tensorflow as tf


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
Rng = Union[jnp.ndarray, Dict[str, jnp.ndarray]]


def maybe_pad_batch(batch: Batch, train: bool, batch_size: int,
                    return_as_dict: bool):
  """Zero pad the batch on the right to the batch_size."""
  if not return_as_dict:
    return dataset_utils.maybe_pad_batch(batch, train, batch_size)

  assert 'batch_mask' not in batch
  if 'rgb' in batch['inputs']:
    unpadded_mask_shape = batch['inputs']['rgb'].shape[0]
    batch_pad = batch_size - unpadded_mask_shape
  elif 'flow' in batch['inputs']:
    unpadded_mask_shape = batch['inputs']['flow'].shape[0]
    batch_pad = batch_size - unpadded_mask_shape
  elif 'spectrogram' in batch['inputs']:
    unpadded_mask_shape = batch['inputs']['spectrogram'].shape[0]
    batch_pad = batch_size - unpadded_mask_shape
  else:
    raise ValueError('invalid input batch')

  if train and batch_pad != 0:
    raise ValueError('In this codebase, we assumed that we always drop the '
                     'last partial batch of the train set. Please use '
                     '` drop_remainder=True` for the training set.')

  # Most batches will not need padding, so we quickly return to avoid slowdown.
  if train or batch_pad == 0:
    if 'batch_mask' not in batch:
      batch['batch_mask'] = np.ones(unpadded_mask_shape, dtype=np.float32)
    return batch

  def zero_pad(array):
    pad_with = [(0, batch_pad)] + [(0, 0)] * (array.ndim - 1)
    return np.pad(array, pad_with, mode='constant')

  padded_batch = jax.tree_util.tree_map(zero_pad, batch)
  padded_batch_mask = zero_pad(np.ones(unpadded_mask_shape, dtype=np.float32))
  padded_batch['batch_mask'] = padded_batch_mask
  return padded_batch


class DatasetFactory(video_tfrecord_dataset.TFRecordDatasetFactory):
  """Reader for TFRecords using the MediaSequence format.

  Attributes:
    num_classes: int. The number of classes in the dataset.
    base_dir: str. The base directory from which the TFRecords are read.
    subset: str. The subset of the dataset. In Scenic, the subsets are "train",
      "validation" and "test".
  """

  _MODALITIES = ('rgb', 'spectrogram')

  def __init__(self,
               base_dir: str,
               tables: Dict[str, Union[str, List[str]]],
               num_classes: int,
               subset: str = 'train',
               modalities: Tuple[str] = ('rgb',),
               prop_data: float = 1.0,
               prop_seed: Optional[int] = None,
               num_groups: Optional[int] = None,
               group_index: Optional[int] = None,
               examples_per_subset: Optional[Dict[str, int]] = None):
    """Initializes the instance of DatasetFactory.

    Initializes a data-loader using DeepMind Video Reader (DMVR) pre-processing.
    TFRecord files are assumed to consist of tf.SequenceExample protos in the
    MediaSequence format.

    Args:
      base_dir: The base directory of the TFRecord files.
      tables: A dictionary mapping the subset name (train, val or test) to the
        relative path of the TFRecord containing them. Follows DMVR convention.
        The values of the dictionary can either be a string or a list. If it is
        a string, it specifies all the shards in the TFRecord. Example -
        "/path/to/tfrecord@10". If passing a list, each entry is a shard of the
        TFRecord. Example - "[/path/to/tfrecord_shard_1_of_10, ...,
        /path/to/tfrecord_shard_10_of_10]." The latter scenario is useful for
        debugging.
      num_classes: The number of classes in the dataset.
      subset: The subset of the dataset to load. Must be a key of "tables"
      modalities: The modalities to be loaded.
      prop_data: The proportion of the data to load. If less than 1.0, this
        proportion of the total TFRecord shards are read.
      prop_seed: Whether to shuffle the shards (with the given seed) before
        choosing the data used (given the proportion).
      num_groups: If specified will reshard the data according to `num_groups`.
        A `group_index` should be specified if using `num_groups`.
      group_index: Index of the shard to return after resharding. `num_groups`
        should be specified if using `group_index`. This is useful in
        distributed setting where one wants to ensure that different data is
        read by different workers.
      examples_per_subset:  A dictionary mapping the subset name (train, val or
        test) to the number of examples in the dataset for that subset. If None,
        the number of entries in the TFRecord are counted manually. This flag is
        useful if the TFRecord file being read is large.
    """
    if examples_per_subset and subset in examples_per_subset:
      self._num_examples = examples_per_subset[subset]
      logging.info('Reading number of examples in subset %s from config: %d',
                   subset, self._num_examples)
    else:
      raise AssertionError('Number of examples per subset must be given.')

    for modality in modalities:
      if modality not in DatasetFactory._MODALITIES:
        raise ValueError('Invalid modality %s.' % modality)
    self._modalities = modalities

    super().__init__(
        base_dir=base_dir,
        tables=tables,
        examples_per_subset=examples_per_subset,
        subset=subset,
        num_classes=num_classes,
        fraction_data=prop_data,
        num_groups=num_groups,
        group_index=group_index)

  def _build(
      self,
      is_training: bool = True,
      # Video related parameters.
      num_frames: int = 32,
      stride: int = 1,
      num_test_clips: int = 1,
      min_resize: int = 256,
      crop_size: int = 224,
      resize_keep_aspect_ratio: bool = True,
      zero_centering_image: bool = False,
      random_flip: bool = True,
      normalization_mean: Union[tf.Tensor, float] = 0,
      normalization_std: Union[tf.Tensor, float] = 1,
      train_frame_sampling_mode: str = 'random',
      include_rgb: bool = True,
      use_crop_and_resize_video_mae: bool = False,
      include_flow: bool = False,
      # Label related parameters.
      one_hot_label: bool = True,
      get_label_str: bool = False,
      label_offset: int = 0,
      # Spectogram related parameters
      include_spectrogram: bool = False,
      num_spec_frames: int = 5,
      spec_stride: int = 1,
      spec_shape: Tuple[int, int] = (100, 128),
      spec_augment: bool = False,
      spec_augment_params: Optional[ml_collections.ConfigDict] = None,
      circular_time_shift: bool = False,
      inflate_spectrograms: bool = True,
      normalization_mean_spec: Union[tf.Tensor, float] = 0,
      normalization_std_spec: Union[tf.Tensor, float] = 1,
      ):
    """Adds DMVR pre-processors to the dataset.

    Args:
      is_training: whether in training mode.
      num_frames: number of frames per subclip.
      stride: temporal stride to sample frames.
      num_test_clips: number of test clip (1 by default). If more than one, this
        will sample multiple linearly spaced clips within each video at test
        time. If 1, then a single clip in the middle of the video is sampled.
      min_resize: frames are resized so that min width/height is min_resize.
      crop_size: final size of the frame after cropping the resized frames.
      resize_keep_aspect_ratio: If True, the image is first resized to have a
        shorter side equal to min_resize and then cropped to (crop_size,
        crop_size). Otherwise, the image is directly resized to (crop_size,
        crop_size).
      zero_centering_image: whether to have image values in the interval [-1, 1]
        or [0, 1].
      random_flip: Whether to perform horizontal flipping during training.
      normalization_mean: value to subtract from the input image to normalize
        it.
      normalization_std: value to divide by from the input image to normalize
        it.
      train_frame_sampling_mode: Method of sampling frames during training.
        Options are one of {random, random_sample_with_centre, centre}.
      include_rgb: Whether to include RGB images.
      use_crop_and_resize_video_mae: Whether to use the crop and resize function
      used in the VideoMAE paper.
      include_flow: Whether to include optical flow images.
      one_hot_label: whether to return one hot version of labels.
      get_label_str: whether to return label as text.
      label_offset: If non-zero, this value is subtracted from the parsed label.
        Useful when dataset is 1-indexed.
      include_spectrogram: Whether to include spectrogram.
      num_spec_frames: number of spectrogram frames.
      spec_stride: stride to sample spectrogram.
      spec_shape: input size of spectrogram per frame.
      spec_augment: whether to apply augmentation using SpecAugment.
      spec_augment_params: parameters for SpecAugment.
      circular_time_shift: If `True`, apply random time shift to spectrograms.
      inflate_spectrograms: whether or not to repeat the single spectrogram
        channel into 3 channels.
      normalization_mean_spec: value to subtract from the spectogram image to
        normalize it.
      normalization_std_spec: value to divide by from the spectogram image to
        normalize it.
    """

    if include_rgb:
      dataset_util_avmae.add_image(
          parser_builder=self.parser_builder,
          sampler_builder=self.sampler_builder,
          decoder_builder=self.decoder_builder,
          preprocessor_builder=self.preprocessor_builder,
          postprocessor_builder=self.postprocessor_builder,
          input_feature_name='image/encoded',
          output_feature_name=builders.IMAGE_FEATURE_NAME,
          is_training=is_training,
          random_flip=random_flip,
          num_frames=num_frames,
          stride=stride,
          num_test_clips=num_test_clips,
          min_resize=min_resize,
          crop_size=crop_size,
          use_crop_and_resize_video_mae=use_crop_and_resize_video_mae,
          train_frame_sampling_mode=train_frame_sampling_mode,
          zero_centering_image=zero_centering_image,
          normalization_mean=normalization_mean,
          normalization_std=normalization_std,
          is_rgb=True)

    if include_flow:
      data_utils.add_image(
          parser_builder=self.parser_builder,
          sampler_builder=self.sampler_builder,
          decoder_builder=self.decoder_builder,
          preprocessor_builder=self.preprocessor_builder,
          postprocessor_builder=self.postprocessor_builder,
          input_feature_name='FORWARD_FLOW/image/encoded',
          output_feature_name=builders.FLOW_FEATURE_NAME,
          is_training=is_training,
          random_flip=random_flip,
          num_frames=num_frames,
          stride=stride,
          num_test_clips=num_test_clips,
          min_resize=min_resize,
          crop_size=crop_size,
          zero_centering_image=True,
          sync_random_state=False,
          is_rgb=None,
          is_flow=True)

    if include_spectrogram:
      dataset_util_avmae.add_spectrogram(
          parser_builder=self.parser_builder,
          sampler_builder=self.sampler_builder,
          decoder_builder=self.decoder_builder,
          preprocessor_builder=self.preprocessor_builder,
          postprocessor_builder=self.postprocessor_builder,
          # TODO(lgeorgescu): make this a parameter
          input_feature_name='melspec/feature/floats',
          output_feature_name='spectrogram',
          input_shape=spec_shape,
          is_training=is_training,
          num_frames=num_spec_frames,
          stride=spec_stride,
          num_test_clips=num_test_clips,
          spec_augment=spec_augment,
          spec_augment_params=spec_augment_params,
          circular_time_shift=circular_time_shift,
          zero_centering_image=zero_centering_image,
          dataset_mean=normalization_mean_spec,
          dataset_stddev=normalization_std_spec,
          sync_random_state=True,
          inflate_spectrograms=inflate_spectrograms)

    if is_training and train_frame_sampling_mode != 'random':
      # We modify the data-processing graph after its construction, as upstream
      # changes to DMVR are not being accepted.
      logging.info('Train frame sampling mode is %s', train_frame_sampling_mode)

      def random_sampling_with_centre(x, state=None):
        return video_ops.random_sample_sequence_with_centre(
            x, num_frames, stride, state=state)

      def deterministic_sampling_from_centre(x, state=None):
        return processors.sample_sequence(
            x, num_frames, False, stride, state=state)

      def random_sampling_entire_interval(x, state=None):
        del state  # Parameter was required by caller API, but is unused.
        return dataset_util_avmae.sample_sequence_uniformly(
            x, num_frames)

      if train_frame_sampling_mode == 'random_sample_with_centre':
        sampling_function = random_sampling_with_centre
      elif train_frame_sampling_mode == 'centre':
        sampling_function = deterministic_sampling_from_centre
      elif train_frame_sampling_mode == 'segment':
        sampling_function = random_sampling_entire_interval
      else:
        raise AssertionError(
            f'Unknown train frame sampling mode {train_frame_sampling_mode}')

      self.sampler_builder.replace_fn(
          fn_name=f'{builders.IMAGE_FEATURE_NAME}_random_sample',
          fn=sampling_function)

    data_utils.add_label(
        parser_builder=self.parser_builder,
        decoder_builder=self.decoder_builder,
        preprocessor_builder=self.preprocessor_builder,
        one_hot_label=one_hot_label,
        num_classes=self.num_classes,
        add_label_name=get_label_str)

    if label_offset:
      self.preprocessor_builder.add_fn(
          fn=lambda x: x - label_offset,
          feature_name=builders.LABEL_INDEX_FEATURE_NAME,
          fn_name=f'label_offset_{label_offset}',
          add_before_fn_name=(f'{builders.LABEL_INDEX_FEATURE_NAME}_one_hot'))

  def get_num_examples(self) -> int:  # Override.
    """Returns the number of examples in the TFRecord files."""
    return self._num_examples


def load_split(ds_factory,
               batch_size: int,
               shuffle_buffer_size: int,
               modalities: Tuple[str] = ('rgb',),
               subset: Text = 'train',
               num_frames: int = 32,
               stride: int = 2,
               num_test_clips: int = 1,
               min_resize: int = 256,
               crop_size: int = 224,
               resize_keep_aspect_ratio: bool = True,
               one_hot_label: bool = True,
               zero_centering: bool = True,
               random_flip: bool = True,
               normalization_mean: Union[tf.Tensor, float] = 0.0,
               normalization_std: Union[tf.Tensor, float] = 1.0,
               get_label_str: bool = False,
               augmentation_params: Optional[ml_collections.ConfigDict] = None,
               keep_key: bool = False,
               do_three_spatial_crops: bool = False,
               label_offset: int = 0,
               train_frame_sampling_mode: str = 'random',
               include_rgb: bool = True,
               use_crop_and_resize_video_mae: bool = False,
               include_flow: bool = False,
               include_spectrogram: bool = True,
               num_spec_frames: int = 5,
               spec_stride: int = 1,
               spec_shape: Tuple[int, int] = (100, 128),
               spec_augment: bool = False,
               spec_augment_params: Optional[ml_collections.ConfigDict] = None,
               circular_time_shift=False,
               inflate_spectrograms: bool = True,
               normalization_mean_spec: Union[tf.Tensor, float] = 0,
               normalization_std_spec: Union[tf.Tensor, float] = 1,
               ) -> Tuple[tf.data.Dataset, int]:
  """Loads dataset using DMVR for pre-processing.

  DMVR dataset loader already does basic augmentation (random crop and flip in
    train mode). It also already shuffles and batches the data.

  Args:
    ds_factory: A DMVR factory to instantiate with the subset.
    batch_size: The batch_size to use.
    shuffle_buffer_size: The buffer size for shuffling the data.
    modalities: list of input modalities.
    subset: train, validation or test
    num_frames: Number of frames per subclip.
    stride: Temporal stride to sample frames.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggregated in the batch dimension.
    min_resize: Frames are resized so that min(height, width) is min_resize.
    crop_size: Final size of the frame after cropping the resized frames. Both
      height and width are the same.
    resize_keep_aspect_ratio: If True, the image is first resized to have a
      shorter side equal to min_resize and then cropped to (crop_size,
      crop_size). Otherwise, the image is directly resized to (crop_size,
      crop_size).
    one_hot_label: If True, return one-hot version of the labels (ie [N, C])
      array. Otherwise, return [N]-array of labels.
    zero_centering: If True, frames are normalized to values in the interval
      [-1, 1]. If False, values are in the interval [0, 1].
    random_flip: Whether to perform horizontal flipping during training.
    normalization_mean: value to subtract from the input image to normalize
      it.
    normalization_std: value to divide by from the input image to normalize
      it.
    get_label_str: Whether to return label as text. Note that strings cannot be
      used in pmapped functions in Jax!
    augmentation_params: Augmentation configurations in train mode.
    keep_key: bool; If true, also return the key for each example.
    do_three_spatial_crops: If true, take three spatial crops of each clip
      during testing.
    label_offset: If non-zero, this value is subtracted from the parsed label.
      Useful when dataset is 1-indexed.
    train_frame_sampling_mode: Method of sampling frames during training.
      Options are one of {random, random_sample_with_centre, centre}.
    include_rgb: Whether to include RGB images.
    use_crop_and_resize_video_mae: Whether to use the crop and resize function
      used in the VideoMAE paper.
    include_flow: Whether to include optical flow images.
    include_spectrogram: Whether to include spectrogram.
    num_spec_frames: Number of spectrogram frames per subclip.
    spec_stride: Temporal stride to sample spectrogram.
    spec_shape: Input size of spectrogram per frame.
    spec_augment: whether to apply augmentation using SpecAugment.
    spec_augment_params: dict; augmentation configurations for SpecAugment.
    circular_time_shift: If `True`, apply random time shift to spectrograms.
    inflate_spectrograms: whether or not to repeat the single spectrogram
      channel into 3 channels.
    normalization_mean_spec: value to subtract from the spectogram image to
      normalize it.
    normalization_std_spec: value to divide by from the spectogram image to
      normalize it.

  Returns:
    A pair `(ds, num_examples)` with
      ds: A `tf.data.Dataset` object
      num_examples: Number of examples in the dataset.
  """
  dataset = ds_factory(subset=subset, modalities=modalities).configure(
      is_training=(subset == 'train'),
      num_frames=num_frames,
      stride=stride,
      num_test_clips=num_test_clips,
      min_resize=min_resize,
      crop_size=crop_size,
      resize_keep_aspect_ratio=resize_keep_aspect_ratio,
      zero_centering_image=zero_centering,
      random_flip=random_flip,
      train_frame_sampling_mode=train_frame_sampling_mode,
      one_hot_label=one_hot_label,
      get_label_str=get_label_str,
      label_offset=label_offset,
      include_rgb=include_rgb,
      use_crop_and_resize_video_mae=use_crop_and_resize_video_mae,
      include_flow=include_flow,
      normalization_mean=normalization_mean,
      normalization_std=normalization_std,
      include_spectrogram=include_spectrogram,
      num_spec_frames=num_spec_frames,
      spec_stride=spec_stride,
      spec_shape=spec_shape,
      spec_augment=spec_augment,
      spec_augment_params=spec_augment_params,
      circular_time_shift=circular_time_shift,
      inflate_spectrograms=inflate_spectrograms,
      normalization_mean_spec=normalization_mean_spec,
      normalization_std_spec=normalization_std_spec
      )

  if subset == 'train' and augmentation_params:
    if include_rgb and include_flow:
      dataset = video_ops.additional_augmentations(
          dataset,
          augmentation_params.get('rgb'),
          crop_size,
          num_frames,
          zero_centering,
          rgb_feature_name=builders.IMAGE_FEATURE_NAME)
      dataset = video_ops.additional_augmentations(
          dataset,
          augmentation_params.get(builders.FLOW_FEATURE_NAME),
          crop_size,
          num_frames,
          zero_centering,
          rgb_feature_name=builders.FLOW_FEATURE_NAME)
    elif include_rgb:
      dataset = video_ops.additional_augmentations(
          dataset,
          augmentation_params,
          crop_size,
          num_frames,
          zero_centering,
          rgb_feature_name=builders.IMAGE_FEATURE_NAME)
    elif include_flow:
      dataset = video_ops.additional_augmentations(
          dataset,
          augmentation_params,
          crop_size,
          num_frames,
          zero_centering,
          rgb_feature_name=builders.FLOW_FEATURE_NAME)

  if subset != 'train' and do_three_spatial_crops and resize_keep_aspect_ratio:
    if include_rgb:
      dataset.preprocessor_builder.replace_fn(
          f'{builders.IMAGE_FEATURE_NAME}_central_crop',
          functools.partial(video_ops.three_spatial_crops, crop_size=crop_size))
    if include_flow:
      dataset.preprocessor_builder.replace_fn(
          f'{builders.FLOW_FEATURE_NAME}_central_crop',
          functools.partial(video_ops.three_spatial_crops, crop_size=crop_size))

    if num_test_clips == 1:
      # This means that reshaping is not part of the post-processing graph.
      if include_rgb:
        dataset.postprocessor_builder.add_fn(
            fn=lambda x: tf.reshape(  # pylint: disable=g-long-lambda
                x, (-1, num_frames, crop_size, crop_size, 3)),
            feature_name=builders.IMAGE_FEATURE_NAME,
            fn_name=f'{builders.IMAGE_FEATURE_NAME}_reshape')
      if include_flow:
        dataset.postprocessor_builder.add_fn(
            fn=lambda x: tf.reshape(  # pylint: disable=g-long-lambda
                x, (-1, num_frames, crop_size, crop_size, 2)),
            feature_name=builders.FLOW_FEATURE_NAME,
            fn_name=f'{builders.FLOW_FEATURE_NAME}_reshape')

  logging.info('Frame sampling graph: %s',
               dataset.sampler_builder.get_summary())
  logging.info('Preprocessing graph: %s',
               dataset.preprocessor_builder.get_summary())
  logging.info('Postprocessing graph: %s',
               dataset.postprocessor_builder.get_summary())
  num_examples = dataset.get_num_examples()

  if subset == 'train':
    dataset.tune(shuffle_buffer=shuffle_buffer_size)

  # Validation and test splits are a single epoch, so that the last batch
  # is padded with zeroes. This is then repeated.
  ds = dataset.make_dataset(
      batch_size=batch_size,
      shuffle=(subset == 'train'),
      num_epochs=None if (subset == 'train') else 1,
      drop_remainder=(subset == 'train'),
      keep_key=(subset != 'train' and keep_key))

  if subset != 'train':
    ds = ds.repeat(None)

  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  return ds, num_examples


def map_keys(batch: Batch,
             include_rgb: bool,
             include_flow: bool,
             include_spectrogram: bool,
             return_as_dict: bool = False) -> Batch:
  """DMVR dataset returns 'image' and 'label'. We want 'inputs' and 'label'."""

  if return_as_dict:
    batch['inputs'] = {}  # pytype: disable=container-type-mismatch  # jax-ndarray
    if include_rgb:
      batch['inputs']['rgb'] = batch.pop(builders.IMAGE_FEATURE_NAME)
    if include_flow:
      batch['inputs']['flow'] = batch.pop(builders.FLOW_FEATURE_NAME)
    if include_spectrogram:
      batch['inputs']['spectrogram'] = batch.pop('spectrogram')
  else:
    assert (include_rgb + include_flow + include_spectrogram) == 1
    if include_rgb:
      batch['inputs'] = batch.pop(builders.IMAGE_FEATURE_NAME)
    if include_flow:
      batch['inputs'] = batch.pop(builders.FLOW_FEATURE_NAME)
    if include_spectrogram:
      batch['inputs'] = batch.pop('spectrogram')
  return batch  # pytype: disable=bad-return-type  # jax-ndarray


def tile_label_key(batch: Batch,
                   include_rgb: bool,
                   include_flow: bool,
                   include_spectrogram: bool,
                   return_as_dict: bool = False) -> Batch:
  """Tile labels and keys to match input videos when num_test_clips > 1.

  When multiple test crops are used (ie num_test_clips > 1), the batch dimension
  of batch['inputs'] = test_batch_size * num_test_clips.
  However, labels and keys remain of size [test_batch_size].
  This function repeats label and key to match the inputs.

  Args:
    batch: Batch from iterator
    include_rgb: Whether to include RGB images.
    include_flow: Whether to include optical flow images.
    include_spectrogram: Whether to include spectrogram.
    return_as_dict: Whether to return inputs as a dict.

  Returns:
    Batch with 'label' and 'key' tiled to match 'inputs'. The input batch is
    mutated by the function.
  """
  if return_as_dict:
    assert include_rgb or include_flow or include_spectrogram, (
        '"include_rgb", "include_flow" or "include_spectrogram" must be True.')
    if include_rgb:
      n_repeats = batch['inputs']['rgb'].shape[0] // batch['label'].shape[0]
    elif include_flow:
      n_repeats = batch['inputs']['flow'].shape[0] // batch['label'].shape[0]
    elif include_spectrogram:
      n_repeats = (
          batch['inputs']['spectrogram'].shape[0] // batch['label'].shape[0])
  else:
    n_repeats = batch['inputs'].shape[0] // batch['label'].shape[0]

  batch['label'] = np.repeat(batch['label'], n_repeats, axis=0)
  if 'key' in batch:
    batch['key'] = np.repeat(batch['key'], n_repeats, axis=0)
  return batch


def reshape_spectrogram(batch: Dict[str, Any], spec_shape: Tuple[int, int],
                        num_frames: int):
  batch['spectrogram'] = np.reshape(
      batch['spectrogram'], (-1, num_frames, spec_shape[0], spec_shape[1],
                             batch['spectrogram'].shape[-1]))
  return batch


@datasets.add_dataset('avmae_audiovisual_tfrecord_dataset')
def get_dataset(
    *,
    batch_size: int,
    eval_batch_size: int,
    num_shards: int,
    dtype_str: Text = 'float32',
    shuffle_seed: Optional[int] = 0,
    rng: Optional[Rng] = None,
    dataset_configs: ml_collections.ConfigDict,
    dataset_service_address: Optional[str] = None) -> dataset_utils.Dataset:
  """Returns a generator for dataset."""
  del rng  # Parameter was required by caller API, but is unused.

  shuffle_buffer_size = dataset_configs.get('shuffle_buffer_size', 256)
  modalities = dataset_configs.get('modalities', ['rgb'])
  num_frames = dataset_configs.get('num_frames', 32)
  num_test_clips = dataset_configs.get('num_test_clips', 1)
  stride = dataset_configs.get('stride', 2)
  min_resize_train = dataset_configs.get('min_resize_train')
  min_resize_test = dataset_configs.get('min_resize_test')
  crop_size = dataset_configs.get('crop_size', 224)
  resize_keep_aspect_ratio = dataset_configs.get('resize_keep_aspect_ratio',
                                                 True)
  one_hot_label = dataset_configs.get('one_hot_label', True)
  zero_centre_data = dataset_configs.get('zero_centering', True)
  random_flip = dataset_configs.get('random_flip', True)
  augmentation_params = dataset_configs.get('augmentation_params', None)
  num_train_val_clips = dataset_configs.get('num_train_val_clips', 1)
  keep_test_key = dataset_configs.get('keep_test_key', False)
  # For the test set, the actual batch size is test_batch_size*num_test_clips.
  test_batch_size = dataset_configs.get('test_batch_size', eval_batch_size)
  do_three_spatial_crops = dataset_configs.get('do_three_spatial_crops', False)
  num_spatial_crops = 3 if do_three_spatial_crops else 1
  test_split = dataset_configs.get('test_split', 'test')
  label_offset = dataset_configs.get('label_offset', 0)
  train_frame_sampling_mode = dataset_configs.get('train_frame_sampling_mode',
                                                  'random')
  examples_per_subset = dataset_configs.get('examples_per_subset', None)
  include_flow = 'flow' in modalities
  include_rgb = 'rgb'in modalities
  use_crop_and_resize_video_mae = augmentation_params.get(
      'crop_and_resize_video_mae', False) if (augmentation_params
                                              is not None)  else False
  return_as_dict = dataset_configs.get('return_as_dict', False)

  normalization_mean = dataset_configs.get('normalization_mean', 0)
  normalization_std = dataset_configs.get('normalization_std', 1)
  if isinstance(normalization_mean, (list, tuple)):
    normalization_mean = tf.constant(normalization_mean, tf.float32)
  if isinstance(normalization_std, (list, tuple)):
    normalization_std = tf.constant(normalization_std, tf.float32)

  # Spectrogram related configs.
  include_spectrogram = 'spectrogram'in modalities
  num_spec_frames = dataset_configs.get('num_spec_frames', 5)
  spec_stride = dataset_configs.get('spec_stride', 1)
  spec_shape = dataset_configs.get('spec_shape', (100, 128))
  spec_augment = dataset_configs.get('spec_augment', False)
  spec_augment_params = dataset_configs.get('spec_augment_params', None)
  circular_time_shift = dataset_configs.get('circular_time_shift', False)
  return_spec_as_2d = dataset_configs.get('return_spec_as_2d', True)
  inflate_spectrograms = dataset_configs.get('inflate_spectrograms', True)
  normalization_mean_spec = dataset_configs.get('normalization_mean_spec', 0)
  normalization_std_spec = dataset_configs.get('normalization_std_spec', 1)

  if dataset_configs.get('base_dir') is None:
    raise ValueError('base_dir must be specified for the dataset')
  if not dataset_configs.get('tables'):
    raise ValueError('tables mapping must be specified for the dataset')
  if not dataset_configs.get('num_classes'):
    raise ValueError('num_classes must be specified for the dataset')

  ds_factory = functools.partial(
      DatasetFactory,
      base_dir=dataset_configs.base_dir,
      tables=dataset_configs.tables,
      num_classes=dataset_configs.num_classes,
      num_groups=jax.process_count(),
      group_index=jax.process_index(),
      examples_per_subset=examples_per_subset)

  def create_dataset_iterator(
      subset: str,
      batch_size_local: int,
      num_clips: int,
      keep_key_local: bool = False,
      is_test: bool = False) -> Tuple[Iterator[Batch], int]:
    is_training = subset == 'train'
    is_test = (subset == 'test' or is_test)
    logging.info('Loading split %s', subset)

    dataset, num_examples = load_split(
        ds_factory,
        batch_size=batch_size_local,
        shuffle_buffer_size=shuffle_buffer_size,
        subset=subset,
        num_frames=num_frames,
        stride=stride,
        num_test_clips=num_clips,
        min_resize=min_resize_train if is_training else min_resize_test,
        crop_size=crop_size,
        resize_keep_aspect_ratio=resize_keep_aspect_ratio,
        one_hot_label=one_hot_label,
        zero_centering=zero_centre_data,
        random_flip=random_flip,
        augmentation_params=augmentation_params,
        keep_key=keep_key_local,
        do_three_spatial_crops=do_three_spatial_crops and is_test,
        label_offset=label_offset,
        train_frame_sampling_mode=train_frame_sampling_mode,
        include_rgb=include_rgb,
        use_crop_and_resize_video_mae=use_crop_and_resize_video_mae,
        include_flow=include_flow,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        include_spectrogram=include_spectrogram,
        num_spec_frames=num_spec_frames,
        spec_stride=spec_stride,
        spec_shape=spec_shape,
        spec_augment=spec_augment,
        spec_augment_params=spec_augment_params,
        circular_time_shift=circular_time_shift,
        inflate_spectrograms=inflate_spectrograms,
        normalization_mean_spec=normalization_mean_spec,
        normalization_std_spec=normalization_std_spec
        )

    if dataset_service_address and is_training:
      if shuffle_seed is not None:
        raise ValueError('Using dataset service with a random seed causes each '
                         'worker to produce exactly the same data. Add '
                         'config.shuffle_seed = None to your config if you '
                         'want to run with dataset service.')
      logging.info('Using the tf.data service at %s', dataset_service_address)
      dataset = dataset_utils.distribute(dataset, dataset_service_address)

    pad_batch_size = batch_size_local * num_train_val_clips

    if is_test:
      pad_batch_size = batch_size_local * num_clips * num_spatial_crops
    shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)

    current_ds_iterator = iter(dataset)
    current_ds_iterator = map(dataset_utils.tf_to_numpy, current_ds_iterator)
    current_ds_iterator = map(
        functools.partial(
            map_keys,
            include_rgb=include_rgb,
            include_flow=include_flow,
            include_spectrogram=include_spectrogram,
            return_as_dict=return_as_dict), current_ds_iterator)
    if is_test and num_clips * num_spatial_crops > 1:
      current_ds_iterator = map(
          functools.partial(
              tile_label_key,
              include_rgb=include_rgb,
              include_flow=include_flow,
              include_spectrogram=include_spectrogram,
              return_as_dict=return_as_dict), current_ds_iterator)
    current_ds_iterator = map(
        functools.partial(
            maybe_pad_batch,
            train=is_training,
            batch_size=pad_batch_size,
            return_as_dict=return_as_dict), current_ds_iterator)

    if is_training and augmentation_params and augmentation_params.get(
        'do_mixup', False):
      mixup_alpha = augmentation_params.get('mixup_alpha', 1.0)
      mixup_batches = functools.partial(
          dataset_utils.mixup, alpha=mixup_alpha, image_format='NTHWC')
      logging.info('Doing mixup with alpha %f', mixup_alpha)
      current_ds_iterator = map(mixup_batches, current_ds_iterator)
    current_ds_iterator = map(shard_batches, current_ds_iterator)
    if is_training and dataset_configs.get('prefetch_to_device'):
      # Async bind batch to device which speeds up training.
      current_ds_iterator = jax_utils.prefetch_to_device(
          current_ds_iterator, dataset_configs.get('prefetch_to_device'))

    if not return_spec_as_2d:
      current_ds_iterator = map(
          functools.partial(
              reshape_spectrogram,
              spec_shape=spec_shape,
              num_frames=num_spec_frames), current_ds_iterator)

    return current_ds_iterator, num_examples

  train_iter, n_train_examples = create_dataset_iterator(
      'train', batch_size, num_train_val_clips)
  eval_iter, n_eval_examples = create_dataset_iterator('validation',
                                                       eval_batch_size,
                                                       1)
  test_iter, n_test_examples = create_dataset_iterator(test_split,
                                                       test_batch_size,
                                                       num_test_clips,
                                                       keep_test_key,
                                                       is_test=True)

  meta_data = {
      'num_classes': dataset_configs.num_classes,
      'num_train_examples': n_train_examples * num_train_val_clips,
      'num_eval_examples': n_eval_examples * num_train_val_clips,
      'num_test_examples':
          (n_test_examples * num_test_clips * num_spatial_crops),
      'target_is_onehot': one_hot_label,
  }

  channels_spectogram = 3 if inflate_spectrograms else 1

  if return_as_dict:
    meta_data['input_shape'] = {}
    meta_data['input_dtype'] = {}
    if include_rgb:
      meta_data['input_shape']['rgb'] = (-1, num_frames, crop_size, crop_size,
                                         3)
      meta_data['input_dtype']['rgb'] = getattr(jnp, dtype_str)
    if include_flow:
      meta_data['input_shape']['flow'] = (-1, num_frames, crop_size, crop_size,
                                          2)
      meta_data['input_dtype']['flow'] = getattr(jnp, dtype_str)

    if include_spectrogram:
      meta_data['input_shape']['spectrogram'] = (  # pylint:disable=g-long-ternary
          -1, num_spec_frames * spec_shape[0], spec_shape[1],
          channels_spectogram) if return_spec_as_2d else (
              -1, num_spec_frames, spec_shape[0], spec_shape[1],
              channels_spectogram)
      meta_data['input_dtype']['spectrogram'] = getattr(jnp, dtype_str)

  else:
    raise ValueError('Only returning a dictionary of inputs is supported.')

  logging.info('Dataset metadata:\n%s', meta_data)

  return dataset_utils.Dataset(train_iter, eval_iter, test_iter, meta_data)
