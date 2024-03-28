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

"""Data-loader to read from SSTables using the MediaSequence format."""

import functools
import os
from typing import Dict, Iterator, List, Optional, Text, Tuple, Union

from absl import logging
from dmvr import builders
from dmvr import modalities
from dmvr import video_dataset
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.dataset_lib import video_ops
from scenic.projects.vivit.data import file_utils
import tensorflow as tf

# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
Rng = Union[jnp.ndarray, Dict[str, jnp.ndarray]]


def get_sharded_files(
    data_path: str,
    fraction_data: float = 1.0,
    num_groups: Optional[int] = None,
    group_index: Optional[int] = None) -> List[str]:
  """Returns a list of shards, which may be postprocessed.

  Args:
    data_path: Path to the data, either sharded or a single file.
    fraction_data: Fraction of the data to be consumed. Only that fraction of
      the shards is returned.
    num_groups: Number of groups to split the data. All the shards will be split
      in `num_groups` groups (of approximately same number of files) and the
      given `group_index` group only will be returned. This is useful when
      distributing the data over multiple hosts, which will make sure that the
      same shard is not processed in two different hosts. If `num_groups` is
      provided `group_index` must be provided as well.
    group_index: Index of the group of data being returned. See `num_groups`.

  Returns:
    A list of shard filenames.

  Raises:
    ValueError: If `fraction_data` is not between 0 and 1.
    ValueError: If `num_groups` requested is not consistent with the number of
      shards available.
    ValueError: If `group_index` >= `num_groups`
    ValueError: If only one of `num_groups` and `group_index` is given.
  """
  if fraction_data <= 0 or fraction_data > 1:
    raise ValueError(
        f'The fraction of data must be in (0, 1] but is {fraction_data}.')

  if file_utils.is_sharded_file_spec(data_path):
    shards = list(file_utils.generate_sharded_filenames(data_path))
  else:
    shards = [data_path]

  num_used_shards = int(np.ceil(fraction_data * len(shards)))
  shards = shards[:num_used_shards]

  if num_groups is None and group_index is None:
    return shards
  if num_groups is None or group_index is None:
    raise ValueError('Both `num_groups` and `group_index` should be specified.')
  if group_index >= num_groups:
    raise ValueError(
        f'Cannot request index {group_index} of {num_groups} groups')
  if num_groups > num_used_shards:
    raise ValueError(
        f'After applying `fraction_data={fraction_data}` we have '
        f'{num_used_shards} data shards, which cannot be split into '
        f'{num_groups} groups.')

  split_shard_ids = np.array_split(np.arange(num_used_shards), num_groups)
  begin_loc = split_shard_ids[group_index][0]
  end_loc = split_shard_ids[group_index][-1] + 1
  shards = shards[begin_loc:end_loc]
  return shards


class TFRecordDatasetFactory(video_dataset.BaseVideoDatasetFactory):
  """Reader for TFRecords using the MediaSequence format.

  Attributes:
    num_classes: int. The number of classes in the dataset.
    base_dir: str. The base directory from which the TFRecords are read.
    subset: str. The subset of the dataset. In Scenic, the subsets are
      "train", "validation" and "test".
  """

  def __init__(
      self,
      base_dir: str,
      tables: Dict[str, Union[str, List[str]]],
      examples_per_subset: Dict[str, int],
      num_classes: int,
      subset: str = 'train',
      fraction_data: float = 1.0,
      num_groups: Optional[int] = None,
      group_index: Optional[int] = None):
    """Initializes the instance of TFRecordDatasetFactory.

    Initializes a data-loader using DeepMind Video Reader (DMVR) pre-processing
    (https://github.com/deepmind/dmvr).
    TFRecords are assumed to consist of tf.SequenceExample protocol buffers in
    the MediaSequence
    (https://github.com/google/mediapipe/tree/master/mediapipe/util/sequence)
    format.

    Args:
      base_dir: The base directory of the TFRecords.
      tables: A dictionary mapping the subset name (train, val or test) to the
        relative path of the TFRecord containing them. Follows DMVR convention.
        The values of the dictionary can either be a string or a list. If it is
        a string, it specifies all the shards in the TFRecord.
        Example - "/path/to/tfrecord@10".
        If passing a list, each entry is a shard of the TFRecord.
        Example - "[/path/to/tfrecord_shard_1_of_10, ...,
                    /path/to/tfrecord_shard_10_of_10]."
        The latter scenario is useful for debugging.
      examples_per_subset:  A dictionary mapping the subset name (train, val or
        test) to the number of examples in the dataset for that subset.
      num_classes: The number of classes in the dataset.
      subset: The subset of the dataset to load. Must be a key of "tables"
      fraction_data: The fraction of the data to load. If less than 1.0, this
        fraction of the total TFRecord shards are read.
      num_groups: If specified will reshard the data according to `num_groups`.
        A `group_index` should be specified if using `num_groups`.
      group_index: Index of the shard to return after resharding. `num_groups`
        should be specified if using `group_index`. This is useful in
        distributed setting where one wants to ensure that different data is
        read by different workers.

    Raises:
      ValueError: If subset is not a key of tables or examples_per_subset
    """
    if (subset not in tables) or (subset not in examples_per_subset):
      raise ValueError(f'Invalid subset {subset!r}. '
                       f'The available subsets are: {set(tables)!r}')

    self.num_classes = num_classes
    self.base_dir = base_dir
    self.subset = subset
    self.num_examples = examples_per_subset[subset]

    data_relative_path = tables[subset]
    if isinstance(data_relative_path, list):
      shards = [os.path.join(self.base_dir, x) for x in data_relative_path]
    else:
      data_path = os.path.join(self.base_dir, data_relative_path)
      shards = get_sharded_files(data_path=data_path,
                                 fraction_data=fraction_data,
                                 num_groups=num_groups,
                                 group_index=group_index)

    super().__init__(shards=shards)

  def _build(self,
             is_training: bool = True,
             # Video related parameters.
             num_frames: int = 32,
             stride: int = 1,
             num_test_clips: int = 1,
             min_resize: int = 256,
             crop_size: int = 224,
             zero_centering_image: bool = False,
             train_frame_sampling_mode: str = 'random',
             # Label related parameters.
             one_hot_label: bool = True,
             get_label_str: bool = False,
             label_offset: int = 0):
    """Adds DMVR pre-processors to the dataset.

    Args:
      is_training: whether or not in training mode.
      num_frames: number of frames per subclip.
      stride: temporal stride to sample frames.
      num_test_clips: number of test clip (1 by default). If more than one,
        this will sample multiple linearly spaced clips within each video at
        test time. If 1, then a single clip in the middle of the video is
        sampled.
      min_resize: frames are resized so that min width/height is min_resize.
      crop_size: final size of the frame after cropping the resized frames.
      zero_centering_image: whether to have image values in the interval [-1, 1]
        or [0, 1].
      train_frame_sampling_mode: Method of sampling frames during training.
        Options are one of {random, random_sample_with_centre, centre}
      one_hot_label: whether or not to return one hot version of labels.
      get_label_str: whether or not to return label as text.
      label_offset: If non-zero, this value is subtracted from the parsed label.
        Useful when dataset is 1-indexed.
    """
    modalities.add_image(
        parser_builder=self.parser_builder,
        sampler_builder=self.sampler_builder,
        decoder_builder=self.decoder_builder,
        preprocessor_builder=self.preprocessor_builder,
        postprocessor_builder=self.postprocessor_builder,
        is_training=is_training,
        num_frames=num_frames, stride=stride,
        num_test_clips=num_test_clips,
        min_resize=min_resize, crop_size=crop_size,
        zero_centering_image=zero_centering_image)

    modalities.add_label(
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
          add_before_fn_name=(
              f'{builders.LABEL_INDEX_FEATURE_NAME}_one_hot'))


def load_split(
    ds_factory,
    batch_size: int,
    subset: Text = 'train',
    num_frames: int = 32,
    stride: int = 2,
    num_test_clips: int = 1,
    min_resize: int = 256,
    crop_size: int = 224,
    one_hot_label: bool = True,
    zero_centering: bool = True,
    get_label_str: bool = False,
    augmentation_params: Optional[ml_collections.ConfigDict] = None,
    keep_key: bool = False,
    do_three_spatial_crops: bool = False,
    label_offset: int = 0) -> Tuple[tf.data.Dataset, int]:
  """Loads dataset using DMVR for pre-processing.

  DMVR dataset loader already does basic augmentation (random crop and flip in
    train mode. It also already shuffles and batches the data.

  Args:
    ds_factory: A DMVR factory to instantiate with the subset.
    batch_size: The batch_size to use.
    subset: train, validation or test
    num_frames: Number of frames per subclip.
    stride: Temporal stride to sample frames.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggreagated in the batch dimension.
    min_resize: Frames are resized so that min(height, width) is min_resize.
    crop_size: Final size of the frame after cropping the resized frames. Both
      height and width are the same.
    one_hot_label: If True, return one-hot version of the labels (ie [N, C])
      array. Otherwise, return [N]-array of labels.
    zero_centering: If True, frames are normalized to values in the interval
      [-1, 1]. If False, values are in the interval [0, 1].
    get_label_str: whether or not to return label as text.
      Note that strings cannot be used in pmapped functions in Jax!
    augmentation_params: Augmentation configurations in train mode.
    keep_key: bool; If true, also return the key for each example.
    do_three_spatial_crops: If true, take three spatial crops of each clip
      during testing.
    label_offset: If non-zero, this value is subtracted from the parsed label.
      Useful when dataset is 1-indexed.

  Returns:
    A pair `(ds, num_examples)` with
      ds: A `tf.data.Dataset` object
      num_examples: Number of examples in the dataset.
  """
  dataset = ds_factory(subset=subset).configure(
      is_training=(subset == 'train'),
      num_frames=num_frames,
      stride=stride,
      num_test_clips=num_test_clips,
      min_resize=min_resize,
      crop_size=crop_size,
      zero_centering_image=zero_centering,
      one_hot_label=one_hot_label,
      get_label_str=get_label_str,
      label_offset=label_offset)

  if subset == 'train' and augmentation_params:
    dataset = video_ops.additional_augmentations(dataset, augmentation_params,
                                                 crop_size, num_frames,
                                                 zero_centering)

  if subset != 'train' and do_three_spatial_crops:
    rgb_feature_name = builders.IMAGE_FEATURE_NAME

    dataset.preprocessor_builder.replace_fn(
        f'{rgb_feature_name}_central_crop',
        functools.partial(video_ops.three_spatial_crops, crop_size=crop_size))

    if num_test_clips == 1:
      # This means that reshaping is not part of the post-processing graph
      output_shape = (-1, num_frames, crop_size, crop_size, 3)
      dataset.postprocessor_builder.add_fn(
          fn=lambda x: tf.reshape(x, output_shape),
          feature_name=rgb_feature_name,
          fn_name=f'{rgb_feature_name}_reshape')

  logging.info('Preprocessing graph: %s',
               dataset.preprocessor_builder.get_summary())
  logging.info('Postprocessing graph: %s',
               dataset.postprocessor_builder.get_summary())
  num_examples = dataset.num_examples

  ds = dataset.make_dataset(batch_size=batch_size,
                            shuffle=(subset == 'train'),
                            drop_remainder=(subset == 'train'),
                            keep_key=(subset != 'train' and keep_key))

  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  return ds, num_examples


def map_keys(batch: Batch) -> Batch:
  """DMVR dataset returns 'image' and 'label'. We want 'inputs' and 'label'."""

  batch['inputs'] = batch['image']
  return batch


def tile_label_key(batch: Batch) -> Batch:
  """Tile labels and keys to match input videos when num_test_clips > 1.

  When multiple test crops are used (ie num_test_clips > 1), the batch dimension
  of batch['inputs'] = test_batch_size * num_test_clips.
  However, labels and keys remain of size [test_batch_size].
  This function repeats label and key to match the inputs.

  Args:
    batch: Batch from iterator

  Returns:
    Batch with 'label' and 'key' tiled to match 'inputs'. The input batch is
    mutated by the function.
  """
  n_repeats = batch['inputs'].shape[0] // batch['label'].shape[0]
  batch['label'] = np.repeat(batch['label'], n_repeats, axis=0)
  if 'key' in batch:
    batch['key'] = np.repeat(batch['key'], n_repeats, axis=0)
  return batch


@datasets.add_dataset('video_tfrecord_dataset')
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
  """Returns a generator for the dataset."""
  del rng  # Parameter was required by caller API, but is unused.

  def validate_config(field):
    if dataset_configs.get(field) is None:
      raise ValueError(f'{field} must be specified for TFRecord dataset.')
  validate_config('base_dir')
  validate_config('tables')
  validate_config('examples_per_subset')
  validate_config('num_classes')

  num_frames = dataset_configs.get('num_frames', 32)
  num_test_clips = dataset_configs.get('num_test_clips', 1)
  stride = dataset_configs.get('stride', 2)
  min_resize = dataset_configs.get('min_resize', 256)
  crop_size = dataset_configs.get('crop_size', 224)
  one_hot_label = dataset_configs.get('one_hot_label', True)
  zero_centre_data = dataset_configs.get('zero_centering', True)
  augmentation_params = dataset_configs.get('augmentation_params', None)
  num_train_val_clips = dataset_configs.get('num_train_val_clips', 1)
  keep_test_key = dataset_configs.get('keep_test_key', False)
  # For the test set, the actual batch size is
  # test_batch_size * num_test_clips.
  test_batch_size = dataset_configs.get('test_batch_size', eval_batch_size)
  do_three_spatial_crops = dataset_configs.get('do_three_spatial_crops', False)
  num_spatial_crops = 3 if do_three_spatial_crops else 1
  test_split = dataset_configs.get('test_split', 'test')
  label_offset = dataset_configs.get('label_offset', 0)

  ds_factory = functools.partial(
      TFRecordDatasetFactory,
      base_dir=dataset_configs.base_dir,
      tables=dataset_configs.tables,
      examples_per_subset=dataset_configs.examples_per_subset,
      num_classes=dataset_configs.num_classes,
      num_groups=jax.process_count(),
      group_index=jax.process_index())

  def create_dataset_iterator(
      subset: Text,
      batch_size_local: int,
      num_clips: int,
      keep_key_local: bool = False) -> Tuple[Iterator[Batch], int]:
    is_training = subset == 'train'
    is_test = subset == 'test'
    logging.info('Loading split %s', subset)

    dataset, num_examples = load_split(
        ds_factory,
        batch_size=batch_size_local,
        subset=subset,
        num_frames=num_frames,
        stride=stride,
        num_test_clips=num_clips,
        min_resize=min_resize,
        crop_size=crop_size,
        one_hot_label=one_hot_label,
        zero_centering=zero_centre_data,
        augmentation_params=augmentation_params,
        keep_key=keep_key_local,
        do_three_spatial_crops=do_three_spatial_crops and is_test,
        label_offset=label_offset)

    if dataset_service_address and is_training:
      if shuffle_seed is not None:
        raise ValueError('Using dataset service with a random seed causes each '
                         'worker to produce exactly the same data. Add '
                         'config.shuffle_seed = None to your config if you '
                         'want to run with dataset service.')
      logging.info('Using the tf.data service at %s', dataset_service_address)
      dataset = dataset_utils.distribute(dataset, dataset_service_address)

    pad_batch_size = batch_size_local
    if is_test:
      pad_batch_size = batch_size_local * num_clips * num_spatial_crops
    maybe_pad_batches = functools.partial(
        dataset_utils.maybe_pad_batch,
        train=is_training,
        batch_size=pad_batch_size)
    shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)

    current_ds_iterator = (
        map_keys(dataset_utils.tf_to_numpy(data)) for data in iter(dataset)
    )

    if is_test and num_clips * num_spatial_crops > 1:
      current_ds_iterator = map(tile_label_key, current_ds_iterator)

    current_ds_iterator = map(maybe_pad_batches, current_ds_iterator)
    current_ds_iterator = map(shard_batches, current_ds_iterator)
    if is_training and dataset_configs.get('prefetch_to_device'):
      # Async bind batch to device which speeds up training.
      current_ds_iterator = jax_utils.prefetch_to_device(
          current_ds_iterator, dataset_configs.get('prefetch_to_device'))

    return current_ds_iterator, num_examples

  train_iter, n_train_examples = create_dataset_iterator(
      'train', batch_size, num_train_val_clips)
  eval_iter, n_eval_examples = create_dataset_iterator(
      'validation', eval_batch_size, num_train_val_clips)
  test_iter, n_test_examples = create_dataset_iterator(
      test_split, test_batch_size, num_test_clips, keep_test_key)

  meta_data = {
      'num_classes': dataset_configs.num_classes,
      'input_shape': (-1, num_frames, crop_size, crop_size, 3),
      'num_train_examples': (n_train_examples * num_train_val_clips),
      'num_eval_examples': (n_eval_examples * num_train_val_clips),
      'num_test_examples':
          (n_test_examples * num_test_clips * num_spatial_crops),
      'input_dtype': getattr(jnp, dtype_str),
      'target_is_onehot': True,
  }
  logging.info('Dataset metadata:\n%s', meta_data)

  return dataset_utils.Dataset(train_iter, eval_iter, test_iter, meta_data)
