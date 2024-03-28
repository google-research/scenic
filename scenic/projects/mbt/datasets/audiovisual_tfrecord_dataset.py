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

"""TFRecords data-loader for audiovisual datasets."""
import functools
from typing import Dict, Iterator, List, Optional, Text, Tuple, Union

from absl import logging
from dmvr import modalities as load_modalities
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.dataset_lib import video_ops
from scenic.projects.mbt.datasets.dataset_utils import add_spectrogram
from scenic.projects.vivit.data import video_tfrecord_dataset
import tensorflow as tf

# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]


def maybe_pad_batch(batch, train, batch_size, return_as_dict):
  """Zero pad the batch on the right to the batch_size."""
  if not return_as_dict:
    return dataset_utils.maybe_pad_batch(batch, train, batch_size)

  assert 'batch_mask' not in batch
  if 'rgb' in batch['inputs']:
    unpadded_mask_shape = batch['inputs']['rgb'].shape[0]
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

  # Most batches will not need padding so we quickly return to avoid slowdown.
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


class AVTFRecordDatasetFactory(video_tfrecord_dataset.TFRecordDatasetFactory):
  """Reader for TFRecords using the MediaSequence format.

  The TFrecords already contain images and spectrograms. Spectrograms are
  extracted per second and stored with size 128x100 for each second of audio.
  """

  _MODALITIES = ('rgb', 'spectrogram')

  def __init__(self,
               base_dir: str,
               tables: Dict[str, Union[str, List[str]]],
               num_classes: int,
               examples_per_subset: Dict[str, int],
               subset: str = 'train',
               modalities: Tuple[str] = ('rgb',),
               prop_data: float = 1.0,
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
        relative path of the SSTable containing them. Follows DMVR convention.
        The values of the dictionary can either be a string or a list. If it is
        a string, it specifies all the shards in the SSTable. Example -
        "/path/to/sstable@10". If passing a list, each entry is a shard of the
        SSTable. Example - "[/path/to/sstable_shard_1_of_10, ...,
        /path/to/sstabble_shard_10_of_10]." The latter scenario is useful for
        debugging.
      num_classes: The number of classes in the dataset.
      examples_per_subset:  A dictionary mapping the subset name (train, val or
        test) to the number of examples in the dataset for that subset.
      subset: The subset of the dataset to load. Must be a key of "tables"
      modalities: Which modality to load. Currently supports 'rgb' and
        'spectrogram'
      prop_data: The proportion of the data to load. If less than 1.0, this
        proportion of the total TFRecord shards are read.
      num_groups: If specified will reshard the data according to `num_groups`.
        A `group_index` should be specified if using `num_groups`.
      group_index: Index of the shard to return after resharding. `num_groups`
        should be specified if using `group_index`. This is useful in
        distributed setting where one wants to ensure that different data is
        read by different workers.
    """

    for modality in modalities:
      if modality not in AVTFRecordDatasetFactory._MODALITIES:
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
      num_spec_frames: int = 5,
      spec_stride: int = 1,
      dataset_spec_mean: float = 0.,
      dataset_spec_stddev: float = 1.,
      num_test_clips: int = 1,
      min_resize: int = 256,
      crop_size: int = 224,
      # Audio related parameters.
      spec_shape: Tuple[int, int] = (100, 128),
      spec_augment: bool = False,
      spec_augment_params=None,
      zero_centering_image: bool = False,
      # Label related parameters.
      one_hot_label: bool = True,
      get_label_str: bool = False):
    """Adds DMVR pre-processors to the dataset.

    Args:
      is_training: whether or not in training mode.
      num_frames: number of frames per subclip.
      stride: temporal stride to sample frames.
      num_spec_frames: number of spectrogram frames.
      spec_stride: stride to sample spectrogram.
      dataset_spec_mean: Mean of spectrograms in the dataset.
      dataset_spec_stddev: Std dev of spectrograms in the dataset.
      num_test_clips: number of test clip (1 by default). If more than one, this
        will sample multiple linearly spaced clips within each video at test
        time. If 1, then a single clip in the middle of the video is sampled.
      min_resize: frames are resized so that min width/height is min_resize.
      crop_size: final size of the frame after cropping the resized frames.
      spec_shape: input size of spectrogram per frame.
      spec_augment: whether to apply augmentation using SpecAugment.
      spec_augment_params: parameters for SpecAugment.
      zero_centering_image: whether to have images between [-1, 1] or [0, 1].
      one_hot_label: whether or not to return one hot version of labels.
      get_label_str: whether or not to return label as text.
    """
    # We set sync_random_state to True so that sample_offset_proportion is
    # the same for all modalities.
    if 'rgb' in self._modalities:
      load_modalities.add_image(
          parser_builder=self.parser_builder,
          sampler_builder=self.sampler_builder,
          decoder_builder=self.decoder_builder,
          preprocessor_builder=self.preprocessor_builder,
          postprocessor_builder=self.postprocessor_builder,
          is_training=is_training,
          num_frames=num_frames,
          stride=stride,
          num_test_clips=num_test_clips,
          min_resize=min_resize,
          crop_size=crop_size,
          zero_centering_image=zero_centering_image,
          sync_random_state=True)
    if 'spectrogram' in self._modalities:
      add_spectrogram(
          parser_builder=self.parser_builder,
          sampler_builder=self.sampler_builder,
          decoder_builder=self.decoder_builder,
          preprocessor_builder=self.preprocessor_builder,
          postprocessor_builder=self.postprocessor_builder,
          input_shape=spec_shape,
          is_training=is_training,
          num_frames=num_spec_frames,
          stride=spec_stride,
          num_test_clips=num_test_clips,
          spec_augment=spec_augment,
          spec_augment_params=spec_augment_params,
          zero_centering_image=zero_centering_image,
          dataset_mean=dataset_spec_mean,
          dataset_stddev=dataset_spec_stddev,
          sync_random_state=True)

    load_modalities.add_label(
        parser_builder=self.parser_builder,
        decoder_builder=self.decoder_builder,
        preprocessor_builder=self.preprocessor_builder,
        is_multi_label=False,
        one_hot_label=True,
        num_classes=self.num_classes,
        add_label_name=False)


def load_split_from_dmvr(ds_factory,
                         batch_size,
                         subset='train',
                         modalities=('rgb'),
                         num_frames=32,
                         stride=2,
                         num_spec_frames=5,
                         spec_stride=1,
                         num_test_clips=1,
                         min_resize=256,
                         crop_size=224,
                         spec_shape=(100, 128),
                         dataset_spec_mean=0.,
                         dataset_spec_stddev=1.,
                         spec_augment=False,
                         spec_augment_params=None,
                         one_hot_label=True,
                         zero_centering=True,
                         get_label_str=False,
                         augmentation_params=None,
                         keep_key=False):
  """Loads dataset using DMVR for pre-processing.

  DMVR dataset loader already does basic augmentation (random crop and flip in
    train mode. It also already shuffles and batches the data.

  Args:
    ds_factory: A DMVR factory to instantiate with the subset.
    batch_size: The batch_size to use.
    subset: train, validation or test.
    modalities: list of input modalities.
    num_frames: Number of RGB frames per subclip.
    stride: Temporal stride to sample RGB frames.
    num_spec_frames: Number of spectrogram frames per subclip.
    spec_stride: Temporal stride to sample spectrogram.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggreagated in the batch dimension.
    min_resize: Frames are resized so that min(height, width) is min_resize.
    crop_size: Final size of the frame after cropping the resized frames. Both
      height and width are the same.
    spec_shape: Input size of spectrogram per frame.
    dataset_spec_mean: Mean of spectrograms in the dataset.
    dataset_spec_stddev: Std dev of spectrograms in the dataset.
    spec_augment: whether to apply augmentation using SpecAugment.
    spec_augment_params: dict; augmentation configurations for SpecAugment
    one_hot_label: If True, return one-hot version of the labels (ie [N, C])
      array. Otherwise, return [N]-dimensional array of labels.
    zero_centering: If True, frames are normalized to values in [-1, 1]. If
      False, values in [0, 1].
    get_label_str: whether or not to return label as text. This does not work on
      TPU!.
    augmentation_params: dict; augmentation configurations in train mode.
    keep_key: bool; If true, also return the key for each example.

  Returns:
    A pair `(ds, num_examples)` with
      ds: A `tf.data.Dataset` object
      num_examples: Number of examples in the dataset.
  """
  is_training = (subset == 'train')

  ds_factory = ds_factory(
      subset=subset, modalities=modalities).configure(
          is_training=is_training,
          num_frames=num_frames,
          stride=stride,
          num_spec_frames=num_spec_frames,
          spec_stride=spec_stride,
          num_test_clips=num_test_clips,
          min_resize=min_resize,
          crop_size=crop_size,
          spec_shape=spec_shape,
          dataset_spec_mean=dataset_spec_mean,
          dataset_spec_stddev=dataset_spec_stddev,
          spec_augment=spec_augment,
          spec_augment_params=spec_augment_params,
          zero_centering_image=zero_centering,
          one_hot_label=one_hot_label,
          get_label_str=get_label_str)

  if 'rgb' in modalities and is_training and augmentation_params:
    # additional augmentation for the RGB features.
    ds_factory = video_ops.additional_augmentations(ds_factory,
                                                    augmentation_params,
                                                    crop_size, num_frames,
                                                    zero_centering)

  logging.info('Preprocessing graph: %s',
               ds_factory.preprocessor_builder.get_summary())
  logging.info('Postprocessing graph: %s',
               ds_factory.postprocessor_builder.get_summary())

  num_examples = ds_factory.num_examples

  ds = ds_factory.make_dataset(
      batch_size=batch_size,
      shuffle=is_training,
      num_epochs=None if is_training else 1,
      drop_remainder=is_training,
      keep_key=(not is_training and keep_key))

  if not is_training:
    ds = ds.repeat(None)

  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  return ds, num_examples


def map_keys(batch, modalities=('rgb'), return_as_dict=False):
  """DMVR dataset returns 'image' and 'label'. We want 'inputs' and 'label'."""
  if not return_as_dict:
    if len(modalities) == 1 and modalities[0] == 'rgb':
      batch['inputs'] = batch['image']
    elif len(modalities) == 1 and modalities[0] == 'spectrogram':
      batch['inputs'] = batch['spectrogram']
    else:
      raise NotImplementedError('modality not supported by map_keys.')
  else:
    batch['inputs'] = {}
    if 'rgb' in modalities:
      batch['inputs']['rgb'] = batch['image']
    if 'spectrogram' in modalities:
      batch['inputs']['spectrogram'] = batch['spectrogram']
  return batch


def tile_label_key(batch, return_as_dict=False):
  """Tile labels and keys to match input videos when num_test_clips > 1.

  When multiple test crops are used (ie num_test_clips > 1), the batch dimension
  of batch['inputs'] = test_batch_size * num_test_clips.
  However, labels and keys remain of size [test_batch_size].
  This function repeats label and key to match the inputs.

  Args:
    batch: Batch from iterator
    return_as_dict: Whether to return multimodal inputs as a dictionary.

  Returns:
    batch: Batch with 'label' and 'key' tiled to match 'inputs'.
  """
  if not return_as_dict:
    n_repeats = batch['inputs'].shape[0] // batch['label'].shape[0]
  elif 'rgb' in batch['inputs']:
    n_repeats = batch['inputs']['rgb'].shape[0] // batch['label'].shape[0]
  elif 'spectrogram' in batch['inputs']:
    n_repeats = (
        batch['inputs']['spectrogram'].shape[0] // batch['label'].shape[0])
  batch['label'] = np.repeat(batch['label'], n_repeats, axis=0)
  if 'key' in batch:
    batch['key'] = np.repeat(batch['key'], n_repeats, axis=0)
  return batch


@datasets.add_dataset('audiovisual_tfrecord_dataset')
def get_dataset(
    *,
    batch_size,
    eval_batch_size,
    num_shards,
    dtype_str='float32',
    shuffle_seed=0,  # pylint:disable=unused-argument
    rng=None,
    dataset_configs: ml_collections.ConfigDict,
    dataset_service_address: Optional[str] = None):
  """Returns a generator for the audiovisual dataset."""
  del rng
  modalities = dataset_configs.get('modalities', ['rgb'])
  return_as_dict = dataset_configs.get('return_as_dict', False)
  # RGB related configs.
  num_frames = dataset_configs.get('num_frames', 32)
  stride = dataset_configs.get('stride', 2)
  min_resize = dataset_configs.get('min_resize', 256)
  crop_size = dataset_configs.get('crop_size', 224)
  # Spectrogram related configs.
  num_spec_frames = dataset_configs.get('num_spec_frames', 5)
  spec_stride = dataset_configs.get('spec_stride', 1)
  spec_shape = dataset_configs.get('spec_shape', (100, 128))
  spec_augment = dataset_configs.get('spec_augment', False)
  spec_augment_params = dataset_configs.get('spec_augment_params', None)
  dataset_spec_mean = dataset_configs.get('spec_mean', 0.)
  dataset_spec_stddev = dataset_configs.get('spec_stddev', 1.)
  # General configs.
  num_test_clips = dataset_configs.get('num_test_clips', 1)
  one_hot_label = dataset_configs.get('one_hot_label', True)
  zero_centre_data = dataset_configs.get('zero_centering', True)
  augmentation_params = dataset_configs.get('augmentation_params', None)
  num_train_val_clips = dataset_configs.get('num_train_val_clips', 1)
  do_three_spatial_crops = dataset_configs.get('do_three_spatial_crops', False)
  num_spatial_crops = 3 if do_three_spatial_crops else 1
  keep_test_key = dataset_configs.get('keep_test_key', False)
  test_split = dataset_configs.get('test_split', 'test')
  # For the test set, the actual batch size is
  # test_batch_size * num_test_clips
  test_batch_size = dataset_configs.get('test_batch_size', eval_batch_size)

  def validate_config(field):
    if dataset_configs.get(field) is None:
      raise ValueError(f'{field} must be specified for TFRecord dataset.')
  validate_config('base_dir')
  validate_config('tables')
  validate_config('examples_per_subset')
  validate_config('num_classes')

  ds_factory = functools.partial(
      AVTFRecordDatasetFactory,
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

    dataset, num_examples = load_split_from_dmvr(
        ds_factory,
        batch_size=batch_size_local,
        subset=subset,
        modalities=modalities,
        num_frames=num_frames,
        stride=stride,
        num_spec_frames=num_spec_frames,
        spec_stride=spec_stride,
        num_test_clips=num_clips,
        min_resize=min_resize,
        crop_size=crop_size,
        spec_shape=spec_shape,
        dataset_spec_mean=dataset_spec_mean,
        dataset_spec_stddev=dataset_spec_stddev,
        spec_augment=spec_augment,
        spec_augment_params=spec_augment_params,
        one_hot_label=one_hot_label,
        zero_centering=zero_centre_data,
        augmentation_params=augmentation_params,
        keep_key=keep_key_local)
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
        maybe_pad_batch,
        train=is_training,
        batch_size=pad_batch_size,
        return_as_dict=return_as_dict)

    shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
    current_iter = iter(dataset)
    current_iter = map(dataset_utils.tf_to_numpy, current_iter)
    current_iter = map(
        functools.partial(
            map_keys, modalities=modalities, return_as_dict=return_as_dict),
        current_iter)
    current_iter = map(
        functools.partial(
            tile_label_key, return_as_dict=return_as_dict),
        current_iter)
    current_iter = map(maybe_pad_batches, current_iter)

    if augmentation_params and augmentation_params.get('do_mixup', False):
      raise ValueError('mixup should be done in the trainer.')

    current_iter = map(shard_batches, current_iter)

    if is_training and dataset_configs.get('prefetch_to_device'):
      # Async bind batch to device which speeds up training.
      current_iter = jax_utils.prefetch_to_device(
          current_iter, dataset_configs.get('prefetch_to_device'))

    return current_iter, num_examples

  train_iter, n_train_examples = create_dataset_iterator(
      'train', batch_size, num_train_val_clips)
  eval_iter, n_eval_examples = create_dataset_iterator('validation',
                                                       eval_batch_size,
                                                       num_train_val_clips)
  test_iter, n_test_examples = create_dataset_iterator(test_split,
                                                       test_batch_size,
                                                       num_test_clips,
                                                       keep_test_key)

  meta_data = {
      'num_classes': dataset_configs.num_classes,  # pylint:disable=protected-access
      'num_train_examples': (n_train_examples * num_train_val_clips),
      'num_eval_examples': (n_eval_examples * num_train_val_clips),
      'num_test_examples':
          (n_test_examples * num_test_clips * num_spatial_crops),
      'input_dtype': getattr(jnp, dtype_str),
      'target_is_onehot': True,
  }
  if return_as_dict:
    meta_data['input_shape'] = {
        'rgb': (-1, num_frames, crop_size, crop_size, 3),
        'spectrogram': (-1, num_spec_frames * spec_shape[0], spec_shape[1], 3)
    }
  elif len(modalities) == 1 and modalities[0] == 'rgb':
    meta_data['input_shape'] = (-1, num_frames, crop_size, crop_size, 3)
  elif len(modalities) == 1 and modalities[0] == 'spectrogram':
    meta_data['input_shape'] = (-1, num_spec_frames * spec_shape[0],
                                spec_shape[1], 3)
  else:
    raise NotImplementedError('modality not supported')
  logging.info('Dataset metadata:\n%s', meta_data)

  return dataset_utils.Dataset(train_iter, eval_iter, test_iter, meta_data)
