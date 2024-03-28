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

"""Data-loader to read from TFRecord using the MediaSequence format."""

import functools
from typing import Dict, Iterator, Optional, Text, Tuple, Union

from absl import logging
from dmvr import builders
from dmvr import modalities

from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import scenic.projects.objectvivit.dataset_utils as objects_dataset_utils
from scenic.projects.vivit.data.video_tfrecord_dataset import TFRecordDatasetFactory
import tensorflow as tf


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
PRNGKey = jnp.ndarray
Rng = Union[jnp.ndarray, Dict[str, jnp.ndarray]]


class ObjectsTFRecordDatasetFactory(TFRecordDatasetFactory):
  """Support bounding boxes."""

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
      use_crop_and_resize_video_mae: bool = False,
      # Label related parameters.
      one_hot_label: bool = True,
      get_label_str: bool = False,
      label_offset: int = 0,
      object_configs: ml_collections.ConfigDict = ml_collections.ConfigDict(),
    ):
    """Adds DMVR pre-processors to the dataset."""

    objects_dataset_utils.add_image_and_boxes(
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
        object_configs=object_configs
    )

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
          add_before_fn_name=(f'{builders.LABEL_INDEX_FEATURE_NAME}_one_hot'))


def load_split(
    ds_factory,
    batch_size: int,
    shuffle_buffer_size: int,
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
    use_crop_and_resize_video_mae: bool = False,
    object_configs: ml_collections.ConfigDict = ml_collections.ConfigDict(),
    num_channels: int = 3,
    ) -> Tuple[tf.data.Dataset, int]:
  """Additionally support loading object boxes."""
  dataset = ds_factory(subset=subset).configure(
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
      use_crop_and_resize_video_mae=use_crop_and_resize_video_mae,
      normalization_mean=normalization_mean,
      normalization_std=normalization_std,
      object_configs=object_configs,
  )
  del augmentation_params

  if subset != 'train' and do_three_spatial_crops and resize_keep_aspect_ratio:
    with_boxes = object_configs.get('with_boxes', False)
    if with_boxes:
      dataset.preprocessor_builder.replace_fn(
          f'{builders.IMAGE_FEATURE_NAME}_central_crop',
          functools.partial(
              objects_dataset_utils.three_spatial_crops_with_state,
              crop_size=crop_size))
      dataset.preprocessor_builder.replace_fn(
          'bboxes_central_crop',
          functools.partial(
              objects_dataset_utils.three_spatial_transform_box,
              crop_size=crop_size))
    else:
      dataset.preprocessor_builder.replace_fn(
          f'{builders.IMAGE_FEATURE_NAME}_central_crop',
          functools.partial(
              objects_dataset_utils.three_spatial_crops, crop_size=crop_size))

    if num_test_clips == 1:
      # This means that reshaping is not part of the post-processing graph.
      dataset.postprocessor_builder.add_fn(
          fn=lambda x: tf.reshape(  # pylint: disable=g-long-lambda
              x, (-1, num_frames, crop_size, crop_size, num_channels)),
          feature_name=builders.IMAGE_FEATURE_NAME,
          fn_name=f'{builders.IMAGE_FEATURE_NAME}_reshape')
      if object_configs.get('return_boxes', -1) > 0:
        dataset.postprocessor_builder.add_fn(
            fn=lambda x: tf.reshape(  # pylint: disable=g-long-lambda
                x,
                (-1, num_frames, object_configs.get('return_boxes'), 4)),
            feature_name='bboxes',
            fn_name='bboxes_reshape')

  logging.info('Frame sampling graph: %s',
               dataset.sampler_builder.get_summary())
  logging.info('Preprocessing graph: %s',
               dataset.preprocessor_builder.get_summary())
  logging.info('Postprocessing graph: %s',
               dataset.postprocessor_builder.get_summary())
  num_examples = dataset.num_examples

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


@datasets.add_dataset('objects_video_tfrecord_dataset')
def objects_video_tfrecord_dataset(
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
  num_frames = dataset_configs.get('num_frames', 32)
  num_test_clips = dataset_configs.get('num_test_clips', 1)
  stride = dataset_configs.get('stride', 2)
  min_resize = dataset_configs.get('min_resize', 256)
  min_resize_train = dataset_configs.get('min_resize_train', min_resize)
  min_resize_test = dataset_configs.get('min_resize_test', min_resize)
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
  use_crop_and_resize_video_mae = augmentation_params.get(
      'crop_and_resize_video_mae', False) if (augmentation_params
                                              is not None)  else False
  normalization_mean = dataset_configs.get('normalization_mean', 0)
  normalization_std = dataset_configs.get('normalization_std', 1)
  object_configs = dataset_configs.get(
      'object_configs', ml_collections.ConfigDict())
  num_channels = 3
  if object_configs.get('with_boxes', False) and object_configs.get(
      'concat_mask', False):
    num_channels += 1 if not object_configs.get(
        'tracked_objects', False) else object_configs['bbox_num']

  if isinstance(normalization_mean, (list, tuple)):
    normalization_mean = tf.constant(normalization_mean, tf.float32)
  if isinstance(normalization_std, (list, tuple)):
    normalization_std = tf.constant(normalization_std, tf.float32)

  if dataset_configs.get('base_dir') is None:
    raise ValueError('base_dir must be specified for TFRecord dataset')
  if not dataset_configs.get('tables'):
    raise ValueError('tables mapping must be specified for TFRecord dataset')
  if not dataset_configs.get('num_classes'):
    raise ValueError('num_classes must be specified for TFRecord dataset')

  ds_factory = functools.partial(
      ObjectsTFRecordDatasetFactory,
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
        use_crop_and_resize_video_mae=use_crop_and_resize_video_mae,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        object_configs=object_configs,
        num_channels=num_channels,
    )

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
    shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)

    current_ds_iterator = iter(dataset)
    current_ds_iterator = map(dataset_utils.tf_to_numpy, current_ds_iterator)
    current_ds_iterator = map(map_keys, current_ds_iterator)

    if is_test and num_clips * num_spatial_crops > 1:
      current_ds_iterator = map(custom_tile_label_key, current_ds_iterator)

    current_ds_iterator = map(
        functools.partial(
            dataset_utils.maybe_pad_batch,
            train=is_training,
            batch_size=pad_batch_size), current_ds_iterator)

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

    return current_ds_iterator, num_examples

  train_iter, n_train_examples = create_dataset_iterator(
      'train', batch_size, num_train_val_clips)
  eval_iter, n_eval_examples = create_dataset_iterator('validation',
                                                       eval_batch_size,
                                                       num_train_val_clips)
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

  meta_data['input_shape'] = (
      -1, num_frames, crop_size, crop_size, num_channels)
  meta_data['input_dtype'] = getattr(jnp, dtype_str)
  logging.info('Dataset metadata:\n%s', meta_data)

  return dataset_utils.Dataset(train_iter, eval_iter, test_iter, meta_data)


def map_keys(batch: Batch) -> Batch:
  """DMVR dataset returns 'image' and 'label'. We want 'inputs' and 'label'."""
  batch['inputs'] = batch.pop(builders.IMAGE_FEATURE_NAME)
  return batch  # pytype: disable=bad-return-type  # jax-ndarray


def custom_tile_label_key(
    batch: Batch) -> Batch:
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
