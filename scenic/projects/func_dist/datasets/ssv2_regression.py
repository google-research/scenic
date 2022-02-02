"""TFRecords data-loader for distance prediction tasks."""

import functools
import json
from typing import Dict, Iterator, Iterable, Optional, Text, Tuple, Union

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
from scenic.dataset_lib import video_ops
from scenic.projects.func_dist.datasets import regression_utils
from scenic.projects.vivit.data import video_tfrecord_dataset
import tensorflow as tf

from tensorflow.io import gfile


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
Rng = Union[jnp.ndarray, Dict[str, jnp.ndarray]]


class RegressionDatasetFactory(video_tfrecord_dataset.TFRecordDatasetFactory):
  """TFRecords reader to load distance prediction data."""

  def _build(self,
             is_training: bool = True,
             num_frames: int = 3,
             stride: int = 1,
             min_resize: int = 256,
             crop_size: int = 224,
             zero_centering_image: bool = False,
             augment_goals: bool = True):
    """Adds DMVR pre-processors to the dataset.

    Args:
      is_training: whether or not in training mode.
      num_frames: number of frames per subclip.
      stride: temporal stride to sample frames.
      min_resize: frames are resized so that min width/height is min_resize.
      crop_size: final size of the frame after cropping the resized frames.
      zero_centering_image: whether to have images between [-1, 1] or [0, 1].
      augment_goals: whether to use any future frame or only the final frame as
        the goal.
    """
    modalities.add_label(
        parser_builder=self.parser_builder,
        decoder_builder=self.decoder_builder,
        preprocessor_builder=self.preprocessor_builder,
        one_hot_label=False,
        output_label_index_feature_name='task_label',
        add_label_name=False)

    regression_utils.sample_frames(
        parser_builder=self.parser_builder,
        sampler_builder=self.sampler_builder,
        decoder_builder=self.decoder_builder,
        preprocessor_builder=self.preprocessor_builder,
        is_training=is_training,
        num_frames=num_frames,
        stride=stride,
        min_resize=min_resize,
        crop_size=crop_size,
        zero_centering_image=zero_centering_image,
        augment_goals=augment_goals)


def load_split(
    ds_factory,
    batch_size: int,
    subset: Text = 'train',
    num_frames: int = 32,
    stride: int = 2,
    min_resize: int = 256,
    crop_size: int = 224,
    zero_centering: bool = True,
    included_tasks: Optional[Iterable[int]] = None,
    augment_goals: bool = False,
    augmentation_params: Optional[ml_collections.ConfigDict] = None,
    keep_key: bool = False,
    do_three_spatial_crops: bool = False) -> Tuple[tf.data.Dataset, int]:
  """Loads dataset using DMVR for pre-processing.


  Args:
    ds_factory: A DMVR factory to instantiate with the subset.
    batch_size: The batch_size to use.
    subset: train or test
    num_frames: Number of frames per subclip.
    stride: Temporal stride to sample frames.
    min_resize: Frames are resized so that min(height, width) is min_resize.
    crop_size: Final size of the frame after cropping the resized frames. Both
      height and width are the same.
    zero_centering: If True, frames are normalized to values in [-1, 1].
      If False, values in [0, 1].
    included_tasks: If set, a subset of task IDs to load.
    augment_goals: bool; If True, use any future video frame as a goal frame.
      Else use last frame.
    augmentation_params: dict; augmentation configurations in train mode.
    keep_key: bool; If true, also return the key for each example.
    do_three_spatial_crops: If true, take three spatial crops of each clip
      during testing.

  Returns:
    A pair `(ds, num_examples)` with
      ds: A `tf.data.Dataset` object
      num_examples: Number of examples in the dataset.
  """
  def filter_on_task(features_dict: Dict[str, tf.Tensor],
                     included_tasks: tf.Tensor) -> tf.Tensor:
    label = tf.sparse.to_dense(features_dict['task_label'])
    label_tile = tf.tile(label, [included_tasks.shape[0]])
    return tf.reduce_any(tf.equal(label_tile, included_tasks), -1)

  dataset = ds_factory(subset=subset).configure(
      is_training=(subset == 'train'),
      num_frames=num_frames,
      stride=stride,
      min_resize=min_resize,
      crop_size=crop_size,
      zero_centering_image=zero_centering,
      augment_goals=(subset == 'train' and augment_goals))

  if included_tasks is not None:
    dataset.filter_builder.add_filter_fn(
        functools.partial(filter_on_task, included_tasks=included_tasks),
        builders.Phase.PARSE)

  if subset == 'train' and augmentation_params:
    dataset = video_ops.additional_augmentations(dataset, augmentation_params,
                                                 crop_size, num_frames,
                                                 zero_centering)

  if subset != 'train' and do_three_spatial_crops:
    rgb_feature_name = builders.IMAGE_FEATURE_NAME

    dataset.preprocessor_builder.replace_fn(
        f'{rgb_feature_name}_central_crop',
        functools.partial(video_ops.three_spatial_crops, crop_size=crop_size))

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


def map_keys(batch):
  """Dataset returns 'image' and 'targets'. We want 'inputs' and 'label'."""
  batch['inputs'] = batch['image']
  batch.pop('image')
  batch['label'] = batch['targets']
  return batch


def count_included_videos(video_metadata_path, included_tasks):
  with gfile.GFile(video_metadata_path) as f:
    videos = json.load(f)
  include_video = [
      video['template'].replace('[', '').replace(']', '') in included_tasks
      for video in videos]
  return np.sum(include_video)


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


@datasets.add_dataset('ssv2_regression_tfrecord')
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
  """Returns a generator for Something Something v2 train or validation set."""
  del rng  # Parameter was required by caller API, but is unused.

  def validate_config(field):
    if dataset_configs.get(field) is None:
      raise ValueError(f'{field} must be specified for TFRecord dataset.')
  validate_config('base_dir')
  validate_config('tables')
  validate_config('examples_per_subset')
  validate_config('num_classes')

  num_frames = dataset_configs.get('num_frames', 4)
  stride = dataset_configs.get('stride', 1)
  min_resize = dataset_configs.get('min_resize', 256)
  crop_size = dataset_configs.get('crop_size', 224)
  zero_centre_data = dataset_configs.get('zero_centering', True)
  augmentation_params = dataset_configs.get('augmentation_params', None)
  keep_train_key = dataset_configs.get('keep_train_key', False)
  keep_test_key = dataset_configs.get('keep_test_key', False)
  test_batch_size = dataset_configs.get('test_batch_size', eval_batch_size)
  do_three_spatial_crops = dataset_configs.get('do_three_spatial_crops', False)
  included_tasks = dataset_configs.get('included_tasks_path', None)
  train_videos_per_task = dataset_configs.get('train_metadata_path', None)
  validation_videos_per_task = dataset_configs.get('validation_metadata_path',
                                                   None)
  num_spatial_crops = 3 if do_three_spatial_crops else 1
  test_split = dataset_configs.get('test_split', 'test')

  n_train_videos = None
  n_validation_videos = None
  if included_tasks is not None:
    with gfile.GFile(included_tasks) as f:
      included_tasks = json.load(f)
    if train_videos_per_task is None or validation_videos_per_task is None:
      raise ValueError('Train and validation metadata paths must be set to '
                       'correctly determine split sizes when filtering tasks.')
    n_train_videos = count_included_videos(train_videos_per_task,
                                           included_tasks)
    n_validation_videos = count_included_videos(validation_videos_per_task,
                                                included_tasks)
    # Keep task IDs only.
    included_tasks = np.array([int(task) for task in included_tasks.values()])
    logging.info('%d included tasks: %s',
                 len(included_tasks), str(included_tasks))

  augment_goals = False
  if augmentation_params:
    augment_goals = augmentation_params.get('augment_goals', False)

  ds_factory = functools.partial(
      RegressionDatasetFactory,
      base_dir=dataset_configs.base_dir,
      tables=dataset_configs.tables,
      examples_per_subset=dataset_configs.examples_per_subset,
      num_classes=dataset_configs.num_classes,
      num_groups=jax.process_count(),
      group_index=jax.process_index())

  def create_dataset_iterator(
      subset: Text,
      batch_size_local: int,
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
        min_resize=min_resize,
        crop_size=crop_size,
        zero_centering=zero_centre_data,
        included_tasks=included_tasks,
        augment_goals=augment_goals,
        augmentation_params=augmentation_params,
        keep_key=keep_key_local,
        do_three_spatial_crops=do_three_spatial_crops and is_test)

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
      pad_batch_size = batch_size_local * num_spatial_crops
    maybe_pad_batches = functools.partial(
        dataset_utils.maybe_pad_batch,
        train=is_training,
        batch_size=pad_batch_size)
    shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)

    current_ds_iterator = (
        map_keys(dataset_utils.tf_to_numpy(data)) for data in iter(dataset)
    )

    if is_test and num_spatial_crops > 1:
      current_ds_iterator = map(tile_label_key, current_ds_iterator)

    current_ds_iterator = map(maybe_pad_batches, current_ds_iterator)
    current_ds_iterator = map(shard_batches, current_ds_iterator)
    if is_training and dataset_configs.get('prefetch_to_device'):
      # Async bind batch to device which speeds up training.
      current_ds_iterator = jax_utils.prefetch_to_device(
          current_ds_iterator, dataset_configs.get('prefetch_to_device'))

    return current_ds_iterator, num_examples

  train_iter, n_train_examples = create_dataset_iterator(
      'train', batch_size, keep_train_key)
  eval_iter, n_eval_examples = create_dataset_iterator(
      'validation', eval_batch_size, keep_train_key)
  test_iter, _ = create_dataset_iterator(
      test_split, test_batch_size, keep_test_key)
  n_train_videos = n_train_videos or n_train_examples
  n_validation_videos = n_validation_videos or n_eval_examples

  meta_data = {
      'num_targets': 1,
      'input_shape': (-1, num_frames, crop_size, crop_size, 3),
      'num_train_examples': n_train_videos,
      'num_eval_examples': n_validation_videos,
      'num_test_examples': n_validation_videos * num_spatial_crops,
      'input_dtype': getattr(jnp, dtype_str),
      'target_is_onehot': False,
  }
  logging.info(
      'Number of training examples: %d', meta_data['num_train_examples'])
  logging.info(
      'Number of validation examples: %d', meta_data['num_eval_examples'])
  logging.info(
      'Number of test examples: %d', meta_data['num_test_examples'])

  return dataset_utils.Dataset(train_iter, eval_iter, test_iter, meta_data)
