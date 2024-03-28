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

"""Data generators for the object detection.

The file is modified from
https://github.com/google-research/scenic/blob/main/scenic/projects/baselines/
detr/input_pipeline_detection.py
"""

import functools
from typing import Optional
from absl import logging

from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections

from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib.coco_dataset import coco_utils
from scenic.projects.baselines.centernet import transforms
import tensorflow as tf
import tensorflow_datasets as tfds

PRNGKey = jnp.ndarray


def make_resize_crop_transforms(
    image_set,
    scale_range=(0.1, 2.0),
    crop_size=1024):
  """Preprocessing and data-augmentation functions.

    Currently it only supports the default data augmentation in detectron2.

  Args:
    image_set: 'train' or 'validation'
    scale_range: list of integers. Sizes of the shorter edge.
    crop_size: integer. Size of the longer edge.
  Returns:
    The data-augmentation functions.
  """
  init_padding_mask = transforms.InitPaddingMask()
  if image_set == 'train':
    return transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomRatioResize(scale_range, crop_size),
         transforms.FixedSizeCrop(crop_size),
         init_padding_mask])
  elif image_set == 'validation':
    return transforms.Compose(
        [transforms.Resize(crop_size, max_size=crop_size),
         init_padding_mask])
  else:
    raise ValueError(f'Unknown image_set: {image_set}')


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


# From tfrecord official: https://www.tensorflow.org/datasets/catalog/coco
coco_feature_description = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/id': tf.io.FixedLenFeature([], tf.int64),
    'objects/bbox': tf.io.VarLenFeature(dtype=tf.float32),
    'objects/area': tf.io.VarLenFeature(dtype=tf.int64),
    'objects/is_crowd': tf.io.VarLenFeature(dtype=tf.int64),
    'objects/id': tf.io.VarLenFeature(dtype=tf.int64),
    'objects/label': tf.io.VarLenFeature(dtype=tf.int64),
    'objects/segmentation': tf.io.VarLenFeature(tf.string),
}



def coco_decode_example(data, with_masks=False):
  """Convert custom tfrecord into tfds builder format."""
  example = {}
  example['image'] = tf.io.decode_jpeg(data['image/encoded'], channels=3)
  example['image/id'] = data['image/id']
  example['objects'] = {}
  example['objects']['bbox'] = tf.reshape(
      tf.sparse.to_dense(data['objects/bbox']), [-1, 4])
  example['objects']['id'] = tf.sparse.to_dense(data['objects/id'])
  example['objects']['area'] = tf.sparse.to_dense(data['objects/area'])
  example['objects']['is_crowd'] = tf.cast(
      tf.sparse.to_dense(data['objects/is_crowd']), tf.bool)
  example['objects']['label'] = tf.sparse.to_dense(data['objects/label'])
  num_objs = tf.shape(example['objects']['id'])[0]
  if with_masks:
    segmentation = tf.sparse.to_dense(data['objects/segmentation'])
    tf.debugging.assert_equal(num_objs, tf.shape(segmentation)[0])
    height, width, _ = tf.unstack(tf.shape(example['image']))
    if num_objs > 0:
      segmentation = tf.map_fn(
          lambda x: tf.image.decode_jpeg(x, channels=1),
          segmentation, back_prop=False, dtype=tf.uint8)
    else:
      segmentation = tf.zeros((0,), dtype=tf.uint8)
    example['objects']['segmentation'] = tf.reshape(
        segmentation, [num_objs, height, width, 1])
  return example


def decode_sharded_names(paths, end=''):
  """Convert sharded file names into a list."""
  ret = []
  paths = paths.split(',')
  for name in paths:
    if '@' in name:
      idx = name.find('@')
      if end:
        num_shards = int(name[idx + 1:-len(end)])
      else:
        num_shards = int(name[idx + 1:])
      names = ['{}-{:05d}-of-{:05d}{}'.format(
          name[:idx], i, num_shards, end) for i in range(num_shards)]
      ret.extend(names)
    else:
      ret.append(name)
  return ret


def decode_coco_detection_example(
    example, max_boxes=100, model_input_format='RGB',
    remove_crowd=False, class_id_base=0, with_masks=False):
  """Given an instance and raw labels, creates <inputs, label> pair.

  Modified: not add 1 to class label

  Decoding includes.
  1. Convert RGB to BGR if the model uses BGR format.
  2. Convert boxes from yxyx [0-1] to xyxy un-normalized.
  3. Shuffling dictionary keys to be consistent with the rest of the code.

  Args:
    example: dict; Input image and raw labels.
    max_boxes: int; max number of objects to load.
    model_input_format: string "RGB" or "BGR". The input format from tfds is
      always RGB. Reverse the pixel order here if the model needs BGR.
    remove_crowd: remove objects labeled as 'is_crowd'
    class_id_base: int; make sure the output class id is 0-based.
    with_masks: bool;
  Returns:
    A dictionary of {'inputs': input image, 'labels': task label}.
  """
  image = tf.cast(example['image'], tf.float32)
  if model_input_format == 'BGR':
    image = image[:, :, ::-1]  # RGB to BGR

  boxes = decode_boxes(example['objects']['bbox'], tf.shape(image)[0:2])
  target = {
      'area': example['objects']['area'],
      'boxes': boxes,
      'objects/id': example['objects']['id'],
      'is_crowd': example['objects']['is_crowd'],
      'labels': example['objects']['label'] - class_id_base,
  }
  if with_masks:
    target['masks'] = example['objects']['segmentation']

  if remove_crowd:
    keep = tf.where(
        tf.logical_and(
            tf.logical_not(example['objects']['is_crowd']),
            tf.logical_and(
                boxes[:, 2] > boxes[:, 0], boxes[:, 3] > boxes[:, 1])
        )
    )[:, 0]
  else:
    # Filters objects to exclude degenerate boxes.
    keep = tf.where(tf.logical_and(
        boxes[:, 2] > boxes[:, 0], boxes[:, 3] > boxes[:, 1]))[:, 0]
  target_kept = {k: tf.gather(v, keep)[:max_boxes] for k, v in target.items()}

  target_kept['orig_size'] = tf.cast(tf.shape(image)[0:2], dtype=tf.int32)
  target_kept['size'] = tf.identity(target_kept['orig_size'])
  target_kept['image/id'] = example['image/id']

  return {
      'inputs': image,
      'label': target_kept,
  }


def coco_load_split_from_tfds(
    batch_size,
    train,
    preprocess_fn,
    decode_fn,
    dataset_path='coco/2017',
    cache=False,
    max_size=1333,
    max_boxes=100,
    remove_crowd=True,
    with_masks=False,
    shuffle_buffer_size=1000,
    shuffle_seed=0):
  """Loads a split from the COCO dataset using TensorFlow Datasets.

  Args:
    batch_size: int; The batch size returned by the data pipeline.
    train: bool; Whether to load the train or evaluation split.
    preprocess_fn: function; A function that given an example, train flag,
      and dtype returns the preprocessed the example. Note that the
      preprocessing is done BEFORE caching to re-use them.
    decode_fn: A function that given an example decodes the image, converts
      it to float32, mean-subtracts it, and pulls out the relevant parts from
      the tfds features.
    dataset_path: string; path of the dataset; by default load from tfds
    cache: bool; whether to use the ds.cache or nor.
    max_size: int; Maximum image size.
    max_boxes: int; Maximum number of boxes.
    remove_crowd: bool; Remove objects labeled with 'is_crowd'
    with_masks: bool; If include instance segmentation masks.
    shuffle_buffer_size: int; Size of the shuffle buffer.
    shuffle_seed: int; Seed for shuffling the training data.

  Returns:
    A `tf.data.Dataset`, and dataset info.
  """
  split = 'train' if train else 'validation'

  if dataset_path == 'coco/2017':
    builder = tfds.builder('coco/2017')
    # Each host is responsible for a fixed subset of data.
    data_range = tfds.even_splits(
        split, jax.process_count())[jax.process_index()]
    ds = builder.as_dataset(split=data_range, shuffle_files=False)
    ds_info = {
        'num_classes': builder.info.features['objects']['label'].num_classes}
    class_id_base = 0
  else:
    feature_description = coco_feature_description
    end = ''
    decode_example_fn = lambda x: coco_decode_example(x, with_masks)
    class_id_base = 0
    ds = tf.data.TFRecordDataset(decode_sharded_names(dataset_path, end=end))
    # Split datasets into machines. Otherwise multi-machine evaluation takes the
    # same images.
    ds = ds.shard(jax.process_count(), jax.process_index())
    ds = ds.map(
        lambda x: tf.io.parse_single_example(x, feature_description))
    ds = ds.map(decode_example_fn)
    ds_info = {}
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)
  ds = ds.map(
      lambda x: decode_fn(  # pylint: disable=g-long-lambda
          x, remove_crowd=train and remove_crowd, class_id_base=class_id_base,
          with_masks=with_masks),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if cache:
    ds = ds.cache()

  padded_shapes = {
      'inputs': [max_size, max_size, 3],
      'padding_mask': [max_size, max_size],
      'label': {
          'area': [max_boxes,],
          'boxes': [max_boxes, 4],
          'objects/id': [max_boxes,],
          'is_crowd': [max_boxes,],
          'labels': [max_boxes,],
          'image/id': [],
          'orig_size': [2,],
          'size': [2,]
      },
  }

  if with_masks:
    padded_shapes['label']['masks'] = [max_boxes, max_size, max_size, 1]

  if train:
    # First repeat then batch.
    ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    ds = ds.repeat()
    # Augmentation should be done after repeat for true randomness.
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.padded_batch(batch_size, padded_shapes=padded_shapes,
                         drop_remainder=True)

  else:
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # First batch then repeat.
    ds = ds.padded_batch(batch_size, padded_shapes=padded_shapes,
                         drop_remainder=False)
    ds = ds.repeat()

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds, ds_info


def dataset_builder(*,
                    batch_size,
                    eval_batch_size,
                    num_shards,
                    dtype_str='float32',
                    shuffle_seed=0,
                    rng=None,
                    dataset_configs=None,
                    dataset_service_address: Optional[str] = None):
  """Returns generators for COCO object detection 2017 train & validation set.

  Args:
    batch_size: int; Determines the train batch size.
    eval_batch_size: int; Determines the evaluation batch size.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image. Only 'float32' is currently supported.
    shuffle_seed: int; Seed for shuffling the training data.
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    dataset_configs: dict; Dataset specific configurations. Must be empty.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
    a test_iter, and a dict of meta_data.
  """
  del rng
  assert dtype_str == 'float32', (
      f'coco_detr_dataset invoked with unsupported dtype_str: {dtype_str}')
  del dtype_str

  dataset_configs = dataset_configs or {}

  scale_range = dataset_configs.get('scale_range', (0.1, 2.0))
  crop_size = dataset_configs.get('crop_size', 1024)
  max_boxes = dataset_configs.get('max_boxes', 100)
  size_divisibility = dataset_configs.get('size_divisibility', 1)
  model_input_format = dataset_configs.get('model_input_format', 'RGB')
  remove_crowd = dataset_configs.get('remove_crowd', True)
  train_data_path = dataset_configs.get('train_data_path', 'coco/2017')
  test_data_path = dataset_configs.get('test_data_path', 'coco/2017')
  with_masks = dataset_configs.get('with_masks', False)

  assert model_input_format in ['RGB', 'BGR'], model_input_format
  crop_size = ((crop_size - 1) // size_divisibility + 1) * size_divisibility

  train_preprocess_fn = make_resize_crop_transforms(
      'train', scale_range=scale_range, crop_size=crop_size)
  eval_preprocess_fn = make_resize_crop_transforms(
      'validation', scale_range=(1.0, 1.0), crop_size=crop_size)

  decode_fn = functools.partial(
      decode_coco_detection_example, max_boxes=max_boxes,
      model_input_format=model_input_format)

  train_ds, train_ds_info = coco_load_split_from_tfds(
      batch_size, train=True,
      preprocess_fn=train_preprocess_fn,
      decode_fn=decode_fn,
      dataset_path=train_data_path,
      shuffle_buffer_size=dataset_configs.get('shuffle_buffer_size', 1000),
      max_size=crop_size,
      max_boxes=max_boxes,
      remove_crowd=remove_crowd,
      with_masks=with_masks,
      shuffle_seed=shuffle_seed)

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you '
                       'want to run with dataset service.')
    logging.info('Using the tf.data service at %s', dataset_service_address)
    train_ds = dataset_utils.distribute(train_ds, dataset_service_address)

  eval_ds, _ = coco_load_split_from_tfds(
      eval_batch_size,
      train=False,
      preprocess_fn=eval_preprocess_fn,
      max_size=crop_size,
      max_boxes=max_boxes,
      decode_fn=decode_fn,
      with_masks=with_masks,
      dataset_path=test_data_path)

  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size)
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch, train=False, batch_size=eval_batch_size)
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)

  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(shard_batches, train_iter)
  if dataset_configs.get('prefetch_to_device'):
    # Async bind batch to device which speeds up training.
    train_iter = jax_utils.prefetch_to_device(
        train_iter, dataset_configs.get('prefetch_to_device'))

  eval_iter = iter(eval_ds)
  eval_iter = map(dataset_utils.tf_to_numpy, eval_iter)
  eval_iter = map(maybe_pad_batches_eval, eval_iter)
  eval_iter = map(shard_batches, eval_iter)

  num_classes = dataset_configs.get(
      'num_classes', train_ds_info.get('num_classes', 1))
  num_train_examples = dataset_configs.get(
      'num_train_examples',
      dataset_utils.get_num_examples('coco/2017', 'train'))
  num_eval_examples = dataset_configs.get(
      'num_eval_examples',
      dataset_utils.get_num_examples('coco/2017', 'validation'))
  label_to_name = coco_utils.get_label_map('coco/2017_panoptic')

  meta_data = {
      'num_classes': num_classes,
      'input_shape': [-1, crop_size, crop_size, 3],
      'num_train_examples': num_train_examples,
      'num_eval_examples': num_eval_examples,
      'input_dtype': jnp.float32,
      'target_is_onehot': False,
      'label_to_name': label_to_name,
  }
  return dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)


def get_dataset(
    config: ml_collections.ConfigDict,
    data_rng: PRNGKey,
    *,
    dataset_service_address: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_configs: Optional[ml_collections.ConfigDict] = None
) -> dataset_utils.Dataset:
  """Creates dataset.

  By default, the values in the config file are used.
  However, if the optional `dataset_name` and `dataset_configs` are passed,
    those are used instead.

  Args:
    config: The configuration of the experiment.
    data_rng: Random number generator key to use for the dataset.
    dataset_service_address: Used when using the tf.data.experimental.service
    dataset_name: Name of dataset to load, if not reading from the config.
    dataset_configs: Configuration of the dataset, if not reading directly from
      the config.

  Returns:
    A dataset_utils.Dataset object.
  """
  device_count = jax.device_count()
  logging.info('device_count: %d', device_count)
  logging.info('num_hosts : %d', jax.process_count())
  logging.info('host_id : %d', jax.process_index())

  del dataset_name

  batch_size = config.batch_size
  if batch_size % device_count > 0:
    raise ValueError(f'Batch size ({batch_size}) must be divisible by the '
                     f'number of devices ({device_count})')

  eval_batch_size = config.get('eval_batch_size', batch_size)
  if eval_batch_size % device_count > 0:
    raise ValueError(f'Eval batch size ({eval_batch_size}) must be divisible '
                     f'by the number of devices ({device_count})')

  local_batch_size = batch_size // jax.process_count()
  eval_local_batch_size = eval_batch_size // jax.process_count()
  device_batch_size = batch_size // device_count
  logging.info('local_batch_size : %d', local_batch_size)
  logging.info('device_batch_size : %d', device_batch_size)

  shuffle_seed = config.get('shuffle_seed', None)
  if dataset_service_address and shuffle_seed is not None:
    raise ValueError('Using dataset service with a random seed causes each '
                     'worker to produce exactly the same data. Add '
                     'config.shuffle_seed = None to your config if you want '
                     'to run with dataset service.')

  dataset_configs = dataset_configs or config.get('dataset_configs')
  dataset = dataset_builder(
      batch_size=local_batch_size,
      eval_batch_size=eval_local_batch_size,
      num_shards=jax.local_device_count(),
      dtype_str=config.data_dtype_str,
      rng=data_rng,
      shuffle_seed=shuffle_seed,
      dataset_configs=dataset_configs,
      dataset_service_address=dataset_service_address)

  return dataset
