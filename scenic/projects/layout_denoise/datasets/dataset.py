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

"""Data generators for the uicomplete dataset."""

import functools
from typing import Optional

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.projects.baselines.detr import transforms
from scenic.projects.layout_denoise.datasets import parsers
import tensorflow as tf

# Computed from the coco training set by taking the per-channel mean/std-dev
# over sample, height and width axes of all training samples.
MEAN_RGB = [0.48, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]

LAYOUT_LABEL_MAP = {
    0: 'INVALID',
    1: 'IMAGE',
    2: 'PICTOGRAM',
    3: 'BUTTON',
    4: 'TEXT',
    5: 'LABEL',
    6: 'TEXT_INPUT',
    7: 'MAP',
    8: 'CHECK_BOX',
    9: 'SWITCH',
    10: 'PAGER_INDICATOR',
    11: 'SLIDER',
    12: 'RADIO_BUTTON',
    13: 'SPINNER',
    14: 'PROGRESS_BAR',
    15: 'ADVERTISEMENT',
    16: 'DRAWER',
    17: 'NAVIGATION_BAR',
    18: 'TOOLBAR',
    19: 'LIST_ITEM',
    20: 'CARD_VIEW',
    21: 'CONTAINER',
    22: 'DATE_PICKER',
    23: 'NUMBER_STEPPER',
}


def preprocess_fn(max_size=1333, train=True):
  """Returns a preprocessing function that operates on inputs and labels."""
  scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
  if not train:
    scales = [800]
  ratio = max_size / 1333.

  scales = [int(s * ratio) for s in scales]

  normalize_boxes = transforms.NormalizeBoxes()
  init_padding_mask = transforms.InitPaddingMask()

  return transforms.Compose([
      transforms.RandomResize(scales, max_size=max_size), normalize_boxes,
      init_padding_mask
  ])


def decode_layout_example(example, input_range=None, add_node_id=False):
  """Given an instance and raw labels, creates <inputs, label> pair.

  Decoding includes.
  1. Converting images from uint8 [0, 255] to [0, 1.] float32.
  2. Mean subtraction and standardization using hard-coded mean and std.
  3. Convert boxes from yxyx [0-1] to xyxy un-normalized.
  4. Add 1 to all labels to account for background/padding object at label 0.
  5. Shuffling dictionary keys to be consistent with the rest of the code.

  Args:
    example: dict; Input image and raw labels.
    input_range: tuple; Range of input. By default we use Mean and StdDev
      normalization.
    add_node_id: bool; Whether to add the node id feature.

  Returns:
    A dictionary of {'inputs': input image, 'labels': task label}.
  """
  image = tf.image.convert_image_dtype(example['image'], dtype=tf.float32)

  # Normalize.
  if input_range:
    image = image * (input_range[1] - input_range[0]) + input_range[0]
  else:
    mean_rgb = tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=tf.float32)
    std_rgb = tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=tf.float32)
    image = (image - mean_rgb) / std_rgb

  boxes = example['objects']['boxes']

  target = {
      'boxes': boxes,
      'labels': example['objects']['label'] + 1,  # 0'th class is padding.
      'binary_labels': example['objects']['binary_label'] + 1,
      'desc_id': example['objects']['desc_id'],
      'resource_id': example['objects']['resource_id'],
      'name_id': example['objects']['name_id'],
      'obj_mask': example['objects']['obj_mask'],
  }
  if add_node_id:
    target.update({
        'node_id': example['objects']['node_id'],
    })

  # Filters objects to exclude degenerate boxes.
  valid_bbx = tf.logical_and(boxes[:, 2] > boxes[:, 0],
                             boxes[:, 3] > boxes[:, 1])
  # -1 is ROOT node, remove it for training & eval.
  valid_node = tf.greater(example['objects']['label'], -1)
  keep = tf.where(tf.logical_and(valid_bbx, valid_node))[:, 0]
  target_kept = {k: tf.gather(v, keep) for k, v in target.items()}

  target_kept['orig_size'] = tf.cast(tf.shape(image)[0:2], dtype=tf.int32)
  target_kept['size'] = tf.identity(target_kept['orig_size'])
  return {
      'inputs': image,
      'label': target_kept,
  }


def _filter_tree_size(example, max_num_boxes):
  """The dataset filter fn."""
  return tf.less_equal(tf.size(example['objects']['label']), max_num_boxes)


def _filter_invalid_bbx(example):
  valid_box = tf.reduce_all(tf.greater(example['label']['boxes'][:, 2:], 0))
  least_num_boxes = tf.greater(tf.size(example['label']['boxes']), 3)
  return tf.logical_and(valid_box, least_num_boxes)


def get_data_split(ds, host_id, host_count, data_length):
  """Return a (sub)split adapted to a given host."""
  full_start = 0
  full_end = data_length
  examples_per_host = (full_end - full_start) // host_count
  host_start = full_start + examples_per_host * host_id
  host_end = full_start + examples_per_host * (host_id + 1)
  ds = ds.skip(host_start)
  ds = ds.take(host_end - host_start)
  return ds


def load_dataset(file_patterns,
                 dataset_configs,
                 max_num_boxes,
                 num_examples,
                 cache=False):
  """Loads a split from the COCO dataset using TensorFlow Datasets.

  Args:
    file_patterns: the data file patterns.
    dataset_configs: the dataset_configs dict.
    max_num_boxes: the maximum number of boxes allowed.
    num_examples: the number of examples in the data.
    cache: bool; whether to use the ds.cache or nor.

  Returns:
    A `tf.data.Dataset`, and dataset info.
  """
  del num_examples

  if not isinstance(file_patterns, (list,)):
    file_patterns = [file_patterns]
  data_files = [tf.io.matching_files(f) for f in file_patterns]
  logging.info('File patterns: %s', file_patterns)
  logging.info('Data files: %s', data_files)
  ds = tf.data.Dataset.from_tensor_slices(data_files)
  ds = ds.interleave(
      tf.data.Dataset.from_tensor_slices,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=True)

  # Each host is responsible for a fixed subset of data. We shard based on the
  # input files.
  num_data_files = sum([int(files.shape[0]) for files in data_files])
  logging.info('Number of data files: %d (before split)', num_data_files)
  assert num_data_files > jax.process_count(), (
      'Number of files must be larger '
      'than the number of hosts.')
  ds = get_data_split(ds, jax.process_index(), jax.process_count(),
                      num_data_files)

  ds = ds.interleave(
      tf.data.TFRecordDataset,
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False)

  def parse_fn(v):
    return parsers.parse(v)

  ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
  filter_fn = functools.partial(
      _filter_tree_size, max_num_boxes=max_num_boxes - 1)
  ds = ds.filter(filter_fn)
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  ds = ds.with_options(options)
  decode_fn = functools.partial(
      decode_layout_example, input_range=dataset_configs.get('input_range'))
  ds = ds.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)
  if cache:
    ds = ds.cache()
  return ds


@datasets.add_dataset('layout_denoise')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                config=None,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None):
  """Returns generators for UIcomplete train, validation and test set.

  Args:
    batch_size: the train batch size.
    eval_batch_size: the eval batch size.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image. Only 'float32' is currently supported.
    shuffle_seed: int; Seed for shuffling the training data.
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    config: the overall config.
    dataset_configs: the dataset config.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
    a test_iter, and a dict of meta_data.
  """
  del rng
  if dataset_service_address:
    raise ValueError('Dataset service is not supported for this dataset yet.')

  assert dtype_str == 'float32', (
      f'coco_detr_dataset invoked with unsupported dtype_str: {dtype_str}')
  del dtype_str

  config = config or ml_collections.ConfigDict({
      # These can be used for testing the dataset:
      'max_num_boxes': 50,
      'max_image_size': 1333,
  })

  max_size = config.max_image_size
  max_num_boxes = config.max_num_boxes
  train_ds = load_dataset(
      dataset_configs['train_files'],
      dataset_configs,
      max_num_boxes=max_num_boxes,
      num_examples=dataset_configs['num_train_examples'],
      cache=False)
  eval_ds = load_dataset(
      dataset_configs['eval_files'],
      dataset_configs,
      max_num_boxes=max_num_boxes,
      num_examples=dataset_configs['num_eval_examples'],
      cache=False)

  padded_shapes = {
      'inputs': [max_size, max_size, 3],
      'padding_mask': [max_size, max_size],
      'label': {
          'boxes': [max_num_boxes, 4],
          'labels': [max_num_boxes,],
          'binary_labels': [max_num_boxes,],
          'orig_size': [2,],
          'size': [2,],
          'desc_id': [max_num_boxes, 10],
          'resource_id': [max_num_boxes, 10],
          'name_id': [max_num_boxes, 10],
          'obj_mask': [max_num_boxes,],
      }
  }

  def _shuffle_batch(ds, bs, train):
    if train:
      # First repeat then batch.
      ds = ds.shuffle(
          dataset_configs.get('shuffle_buffer_size', 1000), seed=shuffle_seed)
      ds = ds.repeat()
      # Augmentation should be done after repeat for true randomness.
      ds = ds.map(
          preprocess_fn(max_size=max_size, train=True),
          num_parallel_calls=tf.data.AUTOTUNE)
      ds = ds.filter(_filter_invalid_bbx)
      ds = ds.padded_batch(bs, padded_shapes=padded_shapes, drop_remainder=True)

    else:
      ds = ds.map(
          preprocess_fn(max_size=max_size, train=False),
          num_parallel_calls=tf.data.AUTOTUNE)
      ds = ds.filter(_filter_invalid_bbx)
      # First batch then repeat.
      ds = ds.padded_batch(
          bs, padded_shapes=padded_shapes, drop_remainder=False)
      ds = ds.repeat()

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

  eval_ds = _shuffle_batch(eval_ds, eval_batch_size, train=False)
  train_ds = _shuffle_batch(train_ds, batch_size, train=True)

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

  meta_data = {
      'task_name': dataset_configs['task_name'],
      'input_shape': (-1, max_size, max_size, 3),
      'input_dtype': jnp.float32,
      'obj_bbx_shape': (-1, max_num_boxes, 4),
      'obj_bbx_dtype': jnp.float32,
      'obj_desc_id_shape': (-1, max_num_boxes, parsers.MAX_WORD_NUM),
      'obj_desc_id_dtype': jnp.int32,
      'obj_resource_id_shape': (-1, max_num_boxes, parsers.MAX_WORD_NUM),
      'obj_resource_id_dtype': jnp.int32,
      'obj_name_id_shape': (-1, max_num_boxes, parsers.MAX_WORD_NUM),
      'obj_name_id_dtype': jnp.int32,
      'obj_mask_shape': (-1, max_num_boxes),
      'obj_mask_dtype': jnp.int32,
      'num_train_examples': dataset_configs.num_train_examples,
      'num_eval_examples': dataset_configs.num_eval_examples,
  }
  return dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)
