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

"""Data generator for COCO with Deformable DETR."""
import functools
from typing import Any, Callable, Mapping, Optional

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.dataset_lib.coco_dataset import coco_utils
from scenic.projects.baselines.detr import transforms
import tensorflow as tf
import tensorflow_datasets as tfds

DecodeExampleFn = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class ImageNetNormalization:
  """Performs standard ImageNet normalization.

  Note: ImageNet normalization is separated out as a module to fix bugs in the
  original DETR porting to JAX. Bugs:
    (1) 0.485 in mean_rgb was typoed as 0.48. (2) To be consistent with the
    pytorch implementation, rgb normalization should happen after data
    augmentation instead of before.
  """

  def __init__(self):
    # Computed from a subset of the training set. The original (randomly
    # generated) subset used for calculation and the exact way of calculation is
    # not known to the public, but the numbers have been widely adopted since.
    self.mean_rgb = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    self.std_rgb = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

  def __call__(self, features):
    features['inputs'] = (features['inputs'] - self.mean_rgb) / self.std_rgb
    return features


def make_coco_transforms(image_set, max_size=1333):
  """Returns a preprocessing function that operates on inputs and labels."""
  scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
  ratio = max_size / 1333.

  scales = [int(s * ratio) for s in scales]
  scales2 = [int(s * ratio) for s in [400, 500, 600]]

  normalize_boxes = transforms.NormalizeBoxes()
  normalize_image = ImageNetNormalization()
  init_padding_mask = transforms.InitPaddingMask()

  if image_set == 'train':
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomSelect(
            transforms.RandomResize(scales, max_size=max_size),
            transforms.Compose([
                transforms.RandomResize(scales2),
                transforms.RandomSizeCrop(int(ratio * 384), int(ratio * 600)),
                transforms.RandomResize(scales, max_size=max_size),
            ])), normalize_boxes, normalize_image, init_padding_mask
    ])

  elif image_set == 'validation':
    return transforms.Compose([
        transforms.Resize(max(scales), max_size=max_size),
        normalize_boxes, normalize_image, init_padding_mask
    ])

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


def decode_coco_detection_example(example):
  """Given an instance and raw labels, creates <inputs, label> pair.

  Decoding includes.
  1. Converting images from uint8 [0, 255] to [0, 1.] float32.
  2. Mean subtraction and standardization using hard-coded mean and std.
  3. Convert boxes from yxyx [0-1] to xyxy un-normalized.
  4. Add 1 to all labels to account for background/padding object at label 0.
  5. Shuffling dictionary keys to be consistent with the rest of the code.

  Args:
    example: dict; Input image and raw labels.

  Returns:
    A dictionary of {'inputs': input image, 'labels': task label}.
  """
  image = tf.image.convert_image_dtype(example['image'], dtype=tf.float32)

  boxes = decode_boxes(example['objects']['bbox'], tf.shape(image)[0:2])

  target = {
      'area': example['objects']['area'],
      'boxes': boxes,
      'objects/id': example['objects']['id'],
      'is_crowd': example['objects']['is_crowd'],
      'labels': example['objects']['label'] + 1,  # 0'th class is padding.
  }

  # Filters objects to exclude degenerate boxes.
  keep = tf.where(tf.logical_and(boxes[:, 2] > boxes[:, 0],
                                 boxes[:, 3] > boxes[:, 1]))[:, 0]
  target_kept = {k: tf.gather(v, keep) for k, v in target.items()}

  target_kept['orig_size'] = tf.cast(tf.shape(image)[0:2], dtype=tf.int32)
  target_kept['size'] = tf.identity(target_kept['orig_size'])
  target_kept['image/id'] = example['image/id']

  return {
      'inputs': image,
      'label': target_kept,
  }


# This is only overridden from DETR to take a data_dir.
def coco_load_split_from_tfds(batch_size: int,
                              *,
                              train: bool,
                              preprocess_fn: Callable[..., Any],
                              decode_fn: DecodeExampleFn,
                              cache: bool = False,
                              max_size: int = 1333,
                              max_boxes: int = 100,
                              shuffle_buffer_size: int = 1000,
                              shuffle_seed: int = 0,
                              data_dir: Optional[str] = None):
  """Loads a split from the COCO dataset using TensorFlow Datasets.

  Args:
    batch_size: The batch size returned by the data pipeline.
    train: Whether to load the train or evaluation split.
    preprocess_fn: A function that given an example, train flag, and dtype
      returns the preprocessed the example. Note that the preprocessing is done
      BEFORE caching to re-use them.
    decode_fn: A function that given an example decodes the image, converts it
      to float32, and pulls out the relevant parts from the tfds features.
    cache: whether to use the ds.cache or nor.
    max_size: Maximum image size.
    max_boxes: Maximum number of boxes.
    shuffle_buffer_size: Size of the shuffle buffer.
    shuffle_seed: Seed for shuffling the training data.
    data_dir: Passed to tfds.builder to optionally specify where to load from.

  Returns:
    A `tf.data.Dataset`, and dataset info.
  """
  split = 'train' if train else 'validation'
  builder = tfds.builder('coco/2017', data_dir=data_dir)

  # Each host is responsible for a fixed subset of data.
  data_range = tfds.even_splits(split, jax.process_count())[jax.process_index()]
  ds = builder.as_dataset(split=data_range, shuffle_files=False)
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)
  ds = ds.map(decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if cache:
    ds = ds.cache()

  # TLDR: make sure max_boxes is set >=64.
  # NOTE: the number of boxes/labels always needs to be strictly larger than 63
  #   to ensure that there is at least one dummy target corresponding
  #   to an empty bounding box, and that the last target box is such a dummy
  #   empty target. This is needed for matching functions that in principle only
  #   produce matches with non-empty target boxes, and produce dummy matches
  #   with an empty target for the rest of the unmatched predicted boxes. The
  #   latter behaviour is necessary to ensure that the number of matches per
  #   datapoint is the same for all datapoints and shapes are static and jit
  #   compatible.
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

  if train:
    # First repeat then batch.
    ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    ds = ds.repeat()
    # Augmentation should be done after repeat for true randomness.
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.padded_batch(
        batch_size, padded_shapes=padded_shapes, drop_remainder=True)

  else:
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # First batch then repeat.
    ds = ds.padded_batch(
        batch_size, padded_shapes=padded_shapes, drop_remainder=False)
    ds = ds.repeat()

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds, builder.info


def decode_coco_deformable_detr_example(
    example: Mapping[str, Any]) -> Mapping[str, Any]:
  """Given an instance and raw labels, creates <inputs, label> pair.

  Calls DETR decoder and then just converts the labels to those expected by
  deformable detr. Labels are the exact label indices in the coco map where
  max id is 90. Whereas, detr labels are the index of the sorted COCO ids.

  Args:
    example: Input image and raw labels.

  Returns:
    A dictionary of {'inputs': input image, 'labels': task label}.
  """
  result = decode_coco_detection_example(example)

  # Remap labels.
  label_ref_keys = sorted(list(coco_utils.get_label_map('ref_coco').keys()))
  label_keys = tf.range(len(label_ref_keys), dtype=tf.int64)
  label_vals = tf.constant(label_ref_keys, dtype=tf.int64)
  label_map = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(label_keys, label_vals),
      default_value=0)
  labels = label_map.lookup(result['label']['labels'])
  labels = tf.cast(labels, tf.int32)

  result['label']['labels'] = labels
  return result


@datasets.add_dataset('coco_deformable_detr_detection')
def get_dataset(*,
                batch_size: int,
                eval_batch_size: int,
                num_shards: int,
                dtype_str: str = 'float32',
                shuffle_seed: int = 0,
                rng: Optional[jnp.ndarray] = None,
                dataset_configs: Optional[Mapping[str, Any]] = None,
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

  max_size = dataset_configs.get('max_size', 1333)
  max_boxes = dataset_configs.get('max_boxes', 100)

  train_preprocess_fn = make_coco_transforms('train', max_size)
  valid_max_size = dataset_configs.get('valid_max_size', max_size)
  eval_preprocess_fn = make_coco_transforms('validation', valid_max_size)

  decode_fn = decode_coco_deformable_detr_example

  train_ds, train_ds_info = coco_load_split_from_tfds(
      batch_size,
      train=True,
      preprocess_fn=train_preprocess_fn,
      decode_fn=decode_fn,
      shuffle_buffer_size=dataset_configs.get('shuffle_buffer_size', 1000),
      max_size=max_size,
      max_boxes=max_boxes,
      shuffle_seed=shuffle_seed,
      data_dir=dataset_configs.get('data_dir'))

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
      max_size=valid_max_size,
      max_boxes=max_boxes,
      decode_fn=decode_fn,
      data_dir=dataset_configs.get('data_dir'))

  # Labels take on values 1-90. We set 0 to be padded objects.
  num_classes = train_ds_info.features['objects']['label'].num_classes + 1

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
      'num_classes':
          num_classes,
      'input_shape': [-1, max_size, max_size, 3],
      'num_train_examples':
          dataset_utils.get_num_examples(
              'coco/2017', 'train', data_dir=dataset_configs.get('data_dir')),
      'num_eval_examples':
          dataset_utils.get_num_examples(
              'coco/2017',
              'validation',
              data_dir=dataset_configs.get('data_dir')),
      'input_dtype':
          jnp.float32,
      'target_is_onehot':
          False,
      'label_to_name':
          coco_utils.get_label_map('ref_coco'),
  }
  return dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)
