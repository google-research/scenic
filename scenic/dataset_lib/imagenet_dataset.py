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

"""Data generators for the ImageNet dataset."""

import functools
from typing import Optional

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import tensorflow as tf
import tensorflow_datasets as tfds

TRAIN_IMAGES = 1281167
EVAL_IMAGES = 50000
NUM_CLASSES = 1000

IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: TF tensor; Binary image data.
    bbox: `Tensor; Bounding boxes arranged `[1, num_boxes, coords]` where each
      coordinate is [0, 1) and the coordinates are arranged as `[ymin, xmin,
      ymax, xmax]`. If num_boxes is 0 then use the whole image.
    min_object_covered: float; Defaults to `0.1`. The cropped area of the image
      must contain at least this fraction of any bounding box supplied.
    aspect_ratio_range: list[float]; The cropped area of the image must have an
      aspect ratio = width / height within this range.
    area_range: list[float]; The cropped area of the image must contain a
      fraction of the supplied image within in this range.
    max_attempts: int; Number of attempts at generating a cropped region of the
      image of the specified constraints. After `max_attempts` failures, return
      the entire image.

  Returns:
    Cropped image TF Tensor.
  """
  shape = tf.image.extract_jpeg_shape(image_bytes)
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      shape,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

  return image


def _resize(image, image_size):
  """Resizes the image.

  Args:
    image: Tensor; Input image.
    image_size: int; Image size.

  Returns:
    Resized image.
  """
  return tf.image.resize([image], [image_size, image_size],
                         method=tf.image.ResizeMethod.BICUBIC)[0]


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes, image_size):
  """Make a random crop of `image_size`."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10)
  original_shape = tf.image.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(bad, lambda: _decode_and_center_crop(image_bytes, image_size),
                  lambda: _resize(image, image_size))

  return image


def _decode_and_center_crop(image_bytes, image_size):
  """Crops to center of image with padding then scales `image_size`."""
  shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([
      offset_height, offset_width, padded_center_crop_size,
      padded_center_crop_size
  ])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = _resize(image, image_size)

  return image


def normalize_image(image):
  image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
  return image


def preprocess_for_train(image_bytes,
                         dtype=tf.float32,
                         image_size=IMAGE_SIZE,
                         data_augmentations=None):
  """Preprocesses the given image for training.

  Args:
    image_bytes: Tensor; Representing an image binary of arbitrary size.
    dtype: TF data type; Data type of the image.
    image_size: int; The target size of the images.
    data_augmentations: list(str); Types of data augmentation applied on
      training data.

  Returns:
    A preprocessed image `Tensor`.
  """
  if data_augmentations is not None:
    if 'default' in data_augmentations:
      image = _decode_and_random_crop(image_bytes, image_size)
      image = tf.reshape(image, [image_size, image_size, 3])
      image = tf.image.random_flip_left_right(image)
  else:
    image = _decode_and_center_crop(image_bytes, image_size)
    image = tf.reshape(image, [image_size, image_size, 3])

  if dtype not in [tf.int32, tf.int64, tf.uint32, tf.uint64]:
    image = normalize_image(image)
    image = tf.image.convert_image_dtype(image, dtype=dtype)
  else:
    image = tf.cast(image, dtype=dtype)
  return image


def preprocess_for_eval(image_bytes, dtype=tf.float32, image_size=IMAGE_SIZE):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: Tensor; Representing an image binary of arbitrary size.
    dtype: TF data type; Data type of the image.
    image_size: int; The target size of the images.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_center_crop(image_bytes, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  if dtype not in [tf.int32, tf.int64, tf.uint32, tf.uint64]:
    image = normalize_image(image)
    image = tf.image.convert_image_dtype(image, dtype=dtype)
  else:
    image = tf.cast(image, dtype=dtype)
  return image


def imagenet_load_split(batch_size,
                        train,
                        onehot_labels,
                        dtype=tf.float32,
                        image_size=IMAGE_SIZE,
                        prefetch_buffer_size=10,
                        shuffle_seed=None,
                        data_augmentations=None):
  """Creates a split from the ImageNet dataset using TensorFlow Datasets.

  For the training set, we drop the last partial batch. This is fine to do
  because we additionally shuffle the data randomly each epoch, thus the trainer
  will see all data in expectation. For the validation set, we pad the final
  batch to the desired batch size.

  Args:
    batch_size: int; The batch size returned by the data pipeline.
    train: bool; Whether to load the train or evaluation split.
    onehot_labels: Whether to transform the labels to one hot.
    dtype: TF data type; Data type of the image.
    image_size: int; The target size of the images.
    prefetch_buffer_size: int; Buffer size for the TFDS prefetch.
    shuffle_seed: The seed to use when shuffling the train split.
    data_augmentations: list(str); Types of data augmentation applied on
      training data.

  Returns:
    A `tf.data.Dataset`.
  """
  if train:
    split_size = TRAIN_IMAGES // jax.process_count()
    start = jax.process_index() * split_size
    split = 'train[{}:{}]'.format(start, start + split_size)
  else:
    split_size = EVAL_IMAGES // jax.process_count()
    start = jax.process_index() * split_size
    split = 'validation[{}:{}]'.format(start, start + split_size)

  def decode_example(example):
    if train:
      image = preprocess_for_train(example['image'], dtype, image_size,
                                   data_augmentations)
    else:
      image = preprocess_for_eval(example['image'], dtype, image_size)

    label = example['label']
    label = tf.one_hot(label, NUM_CLASSES) if onehot_labels else label
    return {'inputs': image, 'label': label}

  dataset_builder = tfds.builder('imagenet2012:5.*.*')
  # Download dataset:
  dataset_builder.download_and_prepare()
  ds = dataset_builder.as_dataset(
      split=split, decoders={
          'image': tfds.decode.SkipDecoding(),
      })
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  ds = ds.cache()

  if train:
    ds = ds.repeat()
    ds = ds.shuffle(16 * batch_size, seed=shuffle_seed)

  # decode_example should be applied after caching as it also does augmentation
  ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size, drop_remainder=train)

  if not train:
    ds = ds.repeat()

  ds = ds.prefetch(prefetch_buffer_size)
  return ds


@datasets.add_dataset('imagenet')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                prefetch_buffer_size=2,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None):
  """Returns generators for the ImageNet train, validation, and test sets.

  Args:
    batch_size: int; Determines the train batch size.
    eval_batch_size: int; Determines the evaluation batch size.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image (e.g. 'float32').
    shuffle_seed: int; Seed for shuffling the training data.
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    prefetch_buffer_size: int; Buffer size for the device prefetch.
    dataset_configs: dict; Dataset specific configurations.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """
  dataset_configs = dataset_configs or {}
  del rng
  data_augmentations = dataset_configs.get('data_augmentations', ['default'])
  # TODO(dehghani): add mixup data augmentation.
  for da in data_augmentations:
    if da not in ['default']:
      raise ValueError(f'Data augmentation {data_augmentations} is not '
                       f'(yet) supported in the ImageNet dataset.')
  dtype = getattr(tf, dtype_str)
  onehot_labels = dataset_configs.get('onehot_labels', False)

  logging.info('Loading train split of the ImageNet dataset.')
  train_ds = imagenet_load_split(
      batch_size,
      train=True,
      onehot_labels=onehot_labels,
      dtype=dtype,
      shuffle_seed=shuffle_seed,
      data_augmentations=data_augmentations)

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you '
                       'want to run with dataset service.')
    logging.info('Using the tf.data service at %s', dataset_service_address)
    train_ds = dataset_utils.distribute(train_ds, dataset_service_address)

  logging.info('Loading test split of the ImageNet dataset.')
  eval_ds = imagenet_load_split(eval_batch_size, train=False,
                                onehot_labels=onehot_labels, dtype=dtype)

  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size)
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch, train=False, batch_size=eval_batch_size)
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)

  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(shard_batches, train_iter)
  train_iter = jax_utils.prefetch_to_device(train_iter, prefetch_buffer_size)

  eval_iter = iter(eval_ds)
  eval_iter = map(dataset_utils.tf_to_numpy, eval_iter)
  eval_iter = map(maybe_pad_batches_eval, eval_iter)
  eval_iter = map(shard_batches, eval_iter)
  eval_iter = jax_utils.prefetch_to_device(eval_iter, prefetch_buffer_size)

  input_shape = (-1, IMAGE_SIZE, IMAGE_SIZE, 3)

  meta_data = {
      'num_classes': NUM_CLASSES,
      'input_shape': input_shape,
      'num_train_examples': TRAIN_IMAGES,
      'num_eval_examples': EVAL_IMAGES,
      'input_dtype': getattr(jnp, dtype_str),
      'target_is_onehot': onehot_labels,
  }
  return dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)
