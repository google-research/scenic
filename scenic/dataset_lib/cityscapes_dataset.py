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

"""Data generators for the Cityscapes dataset."""

import collections
import functools
from typing import Optional

from absl import logging
import jax.numpy as jnp
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import tensorflow as tf

# Based on https://github.com/mcordts/cityscapesScripts
CityscapesClass = collections.namedtuple(
    'CityscapesClass',
    ['name', 'id', 'train_id', 'category', 'category_id', 'has_instances',
     'ignore_in_eval', 'color'])

CLASSES = [
    CityscapesClass(
        'unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass(
        'ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass(
        'rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass(
        'out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass(
        'static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass(
        'dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass(
        'ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass(
        'road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass(
        'sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass(
        'parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass(
        'rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass(
        'building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass(
        'wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass(
        'fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass(
        'guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass(
        'bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass(
        'tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass(
        'pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass(
        'polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass(
        'traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass(
        'traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass(
        'vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass(
        'terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass(
        'sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass(
        'person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass(
        'rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass(
        'car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass(
        'truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass(
        'bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass(
        'caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass(
        'trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass(
        'train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass(
        'motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass(
        'bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass(
        'license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

# Number of pixels per Cityscapes class ID in the training set:
PIXELS_PER_CID = {
    7: 3806423808,
    8: 629490880,
    11: 2354443008,
    12: 67089092,
    13: 91210616,
    17: 126753000,
    19: 21555918,
    20: 57031712,
    21: 1647446144,
    22: 119165328,
    23: 415038624,
    24: 126403824,
    25: 13856368,
    26: 725164864,
    27: 27588982,
    28: 24276994,
    31: 24195352,
    32: 10207740,
    33: 42616088
}


def preprocess_example(example, train, dtype=tf.float32, resize=None):
  """Preprocesses the given image.

  Args:
    example: dict; Example coming from TFDS.
    train: bool; Whether to apply training-specific preprocessing or not.
    dtype: Tensorflow data type; Data type of the image.
    resize: sequence; [H, W] to which image and labels should be resized.

  Returns:
    An example dict as required by the model.
  """
  image = dataset_utils.normalize(example['image_left'], dtype)
  mask = example['segmentation_label']

  # Resize test images (train images are cropped/resized during augmentation):
  if not train:
    if resize is not None:
      image = tf.image.resize(image, resize, 'bilinear')
      mask = tf.image.resize(mask, resize, 'nearest')

  image = tf.cast(image, dtype)
  mask = tf.cast(mask, dtype)
  mask = tf.squeeze(mask, axis=2)
  return {'inputs': image, 'label': mask}


def augment_example(
    example, dtype=tf.float32, resize=None, **inception_crop_kws):
  """Augments the given train image.

  Args:
    example: dict; Example coming from TFDS.
    dtype: Tensorflow data type; Data type of the image.
    resize: sequence; [H, W] to which image and labels should be resized.
    **inception_crop_kws: Keyword arguments passed on to
      inception_crop_with_mask.

  Returns:
    An example dict as required by the model.
  """
  image = example['inputs']
  mask = example['label'][..., tf.newaxis]

  # Random crop and resize ("Inception crop"):
  image, mask = dataset_utils.inception_crop_with_mask(
      image,
      mask,
      resize_size=image.shape[-3:-1] if resize is None else resize,
      **inception_crop_kws)

  # Random LR flip:
  seed = tf.random.uniform(shape=[2], maxval=2**31 - 1, dtype=tf.int32)
  image = tf.image.stateless_random_flip_left_right(image, seed)
  mask = tf.image.stateless_random_flip_left_right(mask, seed)

  image = tf.cast(image, dtype)
  mask = tf.cast(mask, dtype)
  mask = tf.squeeze(mask, axis=2)
  return {'inputs': image, 'label': mask}


def get_post_exclusion_labels():
  """Determines new labels after excluding bad classes.

  See Figure 1 in https://arxiv.org/abs/1604.01685 for which classes are
  excluded. Excluded classes get the new label -1.

  Returns:
    An array of length num_old_classes, containing new labels.
  """
  old_to_new_labels = np.array(
      [-1 if c.ignore_in_eval else c.train_id for c in CLASSES])
  assert np.all(np.diff([i for i in old_to_new_labels if i >= 0]) == 1)
  return old_to_new_labels


def get_class_colors():
  """Returns a [num_classes, 3] array of colors for the model output labels."""
  cm = np.stack([c.color for c in CLASSES if not c.ignore_in_eval], axis=0)
  return cm / 255.0


def get_class_names():
  """Returns a list with the class names of the model output labels."""
  return [c.name for c in CLASSES if not c.ignore_in_eval]


def get_class_proportions():
  """Returns a [num_classes] array of pixel frequency proportions."""
  p = [PIXELS_PER_CID[c.id] for c in CLASSES if not c.ignore_in_eval]
  return np.array(p) / np.sum(p)


def exclude_bad_classes(batch, new_labels):
  """Adjusts masks and batch_masks to exclude void and rare classes.

  This must be applied after dataset_utils.maybe_pad_batch() because we also
  update the batch_mask. Note that the data is already converted to Numpy by
  then.

  Args:
    batch: dict; Batch of data examples.
    new_labels: nd-array; array of length num_old_classes, containing new
      labels.

  Returns:
    Updated batch dict.
  """
  # Convert old labels to new labels:
  batch['label'] = new_labels[batch['label'].astype(np.int32)]

  # Set batch_mask to 0 at pixels that have an excluded label:
  mask_dtype = batch['batch_mask'].dtype
  batch['batch_mask'] = (
      batch['batch_mask'].astype(np.bool_) & (batch['label'] != -1))
  batch['batch_mask'] = batch['batch_mask'].astype(mask_dtype)

  return batch


@datasets.add_dataset('cityscapes')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None):
  """Returns generators for the Cityscapes train, validation, and test set.

  Args:
    batch_size: int; Determines the train batch size.
    eval_batch_size: int; Determines the evaluation batch size.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image (e.g. 'float32').
    shuffle_seed: int; Seed for shuffling the training data.
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    dataset_configs: dict; Dataset specific configurations.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """
  del rng
  dtype = getattr(tf, dtype_str)
  dataset_configs = dataset_configs or {}
  target_size = dataset_configs.get('target_size', None)

  logging.info('Loading train split of the Cityscapes dataset.')
  preprocess_ex_train = functools.partial(
      preprocess_example, train=True, dtype=dtype, resize=None)
  augment_ex = functools.partial(
      augment_example, dtype=dtype, resize=target_size, area_min=30,
      area_max=100)

  train_split = dataset_configs.get('train_split', 'train')
  train_ds, _ = dataset_utils.load_split_from_tfds(
      'cityscapes',
      batch_size,
      split=train_split,
      preprocess_example=preprocess_ex_train,
      augment_train_example=augment_ex,
      shuffle_seed=shuffle_seed)

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you '
                       'want to run with dataset service.')
    logging.info('Using the tf.data service at %s', dataset_service_address)
    train_ds = dataset_utils.distribute(train_ds, dataset_service_address)

  logging.info('Loading validation split of the Cityscapes dataset.')
  preprocess_ex_eval = functools.partial(
      preprocess_example, train=False, dtype=dtype, resize=target_size)
  eval_ds, _ = dataset_utils.load_split_from_tfds(
      'cityscapes', eval_batch_size, split='validation',
      preprocess_example=preprocess_ex_eval)

  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size,
      pixel_level=True)
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch, train=False, batch_size=eval_batch_size,
      pixel_level=True)
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
  exclude_classes = functools.partial(
      exclude_bad_classes, new_labels=get_post_exclusion_labels())

  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(exclude_classes, train_iter)
  train_iter = map(shard_batches, train_iter)

  eval_iter = iter(eval_ds)
  eval_iter = map(dataset_utils.tf_to_numpy, eval_iter)
  eval_iter = map(maybe_pad_batches_eval, eval_iter)
  eval_iter = map(exclude_classes, eval_iter)
  eval_iter = map(shard_batches, eval_iter)

  if target_size is None:
    input_shape = (-1, 1024, 2048, 3)
  else:
    input_shape = (-1,) + tuple(target_size) + (3,)

  meta_data = {
      'num_classes':
          len([c.id for c in CLASSES if not c.ignore_in_eval]),
      'input_shape':
          input_shape,
      'num_train_examples':
          dataset_utils.get_num_examples('cityscapes', train_split),
      'num_eval_examples':
          dataset_utils.get_num_examples('cityscapes', 'validation'),
      'input_dtype':
          getattr(jnp, dtype_str),
      'target_is_onehot':
          False,
      'class_names':
          get_class_names(),
      'class_colors':
          get_class_colors(),
      'class_proportions':
          get_class_proportions(),
  }
  return dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)
