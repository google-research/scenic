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

"""Data generators for ADE20k datasets."""

import functools
from typing import Dict, List, Optional, Tuple

from absl import logging
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.projects.robust_segvit.datasets import datasets_info
from scenic.projects.robust_segvit.datasets import denoise_utils
import tensorflow as tf


def preprocess_example(
    example: Dict[str, tf.Tensor],
    train: bool,
    dataset_configs: ml_collections.ConfigDict,
    dataset_info: datasets_info.DatasetInfo,
    dtype: tf.DType = tf.float32,
    resize: Optional[List[int]] = None,
    rng: int = 0):
  """Preprocesses the given image.

  Args:
    example: Example coming from TFDS.
    train: Whether to apply training-specific preprocessing or not.
    dataset_configs: Dataset configurations.
    dataset_info: Dataset specific information.
    dtype: Data type of the image.
    resize: Height and width to which image and labels should be resized.
    rng: Seed for sampling the noise for denoising.

  Returns:
    An example dict as required by the model.
  """
  image = dataset_utils.normalize(example[dataset_info.image_key], dtype)
  mask = example[dataset_info.label_key]

  # Resize test images (train images are cropped/resized during augmentation):
  if not train:
    if resize is not None:
      image = tf.image.resize(image, resize, 'bilinear')
      mask = tf.image.resize(mask, resize, 'nearest')

  # adding noise for training images is applied during augmentation
  image = tf.cast(image, dtype)
  if dataset_configs.denoise and not train:
    noised_image, noise, timestep, gamma, patch = denoise_utils.add_noise(
        image, rng, dataset_configs.denoise, dtype)
    example = {
        'inputs': noised_image,
        'label': noise,
        'image': image,
        'timestep': timestep,
        'gamma': gamma,
        'patch': patch
    }
  else:
    mask = tf.cast(mask, dtype)
    mask = tf.squeeze(mask, axis=2)
    timestep = tf.constant(dataset_configs.use_timestep or 0)
    example = {'inputs': image, 'label': mask, 'timestep': timestep}

  return example


def augment_example(example: Dict[str, tf.Tensor],
                    dataset_configs: ml_collections.ConfigDict,
                    dtype: tf.DType = tf.float32,
                    resize: Optional[List[int]] = None,
                    rng: int = 0,
                    **inception_crop_kws):
  """Augments the given train image.

  Args:
    example: Example coming from TFDS.
    dataset_configs: Dataset configurations.
    dtype: Data type of the image.
    resize: Height and width to which image and labels should be resized.
    rng: Seed for sampling the noise for denoising.
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
  if dataset_configs.denoise:
    noised_image, noise, timestep, gamma, patch = denoise_utils.add_noise(
        image, rng, dataset_configs.denoise, dtype)
    example = {
        'inputs': noised_image,
        'label': noise,
        'image': image,
        'timestep': timestep,
        'gamma': gamma,
        'patch': patch
    }
  else:
    mask = tf.cast(mask, dtype)
    mask = tf.squeeze(mask, axis=2)
    timestep = tf.constant(dataset_configs.use_timestep or 0)
    example = {'inputs': image, 'label': mask, 'timestep': timestep}
  return example


def get_post_exclusion_labels(classes):
  """Determines new labels after excluding bad classes.

  Excluded classes get the new label -1.

  Args:
    classes:  List of tuples containing information about each class.

  Returns:
    An array of length num_old_classes, containing new labels.
  """
  old_to_new_labels = np.array(
      [-1 if c.ignore_in_eval else c.train_id for c in classes])
  return old_to_new_labels


def get_class_colors(classes):
  """Returns a [num_classes, 3] array of colors for the model output labels."""
  cm = np.stack([c.color for c in classes if not c.ignore_in_eval], axis=0)
  return cm / 255.0


def get_class_names(classes):
  """Returns a list with the class names of the model output labels."""
  return [c.name for c in classes if not c.ignore_in_eval]


def get_class_proportions(classes, pixels_per_cid):
  """Returns a [num_classes] array of pixel frequency proportions."""
  p = [pixels_per_cid[c.id] for c in classes if not c.ignore_in_eval]
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


@datasets.add_dataset('robust_segvit_segmentation')
def get_dataset(
    *,
    batch_size: int,
    eval_batch_size: int,
    num_shards: int,
    dtype_str: str = 'float32',
    shuffle_seed: int = 0,
    rng: Optional[Tuple[int, int]] = None,
    dataset_configs: ml_collections.ConfigDict = ml_collections.ConfigDict(),
    dataset_service_address: Optional[str] = None) -> dataset_utils.Dataset:
  """Returns generators for train and validation splits of the specified dataset.

  Args:
    batch_size: Determines the train batch size.
    eval_batch_size: Determines the evaluation batch size.
    num_shards: Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image (e.g. 'float32').
    shuffle_seed: Seed for shuffling the training data.
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    dataset_configs: Dataset specific configurations.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """
  dtype = getattr(tf, dtype_str)

  if dataset_configs.name is None:
    raise ValueError('The name of the dataset must be specified')
  if dataset_configs.train_target_size is None:
    raise ValueError('Target size must be specified')

  denoise_configs = dataset_configs.get('denoise')
  dataset_info = datasets_info.get_info(dataset_configs.name)

  logging.info('Loading train split of the %s dataset.', dataset_configs.name)
  preprocess_ex_train = functools.partial(
      preprocess_example,
      train=True,
      dtype=dtype,
      resize=None,
      dataset_configs=dataset_configs,
      dataset_info=dataset_info,
      rng=int(rng[0]))
  augment_ex = functools.partial(
      augment_example,
      dtype=dtype,
      resize=dataset_configs.train_target_size,
      dataset_configs=dataset_configs,
      rng=int(rng[0]),
      area_min=30,
      area_max=100)

  train_split = dataset_configs.get('train_split', 'train')
  num_train_examples = fine_train_size = dataset_utils.get_num_examples(
      dataset_info.tfds_name, split=train_split)
  validation_split = dataset_configs.get('validation_split', 'validation')
  num_eval_examples = dataset_utils.get_num_examples(dataset_info.tfds_name,
                                                     validation_split)

  logging.info('number of examples in %s: %d', train_split, fine_train_size)

  train_ds, _ = dataset_utils.load_split_from_tfds(
      dataset_info.tfds_name,
      batch_size,
      split=train_split,
      preprocess_example=preprocess_ex_train,
      augment_train_example=augment_ex,
      shuffle_seed=shuffle_seed,
      data_dir=dataset_info.data_dir)

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you '
                       'want to run with dataset service.')
    logging.info('Using the tf.data service at %s', dataset_service_address)
    train_ds = dataset_utils.distribute(train_ds, dataset_service_address)

  logging.info('Loading validation split of the %s dataset.',
               dataset_configs.name)
  preprocess_ex_eval = functools.partial(
      preprocess_example,
      train=False,
      dtype=dtype,
      resize=dataset_configs.get('eval_target_size',
                                 dataset_configs.train_target_size),
      dataset_configs=dataset_configs,
      dataset_info=dataset_info,
      rng=int(rng[0]))
  eval_ds, _ = dataset_utils.load_split_from_tfds(
      dataset_info.tfds_name,
      eval_batch_size,
      split=validation_split,
      preprocess_example=preprocess_ex_eval,
      data_dir=dataset_info.data_dir)

  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch, train=True, batch_size=batch_size,
      pixel_level=True)
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch, train=False, batch_size=eval_batch_size,
      pixel_level=True)
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
  # define classes to exclude
  # if ood classes are present use those to remap the classes
  class_to_exclude = dataset_info.ood_classes if dataset_info.ood_classes else dataset_info.classes
  exclude_classes = functools.partial(
      exclude_bad_classes,
      new_labels=get_post_exclusion_labels(class_to_exclude))

  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  if not denoise_configs and dataset_configs.name != 'pascal_voc':
    train_iter = map(exclude_classes, train_iter)
  train_iter = map(shard_batches, train_iter)

  eval_iter = iter(eval_ds)
  eval_iter = map(dataset_utils.tf_to_numpy, eval_iter)
  eval_iter = map(maybe_pad_batches_eval, eval_iter)
  if not denoise_configs and dataset_configs.name != 'pascal_voc':
    eval_iter = map(exclude_classes, eval_iter)
  eval_iter = map(shard_batches, eval_iter)

  input_shape = (-1,) + tuple(dataset_configs.train_target_size) + (3,)

  class_proportions = get_class_proportions(
      dataset_info.classes,
      dataset_info.pixels_per_class) if dataset_info.pixels_per_class else None

  meta_data = {
      'num_classes':
          3 if denoise_configs else len(
              [c.id for c in dataset_info.classes if not c.ignore_in_eval]),
      'input_shape':
          input_shape,
      'num_train_examples':
          num_train_examples,
      'num_eval_examples':
          num_eval_examples,
      'input_dtype':
          getattr(jnp, dtype_str),
      'target_is_onehot':
          False,
      'class_names':
          get_class_names(dataset_info.classes),
      'class_colors':
          get_class_colors(dataset_info.classes),
      'class_proportions':
          class_proportions,
  }
  return dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)
