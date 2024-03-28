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

"""Data generators for Segmentation variants dataset.

The datasets include:
ADE20K_C
"""

import functools
from typing import Dict, List, Optional, Tuple

from absl import logging
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.projects.robust_segvit.datasets import datasets_info
from scenic.projects.robust_segvit.datasets import denoise_utils
from scenic.projects.robust_segvit.datasets.segmentation_datasets import exclude_bad_classes
from scenic.projects.robust_segvit.datasets.segmentation_datasets import get_class_colors
from scenic.projects.robust_segvit.datasets.segmentation_datasets import get_class_names
from scenic.projects.robust_segvit.datasets.segmentation_datasets import get_class_proportions
from scenic.projects.robust_segvit.datasets.segmentation_datasets import get_post_exclusion_labels
import tensorflow as tf


def unique_with_inverse(x):
  x = tf.reshape(x, [-1])
  y, idx, _ = tf.unique_with_counts(x)
  return tf.gather(y, idx)


def get_instance_mask(instance_segmentation):
  """Obtain the instance mask from the blue channel of the segmentation file."""
  # Based on DevKit:
  # https://github.com/CSAILVision/ADE20K/blob/main/utils/utils_ade20k.py
  instance_segmentation_blue = instance_segmentation[:, :, 2]
  instance_mask = unique_with_inverse(instance_segmentation_blue)
  instance_mask = tf.expand_dims(
      tf.reshape(instance_mask, tf.shape(instance_segmentation_blue)), -1)
  return tf.cast(instance_mask, tf.uint16)


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

  # preprocess mask following:
  # https://github.com/CSAILVision/ADE20K/blob/main/utils/utils_ade20k.py

  if mask.shape[-1] == 3:
    mask = get_instance_mask(mask)

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


@datasets.add_dataset('robust_segvit_variants')
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
  del batch_size, shuffle_seed, dataset_service_address

  dtype = getattr(tf, dtype_str)

  if dataset_configs.name is None:
    raise ValueError('The name of the dataset must be specified')
  if dataset_configs.train_target_size is None:
    raise ValueError('Target size must be specified')

  denoise_configs = dataset_configs.get('denoise')
  dataset_info = datasets_info.get_info(dataset_configs.name)
  validation_split = dataset_configs.get('validation_split', 'validation')
  num_eval_examples = dataset_utils.get_num_examples(
      dataset=dataset_info.tfds_name,
      split=validation_split,
      data_dir=dataset_info.data_dir)

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

  # TODO(kellybuchanan): merge cityscapes_c and ade20k_c
  eval_ds, _ = dataset_utils.load_split_from_tfds(
      dataset_info.tfds_name,
      eval_batch_size,
      split=validation_split,
      preprocess_example=preprocess_ex_eval,
      data_dir=dataset_info.data_dir)

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
          0,
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
  return dataset_utils.Dataset(None, eval_iter, None, meta_data)
