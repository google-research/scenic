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

"""Data generators for the Cityscapes dataset variants.


Supported datasets, set by dataset_configs.dataset_name in the config file:

cityscapes_corrupted: https://arxiv.org/pdf/1907.07484.pdf
fishyscapes: https://link.springer.com/article/10.1007/s11263-021-01511-6

Implementation details:
cityscapes_c: https://github.com/ekellbuch/cityscapes-c
"""

import functools
from typing import Optional

from absl import logging
from flax import jax_utils
import jax.numpy as jnp
from scenic.dataset_lib import cityscapes_dataset
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import tensorflow as tf
import tensorflow_datasets as tfds


CITYSCAPES_C_CORRUPTIONS = [
    'gaussian_noise',
]

FISHYSCAPES_CORRUPTIONS = [
    'Static',
]

CITYSCAPES_C_SEVERITIES = range(1, 6)

DATASET_INFO = {
    'cityscapes': {
        'tfds_name': 'cityscapes',
        'split': 'validation',
        'num_of_examples': 500,
    },
    'cityscapes_corrupted': {
        'tfds_name': 'internal',
        'split': 'validation',
        'num_of_examples': 500,
    },
    'fishycapes': {
        'tfds_name': 'internal',
        'split': 'validation',
        'num_of_examples': 30,
    },
}

# Adds cityscapes_c
for severity in CITYSCAPES_C_SEVERITIES:
  for corruption in CITYSCAPES_C_CORRUPTIONS:
    temp_dataset_name = f'cityscapes_corrupted/semantic_segmentation_{corruption}_{severity}'
    DATASET_INFO[temp_dataset_name] = {
        'tfds_name': temp_dataset_name,
        'split': 'validation',
        'num_of_examples': 500,
    }

# Adds fishyscapes
for corruption in FISHYSCAPES_CORRUPTIONS:
  temp_dataset_name = f'fishyscapes/{corruption}'
  DATASET_INFO[temp_dataset_name] = {
      'tfds_name': temp_dataset_name,
      'split': 'validation',
      'num_of_examples': 30,
  }

cityscapes_meta_data = {
    'num_classes':
        len([c.id for c in cityscapes_dataset.CLASSES if not c.ignore_in_eval]),
    'class_names':
        cityscapes_dataset.get_class_names(),
    'class_colors':
        cityscapes_dataset.get_class_colors(),
    'class_proportions':
        cityscapes_dataset.get_class_proportions(),
}

fishyscapes_meta_data = {
    'num_classes': 2,
    'class_names': ['ind', 'ood'],
    'class_colors': [(0, 0, 1), (1, 0, 0)],
}


def normalize(image, dtype=tf.float32):
  """Normalizes the value of pixels in the given image.

  Args:
    image: `Tensor` representing an image binary of arbitrary size.
    dtype: Tensorflow data type, Data type of the image.

  Returns:
    A normalized image `Tensor`.
  """
  image = tf.cast(image, dtype=dtype)
  if dtype not in [tf.int32, tf.int64, tf.uint32, tf.uint64]:
    image /= tf.constant(255.0, shape=[1, 1, 1], dtype=dtype)
  return image


def preprocess_example_fishyscapes(example,
                                   train,
                                   dtype=tf.float32,
                                   resize=None,
                                   include_mask=True):
  """Preprocesses the given image.

  Args:
    example: dict; Example coming from TFDS.
    train: bool; Whether to apply training-specific preprocessing or not.
    dtype: Tensorflow data type; Data type of the image.
    resize: sequence; [H, W] to which image and labels should be resized.
    include_mask: include batch_mask to ignore specific classes.
  Returns:
    An example dict as required by the model.
  """
  image = normalize(example['image_left'], dtype)
  mask = example['mask']

  # Resize test images (train images are cropped/resized during augmentation):
  if not train:
    if resize is not None:
      image = tf.image.resize(image, resize, 'bilinear')
      mask = tf.image.resize(mask, resize, 'nearest')

  image = tf.cast(image, dtype)
  mask = tf.cast(mask, dtype)
  mask = tf.squeeze(mask, axis=2)

  outputs = {'inputs': image, 'label': mask}
  if include_mask:
    # Fishyscapes mask has values 0,1, 255, background pixels are set as 255.
    # create batch_mask array and set background pixels to 0 and
    # pixels that should be included during eval to 1
    batch_mask = tf.ones_like(mask, dtype)
    batch_mask = tf.cast(batch_mask*(1-tf.cast(mask == 255, dtype)), dtype)
    # update the mask array to be 0 or  by setting cls 255 to cls 0.
    mask = tf.cast(mask*(1-tf.cast(mask == 255, dtype)), dtype)

    outputs = {'inputs': image, 'label': mask, 'batch_mask': batch_mask}

  return outputs


preprocess_examples = {
    'cityscapes': cityscapes_dataset.preprocess_example,
    'fishyscapes': preprocess_example_fishyscapes,
}


def cityscapes_load_split(
    dataset_name,
    batch_size,
    train=False,
    dtype=tf.float32,
    shuffle_buffer_size=10,
    shuffle_seed=None,
    data_augmentations=None,
    preprocess_ex_eval=None,
    cache=True,
    data_dir: Optional[str] = None,
):
  """Creates a split from the Cityscapes dataset using TensorFlow Datasets.

  For the training set, we drop the last partial batch. This is fine to do
  because we additionally shuffle the data randomly each epoch, thus the trainer
  will see all data in expectation. For the validation set, we pad the final
  batch to the desired batch size.

  Args:
    dataset_name: string; Dataset name defined in DATASET_INFO.
    batch_size: int; The batch size returned by the data pipeline.
    train: bool; Whether to load the train or evaluation split.
    dtype: TF data type; Data type of the image.
    shuffle_buffer_size: int; Buffer size for the TFDS prefetch.
    shuffle_seed: The seed to use when shuffling the train split.
    data_augmentations: list(str); Types of data augmentation applied on
    preprocess_ex_eval: preprocessing function. Default None.
    cache: bool; Whether to cache dataset in memory.
    data_dir: directory with data.

  Returns:
    A `tf.data.Dataset`.
  """
  assert not train, 'Only evaluation is supported.'
  assert dataset_name in DATASET_INFO
  del data_augmentations
  cityscapes_variant_info = DATASET_INFO.get(dataset_name, {})
  split = cityscapes_variant_info['split']  # only supports validation

  # Load the preprocessing function
  if 'cityscapes' in cityscapes_variant_info.get('tfds_name'):
    if dataset_name == 'cityscapes':
      builder = tfds.builder(dataset_name, dtype=dtype)
    elif 'cityscapes_corrupted' in dataset_name:
      if data_dir is None:
        # pylint: disable=line-too-long
        data_dir = 'gs://ub-ekb/tensorflow_datasets/cityscapes_corrupted/tfrecords/v.0.0'  # pylint: disable=line-too-long
        # pylint: enable=line-too-long
      builder = tfds.builder(dataset_name, data_dir=data_dir)
  elif 'fishyscapes' in cityscapes_variant_info.get('tfds_name'):
    if data_dir is None:
      data_dir = 'gs://ub-ekb/tensorflow_datasets/fishyscapes/tfrecords/v.0.0'
    builder = tfds.builder(dataset_name, data_dir=data_dir)
  else:
    raise NotImplementedError(f'{dataset_name} not available')

  ds, ds_info = dataset_utils.load_split_from_tfds_builder(
      builder=builder,
      batch_size=batch_size,
      split=split,
      preprocess_example=preprocess_ex_eval,
      shuffle_buffer_size=shuffle_buffer_size,
      shuffle_seed=shuffle_seed,
      cache=cache)
  return ds, ds_info


def _check_dataset_exists(dataset_configs):
  assert 'dataset_name' in dataset_configs, ('Must specify dataset_name in '
                                             'dataset_configs.')
  dataset_name = dataset_configs['dataset_name']
  assert dataset_configs[
      'dataset_name'] in DATASET_INFO, f'{dataset_name} is not supported.'
  return dataset_name


@datasets.add_dataset('cityscapes_variants')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                prefetch_buffer_size=2,
                rng=None,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None):
  """Returns generators for the Cityscapes validation, and test set.

  Args:
    batch_size: int; Determines the train batch size.
    eval_batch_size: int; Determines the evaluation batch size.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image (e.g. 'float32').
    shuffle_seed: int; Seed for shuffling the training data.
    prefetch_buffer_size: int; Buffer size for the TFDS prefetch.
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    dataset_configs: dict; Dataset specific configurations.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """
  del batch_size
  del shuffle_seed, rng
  del dataset_service_address

  dtype = getattr(tf, dtype_str)
  dataset_configs = dataset_configs or {}
  dataset_name = _check_dataset_exists(dataset_configs)
  cityscapes_variant_info = DATASET_INFO.get(dataset_name)
  target_size = dataset_configs.get('target_size', None)

  if 'cityscapes' in dataset_name:
    preprocess_example = preprocess_examples['cityscapes']
  elif 'fishyscapes' in dataset_name:
    preprocess_example = preprocess_examples['fishyscapes']

  preprocess_ex_eval = functools.partial(
      preprocess_example, train=False, dtype=dtype, resize=target_size)

  logging.info('Loading validation split of the %s dataset.', dataset_name)

  eval_ds, _ = cityscapes_load_split(
      dataset_name=dataset_name,
      batch_size=eval_batch_size,
      train=False,
      dtype=dtype,
      preprocess_ex_eval=preprocess_ex_eval)

  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch,
      train=False,
      batch_size=eval_batch_size,
      pixel_level=True)
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
  exclude_classes = functools.partial(
      cityscapes_dataset.exclude_bad_classes,
      new_labels=cityscapes_dataset.get_post_exclusion_labels())

  eval_iter = iter(eval_ds)
  eval_iter = map(dataset_utils.tf_to_numpy, eval_iter)
  eval_iter = map(maybe_pad_batches_eval, eval_iter)

  if 'cityscapes' in dataset_name:
    eval_iter = map(exclude_classes, eval_iter)
  eval_iter = map(shard_batches, eval_iter)
  eval_iter = jax_utils.prefetch_to_device(eval_iter, prefetch_buffer_size)

  if target_size is None:
    input_shape = (-1, 1024, 2048, 3)
  else:
    input_shape = (-1,) + tuple(target_size) + (3,)

  meta_data = {
      'input_shape': input_shape,
      'num_train_examples': 0,
      'num_eval_examples': cityscapes_variant_info['num_of_examples'],
      'input_dtype': getattr(jnp, dtype_str),
      'target_is_onehot': False,
  }

  if 'cityscapes' in dataset_name:
    meta_data.update(cityscapes_meta_data)
  elif 'fishyscapes' in dataset_name:
    meta_data.update(fishyscapes_meta_data)

  return dataset_utils.Dataset(None, eval_iter, None, meta_data)
