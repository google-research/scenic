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

"""Defines dataset made of pickled objects."""

import pickle
import jax.numpy as jnp
import tensorflow as tf
from tensorflow.io import gfile


def normalize_input(sample):
  """Normalize input.

  Args:
    sample: The input data.

  Returns:
    The normalized input data.
  """
  sample['boundaries'] = tf.cast(sample['boundaries'], tf.float32) / 255.0

  # Threshold noisy image:
  sample['image'] = tf.where(
      sample['image'] > 1.0, tf.ones_like(sample['image']), sample['image']
  )
  sample['image'] = tf.where(
      sample['image'] < 0.0, tf.zeros_like(sample['image']), sample['image']
  )

  return sample


def get_dataset_and_count(
    dataset_config,
    batch_size=1,
    eval_batch_size=1,
    split_train_eval=True,
    shuffle_buffer_size=1,  # pylint: disable=unused-argument
    dataset_shuffle=0,  # pylint: disable=unused-argument
    prebatch=True,  # pylint: disable=unused-argument
    readahead_buffer=256,  # pylint: disable=unused-argument
    dtype_str='float32',
):
  """Get dataset and count.

  Args:
    dataset_config: A config object that contains the paths and num_samples_use.
    batch_size: The batch size.
    eval_batch_size: The eval batch size.
    split_train_eval: If True, split the dataset into train and eval.
    shuffle_buffer_size: The shuffle buffer size.
    dataset_shuffle: The shuffle buffer size.
    prebatch: If True, prebatch the dataset.
    readahead_buffer: The readahead buffer size.
    dtype_str: The dtype string.

  Returns:
    A tuple of datasets, num_images, metadata.
  """

  file_paths = dataset_config.paths
  num_samples_use = dataset_config.num_samples_use

  images = []
  clean_images = []
  distances = []
  boundaries = []
  num_images = 0

  for ii, file_name in enumerate(file_paths):

    with gfile.Gfile(file_name, 'rb') as f:
      data = pickle.load(f)

    images.append(
        data['images'].squeeze().transpose(0, 3, 1, 2)[: num_samples_use[ii]]
    )
    clean_images.append(
        data['clean_images']
        .squeeze()
        .transpose(0, 3, 1, 2)[: num_samples_use[ii]]
    )
    distances.append(
        jnp.expand_dims(data['dist_boundaries'][: num_samples_use[ii]], 1)
    )
    boundaries.append(
        jnp.expand_dims(data['boundaries'][: num_samples_use[ii]], 1)
    )
    num_images = num_images + num_samples_use[ii]

  ######### DEFINE DATA STRUCTURE ########
  data = {
      'image': jnp.concatenate(images, axis=0),
      'clean_image': jnp.concatenate(clean_images, axis=0),
      'distances': jnp.concatenate(distances, axis=0),
      'boundaries': jnp.concatenate(boundaries, axis=0),
  }

  dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(
      num_images, seed=1, reshuffle_each_iteration=False
  )
  dataset = dataset.map(normalize_input)

  if split_train_eval:
    num_train_batches = int(0.9 * num_images) // batch_size
    num_train_images = num_train_batches * batch_size

    num_eval_batches = (num_images - num_train_images) // eval_batch_size
    num_eval_images = num_eval_batches * eval_batch_size

    train_dataset = (
        dataset.take(num_train_images)
        .repeat()
        .batch(batch_size, drop_remainder=True)
    )
    eval_dataset = dataset.skip(num_train_images).batch(
        eval_batch_size, drop_remainder=True
    )

    datasets = (
        train_dataset,
        eval_dataset,
    )
    num_images = (num_train_images, num_eval_images)

  else:
    num_train_batches = int(0.9 * num_images) // batch_size
    num_train_images = num_train_batches * batch_size
    num_eval_images = 0

    datasets = dataset
    num_images = num_train_images

  metadata = {
      'input_dtype': getattr(jnp, dtype_str),
      'num_train_examples': num_train_images,
      'num_eval_examples': num_eval_images,
      'input_shape': [[1, *data['image'][0].shape]],
  }

  return datasets, num_images, metadata
