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

"""Dataset library for KaleidoShapes."""

import functools

import jax.numpy as jnp
from scenic.dataset_lib import dataset_utils
from scenic.projects.boundary_attention.dataset_lib.datasets import kaleidoshapes_dataset_utils
from scenic.projects.boundary_attention.helpers import additive_noise_model
import scenic.projects.boundary_attention.kaleidoshapes.kaleidoshapes  # pylint: disable=unused-import
import tensorflow as tf


def get_input_shape(dataset_config):
  """Returns input shape for the dataset."""

  if dataset_config.get('crop', True):
    input_shape = [[1, 3] +
                   list(dataset_config.get('crop_size', (100, 100, 3))[:2])]
  else:
    input_shape = [[1, 3] +
                   list(dataset_config.get('input_size', (240, 320, 3))[:2])]

  return input_shape


def get_dataset_and_count(dataset_config,
                          batch_size=4,
                          eval_batch_size=4,
                          shuffle_buffer_size=1,
                          dataset_shuffle=0,
                          prebatch=True,
                          readahead_buffer=256,
                          dtype_str='float32'):
  """Returns generators for KaleidoShapes dataset.

  Args:
    dataset_config: ml_collections dict with dataset specific configuration.
    batch_size: int; Batch size per device.
    eval_batch_size: int; Eval batch size.
    shuffle_buffer_size: int; Size of shuffle buffer.
    dataset_shuffle: int; Number of datasets to shuffle.
    prebatch: bool; Whether to prebatch images.
    readahead_buffer: int; Size of readahead buffer.
    dtype_str: str; Data type of the image (e.g. 'float32').

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
    and a dict of metadata.
  """

  del dataset_shuffle, prebatch, readahead_buffer

  num_train_images = dataset_config.num_train_images
  num_eval_images = dataset_config.num_eval_images

  tf.random.set_seed(1234)

  # Define noise model and dataset preprocessing function
  noise_model = additive_noise_model.NoiseModel(
      min_noise_level=dataset_config.get('min_noise_level', 0.3),
      max_noise_level=dataset_config.get('max_noise_level', 0.9),
      normalize=True,
  )
  preprocess_fn = functools.partial(
      kaleidoshapes_dataset_utils.process_kaleido_images,
      dataset_config=dataset_config,
      nmodel=noise_model,
  )

  train_ds = dataset_utils.get_data(
      dataset=dataset_config.get('dataset_name', 'kaleidoshapes'),
      split=dataset_config.get('split', 'train'),
      data_dir=dataset_config.dataset_dir,
      batch_size=batch_size,
      preprocess_fn=preprocess_fn,
      shuffle_buffer_size=shuffle_buffer_size,
      prefetch=dataset_config.get('prefetch_to_host', 0),
      drop_remainder=True,
      cache=False,
      repeats=-1,  # infinite repeats
      ignore_errors=False,
      skip_decode=[],
  )

  eval_ds = dataset_utils.get_data(
      dataset=dataset_config.get('dataset_name', 'kaleidoshapes'),
      split=dataset_config.get('split', 'test'),
      data_dir=dataset_config.dataset_dir,
      batch_size=eval_batch_size,
      preprocess_fn=preprocess_fn,
      cache=False,
      shuffle_files=False,
      repeats=-1,
      drop_remainder=True,
      skip_decode=[],
  )

  input_shape = get_input_shape(dataset_config)

  metadata = {
      'max_num_shapes': dataset_config.max_num_shapes,
      'num_train_examples': num_train_images,
      'num_eval_examples': num_eval_images,
      'input_shape': input_shape,
      'input_dtype': getattr(jnp, dtype_str),
  }

  return train_ds, eval_ds, metadata
