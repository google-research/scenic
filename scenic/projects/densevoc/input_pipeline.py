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

"""Data generators for the Dense VOC tasks."""

import functools
from typing import Optional

from absl import logging
from dmvr import tokenizers
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils
from scenic.projects.baselines.centernet import input_pipeline as centernet_input_pipeline
from scenic.projects.densevoc import input_utils
import tensorflow as tf


PRNGKey = jnp.ndarray


VG_NUM_TRAIN_IMAGES = 77396
VG_NUM_TEST_IMAGES = 5000


def vg_decode_example(data):
  """Convert custom tfrecord into tfds builder format."""
  example = {}
  example['image'] = tf.io.decode_jpeg(data['image'])
  example['image/id'] = data['img_id']
  example['objects'] = {}
  example['objects']['bbox'] = tf.reshape(
      tf.sparse.to_dense(data['regions/bbox']), [-1, 4])
  example['objects']['phrase'] = tf.sparse.to_dense(data['regions/phrase'])
  example['objects']['id'] = tf.sparse.to_dense(data['regions/id'])
  return example


def decode_dense_caption_example(
    example,
    tokenizer,
    max_boxes=100,
    max_text_tokens=40,
    ):
  """Given an instance and raw labels, creates <inputs, label> pair.

  Args:
    example: dict; Input image and raw labels.
    tokenizer: tokenizer that convert string tensor to int tensors.
    max_boxes: int; max number of objects to load.
    max_text_tokens: int; max number of tokens per text.
  Returns:
    A dictionary of {'inputs': input image, 'labels': task label}.
  """
  image = tf.cast(example['image'], tf.float32)
  boxes = centernet_input_pipeline.decode_boxes(
      example['objects']['bbox'], tf.shape(image)[0:2])
  target = {
      'boxes': boxes,
      'text_tokens': tokenizer.string_tensor_to_indices(
          example['objects']['phrase'],
          prepend_bos=True, append_eos=True, max_num_tokens=max_text_tokens),
      'labels': tf.zeros((tf.shape(boxes)[0],), dtype=tf.int32),
  }
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


def load_split_from_tfds(
    batch_size,
    *,
    train,
    preprocess_fn,
    decode_fn,
    dataset_path,
    cache=False,
    max_size=1024,
    max_boxes=100,
    max_text_tokens=40,
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
    max_text_tokens: int; max number of text tokens.
    shuffle_buffer_size: int; Size of the shuffle buffer.
    shuffle_seed: int; Seed for shuffling the training data.

  Returns:
    A `tf.data.Dataset`, and dataset info.
  """
  ds = tf.data.TFRecordDataset(
      centernet_input_pipeline.decode_sharded_names(dataset_path))
  # Split datasets into machines. Otherwise multi-machine evaluation takes the
  # same images.
  ds = ds.shard(jax.process_count(), jax.process_index())
  ds = ds.map(
      lambda x: tf.io.parse_single_example(
          x, input_utils.vg_feature_description))
  ds = ds.map(vg_decode_example)
  ds_info = {}

  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)
  ds = ds.map(
      decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if cache:
    ds = ds.cache()

  padded_shapes = {
      'inputs': [max_size, max_size, 3],
      'padding_mask': [max_size, max_size],
      'label': {
          'boxes': [max_boxes, 4],
          'text_tokens': [max_boxes, max_text_tokens],
          'labels': [max_boxes],
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

  multi_dataset_training = dataset_configs.get('multi_dataset_training', False)
  is_video_dataset = dataset_configs.get('is_video_dataset', False)
  is_video_dataset_train = dataset_configs.get(
      'is_video_dataset_train', is_video_dataset)
  is_video_dataset_test = dataset_configs.get(
      'is_video_dataset_test', is_video_dataset)

  scale_range = dataset_configs.get('scale_range', (0.1, 2.0))
  crop_size = dataset_configs.get('crop_size', 1024)
  max_boxes = dataset_configs.get('max_boxes', 100)
  max_text_tokens = dataset_configs.get('max_text_tokens', 40)
  size_divisibility = dataset_configs.get('size_divisibility', 1)
  train_data_path = dataset_configs['train_data_path']
  test_data_path = dataset_configs['test_data_path']
  tokenizer_weight_path = dataset_configs['tokenizer_weight_path']
  crop_size = ((crop_size - 1) // size_divisibility + 1) * size_divisibility

  if multi_dataset_training:
    assert is_video_dataset_train, 'multidataset training only supported video.'
    assert isinstance(train_data_path, tuple), train_data_path

  train_preprocess_fn = input_utils.make_resize_crop_transforms(
      'train', scale_range=scale_range, crop_size=crop_size)
  eval_preprocess_fn = input_utils.make_resize_crop_transforms(
      'validation', scale_range=(1.0, 1.0), crop_size=crop_size)

  tokenizer = tokenizers.BertTokenizer(tokenizer_weight_path)
  tokenizer.initialize()
  decode_fn = functools.partial(
      decode_dense_caption_example,
      tokenizer=tokenizer,
      max_boxes=max_boxes,
      max_text_tokens=max_text_tokens)

  if is_video_dataset_train:
    if multi_dataset_training:
      train_ds = []
      dataset_format = dataset_configs['dataset_format']
      assert len(dataset_format) == len(train_data_path)
      for i, train_data_path_i in enumerate(train_data_path):
        train_ds_i, _ = input_utils.load_video_train_tfds(
            batch_size,
            dataset_path=train_data_path_i,
            shuffle_buffer_size=dataset_configs.get(
                'shuffle_buffer_size', 1000),
            max_size=crop_size,
            max_boxes=max_boxes,
            max_text_tokens=max_text_tokens,
            shuffle_seed=shuffle_seed,
            tokenizer=tokenizer,
            max_frames=dataset_configs.get(
                'max_frames_train', dataset_configs.get('max_frames', 8)),
            temporal_stride=dataset_configs.get('temporal_stride', 1),
            ensure_sample_has_objects=dataset_configs.get(
                'ensure_sample_has_objects', True),
            max_video_captions=dataset_configs.get(
                'max_video_captions', max_boxes),
            scale_range=scale_range,
            dataset_format=dataset_format[i],
            track_id_key=dataset_configs.get(
                'track_id_key', 'objects/track_id'),
        )
        train_ds.append(train_ds_i)
      train_ds = tf.data.Dataset.sample_from_datasets(
          train_ds, dataset_configs['dataset_sample_weights'])
    else:
      train_ds, _ = input_utils.load_video_train_tfds(
          batch_size,
          dataset_path=train_data_path,
          shuffle_buffer_size=dataset_configs.get('shuffle_buffer_size', 1000),
          max_size=crop_size,
          max_boxes=max_boxes,
          max_text_tokens=max_text_tokens,
          shuffle_seed=shuffle_seed,
          tokenizer=tokenizer,
          max_frames=dataset_configs.get(
              'max_frames_train', dataset_configs.get('max_frames', 8)),
          temporal_stride=dataset_configs.get('temporal_stride', 1),
          ensure_sample_has_objects=dataset_configs.get(
              'ensure_sample_has_objects', True),
          max_video_captions=dataset_configs.get(
              'max_video_captions', max_boxes),
          dataset_format=dataset_configs.get('dataset_format', 'full'),
          track_id_key=dataset_configs.get('track_id_key', 'objects/track_id'),
      )
  else:
    train_ds, _ = load_split_from_tfds(
        batch_size, train=True,
        preprocess_fn=train_preprocess_fn,
        decode_fn=decode_fn,
        dataset_path=train_data_path,
        shuffle_buffer_size=dataset_configs.get('shuffle_buffer_size', 1000),
        max_size=crop_size,
        max_boxes=max_boxes,
        max_text_tokens=max_text_tokens,
        shuffle_seed=shuffle_seed)

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you '
                       'want to run with dataset service.')
    logging.info('Using the tf.data service at %s', dataset_service_address)
    train_ds = dataset_utils.distribute(train_ds, dataset_service_address)

  if is_video_dataset_test:
    eval_ds, _ = input_utils.load_video_val_tfds(
        eval_batch_size,
        max_size=crop_size,
        max_boxes=max_boxes,
        dataset_path=test_data_path,
        tokenizer=tokenizer,
        max_frames=dataset_configs.get(
            'max_frames_test', dataset_configs.get('max_frames', 200)),
        with_objects=dataset_configs.get('with_objects', True),
        temporal_stride=dataset_configs.get('test_temporal_stride', 1),
        dataset_format=dataset_configs.get('eval_dataset_format', 'full'),
    )
  else:
    eval_ds, _ = load_split_from_tfds(
        eval_batch_size,
        train=False,
        preprocess_fn=eval_preprocess_fn,
        max_size=crop_size,
        max_boxes=max_boxes,
        decode_fn=decode_fn,
        dataset_path=test_data_path,
    )

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

  num_classes = dataset_configs.get('num_classes', 1)
  num_train_examples = dataset_configs.get(
      'num_train_examples', VG_NUM_TRAIN_IMAGES)
  num_eval_examples = dataset_configs.get(
      'num_eval_examples', VG_NUM_TEST_IMAGES)

  meta_data = {
      'num_classes': num_classes,
      'input_shape': [-1, crop_size, crop_size, 3],
      'num_train_examples': num_train_examples,
      'num_eval_examples': num_eval_examples,
      'input_dtype': jnp.float32,
      'target_is_onehot': False,
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
