# Copyright 2021 The Scenic Authors.
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

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utils for used by different dataset builders."""

import collections
from typing import Any, Dict, Optional

from absl import logging
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


Dataset = collections.namedtuple(
    # Each instance of the Dataset has three iterators, train_iter, valid_iter,
    # and test_iter, that yield a batch, where each batch is a (nested) dict of
    # numpy arrays. These iterators are created by normally applying these
    # functions on TFDS instances:
    #
    #  - dataset_utils.tf_to_numpy -> convert tensors to numpy arrays.
    #  - dataset_utils.maybe_pad_batch -> pad partial batches and create
    #   batch_mask if needed.
    #  - dataset_utils.shard_batches -> shard batch across devices by reshaping
    #   `[bs, ...]` to `[num_local_devices, bs/(num_local_devices), ...]`.
    #
    # Beside these iterators, there is a dictionary that stores the meta data
    # information about the dataset, that can be used for different purposes.
    # For instance, these fields are used in most of the datasets:
    #
    #   'input_shape': Used during compiling and initializing the model.
    #   'num_train_examples': Used for computing the number of training steps
    #      and controlling the train_iter.
    #   'num_eval_examples': Same as num_train_examples, but for valid_iter.
    #   'num_test_examples': Same as num_train_examples, but for tes_iter.
    #   'target_is_onehot': Used in the loss and metric functions.
    #
    # Note that each dataset can define it's own meta-data field that is used
    # in the model and/or the trainer, depending on the task. As an example, for
    # classification tasks, `num_classes` is used for the configuring head of
    # the model.
    'Dataset',
    ['train_iter', 'valid_iter', 'test_iter', 'meta_data'])


def maybe_pad_batch(batch, train, batch_size, pixel_level=False):
  """Zero pad the batch on the right to the batch_size.

  All keys in the batch dictionary will have their corresponding arrays padded.
  This function returns a dictionary with the same keys, additionally with the
  key 'batch_mask' added, with 1.0 indicating indices which are true data and
  0.0 indicating a padded index. `batch_mask` will be used for calculating the
  weighted cross entropy, or weighted accuracy.

  Note that in this codebase, we assume we drop the last partial batch from the
  training set, so if the batch is from the training set (i.e. `train=True`),
  or when the batch is from the test/validation set, but it is a complete batch,
  we *modify* the batch dict by adding an array of ones as the `batch_mask` of
  all examples in the batch. Otherwise, we create a new dict that has the padded
  patch and its corresponding `batch_mask` array.

  Note that batch_mask can be also used as the label mask (not input mask), for
  task that are pixel/token level. This is simply done by applying the mask we
  make for padding the partial batches on top of the existing label mask.

  Args:
    batch: dict; A dictionary mapping keys to arrays. We assume that inputs is
      one of the keys.
    train: bool; if the batch is from the training data. In that case, we drop
      the last (incomplete) batch and thus don't do any padding.
    batch_size: int; All arrays in the dict will be padded to have first
      dimension equal to desired_batch_size.
    pixel_level: bool; If True, this will create a pixel-level (instead of
      example-level) mask, e.g. for segmentation models.

  Returns:
    A dictionary mapping the same keys to the padded batches. Additionally we
    add a key representing weights, to indicate how the batch was padded.
  """
  batch_pad = batch_size - batch['inputs'].shape[0]

  if pixel_level:
    unpadded_mask_shape = batch['inputs'].shape[:-1]
  else:
    assert 'batch_mask' not in batch, (
        'When the labels of the task are not pixel-level, batch_mask should '
        'not be already present in the batch.')
    unpadded_mask_shape = batch['inputs'].shape[0]

  if train and batch_pad != 0:
    raise ValueError('In this codebase, we assumed that we alwayse drop the '
                     'last partial batch of the train set. Please use '
                     '` drop_remainder=True` for the training set.')

  # Most batches will not need padding so we quickly return to avoid slowdown.
  if train or batch_pad == 0:
    if 'batch_mask' not in batch:
      batch['batch_mask'] = np.ones(unpadded_mask_shape, dtype=np.float32)
    return batch

  def zero_pad(array):
    pad_with = [(0, batch_pad)] + [(0, 0)] * (array.ndim - 1)
    return np.pad(array, pad_with, mode='constant')

  padded_batch = jax.tree_map(zero_pad, batch)
  padded_batch_mask = zero_pad(np.ones(unpadded_mask_shape, dtype=np.float32))
  if 'batch_mask' in padded_batch:
    padded_batch['batch_mask'] *= padded_batch_mask
  else:
    padded_batch['batch_mask'] = padded_batch_mask
  return padded_batch


def shard(pytree, n_devices=None):
  """Reshapes all arrays in the pytree to add a leading n_devices dimension.

  Note: We assume that all arrays in the pytree have leading dimension divisble
  by n_devices and reshape (host_batch_size, height, width, channel) to
  (local_devices, device_batch_size, height, width, channel).

  Args:
    pytree: A pytree of arrays to be sharded.
    n_devices: If None, this will be set to jax.local_device_count().

  Returns:
    Sharded data.
  """
  if n_devices is None:
    n_devices = jax.local_device_count()

  def _shard_array(array):
    return array.reshape((n_devices, -1) + array.shape[1:])

  return jax.tree_map(_shard_array, pytree)


def unshard(pytree):
  """Reshapes all arrays in the pytree from [ndev, bs, ...] to [host_bs, ...].

  Args:
    pytree: A pytree of arrays to be sharded.

  Returns:
    Sharded data.
  """

  def _unshard_array(array):
    ndev, bs = array.shape[:2]
    return array.reshape((ndev * bs,) + array.shape[2:])

  return jax.tree_map(_unshard_array, pytree)


def tf_to_numpy(batch):
  """Convert a input batch from tf Tensors to numpy arrays.

  Args:
    batch: dict; A dictionary tha has items in a batch: image and labels.

  Returns:
    Numpy arrays of the given tf Tensors.
  """
  # Use _numpy() for zero-copy conversion between TF and NumPy.
  convert_data = lambda x: x._numpy()  # pylint: disable=protected-access
  return jax.tree_map(convert_data, batch)


def augment_random_crop_flip(image,
                             height=None,
                             width=None,
                             num_channels=None,
                             crop_padding=4,
                             flip=True):
  """Augment small image with random crop and h-flip.

  Args:
    image: Input image to augment.
    height: int; Height of the target image.
    width: int; Width of the target image.
    num_channels: int; Number of channels of the target image.
    crop_padding: int; Random crop range.
    flip: bool; If True perform random horizontal flip.

  Returns:
    Augmented image.
  """
  h, w, c = image.get_shape().as_list()
  height = height or h
  width = width or w
  num_channels = num_channels or c

  assert crop_padding >= 0
  if crop_padding > 0:
    # Pad with reflection padding
    # (See https://arxiv.org/abs/1605.07146)
    # Section 3.
    image = tf.pad(image, [[crop_padding, crop_padding],
                           [crop_padding, crop_padding], [0, 0]], 'REFLECT')

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.image.random_crop(image, [height, width, num_channels])

  if flip:
    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  return image


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


def load_split_from_tfds(dataset_name,
                         batch_size,
                         split,
                         data_dir=None,
                         preprocess_example=None,
                         augment_train_example=None,
                         shuffle_buffer_size=None,
                         shuffle_seed=0,
                         cache=True,
                         **kwargs):
  """Loads a split from a dataset using TensorFlow Datasets.

  Args:
    dataset_name: str; Name of the dataset to be used to load from tfds.
    batch_size: int; The batch size returned by the data pipeline.
    split: str; Name of  the split to be loaded.
    data_dir: srt; Data directory.
    preprocess_example: function; A function that given an example, returns the
      preprocessed example. Note that the preprocessing is done BEFORE caching
      to re-use them.
    augment_train_example: A function that given a train example returns the
      augmented example. Note that this function is applied AFTER caching and
      repeat to get true randomness.
    shuffle_buffer_size: int; Size of the tf.data.dataset shuffle buffer.
    shuffle_seed: int; Seed for shuffling the training data.
    cache: bool; Whether to cache the dataset in memory.
    **kwargs: Passed to tfds.builder().

  Returns:
    A `tf.data.Dataset`, and dataset information.
  """
  return load_split_from_tfds_builder(
      builder=tfds.builder(dataset_name, data_dir=data_dir, **kwargs),
      batch_size=batch_size,
      split=split,
      preprocess_example=preprocess_example,
      augment_train_example=augment_train_example,
      shuffle_buffer_size=shuffle_buffer_size,
      shuffle_seed=shuffle_seed,
      cache=cache)


def load_split_from_tfds_builder(builder,
                                 batch_size,
                                 split,
                                 preprocess_example=None,
                                 augment_train_example=None,
                                 shuffle_buffer_size=None,
                                 shuffle_seed=0,
                                 cache=True):
  """Loads a split from a dataset using TensorFlow Datasets compatible builder.

  Args:
    builder: tfds.core.DatasetBuilder; A TFDS compatible dataset builder.
    batch_size: int; The batch size returned by the data pipeline.
    split: str; Name of  the split to be loaded.
    preprocess_example: function; A function that given an example, returns the
      preprocessed example. Note that the preprocessing is done BEFORE caching
      to re-use them.
    augment_train_example: A function that given a train example returns the
      augmented example. Note that this function is applied AFTER caching and
      repeat to get true randomness.
    shuffle_buffer_size: int; Size of the tf.data.dataset shuffle buffer.
    shuffle_seed: int; Seed for shuffling the training data.
    cache: bool; Whether to cache dataset in memory.

  Returns:
    A `tf.data.Dataset`, and dataset information.
  """
  # Prepare map functions.
  preprocess_example = preprocess_example or (lambda ex: ex)
  augment_train_example = augment_train_example or (lambda ex: ex)
  shuffle_buffer_size = shuffle_buffer_size  or (8 * batch_size)

  # Each host is responsible for a fixed subset of data/
  base_split_name, host_start, host_end = get_data_range(
      builder, split, jax.process_index(), jax.process_count())
  data_range = tfds.core.ReadInstruction(
      base_split_name, unit='abs', from_=host_start, to=host_end)
  ds = builder.as_dataset(split=data_range, shuffle_files=False)
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  # Applying preprocessing before `ds.cache()` to re-use it.
  ds = ds.map(
      preprocess_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # Caching.
  if cache:
    ds = ds.cache()

  if 'train' in split:
    # First repeat then batch.
    ds = ds.repeat()
    # Augmentation should be done after repeat for true randomness.
    ds = ds.map(
        augment_train_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Shuffle after augmentation to avoid loading uncropped images into buffer:
    ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    ds = ds.batch(batch_size, drop_remainder=True)

  else:
    # First batch then repeat.
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.repeat()

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds, builder.info


def target_to_one_hot(batch, num_classes):
  """Converts the labels to one-hot targets.

  Args:
    batch: dict; A batch of data with 'inputs' and 'label'.
    num_classes: int; Number of classes.

  Returns:
    Batch with one-hot labels.
  """
  return {
      'inputs': batch['inputs'],
      'label': common_utils.onehot(batch['label'], num_classes)
  }


def mixup(batch: Dict['str', jnp.ndarray],
          alpha: float = 1.0,
          image_format: str = 'NHWC',
          rng: Optional[Any] = None) -> Dict['str', jnp.ndarray]:
  """Mixes images and labels within a single batch.

  For more details, please see https://arxiv.org/abs/1710.09412.

  This function supports both using `numpy` to do mixup in the input-pipeline
  and `jax.numpy` to do mixup within a jitted/pmapped function (e.g. within
  a pmapped train step to apply mixup on device patch).

  Results in a batch with:
    mixed_images[idx] = weight * images[idx] + (1-weight) * images[-(idx+1)],
    where weight is sampled from a beta distribution with parameter alpha.

  Args:
    batch: dict; A batch of data with 'inputs' and 'label'.
    alpha: float; Used to control the beta distribution that weight is sampled
      from.
    image_format: string; The format of the input images.
    rng: JAX rng key. If given, JAX numpy will be used as the backend, and if
      None (default value), normal numpy will be used.

  Returns:
    Tuple (mixed_images, mixed_labels).
  """
  images, labels = batch['inputs'], batch['label']
  if labels.shape[-1] == 1:
    raise ValueError('Mixup requires one-hot targets.')
  if 'N' not in image_format:
    raise ValueError('Mixup requires "N" to be in "image_format".')

  batch_size = labels.shape[0]

  # Setup the the numpy backend and prepare mixup weights.
  if rng is None:
    np_backend = np  # Ordinary numpy
    weight = np_backend.random.beta(alpha, alpha)
  else:
    np_backend = jnp  # JAX numpy
    weight = jax.random.beta(rng, alpha, alpha)
  weight *= np_backend.ones((batch_size, 1))

  # Mixup labels.
  batch['label'] = weight * labels + (1.0 - weight) * labels[::-1]

  # Mixup inputs.
  # Shape calculations use np to avoid device memory fragmentation:
  image_weight_shape = np.ones((images.ndim))
  image_weight_shape[image_format.index('N')] = batch_size
  weight = np_backend.reshape(weight,
                              image_weight_shape.astype(np_backend.int32))
  reverse = tuple(
      slice(images.shape[i]) if d != 'N' else slice(-1, None, -1)
      for i, d in enumerate(image_format))
  batch['inputs'] = weight * images + (1.0 - weight) * images[reverse]

  return batch


def _shard_read_instruction(host_id, host_count, abs_ri, builder):
  """Returns this host's shard of the read instruction `abs_ri`."""
  # Get its start/end indices.
  full_start = abs_ri.from_ or 0
  full_end = abs_ri.to or builder.info.splits[abs_ri.splitname].num_examples

  # Compute each host's subset.
  # All hosts should perfrom exactly the same number iterations. To make sure
  # this is the case we ensure that all hosts get the same number of images.
  # some setups, do padding instead, at least for the validation set.
  examples_per_host = (full_end - full_start) // host_count
  host_start = full_start + examples_per_host * host_id
  host_end = full_start + examples_per_host * (host_id + 1)

  # Return raw info instead of ReadInstruction object due to API funkiness.
  return abs_ri.splitname, host_start, host_end


def _get_data_range(builder, split, host_id, host_count):
  """Return a (sub)split adapted to a given host.

  Args:
    builder: TFDS Builder for the datset.
    split: str; Split for which we want to create the data ranage.
    host_id: int; Id of the host.
    host_count: int; Total Number of hosts.

  Returns:
    Data range for the current host and the given split.
  """
  # Each host reads a (same-size) subset of the given `split` such that they
  # all go through different parts of it. For doing so, we need to do some
  # indexing computation on a given `builder` and `split`, because we want to
  # support using contiguous subsets such as `train[10%:20%]`, mainly because
  # imagenet21k only contains a single `full` split and JFT only has huge val
  # sets that would take forever.

  # Canonicalize read instruction `ri` to absolute one.
  name2len = {k: v.num_examples for k, v in builder.info.splits.items()}
  abs_ri = tfds.core.ReadInstruction.from_spec(split).to_absolute(name2len)

  # Shard each read instruction individually. Usually there's only one, but
  # when doing things like `train[:800]+validation[:200]`, there are more.
  return [
      _shard_read_instruction(host_id, host_count, ri, builder) for ri in abs_ri
  ]


def get_data_range(builder, split, host_id, host_count):
  """Return a (sub)split adapted to a given host.

  Each host reads a (same-size) subset of the given `split` such that they
  all go through different parts of it. For doing so, we need to do some
  indexing computation on a given `builder` and `split`, because we want to
  support using contiguous subsets such as `train[10%:20%]`.

  Args:
    builder: TFDS Builder for the datset.
    split: str; Split for which we want to create the data ranage.
    host_id: int; Id of the host.
    host_count: int; Total Number of hosts.

  Returns:
    Data range for the current host and the given split.
  """
  data_range = _get_data_range(builder, split, host_id, host_count)
  assert len(data_range) == 1, ('Multiple TFDS splits not supported. '
                                'Please use `_get_data_range` instead.')
  splitname, host_start, host_end = data_range[0]
  logging.info('Host %d data range: from %s to %s (from split %s)',
               jax.process_index(), host_start, host_end, splitname)

  return splitname, host_start, host_end


def get_num_examples(dataset, split, data_dir=None):
  """Returns the total number of examples in a dataset split."""
  builder = tfds.builder(dataset, data_dir=data_dir)

  n = 0
  nhosts = jax.process_count()
  for ihost in range(nhosts):
    for _, start, end in _get_data_range(builder, split, ihost, nhosts):
      n += end - start

  remainder = builder.info.splits[split].num_examples - n
  if remainder:
    warning = (f'Dropping {remainder} examples for the '
               f'{builder.info.name} dataset, {split} split. '
               f'The reason is that all hosts should have the same number '
               f'of examples in order to guarantee that they stay in sync.')
    logging.warning(warning)

  return n


def get_dataset_tfds(dataset, split, shuffle_files=True, data_dir=None):
  """Data provider."""
  builder = tfds.builder(dataset, data_dir=data_dir)

  # Compute the read instructions for the current host in order to shard the
  # data spec across hosts. It looks complex, but is mostly tfds boilerplate!
  host_ris = []
  for split, start, end in _get_data_range(builder, split, jax.process_index(),
                                           jax.process_count()):
    logging.info('Host %d data range: from %s to %s (from split %s)',
                 jax.process_index(), start, end, split)
    host_ris.append(
        tfds.core.ReadInstruction(split, unit='abs', from_=start, to=end))

  # Each host is responsible for a fixed subset of data
  return builder.as_dataset(
      split=sum(host_ris[1:], host_ris[0]),  # API wants a sum, not a list O.o
      shuffle_files=shuffle_files,
      read_config=tfds.ReadConfig(
          skip_prefetch=True,  # We prefetch after pipeline.
          try_autocache=False,  # We control this, esp. for few-shot.
      ),
      decoders={'image': tfds.decode.SkipDecoding()})


def make_pipeline(data,
                  preprocess_fn,
                  batch_size,
                  drop_remainder,
                  cache='loaded',
                  repeats=None,
                  repeat_after_batching=False,
                  shuffle_buffer_size=None,
                  prefetch=2,
                  ignore_errors=False,
                  dataset_service_address=None):
  """Makes an input pipeline for `data`."""
  assert cache in ('loaded', 'batched', False, None)

  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  options.experimental_threading.max_intra_op_parallelism = 1
  data = data.with_options(options)

  if cache == 'loaded':
    data = data.cache()

  if not repeat_after_batching:
    data = data.repeat(repeats)

  if shuffle_buffer_size is not None:
    data = data.shuffle(shuffle_buffer_size)

  data = data.map(
      preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if ignore_errors:
    # Skip broken images. This does not slow things down.
    data = data.apply(tf.data.experimental.ignore_errors())

  data = data.batch(batch_size, drop_remainder=drop_remainder)

  if cache == 'batched':
    data = data.cache()

  if repeat_after_batching:
    data = data.repeat(repeats)

  if dataset_service_address:
    data = distribute(data, dataset_service_address)

  if prefetch == 'autotune':
    data = data.prefetch(tf.data.experimental.AUTOTUNE)
  elif prefetch:
    data = data.prefetch(prefetch)
  # And 0 or None mean no prefetching.

  return data


def get_data(dataset,
             split,
             batch_size,
             preprocess_fn=lambda x: x,
             repeats=None,
             shuffle_buffer_size=None,
             prefetch=2,
             cache='loaded',
             repeat_after_batching=False,
             drop_remainder=True,
             data_dir=None,
             ignore_errors=False,
             shuffle_files=True,
             dataset_service_address=None):
  """API kept for backwards compatibility."""
  data = get_dataset_tfds(
      dataset=dataset,
      split=split,
      shuffle_files=shuffle_files,
      data_dir=data_dir)
  if split != 'train':
    dataset_service_address = None
  return make_pipeline(
      data=data,
      preprocess_fn=preprocess_fn,
      batch_size=batch_size,
      drop_remainder=drop_remainder,
      cache=cache,
      repeats=repeats,
      prefetch=prefetch,
      shuffle_buffer_size=shuffle_buffer_size,
      repeat_after_batching=repeat_after_batching,
      ignore_errors=ignore_errors,
      dataset_service_address=dataset_service_address)


def inception_crop_with_mask(
    image, mask, resize_size=None, area_min=5, area_max=100):
  """Applies the same inception-style crop to an image and a mask tensor.

  Inception-style crop is a random image crop (its size and aspect ratio are
  random) that was used for training Inception models, see
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

  Args:
    image: [H, W, C] image tensor.
    mask: [H, W, None] mask tensor. H and W must match the image. Will be
      resized using tf.image.ResizeMethod.NEAREST_NEIGHBOR.
    resize_size: Sequence of 2 ints; Resize image to [height, width] after crop.
    area_min: minimal crop area.
    area_max: maximal crop area.

  Returns:
    Cropped image and mask tensors.
  """
  begin, size, _ = tf.image.sample_distorted_bounding_box(
      tf.shape(image), tf.zeros([0, 0, 4], tf.float32),
      area_range=(area_min / 100, area_max / 100),
      min_object_covered=0,  # Don't enforce a minimum area.
      use_image_if_no_bounding_boxes=True)

  # Process image:
  image_cropped = tf.slice(image, begin, size)
  image_cropped.set_shape([None, None, image.shape[-1]])
  if resize_size:
    image_cropped = tf.image.resize(
        image_cropped, resize_size, tf.image.ResizeMethod.BILINEAR)

  # Process mask:
  mask_cropped = tf.slice(mask, begin, size)
  mask_cropped.set_shape([None, None, mask.shape[-1]])
  if resize_size:
    mask_cropped = tf.image.resize(
        mask_cropped, resize_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return image_cropped, mask_cropped


def distribute(
    dataset: tf.data.Dataset, dataset_service_address: str,
    processing_mode: str = 'parallel_epochs') -> tf.data.Dataset:
  dataset_id = tf.data.experimental.service.register_dataset(
      service=dataset_service_address,
      dataset=dataset
  )
  logging.info('tfds service: process %d got id %d',
               jax.process_index(), dataset_id)
  return tf.data.experimental.service.from_dataset_id(
      processing_mode=processing_mode,
      service=dataset_service_address,
      dataset_id=dataset_id,
      job_name='scenic_data_pipeline',
      element_spec=dataset.element_spec)
