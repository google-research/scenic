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

"""Data pipeline for text-conditional detection training.

The pipeline can combine several source datasets, e.g. Objects365 and Visual
Genome. In addition, the pipeline can merge multiple images into "mosaics" to
increase size and class diversity of the training examples.

This file deals with the high-level logistics of processing the data, e.g.
dataset merging and mosaic creation. Most image and label processing is
implemented in image_ops.py and label_ops.py, and configured in the
config.dataset_configs.train.preproc_spec config field.

Rougly, preprocessing proceeds as follows:

 1. Integer labels are converted to text queries by using the category
    names.

 2. For each image, the pipeline produces a set of text queries, which consist
    of positive queries (categories known to be in the image, based on the
    ground-truth bounding boxes) and negative queries (categories known to be
    absent from the image). Optionally, random prompt templates are added to
    category names.

 3. From the queries, classification targets are constructed. In each image, the
    training target for each object/box is the one-hot-encoded
    index of the query corresponding to that object.

The entry-point into the pipeline is `get_dataset`. See docstrings for details.
"""
import dataclasses
import functools
from typing import Any, Dict, Optional, Sequence

from clu import deterministic_data
from clu import preprocess_spec
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.projects.owl_vit.preprocessing import image_ops
from scenic.projects.owl_vit.preprocessing import label_ops
from scenic.projects.owl_vit.preprocessing import mosaic
import tensorflow as tf
import tensorflow_datasets as tfds

Features = preprocess_spec.Features

NUM_PARALLEL_CALLS = tf.data.AUTOTUNE

DECODERS = {
    'visual_genome:1.0.0':
        label_ops.DecodeVisualGenome,
    'lvis:1.2.0':
        label_ops.DecodeLvis,
    'objects365:0.0.1':
        label_ops.DecodeObjects365,
}

# All ops must be listed in either PRE_MOSAIC_OPS or POST_MOSAIC_OPS or both.
# Maintaining exhaustive lists avoids bugs where ops are incorrectly assumed to
# be pre- or post-mosaic.
PRE_MOSAIC_OPS = tuple(
    dec.func if isinstance(dec, functools.partial) else dec
    for dec in DECODERS.values()) + (
        image_ops.CropOrPad,
        image_ops.CropOrPadMetaData,
        image_ops.Drop,
        image_ops.Keep,
        image_ops.RandomCrop,
        image_ops.RandomFlipLeftRight,
        image_ops.ResizeWithPad,
        label_ops.CanonicalizeTextLabels,
        label_ops.RemoveForbiddenLabels,
    )

POST_MOSAIC_OPS = (
    image_ops.CropOrPad,
    image_ops.CropOrPadMetaData,
    image_ops.MergeOverlappingInstances,
    label_ops.AddQuerySet,
    label_ops.AddRandomNegativeLabels,
    label_ops.AddRandomPrompts,
    label_ops.RemovePromptabilityMarker,
    label_ops.SingleToMultiLabel,
    label_ops.ClipTokenizeQueries,
)


def _get_pre_mosaic_process_fn(
    builder: tfds.core.DatasetBuilder,
    decoder_kwargs: Dict[str, Any],
    spec: str,
    mosaic_size: int = 1,
) -> preprocess_spec.PreprocessFn:
  """Constructs the preprocess_fn that should be applied before mosaicing.

  Args:
    builder: TFDS dataset builder.
    decoder_kwargs: Decoder kwargs.
    spec: Config string specifying the preprocessing.
    mosaic_size: Number of tiles along each mosaic edge. For no mosaicing, set
      to 1.

  Returns:
    PreprocessFns for pre-mosaic processing.
  """
  all_ops = preprocess_spec.get_all_ops(
      'scenic.projects.owl_vit.preprocessing.image_ops')
  all_ops += preprocess_spec.get_all_ops(
      'scenic.projects.owl_vit.preprocessing.label_ops')
  pre_mosaic_fn = preprocess_spec.parse(spec, all_ops, only_jax_types=False)

  # Add decoder:
  tfds_name = f'{builder.name}:{builder.version}'
  decoder_kwargs = decoder_kwargs.copy()
  decoder_name = decoder_kwargs.pop('name', tfds_name)
  if decoder_name not in DECODERS:
    raise ValueError(
        f'Did not find decoder for {decoder_name}. Please specify decoders for '
        'all datasets in DECODERS.')
  pre_mosaic_fn.ops = (
      DECODERS[decoder_name](**decoder_kwargs), *pre_mosaic_fn.ops)

  # Reduce the resize-size of all ops by a factor of `mosaic_size`:
  resize_ops = (
      image_ops.ResizeWithPad,
      image_ops.CropOrPad)
  pre_mosaic_ops = []
  for op in pre_mosaic_fn.ops:
    # Validate op:
    if isinstance(op, PRE_MOSAIC_OPS):
      pass  # This is a pre-mosaic op, continue below.
    elif isinstance(op, POST_MOSAIC_OPS):
      continue  # This is a post-mosaic-only op, skip.
    else:
      raise ValueError(
          f'Op {op!r} not found in PRE_MOSAIC_OPS or POST_MOSAIC_OPS. '
          'Please add op to either or both lists.')

    # Adjust resizing ops to mosaic tile size:
    if isinstance(op, resize_ops):
      assert op.size % mosaic_size == 0, 'Size is not evenly divisible!'
      op = dataclasses.replace(op, size=op.size // mosaic_size)
    pre_mosaic_ops.append(op)
  pre_mosaic_fn.ops = pre_mosaic_ops

  return pre_mosaic_fn


def _get_post_mosaic_process_fn(
    spec: str,
) -> preprocess_spec.PreprocessFn:
  """Constructs the preprocess_fn that should be applied after mosaicing.

  Args:
    spec: Config string specifying the preprocessing.

  Returns:
    PreprocessFns for post-mosaic processing.
  """
  all_ops = preprocess_spec.get_all_ops(
      'scenic.projects.owl_vit.preprocessing.image_ops')
  all_ops += preprocess_spec.get_all_ops(
      'scenic.projects.owl_vit.preprocessing.label_ops')
  post_mosaic_fn = preprocess_spec.parse(spec, all_ops, only_jax_types=False)
  post_mosaic_fn.ops = [
      op for op in post_mosaic_fn.ops if isinstance(op, POST_MOSAIC_OPS)]
  return post_mosaic_fn


def _get_single_tfds_dataset(
    builder: tfds.core.DatasetBuilder,
    split: str,
    batch_size: int,
    preprocess_fn: preprocess_spec.PreprocessFn,
    rng: Any,
    shuffle: bool = False,
    shuffle_buffer_size: int = 1000,
    cache: bool = False,
) -> tf.data.Dataset:
  """Creates dataset from builder and applies pre-mosaic preprocessing.

  Args:
    builder: TFDS dataset builder.
    split: Train/test/validation split.
    batch_size: Batch size.
    preprocess_fn: Preprocess function with pre-mosaic ops.
    rng: Random seed.
    shuffle: Whether to shuffle.
    shuffle_buffer_size: Shuffle buffer size.
    cache: Whether to cache the dataset.

  Returns:
    tf.data.Dataset with pre-mosaic preprocessing applied.
  """

  host_split = deterministic_data.get_read_instruction_for_host(
      split,
      dataset_info=builder.info,
      remainder_options=deterministic_data.RemainderOptions.DROP,
  )

  ds = deterministic_data.create_dataset(
      builder,
      split=host_split,
      preprocess_fn=preprocess_fn,
      cache=cache,
      batch_dims=[batch_size],
      rng=rng,
      num_epochs=None,  # None = repeat forever.
      shuffle=shuffle,
      shuffle_buffer_size=shuffle_buffer_size)

  return ds


def _get_merged_dataset(builders: Sequence[tfds.core.DatasetBuilder],
                        splits: Sequence[str],
                        dataset_probs: Sequence[float],
                        decoder_kwarg_list: Sequence[Dict[str, Any]],
                        preproc_spec: str,
                        mosaic_size: int,
                        rng: Any,
                        shuffle: bool = False,
                        shuffle_buffer_size: int = 10_000,
                        cache: bool = False) -> tf.data.Dataset:
  """Creates datasets from builders, applies preprocessing, and merges them.

  Args:
    builders: List of TFDS dataset builders.
    splits: List containing the split to use for each builder.
    dataset_probs: Sampling probabilities for each dataset.
    decoder_kwarg_list: Kwargs to pass to the decoder.
    preproc_spec: Preprocessing specification string.
    mosaic_size: Number of tiles along each mosaic edge. For no mosaicing, set
      to 1.
    rng: Random seed (JAX format).
    shuffle: Whether to shuffle.
    shuffle_buffer_size: Shuffle buffer size (ignored if shuffle is False).
    cache: Whether to cache the datasets.

  Returns:
    tf.data.Dataset with pre-mosaic preprocessing applied.
  """

  datasets_to_merge = []
  for builder, split, decoder_kwargs in zip(
      builders, splits, decoder_kwarg_list):
    pre_mosaic_processing = _get_pre_mosaic_process_fn(
        builder,
        decoder_kwargs,
        preproc_spec,
        mosaic_size)
    rng, ds_rng = jax.random.split(rng)
    datasets_to_merge.append(_get_single_tfds_dataset(
        builder,
        split,
        batch_size=mosaic_size**2,
        preprocess_fn=pre_mosaic_processing,
        rng=ds_rng,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        cache=cache))

  try:
    return tf.data.Dataset.sample_from_datasets(
        datasets_to_merge, weights=dataset_probs,
        seed=None if rng is None else rng[0])
  except TypeError as e:
    # There is a mismatch between the datasets to be merged. If it's a simple
    # difference in the feature keys, provide a nicer message than tf.data:
    expected_keys = set(datasets_to_merge[0].element_spec.keys())
    for dataset in datasets_to_merge[1:]:
      actual_keys = set(dataset.element_spec.keys())
      if actual_keys != expected_keys:
        raise TypeError(
            'Datasets to be merged must have the same structure, but had keys '
            f'\n\n{expected_keys}\n\n and \n\n{actual_keys}\n\n'
            f'Difference:\n{expected_keys.symmetric_difference(actual_keys)}'
            ) from e
    # If the difference is not in the feature keys, raise the original error:
    raise e


def _build_pipeline(config: ml_collections.ConfigDict,
                    batch_size: int,
                    rng: Any,
                    shuffle: bool = False) -> tf.data.Dataset:
  """Build a tf.data.Dataset pipeline using clu.deterministic_data.

  The pipeline has the following steps:

  0. All used datasets should have text label fields. Convert labels to text in
     their decoders.

  1. Datasets are merged with tf.data.Dataset.sample_from_datasets.

  2. Optionally, mosaics are created.

  3. Prompts are added to text labels.

  4. Per-image label spaces are generated from text labels.

  5. Text queries are tokenized.

  6. Features are converted to Scenic format.

  Args:
    config: Dataset configurations.
    batch_size: Total batch size (sum for all devices).
    rng: Random seed (JAX format).
    shuffle: Whether to shuffle.

  Returns:
    tf.data.Dataset after preprocessing, merging, mosaicing, and batching.
  """

  builders = [
      tfds.builder(name, data_dir=config.data_dirs[0].get(name))
      for name in config.tfds_names
  ]

  decoder_kwarg_list = config.get('decoder_kwarg_list',
                                  [{}] * len(config.tfds_names))
  mosaic_sizes = config.get('mosaic_sizes', (1,))
  mosaic_probs = config.get('mosaic_probs', (1.0,))
  mosaic_datasets = []
  for mosaic_size in mosaic_sizes:
    rng, ds_rng = jax.random.split(rng)
    merged_dataset = _get_merged_dataset(
        builders=builders,
        splits=config.splits,
        dataset_probs=config.dataset_probs,
        decoder_kwarg_list=decoder_kwarg_list,
        preproc_spec=config.preproc_spec,
        mosaic_size=mosaic_size,
        rng=ds_rng,
        shuffle=shuffle,
        shuffle_buffer_size=config.shuffle_buffer_size,
        cache=config.get('cache', False))

    # Build mosaics:
    if mosaic_size == 1:  # I.e. no mosaic.
      merged_dataset = merged_dataset.unbatch()  # Remove mosaic dimension.
    else:
      merged_dataset = merged_dataset.map(
          mosaic.CreateMosaic(mosaic_size),
          num_parallel_calls=NUM_PARALLEL_CALLS)

    mosaic_datasets.append(merged_dataset)

  # Merge mosaic sizes:
  final_dataset = tf.data.Dataset.sample_from_datasets(
      mosaic_datasets, weights=mosaic_probs, seed=rng[0])

  # Apply post-mosaic processing:
  post_mosaic_processing = _get_post_mosaic_process_fn(config.preproc_spec)
  post_mosaic_ops = list(post_mosaic_processing.ops)

  post_mosaic_ops.append(
      label_ops.ConvertToScenic(
          input_range=config.input_range,
      ))

  post_mosaic_processing.ops = post_mosaic_ops
  final_dataset = final_dataset.map(
      post_mosaic_processing, num_parallel_calls=NUM_PARALLEL_CALLS)

  # Batch to the desired output batch size:
  batch_dims = [
      jax.local_device_count(), batch_size // jax.local_device_count()
  ]
  for batch_size in reversed(batch_dims):
    final_dataset = final_dataset.batch(
        batch_size, drop_remainder=True, num_parallel_calls=NUM_PARALLEL_CALLS)

  return final_dataset.prefetch(tf.data.AUTOTUNE)


def _validate_and_normalize_config(
    dataset_configs: ml_collections.ConfigDict,
    train: bool = False) -> ml_collections.ConfigDict:
  """Validates dataset_configs and normalizes it to a common format.

  Common and train/eval-mode-specific configs are merged into one.
  TFDS data dirs are added to the config.

  Args:
    dataset_configs: Dataset configuration as specified in the config file.
    train: Flag to switch between train and eval mode.

  Returns:
    Normalized config with merged common and mode-specific settings.
  """
  # Create a merged config for this mode (i.e. train or test):
  mode_config = dataset_configs.train if train else dataset_configs.eval
  config = ml_collections.ConfigDict({**dataset_configs, **mode_config})

  # Validate config structure:
  decoder_kwarg_list = config.get('decoder_kwarg_list',
                                  [{}] * len(config.tfds_names))
  if not (len(config.tfds_names) == len(config.splits) == len(
      config.dataset_probs) == len(decoder_kwarg_list)):
    raise ValueError(
        'Dataset settings must have matching lengths, got '
        f'{config.tfds_names}, {config.splits}, {config.dataset_probs}, '
        f'{decoder_kwarg_list}.')

  mosaic_sizes = config.get('mosaic_sizes', (1,))
  mosaic_probs = config.get('mosaic_probs', (1.0,))
  if len(mosaic_sizes) != len(mosaic_probs):
    raise ValueError(
        'mosaic_sizes and mosaic_probs must have matching lengths, got '
        f'{mosaic_sizes}, {mosaic_probs}.')

  # Determine data_dirs:
  data_dirs = {}
  for tfds_name, kws in zip(config.tfds_names, decoder_kwarg_list):
    # First look for data dir in decoder_kwargs, otherwise use None:
    data_dirs[tfds_name] = kws.get('tfds_data_dir')
  config.data_dirs = (data_dirs,)  # Wrap tuple avoids ConfigDict conversion.

  return config


@datasets.add_dataset('owl_vit')
def get_dataset(
    *,
    batch_size: int,
    eval_batch_size: int,
    num_shards: int,
    rng: Any,
    dataset_configs: ml_collections.ConfigDict,
    dtype_str: str = 'float32',
    shuffle_seed: int = 0,
    dataset_service_address: Optional[str] = None) -> dataset_utils.Dataset:
  """Returns generators for image-text detection datasets.

  In addition to standard detection features this loader also produces textual
  queries (in features['queries']), which defines the per batch labelset. The
  queries are the (argus) tokenized string queries. Text queries from detection
  datasets are simply their class names.

  Args:
    batch_size: Determines the train batch size.
    eval_batch_size: Determines the evaluation batch size.
    num_shards: Unused; determined by CLU.
    rng: JAX RNG key, which can be used for augmentation, shuffling, etc.
    dataset_configs: Dataset configurations.
    dtype_str: Data type of the image. Only 'float32' is currently supported.
    shuffle_seed: Unsupported; use rng instead.
    dataset_service_address: Unsupported; must be None.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
    a test_iter, and a dict of meta_data.
  """
  if dataset_service_address is not None:
    raise NotImplementedError('This dataset does not support the DS service.')
  if rng is None:
    raise NotImplementedError('This dataset requires a JAX RNG.')
  if shuffle_seed:
    raise NotImplementedError(
        'This dataset requires a JAX RNG, do not use shuffle_seed.')
  if len(dataset_configs.eval.tfds_names) > 1:
    raise NotImplementedError(
        'Evaluation with more than one datasets is not supported.')
  if dtype_str != 'float32':
    raise ValueError(f'Unsupported dtype_str: {dtype_str}')

  # Delete unused arguments (see docstring):
  del num_shards, shuffle_seed

  # Ensure a different key on each worker:
  rng = jax.random.fold_in(rng, jax.process_index())

  rng, train_rng = jax.random.split(rng)

  # Training dataset:
  train_config = _validate_and_normalize_config(dataset_configs, train=True)
  train_ds = _build_pipeline(
      config=train_config,
      batch_size=batch_size,
      rng=train_rng,
      shuffle=True,
    )

  example_batch = next(iter(train_ds))
  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)

  if dataset_configs.get('prefetch_to_device'):
    # Async bind batch to device which speeds up training.
    train_iter = jax_utils.prefetch_to_device(
        train_iter, dataset_configs.get('prefetch_to_device'))

  rng, eval_rng = jax.random.split(rng)
  # Evaluation dataset:
  eval_config = _validate_and_normalize_config(dataset_configs, train=False)
  eval_ds = _build_pipeline(
      config=eval_config,
      batch_size=eval_batch_size,
      rng=eval_rng,
  )
  eval_iter = iter(eval_ds)
  eval_iter = map(dataset_utils.tf_to_numpy, eval_iter)

  total_examples = sum(
      dataset_utils.get_num_examples(name, split, train_config.data_dirs[0].get(
          name)) for name, split in zip(dataset_configs.train.tfds_names,
                                        dataset_configs.train.splits))
  eval_data_dir = eval_config.data_dirs[0].get(
      dataset_configs.eval.tfds_names[0])
  total_eval_examples = dataset_utils.get_num_examples(
      dataset_configs.eval.tfds_names[0], dataset_configs.eval.splits[0],
      data_dir=eval_data_dir)

  eval_builder = tfds.builder(dataset_configs.eval.tfds_names[0],
                              data_dir=eval_data_dir)

  if 'bobjects' in eval_builder.info.features:
    num_classes = eval_builder.info.features['bobjects']['label'].num_classes
  else:
    num_classes = eval_builder.info.features['objects']['label'].num_classes
  num_classes += 1

  meta_data = {
      'input_shape': (-1,) + tuple(example_batch['inputs'].shape[-3:]),
      'query_shape': (-1,) + tuple(example_batch['queries'].shape[-2:]),
      'num_train_examples': total_examples,
      'num_eval_examples': total_eval_examples,
      'input_dtype': jnp.float32,
      'target_is_onehot': True,  # We always use the one/multi-hot format.
      'eval_num_classes': num_classes,
  }

  return dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)
