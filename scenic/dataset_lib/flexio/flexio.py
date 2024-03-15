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

"""FlexIO input pipeline."""

import functools
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
from clu import deterministic_data
from clu import preprocess_spec
import grain.tensorflow as grain
import jax
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
import tensorflow as tf
import tensorflow_datasets as tfds



# Default source of pp ops.
DEFAULT_PP_LIBS = []

Features = preprocess_spec.Features
TfFeature = Union[tf.io.FixedLenFeature, tf.io.VarLenFeature,
                  tf.io.FixedLenSequenceFeature]

# From grain/_src/core/constants.py
GRAIN_META_DATA = [
    '_index', '_record_key', '_dataset_index', '_epoch', '_record', 'mids', 'id'
]


def _get_feature(feature_type: str, shape=(), dtype='string') -> TfFeature:
  dtype = tf.dtypes.as_dtype(dtype)
  if feature_type == 'FixedLen':
    return tf.io.FixedLenFeature(shape=shape, dtype=dtype)
  if feature_type == 'VarLen':
    return tf.io.VarLenFeature(dtype=dtype)
  elif feature_type == 'FixedLenSequence':
    return tf.io.FixedLenSequenceFeature(shape=shape, dtype=dtype)
  raise NotImplementedError(f'Feature type {feature_type} not available yet.')


def tf2jax_dtype(dtype: tf.dtypes.DType) -> Union[jnp.dtype, tf.dtypes.DType]:
  """Convert TF dtype to JAX."""
  conv = {
      tf.int8: jnp.int8,
      tf.int16: jnp.int16,
      tf.int32: jnp.int32,
      tf.int64: jnp.int64,
      tf.uint8: jnp.uint8,
      tf.uint16: jnp.uint16,
      tf.uint32: jnp.uint32,
      tf.uint64: jnp.uint64,
      tf.float16: jnp.float16,
      tf.float32: jnp.float32,
      tf.float64: jnp.float64,
      tf.bfloat16: jnp.bfloat16,
      tf.bool: jnp.bool_
  }
  return conv.get(dtype) or dtype




def apply_process_fn_with_populated_seed(ds: tf.data.Dataset,
                                         preprocess_fn: Callable[[Features],
                                                                 Features], *,
                                         rng: jnp.ndarray) -> tf.data.Dataset:
  """Maps `ds` using the preprocess_fn and a deterministic RNG per example.

  Args:
    ds: Dataset containing Python dictionary with the features. The 'rng'
      feature should not exist.
    preprocess_fn: Preprocessing function that takes a Python dictionary of
      tensors and returns a Python dictionary of tensors. The function should be
      convertible into a TF graph.
    rng: Base RNG to use. Per example RNGs will be derived from this by folding
      in the example index.

  Returns:
    The dataset mapped by the `preprocess_fn`.
  """

  def _fn(example_index: int, features: Features) -> Features:
    example_index = tf.cast(example_index, tf.int64)
    if preprocess_spec.SEED_KEY in features:
      logging.warning(('Seed key (%s) already exists in the feature dict -> '
                       '*not* overwriting'), preprocess_spec.SEED_KEY)
    else:
      features[
          preprocess_spec.SEED_KEY] = tf.random.experimental.stateless_fold_in(
              tf.cast(rng, tf.int64), example_index)
    processed = preprocess_fn(features)  # Note: we keep the RNG in the dict.
    return processed

  return ds.enumerate().map(_fn, num_parallel_calls=tf.data.AUTOTUNE)


def get_number_of_examples(config: ml_collections.ConfigDict) -> int:
  """Obtain the number of examples in a thin DMVR or TFDS dataset."""
  if hasattr(config, 'num_examples'):
    return config.num_examples


  if config.source in ['tfds', 'grain']:
    data_dir = config.get('data_dir', None)
    return dataset_utils.get_num_examples(
        config.tfds_name, config.split, data_dir=data_dir)
  raise ValueError(f'Unknown data source: {config.source}')


def get_process_fn(spec: str,
                   pp_libs: Sequence[str]) -> preprocess_spec.PreprocessFn:
  """Constructs the preprocess_fn that should be applied on the data.

  Args:
    spec: Config string specifying the preprocessing.
    pp_libs: List of libraries to collect pp ops from.

  Returns:
    PreprocessFns for pre-processing.
  """
  all_ops = sum(map(preprocess_spec.get_all_ops, pp_libs), [])
  preprocess_fn = preprocess_spec.parse(spec, all_ops, only_jax_types=False)
  return preprocess_fn


def _get_single_tfds_dataset(
    builder: tfds.core.DatasetBuilder,
    split: str,
    batch_size: Optional[int],
    preprocess_fn: Optional[preprocess_spec.PreprocessFn] = None,
    postprocess_fn: Optional[preprocess_spec.PreprocessFn] = None,
    rng: Union[None, jnp.ndarray, tf.Tensor] = None,
    global_rng: Union[None, jnp.ndarray, tf.Tensor] = None,
    shuffle: bool = False,
    shuffle_buffer_size: int = 1000,
    cache: bool = False,
    skip_decoders: Sequence[str] | None = None,
    repeat_dataset: bool = True,
) -> tf.data.Dataset:
  """Creates dataset from builder and applies preprocessing.

  Args:
    builder: TFDS dataset builder.
    split: Train/test/validation split.
    batch_size: Batch size.
    preprocess_fn: Preprocess function with pre-caching ops.
    postprocess_fn: Postprocess function executed *after* batching.
    rng: Random seed.
    global_rng: Global random seed (same across hosts).
    shuffle: Whether to shuffle.
    shuffle_buffer_size: Shuffle buffer size.
    cache: Whether to cache the dataset.
    skip_decoders: Pass decoders to skip to create_dataset (mainly for image).
    repeat_dataset: If True, the dataset is repeated indefinitely.

  Returns:
    tf.data.Dataset with preprocessing applied.
  """

  host_split = deterministic_data.get_read_instruction_for_host(
      split,
      dataset_info=builder.info,
      remainder_options=deterministic_data.RemainderOptions.DROP,
  )
  ds = deterministic_data.create_dataset(
      builder,
      split=host_split,
      preprocess_fn=None,
      cache=cache,
      batch_dims=(),
      rng=global_rng,
      num_epochs=1,  # None = repeat forever.
      shuffle=False,
      pad_up_to_batches=None,
      decoders={d: tfds.decode.SkipDecoding() for d in skip_decoders or []},
  )
  if cache:
    ds = ds.cache()
  if repeat_dataset:
    ds = ds.repeat()  # Repeat indefinitly.
  if shuffle:
    ds = ds.shuffle(shuffle_buffer_size, seed=rng[0])
  if preprocess_fn is not None:
    if rng is not None:
      ds = apply_process_fn_with_populated_seed(ds, preprocess_fn, rng=rng)
    else:
      ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
  if batch_size:
    ds = ds.batch(
        batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True)
  if postprocess_fn is not None:
    if rng is not None:
      ds = apply_process_fn_with_populated_seed(ds, postprocess_fn, rng=rng)
    else:
      ds = ds.map(postprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

  return ds


def _get_single_grain_dataset(
    builder: tfds.core.DatasetBuilder,
    start_index: int,
    split: str,
    batch_size: Optional[int],
    grain_configs: Optional[Dict[str, Any]] = None,
    preprocess_fn: Optional[preprocess_spec.PreprocessFn] = None,
    postprocess_fn: Optional[preprocess_spec.PreprocessFn] = None,
    rng: Union[None, jnp.ndarray, tf.Tensor] = None,
    global_rng: Union[None, jnp.ndarray, tf.Tensor] = None,
    shuffle: bool = False,
    cache: bool = False,
    repeat_dataset: bool = True,
) -> tf.data.Dataset:
  """Creates a Grain-backed dataset from builder and applies preprocessing.

  Args:
    builder: TFDS dataset builder.
    start_index: Index dataset (Grain) start index.
    split: Train/test/validation split.
    batch_size: Batch size.
    grain_configs: To handle Grain config options.
    preprocess_fn: Preprocess function with pre-caching ops.
    postprocess_fn: Postprocess function executed *after* batching.
    rng: Random seed.
    global_rng: Global random seed (same across hosts).
    shuffle: Whether to shuffle.
    cache: Whether to cache the dataset.
    repeat_dataset: If True, the dataset is repeated indefinitely.

  Returns:
    tf.data.Dataset with preprocessing applied.
  """

  if rng is not None:
    raise ValueError(
        'For Grain-backed datasets `global_rng` controls per-example seeds.')
  if cache:
    raise ValueError('Grain datasets are created as inifinitly repeating and '
                     'cannot be cached.')

  # TODO(dehghani): These settings are *not* per-dataset but rather global
  #  grain flags. This will be problematic if we have more than one Grain-backed
  #  source but wishing different setting for them. Find a way for setting
  #  these in a better way.
  grain_configs = grain_configs or {}
  for config_k, config_v in grain_configs.items():
    grain.config.update(config_k, config_v)

  ds = grain.load_from_tfds(
      tfds_info=builder.info,
      split=split,
      num_epochs=None if repeat_dataset else 1,  # None = repeat forever.
      shuffle=shuffle,
      seed=global_rng,
      shard_options=grain.ShardByJaxProcess(drop_remainder=True),
      transformations=preprocess_fn or (),
      batch_size=batch_size).as_dataset(start_index=start_index)

  if postprocess_fn is not None:
    ds = ds.map(postprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

  return ds






def _build_pipeline(
    split: str,
    start_step: Optional[int],
    dataset_configs: ml_collections.ConfigDict,
    batch_size: Optional[int],
    num_local_shards: int,
    rng: Union[None, jnp.ndarray, tf.Tensor] = None,
    global_rng: Union[None, jnp.ndarray, tf.Tensor] = None,
    shuffle: bool = False
) -> Optional[Union[tf.data.Dataset, Dict[str, tf.data.Dataset]]]:
  """Build a tf.data.Dataset pipeline using clu.deterministic_data or DMVR.

  Args:
    split: The split to be used.
    start_step: Start step for GRAIN-backed datasets.
    dataset_configs: Dataset configurations.
    batch_size: Total batch size (sum for all devices).
    num_local_shards: Number of local shards (usually num local devices).
    rng: Per-host random seed (JAX format).
    global_rng: Global random seed (JAX format).
    shuffle: Whether to shuffle.

  Returns:
    tf.data.Dataset after preprocessing, merging, mosaicing, and batching.
  """
  # Pre-processing libs:
  pp_libs = dataset_configs.get('pp_libs', DEFAULT_PP_LIBS)
  process_fn = functools.partial(get_process_fn, pp_libs=pp_libs)

  if split not in dataset_configs:
    return None

  mode_config = dataset_configs.get(split)
  config = ml_collections.ConfigDict({**dataset_configs, **mode_config})

  if len(config.sources) > 1:
    merge_sources = config.merge_sources
  else:
    merge_sources = True

  any_grain = any([src.source == 'grain' for src in config.sources])
  if any_grain:
    if len(config.sources) > 1 and merge_sources:
      raise NotImplementedError(
          'Mixing of GRAIN-backed datasets is not yet '
          'implemented in FlexIO, but can be accomplished '
          'via `TfMixtureIndexSampler` and '
          '`TfMixtureDataLoader`.')
    if start_step is None:
      raise ValueError(
          'For GRAIN-backed datasets you need to provide a '
          '`start_step` to `get_dataset`.'
      )
  elif start_step is not None:
    logging.warning('Start step (%s) provided for non-GRAIN dataset.',
                    start_step)

  sources, weights = {}, {}
  for src_id, src in enumerate(config.sources):
    src_name = src.get('name', f'src_{src_id}')
    if rng is not None:
      rng, ds_rng = jax.random.split(rng)
    else:
      ds_rng = None

    if src.source == 'tfds':
      builder = tfds.builder(src.tfds_name, data_dir=src.get('data_dir'))
      ds = _get_single_tfds_dataset(
          builder,
          src.split,
          batch_size=src.get('batch_size'),
          preprocess_fn=process_fn(src.get('preproc_spec') or ''),
          postprocess_fn=process_fn(src.get('postproc_spec') or ''),
          rng=ds_rng,
          global_rng=global_rng,
          shuffle=shuffle,
          shuffle_buffer_size=src.shuffle_buffer_size,
          cache=src.get('cache', False),
          skip_decoders=src.get('skip_decoders'),
          repeat_dataset=src.get('repeat_dataset', True),
      )
    elif src.source == 'grain':
      if src.get('shuffle_buffer_size') is not None:
        raise ValueError('GRAIN-backed datasets always use a global shuffle.')
      if batch_size is not None:
        global_batch_size = batch_size * jax.process_count()
      else:
        global_batch_size = jax.process_count()
      # TODO(dehghani): Calculating `start_index` based on step like this
      #  works only if there is no filtering or example packing. Switch to
      #  grain checkpointing when it's mature.
      start_index = int(start_step * global_batch_size + jax.process_index())
      builder = tfds.builder(src.tfds_name, data_dir=src.get('data_dir'))
      ds = _get_single_grain_dataset(
          builder,
          start_index,
          src.split,
          batch_size=src.get('batch_size'),
          grain_configs=src.get('grain_configs'),
          preprocess_fn=process_fn(src.get('preproc_spec') or ''),
          postprocess_fn=process_fn(src.get('postproc_spec') or ''),
          rng=None,
          global_rng=global_rng,
          shuffle=shuffle,
          repeat_dataset=src.get('repeat_dataset', True),
          )
      if src.get('drop_grain_meta_features', True):

        def _drop_grain_meta_features(
            features: Mapping[str, Any]) -> Mapping[str, Any]:
          """Returns the features with any Grain meta features."""
          result = {}
          for k, v in features.items():
            if k not in GRAIN_META_DATA:
              result[k] = v
          return result

        ds = ds.map(
            _drop_grain_meta_features, num_parallel_calls=tf.data.AUTOTUNE)
    else:
      raise ValueError(f'Unknown dataset source: {src.source}')
    sources[src_name] = ds
    if merge_sources:
      weights[src_name] = src.get('weight', 1.0)
    else:
      if src.get('weight'):
        raise ValueError(
            'Per source `weight` should not be provided unless you are merging '
            'datasets (i.e., merge_sources=True).')

  def _batch_and_prefetch(ds, batch_size):
    if batch_size is None:
      return ds

    # Batch to the desired output batch size:
    if batch_size % num_local_shards != 0:
      raise ValueError(
          f'Local (host) batch size of {batch_size} is not divisible'
          f'to num_local_shard={num_local_shards}.')
    batch_dims = [num_local_shards, batch_size // num_local_shards]
    for batch_size in reversed(batch_dims):
      if dataset_configs.get('padded_batch'):
        ds = ds.padded_batch(batch_size, drop_remainder=True)
      else:
        ds = ds.batch(
            batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

    # Having prefetch as the last transformation will prevent automatic
    # injection of prefetch(AUTOTUNE).
    ds = ds.prefetch(2)

    # Configure parallelism.
    # TODO(agritsenko, josipd): make these settings configurable as the defaults
    # may leads to OOM.
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    options.threading.max_intra_op_parallelism = 1
    return ds.with_options(options)

  if merge_sources:
    ds_sources = list(sources.values())
    if len(ds_sources) > 1:
      ds_weights = list(weights.values())
      # Normalize sampling weights.
      sum_weights = sum(ds_weights)
      ds_weights = [w / sum_weights for w in ds_weights]
      ds = tf.data.Dataset.sample_from_datasets(
          ds_sources, ds_weights, seed=rng[0] if rng is not None else None)
    else:
      ds = ds_sources[0]

    # Map with shared pp spec, only possible if we are merging the sources:
    def _apply_global_processing(
        ds_pp: tf.data.Dataset, pp_str: str) -> tf.data.Dataset:
      if rng is not None:
        return apply_process_fn_with_populated_seed(
            ds_pp, process_fn(pp_str), rng=rng)
      else:
        return ds_pp.map(
            process_fn(pp_str),
            num_parallel_calls=tf.data.AUTOTUNE)

    ds = _apply_global_processing(ds, config.get('preproc_spec') or '')
    ds = _batch_and_prefetch(ds, batch_size)
    return _apply_global_processing(ds, config.get('postproc_spec') or '')

  else:
    for ds_name, ds in sources.items():
      # TODO(dehghani): Add support for have different batch_sizes for
      #  different sources.
      sources[ds_name] = _batch_and_prefetch(ds, batch_size)
  return sources


def get_iterator(
    ds: Union[tf.data.Dataset, Dict[str, tf.data.Dataset]],
    configs=ml_collections.ConfigDict,
    *,
    return_iterator: bool = False
) -> Tuple[Union[Iterable[Any] | None, Dict[str, Iterable[Any] | None]], Union[
    Tuple[Any, ...], Dict[str, Tuple[Any, ...]]], Union[int, Dict[str, int]]]:
  """Given a (dict of) Dataset object(s), returns iterators and metadata.

  Args:
    ds: A tf.data.Dataset instance or a dictionary of TFDS instances.
    configs: A Config dict.
    return_iterator: If False, the function returns a None instead of an
      iterator.

  Returns:
    Iterators, input specification and num_examples.
  """

  def _get_input_spec(ds):
    return jax.tree_util.tree_map(
        # Remove host dimension from the shapes.
        lambda x: (tuple(x.shape.as_list()[1:]), tf2jax_dtype(x.dtype)),
        ds.element_spec)

  if ds is not None:
    total_examples = {}
    for src_id, src in enumerate(configs.sources):
      total_examples[src.get('name',
                             f'src_{src_id}')] = get_number_of_examples(src)
    if isinstance(ds, dict):
      ds_iter, input_spec = {}, {}
      for dataset_name, dataset in ds.items():
        if not return_iterator:
          ds_iter[dataset_name] = None
        else:
          ds_it = iter(dataset)
          ds_iter[dataset_name] = map(dataset_utils.tf_to_numpy, ds_it)
        input_spec[dataset_name] = _get_input_spec(dataset)
      # TODO(dehghani): Add support for having different input specs.
      first_input_spec = list(input_spec.values())[0]
      for in_spec in input_spec.values():
        assert in_spec == first_input_spec, (
            'For now, input specs for all sources should be the same.')
      input_spec = first_input_spec
    else:
      # Either a single dataset, or we merged them into a single dataset.
      if not return_iterator:
        ds_iter = None
      else:
        ds_it = iter(ds)
        ds_iter = map(dataset_utils.tf_to_numpy, ds_it)
      total_examples = sum(list(total_examples.values()))
      input_spec = _get_input_spec(ds)
  else:
    ds_iter = None
    input_spec = None
    total_examples = -1

  return ds_iter, input_spec, total_examples


@datasets.add_dataset('flexio')
def get_dataset(
    *,
    batch_size: Optional[int],
    eval_batch_size: Optional[int],
    num_shards: int,
    rng: Union[None, jnp.ndarray, tf.Tensor],
    dataset_configs: ml_collections.ConfigDict,
    start_step: Optional[int] = None,
    dtype_str: str = 'float32',
    shuffle_seed: int = 0,
    dataset_service_address: Optional[str] = None) -> dataset_utils.Dataset:
  """Returns generators for video datasets.

  Args:
    batch_size: Determines the train batch size.
    eval_batch_size: Determines the evaluation batch size.
    num_shards: Number of local shards (usually num local devices).
    rng: JAX rng key, which can be used for augmentation, shuffling, etc.
    dataset_configs: Dataset configurations.
    start_step: Current step, used for deterministic input pipeline backed by
      GRAIN.
    dtype_str: Data type of the image. Only 'float32' is currently supported.
    shuffle_seed: Unsupported; use rng instead.
    dataset_service_address: Unsupported; must be None.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
    a test_iter, and a dict of meta_data.
  """

  if rng is None:
    raise NotImplementedError('This dataset requires a JAX RNG.')
  if shuffle_seed:
    raise NotImplementedError(
        'This dataset requires a JAX RNG, do not use shuffle_seed.')
  if dataset_service_address:
    raise ValueError('FlexIO pipeline does not support data service.')
  if dtype_str != 'float32':
    raise ValueError(f'Unsupported dtype_str: {dtype_str}')

  # Delete unused arguments (see docstring):
  del shuffle_seed

  # Ensure a different key on each worker:
  global_rng = rng
  rng = jax.random.fold_in(rng, jax.process_index())

  # Training dataset:
  rng, train_rng = jax.random.split(rng)
  train_ds = _build_pipeline(
      split='train',
      start_step=start_step,
      dataset_configs=dataset_configs,
      batch_size=batch_size,
      num_local_shards=num_shards,
      rng=train_rng,
      global_rng=global_rng,
      shuffle=True)

  # Evaluation dataset:
  rng, eval_rng = jax.random.split(rng)
  eval_ds = _build_pipeline(
      split='eval',
      start_step=0,
      dataset_configs=dataset_configs,
      batch_size=eval_batch_size,
      num_local_shards=num_shards,
      global_rng=global_rng,
      rng=eval_rng)

  return_iterators = dataset_configs.get('return_iterators', True)
  train_iter, train_input_spec, total_train_examples = get_iterator(
      train_ds,
      dataset_configs.get('train'),
      return_iterator=return_iterators)
  eval_iter, eval_input_spec, total_eval_examples = get_iterator(
      eval_ds,
      dataset_configs.get('eval'),
      return_iterator=return_iterators)

  # Testing dataset:
  rng, test_rng = jax.random.split(rng)
  test_ds = _build_pipeline(
      split='test',
      start_step=0,
      dataset_configs=dataset_configs,
      batch_size=eval_batch_size,
      num_local_shards=num_shards,
      global_rng=global_rng,
      rng=test_rng)

  test_iter, test_input_spec, total_test_examples = get_iterator(
      test_ds,
      dataset_configs.get('test'),
      return_iterator=return_iterators)

  # Collect dataset metadata.
  meta_data = {
      'num_train_examples': total_train_examples,
      'num_eval_examples': total_eval_examples,
      'num_test_examples': total_test_examples,
  }

  if train_ds is not None:
    meta_data['input_spec'] = train_input_spec
  if eval_ds is not None:
    meta_data['eval_input_spec'] = eval_input_spec
  if test_ds is not None:
    meta_data['test_input_spec'] = test_input_spec

  # Update metadata if any extra was provided via config.
  meta_data.update(dataset_configs.get('extra_meta_data', {}))
  dataset = {'train_iter': train_iter, 'valid_iter': eval_iter,
             'test_iter': test_iter, 'meta_data': meta_data}
  return_datasets = dataset_configs.get('return_datasets', False)
  if return_datasets:
    dataset.update(
        {'train_ds': train_ds, 'valid_ds': eval_ds, 'test_ds': test_ds})
  logging.info('Dataset metadata: %s', dataset['meta_data'])
  return dataset_utils.Dataset(**dataset)
