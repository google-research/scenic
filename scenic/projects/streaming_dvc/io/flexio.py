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

"""FlexIO input pipeline for TFRecord sources."""

import functools
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
from clu import preprocess_spec
import jax
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.dataset_lib.flexio import flexio
import tensorflow as tf

TfFeature = flexio.TfFeature


def _get_feature(feature_type: str, shape=(), dtype='string') -> TfFeature:
  dtype = tf.dtypes.as_dtype(dtype)
  if feature_type == 'FixedLen':
    return tf.io.FixedLenFeature(shape=shape, dtype=dtype)
  if feature_type == 'VarLen':
    return tf.io.VarLenFeature(dtype=dtype)
  elif feature_type == 'FixedLenSequence':
    return tf.io.FixedLenSequenceFeature(shape=shape, dtype=dtype)
  raise NotImplementedError(f'Feature type {feature_type} not available yet.')


def get_number_of_examples(config: ml_collections.ConfigDict) -> int:
  """Obtain the number of examples in a TFRecord dataset."""
  if hasattr(config, 'num_examples'):
    return config.num_examples

  if config.source == 'tfrecord':
    size = config.get('size', None)
    if size is None:
      raise ValueError('size is required for tfrecord datasets')
    return size
  raise ValueError(f'Unknown data source: {config.source}')


def decode_sharded_names(path):
  """Convert sharded file names into a list."""
  ret = []
  path = path.split(',')
  for name in path:
    if '@' in name:
      num_shards = int(name.split('@')[1].split('.')[0])
      suffix = name.split(f'@{num_shards}')[-1]
      prefix = name.split('@')[0]
      names = [
          f'{prefix}-{i:05d}-of-{num_shards:05d}{suffix}'
          for i in range(num_shards)
      ]
      ret.extend(names)
    else:
      ret.append(name)
  return ret


def _get_single_tfrecord_dataset(
    tfrecords: Union[str, Sequence[str]],
    context_features: Mapping[str, TfFeature],
    sequence_features: Mapping[str, TfFeature],
    batch_size: Optional[int],
    preprocess_fn: Optional[preprocess_spec.PreprocessFn] = None,
    postprocess_fn: Optional[preprocess_spec.PreprocessFn] = None,
    rng: Union[None, jnp.ndarray, tf.Tensor] = None,
    global_rng: Union[None, jnp.ndarray, tf.Tensor] = None,
    shuffle: bool = False,
    shuffle_buffer_size: int = 1000,
    cache: bool = False,
    repeat_dataset: bool = True,
) -> tf.data.Dataset:
  """Creates dataset using DMVR and applies preprocessing.

  Args:
    tfrecords: Path to tfrecords.
    context_features: Dictionary of context features to parse.
    sequence_features: Dictionary of sequence features to parse.
    batch_size: Batch size.
    preprocess_fn: Preprocess function with pre-caching ops.
    postprocess_fn: Postprocess function executed *after* batching.
    rng: Random seed.
    global_rng: Global random seed (same across hosts).
    shuffle: Whether to shuffle.
    shuffle_buffer_size: Shuffle buffer size.
    cache: Whether to cache the dataset.
    repeat_dataset: If True, the dataset is repeated indefinitely.

  Returns:
    tf.data.Dataset with preprocessing applied.
  """
  del global_rng

  if rng is None and shuffle:
    raise ValueError("Please set 'rng' when shuffling.")

  ds = tf.data.TFRecordDataset(tfrecords)
  # Split datasets into machines. Otherwise multi-machine evaluation takes the
  # same images.
  ds = ds.shard(jax.process_count(), jax.process_index())
  if sequence_features:
    # pylint: disable=g-long-lambda
    ds = ds.map(
        lambda x: tf.io.parse_single_sequence_example(
            x, context_features, sequence_features
        )
    )
    # merge two into one
    ds = ds.map(lambda x, y: {**x, **y})
    # pylint: enable=g-long-lambda
  else:
    ds = ds.map(lambda x: tf.io.parse_single_example(x, context_features))

  if cache:
    # Caching is done after pre-processing. This means that only deterministic
    # pre-processing should be used here. This includes things like frame
    # sampling, JPEG decoding, etc.
    ds = ds.cache()
  if repeat_dataset:
    ds = ds.repeat()  # Repeat indefinitly.
  if shuffle:
    ds = ds.shuffle(shuffle_buffer_size, seed=rng[0])
  if preprocess_fn is not None:
    if rng is not None:
      ds = flexio.apply_process_fn_with_populated_seed(
          ds, preprocess_fn, rng=rng)
    else:
      ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
  if batch_size:
    ds = ds.batch(
        batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
  if postprocess_fn is not None:
    if rng is not None:
      ds = flexio.apply_process_fn_with_populated_seed(
          ds, postprocess_fn, rng=rng)
    else:
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
  """Build a tf.data.Dataset pipeline using clu.deterministic_data.

  Different from the original flexio, this function only support TFRecord.

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
  pp_libs = dataset_configs.get('pp_libs', flexio.DEFAULT_PP_LIBS)
  process_fn = functools.partial(flexio.get_process_fn, pp_libs=pp_libs)

  if split not in dataset_configs:
    return None

  mode_config = dataset_configs.get(split)
  config = ml_collections.ConfigDict({**dataset_configs, **mode_config})

  if len(config.sources) > 1:
    merge_sources = config.merge_sources
  else:
    merge_sources = True

  del start_step

  sources, weights = {}, {}
  for src_id, src in enumerate(config.sources):
    src_name = src.get('name', f'src_{src_id}')
    if rng is not None:
      rng, ds_rng = jax.random.split(rng)
    else:
      ds_rng = None

    if src.source == 'tfrecord':
      context_features = dict(src.get('context_features', {}))
      sequence_features = dict(src.get('sequence_features', {}))
      context_features = {
          k: _get_feature(**f) for k, f in context_features.items()
      }
      sequence_features = {
          k: _get_feature(**f) for k, f in sequence_features.items()
      }
      tfrecord_path = decode_sharded_names(src.tfrecords)
      ds = _get_single_tfrecord_dataset(
          tfrecord_path,
          context_features,
          sequence_features,
          batch_size=src.get('batch_size'),
          preprocess_fn=process_fn(src.get('preproc_spec') or ''),
          postprocess_fn=process_fn(src.get('postproc_spec') or ''),
          rng=ds_rng,
          global_rng=global_rng,
          shuffle=shuffle,
          shuffle_buffer_size=src.shuffle_buffer_size,
          cache=src.get('cache', False),
          repeat_dataset=src.get('repeat_dataset', True),
      )
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
        return flexio.apply_process_fn_with_populated_seed(
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

  Different from the original flexio, this function uses a custom
  get_number_of_examples funtion for TFRecord.

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
        lambda x: (tuple(x.shape.as_list()[1:]), flexio.tf2jax_dtype(x.dtype)),
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


@datasets.add_dataset('flexio_tfrecord')
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
  unused_rng, test_rng = jax.random.split(rng)
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
