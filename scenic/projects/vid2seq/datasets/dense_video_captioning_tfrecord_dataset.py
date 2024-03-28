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

"""TFRecords data-loader for dense video captioning datasets."""

import functools
from typing import Dict, Iterator, List, Optional, Text, Tuple, Union

from absl import logging
from dmvr import tokenizers
from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.projects.t5 import tokenizer as t5_tokenizer
from scenic.projects.vid2seq import data_utils as vid2seq_data_utils
from scenic.projects.vivit.data import video_tfrecord_dataset
import tensorflow as tf


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]


class VTTFRecordDatasetFactory(video_tfrecord_dataset.TFRecordDatasetFactory):
  """Reader for TFRecords using the MediaSequence format.

  The TFRecords already contain features, ASR and captions.

  Attributes:
    base_dir: str. The base directory from which the SSTables are read.
    subset: str. The subset of the dataset. The subsets are are determined by
      the tables dictionary..
  """

  _MODALITIES = ('features', 'text')

  def __init__(self,
               base_dir: str,
               tables: Dict[str, Union[str, List[str]]],
               examples_per_subset: Dict[str, int],
               subset: str = 'train',
               modalities: Tuple[str] = ('rgb',),
               prop_data: float = 1.0,
               num_groups: Optional[int] = None,
               group_index: Optional[int] = None):
    """Initializes the instance of TFRecordDatasetFactory.

    Initializes a data-loader using DeepMind Video Reader (DMVR) pre-processing.
    TFRecords are assumed to consist of tf.SequenceExample protocol buffers in
    the MediaSequence
    (https://github.com/google/mediapipe/tree/master/mediapipe/util/sequence)
    format.

    Args:
      base_dir: The base directory of the TFRecords.
      tables: A dictionary mapping the subset name (train, val or test) to the
        relative path of the SSTable containing them. Follows DMVR convention.
        The values of the dictionary can either be a string or a list. If it is
        a string, it specifies all the shards in the SSTable. Example -
        "/path/to/sstable@10". If passing a list, each entry is a shard of the
        SSTable. Example - "[/path/to/sstable_shard_1_of_10, ...,
        /path/to/sstabble_shard_10_of_10]." The latter scenario is useful for
        debugging.
      examples_per_subset:  A dictionary mapping the subset name (train, val or
        test) to the number of examples in the dataset for that subset.
      subset: The subset of the dataset to load. Must be a key of "tables"
      modalities: Which modality to load. Currently supports 'rgb' and
        'spectrogram'
      prop_data: The proportion of the data to load. If less than 1.0, this
        proportion of the total TFRecord shards are read.
      num_groups: If specified will reshard the data according to `num_groups`.
        A `group_index` should be specified if using `num_groups`.
      group_index: Index of the shard to return after resharding. `num_groups`
        should be specified if using `group_index`. This is useful in
        distributed setting where one wants to ensure that different data is
        read by different workers.
    """

    for modality in modalities:
      if modality not in VTTFRecordDatasetFactory._MODALITIES:
        raise ValueError('Invalid modality %s.' % modality)
    self._modalities = modalities

    super().__init__(
        base_dir=base_dir,
        tables=tables,
        examples_per_subset=examples_per_subset,
        num_classes=1,  # no class for captioning
        subset=subset,
        fraction_data=prop_data,
        num_groups=num_groups,
        group_index=group_index)

  def _build(
      self,
      dataset_configs,  # pytype: disable=signature-mismatch
      is_training: bool = True,
      # Video related parameters.
      num_frames: int = 100,
      stride: int = 1,
      num_test_clips: int = 1,
      max_num_captions: int = 1,
      tokenizer: Optional[tokenizers.TextTokenizer] = None,
      append_eos: bool = True,
      caption_string: str = 'caption/string',
      ):
    """Default build for this dataset.

    Args:
      dataset_configs: dataset configuration.
      is_training: whether or not in training mode.
      num_frames: number of frames per subclip.
      stride: temporal stride to sample frames.
      num_test_clips: number of test clip (1 by default). If more than one, this
        will sample multiple linearly spaced clips within each video at test
        time. If 1, then a single clip in the middle of the video is sampled.
      max_num_captions: Maximum number of captions to keep. If there are more
        captions in the proto, only the first `max_num_captions` will be
        returned is `is_training` is set to `False`. If `is_training` is `True`,
        then `max_num_captions` will be randomly sampled. Finally if the proto
        contains less than `max_num_captions`, we pad with empty srings to make
        sure there are `max_num_captions` in total.
      tokenizer: An instance of a tokenizer.
      append_eos: Whether to append EOS token.
      caption_string: Input feature name in sstable for caption.
    """

    modalities = dataset_configs.get('modalities', ('features',))
    num_frames = dataset_configs.get('num_frames', 100)
    num_bins = dataset_configs.get('num_bins', 100)
    stride = dataset_configs.get('stride', 2)
    max_num_output_words = dataset_configs.get('max_num_output_words', 128)
    max_num_captions = dataset_configs.get('max_num_captions', 1)
    caption_string = dataset_configs.get('caption_string', 'caption/string')
    input_timestamp_name = dataset_configs.get('input_timestamp_name',
                                               'caption/timestamp')
    if 'input_timestamp_start_name' in dataset_configs:
      input_timestamp_start_name = dataset_configs.get(
          'input_timestamp_start_name')
      input_timestamp_end_name = dataset_configs.get('input_timestamp_end_name')
    else:
      input_timestamp_start_name = input_timestamp_name + '/start'
      input_timestamp_end_name = input_timestamp_name + '/end'
    input_duration_name = dataset_configs.get('input_duration_name',
                                              'video/duration')
    output_raw_timestamp_name = dataset_configs.get('output_raw_timestamp_name',
                                                    'timestamp')
    output_raw_duration_name = dataset_configs.get('output_raw_duration_name',
                                                   'duration')
    vocabulary_size = dataset_configs.get('vocabulary_size', 32128)
    input_feature_name = dataset_configs.get('input_feature_name',
                                             'image/clip_embeddings')
    output_raw_feature_name = dataset_configs.get('output_raw_feature_name',
                                                  'features')
    features_dim = dataset_configs.get('features_dim', 768)
    max_events = dataset_configs.get('max_events', 50)
    abs_time_token = dataset_configs.get('abs_time_token', False)
    p = dataset_configs.get('random_temporal_crop_proba', 0.5)
    tmp_only = dataset_configs.get('tmp_only', False)
    split = dataset_configs.get('split', False) and not is_training
    time_format = dataset_configs.get('time_format', 'se')
    order = dataset_configs.get('order', 'ld')
    notime = dataset_configs.get('notime', False)
    preserve = dataset_configs.get('preserve', True)
    asr_input = 'text' in modalities
    corrupt = dataset_configs.get('corrupt', 0.)
    span_len = dataset_configs.get('span_len', 3.)
    max_num_input_words = dataset_configs.get('max_num_input_words', 512)
    asr_notime = dataset_configs.get('asr_notime', False)
    max_segments = dataset_configs.get('max_segments', 0)
    output_raw_string_name = 'caption_strings'
    asr_raw_string_name = 'ASR/string'
    keep_raw_string = not is_training

    # Init the TF models of the tokenizer.
    tokenizer.initialize()  # pytype: disable=attribute-error

    # Visual features
    vid2seq_data_utils.add_embeddings(
        parser_builder=self.parser_builder,
        sampler_builder=self.sampler_builder,
        input_feature_lists_name=input_feature_name,
        output_feature_lists_name=output_raw_feature_name,
        num_frames=num_frames,
        features_dim=features_dim,
        sync_random_state=True,
        output_raw_timestamp_name=output_raw_timestamp_name,
        output_raw_duration_name=output_raw_duration_name,
        is_training=is_training,
        output_raw_string_name=output_raw_string_name,
        p=p,
        t=1000000,  # 1FPS
        preserve=preserve,
        asr_input=asr_input,
        max_segments=max_segments)

    # Prep caption, ASR etc...
    vid2seq_data_utils.add_text_with_timestamps(
        parser_builder=self.parser_builder,
        preprocessor_builder=self.preprocessor_builder,
        tokenizer=tokenizer,
        input_feature_name=caption_string,
        input_timestamp_start_name=input_timestamp_start_name,
        input_timestamp_end_name=input_timestamp_end_name,
        input_duration_name=input_duration_name,
        output_raw_timestamp_name=output_raw_timestamp_name,
        output_raw_duration_name=output_raw_duration_name,
        max_events=max_events,
        vocabulary_size=vocabulary_size,
        num_bins=num_bins,
        output_raw_string_name=output_raw_string_name,
        # Always prepend the BOS token to init the generation, and +1 tokens
        # are loaded, which are then splitted into input and target tokens.
        prepend_bos=True,
        append_eos=append_eos,
        keep_raw_string=keep_raw_string,
        max_num_tokens=max_num_output_words + 1,
        abs_time_token=abs_time_token,
        time_format=time_format,
        order=order,
        notime=notime,
        asr_input=asr_input,
        max_num_input_words=max_num_input_words,
        asr_raw_string_name=asr_raw_string_name,
        corrupt=corrupt,
        span_len=span_len,
        tmp_only=tmp_only,
        asr_notime=asr_notime,
        t=1000000)  # 1 FPS

    if split:
      self.parser_builder.parse_feature(
          feature_name='split',
          feature_type=tf.io.VarLenFeature(dtype=tf.int64),
          output_name='split',
          is_context=True)
      self.decoder_builder.add_fn(
          fn=tf.sparse.to_dense,
          feature_name='split',
          fn_name='split_sparse_to_dense')

    if not is_training:
      self.parser_builder.parse_feature(
          feature_name='videoid',
          feature_type=tf.io.VarLenFeature(dtype=tf.string),
          output_name='videoid',
          is_context=True)
      self.decoder_builder.add_fn(
          fn=tf.sparse.to_dense,
          feature_name='videoid',
          fn_name='videoid_sparse_to_dense')


def load_split_from_dmvr(
    ds_factory,
    dataset_configs,
    batch_size,
    subset='train',
    stride=2,
    num_test_clips=1,
    keep_key=False,
    max_num_captions: int = 1,
    caption_string='caption/string'):
  """Loads dataset using DMVR for pre-processing.

  DMVR dataset loader already does basic augmentation (random crop and flip in
    train mode. It also already shuffles and batches the data.

  Args:
    ds_factory: A DMVR factory to instantiate with the subset.
    dataset_configs: Dataset configurations.
    batch_size: The batch_size to use.
    subset: train, validation or test.
    stride: Temporal stride to sample RGB frames.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggreagated in the batch dimension.
    keep_key: bool; If true, also return the key for each example.
    max_num_captions: Maximum number of captions to keep. If there are more
      captions in the proto, only the first `max_num_captions` will be returned
      is `is_training` is set to `False`. If `is_training` is `True`, then
      `max_num_captions` will be randomly sampled. Finally if the proto contains
      less than `max_num_captions`, we pad with empty srings to make sure there
      are `max_num_captions` in total.
    caption_string: Input feature name in sstable for caption.

  Returns:
    A pair `(ds, num_examples)` with
      ds: A `tf.data.Dataset` object
      num_examples: Number of examples in the dataset.
  """
  # Should hold two fields: tokenizer_type and tokenizer_vocab.
  is_training = subset == 'train'
  tokenizer_config = dataset_configs.get('tokenizer', {})
  tokenizer_type = tokenizer_config.get('tokenizer_type', 'sentence_piece')
  tokenizer_model = tokenizer_config.get('tokenizer_model', None)
  modalities = dataset_configs.get('modalities', ['features'])
  num_frames = dataset_configs.get('num_frames', 100)

  if tokenizer_type == 'sentence_piece':
    if tokenizer_model is not None:
      tokenizer = t5_tokenizer.build_dmvr_sp_model(tokenizer_model)
    else:
      tokenizer = t5_tokenizer.build_dmvr_sp_model()
  else:
    raise NotImplementedError

  ds_factory = ds_factory(
      subset=subset, modalities=modalities).configure(
          dataset_configs=dataset_configs,
          is_training=is_training,
          num_frames=num_frames,
          stride=stride,
          num_test_clips=num_test_clips,
          max_num_captions=max_num_captions,
          tokenizer=tokenizer,
          append_eos=True,
          caption_string=caption_string,
          )

  logging.info('Preprocessing graph: %s',
               ds_factory.preprocessor_builder.get_summary())
  logging.info('Postprocessing graph: %s',
               ds_factory.postprocessor_builder.get_summary())

  num_examples = ds_factory.num_examples

  ds = ds_factory.make_dataset(
      batch_size=batch_size,
      shuffle=is_training,
      num_epochs=None if is_training else 1,
      drop_remainder=is_training,
      keep_key=(not is_training and keep_key))

  if not is_training:
    ds = ds.repeat(None)

  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  return ds, num_examples


def map_input_keys(batch,
                   modalities=('features',),
                   return_as_dict=False):
  """DMVR dataset returns 'image' and 'label'. We want 'inputs' and 'label'."""
  if not return_as_dict:
    if len(modalities) == 1 and modalities[0] == 'features':
      batch['encoder_inputs'] = batch['features']
    elif len(modalities) == 1 and 'text' == modalities[0]:
      batch['encoder_inputs'] = batch['asr_indices']
    else:
      raise NotImplementedError('modality not supported by map_keys.')
  else:
    batch['encoder_inputs'] = {}
    if 'text' in modalities:
      if 'asr_indices' in batch:
        batch['encoder_inputs']['text'] = batch['asr_indices']
    if 'features' in modalities:
      batch['encoder_inputs']['features'] = batch['features']
  return batch


def tile_label_key(batch, return_as_dict=False):
  """Tile labels and keys to match input videos when num_test_clips > 1.

  When multiple test crops are used (ie num_test_clips > 1), the batch dimension
  of batch['inputs'] = test_batch_size * num_test_clips.
  However, labels and keys remain of size [test_batch_size].
  This function repeats label and key to match the inputs.

  Args:
    batch: Batch from iterator
    return_as_dict: Whether to return multimodal inputs as a dictionary.

  Returns:
    batch: Batch with 'label' and 'key' tiled to match 'inputs'.
  """
  if not return_as_dict:
    n_repeats = batch['inputs'].shape[0] // batch['label'].shape[0]
  if 'key' in batch:
    batch['key'] = np.repeat(batch['key'], n_repeats, axis=0)
  return batch


def get_dataset(
    *,
    batch_size,
    eval_batch_size,
    num_shards,
    dtype_str='float32',
    shuffle_seed=0,  # pylint:disable=unused-argument
    rng=None,
    dataset_configs=None,
    dataset_service_address: Optional[str] = None):
  """Returns a generator for the audiovisual dataset."""
  del rng
  dataset_configs = dataset_configs or {}
  modalities = dataset_configs.get('modalities', ['features'])
  return_as_dict = dataset_configs.get('return_as_dict', False)
  num_frames = dataset_configs.get('num_frames', 100)
  stride = dataset_configs.get('stride', 2)
  eval_stride = dataset_configs.get('eval_stride', 2)
  augmentation_params = dataset_configs.get('augmentation_params', None)
  num_train_clips = dataset_configs.get('num_train_clips', 1)
  num_eval_clips = dataset_configs.get('num_eval_clips', 1)
  max_num_output_words = dataset_configs.get('max_num_output_words', 128)
  max_num_captions = dataset_configs.get('max_num_captions', 1)
  caption_string = dataset_configs.get('caption_string', 'caption/string')
  train_caption_string = dataset_configs.get('train_caption_string',
                                             'caption/string')
  num_train_captions_per_clip = dataset_configs.get(
      'num_train_captions_per_clip', 1)
  features_dim = dataset_configs.get('features_dim', 768)

  max_num_input_words = dataset_configs.get('max_num_input_words', 512)

  def validate_config(field):
    if dataset_configs.get(field) is None:
      raise ValueError(f'{field} must be specified for TFRecord dataset.')
  validate_config('base_dir')
  validate_config('tables')
  validate_config('examples_per_subset')

  ds_factory = functools.partial(
      VTTFRecordDatasetFactory,
      base_dir=dataset_configs.get('base_dir'),  # pytype: disable=attribute-error
      tables=dict(dataset_configs.get('tables')),  # pytype: disable=attribute-error
      examples_per_subset=dataset_configs.get('examples_per_subset'),  # pytype: disable=attribute-error
      num_groups=jax.process_count(),
      group_index=jax.process_index())

  def create_dataset_iterator(
      subset: Text,
      batch_size_local: int,
      num_clips: int,
      caption_string: str,
      stride: int,
      keep_key_local: bool = False,
      max_num_captions_local: int = 1) -> Tuple[Iterator[Batch], int]:

    is_training = subset == 'train'
    logging.info('Loading split %s', subset)

    dataset, num_examples = load_split_from_dmvr(
        ds_factory,
        dataset_configs=dataset_configs,
        batch_size=batch_size_local,
        subset=subset,
        stride=stride,
        num_test_clips=num_clips,
        keep_key=keep_key_local,
        max_num_captions=max_num_captions_local,
        caption_string=caption_string)

    if dataset_service_address and is_training:
      if shuffle_seed is not None:
        raise ValueError('Using dataset service with a random seed causes each '
                         'worker to produce exactly the same data. Add '
                         'config.shuffle_seed = None to your config if you '
                         'want to run with dataset service.')
      logging.info('Using the tf.data service at %s', dataset_service_address)
      dataset = dataset_utils.distribute(dataset, dataset_service_address)

    maybe_pad_batches = functools.partial(
        dataset_utils.maybe_pad_batch,
        train=is_training,
        batch_size=batch_size_local,
        inputs_key=None)
    shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
    current_iter = iter(dataset)
    current_iter = map(dataset_utils.tf_to_numpy, current_iter)
    current_iter = map(
        functools.partial(
            map_input_keys,
            modalities=modalities,
            return_as_dict=return_as_dict), current_iter)
    current_iter = map(maybe_pad_batches, current_iter)

    if augmentation_params and augmentation_params.get('do_mixup', False):
      raise ValueError('mixup should be done in the trainer.')

    current_iter = map(shard_batches, current_iter)

    if is_training and dataset_configs.get('prefetch_to_device'):
      # Async bind batch to device which speeds up training.
      current_iter = jax_utils.prefetch_to_device(
          current_iter, dataset_configs.get('prefetch_to_device'))

    return current_iter, num_examples

  train_iter, n_train_examples = create_dataset_iterator(
      'train', batch_size, num_train_clips, train_caption_string, stride)

  eval_iters = {}
  num_eval_examples = {}
  for k in dataset_configs.tables.keys():  # pytype: disable=attribute-error
    if k == 'train':
      continue
    eval_iter, n_eval_examples = create_dataset_iterator(
        k, eval_batch_size, num_eval_clips, caption_string, eval_stride, True,
        max_num_captions)
    eval_iters[k] = eval_iter
    num_eval_examples[k] = n_eval_examples * num_eval_clips

  meta_data = {
      # The training loop iterates each caption sample individually. Therefore,
      # the number of captions per clip is muliplied for computing the number of
      # training examples. In contrast, for the eval sets, all GT captions are
      # used together for evaluating a single clip.
      'num_train_examples':
          (n_train_examples * num_train_clips * num_train_captions_per_clip),
      'num_eval_examples': num_eval_examples,
      'encoder_input_dtype': getattr(jnp, dtype_str),
      'encoder_input_text_dtype': jnp.int32,
      # The sahpes below is for the train model where the number of captions is
      # set to 1 and the dimension 1 corresponding to the number of captions is
      # squeezed.
      'decoder_input_shape': {
          'decoder_input_tokens': (-1, max_num_output_words),
          'decoder_target_tokens': (-1, max_num_output_words),
      },
      'decoder_input_dtype': jnp.int32
  }

  if return_as_dict:
    input_shape_dict = {
        'text': (-1, max_num_input_words),
        'features': (-1, num_frames, features_dim)
    }

    meta_data['encoder_input_shape'] = {
        m: input_shape_dict[m] for m in modalities}
  elif len(modalities) == 1 and modalities[0] == 'features':
    meta_data['encoder_input_shape'] = (-1, num_frames, features_dim)
  elif len(modalities) == 1 and modalities[0] == 'text':
    meta_data['encoder_input_shape'] = (-1, max_num_input_words)
  else:
    raise NotImplementedError('modality not supported')
  logging.info('Dataset metadata:\n%s', meta_data)

  return dataset_utils.Dataset(train_iter, eval_iters, None, meta_data)


def get_datasets(config,
                 data_rng: jnp.ndarray,
                 dataset_service_address: Optional[str] = None):
  """Creates dataset from config."""

  device_count = jax.device_count()
  logging.info('device_count: %d', device_count)
  logging.info('num_hosts : %d', jax.process_count())
  logging.info('host_id : %d', jax.process_index())

  dataset_dict = {}
  for ds_name, cfg in config.datasets.items():

    if config.get('batch_sizes') is not None:
      batch_size = config.batch_sizes.get(ds_name)
    else:
      batch_size = config.batch_size

    if batch_size % device_count > 0:
      raise ValueError(
          f'Batch size ({batch_size}) of {ds_name} must be divisible '
          f'by the number of devices ({device_count})')

    if config.get('eval_batch_sizes') is not None:
      eval_batch_size = config.eval_batch_sizes.get(ds_name)
    else:
      eval_batch_size = config.get('eval_batch_size', batch_size)

    if eval_batch_size % device_count > 0:
      raise ValueError(
          f'Eval batch size ({eval_batch_size}) of {ds_name} must be '
          f'divisible by the number of devices ({device_count})')

    local_batch_size = batch_size // jax.process_count()
    eval_local_batch_size = eval_batch_size // jax.process_count()
    device_batch_size = batch_size // device_count
    logging.info('local_batch_size of %s : %d', ds_name, local_batch_size)
    logging.info('device_batch_size of %s : %d', ds_name, device_batch_size)

    shuffle_seed = cfg.get('shuffle_seed', None)
    if dataset_service_address and shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you want '
                       'to run with dataset service.')

    # 'bit' consists of many datasets, so we do this to have a unique dataset
    # key if we train on multiple datasets from 'bit'. E.g. ds_name =
    # 'bit_caltech'.

    dataset_rng, data_rng = jax.random.split(data_rng)
    ds = get_dataset(
        batch_size=local_batch_size,
        eval_batch_size=eval_local_batch_size,
        num_shards=jax.local_device_count(),
        dtype_str='float32',
        rng=dataset_rng,
        shuffle_seed=shuffle_seed,
        dataset_configs=cfg,
        dataset_service_address=dataset_service_address)

    # Add task information to the dataset meta_data:
    dataset_dict[ds_name] = ds

  return dataset_dict
