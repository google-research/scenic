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

"""TFRecords data-loader to read video-text datasets."""
import functools
from typing import Dict, Iterator, List, Optional, Text, Tuple, Union

from absl import logging
from dmvr import modalities as load_modalities
from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.dataset_lib import video_ops
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
from scenic.projects.vivit.data import video_tfrecord_dataset
import tensorflow as tf
from dmvr import tokenizers

Batch = Dict[str, jnp.ndarray]


def maybe_pad_batch(batch, train, batch_size, num_clips, num_captions):
  """Zero pad the batch on the right to the batch_size."""
  assert 'batch_mask' not in batch
  if 'rgb' in batch['inputs']:
    unpadded_mask_shape = batch['inputs']['rgb'].shape[0]
    batch_pad = batch_size - unpadded_mask_shape
  else:
    raise ValueError('invalid input batch')
  if train and batch_pad != 0:
    raise ValueError('In this codebase, we assumed that we always drop the '
                     'last partial batch of the train set. Please use '
                     '` drop_remainder=True` for the training set.')

  # Most batches will not need padding so we quickly return to avoid slowdown.
  if train or batch_pad == 0:
    if 'batch_mask' not in batch:
      batch['batch_mask'] = np.ones(unpadded_mask_shape, dtype=np.float32)
    return batch

  def zero_pad(array, batch_pad=batch_pad):
    pad_with = [(0, batch_pad)] + [(0, 0)] * (array.ndim - 1)
    return np.pad(array, pad_with, mode='constant')

  # For the test set, we only keep one text input and one key for each video
  text_indices = text_mask = keys = labels = verbs = verb_indices = None
  pad_text_mask = False
  pad_keys = False
  pad_labels = False
  pad_verbs = False
  pad_verb_indices = False
  if not train and batch_pad != 0:
    text_indices = batch['text_indices']
    batch_pad_text = (batch_pad * num_captions) // num_clips
    logging.info('Batch pad text in the test loop is %s', batch_pad_text)
    text_indices = zero_pad(text_indices, batch_pad_text)
    del batch['text_indices']
    if 'text_mask' in batch:
      text_mask = batch.pop('text_mask')
      text_mask = zero_pad(text_mask, batch_pad_text)
      pad_text_mask = True
    if 'key' in batch:
      keys = batch['key']
      pad_keys = True
      del batch['key']
      batch_pad_key = batch_pad // num_clips
      keys = zero_pad(keys, batch_pad_key)
    if 'label' in batch:
      labels = batch['label']
      pad_labels = True
      del batch['label']
      batch_pad_key = batch_pad // num_clips
      labels = zero_pad(labels, batch_pad_key)
    if 'verb' in batch:
      verbs = batch['verb']
      pad_verbs = True
      del batch['verb']
      batch_pad_verb = batch_pad // num_clips
      verbs = zero_pad(verbs, batch_pad_verb)
    if 'verb_indices' in batch:
      verb_indices = batch['verb_indices']
      pad_verb_indices = True
      del batch['verb_indices']
      batch_pad_verb_indices = batch_pad // num_clips
      verb_indices = zero_pad(verb_indices, batch_pad_verb_indices)

  # Note here, batch_pad is repeated num_clips times if there are multiple clips
  # sampled per video
  padded_batch = jax.tree_util.tree_map(zero_pad, batch)
  padded_batch_mask = zero_pad(
      np.ones(unpadded_mask_shape, dtype=np.float32), batch_pad)
  padded_batch['batch_mask'] = padded_batch_mask
  if not train and batch_pad != 0:
    padded_batch['text_indices'] = text_indices
    if pad_text_mask:
      padded_batch['text_mask'] = text_mask
    if pad_keys:
      padded_batch['key'] = keys
    if pad_labels:
      padded_batch['label'] = labels
    if pad_verbs:
      padded_batch['verb'] = verbs
    if pad_verb_indices:
      padded_batch['verb_indices'] = verb_indices
  return padded_batch


class AVTFRecordDatasetFactory(video_tfrecord_dataset.TFRecordDatasetFactory):
  """Reader for TFRecords using the MediaSequence format."""

  def __init__(self,
               base_dir: str,
               tables: Dict[str, Union[str, List[str]]],
               examples_per_subset: Dict[str, int],
               subset: str = 'train',
               prop_data: float = 1.0,
               num_groups: Optional[int] = None,
               group_index: Optional[int] = None):
    """Initializes the instance of TFRecordDatasetFactory.

    Initializes a data-loader using DeepMind Video Reader (DMVR) pre-processing
    (https://github.com/deepmind/dmvr).
    TFRecords are assumed to consist of tf.SequenceExample protocol buffers in
    the MediaSequence
    (https://github.com/google/mediapipe/tree/master/mediapipe/util/sequence)

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
      prop_data: The proportion of the data to load. If less than 1.0, this
        proportion of the total TFRecord shards are read.
      num_groups: If specified will reshard the data according to `num_groups`.
        A `group_index` should be specified if using `num_groups`.
      group_index: Index of the shard to return after resharding. `num_groups`
        should be specified if using `group_index`. This is useful in
        distributed setting where one wants to ensure that different data is
        read by different workers.
    """

    self._modalities = ('rgb',)
    super().__init__(
        base_dir=base_dir,
        tables=tables,
        num_classes=0,  # non applicable
        examples_per_subset=examples_per_subset,
        subset=subset,
        fraction_data=prop_data,
        num_groups=num_groups,
        group_index=group_index)

  def _build(self,
             is_training: bool = True,
             # Video related parameters.
             num_frames: int = 32,
             stride: int = 1,
             num_test_clips: int = 1,
             min_resize: int = 256,
             crop_size: int = 224,
             zero_centering_image: bool = False,
             # Text related parameters.
             max_num_words: int = 16,
             max_num_captions: int = 1,
             caption_string: str = 'caption/string',
             num_labels: int = 0,
             include_verb: bool = False,):
    """Adds DMVR pre-processors to the dataset.

    Args:
      is_training: whether or not in training mode.
      num_frames: number of frames per subclip.
      stride: temporal stride to sample frames.
      num_test_clips: number of test clip (1 by default). If more than one, this
        will sample multiple linearly spaced clips within each video at test
        time. If 1, then a single clip in the middle of the video is sampled.
      min_resize: frames are resized so that min width/height is min_resize.
      crop_size: final size of the frame after cropping the resized frames.
      zero_centering_image: whether to have images between [-1, 1] or [0, 1].
      max_num_words: Maximum number of tokens to keep from the text for each
        caption. If there are more tokens, sequence is cropped, if less, the
        caption is padded using the tokenizer pad id.
      max_num_captions: Maximum number of captions to keep. If there are more
        captions in the proto, only the first `max_num_captions` will be
        returned is `is_training` is set to `False`. If `is_training` is `True`,
        then `max_num_captions` will be randomly sampled. Finally if the proto
        contains less than `max_num_captions`, we pad with empty srings to make
        sure there are `max_num_captions` in total.
      caption_string: Input feature name in sstable for caption.
      num_labels: Number of labels (classification).
      include_verb: Including an additional contrastive video-verb phrase loss
        (where the verb is extracted from the caption using PaLM).
    """
    if 'rgb' in self._modalities:
      load_modalities.add_image(
          parser_builder=self.parser_builder,
          sampler_builder=self.sampler_builder,
          decoder_builder=self.decoder_builder,
          preprocessor_builder=self.preprocessor_builder,
          postprocessor_builder=self.postprocessor_builder,
          is_training=is_training,
          num_frames=num_frames,
          stride=stride,
          num_test_clips=num_test_clips,
          min_resize=min_resize,
          crop_size=crop_size,
          zero_centering_image=zero_centering_image,
          sync_random_state=True)
    tokenizer = tokenizers.ClipTokenizer()  # dummy tokenizer
    tokenizer.initialize()
    # Note, output feature name is 'text_indices'
    if ';' in caption_string:
      # this is when we have both positives & hard negatives
      pos_neg_max_captions = [1, max_num_captions-1]
      pos_neg_output_raw_string_name = ['string_pos', 'string_neg']
      pos_neg_output_feature_name = ['text_indices_pos', 'text_indices_neg']
      for idx, caption_to_add in enumerate(caption_string.split(';')):
        load_modalities.add_text(
            parser_builder=self.parser_builder,
            decoder_builder=self.decoder_builder,
            preprocessor_builder=self.preprocessor_builder,
            input_feature_name=caption_to_add,
            max_num_captions=pos_neg_max_captions[idx],
            max_num_tokens=max_num_words,
            keep_raw_string=True,
            tokenizer=tokenizer,  # We do not use this tokenizer.
            is_training=is_training,
            output_raw_string_name=pos_neg_output_raw_string_name[idx],
            output_feature_name=pos_neg_output_feature_name[idx],
        )
    else:
      load_modalities.add_text(
          parser_builder=self.parser_builder,
          decoder_builder=self.decoder_builder,
          preprocessor_builder=self.preprocessor_builder,
          input_feature_name=caption_string,
          max_num_captions=max_num_captions,
          max_num_tokens=max_num_words,
          keep_raw_string=True,
          tokenizer=tokenizer,  # We do not use this tokenizer.
          is_training=is_training)
    if include_verb:
      load_modalities.add_text(
          parser_builder=self.parser_builder,
          decoder_builder=self.decoder_builder,
          preprocessor_builder=self.preprocessor_builder,
          input_feature_name='verb',
          max_num_captions=1,
          max_num_tokens=max_num_words,
          keep_raw_string=True,
          tokenizer=tokenizer,  # We do not use this tokenizer.
          is_training=is_training,
          output_raw_string_name='verb',
          output_feature_name='verb_indices',)


def load_split_from_dmvr(ds_factory,
                         batch_size,
                         subset='train',
                         num_frames=32,
                         stride=2,
                         num_test_clips=1,
                         min_resize=256,
                         crop_size=224,
                         zero_centering=True,
                         augmentation_params=None,
                         keep_key=False,
                         max_num_words: int = 16,
                         max_num_captions: int = 1,
                         caption_string='caption/string',
                         num_labels=0,
                         include_verb: bool = False,):
  """Loads dataset using DMVR for pre-processing.

  DMVR dataset loader already does basic augmentation (random crop and flip in
    train mode. It also already shuffles and batches the data.

  Args:
    ds_factory: A DMVR factory to instantiate with the subset.
    batch_size: The batch_size to use.
    subset: train, validation or test.
    num_frames: Number of RGB frames per subclip.
    stride: Temporal stride to sample RGB frames.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggreagated in the batch dimension.
    min_resize: Frames are resized so that min(height, width) is min_resize.
    crop_size: Final size of the frame after cropping the resized frames. Both
      height and width are the same.
    zero_centering: If True, frames are normalized to values in [-1, 1]. If
      False, values in [0, 1].
    augmentation_params: dict; augmentation configurations in train mode.
    keep_key: bool; If true, also return the key for each example.
    max_num_words: Maximum number of tokens to keep from the text for each
      caption. If there are more tokens, sequence is cropped, if less, the
      caption is padded using the tokenizer pad id.
    max_num_captions: Maximum number of captions to keep. If there are more
      captions in the proto, only the first `max_num_captions` will be returned
      is `is_training` is set to `False`. If `is_training` is `True`, then
      `max_num_captions` will be randomly sampled. Finally if the proto contains
      less than `max_num_captions`, we pad with empty srings to make sure there
      are `max_num_captions` in total.
    caption_string: Input feature name in sstable for caption.
    num_labels: Number of labels (classification).
    include_verb: verb phrase loss.

  Returns:
    A pair `(ds, num_examples)` with
      ds: A `tf.data.Dataset` object
      num_examples: Number of examples in the dataset.
  """
  is_training = (subset == 'train')
  ds_factory = ds_factory(subset=subset).configure(
      is_training=is_training,
      num_frames=num_frames,
      stride=stride,
      num_test_clips=num_test_clips,
      min_resize=min_resize,
      crop_size=crop_size,
      zero_centering_image=zero_centering,
      max_num_words=max_num_words,
      max_num_captions=max_num_captions,
      caption_string=caption_string,
      num_labels=num_labels,
      include_verb=include_verb,
      )

  if is_training and augmentation_params:
    # additional augmentation for the RGB features.
    ds_factory = video_ops.additional_augmentations(ds_factory,
                                                    augmentation_params,
                                                    crop_size, num_frames,
                                                    zero_centering)

  logging.info('Preprocessing graph: %s',
               ds_factory.preprocessor_builder.get_summary())
  logging.info('Postprocessing graph: %s',
               ds_factory.postprocessor_builder.get_summary())
  num_examples = ds_factory.num_examples
  ds = ds_factory.make_dataset(batch_size=batch_size,
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


def map_keys(batch, modalities=('rgb')):
  """DMVR dataset returns 'image' and 'label'. We want 'inputs' and 'label'."""
  batch['inputs'] = {}
  if 'rgb' in modalities:
    batch['inputs']['rgb'] = batch['image']
  return batch


def clip_tokenize(batch, rmv_full_stop,
                  vqa_options_for_ce=False,
                  num_responses=5, include_verb=False):
  """Tokenize the raw string using CLIP tokenizers.

  Args:
    batch: batch
    rmv_full_stop: remove full stop (they occur in hard negs)
    vqa_options_for_ce: having cross entropy loss
    num_responses: default 5 as one correct, 4 wrong in MSR-VTT MC.
    include_verb: whether the verb phrase should be included.

  Returns:
    batch: batch
  """

  if 'text' not in batch:
    pos_strings = batch['string_pos'].reshape(
        (-1,) + batch['string_pos'].shape[2:])
    neg_strings = batch['string_neg'].reshape(
        (-1,) + batch['string_neg'].shape[2:])
    num_hard_neg = batch['string_neg'].shape[1]
    raw_text = []
    for idx in range(len(pos_strings)):
      raw_text.append(pos_strings[idx])
      raw_text += list(neg_strings[
          idx*num_hard_neg:(idx*num_hard_neg)+num_hard_neg])
    del batch['text_indices_pos']
    del batch['text_indices_neg']
    del batch['string_neg']
    del batch['string_pos']
  else:
    raw_text = batch['text']
    raw_text = raw_text.reshape((-1,) + raw_text.shape[2:])
    del batch['text']
  raw_text = [sentence.decode('utf-8') for sentence in raw_text]
  verbs, raw_verbs = False, False
  if include_verb and 'verb' in batch:
    verbs = True
    raw_verbs = batch['verb']
    raw_verbs = raw_verbs.reshape((-1,) + raw_verbs.shape[2:])
    del batch['verb']
    raw_verbs = [verb.decode('utf-8') for verb in raw_verbs]
  if rmv_full_stop:
    raw_text = [sent.replace('.', '') for sent in raw_text]
  if vqa_options_for_ce:
    clean_raw_text = []
    for i in raw_text:
      if i:
        options = i.split(';')[:num_responses]
        # a few examples have only four answers
        if len(options) == 4:
          options += ['']
        clean_raw_text.append(options)
    raw_text = np.concatenate(np.asarray(clean_raw_text))
  tokenizer = clip_tokenizer.build_tokenizer(truncate=True)
  text_tokens = tokenizer(raw_text)
  text_tokens = jnp.asarray(text_tokens)
  text_tokens = jnp.expand_dims(text_tokens, 1)
  text_masking = [int(not i) for i in raw_text]
  text_masking = jnp.asarray(text_masking)
  text_masking = jnp.expand_dims(text_masking, 1)
  batch['text_mask'] = text_masking
  batch['text_indices'] = text_tokens
  if include_verb and verbs:
    verb_tokens = tokenizer(raw_verbs)
    verb_tokens = jnp.asarray(verb_tokens)
    verb_tokens = jnp.expand_dims(verb_tokens, 1)
    verb_masking = [int(bool(i)) for i in raw_verbs]
    verb_masking = jnp.asarray(verb_masking)
    verb_masking = jnp.expand_dims(verb_masking, 1)
    batch['verb_mask'] = verb_masking
    batch['verb_indices'] = verb_tokens
  return batch


@datasets.add_dataset('verbs_in_action_tfrecord_dataset')
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
  """Returns a generator for the video-text dataset."""
  del rng
  dataset_configs = dataset_configs or {}
  # RGB related configs.
  num_frames = dataset_configs.get('num_frames', 32)
  stride = dataset_configs.get('stride', 2)
  test_stride = dataset_configs.get('test_stride', 2)
  min_resize = dataset_configs.get('min_resize', 256)
  crop_size = dataset_configs.get('crop_size', 224)
  # General configs.
  num_test_clips = dataset_configs.get('num_test_clips', 1)
  num_test_captions = dataset_configs.get('num_test_captions', 1)
  num_train_val_clips = dataset_configs.get('num_train_val_clips', 1)
  num_train_val_captions = dataset_configs.get('num_train_val_captions', 0)
  if num_train_val_captions:
    num_train_captions = num_val_captions = num_train_val_captions
  else:
    num_train_captions = dataset_configs.get('num_train_captions', 1)
    num_val_captions = dataset_configs.get('num_val_captions', 1)
  zero_centering = dataset_configs.get('zero_centering', True)
  augmentation_params = dataset_configs.get('augmentation_params', None)
  # Whether to load the key for each test sample from the data sstables.
  keep_test_key = dataset_configs.get('keep_test_key', False)
  test_split = dataset_configs.get('test_split')
  # For the test set, the actual batch size is test_batch_size * num_test_clips
  test_batch_size = dataset_configs.get('test_batch_size', eval_batch_size)
  # Text related configs.
  max_num_words = dataset_configs.get('max_num_words', 16)
  max_num_captions = dataset_configs.get('max_num_captions', 0)
  if max_num_captions > 0:
    num_train_captions = num_val_captions = num_test_captions = max_num_captions
  caption_string = dataset_configs.get('caption_string', 'caption/string')
  num_labels = dataset_configs.get('num_labels', 0)
  caption_string_train = dataset_configs.get('caption_string_train',
                                             'clip/label/string')
  rmv_full_stop = dataset_configs.get('rmv_full_stop', False)
  vqa_options_for_ce = dataset_configs.get('vqa_options_for_ce', False)
  include_verb = dataset_configs.get('include_verb', False)

  def validate_config(field):
    if dataset_configs.get(field) is None:
      raise ValueError(f'{field} must be specified for TFRecord dataset.')
  validate_config('base_dir')
  validate_config('tables')
  validate_config('examples_per_subset')

  ds_factory = functools.partial(
      AVTFRecordDatasetFactory,
      base_dir=dataset_configs.get('base_dir', ''),
      tables=dataset_configs.get('tables', {}),
      examples_per_subset=dataset_configs.get('examples_per_subset'),
      num_groups=jax.process_count(),
      group_index=jax.process_index())

  def create_dataset_iterator(
      subset: Text,
      batch_size_local: int,
      num_clips: int,
      num_captions: int,
      caption_string: str,
      stride: int,
      rmv_full_stop: bool,
      num_labels: int,
      include_verb: bool,
      keep_key_local: bool = False,) -> Tuple[Iterator[Batch], int]:

    is_training = subset == 'train'
    is_test = subset == 'test'
    logging.info('Loading split %s', subset)

    dataset, num_examples = load_split_from_dmvr(
        ds_factory,
        batch_size=batch_size_local,
        subset=subset,
        num_frames=num_frames,
        stride=stride,
        num_test_clips=num_clips,
        min_resize=min_resize,
        crop_size=crop_size,
        zero_centering=zero_centering,
        augmentation_params=augmentation_params,
        keep_key=keep_key_local,
        max_num_words=max_num_words,
        max_num_captions=num_captions,
        caption_string=caption_string,
        num_labels=num_labels,
        include_verb=include_verb)

    if dataset_service_address and is_training:
      if shuffle_seed is not None:
        raise ValueError('Using dataset service with a random seed causes each '
                         'worker to produce exactly the same data. Add '
                         'config.shuffle_seed = None to your config if you '
                         'want to run with dataset service.')
      logging.info('Using the tf.data service at %s', dataset_service_address)
      dataset = dataset_utils.distribute(dataset, dataset_service_address)

    current_iter = iter(dataset)
    current_iter = map(dataset_utils.tf_to_numpy, current_iter)
    current_iter = map(map_keys, current_iter)
    current_iter = map(functools.partial(
        clip_tokenize, rmv_full_stop=rmv_full_stop,
        vqa_options_for_ce=vqa_options_for_ce,
        include_verb=include_verb), current_iter)

    pad_batch_size = batch_size_local
    if is_test:
      pad_batch_size = batch_size_local * num_clips
    maybe_pad_batches = functools.partial(
        maybe_pad_batch,
        train=is_training,
        batch_size=pad_batch_size,
        num_clips=num_clips,
        num_captions=num_captions)
    current_iter = map(maybe_pad_batches, current_iter)

    shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
    current_iter = map(shard_batches, current_iter)

    if is_training and dataset_configs.get('prefetch_to_device'):
      # Async bind batch to device which speeds up training.
      current_iter = jax_utils.prefetch_to_device(
          current_iter, dataset_configs.get('prefetch_to_device'))

    return current_iter, num_examples

  train_iter, n_train_examples = create_dataset_iterator(
      'train', batch_size, num_train_val_clips, num_train_captions,
      caption_string_train, stride,
      rmv_full_stop, num_labels, include_verb)
  eval_iter, n_eval_examples = create_dataset_iterator(
      'validation', eval_batch_size, num_train_val_clips, num_val_captions,
      caption_string, test_stride, rmv_full_stop,
      num_labels, False, keep_test_key)
  n_test_examples, test_iter = 0, None
  if test_split:
    test_iter, n_test_examples = create_dataset_iterator(
        test_split, test_batch_size, num_test_clips, num_test_captions,
        caption_string, test_stride,
        rmv_full_stop, num_labels, False, keep_test_key)

  meta_data = {
      'num_train_examples': (n_train_examples * num_train_val_clips),
      'num_eval_examples': (n_eval_examples * num_train_val_clips),
      'num_test_examples': (n_test_examples * num_test_clips),
      'input_dtype': getattr(jnp, dtype_str)
  }
  meta_data['text_shape'] = [-1, max_num_captions, max_num_words]
  meta_data['text_dtype'] = jnp.int32
  meta_data['input_shape'] = {
      'rgb': (-1, num_frames, crop_size, crop_size, 3),
  }
  logging.info('Dataset metadata:\n%s', meta_data)

  return dataset_utils.Dataset(train_iter, eval_iter, test_iter, meta_data)
