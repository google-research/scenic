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

"""TFRecords data-loader for audiovisual speech recognition datasets."""

import functools
from typing import Any, Dict, Iterator, Optional, Sequence, Text, Tuple

from absl import logging
from dmvr import builders
from dmvr import modalities as load_modalities
from dmvr import tokenizers
from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.dataset_lib import video_ops
from scenic.projects.avatar.datasets.dataset_utils import add_int64
from scenic.projects.avatar.datasets.dataset_utils import add_spectrogram
from scenic.projects.avatar.datasets.dataset_utils import add_spectrogram_from_audio
from scenic.projects.avatar.datasets.dataset_utils import add_text
from scenic.projects.vivit.data import video_tfrecord_dataset
import tensorflow as tf


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]


def maybe_pad_batch(batch, train, batch_size, return_as_dict):
  """Zero pad the batch on the right to the batch_size."""
  if not return_as_dict:
    return dataset_utils.maybe_pad_batch(batch, train, batch_size)

  assert 'batch_mask' not in batch
  if 'rgb' in batch['inputs']:
    unpadded_mask_shape = batch['inputs']['rgb'].shape[0]
    batch_pad = batch_size - unpadded_mask_shape
  elif 'spectrogram' in batch['inputs']:
    unpadded_mask_shape = batch['inputs']['spectrogram'].shape[0]
    batch_pad = batch_size - unpadded_mask_shape
  elif 'waveform' in batch['inputs']:
    unpadded_mask_shape = batch['inputs']['waveform'].shape[0]
    batch_pad = batch_size - unpadded_mask_shape
  elif 'text' in batch['inputs']:
    unpadded_mask_shape = batch['inputs']['text'].shape[0]
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

  def zero_pad(array):
    pad_with = [(0, batch_pad)] + [(0, 0)] * (array.ndim - 1)
    return np.pad(array, pad_with, mode='constant')

  padded_batch = jax.tree_util.tree_map(zero_pad, batch)
  padded_batch_mask = zero_pad(np.ones(unpadded_mask_shape, dtype=np.float32))
  padded_batch['batch_mask'] = padded_batch_mask
  return padded_batch


def _convert_strings_to_int_sequences(in_strs, max_len):
  """Converts string to sequence of ints so that it is XLA compatible."""
  seq_list = []
  for s in in_strs:
    s = s.decode('utf-8')

    int_list = []
    for char in s:
      int_list.append(ord(char))
    nb_char = len(int_list)
    if nb_char <= max_len:
      pad_len = max_len - nb_char
      seq = np.array(int_list, dtype=np.int32)
      seq = np.pad(seq, (0, pad_len), 'constant', constant_values=(0,))
    else:
      seq = np.array(int_list[:max_len])
    seq_list.append(seq)
  return np.stack(seq_list, axis=0)


def _convert_key_string_to_int(features: Dict[str, Any]) -> Dict[str, Any]:
  """Converts keys to sequence of ints."""
  if 'key' in features:
    # Maximum output length
    features['key'] = _convert_strings_to_int_sequences(features['key'], 256)

  return features


def _convert_caption_string_to_int(features: Dict[str, Any]) -> Dict[str, Any]:
  """Converts keys to sequence of ints."""
  if 'key' in features:
    # Maximum output length
    features['raw_caption'] = _convert_strings_to_int_sequences(
        features['raw_caption'][:, 0], 512)

  return features


def _convert_ref_string_to_int(features: Dict[str, Any]) -> Dict[str, Any]:
  """Converts keys to sequence of ints."""
  if 'reference' in features:
    # Maximum output length
    features['reference'] = _convert_strings_to_int_sequences(
        features['reference'], 256)

  return features


class ASRTFRecordDatasetFactory(video_tfrecord_dataset.TFRecordDatasetFactory):
  """Reader for TFRecords using the MediaSequence format.

  The TFrecords already contain images and spectrograms.
  """

  _MODALITIES = ('rgb', 'spectrogram', 'waveform')

  def __init__(
      self,
      base_dir: str,
      tables: Any,
      subset: str = 'train',
      modalities: Tuple[str] = ('rgb',),
      prop_data: float = 1.0,
      num_groups: Optional[int] = None,
      group_index: Optional[int] = None,
  ):
    """Initializes the instance of AVSSTableDatasetFactory.

    Initializes a data-loader using DeepMind Video Reader (DMVR) pre-processing.
    TFrecords are assumed to consist of tf.SequenceExample protocol buffers in
    the MediaSequence format
    (https://github.com/google/mediapipe/tree/master/mediapipe/util/sequence).

    Args:
      base_dir: The base directory of the TFrecords.
      tables: A dictionary mapping the subset name (train, val or test) to the
        relative path of the SSTable containing them. Follows DMVR convention.
        The values of the dictionary can either be a string or a list. If it is
        a string, it specifies all the shards in the SSTable. Example -
        "/path/to/tfrecord@10". If passing a list, each entry is a shard of the
        TFRecord. Example - "[/path/to/sstable_shard_1_of_10, ...,
        /path/to/sstabble_shard_10_of_10]." The latter scenario is useful for
        debugging.
      subset: The subset of the dataset to load. Must be a key of "tables"
      modalities: Which modality to load. Currently supports 'rgb' and
        'spectrogram'
      prop_data: The proportion of the data to load. If less than 1.0, this
        proportion of the total SSTable shards are read.
      num_groups: If specified will reshard the data according to `num_groups`.
        A `group_index` should be specified if using `num_groups`.
      group_index: Index of the shard to return after resharding. `num_groups`
        should be specified if using `group_index`. This is useful in
        distributed setting where one wants to ensure that different data is
        read by different workers.
    """
    if subset not in tables:
      raise ValueError(f'Invalid subset {subset}. '
                       f'The available subsets are: {set(tables)}')

    for modality in modalities:
      if modality not in ASRTFRecordDatasetFactory._MODALITIES:
        raise ValueError('Invalid modality %s.' % modality)
    self._modalities = modalities

    super().__init__(
        base_dir=base_dir,
        tables=self.construct_tables(tables),
        examples_per_subset=self.construct_examples_per_subset(tables),
        subset=subset,
        num_classes=0,
        fraction_data=prop_data,
        num_groups=num_groups,
        group_index=group_index)

  def construct_tables(self, splits):
    ret = {}
    for split, dic in splits.items():
      ret[split] = dic['path']
    return ret

  def construct_examples_per_subset(self, splits):
    ret = {}
    for split, dic in splits.items():
      ret[split] = dic['len']
    return ret

  def _build(
      self,
      is_training: bool = True,
      # Video related parameters.
      num_frames: int = 32,
      stride: int = 1,
      num_spec_frames: int = 5,
      dataset_spec_mean: float = 0.,
      dataset_spec_stddev: float = 1.,
      num_test_clips: int = 1,
      min_resize: int = 256,
      crop_size: int = 224,
      # Audio related parameters.
      spec_compute_online: bool = False,
      spec_shape: Tuple[int, int] = (100, 80),
      spec_augment: bool = False,
      spec_augment_params=None,
      num_waveform_samples: int = 32000,
      waveform_stride: int = 1,
      zero_centering_image: bool = False,
      spec_from_wave_sample_rate: int = 16000,
      spec_from_wave_snr: float = -1.0,
      spec_from_wave_add_gaussian_noise: bool = False,
      spec_from_wave_add_masking_noise: bool = False,
      spec_from_wave_visualness_threshold: float = 0.2,
      spec_from_wave_random_mask_noise_rate: float = 0.4,
      spec_from_wave_max_word_len: int = 128,
      spec_from_wave_extend_mask_boundaries_ms: float = 0.,
      spec_from_wave_max_num_masks: int = -1,
      spec_from_wave_add_word_mask: bool = True,
      spec_from_wave_add_word_mask_info: bool = False,
      spec_from_wave_eval_noise_types: Optional[Sequence[str]] = None,
      spec_from_wave_environment_noise_path: Optional[str] = None,
      spec_from_wave_eval_noise_configs: Optional[Dict[str, Dict[str,
                                                                 Any]]] = None,
      spectrogram_type: str = 'logmf',
      spec_frame_length: int = 400,  # Corresponds to 25ms with 16K sampl rate.
      spec_frame_step: int = 160,  # Corresponds to 10ms with 16K sampl rate.
      spec_num_features: int = 80,
      spec_lower_edge_hertz: float = 0.0,
      spec_upper_edge_hertz: float = 7600.0,
      # Text related parameters.
      max_num_words: int = 16,
      max_num_captions: int = 1,
      tokenizer: Optional[tokenizers.TextTokenizer] = None,
      prepend_bos: bool = True,
      append_eos: bool = True,
      caption_string: str = 'caption/label/string',
      # Masked word prediction related parameters.
      masked_word_pred_aligned_caption_string: str = 'caption/label/string',
      masked_word_pred_max_num_tokens: int = 8,
      masked_word_pred_patch_size: Sequence[int] = (),
      masked_word_pred_max_num_masked_input_indices: int = 64,
  ):
    """Adds DMVR pre-processors to the dataset.

    Args:
      is_training: whether or not in training mode.
      num_frames: number of frames per subclip.
      stride: temporal stride to sample frames.
      num_spec_frames: number of spectrogram frames.
      dataset_spec_mean: Mean of spectrograms in the dataset.
      dataset_spec_stddev: Std dev of spectrograms in the dataset.
      num_test_clips: number of test clip (1 by default). If more than one, this
        will sample multiple linearly spaced clips within each video at test
        time. If 1, then a single clip in the middle of the video is sampled.
      min_resize: frames are resized so that min width/height is min_resize.
      crop_size: final size of the frame after cropping the resized frames.
      spec_compute_online: whether to compute spectrograms on the fly.
      spec_shape: input size of spectrogram per frame.
      spec_augment: whether to apply augmentation using SpecAugment.
      spec_augment_params: parameters for SpecAugment.
      num_waveform_samples: Number of waveform samples to use, defaults to 32000
        which corresponds to two seconds at 16kHz sampling rate.
      waveform_stride: Temporal stride to sample waveform.
      zero_centering_image: whether to have images between [-1, 1] or [0, 1].
      spec_from_wave_sample_rate: spectrogram computation parameter. The sample
        rate of the input audio.
      spec_from_wave_snr: spectrogram computation parameter. Signal-to-noise
        ratio used to inject white noise. The default value None or non-positive
        values disable the noise injection. Noise is add to non-training
        examples only by default. Training noise is added if `add_train_noise`
        is `True`.
      spec_from_wave_add_gaussian_noise: Whether to add Gaussian noise.
      spec_from_wave_add_masking_noise: If the noise type is word masking noise.
      spec_from_wave_visualness_threshold: The threshold for determining the
        visual words.
      spec_from_wave_random_mask_noise_rate: The random mask noise sampling
        ratio. If 1, the word masking is deterministic. If below 1, the masking
        noise is random and each word is masked out with this chance.
      spec_from_wave_max_word_len: Maximum length of the words. This is used to
        pad and truncate noise_word_mask added to the batch dictionary in the
        word masking noise addition.
      spec_from_wave_extend_mask_boundaries_ms: If set, the start and end
        boundaries of each maksing region in the signal is extended.
      spec_from_wave_max_num_masks: Maximum number of word masks to apply. -1
        means unlimited.
      spec_from_wave_add_word_mask: If set, add masks indicating masked words
        into the batch dict. Used for computing recovery rate of the masked
        words.
      spec_from_wave_add_word_mask_info: If set, add word masking related
        information such as timestamps and masks to the batch dict. Used for
        applying masked word prediction loss.
      spec_from_wave_eval_noise_types: A tuple of noise type strings used in
        evaluation. Each string should be either `environment_noise` or
        `packet_loss_noise`.
      spec_from_wave_environment_noise_path: Path to the npy file containing
        noise waveforms in numpy array.
      spec_from_wave_eval_noise_configs: A dict of noise config dicts for each
        noise type listed in `spec_from_wave_eval_noise_types`.
      spectrogram_type: spectrogram computation parameter. The type of the
        spectrogram to be extracted from the waveform. Can be either
        `spectrogram`, `logmf`, and `mfcc`.
      spec_frame_length: spectrogram computation parameter. The length of each
        spectroram frame.
      spec_frame_step: spectrogram computation parameter. The stride of
        spectrogram frames.
      spec_num_features: spectrogram computation parameter. The number of
        spectrogram features.
      spec_lower_edge_hertz: spectrogram computation parameter. Lowest frequency
        to consider.
      spec_upper_edge_hertz: spectrogram computation parameter. Highest
        frequency to consider.
      max_num_words: Maximum number of tokens to keep from the text for each
        caption. If there are more tokens, sequence is cropped, if less, the
        caption is padded using the tokenizer pad id.
      max_num_captions: Maximum number of captions to keep. If there are more
        captions in the proto, only the first `max_num_captions` will be
        returned is `is_training` is set to `False`. If `is_training` is `True`,
        then `max_num_captions` will be randomly sampled. Finally if the proto
        contains less than `max_num_captions`, we pad with empty srings to make
        sure there are `max_num_captions` in total.
      tokenizer: An instance of a tokenizer.
      prepend_bos: Whether to prepend BOS token.
      append_eos: Whether to append EOS token.
      caption_string: Input feature name in sstable for caption.
      masked_word_pred_aligned_caption_string: The feature name to parse to
        extract the caption aligned with the timestamps. Used for the masked
        word prediction.
      masked_word_pred_max_num_tokens: Maximum number of tokens for each masked
        word.
      masked_word_pred_patch_size: The patch size used in the network
        architecture for spectrogram token embedding. Used to compute the token
        index.
      masked_word_pred_max_num_masked_input_indices: Maximum number of masked
        input tokens (spectrogram tokens).
    """
    # We set sync_random_state to True so that sample_offset_proportion is
    # the same for all modalities.
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
          sync_random_state=True,
      )
    if 'spectrogram' in self._modalities:
      if spec_compute_online:
        if spec_from_wave_eval_noise_configs is None:
          spec_from_wave_eval_noise_configs = {}
        add_spectrogram_from_audio(
            parser_builder=self.parser_builder,
            sampler_builder=self.sampler_builder,
            preprocessor_builder=self.preprocessor_builder,
            is_training=is_training,
            spectrogram_type=spectrogram_type,
            sample_rate=spec_from_wave_sample_rate,
            snr=spec_from_wave_snr,
            add_gaussian_noise=spec_from_wave_add_gaussian_noise,
            add_masking_noise=spec_from_wave_add_masking_noise,
            visualness_score_threshold=spec_from_wave_visualness_threshold,
            random_mask_noise_rate=spec_from_wave_random_mask_noise_rate,
            max_word_len=spec_from_wave_max_word_len,
            extend_mask_boundaries_ms=spec_from_wave_extend_mask_boundaries_ms,
            max_num_masks=spec_from_wave_max_num_masks,
            add_word_mask=spec_from_wave_add_word_mask,
            add_word_mask_info=spec_from_wave_add_word_mask_info,
            eval_noise_types=spec_from_wave_eval_noise_types,
            environment_noise_path=spec_from_wave_environment_noise_path,
            **spec_from_wave_eval_noise_configs,
            frame_length=spec_frame_length,
            frame_step=spec_frame_step,
            num_features=spec_num_features,
            lower_edge_hertz=spec_lower_edge_hertz,
            upper_edge_hertz=spec_upper_edge_hertz,
            num_frames=num_spec_frames,
            spec_augment=spec_augment,
            spec_augment_params=spec_augment_params,
            zero_centering_image=zero_centering_image,
            dataset_mean=dataset_spec_mean,
            dataset_stddev=dataset_spec_stddev,
            word_tokenizer=tokenizer,
            aligned_caption_feature_name=masked_word_pred_aligned_caption_string,
            prepend_bos=prepend_bos,
            append_eos=append_eos,
            max_num_word_tokens=masked_word_pred_max_num_tokens,
            patch_size=masked_word_pred_patch_size,
            max_num_masked_input_indices=masked_word_pred_max_num_masked_input_indices,
        )
      else:
        add_spectrogram(
            parser_builder=self.parser_builder,
            sampler_builder=self.sampler_builder,
            decoder_builder=self.decoder_builder,
            preprocessor_builder=self.preprocessor_builder,
            postprocessor_builder=self.postprocessor_builder,
            input_feature_name='melspec/feature/floats',
            input_shape=spec_shape,
            is_training=is_training,
            num_frames=num_spec_frames,
            num_test_clips=num_test_clips,
            spec_augment=spec_augment,
            spec_augment_params=spec_augment_params,
            zero_centering_image=zero_centering_image,
            dataset_mean=dataset_spec_mean,
            dataset_stddev=dataset_spec_stddev,
        )
    if 'waveform' in self._modalities:
      load_modalities.add_audio(
          parser_builder=self.parser_builder,
          sampler_builder=self.sampler_builder,
          postprocessor_builder=self.postprocessor_builder,
          output_feature_name='waveform',
          is_training=is_training,
          num_samples=num_waveform_samples,
          stride=waveform_stride,
          num_test_clips=num_test_clips,
          sync_random_state=True,
      )

    add_text(
        parser_builder=self.parser_builder,
        decoder_builder=self.decoder_builder,
        preprocessor_builder=self.preprocessor_builder,
        tokenizer=tokenizer,
        is_training=is_training,
        input_feature_name=caption_string,
        prepend_bos=prepend_bos,
        append_eos=append_eos,
        max_num_captions=max_num_captions,
        max_num_tokens=max_num_words,
        keep_raw_string=False if is_training else True
    )

    add_int64(
        parser_builder=self.parser_builder,
        decoder_builder=self.decoder_builder,
        feature_name='clip/start/timestamp',
        output_name='clip/start/timestamp')
    add_int64(
        parser_builder=self.parser_builder,
        decoder_builder=self.decoder_builder,
        feature_name='clip/end/timestamp',
        output_name='clip/end/timestamp')

    def keep_only_short_videos(sample: builders.FeaturesDict) -> tf.Tensor:
      """Filter out the videos that are too long."""
      duration = sample['clip/end/timestamp'] - sample['clip/start/timestamp']
      duration = tf.cast(duration, tf.float32)
      duration = duration * 1E-6
      max_duration = float(num_spec_frames)
      duration_ok = tf.reshape(tf.less(duration, max_duration), [])
      return duration_ok

    if is_training:
      # Filter out the videos that are too long to fit in the encoder.
      # Otherwise the model could be trained to hallucinate words corresponding
      # to the cropped-out parts of the audio input.
      self.filter_builder.add_filter_fn(keep_only_short_videos,
                                        builders.Phase.DECODE)


def load_split_from_dmvr(
    ds_factory,
    batch_size,
    subset='train',
    modalities=('rgb'),
    num_frames=32,
    stride=2,
    num_spec_frames=5,
    num_test_clips=1,
    min_resize=256,
    crop_size=224,
    spec_shape=(96, 64),
    dataset_spec_mean=0.,
    dataset_spec_stddev=1.,
    spec_augment=False,
    spec_augment_params=None,
    num_waveform_samples=32000,
    waveform_stride=1,
    zero_centering=True,
    spec_compute_online: bool = False,
    spec_from_wave_sample_rate: int = 16000,
    spec_from_wave_snr: Optional[float] = None,
    spec_from_wave_add_gaussian_noise: bool = False,
    spec_from_wave_add_masking_noise: bool = False,
    spec_from_wave_visualness_threshold: float = 0.2,
    spec_from_wave_random_mask_noise_rate: float = 0.4,
    spec_from_wave_max_word_len: int = 128,
    spec_from_wave_extend_mask_boundaries_ms: float = 0.,
    spec_from_wave_max_num_masks: int = -1,
    spec_from_wave_add_word_mask: bool = True,
    spec_from_wave_add_word_mask_info: bool = False,
    spec_from_wave_eval_noise_types: Optional[Sequence[str]] = None,
    spec_from_wave_environment_noise_path: Optional[str] = None,
    spec_from_wave_eval_noise_configs: Optional[Dict[str, Dict[str,
                                                               Any]]] = None,
    spectrogram_type: str = 'logmf',
    spec_frame_length: int = 400,
    spec_frame_step: int = 160,
    spec_num_features: int = 80,
    spec_lower_edge_hertz: float = 0.0,
    spec_upper_edge_hertz: float = 7600.0,
    augmentation_params=None,
    keep_key=False,
    max_num_words: int = 16,
    max_num_captions: int = 1,
    tokenizer_type='bert',
    tokenizer_vocab=None,
    prepend_bos: bool = False,
    append_eos: bool = True,
    caption_string='caption/string',
    masked_word_pred_aligned_caption_string: str = 'caption/label/string',
    masked_word_pred_max_num_tokens: int = 8,
    masked_word_pred_patch_size: Sequence[int] = (),
    masked_word_pred_max_num_masked_input_indices: int = 64):
  """Loads dataset using DMVR for pre-processing.

  DMVR dataset loader already does basic augmentation (random crop and flip in
    train mode. It also already shuffles and batches the data.

  Args:
    ds_factory: A DMVR factory to instantiate with the subset.
    batch_size: The batch_size to use.
    subset: train, validation or test.
    modalities: list of input modalities.
    num_frames: Number of RGB frames per subclip.
    stride: Temporal stride to sample RGB frames.
    num_spec_frames: Number of spectrogram frames per subclip.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggreagated in the batch dimension.
    min_resize: Frames are resized so that min(height, width) is min_resize.
    crop_size: Final size of the frame after cropping the resized frames. Both
      height and width are the same.
    spec_shape: Input size of spectrogram per frame.
    dataset_spec_mean: Mean of spectrograms in the dataset.
    dataset_spec_stddev: Std dev of spectrograms in the dataset.
    spec_augment: whether to apply augmentation using SpecAugment.
    spec_augment_params: dict; augmentation configurations for SpecAugment
    num_waveform_samples: Number of waveform samples to use, defaults to 32000
      which corresponds to two seconds at 16kHz sampling rate.
    waveform_stride: Temporal stride to sample waveform.
    zero_centering: If True, frames are normalized to values in [-1, 1]. If
      False, values in [0, 1].
    spec_compute_online: whether to compute spectrograms on the fly.
    spec_from_wave_sample_rate: spectrogram computation parameter. The sample
      rate of the input audio.
    spec_from_wave_snr: spectrogram computation parameter. Signal-to-noise ratio
      used to inject white noise. The default value None or non-positive values
      disable the noise injection. Noise is add to non-training examples only by
      default. Training noise is added if `add_train_noise` is `True`.
    spec_from_wave_add_gaussian_noise: Whether to add Gaussian noise.
    spec_from_wave_add_masking_noise: If the noise type is word masking noise.
    spec_from_wave_visualness_threshold: The threshold for determining the
      visual words.
    spec_from_wave_random_mask_noise_rate: The random mask noise sampling
      ratio. If 1, the word masking is deterministic. If below 1,
      the masking noise is random and each word is masked out with this
      chance.
    spec_from_wave_max_word_len: Maximum length of the words. This is used to
      pad and truncate noise_word_mask added to the batch dictionary in the word
      masking noise addition.
    spec_from_wave_extend_mask_boundaries_ms: If set, the start and end
      boundaries of each maksing region in the signal is extended.
    spec_from_wave_max_num_masks: Maximum number of word masks to apply. -1
      means unlimited.
    spec_from_wave_add_word_mask: If set, add masks indicating masked words
      into the batch dict. Used for computing recovery rate of the masked
      words.
    spec_from_wave_add_word_mask_info: If set, add word masking related
      information such as timestamps and masks to the batch dict. Used for
      applying masked word prediction loss.
    spec_from_wave_eval_noise_types: A tuple of noise type strings used in
      evaluation. Each string should be either `environment_noise` or
      `packet_loss_noise`.
    spec_from_wave_environment_noise_path: Path to the npy file containing
      noise waveforms in numpy array.
    spec_from_wave_eval_noise_configs: A dict of noise config dicts for each
      noise type listed in `spec_from_wave_eval_noise_types`.
    spectrogram_type: spectrogram computation parameter. The type of the
      spectrogram to be extracted from the waveform. Can be either
      `spectrogram`, `logmf`, and `mfcc`.
    spec_frame_length: spectrogram computation parameter. The length of each
      spectroram frame.
    spec_frame_step: spectrogram computation parameter. The stride of
      spectrogram frames.
    spec_num_features: spectrogram computation parameter. The number of
      spectrogram features.
    spec_lower_edge_hertz: spectrogram computation parameter. Lowest frequency
      to consider.
    spec_upper_edge_hertz: spectrogram computation parameter. Highest frequency
      to consider.
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
    tokenizer_type: The type of tokenizer. Supported types: ('bert',).
    tokenizer_vocab: The path to the tokenizer vocabulary.
    prepend_bos: Whether to prepend BOS token.
    append_eos: Whether to append EOS token.
    caption_string: Input feature name in sstable for caption.
    masked_word_pred_aligned_caption_string: The feature name to parse to
      extract the caption aligned with the timestamps. Used for the masked
      word prediction.
    masked_word_pred_max_num_tokens: Maximum number of tokens for each masked
      word.
    masked_word_pred_patch_size: The patch size used in the network architecture
      for spectrogram token embedding. Used to compute the token index.
    masked_word_pred_max_num_masked_input_indices: Maximum number of masked
      input tokens (spectrogram tokens).

  Returns:
    A pair `(ds, num_examples)` with
      ds: A `tf.data.Dataset` object
      num_examples: Number of examples in the dataset.
  """
  is_training = (subset == 'train')

  if tokenizer_type == 'bert':
    assert tokenizer_vocab
    tokenizer = tokenizers.BertTokenizer(tokenizer_vocab)
  else:
    raise ValueError('Tokenizer not supported')
  vocab_size = int(tokenizer.vocab_size)
  logging.info('vocab_size %d', vocab_size)
  logging.info('EOS token: %d', tokenizer.eos_token)
  # Init the TF models of the tokenizer.
  tokenizer.initialize()

  ds_factory = ds_factory(
      subset=subset, modalities=modalities
  ).configure(
      is_training=is_training,
      num_frames=num_frames,
      stride=stride,
      num_spec_frames=num_spec_frames,
      num_test_clips=num_test_clips,
      min_resize=min_resize,
      crop_size=crop_size,
      spec_shape=spec_shape,
      dataset_spec_mean=dataset_spec_mean,
      dataset_spec_stddev=dataset_spec_stddev,
      spec_augment=spec_augment,
      spec_augment_params=spec_augment_params,
      spec_compute_online=spec_compute_online,
      spec_from_wave_sample_rate=spec_from_wave_sample_rate,
      spec_from_wave_snr=spec_from_wave_snr,
      spec_from_wave_add_gaussian_noise=spec_from_wave_add_gaussian_noise,
      spec_from_wave_add_masking_noise=spec_from_wave_add_masking_noise,
      spec_from_wave_visualness_threshold=spec_from_wave_visualness_threshold,
      spec_from_wave_random_mask_noise_rate=(
          spec_from_wave_random_mask_noise_rate),
      spec_from_wave_max_word_len=spec_from_wave_max_word_len,
      spec_from_wave_extend_mask_boundaries_ms=(
          spec_from_wave_extend_mask_boundaries_ms),
      spec_from_wave_max_num_masks=spec_from_wave_max_num_masks,
      spec_from_wave_add_word_mask=spec_from_wave_add_word_mask,
      spec_from_wave_add_word_mask_info=spec_from_wave_add_word_mask_info,
      spec_from_wave_eval_noise_types=spec_from_wave_eval_noise_types,
      spec_from_wave_environment_noise_path=spec_from_wave_environment_noise_path,
      spec_from_wave_eval_noise_configs=spec_from_wave_eval_noise_configs,
      spectrogram_type=spectrogram_type,
      spec_frame_length=spec_frame_length,
      spec_frame_step=spec_frame_step,
      spec_num_features=spec_num_features,
      spec_lower_edge_hertz=spec_lower_edge_hertz,
      spec_upper_edge_hertz=spec_upper_edge_hertz,
      num_waveform_samples=num_waveform_samples,
      waveform_stride=waveform_stride,
      zero_centering_image=zero_centering,
      max_num_words=max_num_words,
      max_num_captions=max_num_captions,
      tokenizer=tokenizer,
      prepend_bos=prepend_bos,
      append_eos=append_eos,
      caption_string=caption_string,
      masked_word_pred_aligned_caption_string=masked_word_pred_aligned_caption_string,
      masked_word_pred_max_num_tokens=masked_word_pred_max_num_tokens,
      masked_word_pred_patch_size=masked_word_pred_patch_size,
      masked_word_pred_max_num_masked_input_indices=masked_word_pred_max_num_masked_input_indices,
  )

  if 'rgb' in modalities and is_training and augmentation_params:
    # additional augmentation for the RGB features.
    ds_factory = video_ops.additional_augmentations(ds_factory,
                                                    augmentation_params,
                                                    crop_size, num_frames,
                                                    zero_centering)

  logging.info('Preprocessing graph: %s',
               ds_factory.preprocessor_builder.get_summary())
  logging.info('Postprocessing graph: %s',
               ds_factory.postprocessor_builder.get_summary())

  # Val and test splits are a single epoch otherwise the last
  # batch is not padded with zeros but with valid examples
  num_examples = ds_factory.num_examples
  ds = ds_factory.make_dataset(
      batch_size=batch_size,
      shuffle=is_training,
      num_epochs=None if is_training else 1,
      drop_remainder=is_training,
      keep_key=(not is_training and keep_key))

  if not is_training:
    # Repeat indefinitely
    ds = ds.repeat(None)

  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  return ds, num_examples


def map_keys(batch, modalities=('rgb'), return_as_dict=False):
  """DMVR dataset returns 'image' and 'label'. We want 'inputs' and 'label'."""
  if not return_as_dict:
    if len(modalities) == 1 and modalities[0] == 'rgb':
      batch['inputs'] = batch['image']
    elif len(modalities) == 1 and modalities[0] == 'spectrogram':
      batch['inputs'] = batch['spectrogram']
    elif len(modalities) == 1 and modalities[0] == 'waveform':
      batch['inputs'] = batch['waveform']
    else:
      raise NotImplementedError('modality not supported by map_keys.')
  else:
    batch['inputs'] = {}
    if 'rgb' in modalities:
      batch['inputs']['rgb'] = batch['image']
      batch.pop('image')
    if 'spectrogram' in modalities:
      batch['inputs']['spectrogram'] = batch['spectrogram']
      batch.pop('spectrogram')
    if 'waveform' in modalities:
      batch['inputs']['waveform'] = batch['waveform']
      batch.pop('waveform')
  batch['targets'] = batch['text_indices']
  batch.pop('text_indices')
  return batch


@datasets.add_dataset('av_asr_tfrecord_dataset')
def get_dataset(
    *,
    batch_size,
    eval_batch_size,
    num_shards,
    dtype_str='float32',
    shuffle_seed=0,  # pylint:disable=unused-argument
    rng=None,
    dataset_configs,
    dataset_service_address: Optional[str] = None):
  """Returns a generator for the audiovisual dataset."""
  del rng
  dataset_configs = dataset_configs or {}
  modalities = dataset_configs.get('modalities', ['rgb'])
  return_as_dict = dataset_configs.get('return_as_dict', True)
  # RGB related configs.
  num_frames = dataset_configs.get('num_frames', 32)
  stride = dataset_configs.get('stride', 2)
  eval_stride = dataset_configs.get('eval_stride', stride)
  min_resize = dataset_configs.get('min_resize', 256)
  crop_size = dataset_configs.get('crop_size', 224)
  # Spectrogram related configs.
  num_spec_frames = dataset_configs.get('num_spec_frames', 25)
  eval_num_spec_frames = dataset_configs.get('eval_num_spec_frames',
                                             num_spec_frames)
  spec_shape = dataset_configs.get('spec_shape', (100, 80))
  spec_augment = dataset_configs.get('spec_augment', False)
  spec_augment_params = dataset_configs.get('spec_augment_params', None)
  dataset_spec_mean = dataset_configs.get('spec_mean', 0.)
  dataset_spec_stddev = dataset_configs.get('spec_stddev', 1.)
  # Waveform related configs
  num_waveform_samples = dataset_configs.get('num_waveform_samples', 16000)
  eval_num_waveform_samples = dataset_configs.get('eval_num_waveform_samples',
                                                  num_waveform_samples)
  waveform_stride = dataset_configs.get('waveform_stride', 1)
  # Spectrogram computation related configs
  spec_compute_online = dataset_configs.get('spec_compute_online', False)
  spec_from_wave_sample_rate = dataset_configs.get('spec_from_wave_sample_rate',
                                                   16000)
  spec_from_wave_snr = dataset_configs.get('spec_from_wave_snr', None)
  spec_from_wave_add_gaussian_noise = dataset_configs.get(
      'spec_from_wave_add_gaussian_noise', False)
  spec_from_wave_add_train_masking_noise = dataset_configs.get(
      'spec_from_wave_add_train_masking_noise', False)
  spec_from_wave_visualness_threshold = dataset_configs.get(
      'spec_from_wave_visualness_threshold', None)
  spec_from_wave_random_mask_noise_rate = dataset_configs.get(
      'spec_from_wave_random_mask_noise_rate', 0.4)
  spec_from_wave_max_word_len = dataset_configs.get(
      'spec_from_wave_max_word_len', 0)
  spec_from_wave_extend_mask_boundaries_ms = dataset_configs.get(
      'spec_from_wave_extend_mask_boundaries_ms', 0.)
  spec_from_wave_max_num_masks = dataset_configs.get(
      'spec_from_wave_max_num_masks', 16)
  spec_from_wave_add_word_mask = dataset_configs.get(
      'spec_from_wave_add_word_mask', False)
  spec_from_wave_add_word_mask_info = dataset_configs.get(
      'spec_from_wave_add_word_mask_info', False)
  spec_from_wave_eval_noise_configs = dataset_configs.get(
      'spec_from_wave_eval_noise_configs', None)
  spec_from_wave_environment_noise_path = dataset_configs.get(
      'spec_from_wave_environment_noise_path', None)
  spectrogram_type = dataset_configs.get('spectrogram_type', 'logmf')
  spec_frame_length = dataset_configs.get('spec_frame_length', 400)
  spec_frame_step = dataset_configs.get('spec_frame_step', 160)
  spec_num_features = dataset_configs.get('spec_num_features', 80)
  spec_lower_edge_hertz = dataset_configs.get('spec_lower_edge_hertz', 0.0)
  spec_upper_edge_hertz = dataset_configs.get('spec_upper_edge_hertz', 7600.0)
  masked_word_pred_aligned_caption_string = dataset_configs.get(
      'masked_word_pred_aligned_caption_string', 'caption/label/string')
  masked_word_pred_max_num_tokens = dataset_configs.get(
      'masked_word_pred_max_num_tokens', 8)
  masked_word_pred_patch_size = dataset_configs.get(
      'masked_word_pred_patch_size', [])
  masked_word_pred_max_num_masked_input_indices = dataset_configs.get(
      'masked_word_pred_max_num_masked_input_indices', 32)
  # General configs.
  num_eval_clips = dataset_configs.get('num_eval_clips', 1)
  zero_centre_data = dataset_configs.get('zero_centering', True)
  augmentation_params = dataset_configs.get('augmentation_params', None)
  num_train_val_clips = dataset_configs.get('num_train_val_clips', 1)
  do_three_spatial_crops = dataset_configs.get('do_three_spatial_crops', False)
  num_spatial_crops = 3 if do_three_spatial_crops else 1

  # Should hold two fields: tokenizer_type and tokenizer_vocab.
  tokenizer_config = dataset_configs.get('tokenizer', {})
  max_num_words = dataset_configs.get('max_num_words', 16)
  eval_max_num_words = dataset_configs.get('eval_max_num_words', max_num_words)
  max_num_captions = dataset_configs.get('max_num_captions', 1)
  caption_string = dataset_configs.get('caption_string', 'clip/label/string')

  if (spec_from_wave_add_train_masking_noise and
      spec_from_wave_visualness_threshold is None):
    raise ValueError(
        '`spec_from_wave_visualness_threshold` must be specified when '
        '`spec_from_wave_add_train_masking_noise` is True')

  def validate_config(field):
    if dataset_configs.get(field) is None:
      raise ValueError(f'{field} must be specified for TFRecord dataset.')

  validate_config('base_dir')
  validate_config('tables')

  ds_factory = functools.partial(
      ASRTFRecordDatasetFactory,
      base_dir=dataset_configs.get('base_dir'),
      tables=dataset_configs.get('tables'),
      num_groups=jax.process_count(),
      group_index=jax.process_index(),
  )

  def create_dataset_iterator(
      subset: Text,
      batch_size_local: int,
      num_clips: int,
      caption_string: str,
      stride: int,
      num_spec_frames: int,
      num_waveform_samples: int,
      max_num_words: int,
      keep_key_local: bool = True,
      add_masking_noise: bool = False,
      eval_noise_types: Optional[Sequence[str]] = None,
      eval_noise_configs: Optional[Dict[str, Dict[str, Any]]] = None,
      ) -> Tuple[Iterator[Batch], int]:

    is_training = subset == 'train'
    is_test = subset == 'test'
    logging.info('Loading split %s', subset)

    # TODO(phseo): Remove duplicates and pass the dict itself.
    dataset, num_examples = load_split_from_dmvr(
        ds_factory,
        batch_size=batch_size_local,
        subset=subset,
        modalities=modalities,
        num_frames=num_frames,
        stride=stride,
        num_spec_frames=num_spec_frames,
        num_test_clips=num_clips,
        min_resize=min_resize,
        crop_size=crop_size,
        spec_shape=spec_shape,
        dataset_spec_mean=dataset_spec_mean,
        dataset_spec_stddev=dataset_spec_stddev,
        spec_augment=spec_augment,
        spec_augment_params=spec_augment_params,
        spec_compute_online=spec_compute_online,
        spec_from_wave_sample_rate=spec_from_wave_sample_rate,
        spec_from_wave_snr=spec_from_wave_snr,
        spec_from_wave_add_gaussian_noise=spec_from_wave_add_gaussian_noise,
        spec_from_wave_add_masking_noise=add_masking_noise,
        spec_from_wave_visualness_threshold=(
            spec_from_wave_visualness_threshold),
        spec_from_wave_random_mask_noise_rate=(
            spec_from_wave_random_mask_noise_rate),
        spec_from_wave_max_word_len=spec_from_wave_max_word_len,
        spec_from_wave_extend_mask_boundaries_ms=(
            spec_from_wave_extend_mask_boundaries_ms),
        spec_from_wave_max_num_masks=spec_from_wave_max_num_masks,
        spec_from_wave_add_word_mask=spec_from_wave_add_word_mask,
        spec_from_wave_add_word_mask_info=spec_from_wave_add_word_mask_info,
        spec_from_wave_eval_noise_types=eval_noise_types,
        spec_from_wave_environment_noise_path=spec_from_wave_environment_noise_path,
        spec_from_wave_eval_noise_configs=eval_noise_configs,
        spectrogram_type=spectrogram_type,
        spec_frame_length=spec_frame_length,
        spec_frame_step=spec_frame_step,
        spec_num_features=spec_num_features,
        spec_lower_edge_hertz=spec_lower_edge_hertz,
        spec_upper_edge_hertz=spec_upper_edge_hertz,
        num_waveform_samples=num_waveform_samples,
        waveform_stride=waveform_stride,
        zero_centering=zero_centre_data,
        augmentation_params=augmentation_params,
        keep_key=keep_key_local,
        max_num_words=max_num_words,
        max_num_captions=max_num_captions,
        **tokenizer_config,
        caption_string=caption_string,
        masked_word_pred_aligned_caption_string=masked_word_pred_aligned_caption_string,
        masked_word_pred_max_num_tokens=masked_word_pred_max_num_tokens,
        masked_word_pred_patch_size=masked_word_pred_patch_size,
        masked_word_pred_max_num_masked_input_indices=masked_word_pred_max_num_masked_input_indices,
    )

    if dataset_service_address and is_training:
      if shuffle_seed is not None:
        raise ValueError('Using dataset service with a random seed causes each '
                         'worker to produce exactly the same data. Add '
                         'config.shuffle_seed = None to your config if you '
                         'want to run with dataset service.')
      logging.info('Using the tf.data service at %s', dataset_service_address)
      dataset = dataset_utils.distribute(dataset, dataset_service_address)

    pad_batch_size = batch_size_local
    if is_test:
      pad_batch_size = batch_size_local * num_clips * num_spatial_crops
    maybe_pad_batches = functools.partial(
        maybe_pad_batch,
        train=is_training,
        batch_size=pad_batch_size,
        return_as_dict=return_as_dict)

    shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
    current_iter = iter(dataset)
    current_iter = map(dataset_utils.tf_to_numpy, current_iter)
    current_iter = map(
        functools.partial(
            map_keys, modalities=modalities, return_as_dict=return_as_dict),
        current_iter)
    current_iter = map(_convert_key_string_to_int, current_iter)
    if not is_training:
      current_iter = map(_convert_caption_string_to_int, current_iter)
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
      'train', batch_size, num_train_val_clips, caption_string, stride,
      num_spec_frames, num_waveform_samples, max_num_words,
      add_masking_noise=spec_from_wave_add_train_masking_noise)
  eval_iter, n_eval_examples = create_dataset_iterator(
      'val', eval_batch_size, num_eval_clips, caption_string, eval_stride,
      eval_num_spec_frames, eval_num_waveform_samples, eval_max_num_words)
  test_iter, n_test_examples = create_dataset_iterator(
      'test', eval_batch_size, num_eval_clips, caption_string, eval_stride,
      eval_num_spec_frames, eval_num_waveform_samples, eval_max_num_words)
  if spec_from_wave_eval_noise_configs:
    test_iter = [test_iter]
    for noise_types, configs in spec_from_wave_eval_noise_configs.items():
      noise_types = noise_types.split(',')
      for config in configs:
        test_iter.append(
            create_dataset_iterator(
                'test', eval_batch_size, num_eval_clips, caption_string,
                eval_stride, eval_num_spec_frames, eval_num_waveform_samples,
                eval_max_num_words, eval_noise_types=noise_types,
                eval_noise_configs=config)[0]
        )

  meta_data = {
      # pylint:disable=protected-access
      'num_train_examples': (n_train_examples * num_train_val_clips),
      'num_eval_examples': (n_eval_examples * num_eval_clips),
      'num_test_examples':
          (n_test_examples * num_eval_clips * num_spatial_crops),
      'input_dtype': getattr(jnp, dtype_str)
  }

  # Set the input shapes
  input_shapes = {
      'rgb': (-1, num_frames, crop_size, crop_size, 3),
      'spectrogram': (-1, num_spec_frames * spec_shape[0], spec_shape[1], 3),
      'waveform': (-1, num_waveform_samples),
  }
  meta_data['input_shape'] = {}
  for modality, shape in input_shapes.items():
    if modality in modalities:
      meta_data['input_shape'][modality] = shape

  meta_data['target_shape'] = (-1, max_num_words)
  meta_data['target_dtype'] = jnp.int32

  if spec_from_wave_add_word_mask_info:
    meta_data['masked_token_idxs_shape'] = (
        -1, spec_from_wave_max_num_masks,
        masked_word_pred_max_num_masked_input_indices)
    meta_data['masked_token_idx_masks_shape'] = (
        -1, spec_from_wave_max_num_masks,
        masked_word_pred_max_num_masked_input_indices)
    meta_data['masked_word_targets_shape'] = (
        -1, spec_from_wave_max_num_masks,
        masked_word_pred_max_num_tokens)
    meta_data['masked_token_idxs_dtype'] = jnp.int32
    meta_data['masked_token_idx_masks_dtype'] = jnp.int32
    meta_data['masked_word_targets_dtype'] = jnp.int32

  logging.info('Number of training examples: %d',
               meta_data['num_train_examples'])
  logging.info('Number of validation examples: %d',
               meta_data['num_eval_examples'])
  logging.info('Number of test examples: %d', meta_data['num_test_examples'])

  return dataset_utils.Dataset(train_iter, eval_iter, test_iter, meta_data)
