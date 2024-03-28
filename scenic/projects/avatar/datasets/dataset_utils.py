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

"""Utility functions for datasets that process SequenceExample."""

import functools
from typing import Any, Dict, Optional, Sequence

from absl import logging
from dmvr import builders
from dmvr import processors
from dmvr import tokenizers
import numpy as np
from scenic.projects.mbt.datasets.dataset_utils import _decode_spectrogram
from scenic.projects.mbt.datasets.dataset_utils import apply_specaugment
import tensorflow as tf


VALID_EVAL_NOISE_TYPES = ['environment_noise', 'packet_loss_noise']


def pad_first_dim(sequence, max_len, pad_value):
  """Pads the first dimension of the input `sequence` to `max_len`."""
  sequence_length = tf.shape(sequence)[0]
  padding_pattern = [
      [0, tf.maximum(0, max_len - sequence_length)],
  ]

  num_dim = len(tf.shape(sequence))
  if num_dim > 1:
    padding_pattern.extend([[0, 0]] * (num_dim - 1))
  return tf.pad(
      tensor=sequence, paddings=padding_pattern, constant_values=pad_value)


def normalize_input(signal, normalization_threshold=5e-4):
  """Normalizes the signal to the [-1, 1] range."""
  sig_min = tf.reduce_min(signal, name='norm_min', keepdims=True)
  sig_max = tf.reduce_max(signal, name='norm_max', keepdims=True)
  # Don't normalize silent noise.
  sig_delta = tf.maximum(
      (sig_max - sig_min) * 0.5, normalization_threshold, name='norm_delta')
  return (signal - sig_min) / sig_delta - 1.0


def add_white_noise(wave_tensor, snr):
  """Adds deterministic gaussian white noise to the signal.

  While setting a global seeds does not guarrantee the deterministic per-sample
  noise generation due to the intrinsic randomness in the TF dataloading
  process. Instead, we generate the seeds from the input signal and generate
  noise from the stateless normal sampling process. This effectively sample
  deterministic noise per sample while the noise varies between samples. (A
  constant seed for all sample would generate identical noise for all samples,
  which is not a desired behavior.)

  Args:
    wave_tensor: Input tensor containing the raw wave signal.
    snr: The signal to noise ratio to determine the power of the noise.

  Returns:
    Output signal with gaussian white noise injected.
  """
  pos = tf.reduce_mean(tf.square(wave_tensor))
  nos = pos / snr
  # Deterministically create the noise using the input signal as the seeds.
  # The seeds are determined by the sums of the positive and negative signals.
  positive_sum = tf.cast(tf.reduce_sum(tf.maximum(wave_tensor, 0)), tf.int32)
  negative_sum = tf.cast(tf.reduce_sum(tf.maximum(-wave_tensor, 0)), tf.int32)
  noise = tf.random.stateless_normal(
      tf.shape(wave_tensor),
      seed=[positive_sum, negative_sum],
      stddev=tf.sqrt(nos),
      dtype=wave_tensor.dtype)

  return wave_tensor + noise


def mask_word_with_white_noise(wave_tensor, snr, word_mask, word_sts, word_ets,
                               sample_rate, extend_boundaries_ms):
  """Add word masking noise to the signal.

  Masks out wave signal that correspond to words indicated by word_mask. Each
  masked part is filled with Gaussian white noise whose power is determined by
  snr.

  Args:
    wave_tensor: Input tensor containing the raw wave signal.
    snr: The signal to noise ratio to determine the power of the white noise
      filled in the masked region. If None, the regions are zeroed out.
    word_mask: A boolean mask tensor where the words to mask are marked as True.
    word_sts: The start timestamps of the words.
    word_ets: The end timestamps of the words.
    sample_rate: The sample rate of the input wave signal.
    extend_boundaries_ms: If set, the start and end boundaries of each masking
      region in the signal is extended.

  Returns:
    Output signal with word masking noise injected.
  """
  if snr is not None and snr > 0.0:
    pos = tf.reduce_mean(tf.square(wave_tensor))
    nos = pos / snr
    # Deterministically create the noise using the input signal as the seeds.
    # The seeds are determined by the sums of the positive and negative signals.
    positive_sum = tf.cast(tf.reduce_sum(tf.maximum(wave_tensor, 0)), tf.int32)
    negative_sum = tf.cast(tf.reduce_sum(tf.maximum(-wave_tensor, 0)), tf.int32)
    noise = tf.random.stateless_normal(
        tf.shape(wave_tensor),
        seed=[positive_sum, negative_sum],
        stddev=tf.sqrt(nos),
        dtype=wave_tensor.dtype)
  else:
    noise = 0

  target_word_sts = tf.boolean_mask(word_sts, word_mask) - extend_boundaries_ms
  target_word_ets = tf.boolean_mask(word_ets, word_mask) + extend_boundaries_ms

  timestamps = (tf.range(tf.shape(wave_tensor)[0], dtype=tf.float32) /
                sample_rate) * 1000
  timestamps = tf.repeat(timestamps[None, :], tf.shape(target_word_sts)[0], 0)

  time_mask = tf.logical_and(
      tf.greater_equal(timestamps, target_word_sts[:, None]),
      tf.less_equal(timestamps, target_word_ets[:, None]))
  time_mask = tf.cast(tf.reduce_any(time_mask, 0), tf.float32)

  noised_wave_tensor = wave_tensor * (1 - time_mask) + noise * time_mask
  return noised_wave_tensor, target_word_sts, target_word_ets


def add_visual_word_masking_noise(
    batch,
    wave_feature_name: str,
    snr: float = 1.0,
    sample_rate: int = 16000,
    vis_threshold: float = 0.2,
    max_word_len: int = 128,
    mask_rate: float = 0.4,
    max_num_masks: int = -1,
    extend_boundaries_ms: float = 0.,
    add_word_mask: bool = True,
    add_word_mask_info: bool = False,
):
  """Add word masking noise to the wave signal in the input batch.

  Masks out wave signal that correspond to words indicated by word_mask. Each
  masked part is filled with Gaussian white noise whose power is determined by
  snr. If snr is None, Gaussian white noise filling is not performed. When
  mask_rate < 1.0, the target words are randomly selected with the probability
  of mask_rate and only the selected words are masked out.

  Args:
    batch: The input batch dictionary.
    wave_feature_name: Dictionary key for the raw wave feature.
    snr: The signal to noise ratio to determine the power of the white noise. If
      None, the target regions are simply zeroed out.
    sample_rate: The sample rate of the input wave signal.
    vis_threshold: The threshold for the visualness scores of the input words.
    max_word_len: Maximum length of the words. This is used to pad and truncate
      noise_word_mask which is added in the batch dictionary.
    mask_rate: The masking ratio used for random mask generation.
    max_num_masks: The maximum number of masks to keep. If the number of mask
      candidates are larger than this, the masks are randomly subsampled to
      match this number. If -1, all masks are kept.
    extend_boundaries_ms: If set, the start and end boundaries of each maksing
      region in the signal is extended.
    add_word_mask: If set, add raw word mask into the batch dict. Used to
      compute recovery rate.
    add_word_mask_info: If set, add word masking related information including
      timestamps and length to be used for computing masked word prediction
      loss.

  Returns:
    The updated batch dictionary where the wave signal is updated with the noise
    injected one.
  """

  vis_score = batch['visualness_score']
  word_sts = batch['word_start_timestamp']
  word_ets = batch['word_end_timestamp']
  wave_tensor = batch[wave_feature_name]

  word_mask = vis_score > vis_threshold

  logging.info('mask_rate: %f', mask_rate)
  if mask_rate < 1.0:
    logging.info('doing random masking')
    random_mask = tf.random.uniform(tf.shape(word_mask)) < mask_rate
    word_mask = tf.logical_and(random_mask, word_mask)

  logging.info('max_num_masks: %d', max_num_masks)
  logging.info('word_mask: %s', word_mask)

  if max_num_masks > -1:
    if tf.reduce_sum(tf.cast(word_mask, tf.int32)) > max_num_masks:
      indices = tf.where(word_mask)
      indices = tf.random.shuffle(indices)[:max_num_masks]
      word_mask = tf.cast(
          tf.scatter_nd(
              tf.cast(indices, tf.int32), tf.ones(max_num_masks),
              tf.shape(word_mask)), tf.bool)

  if add_word_mask:
    logging.info('adding noise_word_mask')
    # This is for computing word recovery rate
    batch['noise_word_mask'] = pad_first_dim(word_mask[:max_word_len],
                                             max_word_len, 0)

  batch[wave_feature_name], word_start_timestamp, word_end_timestamp = (
      mask_word_with_white_noise(wave_tensor, snr, word_mask, word_sts,
                                 word_ets, sample_rate, extend_boundaries_ms))

  if add_word_mask_info:
    # This is for masked word prediction loss
    batch['mwp_word_mask'] = word_mask
    batch['mwp_start_timestamp'] = word_start_timestamp
    batch['mwp_end_timestamp'] = word_end_timestamp
    batch['mwp_wave_length'] = tf.math.divide(
        tf.cast(tf.size(wave_tensor), tf.float32),
        tf.cast(sample_rate, tf.float32)) * 1000.

  del batch['word_start_timestamp'], batch['word_end_timestamp'], batch[
      'visualness_score']

  return batch


def tokenize_masked_words(batch, tokenizer, max_num_tokens, prepend_bos,
                          append_eos):
  """Tokenize masked words and add them to the batch in `masked_targets` field."""

  caption_string = batch['mwp_caption_string'][0]
  word_mask = batch['mwp_word_mask']
  masked_word_strings = tf.boolean_mask(
      tf.strings.split(caption_string), word_mask)

  tokenized = tokenizer.string_tensor_to_indices(
      masked_word_strings,
      prepend_bos=prepend_bos,
      append_eos=append_eos,
      max_num_tokens=max_num_tokens)

  batch['masked_targets'] = tokenized

  del batch['mwp_word_mask'], batch['mwp_caption_string']

  return batch


def convert_ms_to_frame_number(batch, spectrogram_feature_name):
  """Convert start timestamps in microsec into the frame numbers."""
  word_start_timestamp = batch['mwp_start_timestamp']
  word_end_timestamp = batch['mwp_end_timestamp']
  wave_length = batch['mwp_wave_length']
  spectrogram = batch[spectrogram_feature_name]

  start_idxs = tf.cast(
      tf.math.divide_no_nan(word_start_timestamp, wave_length)
      * tf.cast((tf.shape(spectrogram)[0] - 1), tf.float32), tf.int32)
  end_idxs = tf.cast(
      tf.math.divide_no_nan(word_end_timestamp, wave_length)
      * tf.cast((tf.shape(spectrogram)[0] - 1), tf.float32), tf.int32)

  batch['mwp_start_indices'] = start_idxs
  batch['mwp_end_indices'] = end_idxs

  del batch['mwp_wave_length'], batch['mwp_start_timestamp'], batch[
      'mwp_end_timestamp']

  return batch


def finalize_word_mask_info(batch, spectrogram_feature_name, patch_size,
                            max_num_word_masks, max_num_masked_input_indices):
  """Format the word mask related information in the batch dict."""
  spectrogram = batch[spectrogram_feature_name]
  len_spec = tf.shape(spectrogram)[0]
  num_feats = tf.shape(spectrogram)[1]

  len_spec = tf.cast(len_spec / patch_size[0], tf.int32) * patch_size[0]

  start_idxs = batch['mwp_start_indices']
  end_idxs = batch['mwp_end_indices']
  masked_targets = batch['masked_targets']

  oor_mask = tf.less(start_idxs, len_spec)

  start_idxs = tf.boolean_mask(start_idxs, oor_mask)
  end_idxs = tf.boolean_mask(end_idxs, oor_mask)
  masked_targets = tf.boolean_mask(masked_targets, oor_mask, axis=0)

  # now convert spectrogram index to token index using patch size
  start_idxs = tf.math.divide(
      tf.cast(start_idxs, tf.float32), tf.cast(patch_size[0], tf.float32))
  end_idxs = tf.math.divide(
      tf.cast(end_idxs, tf.float32), tf.cast(patch_size[0], tf.float32))

  num_tokens_per_step = tf.cast(tf.math.divide(
      tf.cast(num_feats, tf.int32), tf.cast(patch_size[1], tf.int32)), tf.int32)
  start_token_index = tf.cast(start_idxs, tf.int32) * num_tokens_per_step
  end_token_index = tf.cast(end_idxs + 1, tf.int32) * num_tokens_per_step

  token_length = tf.cast(len_spec / patch_size[0],
                         tf.int32) * num_tokens_per_step
  idx_pool = tf.range(token_length, dtype=tf.int32)
  word_token_idx_mask = tf.logical_and(
      tf.greater_equal(idx_pool[None, :], start_token_index[:, None]),
      tf.less(idx_pool[None, :], end_token_index[:, None]))

  masked_input_idxs = tf.ragged.boolean_mask(
      tf.tile(idx_pool[None, :], [tf.size(start_token_index), 1]),
      word_token_idx_mask)
  valid_input_idx_mask = tf.ones_like(masked_input_idxs)

  batch['masked_targets'] = pad_first_dim(masked_targets,
                                          max_num_word_masks, 0)
  batch['masked_input_token_indices'] = masked_input_idxs.to_tensor(
      shape=[max_num_word_masks, max_num_masked_input_indices])
  batch['valid_input_token_index_mask'] = valid_input_idx_mask.to_tensor(
      shape=[max_num_word_masks, max_num_masked_input_indices])

  del batch['mwp_start_indices'], batch['mwp_end_indices']

  return batch


# Eval noise addition should be deterministic for each example.
def add_packet_loss_noise(wave_tensor, max_num_bursts, max_length_rate):
  """Simulate burst packet loss noise deterministically to the input `wave_tensor`.

  Args:
    wave_tensor: The input waveform signals to add noise to.
    max_num_bursts: The number of burst losses to simulate.
    max_length_rate: The maximum ratio of each burst loss length to the input
      signal length. The ratio is uniformly sampled for each burst loss in range
      (0, max_rength_rate].

  Returns:
    Noise injected waveform signals.
  """
  positive_sum = tf.cast(tf.reduce_sum(tf.maximum(wave_tensor, 0)), tf.int32)
  negative_sum = tf.cast(tf.reduce_sum(tf.maximum(-wave_tensor, 0)), tf.int32)

  max_length = tf.cast(
      tf.round(tf.cast(tf.size(wave_tensor), tf.float32) * max_length_rate),
      tf.int32)

  mask_lengths = tf.random.stateless_uniform([max_num_bursts],
                                             [positive_sum, negative_sum],
                                             1,
                                             max_length + 1,
                                             dtype=tf.int32)

  mask_start_idxs = tf.math.floormod(
      tf.abs(
          tf.random.stateless_uniform([max_num_bursts],
                                      [positive_sum + 1, negative_sum + 1],
                                      None,
                                      None,
                                      dtype=tf.int32)),
      tf.size(wave_tensor) - mask_lengths)

  mask_end_idxs = mask_start_idxs + mask_lengths

  timestamps = tf.repeat(
      tf.range(tf.size(wave_tensor))[None, :], max_num_bursts, 0)

  time_mask = tf.logical_and(
      tf.greater_equal(timestamps, mask_start_idxs[:, None]),
      tf.less(timestamps, mask_end_idxs[:, None]))
  time_mask = tf.cast(tf.reduce_any(time_mask, 0), tf.float32)

  noised_wave_tensor = wave_tensor * (1 - time_mask)

  return noised_wave_tensor


def add_environment_noise(wave_tensor, snr, noise_db):
  """Adds random environment noise deterministically chosen from `noise_db`.

  Args:
    wave_tensor: The input waveform signals to add noise to.
    snr: The signal to noise ratio for controlling the power of the noise.
    noise_db: A two dimensional tensor where the first dimension corresponds to
      the number of noise on which we sample the random noise.

  Returns:
    Noise injected waveform signals.
  """

  positive_sum = tf.cast(tf.reduce_sum(tf.maximum(wave_tensor, 0)), tf.int32)
  negative_sum = tf.cast(tf.reduce_sum(tf.maximum(-wave_tensor, 0)), tf.int32)

  snr = tf.cast(snr, wave_tensor.dtype)

  noise_idx = tf.random.stateless_uniform([], [positive_sum, negative_sum], 0,
                                          tf.shape(noise_db)[0], tf.int32)
  target_noise = tf.cast(noise_db[noise_idx], wave_tensor.dtype)

  pos = tf.reduce_mean(tf.square(wave_tensor))
  if tf.size(target_noise) >= tf.size(wave_tensor):
    start_idx = tf.random.stateless_uniform(
        [], [positive_sum + 1, negative_sum + 1], 0,
        tf.size(target_noise) - tf.size(wave_tensor) + 1, tf.int32)
    target_noise = target_noise[start_idx:start_idx + tf.size(wave_tensor)]
    nos = tf.reduce_mean(tf.square(target_noise))
  else:
    start_idx = tf.random.stateless_uniform(
        [], [positive_sum + 1, negative_sum + 1], 0,
        tf.size(wave_tensor) - tf.size(target_noise) + 1, tf.int32)
    nos = tf.reduce_mean(tf.square(target_noise))
    target_noise = tf.pad(target_noise, [[
        start_idx,
        tf.size(wave_tensor) - (start_idx + tf.size(target_noise))
    ]])

  multiplier = tf.math.divide_no_nan(
      1., tf.sqrt(tf.math.divide_no_nan(nos * snr, pos)))
  wave_tensor = wave_tensor + target_noise * multiplier

  return wave_tensor


def add_spectrogram_from_audio(
    parser_builder,
    sampler_builder,
    preprocessor_builder,
    input_feature_name='WAVEFORM/feature/floats',
    output_feature_name='spectrogram',
    is_training=True,
    # Wave related parameters (stride is always assumed to be 1).
    sample_rate: int = 16000,
    add_gaussian_noise: bool = False,
    add_masking_noise: bool = False,
    snr: Optional[float] = None,
    visualness_score_threshold: float = 0.2,
    random_mask_noise_rate: float = 0.4,
    max_word_len: int = 128,
    max_num_masks: int = -1,
    extend_mask_boundaries_ms: float = 0.,
    add_word_mask: bool = True,
    add_word_mask_info: bool = False,
    eval_noise_types: Sequence[str] = tuple(),
    environment_noise_configs: Optional[Dict[str, Any]] = None,
    environment_noise_path: Optional[str] = None,
    packet_loss_noise_configs: Optional[Dict[str, Any]] = None,
    aligned_caption_feature_name: str = 'caption/label/string',
    word_tokenizer: Optional[tokenizers.TextTokenizer] = None,
    max_num_word_tokens: int = 8,
    prepend_bos: bool = True,
    append_eos: bool = True,
    patch_size: Sequence[int] = tuple(),
    max_num_masked_input_indices: int = 64,
    # Spectrogram computation parameters.
    spectrogram_type: str = 'logmf',
    frame_length: int = 400,
    frame_step: int = 160,
    num_features: int = 80,
    lower_edge_hertz: float = 80.0,
    upper_edge_hertz: float = 7600.0,
    # Spectrogram related parameters.
    num_frames=5,
    spec_augment=True,
    spec_augment_params=None,
    zero_centering_image=False,
    dataset_mean=0.0,
    dataset_stddev=1.0,
):
  """Add audio spectrogram computed from waveform.

  Args:
    parser_builder: An instance of a builders.BaseParserBuilder.
    sampler_builder: An instance of a builders.SamplerBuilder.
    preprocessor_builder: An instance of a builders.PreprocessorBuilder.
    input_feature_name: Name of the feature in the input SequenceExample.
      Exposing this as an argument allows using this function for different
      image features.
    output_feature_name: Name of the feature in the output features dictionary.
    is_training: Whether or not in training mode.
    sample_rate: The sample rate of the input audio.
    add_gaussian_noise: Whether to add Gaussian noise.
    add_masking_noise: Whether to add word masking noise.
    snr: Signal-to-noise ratio used to inject white noise. For word masking
      noise, white noise is added to the masked region. If None, no white noise
      is added to the masked region.
    visualness_score_threshold: The threshold for determining the visual words.
    random_mask_noise_rate: The random mask noise sampling ratio. If 1, the word
      masking is deterministic. If below 1, the masking noise is random and each
      word is masked out with this chance.
    max_word_len: Maximum length of the words. This is used to pad and truncate
      noise_word_mask added to the batch dictionary in the word masking noise
      addition.
    max_num_masks: Maximum number of word masks to apply. -1 means unlimited.
    extend_mask_boundaries_ms: If set, the start and end boundaries of each
      maksing region in the signal is extended.
    add_word_mask: If set, add masks indicating masked words into the batch
      dict. Used for computing recovery rate of the masked words.
    add_word_mask_info: If set, add word masking related information such as
      timestamps and masks to the batch dict. Used for applying masked word
      prediction loss.
    eval_noise_types: A tuple of noise type strings used in evaluation. Each
      string should be either `environment_noise` or `packet_loss_noise`.
    environment_noise_configs: A config dict for environment noise addition.
      Used if eval_noise_types contain `environment_noise`. The dict contains
      `snr`.
    environment_noise_path: Path to the npy file containing noise waveforms in
      numpy array.
    packet_loss_noise_configs: A config dict for packet loss noise addition.
      Used if eval_noise_types contain `packet_loss_noise`. The dict contains
      `max_num_bursts` and `max_length_rate`.
    aligned_caption_feature_name: The feature name to parse to extract the
      caption aligned with the timestamps. Used for the masked word prediction.
    word_tokenizer: Tokenizer used to tokenize the masked words for masked word
      prediction.
    max_num_word_tokens: Maximum number of tokens for each masked word.
    prepend_bos: Whether to add BOS token to a tokenized word.
    append_eos: Whether to add EOS token to a tokenized word.
    patch_size: The patch size used in the network architecture for spectrogram
      token embedding. Used to compute the token index.
    max_num_masked_input_indices: Maximum number of masked input tokens
      (spectrogram tokens).
    spectrogram_type: The type of the spectrogram to be extracted from the
      waveform. Can be either `spectrogram`, `logmf`, and `mfcc`.
    frame_length: The length of each spectroram frame.
    frame_step: The stride of spectrogram frames.
    num_features: The number of spectrogram features.
    lower_edge_hertz: Lowest frequency to consider.
    upper_edge_hertz: Highest frequency to consider.  crop are used.
    num_frames: Number of seconds to sample per subclip.
    spec_augment: Whether to apply augmentation using SpecAugment.
    spec_augment_params: Dict of parameters for SpecAugment.
    zero_centering_image: If `True`, frames are normalized to values in [-1, 1].
      If `False`, values in [0, 1].
    dataset_mean: Mean of values over the dataset.
    dataset_stddev: Standard deviation of values of the dataset.
  """

  ##############################################################################
  ### Load audio signal and sample from the beginning.
  ##############################################################################
  # Keep audio signal.
  parser_builder.parse_feature(
      feature_name=input_feature_name,
      # Entire signal stored in one Feature.
      feature_type=tf.io.VarLenFeature(dtype=tf.float32),
      output_name=output_feature_name)

  # Densify and flatten.
  sampler_builder.add_fn(
      fn=lambda x: tf.reshape(tf.sparse.to_dense(x), [-1]),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_sparse_to_dense')

  # Add noise
  if eval_noise_types:
    for noise_type in eval_noise_types:
      if noise_type not in VALID_EVAL_NOISE_TYPES:
        raise ValueError(f'noise_type `{noise_type}` not supported.')
    if 'environment_noise' in eval_noise_types:
      # Load environment noise samples dumped in a numpy array in a .npy file.
      with tf.io.gfile.GFile(environment_noise_path, 'rb') as f:
        noise_db = tf.constant(np.load(f), tf.float32)
      sampler_builder.add_fn(
          fn=functools.partial(
              add_environment_noise,
              **environment_noise_configs,
              noise_db=noise_db),
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_add_environment_noise'
      )
    if 'packet_loss_noise' in eval_noise_types:
      sampler_builder.add_fn(
          fn=functools.partial(add_packet_loss_noise,
                               **packet_loss_noise_configs),
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_add_packet_loss_noise')

  if add_masking_noise:
    logging.info('Adding word masking noise to the signal.')
    # First, parse related data: visualness score, word start and end timestamps
    parser_builder.parse_feature(
        feature_name='clip/label/align/visualness',
        feature_type=tf.io.VarLenFeature(dtype=tf.float32),
        output_name='visualness_score',
        is_context=True)

    parser_builder.parse_feature(
        feature_name='clip/label/align/start_ms',
        feature_type=tf.io.VarLenFeature(dtype=tf.float32),
        output_name='word_start_timestamp',
        is_context=True)

    parser_builder.parse_feature(
        feature_name='clip/label/align/end_ms',
        feature_type=tf.io.VarLenFeature(dtype=tf.float32),
        output_name='word_end_timestamp',
        is_context=True)

    # Densify and flatten.
    sampler_builder.add_fn(
        fn=tf.sparse.to_dense,
        feature_name='visualness_score',
        fn_name='visualness_score_sparse_to_dense')

    sampler_builder.add_fn(
        fn=tf.sparse.to_dense,
        feature_name='word_start_timestamp',
        fn_name='word_start_timestamp_sparse_to_dense')

    sampler_builder.add_fn(
        fn=tf.sparse.to_dense,
        feature_name='word_end_timestamp',
        fn_name='word_end_timestamp_sparse_to_dense')

    sampler_builder.add_fn(
        fn=functools.partial(
            add_visual_word_masking_noise,
            wave_feature_name=output_feature_name,
            snr=snr,
            sample_rate=sample_rate,
            vis_threshold=visualness_score_threshold,
            max_word_len=max_word_len,
            mask_rate=random_mask_noise_rate,
            max_num_masks=max_num_masks,
            extend_boundaries_ms=extend_mask_boundaries_ms,
            add_word_mask=add_word_mask,
            add_word_mask_info=add_word_mask_info),
        fn_name=f'{output_feature_name}_add_masking_noise')
  elif add_gaussian_noise and snr is not None and snr > 0.0:
    logging.info('Adding Gaussian white noise to the signal.')
    sampler_builder.add_fn(
        fn=lambda x: add_white_noise(x, snr),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_add_white_noise')
  else:
    logging.info('Adding no noise to the signal.')

  # Normalize the waveform before spectrogram computation.
  sampler_builder.add_fn(
      fn=normalize_input,
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_normalize')

  ##############################################################################
  ### Compute spectrograms from the loaded audio.
  ##############################################################################
  # Extract audio spectrograms.
  sampler_builder.add_fn(
      functools.partial(
          processors.compute_audio_spectrogram,
          sample_rate=sample_rate,
          spectrogram_type=spectrogram_type,
          frame_length=frame_length,
          frame_step=frame_step,
          num_features=num_features,
          lower_edge_hertz=lower_edge_hertz,
          upper_edge_hertz=upper_edge_hertz,
          audio_feature_name=output_feature_name,
          spectrogram_feature_name=output_feature_name))

  ##############################################################################
  ### Generate meta data needed for masked word prediction.
  ##############################################################################
  if add_masking_noise and add_word_mask_info:
    sampler_builder.add_fn(
        fn=functools.partial(
            convert_ms_to_frame_number,
            spectrogram_feature_name=output_feature_name),
        fn_name='word_mask_info_convert_ms_to_frame_number')
    add_text_untokenized(
        parser_builder, sampler_builder, aligned_caption_feature_name,
        'mwp_caption_string')
    sampler_builder.add_fn(
        fn=functools.partial(
            tokenize_masked_words,
            tokenizer=word_tokenizer,
            max_num_tokens=max_num_word_tokens,
            prepend_bos=prepend_bos,
            append_eos=append_eos),
        fn_name='word_mask_info_tokenize_masked_words')

  ##############################################################################
  ### Preprocess the computed spectrograms.
  ##############################################################################

  # We apply spec_augment after the signal truncation but before padding.
  # In this way, we apply spec_augment to the valid signals only excluding
  # padding and the truncated non-target signals.

  # Temporal sampling (beginning_sample)
  num_time_bins = tf.cast(
      num_frames * (sample_rate / frame_step), dtype=tf.int32)
  preprocessor_builder.add_fn(
      fn=lambda x: x[:num_time_bins],
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_beginning_sample')

  preprocessor_builder.add_fn(
      # pylint: disable=g-long-lambda
      fn=lambda x: _decode_spectrogram(x, True, zero_centering_image,
                                       dataset_mean, dataset_stddev),
      # pylint: enable=g-long-lambda
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_decode_spectrogram')

  if is_training and spec_augment:
    # Apply specaugment
    preprocessor_builder.add_fn(
        fn=lambda x, s=None: apply_specaugment(x, spec_augment_params),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_specaugment')

  # Padding
  pad_value = .0
  preprocessor_builder.add_fn(
      fn=lambda x: pad_first_dim(x, num_time_bins, pad_value),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_pad')

  ##############################################################################
  ### Finalize the word mask info by truncating, padding and converting indices.
  ##############################################################################
  if add_masking_noise and add_word_mask_info:
    preprocessor_builder.add_fn(
        fn=functools.partial(
            finalize_word_mask_info,
            spectrogram_feature_name=output_feature_name,
            patch_size=patch_size,
            max_num_word_masks=max_num_masks,
            max_num_masked_input_indices=max_num_masked_input_indices),
        fn_name='finalize_word_mask_info')


def add_spectrogram(
    parser_builder,
    sampler_builder,
    decoder_builder,
    preprocessor_builder,
    postprocessor_builder,
    input_feature_name='melspec/feature/floats',
    input_shape=(100, 128),  # (frames, num_mel_bins)
    output_feature_name='spectrogram',
    is_training=True,
    num_frames=5,
    num_test_clips=1,
    spec_augment=True,
    spec_augment_params=None,
    zero_centering_image=False,
    dataset_mean=0.0,
    dataset_stddev=1.0,):
  """Add audio spectrogram.

  Args:
    parser_builder: An instance of a builders.BaseParserBuilder.
    sampler_builder: An instance of a builders.SamplerBuilder.
    decoder_builder: An instance of a builders.DecoderBuilder.
    preprocessor_builder: An instance of a builders.PreprocessorBuilder.
    postprocessor_builder: An instance of a builders.PostprocessorBuilder.
    input_feature_name: Name of the feature in the input SequenceExample.
      Exposing this as an argument allows using this function for different
      image features.
    input_shape: Shape of the input spectrogram.
    output_feature_name: Name of the feature in the output features dictionary.
    is_training: Whether or not in training mode. If True, random sample, and
      crop are used.
    num_frames: Number of seconds to sample per subclip.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggreagated in the batch dimension.
    spec_augment: Whether to apply augmentation using SpecAugment.
    spec_augment_params: Dict of parameters for SpecAugment.
    zero_centering_image: If `True`, frames are normalized to values in [-1, 1].
      If `False`, values in [0, 1].
    dataset_mean: Mean of values over the dataset.
    dataset_stddev: Standard deviation of values of the dataset.
  """
  if is_training and num_test_clips != 1:
    logging.info('`num_test_clips` %d is ignored since `is_training` is true.',
                 num_test_clips)
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.FixedLenSequenceFeature(
            shape=input_shape, dtype=tf.float32),
        output_name=output_feature_name)
  else:
    raise ValueError('`parser_builder` has an unexpected type.')

  # Temporal sampler.
  num_time_bins = num_frames * input_shape[0]
  sampler_builder.add_fn(
      fn=lambda x: tf.reshape(x, (-1, input_shape[1])),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_sampler_reshape')

  # Get the first num_time_bins of the sequence.
  sampler_builder.add_fn(
      fn=lambda x: x[:num_time_bins],
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_beginning_sample')

  # pylint: disable=g-long-lambda
  decoder_builder.add_fn(
      fn=lambda x: _decode_spectrogram(x, True, zero_centering_image,
                                       dataset_mean, dataset_stddev),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_decode_spectrogram')
  # pylint: enable=g-long-lambda

  if is_training and spec_augment:
    # Apply specaugment
    preprocessor_builder.add_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x, s=None: apply_specaugment(x, spec_augment_params),
        # pylint: enable=g-long-lambda
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_specaugment')

  # Pad if necessary.
  pad_value = .0
  preprocessor_builder.add_fn(
      fn=lambda x: pad_first_dim(x, num_time_bins, pad_value),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_pad')

  if num_test_clips > 1 and not is_training:
    # In this case, multiple clips are merged together in batch dimenstion which
    # will be `B * num_test_clips`.
    postprocessor_builder.add_fn(
        fn=lambda x: tf.reshape(  # pylint: disable=g-long-lambda
            x, (-1, num_time_bins, x.shape[2], x.shape[3])),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_reshape')


def add_text(
    parser_builder: builders.BaseParserBuilder,
    decoder_builder: builders.DecoderBuilder,
    preprocessor_builder: builders.PreprocessorBuilder,
    tokenizer: tokenizers.TextTokenizer,
    is_training: bool = True,
    input_feature_name: str = 'caption/string',
    output_raw_string_name: str = 'raw_caption',
    output_feature_name: str = builders.TEXT_INDICES_FEATURE_NAME,
    # Text related parameters.
    prepend_bos: bool = False,
    append_eos: bool = False,
    keep_raw_string: bool = False,
    max_num_captions: int = 1,
    max_num_tokens: int = 16,
    sync_random_state: bool = False):
  """Adds functions to process text feature to builders.

  This function expects the input to be either a `tf.train.SequenceExample`
  (with the features in the context) or a `tf.train.Example`. The expected
  structure is (or equivalent for `tf.train.Example`):
  ```
  context {
    feature {
      key: input_feature_name
      value {
        bytes_list {
          value: "Hello world!"
          value: "This is a caption."
          ...
        }
      }
    }
  }
  ```

  The corresponding `builders.ExampleParserBuilder` or
  `builders.SequenceExampleParserBuilder` has to be given as parameter.

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    decoder_builder: An instance of a `builders.DecoderBuilder`.
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    tokenizer: An instance of a tokenizer.
    is_training: Whether or not in training mode. This will be used to randomly
      sample the captions.
    input_feature_name: Name of the feature in the input `tf.train.Example` or
      `tf.train.SequenceExample`. Exposing this as an argument allows using this
      function for different text features within a single dataset.
    output_raw_string_name: Name of the raw string in the output features
      dictionary. Exposing this as an argument allows using this function for
      different text features within a single dataset.
    output_feature_name: Name of the feature in the output features dictionary.
      Exposing this as an argument allows using this function for different text
      features.
    prepend_bos: Whether to prepend BOS token.
    append_eos: Whether to append EOS token.
    keep_raw_string: Whether to keep raw string.
    max_num_captions: Maximum number of captions to keep. If there are more
      captions in the proto, only the first `max_num_captions` will be returned
      is `is_training` is set to `False`. If `is_training` is `True`, then
      `max_num_captions` will be randomly sampled. Finally if the proto contains
      less than `max_num_captions`, we pad with empty srings to make sure there
      are `max_num_captions` in total.
    max_num_tokens: Maximum number of tokens to keep from the text for each
      caption. If there are more tokens, sequence is cropped, if less, the
      caption is padded using the tokenizer pad id. The sequence is unmodified
      if max_num_tokens is None.
    sync_random_state: Whether to use stateful option to keep random operations
      in sync between different modalities. All modalities having this option
      `True` will use the same outcome in random operations used for sampling
      the captions.
  """
  # Parse text indices.
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.VarLenFeature(dtype=tf.string),
        output_name=output_raw_string_name,
        is_context=True)
  elif isinstance(parser_builder, builders.ExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.VarLenFeature(dtype=tf.string),
        output_name=output_raw_string_name)

  # Densify text tensor.
  decoder_builder.add_fn(
      fn=tf.sparse.to_dense,
      feature_name=output_raw_string_name,
      fn_name=f'{output_feature_name}_sparse_to_dense')

  preprocessor_builder.add_fn(
      # pylint: disable=g-long-lambda
      lambda x, s=None: processors.sample_or_pad_non_sorted_sequence(
          x, max_num_captions, b'', random=is_training, state=s),
      # pylint: enable=g-long-lambda
      feature_name=output_raw_string_name,
      fn_name=f'{output_feature_name}_sample_captions',
      # Use state to keep coherence between modalities if requested.
      stateful=sync_random_state)

  preprocessor_builder.add_fn(
      fn=lambda x: format_text(  # pylint: disable=g-long-lambda
          x, output_raw_string_name, output_raw_string_name),
      fn_name=f'{output_feature_name}_formatting')

  # Tokenize the sentence.
  preprocessor_builder.add_fn(
      fn=lambda x: processors.tokenize(  # pylint: disable=g-long-lambda
          x, tokenizer, output_raw_string_name, output_feature_name,
          prepend_bos, append_eos, max_num_tokens, keep_raw_string),
      fn_name=f'{output_feature_name}_tokenization')

  if max_num_tokens is not None:
    # Set text shape.
    shape = (max_num_captions, max_num_tokens)
    preprocessor_builder.add_fn(
        fn=lambda x: processors.set_shape(x, shape),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_set_shape')


def add_text_untokenized(
    parser_builder: builders.BaseParserBuilder,
    decoder_builder: builders.DecoderBuilder,
    input_feature_name: str = 'caption/string',
    output_raw_string_name: str = builders.TEXT_FEATURE_NAME):
  """Adds functions to process text feature to builders."""

  # Parse text indices.
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.VarLenFeature(dtype=tf.string),
        output_name=output_raw_string_name,
        is_context=True)
  elif isinstance(parser_builder, builders.ExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.VarLenFeature(dtype=tf.string),
        output_name=output_raw_string_name)

  # Densify text tensor.
  decoder_builder.add_fn(
      fn=tf.sparse.to_dense,
      feature_name=output_raw_string_name,
      fn_name=f'{output_raw_string_name}_sparse_to_dense')


def add_int64(parser_builder: builders.BaseParserBuilder,
              decoder_builder: builders.DecoderBuilder,
              feature_name: str = 'clip/label/index',
              output_name: str = 'label_index'):
  """Adds functions to process integer feature to builders.

  This function expects the input to be either a `tf.train.SequenceExample`
  (with the features in the context) or a `tf.train.Example`. The expected
  structure is (or equivalent for `tf.train.Example`):
  ```
  context {
    feature {
      key: input_label_index_feature_name
      value {
        int64_list {
          value: 42
          ...
        }
      }
    }
  }
  ```

  The corresponding `builders.ExampleParserBuilder` or
  `builders.SequenceExampleParserBuilder` has to be given as parameter.

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    decoder_builder: An instance of a `builders.DecoderBuilder`.
    feature_name: Name of the label index feature in the input
      `tf.train.Example` or `tf.train.SequenceExample`. Exposing this as an
      argument allows using this function for different label features within a
      single dataset.
    output_name: Name of the label index feature in the output features
      dictionary. Exposing this as an argument allows using this function for
      different label features within a single dataset.
  """

  # Parse label.
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=feature_name,
        feature_type=tf.io.VarLenFeature(dtype=tf.int64),
        output_name=output_name,
        is_context=True)
  elif isinstance(parser_builder, builders.ExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=feature_name,
        feature_type=tf.io.VarLenFeature(dtype=tf.int64),
        output_name=output_name)
  else:
    raise ValueError('`parser_builder` has an unexpected type.')

  # Densify tensor
  decoder_builder.add_fn(
      fn=tf.sparse.to_dense,
      feature_name=output_name,
      fn_name=f'{output_name}_sparse_to_dense')


def custom_standardization(input_data):
  # Removes punctuation
  lowercase = tf.strings.lower(input_data)
  ret = tf.strings.regex_replace(
      lowercase,
      '[!"\\#\\$%\\&\\(\\)\\*\\+,\\-\\./:;<=>\\?@\\[\\\\\\]\\^_`\\{\\|\\}\\~]',
      '')
  return ret


def format_text(features: builders.FeaturesDict, raw_string_name: str,
                formated_name: str,
                keep_raw_string: bool = False) -> builders.FeaturesDict:
  """Tokenize raw string with tokenizer."""
  raw_caption = features[raw_string_name]

  formated = custom_standardization(raw_caption)

  if not keep_raw_string:
    del features[raw_string_name]

  features[formated_name] = formated
  return features
