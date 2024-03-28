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

"""Utility functions for datasets that process audio spectrogram."""

from absl import logging
from dmvr import builders
from dmvr import processors
from lingvo.core import spectrum_augmenter
import tensorflow as tf


def apply_specaugment(spec: tf.Tensor, spec_augment_params=None):
  """Performs SpecAugment on the inputs.

  SpecAugment is a data augmentation technique from arXiv:1904.08779,
  that combines three transformations:
   - a time warping of up to max(time_warp_max_frames,
   time_warp_max_ratio*input_length) frames.
   - a masking of sampled frequencies with zeros along the entire time axis
   (freq_mask)
   - a masking of sampled timesteps with zeros along the entire frequency axis
   (time_mask)

  Args:
    spec: input mel spectrogram of shape [num_clips, time, freq, num_channels]
      or [time, freq, num_channels].
    spec_augment_params: dictionary containing the following -
      freq_mask_max_bins (int), max number of consecutive mel bins to mask in a
      band. - freq_mask_count (int), number of frequency bands to mask. -
      time_mask_max_frames (int), max number of consecutive time frames to mask.
      - time_mask_count (int), number of time bands to mask. -
      time_mask_max_ratio (float), max time mask ratio. - time_warp_max_frames
      (int), max numer of time frames to warp. - time_warp_max_ratio (int), max
      ratio of the time warp.

  Returns:
    Augmented mel spectrogram of shape (num_time_bins, num_freq_bins, channels)
    or
      (num_clips, num_time_bins, num_freq_bins, channels).
  """
  spec_augment_params_obj = spectrum_augmenter.SpectrumAugmenter.Params()
  spec_augment_params_obj.freq_mask_max_bins = spec_augment_params.freq_mask_max_bins
  spec_augment_params_obj.freq_mask_count = spec_augment_params.freq_mask_count
  spec_augment_params_obj.time_mask_max_frames = spec_augment_params.time_mask_max_frames
  spec_augment_params_obj.time_mask_count = spec_augment_params.time_mask_count
  spec_augment_params_obj.time_warp_max_frames = spec_augment_params.time_warp_max_frames
  spec_augment_params_obj.time_warp_max_ratio = spec_augment_params.time_warp_max_ratio
  spec_augment_params_obj.time_mask_max_ratio = spec_augment_params.time_mask_max_ratio
  spec_augment_params_obj.name = 'specaugment'
  spec_augment_layer = spec_augment_params_obj.Instantiate()

  squeeze_axis = []
  if spec.shape.ndims == 3:
    spec = spec[None, :, :, :]
    squeeze_axis = [0]
  elif spec.shape.ndims != 4:
    raise ValueError('Spectrogram shape must have 3 or 4 dimensions')

  outputs, _ = spec_augment_layer.FPropDefaultTheta(
      spec, tf.zeros(tf.shape(spec)[:2]))
  if squeeze_axis:
    outputs = tf.squeeze(outputs, axis=squeeze_axis)
  return outputs


def _decode_spectrogram(spectrogram,
                        inflate=True,
                        zero_centering=True,
                        dataset_mean=0,
                        dataset_stddev=1):

  """Decodes audio spectrogram.

  Args:
    spectrogram: input mel spectrogram
    inflate: if True, adds a channel dimension
    zero_centering: if True, zero centers the spectrogram
    dataset_mean: mean over the dataset.
    dataset_stddev: standard deviation over the dataset.

  Returns:
    spectrogram: decoded spectrogram.

  """
  if inflate:
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = tf.tile(spectrogram, [1, 1, 3])

  # normalize spectrogram by mean and std deviation
  spectrogram = spectrogram - dataset_mean
  spectrogram = spectrogram / dataset_stddev
  if not zero_centering:
    spectrogram = spectrogram + 1.0
    spectrogram = spectrogram / 2
  return spectrogram


def add_spectrogram(parser_builder,
                    sampler_builder,
                    decoder_builder,
                    preprocessor_builder,
                    postprocessor_builder,
                    input_feature_name='melspec/feature/floats',
                    input_shape=(100, 128),  # (frames, num_mel_bins)
                    output_feature_name='spectrogram',
                    is_training=True,
                    num_frames=5,
                    stride=1,
                    num_test_clips=1,
                    spec_augment=True,
                    spec_augment_params=None,
                    zero_centering_image=False,
                    dataset_mean=0.0,
                    dataset_stddev=1.0,
                    sync_random_state=True):
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
    stride: Temporal stride to sample frames.
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
    sync_random_state: Whether to use stateful option to keep random operations
      in sync between different modalities. All modalities having this option
      True will use the same outcome in random operations such as sampling and
      cropping.
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
  if is_training:
    # Sample random clip.
    sampler_builder.add_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x, s=None: processors.sample_sequence(
            x, num_time_bins, True, stride, state=s),
        # pylint: enable=g-long-lambda
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_random_sample',
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state)
  else:
    if num_test_clips > 1:
      # Sample linspace clips.
      sampler_builder.add_fn(
          # pylint: disable=g-long-lambda
          fn=lambda x: processors.sample_linspace_sequence(
              x, num_test_clips, num_time_bins, stride),
          # pylint: enable=g-long-lambda
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_linspace_sample')
    else:
      # Sample middle clip.
      sampler_builder.add_fn(
          # pylint: disable=g-long-lambda
          fn=lambda x: processors.sample_sequence(
              x, num_time_bins, False, stride),
          # pylint: enable=g-long-lambda
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_middle_sample')
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
        fn=lambda x, s=None: apply_specaugment(
            x, spec_augment_params),
        # pylint: enable=g-long-lambda
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_specaugment')

  if num_test_clips > 1 and not is_training:
    # In this case, multiple clips are merged together in batch dimenstion which
    # will be `B * num_test_clips`.
    postprocessor_builder.add_fn(
        fn=lambda x: tf.reshape(  # pylint: disable=g-long-lambda
            x, (-1, num_time_bins, x.shape[2], x.shape[3])),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_reshape')



