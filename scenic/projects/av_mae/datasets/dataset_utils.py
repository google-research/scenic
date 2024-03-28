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

"""Utilities for adding modalities.


Forked from:
https://github.com/google-deepmind/dmvr/blob/master/dmvr/modalities.py.
"""

from typing import Optional
from typing import Union

from absl import logging
from dmvr import builders
from dmvr import processors
from lingvo.core import spectrum_augmenter
import tensorflow as tf


def crop_and_resize_image_vmae(frames: tf.Tensor,
                               resized_size: tuple[int, int] = (224, 224),
                               scales: tf.Tensor = tf.constant(
                                   [1, .875, .75, .66])) -> tf.Tensor:
  """Crops and resizes the images in the given sequence of images.

  Args:
    frames: A tensor of dimension [timesteps, input_h, input_w, channels].
    resized_size: The size for the resize operation.
    scales: The scales for the resize operation. Must be a tensor with 4 values.
  Returns:
    A tensor of shape [timesteps, output_h, output_w, channels] of same type as
    input with the cropped and resized images.
  """

  shape = tf.shape(input=frames)
  timesteps = shape[0]
  image_h = shape[1]
  image_w = shape[2]
  channels = shape[3]

  crop_h, crop_w, offset_h, offset_w = sample_crop_size(
      image_h=image_h, image_w=image_w, resized_size=resized_size,
      scales=scales)
  # offset [0, offset_h, offset_w, 0]
  # size  [timesteps, height, width, channels]

  offset = tf.convert_to_tensor(value=(0, offset_h, offset_w, 0))
  size = tf.convert_to_tensor(value=(timesteps, crop_h, crop_w, channels))
  frames = tf.slice(frames, offset, size)
  frames = tf.image.resize(frames, resized_size)

  return frames


def sample_fixed_offset(image_w: int, image_h: int, crop_w: int, crop_h: int,
                        more_fix_crop: bool = True) -> tf.Tensor:
  """Sample offset of the crop out of 13 fixed offsets.

  The sampling strategy is taken from: https://arxiv.org/abs/2203.12602, Github:
  https://github.com/MCG-NJU/VideoMAE.

  Args:
    image_w: The width of the image.
    image_h: The height of the image.
    crop_w: The width of the crop.
    crop_h: The height of the crop.
    more_fix_crop: Add another 8 fixed crops to the sampling.

  Returns:
    A tensor of shape [1, 2] with the corresponding offset
    [[offset_w, offset_h]].
  """
  w_step = (image_w - crop_w) // 4
  h_step = (image_h - crop_h) // 4

  ret = list()
  ret.append((tf.constant(0), tf.constant(0)))  # upper left
  ret.append((4 * w_step, 0))  # upper right
  ret.append((0, 4 * h_step))  # lower left
  ret.append((4 * w_step, 4 * h_step))  # lower right
  ret.append((2 * w_step, 2 * h_step))  # center

  if more_fix_crop:
    ret.append((0, 2 * h_step))  # center left
    ret.append((4 * w_step, 2 * h_step))  # center right
    ret.append((2 * w_step, 4 * h_step))  # lower center
    ret.append((2 * w_step, 0 * h_step))  # upper center

    ret.append((1 * w_step, 1 * h_step))  # upper left quarter
    ret.append((3 * w_step, 1 * h_step))  # upper right quarter
    ret.append((1 * w_step, 3 * h_step))  # lower left quarter
    ret.append((3 * w_step, 3 * h_step))  # lower right quarter

  ret_index = tf.random.uniform((1, 1), minval=0, maxval=len(ret),
                                dtype=tf.int32)[0, 0]
  ret = tf.stack(ret)

  ret_pair = tf.slice(ret, [ret_index, 0], [1, 2])
  return ret_pair


def sample_crop_size(image_h: int, image_w: int,
                     resized_size: tuple[int, int], scales: tf.Tensor,
                     max_distort: int = 1) -> tuple[int, int, int, int]:
  """Sample a crop size and the offset out of fixed choices.

  Args:
    image_h: The height of the image.
    image_w: The width of the image.
    resized_size: The size of the resized image.
    scales: The scales for the resize operation.
    max_distort: How many adjact possitions in the scales array to combine in
    order to get the pairs for the resize options.

  Returns:
    A tuple of 4 elements -> [crop_h, crop_w, offset_h, offset_w].

  """

  if len(scales) != 4:
    raise NotImplementedError('Only 4 values are supported for the scale.')

  base_size = tf.cast(tf.minimum(image_w, image_h), tf.float32)

  crop_sizes = [tf.cast(base_size * scales[0], tf.int32),
                tf.cast(base_size * scales[1], tf.int32),
                tf.cast(base_size * scales[2], tf.int32),
                tf.cast(base_size * scales[3], tf.int32)]
  rsize_h, rsize_w = resized_size

  crop_h = [
      rsize_h if abs(crop_sizes[0] - rsize_h) < 3 else crop_sizes[0],
      rsize_h if abs(crop_sizes[1] - rsize_h) < 3 else crop_sizes[1],
      rsize_h if abs(crop_sizes[2] - rsize_h) < 3 else crop_sizes[2],
      rsize_h if abs(crop_sizes[3] - rsize_h) < 3 else crop_sizes[3]]

  crop_w = [
      rsize_w if abs(crop_sizes[0] - rsize_w) < 3 else crop_sizes[0],
      rsize_w if abs(crop_sizes[1] - rsize_w) < 3 else crop_sizes[1],
      rsize_w if abs(crop_sizes[2] - rsize_w) < 3 else crop_sizes[2],
      rsize_w if abs(crop_sizes[3] - rsize_w) < 3 else crop_sizes[3]]

  # Get the resized pairs.
  pairs = []
  for i, h in enumerate(crop_h):
    for j, w in enumerate(crop_w):
      if abs(i - j) <= max_distort:
        pairs.append((w, h))

  # Implement random.choice.
  crop_pair_index = tf.random.uniform((1, 1), minval=0, maxval=len(pairs),
                                      dtype=tf.int32)[0, 0]
  pairs = tf.stack(pairs)
  crop_pair = tf.slice(pairs, [crop_pair_index, 0], [1, 2])

  offset = sample_fixed_offset(image_w=image_w, image_h=image_h,
                               crop_w=crop_pair[0][0], crop_h=crop_pair[0][1])
  return crop_pair[0][1], crop_pair[0][0], offset[0][1], offset[0][0]


def add_image(
    parser_builder: builders.BaseParserBuilder,
    sampler_builder: builders.SamplerBuilder,
    decoder_builder: builders.DecoderBuilder,
    preprocessor_builder: builders.PreprocessorBuilder,
    postprocessor_builder: builders.PostprocessorBuilder,
    input_feature_name: str = 'image/encoded',
    output_feature_name: str = builders.IMAGE_FEATURE_NAME,
    is_training: bool = True,
    # Video related parameters.
    num_frames: int = 32,
    stride: int = 1,
    num_test_clips: int = 1,
    min_resize: int = 224,
    resize_method: str = tf.image.ResizeMethod.BILINEAR,
    crop_size: int = 200,
    use_crop_and_resize_video_mae: bool = False,
    train_frame_sampling_mode: Optional[str] = None,
    zero_centering_image: bool = False,
    sync_random_state: bool = True,
    is_rgb: Optional[bool] = True,
    is_flow: bool = False,
    random_flip: bool = True,
    normalization_mean: Union[tf.Tensor, float] = 0,
    normalization_std: Union[tf.Tensor, float] = 1,
) -> None:
  """Adds functions to process image feature to builders.

  This function expects the input to be either a `tf.train.SequenceExample` (for
  videos) and have the following structure:
  ```
  feature_lists {
    feature_list {
      key: input_feature_name
      value {
        feature {
          bytes_list {
            value: jpeg_bytes
          }
        }
      }
    }
  }
  ```

  Or a `tf.train.Example` (for image only) and have the following structure:
  ```
  features {
    feature {
      key: input_feature_name
      value {
        bytes_list {
          value: "JPEG"
        }
      }
    }
  }
  ```

  The corresponding `builders.ExampleParserBuilder` or
  `builders.SequenceExampleParserBuilder` has to be given as parameter.

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    sampler_builder: An instance of a `builders.SamplerBuilder`.
    decoder_builder: An instance of a `builders.DecoderBuilder`.
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    postprocessor_builder: An instance of a `builders.PostprocessorBuilder`.
    input_feature_name: Name of the feature in the input `tf.train.Example` or
      `tf.train.SequenceExample`. Exposing this as an argument allows using this
      function for different image features within a single dataset.
    output_feature_name: Name of the feature in the output features dictionary.
      Exposing this as an argument allows using this function for different
      image features within a single dataset.
    is_training: Whether in training mode. If `True`, random sample, crop and
      left right flip is used.
    num_frames: Number of frames per subclip. For single images, use 1.
    stride: Temporal stride to sample frames.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggregated in the batch dimension.
    min_resize: Frames are resized so that `min(height, width)` is `min_resize`.
    resize_method: A resizing method.
    crop_size: Final size of the frame after cropping the resized frames. Both
      height and width are the same.
    use_crop_and_resize_video_mae: If True cropping stragy used by VideoMAE of
      Tong et al. will be used.
    train_frame_sampling_mode: The temporal sampling strategy used in the
    training.
    zero_centering_image: If `True`, frames are normalized to values in [-1, 1].
      If `False`, values in [0, 1].
    sync_random_state: Whether to use stateful option to keep random operations
      in sync between different modalities. All modalities having this option
      `True` will use the same outcome in random operations such as sampling and
      cropping.
    is_rgb: If `True`, the number of channels in the JPEG is 3, if False, 1. If
      is_flow is `True`, `is_rgb` should be set to `None` (see below).
    is_flow: If `True`, the image is assumed to contain flow and will be
      processed as such. Note that the number of channels in the JPEG for flow
      is 3, but only two channels will be output corresponding to the valid
      horizontal and vertical displacement.
    random_flip: If `True`, a random horizontal flip is applied to the input
      image. This augmentation may not be used if the label set contains
      direction related classes, such as `pointing left`, `pointing right`, etc.
    normalization_mean: value to subtract from the input image to normalize it.
    normalization_std: value to divide by from the input image to normalize it.
  """

  # Validate parameters.
  if is_flow and is_rgb is not None:
    raise ValueError('`is_rgb` should be `None` when requesting flow.')

  if is_flow and not zero_centering_image:
    raise ValueError('Flow contains displacement values that can be negative, '
                     'but `zero_centering_image` was set to `False`.')

  if is_training and num_test_clips != 1:
    logging.info('`num_test_clips` %d is ignored since `is_training` is true.',
                 num_test_clips)

  # Parse frames or single image.
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.FixedLenSequenceFeature((), dtype=tf.string),
        output_name=output_feature_name)
  elif isinstance(parser_builder, builders.ExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.FixedLenFeature((), dtype=tf.string),
        output_name=output_feature_name)
    # Expand dimensions so single images have the same structure as videos.
    sampler_builder.add_fn(
        fn=lambda x: tf.expand_dims(x, axis=0),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_expand_dims')
  else:
    raise ValueError('`parser_builder` has an unexpected type.')

  # Temporal sampler.
  if is_training:
    # Sample random clip.
    sampler_builder.add_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x, s=None: processors.sample_sequence(
            x, num_frames, True, stride, state=s),
        # pylint: enable=g-long-lambda
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_random_sample',
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state)
  else:
    if num_test_clips > 1:
      if train_frame_sampling_mode == 'segment':
        if num_test_clips != 2:
          raise ValueError('For segment sampling only 2 video clips at test'
                           'are implemented.')
        sampler_builder.add_fn(
            fn=lambda x: sample_two_sequences_uniformly(x, num_frames),
            feature_name=output_feature_name,
            fn_name=f'{output_feature_name}_segment_sample')
      else:
        # Sample linspace clips.
        sampler_builder.add_fn(
            # pylint: disable=g-long-lambda
            fn=lambda x: processors.sample_linspace_sequence(
                x, num_test_clips, num_frames, stride),
            # pylint: enable=g-long-lambda
            feature_name=output_feature_name,
            fn_name=f'{output_feature_name}_linspace_sample')
    else:
      if train_frame_sampling_mode == 'segment':
        sampler_builder.add_fn(
            # pylint: disable=g-long-lambda
            fn=lambda x: sample_sequence_uniformly(x, num_frames,
                                                   is_training=is_training),
            # pylint: enable=g-long-lambda
            feature_name=output_feature_name,
            fn_name=f'{output_feature_name}_segment_sample_train')
      else:
        # Sample middle clip.
        sampler_builder.add_fn(
            # pylint: disable=g-long-lambda
            fn=lambda x: processors.sample_sequence(x,
                                                    num_frames, False, stride),
            # pylint: enable=g-long-lambda
            feature_name=output_feature_name,
            fn_name=f'{output_feature_name}_middle_sample')

  # Decode JPEG string to `tf.uint8`.
  # Note that for flow, 3 channels are stored in the JPEG: the first two
  # corresponds to horizontal and vertical displacement, respectively.
  # The last channel contains zeros and is dropped later in the preprocessing.
  # Hence, the output number of channels for flow is 2.
  num_raw_channels = 3 if (is_rgb or is_flow) else 1
  decoder_builder.add_fn(
      fn=lambda x: processors.decode_jpeg(x, channels=num_raw_channels),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_decode_jpeg')

  if is_flow:
    # Cast the flow to `tf.float32`, normalizing between [-1.0, 1.0].
    preprocessor_builder.add_fn(
        fn=lambda x: processors.normalize_image(x, zero_centering_image=True),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_normalize')

  # Resize images (resize happens only if necessary to save compute).
  preprocessor_builder.add_fn(
      # pylint: disable=g-long-lambda
      fn=lambda x: processors.resize_smallest(
          x, min_resize, is_flow=is_flow, method=resize_method),
      # pylint: enable=g-long-lambda
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_resize_smallest')

  if is_training:
    # Standard image data augmentation: random crop and random flip.
    if use_crop_and_resize_video_mae:
      preprocessor_builder.add_fn(
          # pylint: disable=g-long-lambda
          fn=lambda x, s=None: crop_and_resize_image_vmae(
              x),
          # pylint: enable=g-long-lambda
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_crop_and_resize',
          # Use state to keep coherence between modalities if requested.
          stateful=sync_random_state)
    else:
      preprocessor_builder.add_fn(
          # pylint: disable=g-long-lambda
          fn=lambda x, s=None: processors.crop_image(
              x, crop_size, crop_size, True, state=s),
          # pylint: enable=g-long-lambda
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_random_crop',
          # Use state to keep coherence between modalities if requested.
          stateful=sync_random_state)
    if random_flip:
      preprocessor_builder.add_fn(
          # pylint: disable=g-long-lambda
          fn=lambda x, s=None: processors.random_flip_left_right(
              x, state=s, is_flow=is_flow),
          # pylint: enable=g-long-lambda
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_random_flip',
          # Use state to keep coherence between modalities if requested.
          stateful=sync_random_state)
  else:
    # Central crop of the frames.
    preprocessor_builder.add_fn(
        fn=lambda x: processors.crop_image(x, crop_size, crop_size, False),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_central_crop')

  if is_flow:
    # Keep only two channels for the flow: horizontal and vertical displacement.
    preprocessor_builder.add_fn(
        fn=lambda x: x[:, :, :, :2],
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_extract_flow_channels')

    # Clip the flow to stay between [-1.0 and 1.0]
    preprocessor_builder.add_fn(
        fn=lambda x: tf.clip_by_value(x, -1.0, 1.0),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_clip_flow')
  else:
    # Cast the frames to `tf.float32`, normalizing according to
    # `zero_centering_image`.
    preprocessor_builder.add_fn(
        fn=lambda x: processors.normalize_image(x, zero_centering_image),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_normalize')

  preprocessor_builder.add_fn(
      fn=lambda x: x - normalization_mean,
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_subtract_given_mean')

  preprocessor_builder.add_fn(
      fn=lambda x: x / normalization_std,
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_divide_by_given_std')

  if num_test_clips > 1 and not is_training:
    # In this case, multiple clips are merged together in batch dimension which
    # will be `B * num_test_clips`.
    postprocessor_builder.add_fn(
        fn=lambda x: tf.reshape(  # pylint: disable=g-long-lambda
            x, (-1, num_frames, x.shape[2], x.shape[3], x.shape[4])),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_reshape')


def sample_sequence_uniformly(
    sequence: tf.Tensor,
    num_steps: int,
    is_training: bool = True) -> tf.Tensor:
  """Uniform frame sampling.

  Sample frames based on uniform sampling following TSN (Wang et al., 2019)
  used by Tong et al. in VideoMAE. The stride is automatically computed based on
  the length of the sequence and the number of frames to take (`num_steps`). If
  `is_training` is set to False, a deterministic sequence will be returned.

  Args:
    sequence: Any tensor where the first dimension is timesteps.
    num_steps: Number of steps (e.g. frames) to take.
    is_training: If is called during training or not.
  Returns:
     A single tensor with first dimension `num_steps` with the sampled segment.
  """

  sequence_length = tf.shape(input=sequence)[0]
  sequence_length = tf.cast(sequence_length, tf.int32)
  stride = tf.cast(sequence_length // num_steps, tf.int32)

  if stride > 0:
    indices = tf.math.multiply(tf.range(num_steps), stride)
    if is_training:
      indices = indices + tf.random.uniform(shape=(1, num_steps), minval=0,
                                            maxval=stride, dtype=tf.int32)
  else:
    if is_training:
      indices = tf.sort(tf.random.uniform(shape=(1, num_steps),
                                          minval=0, maxval=sequence_length,
                                          dtype=tf.int32))
    else:
      stride_float = tf.cast(sequence_length / num_steps, tf.float32)
      indices = tf.cast(tf.range(num_steps, dtype=tf.float32) * stride_float,
                        tf.int32)
  if is_training:
    indices = indices[0]

  indices.set_shape((num_steps,))
  output = tf.gather(sequence, indices)
  return output


def sample_two_sequences_uniformly(sequence: tf.Tensor, num_steps: int):
  """Uniform sampling two non-overlapping sequences.

  Sample frames based on uniform sampling following TSN (Wang et al., 2019)
  used by Tong et al. in VideoMAE. The stride is automatically computed based on
  the length of the sequence and the number of frames to take (`num_steps`)

  Args:
    sequence: Any tensor where the first dimension is timesteps.
    num_steps: Number of steps (e.g. frames) to take.
  Returns:
     A single tensor with first dimension `2 * num_steps` with the sampled
     segment.
  """

  sequence_length = tf.shape(input=sequence)[0]
  sequence_length = tf.cast(sequence_length, tf.int32)
  average_duration = tf.cast(sequence_length / num_steps, tf.float32)

  index_1 = tf.cast(tf.range(num_steps, dtype=tf.float32)
                    * average_duration + average_duration / 2.0, tf.int32)

  index_2 = tf.cast(tf.range(num_steps, dtype=tf.float32)
                    * average_duration, tf.int32)
  indices = tf.concat((index_1, index_2), axis=0)

  indices.set_shape((2 * num_steps,))
  output = tf.gather(sequence, indices)
  return output


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
      Optionally, the dictionary may contain the following params -
      use_dynamic_time_mask_max_frames (bool), whether to determine the
      time_mask_max_frames dynamically. - time_masks_per_frame (float)

  Returns:
    Augmented mel spectrogram of shape (num_time_bins, num_freq_bins, channels)
    or
      (num_clips, num_time_bins, num_freq_bins, channels).
  """
  # pylint: disable=line-too-long
  spec_augment_params_obj = spectrum_augmenter.SpectrumAugmenter.Params()
  spec_augment_params_obj.freq_mask_max_bins = spec_augment_params.freq_mask_max_bins
  spec_augment_params_obj.freq_mask_count = spec_augment_params.freq_mask_count
  spec_augment_params_obj.time_mask_max_frames = spec_augment_params.time_mask_max_frames
  spec_augment_params_obj.time_mask_count = spec_augment_params.time_mask_count
  spec_augment_params_obj.time_warp_max_frames = spec_augment_params.time_warp_max_frames
  spec_augment_params_obj.time_warp_max_ratio = spec_augment_params.time_warp_max_ratio
  spec_augment_params_obj.time_mask_max_ratio = spec_augment_params.time_mask_max_ratio
  spec_augment_params_obj.use_dynamic_time_mask_max_frames = spec_augment_params.get(
      'use_dynamic_time_mask_max_frames', False)
  spec_augment_params_obj.time_masks_per_frame = spec_augment_params.get(
      'time_masks_per_frame', 0.0)
  spec_augment_params_obj.time_warp_bound = spec_augment_params.get(
      'time_warp_bound', 'static')
  spec_augment_params_obj.name = 'specaugment'
  spec_augment_layer = spec_augment_params_obj.Instantiate()
  # pylint: enable=line-too-long

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
                        circular_time_shift=False,
                        zero_centering=True,
                        dataset_mean=0,
                        dataset_stddev=1):

  """Decodes audio spectrogram.

  Args:
    spectrogram: input mel spectrogram
    inflate: if True, adds a channel dimension
    circular_time_shift: If `True`, apply random time shift to spectrograms
    zero_centering: if True, zero centers the spectrogram
    dataset_mean: mean over the dataset.
    dataset_stddev: standard deviation over the dataset.

  Returns:
    spectrogram: decoded spectrogram.

  """
  if circular_time_shift:
    # randomly sample start time, then cyclically extract whole clip
    shift = tf.random.uniform(
        shape=(), minval=0, maxval=tf.shape(spectrogram)[0], dtype=tf.int32)
    spectrogram = tf.roll(spectrogram, shift=shift, axis=0)

  # Expand the dimension as the specaugmentation always requires the last
  # channel dimension.
  spectrogram = tf.expand_dims(spectrogram, -1)
  if inflate:
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
                    circular_time_shift=False,
                    zero_centering_image=False,
                    dataset_mean=0.0,
                    dataset_stddev=1.0,
                    sync_random_state=True,
                    inflate_spectrograms: bool = True):
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
    circular_time_shift: If `True`, apply random time shift to spectrograms.
    zero_centering_image: If `True`, frames are normalized to values in [-1, 1].
      If `False`, values in [0, 1].
    dataset_mean: Mean of values over the dataset.
    dataset_stddev: Standard deviation of values of the dataset.
    sync_random_state: Whether to use stateful option to keep random operations
      in sync between different modalities. All modalities having this option
      True will use the same outcome in random operations such as sampling and
      cropping.
    inflate_spectrograms: whether or not to repeat the single spectrogram
      channel into 3 channels.
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
      fn=lambda x: _decode_spectrogram(
          x, inflate_spectrograms, circular_time_shift and is_training,
          zero_centering_image, dataset_mean, dataset_stddev),
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
