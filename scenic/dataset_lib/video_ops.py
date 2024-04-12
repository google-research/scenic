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

"""Preprocessing functions for video data loading.

Includes SimCLR-style data augmentation functions adapted to be temporally
consistent throughout the video.

Code is based on:
SimCLR style data augmentation is based on:
https://github.com/google-research/simclr/blob/master/tf2/data_util.py
"""

import functools
import math
from typing import Optional


from absl import logging
from dmvr import builders
from dmvr import processors as dmvr_processors
import simclr.tf2.data_util as simclr_data
import tensorflow as tf
from official.vision.image_classification import augment


def _get_shape(x):
  """Gets tensor shape as a list, allowing mixing static and dynamic shapes."""
  dynamic_shape = tf.shape(x)
  if x.shape.ndims is None:
    return dynamic_shape
  static_shape = x.shape.as_list()
  shapes = [
      static_shape[i] if static_shape[i] is not None else dynamic_shape[i]
      for i in range(x.shape.ndims)
  ]
  return shapes


def _fill_rectangle_video(image,
                          center_width,
                          center_height,
                          half_width,
                          half_height,
                          replace=None):
  """Fills blank area for video."""
  image_time = tf.shape(image)[0]
  image_height = tf.shape(image)[1]
  image_width = tf.shape(image)[2]

  lower_pad = tf.maximum(0, center_height - half_height)
  upper_pad = tf.maximum(0, image_height - center_height - half_height)
  left_pad = tf.maximum(0, center_width - half_width)
  right_pad = tf.maximum(0, image_width - center_width - half_width)

  cutout_shape = [
      image_time, image_height - (lower_pad + upper_pad),
      image_width - (left_pad + right_pad)
  ]
  padding_dims = [[0, 0], [lower_pad, upper_pad], [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims,
      constant_values=1)
  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, 1, 3])

  if replace is None:
    fill = tf.random.normal(tf.shape(image), dtype=image.dtype)
  elif isinstance(replace, tf.Tensor):
    fill = replace
  else:
    fill = tf.ones_like(image, dtype=image.dtype) * replace
  image = tf.where(tf.equal(mask, 0), fill, image)

  return image


class RandomErasing:
  """Applies RandomErasing to a video.


  Reference: https://arxiv.org/abs/1708.04896
  """

  def __init__(self,
               probability: float = 0.25,
               min_area: float = 0.02,
               max_area: float = 1 / 3,
               min_aspect: float = 0.3,
               max_aspect: Optional[float] = None,
               min_count=1,
               max_count=1,
               trials=10):
    """Applies RandomErasing to a video.

    Args:
      probability: Probability of augmenting the image. Defaults to `0.25`.
      min_area: Minimum area of the random erasing rectangle. Defaults to
        `0.02`.
      max_area: Maximum area of the random erasing rectangle. Defaults to `1/3`.
      min_aspect: Minimum aspect rate of the random erasing rectangle. Defaults
        to `0.3`.
      max_aspect: Maximum aspect rate of the random erasing rectangle. Defaults
        to `None`.
      min_count: Minimum number of erased rectangles. Defaults to `1`.
      max_count: Maximum number of erased rectangles. Defaults to `1`.
      trials: Maximum number of trials to randomly sample a rectangle that
        fulfills constraint. Defaults to `10`.
    """
    self._probability = probability
    self._min_area = float(min_area)
    self._max_area = float(max_area)
    self._min_log_aspect = math.log(min_aspect)
    self._max_log_aspect = math.log(max_aspect or 1 / min_aspect)
    self._min_count = min_count
    self._max_count = max_count
    self._trials = trials

  def distort(self, video: tf.Tensor) -> tf.Tensor:
    """Applies RandomErasing to video.

    Args:
      video (tf.Tensor): Of shape [temporal, height, width, 3] representing a
      video.

    Returns:
      tf.Tensor: The augmented version of video.
    """
    uniform_random = tf.random.uniform(shape=[], minval=0., maxval=1.0)
    mirror_cond = tf.less(uniform_random, self._probability)
    video = tf.cond(mirror_cond, lambda: self._erase(video), lambda: video)
    return video

  @tf.function
  def _erase(self, video: tf.Tensor) -> tf.Tensor:
    """Erase an area."""
    if self._min_count == self._max_count:
      count = self._min_count
    else:
      count = tf.random.uniform(
          shape=[],
          minval=int(self._min_count),
          maxval=int(self._max_count - self._min_count + 1),
          dtype=tf.int32)

    image_height = tf.shape(video)[1]
    image_width = tf.shape(video)[2]
    area = tf.cast(image_width * image_height, tf.float32)

    for _ in range(count):
      # Work around since break is not supported in tf.function
      is_trial_successfull = False
      for _ in range(self._trials):
        if not is_trial_successfull:
          erase_area = tf.random.uniform(
              shape=[],
              minval=area * self._min_area,
              maxval=area * self._max_area)
          aspect_ratio = tf.math.exp(
              tf.random.uniform(
                  shape=[],
                  minval=self._min_log_aspect,
                  maxval=self._max_log_aspect))

          half_height = tf.cast(
              tf.math.round(tf.math.sqrt(erase_area * aspect_ratio) / 2),
              dtype=tf.int32)
          half_width = tf.cast(
              tf.math.round(tf.math.sqrt(erase_area / aspect_ratio) / 2),
              dtype=tf.int32)

          if 2 * half_height < image_height and 2 * half_width < image_width:
            center_height = tf.random.uniform(
                shape=[],
                minval=0,
                maxval=int(image_height - 2 * half_height),
                dtype=tf.int32)
            center_width = tf.random.uniform(
                shape=[],
                minval=0,
                maxval=int(image_width - 2 * half_width),
                dtype=tf.int32)

            video = _fill_rectangle_video(
                video,
                center_width,
                center_height,
                half_width,
                half_height,
                replace=None)

            is_trial_successfull = True
    return video


def random_erasing(frames: tf.Tensor,
                   probability: float = 0.25, min_area: float = 0.02,
                   max_area: float = 1 / 3, min_aspect: float = 0.3,
                   max_aspect: Optional[float] = None, min_count=1,
                   max_count=1, trials=10):

  """Applies RandomErasing to a video.

  Args:
    frames: A Tensor of dimension [timesteps, input_h, input_w, channels].
    probability: Probability of augmenting the image. Defaults to `0.25`.
    min_area: Minimum area of the random erasing rectangle. Defaults to
      `0.02`.
    max_area: Maximum area of the random erasing rectangle. Defaults to `1/3`.
    min_aspect: Minimum aspect rate of the random erasing rectangle. Defaults
      to `0.3`.
    max_aspect: Maximum aspect rate of the random erasing rectangle. Defaults
      to `None`.
    min_count: Minimum number of erased rectangles. Defaults to `1`.
    max_count: Maximum number of erased rectangles. Defaults to `1`.
    trials: Maximum number of trials to randomly sample a rectangle that
      fulfills constraint. Defaults to `10`.
  Returns:
    tf.Tensor: The augmented version of video.
  """
  random_eraser = RandomErasing(probability, min_area, max_area, min_aspect,
                                max_aspect, min_count, max_count, trials)
  return random_eraser.distort(frames)


def crop_resize(
    frames: tf.Tensor,
    output_h: int,
    output_w: int,
    num_frames: int,
    num_channels: int,
    area_range=(0.3, 1),
    unused_state=None,
    aspect_ratio=(0.5, 2.0),
    resize_method: str = tf.image.ResizeMethod.BICUBIC,
    resize_antialias: bool = False,
) -> tf.Tensor:
  """First crop clip with jittering and then resizes to (output_h, output_w).

  Args:
    frames: A Tensor of dimension [timesteps, input_h, input_w, channels].
    output_h: Size of the height of output.
    output_w: Size of the width of output.
    num_frames: Number of input frames per clip.
    num_channels: Number of channels of the clip.
    area_range: Random crop will preserve this proportion of the area of the
      original frame.
    unused_state: Argument included to be compatible with DeepMind Video Reader
      preprocessing pipeline functions which pass in a state variable.
    aspect_ratio: Aspect ratio range of area based random resizing.
    resize_method: Method for resizing the frames.
    resize_antialias: If True, apply anti-aliasing when resizing.

  Returns:
    A Tensor of shape [timesteps, output_h, output_w, channels] of type
      frames.dtype.
  """

  shape = tf.shape(frames)
  seq_len, channels = int(shape[0]), int(shape[3])
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  factor = output_w / output_h
  aspect_ratio = (aspect_ratio[0] * factor, aspect_ratio[1] * factor)

  sample_distorted_bbox = tf.image.sample_distorted_bounding_box(
      shape[1:],
      bounding_boxes=bbox,
      min_object_covered=0.1,
      aspect_ratio_range=aspect_ratio,
      area_range=area_range,
      max_attempts=100,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bbox
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  size = tf.convert_to_tensor((seq_len, target_height, target_width, channels))
  offset = tf.convert_to_tensor((0, offset_y, offset_x, 0))

  frames = tf.slice(frames, offset, size)
  frames = tf.cast(
      tf.image.resize(
          frames,
          (output_h, output_w),
          method=resize_method,
          antialias=resize_antialias,
      ),
      frames.dtype,
  )
  frames.set_shape((num_frames, output_h, output_w, num_channels))
  return frames


def simclr_aug_fn(frames, num_frames):
  """Applies the Simclr Augment policy to one video clip.

  Args:
    frames: `Tensor` of shape [timesteps, height, width, 3].
    num_frames: number of frames.

  Returns:
    A Tensor of shape [timesteps, output_h, output_w, channels] being random
    augmented with the same operation.
  """

  def random_color_jitter(image, p=1.0):

    def _transform(image):
      color_jitter_t = functools.partial(
          simclr_data.color_jitter, strength=0.75)
      image = simclr_data.random_apply(color_jitter_t, p=0.8, x=image)
      return simclr_data.random_apply(simclr_data.to_grayscale, p=0.2, x=image)

    return simclr_data.random_apply(_transform, p=p, x=image)

  frame_list = tf.unstack(frames, num_frames, 0)
  # Temporally random version
  # simclr_aug_frame_list = []
  # for image in frame_list:
  #     image = random_color_jitter(image)
  #     simclr_aug_frame_list.append(image)
  # return tf.stack(simclr_aug_frame_list, axis=0)

  # Temporally consistent version
  big_image = tf.concat(frame_list, axis=0)  # [t*h, w, c]
  big_image = random_color_jitter(big_image)
  simclr_aug_frame_list = tf.split(big_image, num_or_size_splits=num_frames)
  return tf.stack(simclr_aug_frame_list, axis=0)  # [t, h, w, c]


def batch_random_blur(images, height, width, blur_probability=0.5):
  """Random blur to all frames.

  All frames have a blur applied to them, or all do not.

  Args:
    images: `Tensor` of shape [timesteps, height, width, 3]..
    height: the height of image.
    width: the width of image.
    blur_probability: the probaility to apply the blur operator.

  Returns:
    Blurred images.
  """

  def generate_selector(p, bsz):
    shape = [bsz, 1, 1, 1]
    selector = tf.cast(
        tf.less(tf.random.uniform(shape, 0, 1, dtype=tf.float32), p),
        tf.float32)
    return selector

  images_new = simclr_data.random_blur(images, height, width, p=1.)
  # All frames have augmentation applied, or not.
  selector = generate_selector(blur_probability, 1)
  images = images_new * selector + images * (1 - selector)
  images = tf.clip_by_value(images, 0., 1.)

  return images


def random_solarization(image, p=0.2):

  def _transform(image):
    image = image * tf.cast(tf.less(image, 0.5), tf.float32) + (
        1.0 - image) * tf.cast(tf.greater_equal(image, 0.5), tf.float32)
    return image

  return simclr_data.random_apply(_transform, p=p, x=image)


def random_time_reverse(image, p=0.5):

  def _transform(image):
    return image[::-1, :, :, :]

  return simclr_data.random_apply(_transform, p=p, x=image)


def simclr_style_augmentation(frames, height, width, zero_centre):
  """Applies SimCLR-style random augmentations to frames.

  Args:
    frames: `Tensor` of shape [timesteps, height, width, 3].
    height: Image height.
    width: Image width.
    zero_centre: Bool. If true, frames are between [-1. 1]. Otherwise, they are
      in the range [0, 1]

  Returns:
    A Tensor of shape [timesteps, height, width, channels] being random
    augmented with the same operation.
  """
  num_frames = frames.shape[0]
  frames = simclr_aug_fn(frames, num_frames)
  blur_frames = batch_random_blur(frames, height, width)
  solarize_frames = random_solarization(blur_frames)
  reversed_frames = random_time_reverse(solarize_frames)
  reversed_frames = tf.clip_by_value(reversed_frames, 0., 1.)

  if zero_centre:
    return reversed_frames * 2.0 - 1.0
  else:
    return reversed_frames


def deterministic_crop(images, size, spatial_idx):
  """Takes a deterministic crop of input images.

  Args:
    images: `Tensor` of shape shape [t, h, w, c]
    size: Integer ; size of height and width to crop the images.
    spatial_idx: 0, 1, or 2 for left, center, and right crop if width is larger
      than height. Or 0, 1, or 2 for top, center, and bottom crop if height is
      larger than width.

  Returns:
    cropped: `Tensor` of shape [t, crop_size, crop_size, c]
  """
  assert spatial_idx in [0, 1, 2]
  height, width = tf.shape(images)[1], tf.shape(images)[2]

  y_offset = tf.cast(tf.math.ceil((height - size) / 2), tf.int32)
  x_offset = tf.cast(tf.math.ceil((width - size) / 2), tf.int32)

  if height > width:
    if spatial_idx == 0:
      y_offset = 0
    elif spatial_idx == 2:
      y_offset = height - size
  else:
    if spatial_idx == 0:
      x_offset = 0
    elif spatial_idx == 2:
      x_offset = width - size

  cropped = tf.slice(images, [0, y_offset, x_offset, 0], [-1, size, size, -1])

  return cropped


def three_spatial_crops(images, crop_size):
  """Returns three spatial crops of the same frame, as done by SlowFast.

  This enables testing using the same protocol as prior works. ie
  (https://arxiv.org/abs/1812.03982, https://arxiv.org/abs/1904.02811,
   https://arxiv.org/abs/2004.04730)
  If width > height, takes left, centre and right crop.
  If height > width, takes top, middle and bottom crop.

  Args:
    images: `Tensor` of shape [t, h, w, c]
    crop_size: The size to crop from the images

  Returns:
    `Tensor` of shape [3 * t, h, w, c]
  """

  result = []
  for spatial_index in range(3):
    images_cropped = deterministic_crop(images, crop_size, spatial_index)
    result.append(images_cropped)

  return tf.concat(result, axis=0)


def additional_augmentations(
    ds_factory,
    augmentation_params,
    crop_size,
    num_frames,
    zero_centering,
    rgb_feature_name=None,
    resize_method: str = tf.image.ResizeMethod.BICUBIC,
    resize_antialias: bool = False,
):
  """Apply additional data augmentations in the DMVR pre-processsing graph."""

  if not rgb_feature_name:
    rgb_feature_name = builders.IMAGE_FEATURE_NAME

  do_simclr_crop_resize = augmentation_params.get('do_simclr_crop_resize',
                                                  False)
  do_simclr_style_augmentations = augmentation_params.get(
      'do_simclr_style_augmentations', False)
  do_rand_augment = augmentation_params.get('do_rand_augment', False)
  do_color_augment = augmentation_params.get('do_color_augment', False)
  do_jitter_scale = augmentation_params.get('do_jitter_scale', False)
  do_random_erasing = augmentation_params.get('do_random_erasing', False)

  if do_simclr_crop_resize and do_jitter_scale:
    logging.warning('Only doing simclr_crop_resize.'
                    'Not compatible with jitter_scale')

  if do_simclr_crop_resize:
    area_range = (augmentation_params.get('simclr_area_lower_bound', 0.5), 1)
    aspect_ratio = augmentation_params.get('aspect_ratio_crop', (0.5, 2.0))

    # Remove resize_smallest and Replace random_crop with crop_resize
    ds_factory.preprocessor_builder.remove_fn(
        f'{rgb_feature_name}_resize_smallest')
    # To replace random_crop with the crop_resize we need to find out which
    # function comes next, as not all datasets have the same list of
    # preprocessing functions (e.g. SSv2 doesn't have a random_flip)
    randcrop_fn_name = f'{rgb_feature_name}_random_crop'
    fns_list = ds_factory.preprocessor_builder.get_summary()
    idx = [i for i, fd in enumerate(fns_list) if fd.fn_name == randcrop_fn_name]
    if not idx:
      raise ValueError(f'No {randcrop_fn_name} in Preprocessing Builder.')
    next_fn_name = fns_list[idx[0] + 1].fn_name
    ds_factory.preprocessor_builder.remove_fn(randcrop_fn_name)
    ds_factory.preprocessor_builder.add_fn(
        functools.partial(
            crop_resize,
            num_frames=num_frames,
            output_h=crop_size,
            output_w=crop_size,
            num_channels=3,
            area_range=area_range,
            aspect_ratio=aspect_ratio,
            resize_method=resize_method,
            resize_antialias=resize_antialias,
        ),
        feature_name=rgb_feature_name,
        fn_name=f'{rgb_feature_name}_crop_resize',
        add_before_fn_name=next_fn_name,
    )

  elif do_jitter_scale:
    ds_factory.preprocessor_builder.add_fn(
        functools.partial(
            dmvr_processors.scale_jitter_augm,
            min_scale_factor=augmentation_params.scale_min_factor,
            max_scale_factor=augmentation_params.scale_max_factor,
            prob=augmentation_params.prob_scale_jitter),
        feature_name=rgb_feature_name,
        fn_name=f'{rgb_feature_name}_jitter_scale',
        add_before_fn_name=f'{rgb_feature_name}_random_crop')

  if do_simclr_style_augmentations and do_color_augment:
    logging.warning('Only doing simclr_style_augmentations as it includes'
                    'color augmentations')

  if sum([do_rand_augment, do_simclr_style_augmentations, do_color_augment
         ]) > 1:
    logging.warning('Priority for different augmentation functions is:'
                    '1) rand_augment. 2) simclr_style_augment.'
                    '3) colour_augment. Only one is performed.')

  if do_rand_augment:
    logging.info('Adding rand_augment')
    ds_factory.preprocessor_builder.add_fn(
        functools.partial(
            distort_image_with_randaugment,
            num_layers=augmentation_params.rand_augment_num_layers,
            magnitude=augmentation_params.rand_augment_magnitude,
        ),
        feature_name=rgb_feature_name,
        fn_name=f'{rgb_feature_name}_rand_augment',
        add_before_fn_name=f'{rgb_feature_name}_normalize')
  elif do_simclr_style_augmentations:
    # Add additional augmentations at the end
    logging.info('Adding simclr_style augmentation')
    ds_factory.preprocessor_builder.add_fn(
        functools.partial(
            simclr_style_augmentation,
            height=crop_size,
            width=crop_size,
            zero_centre=zero_centering), rgb_feature_name)
  elif do_color_augment:
    logging.info('Adding color_augment')
    ds_factory.preprocessor_builder.add_fn(
        functools.partial(
            dmvr_processors.color_default_augm,
            zero_centering_image=zero_centering,
            prob_color_augment=augmentation_params.prob_color_augment,
            prob_color_drop=augmentation_params.prob_color_drop),
        rgb_feature_name)

  if do_random_erasing:
    logging.info('Adding random erasing')
    random_erasing_prob = augmentation_params.get('random_erasing_prob', 0.25)
    ds_factory.preprocessor_builder.add_fn(
        functools.partial(random_erasing, probability=random_erasing_prob),
        rgb_feature_name)

  return ds_factory


def random_sample_sequence_with_centre(
    sequence: tf.Tensor,
    num_steps: int,
    stride: int = 1,
    seed: Optional[int] = None,
    state: Optional[builders.ProcessorState] = None) -> tf.Tensor:
  """Samples a single segment of size `num_steps` from a given sequence.

  The segment is randomly chosen such that it contains the middle element
  of the sequence.

  Args:
    sequence: Any tensor where the first dimension is timesteps.
    num_steps: Number of steps (e.g. frames) to take.
    stride: Distance to sample between timesteps.
    seed: A deterministic seed to use when sampling.
    state: A mutable dictionary where keys are strings. The dictionary might
      contain 'sample_offset_proportion' as key with metadata useful for
      sampling. It will be modified with added metadata if needed. This can be
      used to keep consistency between sampling of different sequences.

  Returns:
    A single tensor with first dimension `num_steps` with the sampled segment.
  """
  sequence_length = tf.shape(input=sequence)[0]
  offset_lower_bound = tf.maximum(sequence_length / 2 - num_steps * stride, 0)
  offset_upper_bound = sequence_length / 2

  offset = tf.random.uniform(
      (),
      minval=tf.cast(offset_lower_bound, dtype=tf.int32),
      maxval=tf.cast(offset_upper_bound, dtype=tf.int32),
      dtype=tf.int32,
      seed=seed)  # Samples from [lower_bound, upper_bound)

  indices = dmvr_processors.sample_or_pad_sequence_indices(
      sequence=sequence,
      num_steps=num_steps,
      repeat_sequence=True,  # Will repeat the sequence if we request more.
      stride=stride,
      offset=offset)
  indices.set_shape((num_steps,))
  output = tf.gather(sequence, indices)

  if state is not None:
    # Update state.
    sample_offset_proportion = (
        tf.cast(offset, tf.float32) / tf.cast(sequence_length, tf.float32))
    state['sample_offset_proportion'] = sample_offset_proportion

  return output


def cutout(big_image, pad_size, num_frames, replace=0) -> tf.Tensor:
  """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

  This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
  a random location within `img`. The pixel values filled in will be of the
  value `replace`. The located where the mask will be applied is randomly
  chosen uniformly over the whole image.

  Args:
    big_image: An image Tensor of type uint8. Shape is [t * h, w, c]
    pad_size: Specifies how big the zero mask that will be generated is that is
      applied to the image. The mask will be of size (2*pad_size x 2*pad_size).
    num_frames: Specifies the t dimension in the input shape.
    replace: What pixel value to fill in the image in the area that has the
      cutout mask applied to it.

  Returns:
    An image Tensor that is of type uint8.
  """
  big_image_shape = _get_shape(big_image)
  image = tf.reshape(big_image, [
      num_frames, big_image_shape[0] // num_frames, big_image_shape[1],
      big_image_shape[2]
  ])
  image_height = tf.shape(image)[1]
  image_width = tf.shape(image)[2]

  # Sample the center location in the image where the zero mask will be applied.
  cutout_center_height = tf.random.uniform(
      shape=[], minval=0, maxval=image_height, dtype=tf.int32)

  cutout_center_width = tf.random.uniform(
      shape=[], minval=0, maxval=image_width, dtype=tf.int32)

  lower_pad = tf.maximum(0, cutout_center_height - pad_size)
  upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
  left_pad = tf.maximum(0, cutout_center_width - pad_size)
  right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

  cutout_shape = [
      image_height - (lower_pad + upper_pad),
      image_width - (left_pad + right_pad)
  ]
  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims,
      constant_values=1)
  mask = tf.expand_dims(mask, -1)
  mask = tf.expand_dims(mask, 0)
  mask = tf.tile(mask, [num_frames, 1, 1, 3])
  image = tf.where(
      tf.equal(mask, 0),
      tf.ones_like(image, dtype=image.dtype) * replace, image)

  big_image = tf.reshape(image, [num_frames * image_height, image_width, 3])
  return big_image


NAME_TO_FUNC = {
    'AutoContrast': augment.autocontrast,
    'Equalize': augment.equalize,
    'Invert': augment.invert,
    # 'Rotate': wrapped_rotate,
    'Posterize': augment.posterize,
    'Solarize': augment.solarize,
    'SolarizeAdd': augment.solarize_add,
    'Color': augment.color,
    'Contrast': augment.contrast,
    'Brightness': augment.brightness,
    'Sharpness': augment.sharpness,
    # 'ShearX': shear_x,
    # 'ShearY': shear_y,
    # 'TranslateX': translate_x,
    # 'TranslateY': translate_y,
    'Cutout': cutout,
}

# Functions that have a 'replace' parameter
REPLACE_FUNCS = frozenset({
    'Rotate',
    'TranslateX',
    'ShearX',
    'ShearY',
    'TranslateY',
    'Cutout',
})


def _parse_policy_info(name, prob, level, replace_value, cutout_const,
                       translate_const):
  """Return the function that corresponds to `name` and update `level` param."""
  func = NAME_TO_FUNC[name]
  args = augment.level_to_arg(cutout_const, translate_const)[name](level)

  if name in REPLACE_FUNCS:
    # Add in replace arg if it is required for the function that is called.
    args = tuple(list(args) + [replace_value])

  return func, prob, args


def distort_image_with_randaugment(frames,
                                   num_layers,
                                   magnitude,
                                   cutout_const=40,
                                   translate_const=100):
  """Applies the RandAugment policy to `image`.

  The original rand_augment implementation is for images. To be temporally
  consistent in video, we
    -- Reshape the video clip [t, h, w, c] to [t * h, w, c]
    -- Only apply functions that do not depend on spatial extent (ie rotate,
        shear, translate)
    -- We do, however, use a modified cutout.

  Args:
    frames: `Tensor` of shape [t, h, w, 3] representing an image.
    num_layers: Integer, the number of augmentation transformations to apply
      sequentially to an image. Represented as (N) in the paper. Usually best
      values will be in the range [1, 3].
    magnitude: Integer, shared magnitude across all augmentation operations.
      Represented as (M) in the paper. Usually best values are in the range [5,
      10].
    cutout_const: multiplier for applying cutout.
    translate_const: multiplier for applying translation.

  Returns:
    The augmented version of `frames`.
  """
  available_ops = [
      'AutoContrast',
      'Equalize',
      'Invert',
      'Posterize',
      'Solarize',
      'Color',
      'Contrast',
      'Brightness',
      'Sharpness',
      'Cutout',
      'SolarizeAdd',
      # 'Rotate', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY',
  ]

  input_shape = _get_shape(frames)
  num_frames = input_shape[0]
  image = tf.reshape(frames, [-1, frames.shape[2], frames.shape[3]])
  input_image_type = image.dtype

  if input_image_type != tf.uint8:
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.cast(image, dtype=tf.uint8)

  replace_value = [128] * 3
  min_prob, max_prob = 0.2, 0.8

  for _ in range(num_layers):
    op_to_select = tf.random.uniform([],
                                     maxval=len(available_ops) + 1,
                                     dtype=tf.int32)

    branch_fns = []
    for (i, op_name) in enumerate(available_ops):
      prob = tf.random.uniform([],
                               minval=min_prob,
                               maxval=max_prob,
                               dtype=tf.float32)
      func, _, args = _parse_policy_info(op_name, prob, magnitude,
                                         replace_value, cutout_const,
                                         translate_const)

      if op_name == 'Cutout':
        args = (args[0], num_frames)

      branch_fns.append((
          i,
          # pylint:disable=g-long-lambda
          lambda selected_func=func, selected_args=args: selected_func(
              image, *selected_args)))
      # pylint:enable=g-long-lambda

    image = tf.switch_case(
        branch_index=op_to_select,
        branch_fns=branch_fns,
        default=lambda: tf.identity(image))

  image = tf.cast(image, dtype=input_image_type)
  return tf.reshape(image, input_shape)
