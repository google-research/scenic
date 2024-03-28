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

"""Implements operations for adding different types of noise to images."""

from typing import Tuple, Optional

from absl import logging
import ml_collections
import tensorflow as tf


def get_mask(image: tf.Tensor,
             denoise_configs: ml_collections.ConfigDict,
             dtype: tf.DType = tf.float32) -> tf.Tensor:
  """Generates random bounding box coordinates and turns them into a mask.

  Args:
    image: The image for which the bounding boxes are being generated.
    denoise_configs: Configurations for the denoising process.
    dtype: Data type of the image.

  Returns:
    The mask for the cut out regions.
  """
  h, w, _ = image.shape
  bboxes = []
  for _ in range(denoise_configs.n_bboxes):
    bboxes.append(tf.image.sample_distorted_bounding_box(
        image_size=image.shape,
        bounding_boxes=tf.zeros((0, 0, 4)),
        min_object_covered=0.0,
        area_range=denoise_configs.area_range,
        aspect_ratio_range=denoise_configs.aspect_ratio_range,
        use_image_if_no_bounding_boxes=True)[2])
  bboxes = tf.stack(bboxes)

  x, y = tf.meshgrid(tf.range(w, dtype=tf.int32),
                     tf.range(h, dtype=tf.int32))
  y_min = tf.cast(bboxes[..., 0] * h, dtype=x.dtype)
  x_min = tf.cast(bboxes[..., 1] * w, dtype=x.dtype)
  y_max = tf.cast(bboxes[..., 2] * h, dtype=x.dtype)
  x_max = tf.cast(bboxes[..., 3] * w, dtype=x.dtype)

  masks = ((x[None, ...] >= x_min) &
           (x[None, ...] < x_max) &
           (y[None, ...] >= y_min) &
           (y[None, ...] < y_max))
  mask = tf.reduce_any(masks, axis=0)

  return tf.cast(mask[..., None], dtype=dtype)


def cutout_bbox(image: tf.Tensor,
                rng: int,
                denoise_configs: ml_collections.ConfigDict,
                dtype: tf.DType = tf.float32) -> Tuple[tf.Tensor, tf.Tensor]:
  """Cuts out portions of the given image by sampling random bounding boxes.

  Args:
    image: A single image, must be in 0-1 range.
    rng: Seed for sampling the noise.
    denoise_configs: Configurations for the denoising process.
    dtype: Data type of the image.

  Returns:
    The corrupted image and the noise.
  """
  del rng

  noise = get_mask(image, denoise_configs, dtype)
  noised_image = image * (1 - noise) + 0.5 * noise
  return noised_image, noise


def cutout_checkerboard(
    image: tf.Tensor,
    noise_magnitude: tf.Tensor,
    rng: int,
    denoise_configs: ml_collections.ConfigDict,
    dtype: tf.DType = tf.float32) -> Tuple[tf.Tensor, tf.Tensor]:
  """Randomly cuts out portions of the given image.

  Noise is sampled at `patch_size` times lower resolution and upsampled to
  form a checkerboard pattern.

  Args:
    image: A single image,must be in 0-1 range.
    noise_magnitude: Used to generate the cutout mask.
    rng: Seed for sampling the noise.
    denoise_configs: Configurations for the denoising process.
    dtype: Data type of the image.

  Returns:
    The corrupted image and the noise.
  """
  del rng

  p = 1 - noise_magnitude
  h, w, _ = image.shape
  fh, fw = denoise_configs.patch_size
  gh, gw = h // fh, w // fw

  noise = tf.random.uniform((gh, gw), dtype=dtype, seed=None)
  noise = tf.cast(noise > p, dtype=dtype)
  noise = tf.image.resize(noise[..., None], [h, w], method='nearest')

  noised_image = image * noise + 0.5 * (1 - noise)
  return noised_image, 1 - noise


def add_gaussian_noise(
    image: tf.Tensor,
    noise_magnitude: tf.Tensor,
    use_additive_noise: bool,
    rng: int,
    denoise_configs: ml_collections.ConfigDict,
    dtype: tf.DType = tf.float32) -> Tuple[tf.Tensor, tf.Tensor]:
  """Add gaussian noise to the given image.

  Args:
    image: A single image,must be in 0-1 range.
    noise_magnitude: Amount of noise to be added to the image.
    use_additive_noise: Whether to use a simple additive noise formulation.
    rng: Seed for sampling the noise.
    denoise_configs: Configurations for the denoising process.
    dtype: Data type of the image.

  Returns:
    The corrupted image and the noise.
  """
  del rng

  h, w, c = image.shape
  if denoise_configs.use_coarse_noise:
    fh, fw = denoise_configs.patch_size
    gh, gw = h // fh, w // fw

    noise = tf.random.normal(
        (gh, gw, c),
        mean=0.0,
        stddev=1.0,
        dtype=dtype,
        seed=None,
        name=None)
    # upsample the noise to the original image resolution
    noise = tf.image.resize(noise, [h, w], method='nearest')
  else:
    noise = tf.random.normal(
        (h, w, c),
        mean=0.0,
        stddev=1.0,
        dtype=dtype,
        seed=None,
        name=None)

  if use_additive_noise:
    logging.info('Using simple additive noise formulation.')
    noise = noise_magnitude * noise
    noised_image = image + noise
    if denoise_configs.clip_values:
      logging.info('Clipping values to be between 0 and 1.')
      noised_image = tf.clip_by_value(noised_image, 0., 1.)
      noise = noised_image - image
  else:
    logging.info('Using ddpm noise and image scaling formulation')
    scale_img = tf.sqrt(noise_magnitude) * image
    scale_noise = ((tf.sqrt(1 - noise_magnitude) * noise) -
                   tf.sqrt(noise_magnitude) + 1.0) / 2
    noised_image = scale_img + scale_noise
    if denoise_configs.clip_values:
      logging.info('Clipping values to be between 0 and 1.')
      noised_image = tf.clip_by_value(noised_image, 0., 1.)

    if noise_magnitude == 1.0:
      noise = noise * 0.
  return noised_image, noise


def patch_noise(
    image: tf.Tensor,
    gamma: tf.Tensor,
    patch_gamma: tf.Tensor,
    rng: int,
    denoise_configs: ml_collections.ConfigDict,
    dtype: tf.DType = tf.float32
) -> Tuple[tf.Tensor, tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
  """Add non uniform gaussian noise to the given image.

  Args:
    image: A single image,must be in 0-1 range.
    gamma: Controls the amount of noise in the background
    patch_gamma: Controls the amount of noise within the patches
    rng: Seed for sampling the noise.
    denoise_configs: Configurations for the denoising process.
    dtype: Data type of the image.

  Returns:
    The corrupted image and the noise.
  """
  noised_image_background, noise_background = add_gaussian_noise(
      image, gamma, False, rng, denoise_configs, dtype)
  noised_image_patch, noise_patch = add_gaussian_noise(image, patch_gamma,
                                                       False, rng,
                                                       denoise_configs, dtype)
  mask = get_mask(image, denoise_configs, dtype)
  noised_image = (noised_image_patch * mask) + (
      noised_image_background * (1 - mask))
  noise = (noise_patch * mask) + (noise_background * (1 - mask))

  return noised_image, noise, (mask, patch_gamma)


def add_noise(
    image: tf.Tensor,
    rng: int,
    denoise_configs: ml_collections.ConfigDict,
    dtype: tf.DType = tf.float32
) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor], tf.Tensor, Tuple[
    Optional[tf.Tensor], Optional[tf.Tensor]]]:
  """Add noise to the given image.

  Args:
    image: A single image,must be in 0-1 range.
    rng: Seed for sampling the noise.
    denoise_configs: Configurations for the denoising process.
    dtype: Data type of the image.

  Returns:
    The corrupted image and either the noise or the image.
  """
  tf.debugging.assert_non_negative(image, message='Image must be in 0-1 range')

  if denoise_configs.gamma != -1 and denoise_configs.sigma != -1:
    raise ValueError(
        'When gamma is being used, sigma must be set to -1 and vice versa.')

  noise_magnitude = denoise_configs.sigma if denoise_configs.sigma != -1 else denoise_configs.gamma
  noise_magnitude = tf.constant(noise_magnitude, dtype=dtype)
  use_additive_noise = denoise_configs.sigma != -1

  # use 0 for runs that won't use timestep embedding
  timestep = tf.constant(0)
  # fallback to simple random gamma sampling scheme
  if (denoise_configs.random_noise_schedule and
      denoise_configs.random_noise_schedule.type == 'simple'):
    logging.info('Using a random noise schedule')
    c = tf.random.uniform(
        shape=[],
        minval=denoise_configs.random_noise_schedule.minval,
        maxval=denoise_configs.random_noise_schedule.maxval)
    # to sample noiseless images maxval is set to a value slightly greater than
    # 1, random values of gamma that are greater than 1 are taken to be 1
    noise_magnitude = tf.minimum(c, 1.0)
  elif denoise_configs.random_noise_schedule:
    logging.info('Using a random noise schedule with timestep embedding')
    logging.info('Number of timesteps: %d',
                 denoise_configs.random_noise_schedule.n_timesteps)
    noise_magnitudes = tf.linspace(
        denoise_configs.random_noise_schedule.maxval,
        denoise_configs.random_noise_schedule.minval,
        denoise_configs.random_noise_schedule.n_timesteps)

    timestep = tf.random.uniform(
        shape=[],
        maxval=denoise_configs.random_noise_schedule.n_timesteps,
        dtype=tf.int32)
    if denoise_configs.random_noise_schedule.type == 'linear':
      noise_magnitude = noise_magnitudes[timestep]
    elif denoise_configs.random_noise_schedule.type == 'multiplicative':
      # Multipy gammas from step 1 to `timestep` as is done in DDPM
      noise_magnitude = tf.math.reduce_prod(noise_magnitudes[:timestep])

  patch = (None, None)
  if denoise_configs.type == 'gaussian':
    logging.info('Adding gaussian noise to the image')
    noised_image, noise = add_gaussian_noise(image, noise_magnitude,
                                             use_additive_noise, rng,
                                             denoise_configs, dtype)
  elif denoise_configs.type == 'cutout_checkerboard':
    logging.info('Apply cutout to random portions of the image')
    noised_image, noise = cutout_checkerboard(image, noise_magnitude, rng,
                                              denoise_configs, dtype)
  elif denoise_configs.type == 'cutout_bbox':
    logging.info('Apply cutout to random portions of the image')
    noised_image, noise = cutout_bbox(image, rng, denoise_configs, dtype)
  elif denoise_configs.type == 'patch_noise':
    logging.info('Apply non uniform gaussian noise to the image')
    noised_image, noise, patch = patch_noise(
        image=image,
        gamma=noise_magnitude,
        patch_gamma=tf.constant(denoise_configs.patch_gamma, dtype=dtype),
        rng=rng,
        denoise_configs=denoise_configs,
        dtype=dtype)
  else:
    raise ValueError(f'Noise type: {denoise_configs.type} is not defined')

  return noised_image, noise, timestep, noise_magnitude, patch
