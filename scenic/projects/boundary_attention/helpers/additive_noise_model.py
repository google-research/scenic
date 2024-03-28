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

"""Noise model for the boundary attention training."""

from scenic.projects.boundary_attention.helpers import perlin_noise
import tensorflow as tf


class NoiseModel:
  """Noise model for the boundary attention training."""

  def __init__(self, min_noise_level, max_noise_level, *, normalize=True):
    self.min_noise_lvl = min_noise_level
    self.max_noise_lvl = max_noise_level
    self.normalize = normalize

  def __call__(self, input_im, input_im_shape):

    # Decide on an overall noise level given a minimum noise level and a maximum
    # noise level
    overall_noise_lvl = tf.random.uniform(
        shape=[], minval=self.min_noise_lvl, maxval=self.max_noise_lvl
    )

    # A toggle that decides which noise mode to implement
    r = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)

    # If r is between 0 and .3, add both camera noise and perlin noise, where
    # the amount of each is uniformly
    # chosen to sum to the maxmimum noise level
    if 0.0 < r < 0.3:
      amnt_perlin = tf.random.uniform(
          shape=[], minval=0.0, maxval=overall_noise_lvl
      )
      noisy_im = self._get_perlin_noise(
          input_im_shape
      ) * amnt_perlin + self._get_camera_noise(input_im, input_im_shape) * (
          1 - amnt_perlin
      )

    # If r is between .4 and .8, add gaussian noise in the same manner as our
    # original training
    elif 0.3 <= r < 0.75:
      noisy_im = (
          overall_noise_lvl * self._get_gaussian_noise(input_im_shape)
          + input_im
      )

    elif 0.75 <= r < 0.8:
      chunk_size = 7
      noisy_im = input_im + overall_noise_lvl * self._get_chunky_noise(
          input_im, input_im_shape, chunk_size
      )

    # If r is between .6 and .8, add camera noise
    elif 0.8 <= r < 0.9:
      noisy_im = overall_noise_lvl * self._get_camera_noise(
          input_im, input_im_shape
      )

    # If r is between .9 and 1.0, add perlin noise
    else:
      noisy_im = input_im + overall_noise_lvl * self._get_perlin_noise(
          input_im_shape
      )

    # Truncate any values below 0 or greater than 1
    noisy_im = tf.where(noisy_im > 1.0, 1.0, noisy_im)
    noisy_im = tf.where(noisy_im < 0.0, 0.0, noisy_im)

    return noisy_im

  def normalize_fn(self, im):
    """Normalizes an image."""
    im = im - tf.math.reduce_min(im, axis=(0, 1), keepdims=True)
    im = im / tf.math.reduce_max(im, axis=(0, 1), keepdims=True)

    return im

  def _get_gaussian_noise(self, y_shape):
    """Generates a gaussian noise."""
    return tf.random.normal(shape=y_shape)

  def _get_chunky_noise(self, y, y_shape, chunk_size):
    """Generates a noise that is a chunky version of the input noise."""

    noise = tf.random.normal(
        shape=tf.concat([tf.cast(([1]), dtype=tf.int32), y_shape], axis=0),
        stddev=tf.expand_dims(tf.norm(y, axis=-1, keepdims=True), 0),
    )
    noise = (
        3
        * tf.squeeze(
            tf.nn.avg_pool2d(noise, chunk_size, strides=1, padding='SAME'), 0
        )
        * tf.random.uniform([3], minval=0, maxval=1, dtype=tf.float32)
    )

    return noise

  def _get_perlin_noise(self, y_shape):
    """Generates perlin noise."""

    # h, w = tf.cast(y.shape[0], tf.float32), tf.cast(y.shape[1], tf.float32)
    h, w = tf.cast(y_shape[0], tf.float32), tf.cast(y_shape[1], tf.float32)

    # f controls the coarseness of the generated perlin noise
    f = tf.math.maximum(
        tf.cast(
            tf.cast(tf.random.normal(shape=[], mean=0, stddev=2), tf.int32) + 5,
            tf.float32,
        ),
        2.0,
    )

    # f needs to be a factor of the rendered image size, so we choose any size
    # large than our image and crop it later
    h_render, w_render = (tf.math.ceil(h / f) * f), (tf.math.ceil(w / f)) * f

    r_h, r_w = tf.cast(h_render / f, tf.int32), tf.cast(w_render / f, tf.int32)

    h_render, w_render = tf.cast(h_render, tf.int32), tf.cast(
        w_render, tf.int32
    )

    colors = tf.stack(
        (
            perlin_noise.rand_perlin_2d_tf(
                (h_render, w_render), res=(r_h, r_w)
            ),
            perlin_noise.rand_perlin_2d_tf(
                (h_render, w_render), res=(r_h, r_w)
            ),
            perlin_noise.rand_perlin_2d_tf(
                (h_render, w_render), res=(r_h, r_w)
            ),
        ),
        -1,
    )
    qcolors = tf.quantization.fake_quant_with_min_max_args(colors, -1, 1, 3)

    return qcolors[:y_shape[0], :y_shape[1], :y_shape[2]]

  def _get_camera_noise(self, y, y_shape, *, model=None, params=None):
    """Generates camera noise."""

    all_models = ['g', 'gP', 'gp']

    if model is None:
      model = tf.gather(
          all_models,
          tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32),
      )

    if params is None:
      kconst, g_scale, saturation_level, ratio = self._sample_camera_params()
    else:
      kconst, g_scale, saturation_level, ratio = params

    y = y * saturation_level
    y = y / ratio

    if tf.strings.regex_full_match(model, '.*P.*'):
      z = tf.random.poisson(shape=[], lam=(y / kconst)) * kconst
    elif tf.strings.regex_full_match(model, '.*p.*'):
      z = y + tf.random.normal(shape=y_shape) * tf.math.sqrt(
          tf.math.maximum(kconst * y, 1e-10)
      )
    else:
      z = y

    if tf.strings.regex_full_match(model, '.*g.*'):
      z = z + tf.random.normal(shape=y_shape) * tf.math.maximum(
          g_scale, 1e-10
      )  # Gaussian noise

    z = z * ratio
    z = z / saturation_level
    return z

  def _sample_camera_params(self):
    """Samples reasonable camera parameters."""
    saturation_level = 16383 - 800

    g_scale_sigma = tf.random.uniform(shape=[], minval=0.2, maxval=0.27)
    g_scale_bias = tf.random.uniform(shape=[], minval=0.7, maxval=2.2)
    g_scale_slope = tf.random.uniform(shape=[], minval=0.5, maxval=0.7)

    log_kconst = tf.random.uniform(
        shape=[], minval=tf.math.log(1e-1), maxval=tf.math.log(30.0)
    )

    log_g_scale = (
        tf.random.normal(shape=[]) * g_scale_sigma * 1
        + g_scale_slope * log_kconst
        + g_scale_bias
    )

    kconst = tf.math.exp(log_kconst)
    g_scale = tf.math.exp(log_g_scale)

    ratio = tf.random.uniform(shape=[], minval=100, maxval=300)

    return (kconst, g_scale, saturation_level, ratio)
