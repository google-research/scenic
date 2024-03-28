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

"""Perlin Noise."""
import math
import tensorflow as tf


def lerp_np(x, y, w):
  fin_out = (y - x) * w + x
  return fin_out


def rand_perlin_2d_tf(shape, res):
  """Perlin noise implementation in tensorflow."""
  def f(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3

  delta = (tf.cast(res[0] / shape[0],
                   tf.float32), tf.cast(res[1] / shape[1], tf.float32))
  d = (shape[0] // res[0], shape[1] // res[1])
  grid = tf.transpose(
      tf.stack(
          tf.meshgrid(
              tf.range(0, res[1], delta[1]),
              tf.range(0, res[0], delta[0]))[::-1], axis=0), (1, 2, 0)) % 1
  # Gradients
  angles = 2 * math.pi * tf.random.uniform((res[0] + 1, res[1] + 1))
  gradients = tf.stack((tf.cos(angles), tf.sin(angles)), axis=-1)
  g00 = tf.repeat(tf.repeat(gradients[0:-1, 0:-1], d[0], 0), d[1], 1)
  g10 = tf.repeat(tf.repeat(gradients[1:, 0:-1], d[0], 0), d[1], 1)
  g01 = tf.repeat(tf.repeat(gradients[0:-1, 1:], d[0], 0), d[1], 1)
  g11 = tf.repeat(tf.repeat(gradients[1:, 1:], d[0], 0), d[1], 1)

  # Ramps
  n00 = tf.reduce_sum(grid * g00, 2)
  n10 = tf.reduce_sum(
      tf.stack((grid[:, :, 0] - 1, grid[:, :, 1]), axis=-1) * g10, 2)
  n01 = tf.reduce_sum(
      tf.stack((grid[:, :, 0], grid[:, :, 1] - 1), axis=-1) * g01, 2)
  n11 = tf.reduce_sum(
      tf.stack((grid[:, :, 0] - 1, grid[:, :, 1] - 1), axis=-1) * g11, 2)

  # Interpolation
  t = f(grid)
  n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
  n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
  return tf.sqrt(2.0) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

