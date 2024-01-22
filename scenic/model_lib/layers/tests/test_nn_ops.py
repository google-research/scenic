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

"""Tests for nn_ops.py."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from scenic.model_lib.layers import nn_ops


class NNOpsTest(parameterized.TestCase):
  """Tests for utilities in nn_ops.py."""

  @parameterized.named_parameters([('test_both', (0, 1), (2, 3, 5, 4, 6)),
                                   ('test_rows', (0,), (1, 3, 4)),
                                   ('test_columns', (1,), (1, 5, 6))])
  def test_compute_relative_positions(self, spatial_axis,
                                      expected_output_shape):
    """Tests compute_relative_positions.

    Args:
      spatial_axis: position axis passed to the compute_relative_positions.
      expected_output_shape: expected shape of the output.
    """
    query_spatial_shape = (3, 5)
    key_spatial_shape = (4, 6)
    relative_positions = nn_ops.compute_relative_positions(
        query_spatial_shape, key_spatial_shape, spatial_axis)

    # test output shape
    self.assertEqual(relative_positions.shape, expected_output_shape)

    # test maximum positional distances
    for dim_i, dim in enumerate(spatial_axis):
      max_positional_distances = (
          query_spatial_shape[dim] + key_spatial_shape[dim] - 2)
      self.assertEqual(max_positional_distances,
                       jnp.max(relative_positions[dim_i]))

  def test_weighted_max_pool(self):
    """Tests weighted_max_pool."""
    inputs_shape = (16, 32, 32, 20)
    window_shape = (4, 4)
    strides = (4, 4)
    inputs = jnp.array(np.random.normal(size=inputs_shape))
    weights = jnp.ones(inputs_shape[:-1])

    outputs, pooled_weights = nn_ops.weighted_max_pool(
        inputs,
        weights,
        window_shape=window_shape,
        strides=strides,
        padding='VALID',
        return_pooled_weights=True)

    expected_outputs = nn.max_pool(
        inputs, window_shape=window_shape, strides=strides, padding='VALID')
    expected_pooled_weights = jnp.ones((16, 8, 8))
    self.assertTrue(jnp.array_equal(outputs, expected_outputs))
    self.assertTrue(jnp.array_equal(pooled_weights, expected_pooled_weights))

  def test_weighted_avg_pool(self):
    """Tests weighted_avg_pool."""
    inputs_shape = (16, 32, 32, 20)
    window_shape = (4, 4)
    strides = (4, 4)
    inputs = jnp.array(np.random.normal(size=inputs_shape))
    weights = jnp.ones(inputs_shape[:-1])

    outputs, pooled_weights = nn_ops.weighted_avg_pool(
        inputs,
        weights,
        window_shape=window_shape,
        strides=strides,
        padding='VALID',
        return_pooled_weights=True)

    expected_outputs = nn.avg_pool(
        inputs, window_shape=window_shape, strides=strides, padding='VALID')
    expected_pooled_weights = jnp.ones((16, 8, 8))
    self.assertTrue(jnp.array_equal(outputs, expected_outputs))
    self.assertTrue(jnp.array_equal(pooled_weights, expected_pooled_weights))

  def test_extract_image_patches(self):
    """Tests extract_image_patches."""
    input_shape = (16, 3, 3, 32)
    inputs = np.array(np.random.normal(size=input_shape))

    # patching a 3x3 image to 3x3 patches, with no stride 1x1 and no dilation
    # and VALID padding should do nothing but reshaping the (bs, h, w, c) to
    # (bs, 1, 1, h, w, c)
    patched = nn_ops.extract_image_patches(
        inputs, (1, 3, 3, 1), (1, 1, 1, 1),
        padding='VALID',
        rhs_dilation=(1, 1, 1, 1))
    self.assertEqual(patched.shape, (16, 1, 1, 3, 3, 32))
    np.testing.assert_allclose(inputs, patched.reshape(input_shape), atol=1e-2)

  def test_upscale2x_nearest_neighbor(self):
    """Tests upscale2x_nearest_neighbor."""
    inputs = jnp.array(np.random.normal(size=(16, 32, 32, 128)))

    outputs = nn_ops.upscale2x_nearest_neighbor(inputs)
    # check the output shape
    self.assertEqual(outputs.shape, (16, 64, 64, 128))

  def test_central_crop(self):
    """Tests upscale2x_nearest_neighbor."""
    inputs = jnp.array(np.random.normal(size=(16, 32, 32, 128)))

    # check the case where the outputs should be same as the inputs
    outputs = nn_ops.central_crop(inputs, target_shape=(16, 32, 32, 128))
    self.assertTrue(jnp.array_equal(outputs, inputs))

    # check the output shape
    outputs = nn_ops.central_crop(inputs, target_shape=(16, 6, 6, 128))
    self.assertEqual(outputs.shape, (16, 6, 6, 128))

    inputs = jnp.arange(100.).reshape((1, 10, 10, 1))
    target_shape = (1, 8, 8, 1)
    output = nn_ops.central_crop(inputs, target_shape)
    # check up-left and down-right pixel of the output
    self.assertEqual(output[0, 0, 0, 0], 11.)
    self.assertEqual(output[0, -1, -1, 0], 88.)

  def test_extract_patches(self):
    """Tests extract_patches."""
    input_shape = (16, 3, 3, 32)
    inputs = np.array(np.random.normal(size=input_shape))

    # patching a 3x3 image to 3x3 patches, with no stride 1x1 should do nothing
    # but reshaping the (bs, h, w, c) to (bs, 1, 1, h, w, c)
    patched = nn_ops.extract_patches(inputs, (3, 3), (1, 1))
    self.assertEqual(patched.shape, (16, 1, 1, 3, 3, 32))
    np.testing.assert_allclose(inputs, patched.reshape(input_shape), atol=1e-2)

  @parameterized.named_parameters([('test_avg_pooling', 'avg_pooling'),
                                   ('test_max_pooling', 'max_pooling'),
                                   ('test_avg_pooling_bu', 'avg_pooling'),
                                   ('test_max_pooling_bu', 'max_pooling'),
                                   ('test_space_to_depth', 'space_to_depth')])
  def test_pooling(self, pooling_type):
    """Test Pooling module.

    Args:
      pooling_type: str; Type of pooling function from `['avg_pooling',
        'max_pooling', 'space_to_depth']`
    """
    inputs_shape = (16, 32, 32, 64)
    window_shape = (4, 4)
    strides = (4, 4)
    inputs = jnp.array(np.random.normal(size=inputs_shape))

    outputs = nn_ops.pooling(
        inputs,
        pooling_configs={'pooling_type': pooling_type},
        window_shape=window_shape,
        strides=strides)

    if pooling_type == 'space_to_depth':
      self.assertEqual(outputs.shape, (16, 8, 8, 1024))
    else:
      self.assertEqual(outputs.shape, (16, 8, 8, 64))

  @parameterized.named_parameters([
      ('test_4', (4, 28, 28, 32), (4, 4), (4, 4), 'VALID', (4, 7, 7, 4, 4, 32)),
      ('test_4_stride', (4, 28, 28, 32), (4, 4), (1, 1), 'VALID', (4, 25, 25, 4,
                                                                   4, 32)),
      ('test_4_stride_pad', (4, 28, 28, 32), (4, 4), (1, 1), 'SAME',
       (4, 28, 28, 4, 4, 32)),
      ('test_6_stride', (4, 28, 28, 32), (6, 6), (1, 1), 'VALID', (4, 23, 23, 6,
                                                                   6, 32)),
  ])
  def test_image_patcher(self, input_shape, patch_size, strides, padding,
                         expected_output_shape):
    """Tests ImagePatcher.

    Args:
      input_shape: tuple; Shape of the input data.
      patch_size: tuple; size of the patch: (height, width).
      strides: tuple; Specifies how far two consecutive patches are in the
        input.
      padding: str; The type of padding algorithm to use.
      expected_output_shape: expected shape of the output.
    """
    inputs = jnp.zeros(input_shape)

    image_patcher = functools.partial(
        nn_ops.patch_image,
        inputs_shape=input_shape,
        patch_size=patch_size,
        strides=strides,
        padding=padding,
        mode='i2p')

    # test output shape
    outputs = image_patcher(inputs)
    self.assertEqual(outputs.shape, expected_output_shape)

  @parameterized.named_parameters([
      ('test_q1k4', 1, 4, np.array([[0, 1, 2, 3]])),
      ('test_q5k1', 5, 1, np.array([[4], [3], [2], [1], [0]])),
      ('test_q2k3', 2, 3, np.array([[1, 2, 3], [0, 1, 2]])),
  ])
  def test_compute_1d_relative_distance(self, lenq, lenk,
                                        expected_relative_distance):
    """Tests compute_relative_positions."""
    relative_distance = nn_ops.compute_1d_relative_distance(lenq, lenk)
    # Test output values.
    self.assertTrue(
        np.array_equal(relative_distance, expected_relative_distance))

  def test_compute_1d_relative_distance_min_and_max(self):
    len_q = np.random.randint(0, 100, (1,))
    len_k = np.random.randint(0, 100, (1,))
    relative_distance = nn_ops.compute_1d_relative_distance(len_q, len_k)
    self.assertEqual(relative_distance.min(), 0)
    self.assertEqual(relative_distance.max(), len_q + len_k - 2)

  def test_truncated_normal_init(self):
    """Tests truncated_normal_initializer."""
    target_stddev = 0.4
    key = jax.random.PRNGKey(42)
    shape = (128, 128, 128)
    init_fn = nn_ops.truncated_normal_initializer(stddev=target_stddev)
    x = init_fn(key, shape, jnp.float32)
    self.assertAlmostEqual(target_stddev, jnp.std(x), places=2)

if __name__ == '__main__':
  absltest.main()
