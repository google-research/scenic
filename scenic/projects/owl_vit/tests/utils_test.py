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

"""Tests OWL-ViT utils."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from scenic.projects.owl_vit import utils

# Config for a tiny transformer for testing.
TINY_VIT_CONFIG = {'depth': 2, 'width': 64, 'mlp_dim': 256, 'num_heads': 2}


class UtilsTest(parameterized.TestCase):
  """Tests for ViT+ utilities."""

  @parameterized.parameters(
      ((2, 224, 224, 3), (2, 49, 50), (2, 7, 7, 50)),  # patch_size: 32
      ((2, 1333, 1333, 3), (2, 7056, 50), (2, 84, 84, 50)),  # patch_size: 16
      )
  def test_seq2img(self, img_shape, features_shape, expected_shape):
    images = jnp.ones(img_shape)
    features = jnp.ones(features_shape)
    features_2d = utils.seq2img(images, features)
    self.assertEqual(expected_shape, features_2d.shape)

  def test_normalized_grid_corner_coordinates(self):
    feature_map = jnp.ones((2, 3, 3, 10))
    corner_coords = utils.normalized_grid_corner_coordinates(feature_map, None)
    expected_coords = jnp.array(
        [[0.33333334, 0.33333334], [0.6666667, 0.33333334],
         [1., 0.33333334], [0.33333334, 0.6666667],
         [0.6666667, 0.6666667], [1., 0.6666667],
         [0.33333334, 1.], [0.6666667, 1.],
         [1., 1.]])
    np.testing.assert_allclose(expected_coords, corner_coords, atol=1e-5)

  def test_compute_box_bias(self):
    feature_map = jnp.ones((2, 3, 3, 10))
    box_bias = utils.compute_box_bias(feature_map)
    expected_bias = jnp.array(
        [[-0.69299614, -0.69299614, -0.692997, -0.692997],
         [0.692996260, -0.69299614, -0.692997, -0.692997],
         [9.210265000, -0.69299614, -0.692997, -0.692997],
         [-0.69299614, 0.692996260, -0.692997, -0.692997],
         [0.692996260, 0.692996260, -0.692997, -0.692997],
         [9.210265000, 0.692996260, -0.692997, -0.692997],
         [-0.69299614, 9.210265000, -0.692997, -0.692997],
         [0.692996260, 9.210265000, -0.692997, -0.692997],
         [9.210265000, 9.210265000, -0.692997, -0.692997]],)
    np.testing.assert_allclose(expected_bias, box_bias, atol=1e-5)

  def test_compute_box_bias_location(self):
    feature_map = jnp.ones((2, 3, 3, 10))
    box_bias = utils.compute_box_bias(feature_map, kind='location')
    expected_bias = jnp.array(
        [[-0.69299614, -0.69299614, 0.0, 0.0],
         [0.692996260, -0.69299614, 0.0, 0.0],
         [9.210265000, -0.69299614, 0.0, 0.0],
         [-0.69299614, 0.692996260, 0.0, 0.0],
         [0.692996260, 0.692996260, 0.0, 0.0],
         [9.210265000, 0.692996260, 0.0, 0.0],
         [-0.69299614, 9.210265000, 0.0, 0.0],
         [0.692996260, 9.210265000, 0.0, 0.0],
         [9.210265000, 9.210265000, 0.0, 0.0]],)
    np.testing.assert_allclose(expected_bias, box_bias, atol=1e-5)

  def test_compute_box_bias_size(self):
    feature_map = jnp.ones((2, 3, 3, 10))
    box_bias = utils.compute_box_bias(feature_map, kind='size')
    expected_bias = jnp.array(
        [[0.0, 0.0, -0.692997, -0.692997],
         [0.0, 0.0, -0.692997, -0.692997],
         [0.0, 0.0, -0.692997, -0.692997],
         [0.0, 0.0, -0.692997, -0.692997],
         [0.0, 0.0, -0.692997, -0.692997],
         [0.0, 0.0, -0.692997, -0.692997],
         [0.0, 0.0, -0.692997, -0.692997],
         [0.0, 0.0, -0.692997, -0.692997],
         [0.0, 0.0, -0.692997, -0.692997]],)
    np.testing.assert_allclose(expected_bias, box_bias, atol=1e-5)

  @parameterized.parameters((0.1, -2.1972246), (0.01, -4.59512))
  def test_init_classification_bias(self, prior_prob, expected_init_bias):
    bias = jnp.zeros((1))
    init_bias = utils.init_classification_bias(bias, prior_prob)
    self.assertAlmostEqual(expected_init_bias, init_bias[0])

  def test_dot_product_similarity(self):
    b, n, m, d = 2, 3, 4, 5
    rng = jax.random.PRNGKey(0)
    rng_x, rng_y = jax.random.split(rng)
    x = jax.random.uniform(rng_x, (b, n, d))
    y = jax.random.uniform(rng_y, (b, m, d))

    self_sim = utils.dot_product_similarity(x, x)
    self.assertEqual(self_sim.shape, (b, n, n))

    cross_sim = utils.dot_product_similarity(x, y)
    self.assertEqual(cross_sim.shape, (b, n, m))


if __name__ == '__main__':
  absltest.main()
