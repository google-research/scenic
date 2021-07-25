# Copyright 2021 The Scenic Authors.
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

"""Tests for nn_layers.py."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import random
import jax.numpy as jnp
import numpy as np
from scenic.model_lib.layers import nn_layers


class NNLayersTest(parameterized.TestCase):
  """Tests for modules in nn_layers.py."""

  @parameterized.named_parameters([('test_residual', 'add'),
                                   ('test_highway', 'highway'),
                                   ('test_rezero', 'rezero'),
                                   ('test_sigtanh', 'sigtanh'),
                                   ('test_gated', 'gated')])
  def test_residual(self, residual_type):
    """Test Residual module."""
    rng = random.PRNGKey(0)
    inputs_shape = (16, 32, 32, 3)

    residual_module_def = nn_layers.Residual(
        residual_type=residual_type)
    x = jnp.array(np.random.normal(size=inputs_shape))
    y = jnp.array(np.random.normal(size=inputs_shape))
    outputs, _ = residual_module_def.init_with_output(rng, x, y)

    # test output shape for 4d inputs
    self.assertEqual(outputs.shape, inputs_shape)

    # test the residual connection
    if residual_type == 'add':
      self.assertTrue(jnp.array_equal(outputs, x + y))

    # make sure the residual connection is an identity mapping at initialization
    else:
      np.testing.assert_allclose(outputs, x, atol=1e-3)

  @parameterized.named_parameters([('test_red_1_axis_1', 1),
                                   ('test_red_4_axis_1', 4)])
  def test_squeeze_and_excite(self, reduction_factor):
    """Test the SqueezeAndExcite module."""
    rng = random.PRNGKey(0)
    inputs_shape = (16, 24, 32, 64)
    inputs = jnp.array(np.random.normal(size=inputs_shape))

    squeeze_and_excite_def = nn_layers.SqueezeAndExcite(
        reduction_factor=reduction_factor)

    # test output shape
    outputs, _ = squeeze_and_excite_def.init_with_output(rng, inputs)
    self.assertEqual(outputs.shape, inputs_shape)


if __name__ == '__main__':
  absltest.main()
