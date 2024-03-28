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

"""Tests for axial_resnet.py."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import random
import jax.numpy as jnp
import ml_collections
from scenic.projects.baselines import axial_resnet


class AxialResNetTest(parameterized.TestCase):
  """Test cases for AxialResNet."""

  def test_self_attention_with_1d_relative_pos_output_shape(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 6, 128))
    sa_module = axial_resnet.SelfAttentionWith1DRelativePos(num_heads=8)
    y, _ = sa_module.init_with_output(rng, x)
    self.assertEqual(y.shape, x.shape)

  def test_self_attention_with_1d_relative_pos_unacceptable_num_features(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 6, 5))
    # The channel size should be divisible by `2 * num_heads`, which is not:
    with self.assertRaises(ValueError):
      axial_resnet.SelfAttentionWith1DRelativePos(num_heads=8).init(rng, x)

  @parameterized.named_parameters(('row', 1), ('col', 2), ('channel', 3))
  def test_axial_self_attention_output_shape(self, attention_axis):
    """Tests AxialSelfAttention module given different attention axis."""
    rng = random.PRNGKey(0)
    x = jnp.ones((6, 8, 10, 128))
    axial_attention_configs = ml_collections.ConfigDict({'num_heads': 4})
    if attention_axis not in [1, 2]:
      with self.assertRaises(ValueError):
        axial_resnet.AxialSelfAttention(
            attention_axis=attention_axis,
            axial_attention_configs=axial_attention_configs).init_with_output(
                rng, x)
    else:
      asa_module = axial_resnet.AxialSelfAttention(
          attention_axis=attention_axis,
          axial_attention_configs=axial_attention_configs)
      y, _ = asa_module.init_with_output(rng, x)
      self.assertEqual(y.shape, x.shape)

  @parameterized.named_parameters(
      ('strides_1', (1, 1), False, (10, 32, 32, 128)),
      ('strides_2', (2, 2), False, (10, 16, 16, 128)),
      ('strides_1_bottleneck', (1, 1), True, (10, 32, 32, 512)),
      ('strides_2_bottleneck', (2, 2), True, (10, 16, 16, 512)),
  )
  def test_axial_residual_unit_output_shape(self, strides, bottleneck,
                                            expected_output_shape):
    """Tests AxialResidualUnit module given different strides."""
    rng = random.PRNGKey(0)
    x = jnp.ones((10, 32, 32, 64))
    axial_attention_configs = ml_collections.ConfigDict({'num_heads': 4})
    aru_module = axial_resnet.AxialResidualUnit(
        nout=128,
        strides=strides,
        bottleneck=bottleneck,
        axial_attention_configs=axial_attention_configs)
    y, _ = aru_module.init_with_output(rng, x)
    self.assertEqual(y.shape, expected_output_shape)

  @parameterized.named_parameters(
      ('strides_1_block1', (1, 1), False, 1, (10, 32, 32, 128)),
      ('strides_2__block1', (2, 2), False, 1, (10, 16, 16, 128)),
      ('strides_1_bottleneck_block1', (1, 1), True, 1, (10, 32, 32, 512)),
      ('strides_2_bottleneck_block1', (2, 2), True, 1, (10, 16, 16, 512)),
      ('strides_1_block2', (1, 1), False, 2, (10, 32, 32, 128)),
      ('strides_2__block2', (2, 2), False, 2, (10, 16, 16, 128)),
      ('strides_1_bottleneck_block2', (1, 1), True, 2, (10, 32, 32, 512)),
      ('strides_2_bottleneck_block2', (2, 2), True, 2, (10, 16, 16, 512)),
  )
  def test_axial_residual_stage_output_shape(self, strides, bottleneck,
                                             block_size, expected_output_shape):
    """Tests AxialResNetStage module given different strides."""
    rng = random.PRNGKey(0)
    x = jnp.ones((10, 32, 32, 64))
    axial_attention_configs = ml_collections.ConfigDict({'num_heads': 4})
    aru_module = axial_resnet.AxialResNetStage(
        block_size=block_size,
        nout=128,
        first_stride=strides,
        bottleneck=bottleneck,
        axial_attention_configs=axial_attention_configs)
    y, _ = aru_module.init_with_output(rng, x)
    self.assertEqual(y.shape, expected_output_shape)


if __name__ == '__main__':
  absltest.main()
