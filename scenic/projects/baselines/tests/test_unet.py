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

"""Tests for unet.py."""

from absl.testing import absltest
from absl.testing import parameterized
import flax
from jax import random
import jax.numpy as jnp
from scenic.common_lib import debug_utils
from scenic.projects.baselines import unet


class UNetTest(parameterized.TestCase):
  """Test cases for UNet."""

  @parameterized.named_parameters(
      ("128_128", (128, 128), 34_491_599),
      # It's fully convolutional => same parameter number.
      ("256_256", (256, 256), 34_491_599),
  )
  def test_output_shape_and_param_count_of_unet_with_different_input_shapes(
      self, hw, param_count: int):
    """Test UNet model.

    We just test the output shape as well as number of trainable parameters,
    using two different input shapes, i.e. 128x128 and 256x256.
    We need to see the same shape as input in the output and given the all
    the components of the model are convolutions, we expect to see no change
    in the parameters of the model, when input resolutions changes.

    Args:
      hw: Height and Width of the input.
      param_count: Expected number of parameters.
    """
    rng = random.PRNGKey(0)
    dummy_input = jnp.zeros((2, *hw, 5), jnp.float32)
    output, init_var = unet.UNet(num_classes=5).init_with_output(
        rng, dummy_input, train=True, debug=False)
    # Check the output shape.
    self.assertEqual((2, *hw, 5), output.shape)

    _, init_params = flax.core.pop(init_var, "params")
    # Check the parameters count.
    num_trainable_params = debug_utils.log_param_shapes(init_params)
    self.assertEqual(param_count, num_trainable_params)


if __name__ == "__main__":
  absltest.main()
