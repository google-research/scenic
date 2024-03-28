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

"""Tests for OmniNet model.py."""

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
from jax import random
import jax.numpy as jnp
import ml_collections
from scenic.projects.omninet import model


class OmniNetModelTest(parameterized.TestCase):
  """Tests for modules in omninet model.py."""

  @parameterized.parameters(
      itertools.product([True, False], [1, 2, 4], ['max', 'last']))
  def test_omnimixer_output_shape(self, skip_standard, partition, pool):
    """Tests validity of output's shape of OmniMixerEncoder module."""
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 16, 32))
    omnimixer_configs = ml_collections.ConfigDict({
        'skip_standard': skip_standard,
        'partition': partition,
        'pool': pool,
        'depth_mlp_dim': 16,
    })

    omnimixer_encoder_def = functools.partial(
        model.OmniMixerEncoder,
        num_layers=4,
        channels_mlp_dim=32,
        sequence_mlp_dim=8,
        omnimixer=omnimixer_configs)
    omnimixer_encoder_vars = omnimixer_encoder_def().init(rng, x, train=False)
    y = omnimixer_encoder_def().apply(omnimixer_encoder_vars, x, train=False)
    # Test outputs shape.
    self.assertEqual(y.shape, x.shape)


if __name__ == '__main__':
  absltest.main()
