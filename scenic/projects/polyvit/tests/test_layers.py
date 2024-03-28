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

"""Tests for PolyViT layers."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from jax import random
import jax.numpy as jnp
import ml_collections
from scenic.projects.polyvit import layers


class PolyViTLayersTests(parameterized.TestCase):
  """Tests for modules in polyvit layers.py."""

  @parameterized.named_parameters([
      ('test_32x32', (4, 32, 32, 32), (4, 64, 128)),
      ('test_64x64', (4, 64, 64, 32), (4, 256, 128)),
      ('test_5d', (4, 32, 32, 32, 32), None),
  ])
  def test_tokenizer2d_output_shape(self, input_shape, expected_output_shape):
    """Tests Tokenizer2D."""
    rng = random.PRNGKey(0)
    x = jnp.ones(input_shape)
    patches = ml_collections.ConfigDict({'size': (4, 4)})
    hidden_size = 128
    tokenizer_2d_def = functools.partial(
        layers.Tokenizer2D,
        hidden_size=hidden_size,
        patches=patches,
        mlp_dim=512,
        num_layers=0,
        num_heads=2,
    )

    if expected_output_shape is not None:
      y, _ = tokenizer_2d_def().init_with_output(
          {'params': rng, 'dropout': rng},
          x,
          train=True,
          stochastic_droplayer_rate=None,
          dataset='',
      )
      # Test outputs shape.
      self.assertEqual(y.shape, expected_output_shape)
    else:
      with self.assertRaises(ValueError):
        tokenizer_2d_def().init_with_output(
            {'params': rng, 'dropout': rng},
            x,
            train=True,
            stochastic_droplayer_rate=None,
            dataset='',
        )

  @parameterized.named_parameters([
      ('test_16x32x32', (4, 16, 32, 32, 32), (4, 512, 128)),
      ('test_16x64x64', (4, 16, 64, 64, 32), (4, 2048, 128)),
      ('test_4d', (4, 32, 32, 32), None),
  ])
  def test_tokenizer3d_output_shape(self, input_shape, expected_output_shape):
    """Tests Tokenizer3D."""
    rng = random.PRNGKey(0)
    x = jnp.ones(input_shape)
    patches = ml_collections.ConfigDict({'size': (2, 4, 4)})
    hidden_size = 128
    tokenizer_3d_def = functools.partial(
        layers.Tokenizer3D,
        hidden_size=hidden_size,
        patches=patches,
        kernel_init_method=None,
        mlp_dim=512,
        num_layers=0,
        num_heads=2,
    )

    if expected_output_shape is not None:
      y, _ = tokenizer_3d_def().init_with_output(
          {'params': rng, 'dropout': rng},
          x,
          train=True,
          stochastic_droplayer_rate=None,
          dataset='',
      )
      # Test outputs shape.
      self.assertEqual(y.shape, expected_output_shape)
    else:
      with self.assertRaises(ValueError):
        tokenizer_3d_def().init_with_output(
            {'params': rng, 'dropout': rng},
            x,
            train=True,
            stochastic_droplayer_rate=None,
            dataset='',
        )


if __name__ == '__main__':
  absltest.main()
