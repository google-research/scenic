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
