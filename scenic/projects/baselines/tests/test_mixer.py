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

"""Tests for mixer.py."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import flax
from jax import random
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.tree_util
import numpy as np
from scenic.projects.baselines import mixer

NUM_OUTPUTS = 5
INPUT_SHAPE = (10, 32, 32, 3)


class MixerTest(parameterized.TestCase):
  """Tests for modules in mixer.py."""

  @parameterized.named_parameters(
      ('without_dropout_without_stochastic_depth', 0.0, 0.0),
      ('with_dropout_without_stochastic_depth', 0.1, 0.0),
      ('without_dropout_with_stochastic_depth', 0.0, 0.1),
      ('with_dropout_with_stochastic_depth', 0.1, 0.1),
      ('with_dropout_stochastic_depth_layer_scale', 0.1, 0.1, 0.1)
  )
  def test_mixer_block(self, dropout_rate, stochastic_depth, layer_scale=None):
    """Tests MixerBlock."""
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 16, 32))
    mixer_block = functools.partial(
        mixer.MixerBlock,
        channels_mlp_dim=32,
        sequence_mlp_dim=32,
        dropout_rate=dropout_rate,
        stochastic_depth=stochastic_depth,
        layer_scale=layer_scale)
    mixer_block_vars = mixer_block().init(rng, x, deterministic=True)
    y = mixer_block().apply(mixer_block_vars, x, deterministic=True)
    # Test outputs shape.
    self.assertEqual(y.shape, x.shape)

  def test_mixer_models(self):
    """Test forward pass of the mixer classification model."""
    rng = jax.random.PRNGKey(0)
    model = mixer.MixerMultiLabelClassificationModel(
        config=None,
        dataset_meta_data={
            'num_classes': NUM_OUTPUTS,
            'target_is_onehot': False,
        })

    xs = jnp.array(np.random.normal(loc=0.0, scale=10.0,
                                    size=INPUT_SHAPE)).astype(jnp.float32)

    rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.zeros(INPUT_SHAPE, jnp.float32)
    init_model_state, init_params = flax.core.pop(model.flax_model.init(
        init_rng, dummy_input, train=False, debug=False), 'params')

    # Check that the forward pass works with mutated model_state.
    rng, dropout_rng = jax.random.split(rng)
    variables = {'params': init_params, **init_model_state}
    outputs, new_model_state = model.flax_model.apply(
        variables,
        xs,
        mutable=['batch_stats'],
        train=True,
        rngs={'dropout': dropout_rng},
        debug=False)
    self.assertEqual(outputs.shape, (INPUT_SHAPE[0], NUM_OUTPUTS))

    # If it's a batch norm model check the batch stats changed.
    if init_model_state:
      bflat, _ = ravel_pytree(init_model_state)
      new_bflat, _ = ravel_pytree(new_model_state)
      self.assertFalse(jnp.array_equal(bflat, new_bflat))

    # Test batch_norm in inference mode.
    outputs = model.flax_model.apply(
        variables, xs, mutable=False, train=False, debug=False)
    self.assertEqual(outputs.shape, (INPUT_SHAPE[0], NUM_OUTPUTS))


if __name__ == '__main__':
  absltest.main()
