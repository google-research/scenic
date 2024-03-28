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

"""Tests for vit.py."""

from absl.testing import absltest
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
import jax.tree_util
import ml_collections
import numpy as np
from scenic.projects.baselines import vit

NUM_OUTPUTS = 5
INPUT_SHAPE = (10, 32, 32, 3)


class ViTTest(parameterized.TestCase):
  """Tests for modules in vit.py."""

  @parameterized.named_parameters(
      ('pos_embed_learned_1d', 'learned_1d', False),
      ('pos_embed_sinusoidal_1d', 'sinusoidal_1d', False),
      ('pos_embed_sinusoidal_2d', 'sinusoidal_2d', False),
      ('pos_embed_none', 'none', False),
      ('default_config', 'ignored', True)
  )
  def test_vit_model(self, positional_embedding, use_default_config):
    """Test forward pass of the ViT classification model."""
    rng = jax.random.PRNGKey(0)

    # Config for model.
    if not use_default_config:
      config = ml_collections.ConfigDict({
          'model':
              dict(
                  num_heads=2,
                  num_layers=1,
                  representation_size=16,
                  mlp_dim=32,
                  dropout_rate=0.,
                  attention_dropout_rate=0.,
                  hidden_size=16,
                  patches={'size': (4, 4)},
                  positional_embedding=positional_embedding,
                  classifier='gap',
                  data_dtype_str='float32')
      })
    else:
      config = None
    model = vit.ViTMultiLabelClassificationModel(
        config=config,
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

    # Check that the forward pass works in train mode.
    rng, dropout_rng = jax.random.split(rng)
    variables = {'params': init_params, **init_model_state}
    outputs = model.flax_model.apply(
        variables,
        xs,
        train=True,
        rngs={'dropout': dropout_rng},
        debug=False)
    self.assertEqual(outputs.shape, (INPUT_SHAPE[0], NUM_OUTPUTS))

    # Test model in inference mode.
    outputs = model.flax_model.apply(
        variables, xs, mutable=False, train=False, debug=False)
    self.assertEqual(outputs.shape, (INPUT_SHAPE[0], NUM_OUTPUTS))


if __name__ == '__main__':
  absltest.main()
