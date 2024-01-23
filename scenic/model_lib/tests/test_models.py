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

"""Tests for models.py."""

from absl.testing import absltest
from absl.testing import parameterized
import flax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.tree_util
import numpy as np
from scenic.model_lib import models

NUM_OUTPUTS = 5
INPUT_SHAPE = (10, 32, 32, 3)


# Automatically test all defined classification models.
CLASSIFICATION_KEYS = [
    ('test_{}'.format(m), m) for m in models.CLASSIFICATION_MODELS.keys()
]

# Automatically test all defined segmentation models.
SEGMENTATION_KEYS = [
    ('test_{}'.format(m), m) for m in models.SEGMENTATION_MODELS.keys()
]


class ModelsTest(parameterized.TestCase):
  """Tests for all models."""

  @parameterized.named_parameters(*CLASSIFICATION_KEYS)
  def test_classification_models(self, model_name):
    """Test forward pass of the classification models."""

    model_cls = models.get_model_cls(model_name)
    rng = jax.random.PRNGKey(0)
    model = model_cls(
        config=None,
        dataset_meta_data={
            'num_classes': NUM_OUTPUTS,
            'target_is_onehot': False,
        })

    model_input_dtype = getattr(
        jnp, model.config.get('data_dtype_str', 'float32'))

    xs = jnp.array(np.random.normal(loc=0.0, scale=10.0,
                                    size=INPUT_SHAPE)).astype(model_input_dtype)

    rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.zeros(INPUT_SHAPE, model_input_dtype)
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

  @parameterized.named_parameters(*SEGMENTATION_KEYS)
  def test_segmentation_models(self, model_name):
    """Test forward pass of the segmentation models."""

    model_cls = models.get_model_cls(model_name)
    rng = jax.random.PRNGKey(0)
    model = model_cls(
        config=None,
        dataset_meta_data={
            'num_classes': NUM_OUTPUTS,
            'target_is_onehot': False,
        })

    model_input_dtype = model.config.get('default_input_dtype', jnp.float32)
    xs = jnp.array(np.random.normal(loc=0.0, scale=10.0,
                                    size=INPUT_SHAPE)).astype(model_input_dtype)

    rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.zeros(INPUT_SHAPE, model_input_dtype)
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
    self.assertEqual(outputs.shape, INPUT_SHAPE[:3] + (NUM_OUTPUTS,))

    # If it's a batch norm model check the batch stats changed.
    if init_model_state:
      bflat, _ = ravel_pytree(init_model_state)
      new_bflat, _ = ravel_pytree(new_model_state)
      self.assertFalse(jnp.array_equal(bflat, new_bflat))

    # Test batch_norm in inference mode.
    outputs = model.flax_model.apply(
        variables, xs, mutable=False, train=False, debug=False)
    self.assertEqual(outputs.shape, INPUT_SHAPE[:3] + (NUM_OUTPUTS,))


if __name__ == '__main__':
  absltest.main()
