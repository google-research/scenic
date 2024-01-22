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

"""Tests for scenic optimizers."""
from absl.testing import absltest
from absl.testing import parameterized
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils
import tensorflow as tf


# Convenience constants for names of the parameters for the MLP used.
DENSE_0_BIAS = 'Dense_0/bias'
DENSE_0_KERNEL = 'Dense_0/kernel'
DENSE_1_BIAS = 'Dense_1/bias'
DENSE_1_KERNEL = 'Dense_1/kernel'


class MLP(nn.Module):
  """A simple MLP model. initialized with ones for testing weight decay."""

  @nn.compact
  def __call__(self, x, train=None, debug=None):
    x = nn.Dense(features=3, bias_init=nn.initializers.ones,
                 kernel_init=nn.initializers.ones)(x)
    x = nn.relu(x)
    x = nn.Dense(features=1, bias_init=nn.initializers.ones,
                 kernel_init=nn.initializers.ones)(x)
    return x


class OptimizersTest(tf.test.TestCase, parameterized.TestCase):
  """Class for testing optimizers.py."""

  def setUp(self):
    """Creates parameters and gradient function."""
    super().setUp()
    rng = jax.random.PRNGKey(0)
    model = MLP()
    config = ml_collections.ConfigDict()

    self.params, _, _, _ = train_utils.initialize_model(
        model_def=model,
        input_spec=[((1, 2), jnp.float32)],
        config=config,
        rngs=rng)

    def training_loss_fn(params, label):
      prediction = model.apply(
          variables={'params': params},
          x=jnp.array([[1., 1.]]))
      return jnp.mean(jnp.square(prediction - label))

    self.compute_gradient_fn = jax.value_and_grad(training_loss_fn)
    self.label_causing_no_loss = jnp.array([10.])
    self.label_causing_loss = jnp.array([9.])

    self.lr = 0.1
    self.expected_sgd_updates = {  # For LR = 0.1
        DENSE_0_BIAS: jnp.array([-0.2, -0.2, -0.2]),
        DENSE_0_KERNEL: jnp.array([[-0.2, -0.2, -0.2], [-0.2, -0.2, -0.2]]),
        DENSE_1_BIAS: jnp.array([-0.2]),
        DENSE_1_KERNEL: jnp.array([[-0.6], [-0.6], [-0.6]]),
    }

  def test_sgd(self):
    """Test obtaining basic sgd optimizer."""
    optimizer_config = ml_collections.ConfigDict()
    optimizer_config.optimizer = 'sgd'
    optimizer = optimizers.get_optimizer(optimizer_config, self.lr, self.params)
    optimizer_state = optimizer.init(self.params)

    _, grad = self.compute_gradient_fn(self.params, self.label_causing_loss)
    updates, _ = optimizer.update(grad, optimizer_state, self.params)
    updates = flax.traverse_util.flatten_dict(updates, sep='/')

    for param_name in self.expected_sgd_updates:
      self.assertAllClose(
          updates[param_name],
          self.expected_sgd_updates[param_name])

  @parameterized.parameters(0.01, 0.1)
  def test_sgd_with_weight_decay(self, weight_decay):
    """Test SGD with weight decay."""
    optimizer_config = ml_collections.ConfigDict()
    optimizer_config.optimizer = 'sgd'
    optimizer_config.weight_decay = weight_decay
    optimizer = optimizers.get_optimizer(optimizer_config, self.lr, self.params)
    optimizer_state = optimizer.init(self.params)

    _, grad = self.compute_gradient_fn(self.params, self.label_causing_loss)
    updates, _ = optimizer.update(grad, optimizer_state, self.params)
    updates = flax.traverse_util.flatten_dict(updates, sep='/')

    for param_name in self.expected_sgd_updates:
      self.assertAllClose(
          updates[param_name],
          self.expected_sgd_updates[param_name] - weight_decay * self.lr)

  @parameterized.parameters(0.01, 0.1)
  def test_sgd_with_weight_decay_skip_bias(self, weight_decay):
    """Test weight decay which skips bias parameters."""
    optimizer_config = ml_collections.ConfigDict()
    optimizer_config.optimizer = 'sgd'
    optimizer_config.weight_decay = weight_decay
    optimizer_config.skip_scale_and_bias_regularization = True
    optimizer = optimizers.get_optimizer(optimizer_config, self.lr, self.params)
    optimizer_state = optimizer.init(self.params)

    _, grad = self.compute_gradient_fn(self.params, self.label_causing_loss)
    updates, _ = optimizer.update(grad, optimizer_state, self.params)
    updates = flax.traverse_util.flatten_dict(updates, sep='/')

    # Biases are unaffected by weight decay.
    self.assertAllClose(
        updates[DENSE_0_BIAS],
        self.expected_sgd_updates[DENSE_0_BIAS])
    self.assertAllClose(
        updates[DENSE_1_BIAS],
        self.expected_sgd_updates[DENSE_1_BIAS])

    # Kernels are affected by weight decay.
    self.assertAllClose(
        updates[DENSE_0_KERNEL],
        self.expected_sgd_updates[DENSE_0_KERNEL] - weight_decay * self.lr)
    self.assertAllClose(
        updates[DENSE_1_KERNEL],
        self.expected_sgd_updates[DENSE_1_KERNEL] - weight_decay * self.lr)

  @parameterized.named_parameters(
      {
          'testcase_name': 'bias',
          'freeze_params_reg_exp': 'bias',
          'frozen_params': [DENSE_0_BIAS, DENSE_1_BIAS]
      },
      {
          'testcase_name': 'kernel',
          'freeze_params_reg_exp': 'kernel',
          'frozen_params': [DENSE_0_KERNEL, DENSE_1_KERNEL]
      },
      {
          'testcase_name': 'bias_1',
          'freeze_params_reg_exp': '_1/bias',
          'frozen_params': [DENSE_1_BIAS]
      },
  )
  def test_sgd_freeze_params(self, freeze_params_reg_exp, frozen_params):
    """Test freezing of parameters."""
    optimizer_config = ml_collections.ConfigDict()
    optimizer_config.optimizer = 'sgd'
    optimizer_config.freeze_params_reg_exp = freeze_params_reg_exp
    optimizer = optimizers.get_optimizer(optimizer_config, self.lr, self.params)
    optimizer_state = optimizer.init(self.params)

    _, grad = self.compute_gradient_fn(self.params, self.label_causing_loss)
    updates, _ = optimizer.update(grad, optimizer_state, self.params)
    updates = flax.traverse_util.flatten_dict(updates, sep='/')

    for param_name in self.expected_sgd_updates:
      if param_name in frozen_params:
        self.assertAllClose(
            updates[param_name],
            jnp.zeros_like(self.expected_sgd_updates[param_name]))
      else:
        self.assertAllClose(
            updates[param_name],
            self.expected_sgd_updates[param_name])

  @parameterized.named_parameters(
      {
          'testcase_name': 'bias',
          'freeze_params_reg_exp': 'bias',
          'frozen_params': [DENSE_0_BIAS, DENSE_1_BIAS]
      },
      {
          'testcase_name': 'kernel',
          'freeze_params_reg_exp': 'kernel',
          'frozen_params': [DENSE_0_KERNEL, DENSE_1_KERNEL]
      },
      {
          'testcase_name': 'bias_1',
          'freeze_params_reg_exp': '_1/bias',
          'frozen_params': [DENSE_1_BIAS]
      },
  )
  def test_freeze_params_optimizer_state_frozen(self, freeze_params_reg_exp,
                                                frozen_params):
    """Optimizer state (sgd with momentum) is not updated for frozen params."""
    optimizer_config = ml_collections.ConfigDict()
    optimizer_config.optimizer = 'sgd'
    optimizer_config.momentum = 0.9
    optimizer_config.freeze_params_reg_exp = freeze_params_reg_exp
    optimizer = optimizers.get_optimizer(optimizer_config, self.lr, self.params)
    optimizer_state = optimizer.init(self.params)

    optimizer_dict_state = flax.traverse_util.flatten_dict(
        optimizer_state[0].inner_state[0].trace, sep='/')

    for param_name in self.expected_sgd_updates:
      if param_name in frozen_params:
        self.assertIsInstance(optimizer_dict_state[param_name],
                              optax.MaskedNode)
      else:
        self.assertAllEqual(
            optimizer_dict_state[param_name],
            jnp.zeros_like(self.expected_sgd_updates[param_name]))

  def test_sgd_freeze_params_with_weight_decay(self):
    """Frozen parameters should be unaffected by weight decay."""
    freeze_params_reg_exp = 'bias'
    frozen_params = [DENSE_0_BIAS, DENSE_1_BIAS]
    weight_decay = 0.1
    optimizer_config = ml_collections.ConfigDict()
    optimizer_config.optimizer = 'sgd'
    optimizer_config.weight_decay = weight_decay
    optimizer_config.freeze_params_reg_exp = freeze_params_reg_exp
    optimizer = optimizers.get_optimizer(optimizer_config, self.lr, self.params)
    optimizer_state = optimizer.init(self.params)

    _, grad = self.compute_gradient_fn(self.params, self.label_causing_loss)
    updates, _ = optimizer.update(grad, optimizer_state, self.params)
    updates = flax.traverse_util.flatten_dict(updates, sep='/')

    for param_name in self.expected_sgd_updates:
      if param_name in frozen_params:
        self.assertAllEqual(
            updates[param_name],
            jnp.zeros_like(self.expected_sgd_updates[param_name]))
      else:
        self.assertAllClose(
            updates[param_name],
            self.expected_sgd_updates[param_name] - weight_decay * self.lr)

  def test_freeze_params_reg_exp_matches_all_params_raises_value_error(self):
    """Freeze parameter reg_exp is not allowed to freeze the complete model."""
    optimizer_config = ml_collections.ConfigDict()
    optimizer_config.optimizer = 'sgd'
    optimizer_config.freeze_params_reg_exp = '.*'

    with self.assertRaises(ValueError):
      optimizers.get_optimizer(optimizer_config, self.lr, self.params)

  def test_invalid_config_raises_type_error(self):
    """Invalid configuration should raise error instead of failing silently."""
    optimizer_config = ml_collections.ConfigDict()
    optimizer_config.optimizer = 'sgd'
    optimizer_config.invalid_field = 'invalid'
    with self.assertRaises(TypeError):
      optimizers.get_optimizer(optimizer_config, self.lr, self.params)


if __name__ == '__main__':
  absltest.main()
