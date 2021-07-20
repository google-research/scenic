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

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for optimizers.py."""

from absl.testing import absltest
from flax import optim
from flax import struct
import jax.numpy as jnp
import numpy as np
from scenic.train_lib import optimizers


class MomentumHPTest(absltest.TestCase):
  """Tests for MomentumHP Optimizer."""

  @struct.dataclass
  class HyperParams:
    learning_rate: np.ndarray
    beta: np.ndarray

  @struct.dataclass
  class State:
    momentum: np.ndarray

  def test_init_state(self):
    """Tests initializing state of the optimizer."""
    params = np.zeros((1,))
    optimizer_def = optimizers.MomentumHP(learning_rate=0.1, beta=0.2)
    state = optimizer_def.init_state(params)
    expected_hyper_params = self.HyperParams(jnp.array(0.1), jnp.array(0.2))
    self.assertEqual(optimizer_def.hyper_params, expected_hyper_params)
    expected_state = optim.OptimizerState(0, self.State(np.zeros((1,))))
    self.assertEqual(state, expected_state)

  def test_expected_state_and_params_after_apply_gradient(self):
    """Tests if apply gradient changes params and sate as expected."""
    optimizer_def = optimizers.MomentumHP(learning_rate=0.1, beta=0.2)
    params = np.ones((1,))
    state = optim.OptimizerState(0, self.State(np.array([1.0])))
    grads = np.array([3.0])
    new_params, new_state = optimizer_def.apply_gradient(
        optimizer_def.hyper_params, params, state, grads)
    expected_new_state = optim.OptimizerState(1, self.State(np.array([3.2])))
    expected_new_params = np.array([1.0 - 0.32])
    self.assertEqual(new_params, expected_new_params)
    self.assertEqual(new_state, expected_new_state)
