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

"""Unit tests for regression_model.py."""

from absl.testing import absltest
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models import regression_model


class FakeRegressionModel(regression_model.RegressionModel):
  """A dummy regression model for testing purposes."""

  def __init__(self):
    dataset_meta_data = {}
    super().__init__(ml_collections.ConfigDict(), dataset_meta_data)

  def build_flax_model(self):
    pass

  def default_flax_model_config(self):
    pass


def get_fake_batch_and_predictions():
  """Generates a fake `batch`."""
  targets = jnp.array(
      [[2.0, 1.0, 0.0, 1.0],
       [2.0, 1.0, 0.0, 1.0],
       [5.0, 7.0, 0.0, 1.0]])
  predictions = jnp.array(
      [[2.0, 0.0, 0.0, 1.0],
       [2.0, 1.0, 0.0, 1.0],
       [4.0, 10.0, 0.0, 1.0]])
  fake_batch = {
      'inputs': None,
      'targets': targets
  }
  return fake_batch, predictions


class TestRegressionModel(absltest.TestCase):
  """Tests for the a fake regression model."""

  def test_loss_function(self):
    """Tests loss_function by checking its output's validity."""
    model = FakeRegressionModel()
    batch, predictions = get_fake_batch_and_predictions()
    batch_replicated, predictions_replicated = (
        jax_utils.replicate(batch), jax_utils.replicate(predictions))

    # Test loss function in the pmapped setup:
    loss_function_pmapped = jax.pmap(model.loss_function, axis_name='batch')
    total_loss = loss_function_pmapped(predictions_replicated, batch_replicated)
    total_loss = jax_utils.unreplicate(total_loss)
    # Loss =  1/3 * (|[0, 1, 0, 0]|^2 + |[0, 0, 0, 0|^2 + |[1, 3, 0, 0]|^2)
    self.assertAlmostEqual(total_loss, 11 / 3)

  def test_metric_function(self):
    """Tests metric_function by checking its output's format and validity."""
    model = FakeRegressionModel()
    batch, predictions = get_fake_batch_and_predictions()
    batch_replicated, predictions_replicated = (
        jax_utils.replicate(batch), jax_utils.replicate(predictions))

    metrics_fn_pmapped = jax.pmap(model.get_metrics_fn(), axis_name='batch')
    all_metrics = metrics_fn_pmapped(predictions_replicated, batch_replicated)
    expected_metrics_keys = ['mean_squared_error']
    self.assertSameElements(expected_metrics_keys, all_metrics.keys())

    all_metrics = jax_utils.unreplicate(all_metrics)
    self.assertLen(all_metrics, 1)

    mse_sum_count = all_metrics['mean_squared_error']
    # (|[0, 1, 0, 0]|^2 + |[0, 0, 0, 0|^2 + |[1, 3, 0, 0]|^2) = 11
    self.assertAlmostEqual(mse_sum_count[0], 11.0)
    self.assertEqual(mse_sum_count[1], 3)

if __name__ == '__main__':
  absltest.main()
