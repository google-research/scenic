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

"""Unit tests for classification_model.py."""

from absl.testing import absltest
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models import classification_model

NUM_CLASSES = 1000
BATCH_SIZE = 4


class FakeClassificationModel(classification_model.ClassificationModel):
  """A dummy classification model for testing purposes."""

  def __init__(self):
    dataset_meta_data = {'num_classes': NUM_CLASSES, 'target_is_onehot': False}
    super().__init__(
        ml_collections.ConfigDict(),  # An empty config dict.
        dataset_meta_data)

  def build_flax_model(self):
    pass

  def default_flax_model_config(self):
    pass


def get_fake_batch_output():
  """Generates a fake `batch`.

  Returns:
    `batch`: Dictionary of None inputs and fake ground truth targets.
        outputs_noaux.pop('aux_outputs')
    `output`: Dictionary of a fake output logits.
  """
  batch = {
      'inputs': None,
      'label': jnp.array(np.random.randint(NUM_CLASSES, size=(BATCH_SIZE,))),
  }
  output = np.random.random(size=(BATCH_SIZE, NUM_CLASSES))
  return batch, output


class TestClassificationModel(absltest.TestCase):
  """Tests for the ClassificationModel."""

  def is_valid(self, t, value_name):
    """Helper function to assert that tensor `t` does not have `nan`, `inf`."""
    self.assertFalse(
        jnp.isnan(t).any(), msg=f'Found nan\'s in {t} for {value_name}')
    self.assertFalse(
        jnp.isinf(t).any(), msg=f'Found inf\'s in {t} for {value_name}')

  def test_loss_function(self):
    """Tests loss_function by checking its output's validity."""
    model = FakeClassificationModel()
    batch, output = get_fake_batch_output()
    batch_replicated, outputs_replicated = (jax_utils.replicate(batch),
                                            jax_utils.replicate(output))

    # Test loss function in the pmapped setup:
    loss_function_pmapped = jax.pmap(model.loss_function, axis_name='batch')
    total_loss = loss_function_pmapped(outputs_replicated, batch_replicated)
    # Check that loss is returning valid values:
    self.is_valid(jax_utils.unreplicate(total_loss), value_name='loss')

  def test_metric_function(self):
    """Tests metric_function by checking its output's format and validity."""
    model = FakeClassificationModel()
    batch, output = get_fake_batch_output()
    batch_replicated, outputs_replicated = (jax_utils.replicate(batch),
                                            jax_utils.replicate(output))

    # Test metric function in the pmapped setup
    metrics_fn_pmapped = jax.pmap(model.get_metrics_fn(), axis_name='batch')
    all_metrics = metrics_fn_pmapped(outputs_replicated, batch_replicated)
    # Check expected metrics exist in the output:
    expected_metrics_keys = ['accuracy', 'loss']
    self.assertSameElements(expected_metrics_keys, all_metrics.keys())

    # For each metric, check that it is a valid value.
    all_metrics = jax_utils.unreplicate(all_metrics)
    for k, v in all_metrics.items():
      self.is_valid(v[0], value_name=f'numerator of {k}')
      self.is_valid(v[1], value_name=f'denominator of {k}')


if __name__ == '__main__':
  absltest.main()
