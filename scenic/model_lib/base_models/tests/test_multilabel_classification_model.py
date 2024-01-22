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

"""Unit tests for multilabel_classification_model.py."""

from absl.testing import absltest
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models import multilabel_classification_model

NUM_CLASSES = 1000
BATCH_SIZE = 4


class FakeMultiLabelClassificationModel(
    multilabel_classification_model.MultiLabelClassificationModel):
  """A dummy multi-label classification model for testing purposes."""

  def __init__(self):
    dataset_meta_data = {'num_classes': NUM_CLASSES, 'target_is_onehot': True}
    super().__init__(
        ml_collections.ConfigDict(),  # An empty config dict.
        dataset_meta_data)

  def build_flax_model(self):
    pass

  def default_flax_model_config(self):
    pass


def get_fake_batch_output(array_size=(BATCH_SIZE, NUM_CLASSES)):
  """Generates a fake `batch`.

  Args:
    array_size: size of the label and output array.

  Returns:
    `batch`: Dictionary of None inputs and fake ground truth targets.
        outputs_noaux.pop('aux_outputs')
    `output`: Dictionary of a fake output logits.
  """
  batch = {
      'inputs': None,
      'label': jnp.array(np.random.randint(2, size=array_size)),
  }
  output = jnp.array(np.random.random(size=array_size))
  return batch, output


class TestMultiLabelClassificationModel(absltest.TestCase):
  """Tests for the MultiLabelClassificationModel."""

  def is_valid(self, t, value_name):
    """Helper function to assert that tensor `t` does not have `nan`, `inf`."""
    self.assertFalse(
        jnp.isnan(t).any(), msg=f'Found nan\'s in {t} for {value_name}')
    self.assertFalse(
        jnp.isinf(t).any(), msg=f'Found inf\'s in {t} for {value_name}')

  def test_loss_function(self):
    """Tests loss_function by checking its output's validity."""
    model = FakeMultiLabelClassificationModel()
    batch, output = get_fake_batch_output()
    batch_replicated, outputs_replicated = (jax_utils.replicate(batch),
                                            jax_utils.replicate(output))

    # Test loss function in the pmapped setup:
    loss_function_pmapped = jax.pmap(model.loss_function, axis_name='batch')
    total_loss = loss_function_pmapped(outputs_replicated, batch_replicated)
    # Check that loss is returning valid values:
    self.is_valid(jax_utils.unreplicate(total_loss), value_name='loss')

  def test_loss_function_masked(self):
    """Tests a masked loss_function by comparing different canonical masks."""
    array_size = (BATCH_SIZE, 50, NUM_CLASSES)

    model = FakeMultiLabelClassificationModel()
    batch, output = get_fake_batch_output(
        array_size=array_size)

    # Unmasked loss
    loss_value_unmasked = model.loss_function(output, batch)

    # Mask with only ones (so effectively no mask).
    batch['batch_mask'] = jnp.ones((BATCH_SIZE, 50))
    loss_value_masked = model.loss_function(output, batch)

    self.assertAlmostEqual(
        float(loss_value_unmasked),
        float(loss_value_masked))

    # Extend the batch with random outputs and labels, but mask them with 0's.
    batch_extended = {
        'label': jnp.concatenate(
            (batch['label'], np.random.randint(2, size=array_size)), axis=1),
        'batch_mask': jnp.concatenate(
            (batch['batch_mask'], np.zeros((BATCH_SIZE, 50))), axis=1),
    }
    output_extended = jnp.concatenate(
        (output, np.random.random(size=array_size)), axis=1)
    loss_value_extended = model.loss_function(output_extended, batch_extended)

    # Test with `places=3` due to JAX issue: github.com/google/jax/issues/6553
    # TODO(robromijnders): follow up with JAX issue and remove `places=3`.
    self.assertAlmostEqual(
        float(loss_value_masked),
        float(loss_value_extended),
        places=3)

  def test_metric_function(self):
    """Tests metric_function by checking its output's format and validity."""
    model = FakeMultiLabelClassificationModel()
    batch, output = get_fake_batch_output()
    batch_replicated, outputs_replicated = (jax_utils.replicate(batch),
                                            jax_utils.replicate(output))

    # Test metric function in the pmapped setup
    metrics_fn_pmapped = jax.pmap(model.get_metrics_fn(), axis_name='batch')
    all_metrics = metrics_fn_pmapped(outputs_replicated, batch_replicated)
    # Check expected metrics exist in the output:
    expected_metrics_keys = ['prec@1', 'loss']
    self.assertSameElements(expected_metrics_keys, all_metrics.keys())

    # For each metric, check that it is a valid value.
    all_metrics = jax_utils.unreplicate(all_metrics)
    for k, v in all_metrics.items():
      self.is_valid(v[0], value_name=f'numerator of {k}')
      self.is_valid(v[1], value_name=f'denominator of {k}')


if __name__ == '__main__':
  absltest.main()
