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

"""Unit tests for base_model.py."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
from scenic.projects.av_mae import base_model


class FakeModel(base_model.MaskedFeatureRegressionModel):
  """A dummy model for testing purposes."""

  def __init__(self, loss_unmasked_tokens):
    dataset_meta_data = {}
    config = ml_collections.ConfigDict({
        'masked_feature_loss': {
            'target': 'rgb',
            'loss_unmasked_tokens': loss_unmasked_tokens
        },
        'model': {'patches': {'size': (2, 2)}}
    })
    super().__init__(config, dataset_meta_data)

  def build_flax_model(self):
    pass

  def default_flax_model_config(self):
    pass


def get_fake_batch_and_predictions():
  """Generates a fake `batch`."""
  # Predictions and targets have shape [batch, num_tokens, channels], and here
  # we set channels=1.
  batch, height, width, channels = 2, 4, 4, 1
  inputs = jnp.arange(1, 33).reshape(batch, height, width, channels)
  predictions = jnp.array([
      [
          [1.0, 3.0, 5.0, 6.0],
          [3.0, 5.0, 11.0, 10.0],
          [9.0, 10.0, 11.0, 12.0],
          [14.0, 13.0, 14.0, 17.0],
      ],
      [
          [17.0, 18.0, 21.0, 22.0],
          [20.0, 19.0, 24.0, 25.0],
          [27.0, 29.0, 30.0, 32.0],
          [27.0, 28.0, 33.0, 32.0],
      ],
  ])
  prediction_masks = jnp.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

  fake_batch = {
      'inputs': inputs,
      'target_rgb': base_model.get_rgb_targets(inputs, (2, 2))
  }
  return fake_batch, predictions, prediction_masks


class TestMaskedFeatureRegressionModel(parameterized.TestCase):
  """Tests for a fake feature regression regression model."""

  @parameterized.named_parameters(
      ('loss_masked_tokens', False),
      ('loss_all_tokens', True),
  )
  def test_loss_function(self, loss_unmasked_tokens):
    """Tests loss_function by checking its output's validity."""
    model = FakeModel(loss_unmasked_tokens=loss_unmasked_tokens)
    batch, predictions, prediction_masks = get_fake_batch_and_predictions()
    batch_replicated, predictions_replicated, prediction_masks_replicated = (
        jax_utils.replicate(batch), jax_utils.replicate(predictions),
        jax_utils.replicate(prediction_masks))

    # Test loss function in the pmapped setup:
    loss_function_pmapped = jax.pmap(model.loss_function, axis_name='batch')
    total_loss = loss_function_pmapped(
        predictions_replicated, prediction_masks_replicated, batch_replicated)
    total_loss = jax_utils.unreplicate(total_loss)
    if loss_unmasked_tokens:
      expected_loss = jnp.mean(jnp.array(
          [1.0, 21.0, 8.0, 12.0, 0.0, 4.0, 18.0, 4.0]))
    else:
      expected_loss = jnp.mean(jnp.array([1.0, 21.0, 18.0]))
    self.assertAlmostEqual(total_loss, expected_loss, delta=1e-6)

  def test_rgb_image_targets(self):
    """Test shapes of generating rgb targets for images."""
    test_image = jnp.arange(1, 65).reshape(8, 8).astype(jnp.float32)
    batch = jnp.expand_dims(test_image, (0, -1))
    targets = base_model.get_rgb_targets(batch, patch_size=(2, 2))

    expected_shape = (1, (8 // 2) * (8 // 2), 2 * 2)
    self.assertEqual(targets.shape, expected_shape)

  @parameterized.named_parameters(
      ('select_central_frame_False', False),
      ('select_central_frame_True', True),
  )
  def test_rgb_video_targets(self, select_central_frame):
    """Test shapes of generating rgb targets for video."""
    test_video = jnp.arange(768).reshape(4, 8, 8, 3).astype(jnp.float32)
    batch = jnp.expand_dims(test_video, 0)
    targets = base_model.get_rgb_targets(
        batch, patch_size=(4, 4, 2), select_central_frame=select_central_frame)

    if select_central_frame:
      expected_shape = (1, (8 // 4) * (8 // 4) * (4 // 2), 4 * 4 * 3)
    else:
      expected_shape = (1, (8 // 4) * (8 // 4) * (4 // 2), 4 * 4 * 2 * 3)
    self.assertEqual(targets.shape, expected_shape)


if __name__ == '__main__':
  absltest.main()
