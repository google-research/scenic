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

"""Unit tests for vit.py."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from jax import random
import jax.numpy as jnp
import ml_collections

from scenic.projects.av_mae import vit
from scenic.train_lib_deprecated import train_utils


class TestViT(parameterized.TestCase):
  """Tests for ViT with token masking."""

  @parameterized.named_parameters(
      ('mae_model', vit.ViTMaskedAutoencoderModel)
  )
  def test_shapes(self, model_class):
    """Tests the output shapes of the ViT model are correct."""

    # Get random input data
    rng = random.PRNGKey(0)
    batch, height, width, channels = 4, 32, 32, 3
    input_shape = (batch, height, width, channels)
    inputs = random.normal(rng, shape=input_shape)
    dataset_meta_data = {'input_shape': (-1, height, width, channels)}

    # Initialise the model
    init_rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    config = None  # Will use the default config.
    scenic_model = model_class(config, dataset_meta_data)
    scenic_model.config.batch_size = batch

    params, _, _, _ = train_utils.initialize_model(
        model_def=scenic_model.flax_model,
        input_spec=[(dataset_meta_data['input_shape'],
                     dataset_meta_data.get('input_dtype', jnp.float32))],
        config=scenic_model.config,
        rngs=init_rngs,
        train=True)

    # Check output shapes match.
    for train in (True, False):
      output, token_mask = scenic_model.flax_model.apply(
          {'params': params}, inputs, train=train, mutable=False,
          rngs=init_rngs)

      # The default patch size is 4x4.
      expected_output_shape = [batch, int(height * width / (4 * 4)), 4 * 4 * 3]
      if model_class == vit.ViTMaskedAutoencoderModel and not train:
        # For MAE, in test mode, we return the features output by the encoder.
        expected_output_shape = [batch, int(height * width / (4 * 4)),
                                 scenic_model.config.model.hidden_size]
        if scenic_model.config.model.classifier == 'token':
          expected_output_shape[1] += 1

      expected_mask_shape = (batch, height * width / (4 * 4))
      self.assertEqual(output.shape, tuple(expected_output_shape))
      self.assertEqual(token_mask.shape, expected_mask_shape)
      chex.assert_tree_all_finite(output)
      chex.assert_tree_all_finite(params)

      if not train:
        self.assertEqual(jnp.sum(token_mask), 0)
      else:
        token_mask_sum = jnp.sum(token_mask)
        mask_prob = scenic_model.config.masked_feature_loss.token_mask_probability  # pylint: disable=line-too-long
        patch_h, patch_w = scenic_model.config.model.patches.size
        expected_sum = (
            mask_prob * (batch * height * width) / (patch_h * patch_w))

        delta = 0.05 * expected_sum
        self.assertAlmostEqual(token_mask_sum, expected_sum, delta=delta)


class TestViTMaeFinetuning(parameterized.TestCase):
  """Tests for ViT for finetuning MAE pretrained models."""

  @parameterized.named_parameters(
      ('learned_false_cls', 'learned_1d', False, 'token',
       vit.ViTMAEMultilabelFinetuning),
      ('learned_false_gap', 'learned_1d', True, 'gap',
       vit.ViTMAEClassificationFinetuning),
      ('sin_1d_True', 'sinusoidal_1d', True, 'token',
       vit.ViTMAEMultilabelFinetuning),
      ('sin_2d_True', 'sinusoidal_2d', True, 'gap',
       vit.ViTMAEClassificationFinetuning),
  )
  def test_shapes(self, positional_embedding, freeze_backbone,
                  classifier, model_class):
    """Tests the output shapes of the ViT model are correct."""

    # Get random input data
    rng = random.PRNGKey(0)
    batch, height, width, channels = 4, 32, 32, 3
    num_classes = 5
    input_shape = (batch, height, width, channels)
    inputs = random.normal(rng, shape=input_shape)
    dataset_meta_data = {
        'input_shape': (-1, height, width, channels),
        'num_classes': num_classes
    }

    config = ml_collections.ConfigDict({
        'batch_size': batch,
        'model':
            dict(
                num_heads=2,
                num_layers=1,
                representation_size=None,
                mlp_dim=64,
                dropout_rate=0.,
                attention_dropout_rate=0.,
                positional_embedding=positional_embedding,
                hidden_size=16,
                patches={'size': (4, 4)},
                classifier=classifier,
                data_dtype_str='float32',
                freeze_backbone=freeze_backbone),
    })

    # Initialise the model
    init_rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    scenic_model = model_class(config, dataset_meta_data)

    params, model_state, _, _ = train_utils.initialize_model(
        model_def=scenic_model.flax_model,
        input_spec=[(dataset_meta_data['input_shape'],
                     dataset_meta_data.get('input_dtype', jnp.float32))],
        config=scenic_model.config,
        rngs=init_rngs,
        train=True)

    # Check output shapes match.
    for train in (True, False):
      if train:
        output, unused_model_state = scenic_model.flax_model.apply(
            {'params': params, **model_state}, inputs, train=train,
            mutable=['batch_stats'],
            rngs=init_rngs)
      else:
        output = scenic_model.flax_model.apply(
            {'params': params, **model_state}, inputs, train=train,
            mutable=False, rngs=init_rngs)

      # The default patch size is 4x4.
      expected_output_shape = (batch, num_classes)
      self.assertEqual(output.shape, expected_output_shape)
      chex.assert_tree_all_finite(output)
      chex.assert_tree_all_finite(params)


if __name__ == '__main__':
  absltest.main()
