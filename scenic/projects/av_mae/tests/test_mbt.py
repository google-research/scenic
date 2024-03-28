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

"""Unit tests for mbt.py."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import chex
from jax import random
import jax.numpy as jnp
import ml_collections

from scenic.projects.av_mae import mbt
from scenic.projects.av_mae import trainer_multimodal


def get_config():
  return ml_collections.ConfigDict({
      'model':
          dict(
              modality_fusion=('rgb', 'spectrogram'),
              fusion_layer=2,
              share_encoder=False,
              use_bottleneck=True,
              n_bottlenecks=4,
              attention_config=dict(type='spacetime'),
              temporal_encoding_config=dict(method='3d_conv'),
              num_heads=2,
              num_layers=4,
              mlp_dim=96,
              hidden_size=24,
              patches={'size': (4, 4, 2)},
              classifier='gap',
              data_dtype_str='float32',
              dropout_rate=0.0,
              attention_dropout_rate=0.0,
              return_preclassifier=False,
              representation_size=None),
  })


class TestMBT(parameterized.TestCase):
  """Tests for MBT."""

  @parameterized.named_parameters(
      ('multilabel_noprecls', mbt.MBTMultilabelClassificationModel, False),
      ('classification_precls', mbt.MBTClassificationModel, True),
  )
  def test_shapes(self, model_class, return_preclassifier):
    """Tests the output shapes of the ViViT model are correct."""

    # Get random input data
    rng = random.PRNGKey(0)
    unused_batch, time, height, width, channels = 2, 16, 32, 32, 3
    batch, height_spec, width_spec, channels_spec = 2, 12, 8, 1
    num_classes = 5
    dataset_meta_data = {
        'input_shape': {
            'rgb': (-1, time, height, width, channels),
            'spectrogram': (-1, height_spec, width_spec, channels_spec)
        },
        'input_dtype': {
            'rgb': jnp.float32,
            'spectrogram': jnp.float32
        },

        'num_classes': num_classes
    }

    config = get_config()
    config.model.return_preclassifier = return_preclassifier
    config.batch_size = batch

    # Initialise the model
    init_rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    scenic_model = model_class(config, dataset_meta_data)

    input_spec_dict = {}
    for key in dataset_meta_data['input_shape']:
      input_spec = (dataset_meta_data['input_shape'][key],
                    dataset_meta_data['input_dtype'][key])
      input_spec_dict[key] = input_spec
      logging.info('input_spec: %s', input_spec_dict)

    params, _, _, _ = trainer_multimodal.initialize_model(
        model_def=scenic_model.flax_model,
        input_spec_dict=input_spec_dict,
        config=config,
        rngs=init_rngs,
        is_train=True)

    # Check output shapes match.
    for train in (False, True):
      logging.info('Running for mode train = %s', train)

      # ATTENTION: MBT mutates the input dictionary. Therefore, we need to
      # recreate the inputs each time in this loop.
      inputs = {
          'rgb': random.normal(
              rng, shape=(batch, time, height, width, channels)),
          'spectrogram': random.normal(
              rng, shape=(batch, height_spec, width_spec, channels_spec))
      }

      output = scenic_model.flax_model.apply(
          {'params': params}, inputs, train=train, mutable=False,
          rngs=init_rngs)

      # The default patch size is 4x4x2.
      if return_preclassifier:
        for modality, tensor in output.items():
          logging.info('modality: %s, output shape: %s', modality, tensor.shape)
          if modality == 'rgb':
            num_tokens = height * width * time / (4 * 4 * 2)
          elif modality == 'spectrogram':
            num_tokens = height_spec * width_spec / (4 * 4)
          else:
            raise ValueError('Unknown modality ' + modality)
          expected_output_shape = (batch, num_tokens, config.model.hidden_size)
          self.assertEqual(tensor.shape, expected_output_shape)
      else:
        expected_output_shape = (batch, num_classes)
        if train:  # Then MBT returns a dictionary of each modality.
          for tensor in output.values():
            self.assertEqual(tensor.shape, expected_output_shape)
        else:
          self.assertEqual(output.shape, expected_output_shape)

      chex.assert_tree_all_finite(output)
      chex.assert_tree_all_finite(params)

if __name__ == '__main__':
  absltest.main()
