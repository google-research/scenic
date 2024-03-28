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

"""Tests for model_utils."""

from absl.testing import parameterized

import jax.numpy as jnp
import ml_collections
import numpy as np

from scenic.projects.mtv import model_utils
import tensorflow as tf


class ModelUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mtv_cfg = ml_collections.ConfigDict({
        'init_from': {
            'restore_input_embedding': True,
            'restore_positional_embedding': True,
            'positional_embed_size_change': 'tile',
        },
        'dataset_configs': {
            'num_frames': 4,
        },
        'model':
            ml_collections.ConfigDict({
                'view_configs': [
                    ml_collections.ConfigDict({'patches': {
                        'size': (4, 4, 2),
                    }}),
                    ml_collections.ConfigDict({'patches': {
                        'size': (4, 4, 1),
                    }})
                ],
                'classifier': 'gap',
                'temporal_encoding_config': {
                    'kernel_init_method': 'central_frame_initializer',
                    'method': '3d_conv',
                }
            })
    })
    self.mtv_params = {
        'embedding_view0': {
            'kernel': np.zeros((2, 4, 4, 3, 8)),
            'bias': np.zeros((8)),
        },
        'embedding_view1': {
            'kernel': np.zeros((1, 4, 4, 3, 16)),
            'bias': np.zeros((16)),
        },
        'MultiviewEncoder': {
            'posembed_input_view0': {
                'pos_embedding': np.zeros((1, 6 * 6, 8)),
            },
            'encoderblock_0_view0': jnp.zeros((10)),
            'encoderblock_1_view0': jnp.zeros((10)),
            'posembed_input_view1': {
                'pos_embedding': np.zeros((1, 6 * 6, 16)),
            },
            'encoderblock_0_view1': jnp.zeros((10)),
            'encoderblock_1_view1': jnp.zeros((10)),
            'encoderblock_2_view1': jnp.zeros((10)),
        },
        'output_projection': jnp.zeros((10)),
    }

    self.expected_updated_mtv_params = {
        'embedding_view0': {
            'kernel': np.ones((2, 4, 4, 3, 8)),
            'bias': np.ones((8)),
        },
        'embedding_view1': {
            'kernel': np.ones((1, 4, 4, 3, 16)),
            'bias': np.ones((16)),
        },
        'MultiviewEncoder': {
            'posembed_input_view0': {
                'pos_embedding': np.ones((1, 6 * 6, 8)),
            },
            'encoderblock_0_view0': jnp.ones((10)),
            'encoderblock_1_view0': jnp.ones((10)),
            'posembed_input_view1': {
                'pos_embedding': np.ones((1, 6 * 6, 16)),
            },
            'encoderblock_0_view1': jnp.ones((10)),
            'encoderblock_1_view1': jnp.ones((10)),
            'encoderblock_2_view1': jnp.ones((10)),
        },
        'output_projection': jnp.zeros((10)),
    }
    self.restored_mtv_cfg = ml_collections.ConfigDict({
        'dataset_configs': {
            'num_frames': 4,
        },
        'model':
            ml_collections.ConfigDict({
                'view_configs': [
                    ml_collections.ConfigDict({'patches': {
                        'size': (6, 6, 4),
                    }}),
                    ml_collections.ConfigDict({'patches': {
                        'size': (6, 6, 2),
                    }})
                ],
                'classifier': 'gap',
                'temporal_encoding_config': {
                    'kernel_init_method': 'central_frame_initializer',
                    'method': '3d_conv',
                }
            })
    })
    self.restored_mtv_params = {
        'embedding_view0': {
            'kernel': np.ones((4, 6, 6, 3, 8)),
            'bias': np.ones((8)),
        },
        'embedding_view1': {
            'kernel': np.ones((2, 6, 6, 3, 16)),
            'bias': np.ones((16)),
        },
        'MultiviewEncoder': {
            'posembed_input_view0': {
                'pos_embedding': np.ones((1, 4 * 4, 8)),
            },
            'encoderblock_0_view0': jnp.ones((10)),
            'encoderblock_1_view0': jnp.ones((10)),
            'posembed_input_view1': {
                'pos_embedding': np.ones((1, 4 * 4, 16)),
            },
            'encoderblock_0_view1': jnp.ones((10)),
            'encoderblock_1_view1': jnp.ones((10)),
            'encoderblock_2_view1': jnp.ones((10)),
        },
        'output_projection': jnp.ones((10)),
    }
    self.restored_vit_params = {
        'embedding': {
            'kernel': np.ones((4, 4, 3, 8)),
            'bias': np.ones((8)),
        },
        'Transformer': {
            'posembed_input': {
                'pos_embedding': jnp.ones((1, 6 * 6, 8)),
            },
            'encoderblock_0': jnp.ones((10)),
            'encoderblock_1': jnp.ones((10)),
            'encoder_norm': jnp.ones((10)),
        },
        'output_projection': jnp.ones((10)),
    }
    self.restored_vit_cfg = ml_collections.ConfigDict(
        {'model': {
            'classifier': 'gap'
        }})
    self.expected_updated_mtv_params_from_vit = {
        'embedding_view0': {
            # Center frame initialization is used.
            'kernel':
                np.concatenate(
                    [np.zeros((1, 4, 4, 3, 8)),
                     np.ones((1, 4, 4, 3, 8))],
                    axis=0),
            'bias':
                np.ones((8)),
        },
        'embedding_view1': {
            'kernel': np.zeros((1, 4, 4, 3, 16)),
            'bias': np.zeros((16)),
        },
        'MultiviewEncoder': {
            'posembed_input_view0': {
                'pos_embedding': jnp.ones((1, 6 * 6, 8)),
            },
            'encoderblock_0_view0': jnp.ones((10)),
            'encoderblock_1_view0': jnp.ones((10)),
            'posembed_input_view1': {
                'pos_embedding': np.zeros((1, 6 * 6, 16)),
            },
            'encoderblock_0_view1': jnp.zeros((10)),
            'encoderblock_1_view1': jnp.zeros((10)),
            'encoderblock_2_view1': jnp.zeros((10)),
        },
        'output_projection': jnp.zeros((10)),
    }

  def assertDictEqualRecursive(self, actual, expected):
    self.assertEqual(type(actual), type(expected))
    if isinstance(actual, dict):
      self.assertSameElements(actual.keys(), expected.keys())
      for key, _ in expected.items():
        self.assertDictEqualRecursive(actual[key], expected[key])
    elif isinstance(actual, jnp.ndarray):
      self.assertTrue(jnp.array_equal(actual, expected))
    elif isinstance(actual, np.ndarray):
      self.assertTrue(np.allclose(actual, expected))
    else:
      self.assertEqual(actual, expected)

  def test_interpolate_cls_tokens(self):
    cls = np.zeros((1, 4, 1, 4))
    restored_cls = np.ones((1, 8, 1, 4))
    actual = model_utils.interpolate_cls_tokens(cls, restored_cls)
    self.assertAllClose(actual, np.ones((1, 4, 1, 4)))

  def test_init_bottleneck_same_dim(self):
    params = {
        'bottleneck': np.zeros((1, 4, 4, 8))
    }
    restored_bottleneck = np.ones((1, 4, 4, 8))
    model_utils.init_bottleneck(params, restored_bottleneck)
    self.assertAllClose(params['bottleneck'], restored_bottleneck)

  def test_init_bottleneck_diff_dim(self):
    params = {
        'bottleneck': np.zeros((1, 4, 4, 8))
    }
    restored_bottleneck = np.ones((1, 6, 4, 8))
    model_utils.init_bottleneck(params, restored_bottleneck)
    self.assertAllClose(params['bottleneck'], np.ones((1, 4, 4, 8)))

  def test_interpolate_input_embedding(self):
    embedding_params = {
        'kernel': np.zeros((4, 6, 6, 3, 8)),
        'bias': np.zeros((8))
    }
    restored_embedding_params = {
        'kernel': np.ones((2, 4, 4, 3, 8)),
        'bias': np.ones((8)),
    }
    model_utils.interpolate_input_embedding(embedding_params,
                                            restored_embedding_params)
    self.assertAllClose(embedding_params['kernel'], np.ones((4, 6, 6, 3, 8)))
    self.assertAllClose(embedding_params['bias'], np.ones((8)))

  def test_initialize_from_mtv_parameters_restore_output_projection(self):
    self.mtv_cfg.init_from.positional_embed_size_change = 'resize'
    model_utils.initialize_from_mtv_parameters(
        self.mtv_cfg,
        self.mtv_params,
        self.restored_mtv_cfg,
        self.restored_mtv_params,
        restore_output_projection=True)
    self.expected_updated_mtv_params['output_projection'] = jnp.ones((10))
    self.assertDictEqualRecursive(self.mtv_params,
                                  self.expected_updated_mtv_params)

  def test_initialize_from_mtv_parameters_classifier_gap(self):
    self.mtv_cfg.init_from.positional_embed_size_change = 'resize'
    model_utils.initialize_from_mtv_parameters(
        self.mtv_cfg,
        self.mtv_params,
        self.restored_mtv_cfg,
        self.restored_mtv_params,
        restore_output_projection=False)
    self.assertDictEqualRecursive(self.mtv_params,
                                  self.expected_updated_mtv_params)

  def test_initialize_from_mtv_parameters_classifier_token(self):
    self.mtv_cfg.init_from.positional_embed_size_change = 'resize'
    self.mtv_params.update({
        'cls_view0': np.zeros((1, 2, 1, 8)),
        'cls_view1': np.zeros((1, 1, 1, 16)),
    })
    self.mtv_params['MultiviewEncoder']['posembed_input_view0'][
        'pos_embedding'] = np.zeros((1, 1 + 6 * 6, 8))
    self.mtv_params['MultiviewEncoder']['posembed_input_view1'][
        'pos_embedding'] = np.zeros((1, 1 + 6 * 6, 16))
    self.restored_mtv_params.update({
        'cls_view0': np.ones((1, 4, 1, 8)),
        'cls_view1': np.ones((1, 2, 1, 16)),
    })
    self.restored_mtv_params['MultiviewEncoder']['posembed_input_view0'][
        'pos_embedding'] = np.ones((1, 1 + 4 * 4, 8))
    self.restored_mtv_params['MultiviewEncoder']['posembed_input_view1'][
        'pos_embedding'] = np.ones((1, 1 + 4 * 4, 16))
    self.mtv_cfg.model.classifier = 'token'
    self.restored_mtv_cfg.model.classifier = 'token'
    model_utils.initialize_from_mtv_parameters(
        self.mtv_cfg,
        self.mtv_params,
        self.restored_mtv_cfg,
        self.restored_mtv_params,
        restore_output_projection=False)
    self.expected_updated_mtv_params.update({
        'cls_view0': np.ones((1, 2, 1, 8)),
        'cls_view1': np.ones((1, 1, 1, 16)),
    })
    self.expected_updated_mtv_params['MultiviewEncoder'][
        'posembed_input_view0']['pos_embedding'] = jnp.ones((1, 1 + 6 * 6, 8))
    self.expected_updated_mtv_params['MultiviewEncoder'][
        'posembed_input_view1']['pos_embedding'] = jnp.ones((1, 1 + 6 * 6, 16))
    self.assertDictEqualRecursive(self.mtv_params,
                                  self.expected_updated_mtv_params)

  @parameterized.parameters('gap', 'token')
  def test_initialize_one_view_from_vit_parameters_classifier_gap(
      self, restored_model_classifier):
    self.restored_vit_cfg.model.classifier = restored_model_classifier
    if restored_model_classifier == 'token':
      self.restored_vit_params['Transformer']['posembed_input'][
          'pos_embedding'] = jnp.ones((1, 6 * 6 + 1, 8))
    model_utils.initialize_one_view_from_vit_parameters(
        self.mtv_cfg,
        self.mtv_params,
        self.restored_vit_cfg,
        self.restored_vit_params,
        view_idx=0)
    self.assertDictEqualRecursive(self.mtv_params,
                                  self.expected_updated_mtv_params_from_vit)

  def test_initialize_one_view_from_vit_parameters_cross_attention(self):
    self.mtv_params['MultiviewEncoder'].pop('encoderblock_1_view0')
    self.mtv_params['MultiviewEncoder'].pop('encoderblock_1_view1')
    self.mtv_params['MultiviewEncoder']['cross_view_encoderblock_1'] = {
        'msa_ln_view0': jnp.zeros((10)),
        'msa_ln_view1': jnp.zeros((10)),
        'msa_view0': jnp.zeros((10)),
        'msa_view1': jnp.zeros((10)),
        'cross_attention_ln_view0': jnp.zeros((10)),
        'cross_attention_ln_view1': jnp.zeros((10)),
        'cross_attention_view0_1': jnp.zeros((10)),
        'mlp_ln_view0': jnp.zeros((10)),
        'mlp_ln_view1': jnp.zeros((10)),
        'mlp_view0': jnp.zeros((10)),
        'mlp_view1': jnp.zeros((10)),
    }
    self.restored_vit_params['Transformer']['encoderblock_1'] = {
        'LayerNorm_0': jnp.ones((10)),
        'LayerNorm_1': jnp.ones((10)),
        'MultiHeadDotProductAttention_0': jnp.ones((10)),
        'MlpBlock_0': jnp.ones((10)),
    }
    self.expected_updated_mtv_params_from_vit['MultiviewEncoder'].pop(
        'encoderblock_1_view0')
    self.expected_updated_mtv_params_from_vit['MultiviewEncoder'].pop(
        'encoderblock_1_view1')
    self.expected_updated_mtv_params_from_vit['MultiviewEncoder'][
        'cross_view_encoderblock_1'] = {
            'msa_ln_view0': jnp.ones((10)),
            'msa_ln_view1': jnp.zeros((10)),
            'msa_view0': jnp.ones((10)),
            'msa_view1': jnp.zeros((10)),
            'cross_attention_ln_view0': jnp.zeros((10)),
            'cross_attention_ln_view1': jnp.zeros((10)),
            'cross_attention_view0_1': jnp.zeros((10)),
            'mlp_ln_view0': jnp.ones((10)),
            'mlp_ln_view1': jnp.zeros((10)),
            'mlp_view0': jnp.ones((10)),
            'mlp_view1': jnp.zeros((10)),
        }
    model_utils.initialize_one_view_from_vit_parameters(
        self.mtv_cfg,
        self.mtv_params,
        self.restored_vit_cfg,
        self.restored_vit_params,
        view_idx=0)
    self.assertDictEqualRecursive(self.mtv_params,
                                  self.expected_updated_mtv_params_from_vit)

  @parameterized.parameters('gap', 'token')
  def test_initialize_one_view_from_vit_parameters_classifier_token(
      self, restored_model_classifier):
    self.mtv_cfg.model.classifier = 'token'
    self.mtv_params.update({
        'cls_view0': np.zeros((1, 2, 1, 8)),
        'cls_view1': np.zeros((1, 1, 1, 16)),
    })
    self.mtv_params['MultiviewEncoder']['posembed_input_view0'][
        'pos_embedding'] = np.zeros((1, 1 + 6 * 6, 8))
    self.mtv_params['MultiviewEncoder']['posembed_input_view1'][
        'pos_embedding'] = np.zeros((1, 1 + 6 * 6, 8))
    self.restored_vit_cfg.model.classifier = restored_model_classifier
    if restored_model_classifier == 'token':
      self.restored_vit_params.update({'cls': np.ones((1, 1, 8))})
      self.restored_vit_params['Transformer']['posembed_input'][
          'pos_embedding'] = jnp.ones((1, 1 + 6 * 6, 8))
      self.expected_updated_mtv_params_from_vit['MultiviewEncoder'][
          'posembed_input_view0']['pos_embedding'] = jnp.ones((1, 1 + 6 * 6, 8))
    else:
      self.expected_updated_mtv_params_from_vit['MultiviewEncoder'][
          'posembed_input_view0']['pos_embedding'] = jnp.concatenate([
              jnp.zeros((1, 1, 8)),
              jnp.ones((1, 6 * 6, 8)),
          ],
                                                                     axis=1)
    self.expected_updated_mtv_params_from_vit['MultiviewEncoder'][
        'posembed_input_view1']['pos_embedding'] = np.zeros((1, 1 + 6 * 6, 8))
    if restored_model_classifier == 'token':
      self.expected_updated_mtv_params_from_vit.update({
          'cls_view0': jnp.ones((1, 2, 1, 8)),
          'cls_view1': np.zeros((1, 1, 1, 16)),
      })
    else:
      self.expected_updated_mtv_params_from_vit.update({
          'cls_view0': np.zeros((1, 2, 1, 8)),
          'cls_view1': np.zeros((1, 1, 1, 16)),
      })
    model_utils.initialize_one_view_from_vit_parameters(
        self.mtv_cfg,
        self.mtv_params,
        self.restored_vit_cfg,
        self.restored_vit_params,
        view_idx=0)
    self.assertDictEqualRecursive(self.mtv_params,
                                  self.expected_updated_mtv_params_from_vit)


if __name__ == '__main__':
  tf.test.main()
