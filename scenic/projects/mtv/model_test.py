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

"""Tests for model."""

from absl.testing import parameterized

import flax
from jax import random
import jax.numpy as jnp
import ml_collections
from scenic.projects.mtv import model
import tensorflow as tf


class ModelTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.view_configs = [
        ml_collections.ConfigDict({
            'hidden_size': 8,
            'patches': {
                'size': (4, 4, 4)
            },
            'num_heads': 2,
            'mlp_dim': 16,
            'num_layers': 1,
        }),
        ml_collections.ConfigDict({
            'hidden_size': 16,
            'patches': {
                'size': (4, 4, 2)
            },
            'num_heads': 2,
            'mlp_dim': 32,
            'num_layers': 1,
        }),
    ]
    self.cross_view_fusion = None
    self.temporal_encoding_config = ml_collections.ConfigDict({
        'method': '3d_conv',
        'kernel_init_method': 'central_frame_initializer',
    })
    self.global_encoder_config = ml_collections.ConfigDict({
        'num_heads': 2,
        'mlp_dim': 32,
        'num_layers': 1,
        'hidden_size': 16,
        'merge_axis': 'channel',
    })

  def test_cross_view_attention_encoder_block_same_layers(self):
    self.cross_view_fusion = ml_collections.ConfigDict({
        'use_query_config': True,
        'fusion_layers': (0,),
    })
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    tokens = [jnp.ones((2, 8, 8)), jnp.ones((2, 16, 16))]
    outputs, vars_dict = model.CrossViewAttentionEncoderBlock(
        view_configs=self.view_configs,
        cross_view_fusion=self.cross_view_fusion).init_with_output(
            rngs, tokens, 0, deterministic=True)
    self.assertEqual(outputs[0].shape, (2, 8, 8))
    self.assertEqual(outputs[1].shape, (2, 16, 16))
    expected_keys = {
        'msa_ln_view0',
        'msa_ln_view1',
        'msa_view0',
        'msa_view1',
        'cross_attention_ln_view0',
        'cross_attention_ln_view1',
        'cross_attention_view0_1',
        'mlp_ln_view0',
        'mlp_ln_view1',
        'mlp_view0',
        'mlp_view1',
    }
    self.assertSetEqual(set(vars_dict['params'].keys()), expected_keys)

  def test_cross_view_attention_encoder_block_different_layers(self):
    self.view_configs[1].num_layers = 2
    self.cross_view_fusion = ml_collections.ConfigDict({
        'use_query_config': True,
        'fusion_layers': (0, 1),
    })
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    tokens = [jnp.ones((2, 8, 8)), jnp.ones((2, 16, 16))]
    outputs, vars_dict = model.CrossViewAttentionEncoderBlock(
        view_configs=self.view_configs,
        cross_view_fusion=self.cross_view_fusion).init_with_output(
            rngs, tokens, 1, deterministic=True)
    self.assertEqual(outputs[0].shape, (2, 8, 8))
    self.assertEqual(outputs[1].shape, (2, 16, 16))
    expected_keys = {
        'msa_ln_view1',
        'msa_view1',
        'cross_attention_ln_view0',
        'cross_attention_ln_view1',
        'cross_attention_view0_1',
        'mlp_ln_view1',
        'mlp_view1',
    }
    self.assertSetEqual(set(vars_dict['params'].keys()), expected_keys)

  def test_multiview_encoder_wo_cross_view_fusion(self):
    rng = random.PRNGKey(0)
    inputs = [jnp.ones((2, 4, 8)), jnp.ones((2, 8, 16))]
    temporal_dims = [2, 4]
    outputs, vars_dict = model.MultiviewEncoder(
        self.view_configs, self.cross_view_fusion,
        temporal_dims).init_with_output(rng, inputs, temporal_dims, None)
    self.assertEqual(outputs[0].shape, (2, 4, 8))
    self.assertEqual(outputs[1].shape, (2, 8, 16))
    self.assertSetEqual(
        set(vars_dict['params'].keys()), {
            'posembed_input_view0',
            'posembed_input_view1',
            'encoderblock_0_view0',
            'encoderblock_0_view1',
        })

  @parameterized.parameters(True, False)
  def test_multiview_encoder_w_cross_view_attention(self,
                                                    fuse_in_descending_order):
    self.cross_view_fusion = ml_collections.ConfigDict({
        'type': 'cross_view_attention',
        'use_query_config': True,
        'fusion_layers': (0,),
        'fuse_in_descending_order': fuse_in_descending_order,
    })
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    inputs = [jnp.ones((2, 4, 8)), jnp.ones((2, 8, 16))]
    temporal_dims = [2, 4]
    outputs, vars_dict = model.MultiviewEncoder(self.view_configs,
                                                self.cross_view_fusion,
                                                temporal_dims).init_with_output(
                                                    rngs, inputs, None)
    self.assertEqual(outputs[0].shape, (2, 4, 8))
    self.assertEqual(outputs[1].shape, (2, 8, 16))
    self.assertSetEqual(
        set(vars_dict['params'].keys()), {
            'posembed_input_view0',
            'posembed_input_view1',
            'cross_view_encoderblock_0',
        })

  @parameterized.parameters(
      (True, 16, {'bottleneck_linear_0_view0'}),
      (False, 8, {'bottleneck_linear_0_view1'}),
  )
  def test_multiview_encoder_w_bottleneck(self, fuse_in_descending_order,
                                          bottleneck_channels,
                                          expected_bottleneck_key):
    self.cross_view_fusion = ml_collections.ConfigDict({
        'type': 'bottleneck',
        'bottleneck_tokens': 4,
        'fusion_layers': (0,),
        'fuse_in_descending_order': fuse_in_descending_order,
    })
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    inputs = [jnp.ones((2, 4, 8)), jnp.ones((2, 8, 16))]
    bottleneck = jnp.ones((2, 4, bottleneck_channels))
    temporal_dims = [2, 4]
    outputs, vars_dict = model.MultiviewEncoder(self.view_configs,
                                                self.cross_view_fusion,
                                                temporal_dims).init_with_output(
                                                    rngs, inputs, bottleneck)
    self.assertEqual(outputs[0].shape, (2, 4, 8))
    self.assertEqual(outputs[1].shape, (2, 8, 16))
    self.assertSetEqual(
        set(vars_dict['params'].keys()), {
            'posembed_input_view0',
            'posembed_input_view1',
            'encoderblock_0_view0',
            'encoderblock_0_view1',
        }.union(expected_bottleneck_key))

  @parameterized.parameters(
      ('time', 'token'),
      ('time', 'gap'),
      ('channel', 'gap'),
      ('channel', 'gap'),
  )
  def test_mtv_without_cross_fusion(self, merge_axis, classifier):
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    images = jnp.ones((2, 4, 8, 8, 3))
    self.global_encoder_config.merge_axis = merge_axis
    self.global_encoder_config.hidden_size = 16
    outputs, vars_dict = model.MTV(
        view_configs=self.view_configs,
        cross_view_fusion=self.cross_view_fusion,
        temporal_encoding_config=self.temporal_encoding_config,
        global_encoder_config=self.global_encoder_config,
        input_token_temporal_dims=[1, 2],
        num_classes=10,
        classifier=classifier,
    ).init_with_output(rngs, images)
    self.assertEqual(outputs.shape, (2, 10))
    actual_var_keys = set(vars_dict['params'].keys())
    expected_keys = {
        'embedding_view0',
        'embedding_view1',
        'MultiviewEncoder',
        'global_encoder',
        'output_projection',
    }
    if classifier == 'token':
      expected_keys.update({
          'cls_view0',
          'cls_view1',
          'cls_global',
      })
    if merge_axis == 'time':
      expected_keys.update({
          'global_encoder_linear_view0',
          'global_encoder_linear_view1',
      })
    self.assertSetEqual(actual_var_keys, expected_keys)

  @parameterized.parameters(
      ('pre_logits', (2, 2, 2, 2, 24)),
      ('logits', (2, 10)),
  )
  def test_mtv_keep_spatiotemporal_features(self, final_endpoint,
                                            expected_output_shape):
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    images = jnp.ones((2, 4, 8, 8, 3))
    outputs, _ = model.MTV(
        view_configs=self.view_configs,
        cross_view_fusion=self.cross_view_fusion,
        temporal_encoding_config=self.temporal_encoding_config,
        global_encoder_config=self.global_encoder_config,
        input_token_temporal_dims=[1, 2],
        num_classes=10,
        classifier='gap',
        keep_spatiotemporal_features=True,
        final_endpoint=final_endpoint,
    ).init_with_output(rngs, images)
    self.assertEqual(outputs.shape, expected_output_shape)

  @parameterized.parameters(
      (True, 'bottleneck_linear_0_view0'),
      (False, 'bottleneck_linear_0_view1'),
  )
  def test_mtv_with_bottleneck(self, fuse_in_descending_order,
                               expected_bottleneck_key):
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    images = jnp.ones((2, 4, 8, 8, 3))
    self.cross_view_fusion = ml_collections.ConfigDict({
        'type': 'bottleneck',
        'bottleneck_tokens': 4,
        'fusion_layers': (0,),
        'fuse_in_descending_order': fuse_in_descending_order,
    })
    outputs, vars_dict = model.MTV(
        view_configs=self.view_configs,
        cross_view_fusion=self.cross_view_fusion,
        temporal_encoding_config=self.temporal_encoding_config,
        global_encoder_config=self.global_encoder_config,
        input_token_temporal_dims=[1, 2],
        num_classes=10,
        classifier='token',
    ).init_with_output(rngs, images)
    self.assertEqual(outputs.shape, (2, 10))
    actual_var_keys = set(vars_dict['params'].keys())
    self.assertIn(expected_bottleneck_key,
                  vars_dict['params']['MultiviewEncoder'].keys())
    expected_keys = {
        'cls_view0',
        'cls_view1',
        'cls_global',
        'embedding_view0',
        'embedding_view1',
        'MultiviewEncoder',
        'bottleneck',
        'global_encoder',
        'output_projection',
    }
    self.assertSetEqual(actual_var_keys, expected_keys)

  @parameterized.parameters(True, False)
  def test_mtv_with_cross_view_attention(self, fuse_in_descending_order):
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    images = jnp.ones((2, 4, 8, 8, 3))
    self.cross_view_fusion = ml_collections.ConfigDict({
        'type': 'cross_view_attention',
        'use_query_config': True,
        'add_mlp': True,
        'fusion_layers': (0,),
        'fuse_in_descending_order': fuse_in_descending_order,
    })
    outputs, vars_dict = model.MTV(
        view_configs=self.view_configs,
        cross_view_fusion=self.cross_view_fusion,
        temporal_encoding_config=self.temporal_encoding_config,
        global_encoder_config=self.global_encoder_config,
        input_token_temporal_dims=[1, 2],
        num_classes=10,
        classifier='token',
    ).init_with_output(rngs, images)
    self.assertEqual(outputs.shape, (2, 10))
    actual_var_keys = set(vars_dict['params'].keys())
    self.assertIn('cross_view_encoderblock_0',
                  vars_dict['params']['MultiviewEncoder'].keys())
    expected_keys = {
        'cls_view0',
        'cls_view1',
        'cls_global',
        'embedding_view0',
        'embedding_view1',
        'MultiviewEncoder',
        'global_encoder',
        'output_projection',
    }
    self.assertSetEqual(actual_var_keys, expected_keys)

  def test_mtv_classification_model(self):
    rng = random.PRNGKey(0)
    mtv = model.MTVClassificationModel(
        config=None,
        dataset_meta_data={
            'num_classes': 10,
            'target_is_onehot': False,
        })
    num_frames = 8  # matches with the default config.
    inputs = jnp.ones((2, num_frames, 8, 8, 3))
    rng, init_rng = random.split(rng)
    init_model_state, init_params = flax.core.pop(mtv.flax_model.init(
        init_rng, inputs, train=False), 'params')

    # Check that the forward pass works with mutated model_state.
    rng, dropout_rng = random.split(rng)
    variables = {'params': init_params, **init_model_state}
    outputs = mtv.flax_model.apply(
        variables,
        inputs,
        train=True,
        rngs={'dropout': dropout_rng})
    self.assertEqual(outputs.shape, (2, 10))

  def test_mtv_multihead_classification_model(self):
    rng = random.PRNGKey(0)
    config = model._DEFAULT_MTV_CONFIG
    config.dataset_configs.class_splits = [7, 3]
    mtv = model.MTVMultiheadClassificationModel(
        config=config,
        dataset_meta_data={
            'num_classes': 10,
            'target_is_onehot': False,
        })
    num_frames = 8  # matches with the default config.
    inputs = jnp.ones((2, num_frames, 8, 8, 3))
    rng, init_rng = random.split(rng)
    init_model_state, init_params = flax.core.pop(mtv.flax_model.init(
        init_rng, inputs, train=False), 'params')

    # Check that the forward pass works with mutated model_state.
    rng, dropout_rng = random.split(rng)
    variables = {'params': init_params, **init_model_state}
    outputs = mtv.flax_model.apply(
        variables,
        inputs,
        train=True,
        rngs={'dropout': dropout_rng})
    self.assertEqual(outputs.shape, (2, 10))


if __name__ == '__main__':
  tf.test.main()
