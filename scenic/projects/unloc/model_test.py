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

import copy
from absl.testing import parameterized
import flax
from jax import random
import jax.numpy as jnp
import ml_collections
from scenic.projects.unloc import model
import tensorflow as tf


class ModelTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.images = jnp.ones((2, 4, 8, 8, 3))
    self.text_inputs = {
        'input_word_ids': jnp.ones((2, 10, 8), dtype=jnp.int32),
        'input_type_ids': jnp.zeros((2, 10, 8), dtype=jnp.int32),
        'input_mask': jnp.ones((2, 10, 8), dtype=jnp.int32),
    }
    self.caption = {
        'input_word_ids': jnp.ones((2, 2, 10), dtype=jnp.int32),
        'input_type_ids': jnp.zeros((2, 2, 10), dtype=jnp.int32),
        'input_mask': jnp.ones((2, 2, 10), dtype=jnp.int32),
    }
    self.video_title = {
        'input_word_ids': jnp.ones((2, 10), dtype=jnp.int32),
        'input_type_ids': jnp.zeros((2, 10), dtype=jnp.int32),
        'input_mask': jnp.ones((2, 10), dtype=jnp.int32),
    }
    self.text_emb_inputs = jnp.ones((2, 10, 8), dtype=jnp.float32)
    self.caption_emb_inputs = jnp.ones((2, 1, 8), dtype=jnp.float32)
    self.inputs = {
        'rgb': self.images,
        'class_names': self.text_inputs,
        'caption': self.caption,
        'video_title': self.video_title,
    }
    self.clip_video_tower_config = ml_collections.ConfigDict(
        {
            'modality_configs': {
                'rgb': ml_collections.ConfigDict({
                    'encoder_name': 'clip_video_encoder',
                    'encoder_config': ml_collections.ConfigDict({
                        'num_classes': -1,
                        'image_encoder_config': ml_collections.ConfigDict(
                            dict(
                                patches=ml_collections.ConfigDict(
                                    {'size': (4, 4, 1)}
                                ),
                                features=8,
                                num_layers=2,
                                num_heads=2,
                            )
                        ),
                        'temporal_encoding_config': ml_collections.ConfigDict({
                            'method': '3d_conv',
                            'kernel_init_method': 'central_frame_initializer',
                        }),
                        'temporal_encoder_config': None,
                        'final_endpoint': 'temporal_tokens',
                        'classifier': 'token',
                    }),
                    'projection_size': 8,
                    'projection_use_bias': False,
                })
            }
        }
    )
    self.clip_text_tower_config = ml_collections.ConfigDict({
        'encoder_name':
            'clip_text_encoder',
        'encoder_config':
            ml_collections.ConfigDict(
                dict(
                    vocab_size=100,
                    num_layers=2,
                    hidden_size=8,
                    num_heads=2,
                    classifier='eos')),
        'projection_size':
            8,
        'projection_use_bias':
            False,
    })
    self.pass_through_encoder_config = ml_collections.ConfigDict({
        'encoder_name': 'pass_through_encoder',
        'encoder_config': {},
        'input_type': 'text_emb',
    })
    self.video_textemb_fusion_encoder_config = ml_collections.ConfigDict({
        'self_attention_encoder_config': ml_collections.ConfigDict({
            'num_heads': 2,
            'mlp_dim': 16,
            'num_layers': 1,
            'dropout_rate': 0.0,
            'attention_dropout_rate': 0.0,
            'stochastic_depth': 0.1,
        }),
        'self_attention_encoder_name': 'transformer',
    })

  @parameterized.named_parameters(
      (
          'video_text_self_attention_moment_retrieval',
          'video_text_self_attention',
          'tokenized_text',
          None,
          'query_dependent_localization_head',
          'moment_retrieval',
          {
              'rgb_encoder',
              'text_encoder',
              'video_text_fusion',
              'moment_retrieval_head',
          },
      ),
      (
          'video_text_self_attention_highlight_detection',
          'video_text_self_attention',
          'tokenized_text',
          None,
          'query_dependent_localization_head',
          'highlight_detection',
          {
              'rgb_encoder',
              'text_encoder',
              'video_text_fusion',
              'highlight_detection_head',
          },
      ),
      (
          'video_text_emb_self_attention_temporal_localization',
          'video_text_emb_self_attention',
          'text_emb',
          None,
          'query_dependent_localization_head',
          'temporal_localization',
          {'rgb_encoder', 'video_text_fusion', 'temporal_localization_head'},
      ),
      (
          'video_text_emb_self_attention_action_segmentation',
          'video_text_emb_self_attention',
          'text_emb',
          None,
          'linear_head',
          'action_segmentation',
          {'rgb_encoder', 'video_text_fusion', 'action_segmentation_head'},
      ),
  )
  def test_video_text_single_tower_clip_encoder(self, fusion_type,
                                                text_input_type,
                                                projection_size, head_type,
                                                task, expected_keys):
    rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    config = copy.deepcopy(model._DEFAULT_UNLOC_CONFIG)
    config.model.classifier = 'token'
    config.model.num_classes = 1 if task == 'highlight_detection' else 10
    config.model.text_tower_config = self.clip_text_tower_config
    config.model.text_tower_config.projection_size = projection_size
    config.model.text_tower_config.input_type = text_input_type
    config.model.video_tower_config = self.clip_video_tower_config
    config.model.video_tower_config.modality_configs['rgb'].projection_size = (
        projection_size
    )
    config.model.video_text_fusion_config.type = fusion_type
    config.model.head_config.get(task).type = head_type

    if fusion_type == 'video_text_emb_self_attention':
      self.inputs['class_names'] = self.text_emb_inputs
      self.inputs['caption'] = self.caption_emb_inputs
      config.model.text_tower_config = self.pass_through_encoder_config
      config.model.video_text_fusion_config.config = (
          self.video_textemb_fusion_encoder_config
      )
    elif fusion_type == 'video_identity_text_emb':
      config.model.video_text_fusion_config.config = ml_collections.ConfigDict()

    output, params = model.VideoTextSingleTower(
        num_classes=config.model.num_classes,
        video_tower_config=config.model.video_tower_config,
        text_tower_config=config.model.text_tower_config,
        video_text_fusion_config=config.model.video_text_fusion_config,
        head_config=config.model.head_config,
        classifier=config.model.classifier,
    ).init_with_output(
        rngs, self.inputs, task=task, train=False, debug=False)
    if task == 'moment_retrieval':
      self.assertTupleEqual(output.shape, (2, 4, 4, 3))
    elif task == 'temporal_localization':
      self.assertTupleEqual(output.shape, (2, 4, 30))
    elif task == 'action_segmentation':
      self.assertTupleEqual(output.shape, (2, 4, 10))
    self.assertSetEqual(set(params['params'].keys()), expected_keys)

  @parameterized.parameters(
      (
          model.UnlocTemporalLocalizationModel,
          'temporal_localization',
          (2, 4, 30),
      ),
      (model.UnlocActionSegmentationModel, 'action_segmentation', (2, 4, 10)),
      (model.UnlocMomentRetrievalModel, 'moment_retrieval', (2, 4, 4, 3)),
  )
  def test_video_text_single_tower_model(self, model_cls, task,
                                         expected_output_shape):
    rng = random.PRNGKey(0)
    unloc = model_cls(config=None, dataset_meta_data={'num_classes': 10})
    rng, init_rng = random.split(rng)
    init_model_state, init_params = flax.core.pop(unloc.flax_model.init(
        init_rng, self.inputs, task=task, train=False), 'params')

    _, dropout_rng = random.split(rng)
    variables = {'params': init_params, **init_model_state}
    logits = unloc.flax_model.apply(
        variables,
        self.inputs,
        task=task,
        train=False,
        rngs={'dropout': dropout_rng})
    self.assertEqual(logits.shape, expected_output_shape)

  @parameterized.parameters(
      (
          model.UnlocTemporalLocalizationModel,
          'temporal_localization',
          (2, 4, 30),
      ),
      (model.UnlocActionSegmentationModel, 'action_segmentation', (2, 4, 10)),
      (model.UnlocMomentRetrievalModel, 'moment_retrieval', (2, 4, 4, 3)),
  )
  def test_video_text_single_tower_model_separate_steps(self, model_cls, task,
                                                        expected_output_shape):
    rng = random.PRNGKey(0)
    unloc = model_cls(config=None, dataset_meta_data={'num_classes': 10})
    rng, init_rng = random.split(rng)
    init_model_state, init_params = flax.core.pop(unloc.flax_model.init(
        init_rng, self.inputs, task=task, train=False), 'params')

    _, dropout_rng = random.split(rng)
    variables = {'params': init_params, **init_model_state}
    video_tokens = unloc.flax_model.apply(
        variables,
        self.inputs,
        train=False,
        rngs={'dropout': dropout_rng},
        method=unloc.flax_model.encode_video)
    text_tokens = unloc.flax_model.apply(
        variables,
        self.inputs,
        task=task,
        train=False,
        rngs={'dropout': dropout_rng},
        method=unloc.flax_model.encode_text)
    if task == 'moment_retrieval':
      input_word_ids = jnp.reshape(self.caption['input_word_ids'], (-1, 10))
      text_input_mask = jnp.reshape(self.caption['input_mask'], (-1, 10))
    else:
      input_word_ids = self.text_inputs['input_word_ids'][0]
      text_input_mask = self.text_inputs['input_mask'][0]
    logits = unloc.flax_model.apply(
        variables,
        video_tokens,
        text_tokens,
        task=task,
        input_word_ids=input_word_ids,
        text_input_mask=text_input_mask,
        video_input_mask=self.inputs.get('input_mask'),
        train=False,
        rngs={'dropout': dropout_rng},
        method=unloc.flax_model.fuse_video_text)
    self.assertEqual(logits.shape, expected_output_shape)


if __name__ == '__main__':
  tf.test.main()
