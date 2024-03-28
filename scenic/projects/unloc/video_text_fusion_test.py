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

"""Tests for video_text_fusion."""

from absl.testing import parameterized
from jax import random
import ml_collections
import numpy as np
from scenic.projects.unloc import video_text_fusion
import tensorflow as tf


class LayersTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('moment_retrieval', None, (2, 2, 8, 16), (2, 2, 16)),
      ('temporal_localization', None, (2, 4, 8, 16), (2, 4, 16)),
      (
          'temporal_localization',
          np.ones((2, 8), dtype=np.int32),
          (2, 4, 8, 16),
          (2, 4, 16),
      ),
      (
          'action_segmentation',
          np.ones((2, 8), dtype=np.int32),
          (2, 8, 4, 16),
          (2, 4, 16),
      ),
  )
  def test_video_text_emb_self_attention_fusion(
      self, task, video_input_mask, expected_video_shape, expected_text_shape
  ):
    video_tokens = np.ones((2, 8, 16), dtype=np.float32)
    text_embs = (
        np.ones((2, 16), np.float32) if task == 'moment_retrieval' else np.ones(
            (4, 16), np.float32))
    config = ml_collections.ConfigDict({
        'num_heads': 2,
        'mlp_dim': 32,
        'num_layers': 2,
        'dropout_rate': 0.0,
        'attention_dropout_rate': 0.0,
        'stochastic_depth': 0.0,
    })
    rng = random.PRNGKey(0)
    (
        actual_video_tokens,
        actual_text_tokens,
    ), _ = video_text_fusion.VideoTextEmbSelfAttentionFusion(
        self_attention_encoder_config=config
    ).init_with_output(
        rng,
        video_tokens,
        text_embs,
        video_input_mask=video_input_mask,
        task=task,
    )
    self.assertTupleEqual(actual_video_tokens.shape, expected_video_shape)
    self.assertTupleEqual(actual_text_tokens.shape, expected_text_shape)

  @parameterized.parameters(
      ('temporal_localization', (2, 4, 12, 16), (2, 4, 16)),
      ('moment_retrieval', (2, 2, 12, 16), (2, 2, 16)),
  )
  def test_video_text_emb_self_attention_fusion_fpn(self, task,
                                                    expected_video_shape,
                                                    expected_text_shape):
    batch_size = 2
    num_classes = 4
    video_seq = 8
    video_tokens = np.ones((batch_size, video_seq, 16), dtype=np.float32)
    if task == 'moment_retrieval':
      text_tokens = np.ones((batch_size, 16), np.float32)
    else:
      text_tokens = np.ones((num_classes, 16), np.float32)
    video_input_mask = np.ones((batch_size, video_seq + video_seq // 2),
                               np.int32)
    config = ml_collections.ConfigDict({
        'num_heads':
            2,
        'mlp_dim':
            32,
        'num_layers':
            3,
        'dropout_rate':
            0.0,
        'attention_dropout_rate':
            0.0,
        'stochastic_depth':
            0.0,
        'feature_pyramid_config':
            ml_collections.ConfigDict({
                'num_features_level0': 8,
                'feature_pyramid_levels': [1, 2],
                'feature_pyramid_downsample_stride': 2,
            })
    })
    rng = random.PRNGKey(0)
    (actual_video_tokens, actual_text_tokens), _ = (
        video_text_fusion.VideoTextEmbSelfAttentionFusion(
            self_attention_encoder_name='fpn',
            self_attention_encoder_config=config,
        ).init_with_output(
            rng, video_tokens, text_tokens, task, video_input_mask
        )
    )
    self.assertTupleEqual(actual_video_tokens.shape, expected_video_shape)
    self.assertTupleEqual(actual_text_tokens.shape, expected_text_shape)

  @parameterized.parameters(
      (False, 'moment_retrieval', (2, 2, 8, 16), (2, 2, 16)),
      (False, 'highlight_detection', (2, 1, 8, 16), (2, 1, 16)),
      (True, 'highlight_detection', (2, 1, 8, 16), (2, 1, 16)),
      (False, 'temporal_localization', (2, 4, 8, 16), (2, 4, 16)),
      (False, 'action_segmentation', (2, 8, 4, 16), (2, 4, 16)),
  )
  def test_video_text_self_attention_fusion(
      self,
      use_all_text_tokens,
      task,
      expected_video_shape,
      expected_text_shape,
  ):
    video_tokens = np.ones((2, 8, 16), dtype=np.float32)
    if task == 'moment_retrieval':
      text_tokens = np.ones((2, 8, 16), np.float32)
      input_word_ids = np.ones((2, 8), np.int32)
      input_mask = np.ones((2, 8), np.int32)
    elif task == 'highlight_detection':
      text_tokens = np.ones((2, 16, 16), np.float32)
      input_word_ids = np.ones((2, 16), np.int32)
      input_mask = np.ones((2, 16), np.int32)
    else:
      text_tokens = np.ones((4, 8, 16), np.float32)
      input_word_ids = np.ones((4, 8), np.int32)
      input_mask = np.ones((4, 8), np.int32)
    config = ml_collections.ConfigDict({
        'num_heads': 2,
        'mlp_dim': 32,
        'num_layers': 2,
        'dropout_rate': 0.0,
        'attention_dropout_rate': 0.0,
        'stochastic_depth': 0.0,
    })
    rng = random.PRNGKey(0)
    (
        actual_video_tokens,
        actual_text_tokens,
    ), _ = video_text_fusion.VideoTextSelfAttentionFusion(
        text_tower_classifier='eos',
        self_attention_encoder_name='transformer',
        self_attention_encoder_config=config,
        use_all_text_tokens=use_all_text_tokens,
    ).init_with_output(
        rng, video_tokens, text_tokens, task, input_word_ids, input_mask
    )
    self.assertTupleEqual(actual_video_tokens.shape, expected_video_shape)
    self.assertTupleEqual(actual_text_tokens.shape, expected_text_shape)

  @parameterized.parameters(
      ('temporal_localization', (2, 4, 12, 16), (2, 4, 16)),
      ('moment_retrieval', (2, 2, 12, 16), (2, 2, 16)),
      ('highlight_detection', (2, 1, 12, 16), (2, 1, 16)),
  )
  def test_video_text_self_attention_fusion_fpn(self, task,
                                                expected_video_shape,
                                                expected_text_shape):
    batch_size = 2
    num_classes = 4
    video_seq = 8
    text_seq = 10
    video_tokens = np.ones((batch_size, video_seq, 16), dtype=np.float32)
    if task == 'moment_retrieval' or task == 'highlight_detection':
      text_tokens = np.ones((batch_size, text_seq, 16), np.float32)
      input_word_ids = np.ones((batch_size, text_seq), np.int32)
      text_input_mask = np.ones((batch_size, text_seq), np.int32)
    else:
      text_tokens = np.ones((num_classes, text_seq, 16), np.float32)
      input_word_ids = np.ones((num_classes, text_seq), np.int32)
      text_input_mask = np.ones((num_classes, text_seq), np.int32)
    video_input_mask = np.ones((batch_size, video_seq + video_seq // 2),
                               np.int32)
    config = ml_collections.ConfigDict({
        'num_heads':
            2,
        'mlp_dim':
            32,
        'num_layers':
            3,
        'dropout_rate':
            0.0,
        'attention_dropout_rate':
            0.0,
        'stochastic_depth':
            0.0,
        'feature_pyramid_config':
            ml_collections.ConfigDict({
                'num_features_level0': 8,
                'feature_pyramid_levels': [1, 2],
                'feature_pyramid_downsample_stride': 2,
            })
    })
    rng = random.PRNGKey(0)
    (
        actual_video_tokens,
        actual_text_tokens,
    ), _ = video_text_fusion.VideoTextSelfAttentionFusion(
        text_tower_classifier='eos',
        self_attention_encoder_name='fpn',
        self_attention_encoder_config=config,
        use_all_text_tokens=False,
    ).init_with_output(
        rng,
        video_tokens,
        text_tokens,
        task,
        input_word_ids,
        text_input_mask,
        video_input_mask,
    )
    self.assertTupleEqual(actual_video_tokens.shape, expected_video_shape)
    self.assertTupleEqual(actual_text_tokens.shape, expected_text_shape)

  @parameterized.parameters(
      (None, 0, (0, 1, 2, 3)),
      (None, 3, (0, 1, 2)),
      (np.ones((2, 8 + 4), np.float32), 0, (0, 1, 2),),
      (np.ones((2, 8 + 4), np.float32), 3, (0, 1, 2),),
  )
  def test_feature_pyramid_encoder_no_text_token(
      self, input_mask, window_size, window_block_indexes
  ):
    rng = random.PRNGKey(0)
    x = np.ones((2, 8, 8), dtype=np.float32)
    output, params = video_text_fusion.FeaturePyramidEncoder(
        num_layers=4,
        mlp_dim=16,
        num_heads=2,
        window_size=window_size,
        window_block_indexes=window_block_indexes,
        feature_pyramid_config=ml_collections.ConfigDict({
            'num_features_level0': 8,
            'feature_pyramid_levels': [2, 3],
            'feature_pyramid_downsample_stride': 2,
        }),
    ).init_with_output(rng, x, input_mask, train=False)
    self.assertTupleEqual(output.shape, (2, 12, 8))
    self.assertSetEqual(
        set(params['params'].keys()), {
            'posembed_input', 'encoderblock_0', 'encoderblock_1',
            'encoderblock_2', 'encoderblock_3', 'output_conv_0',
            'output_conv_1', 'output_ln_0', 'output_ln_1'
        })

  @parameterized.parameters(
      (None, 3, (0, 1, 2)),
      (np.ones((2, 8 + 1 + 4 + 1 + 2 + 1), np.float32), 0, (0, 1, 2),),
      (np.ones((2, 8 + 1 + 4 + 1 + 2 + 1), np.float32), 3, (0, 1, 2),),
  )
  def test_feature_pyramid_encoder_w_text_token(
      self, input_mask, window_size, window_block_indexes
  ):
    rng = random.PRNGKey(0)
    x = np.ones((2, 8 + 1, 8), dtype=np.float32)
    output, params = video_text_fusion.FeaturePyramidEncoder(
        num_layers=4,
        mlp_dim=16,
        num_heads=2,
        window_size=window_size,
        window_block_indexes=window_block_indexes,
        feature_pyramid_config=ml_collections.ConfigDict({
            'num_features_level0': 8,
            'feature_pyramid_levels': [1, 2, 3],
            'feature_pyramid_downsample_stride': 2,
        }),
    ).init_with_output(rng, x, input_mask, train=False)
    self.assertTupleEqual(output.shape, (2, 14 + 1, 8))
    self.assertSetEqual(
        set(params['params'].keys()), {
            'posembed_input', 'encoderblock_0', 'encoderblock_1',
            'encoderblock_2', 'encoderblock_3', 'output_conv_0',
            'output_conv_1', 'output_conv_2', 'output_ln_0', 'output_ln_1',
            'output_ln_2'
        })

  @parameterized.parameters(
      (None, 3, (0, 1, 2)),
      (np.ones((2, 8 + 1 + 4 + 1 + 2 + 1), np.float32), 0, (0, 1, 2),),
      (np.ones((2, 8 + 1 + 4 + 1 + 2 + 1), np.float32), 3, (0, 1, 2),),
  )
  def test_simple_pyramid_encoder_w_text_token(
      self, input_mask, window_size, window_block_indexes
  ):
    rng = random.PRNGKey(0)
    x = np.ones((2, 8 + 1, 8), dtype=np.float32)
    output, params = video_text_fusion.SimplePyramidEncoder(
        num_layers=4,
        mlp_dim=16,
        num_heads=2,
        window_size=window_size,
        window_block_indexes=window_block_indexes,
        feature_pyramid_config=ml_collections.ConfigDict({
            'num_features_level0': 8,
            'feature_pyramid_levels': [1, 2, 3],
            'feature_pyramid_downsample_stride': 2,
        }),
    ).init_with_output(rng, x, input_mask, train=False)
    self.assertTupleEqual(output.shape, (2, 14 + 1, 8))
    self.assertSetEqual(
        set(params['params'].keys()),
        {
            'posembed_input',
            'encoderblock_0',
            'encoderblock_1',
            'encoderblock_2',
            'encoderblock_3',
            'output_conv_0',
            'output_conv_1',
            'output_conv_2',
            'output_ln_0',
            'output_ln_1',
            'output_ln_2',
        },
    )

if __name__ == '__main__':
  tf.test.main()
