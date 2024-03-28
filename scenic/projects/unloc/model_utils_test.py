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

from absl.testing import absltest
from absl.testing import parameterized
import flax
from flax import struct
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.projects.unloc import model_utils
from scenic.train_lib import train_utils


class ModelUtilsTest(parameterized.TestCase):

  def assertDictEqualRecursive(self, actual, expected):
    self.assertEqual(type(actual), type(expected))
    if isinstance(actual, dict):
      self.assertSameElements(actual.keys(), expected.keys())
      for key, _ in expected.items():
        self.assertDictEqualRecursive(actual[key], expected[key])
    elif isinstance(actual, jnp.ndarray):
      self.assertTrue(jnp.array_equal(actual, expected))
    elif isinstance(actual, np.ndarray):
      np.testing.assert_allclose(actual, expected)
    else:
      self.assertEqual(actual, expected)

  @parameterized.named_parameters(
      (
          'one_level_fpn',
          1,
          np.ones((4, 8 + 1, 16), dtype=np.float32),
          np.ones((4, 8, 16), dtype=np.float32),
      ),
      (
          'three_level_fpn',
          3,
          np.ones((4, 8 + 1 + 4 + 1 + 2 + 1, 16), dtype=np.float32),
          np.ones((4, 8 + 4 + 2, 16), dtype=np.float32),
      ),
  )
  def test_extract_pyramid_video_tokens(self, num_pyramid_levels, input_tokens,
                                        expected_output):
    actual = model_utils.extract_pyramid_video_tokens(
        input_tokens,
        num_pyramid_levels=num_pyramid_levels,
        feature_pyramid_downsample_stride=2,
        num_video_tokens_level0=8,
        num_text_tokens=1)
    np.testing.assert_equal(actual, expected_output)

  @parameterized.parameters(
      (1, 1, 0, np.array((), dtype=np.int32)),
      (3, 2, 0, np.array((128, 192), dtype=np.int32)),
      (4, 2, 1, np.array((129, 194, 227), dtype=np.int32)),
  )
  def test_create_pyramid_split_indices(self, num_pyramid_levels,
                                        feature_pyramid_downsample_stride,
                                        num_extra_features_per_level,
                                        expected_indices):
    actual = model_utils.create_pyramid_split_indices(
        num_features_level0=128,
        num_pyramid_levels=num_pyramid_levels,
        feature_pyramid_downsample_stride=feature_pyramid_downsample_stride,
        num_extra_features_per_level=num_extra_features_per_level)
    np.testing.assert_equal(actual, expected_indices)

  def test_create_pyramid_input_masks(self):
    actual = model_utils.create_pyramid_input_masks(
        input_mask=np.ones((2, 128 + 64 + 32), dtype=np.int32),
        num_features_level0=128,
        num_pyramid_levels=3,
        feature_pyramid_downsample_stride=2,
        num_text_tokens=0)
    self.assertLen(actual, 3)
    np.testing.assert_equal(actual[0], np.ones((2, 128), dtype=np.int32))
    np.testing.assert_equal(actual[1], np.ones((2, 64), dtype=np.int32))
    np.testing.assert_equal(actual[2], np.ones((2, 32), dtype=np.int32))

  def test_merge_pyramid_input_masks(self):
    actual = model_utils.merge_pyramid_input_masks(
        [np.ones((2, 128), dtype=np.int32),
         np.ones((2, 64), dtype=np.int32)])
    np.testing.assert_equal(actual, np.ones((2, 128 + 64), dtype=np.int32))

  def test_merge_pyramid_input_masks_with_text_mask(self):
    actual = model_utils.merge_pyramid_input_masks(
        [np.ones((2, 128), dtype=np.int32),
         np.ones((2, 64), dtype=np.int32)],
        input_text_mask=np.ones((2, 8), dtype=np.int32))
    np.testing.assert_equal(actual,
                            np.ones((2, 128 + 8 + 64 + 8), dtype=np.int32))

  @parameterized.parameters(
      (
          ml_collections.ConfigDict({
              'init_from': ml_collections.ConfigDict({
                  'video_encoders': {
                      'rgb': ml_collections.ConfigDict(
                          {'load_projection': True}
                      ),
                  },
                  'text_encoder': ml_collections.ConfigDict(
                      {'load_projection': True}
                  ),
              }),
              'model': ml_collections.ConfigDict({
                  'temporal_encoding_config': ml_collections.ConfigDict({
                      'kernel_init_method': 'central_frame_initializer',
                      'method': '3d_conv',
                  }),
                  'video_tower_config': ml_collections.ConfigDict(
                      {
                          'modality_configs': {
                              'rgb': ml_collections.ConfigDict({
                                  'projection_size': 16,
                                  'encoder_config': ml_collections.ConfigDict({
                                      'classifier': 'token',
                                      'image_encoder_config': (
                                          ml_collections.ConfigDict(
                                              {
                                                  'classifier': 'token',
                                              }
                                          )
                                      ),
                                  }),
                              }),
                          }
                      }
                  ),
                  'text_tower_config': ml_collections.ConfigDict(
                      {'projection_size': 16}
                  ),
              }),
          }),
          'rgb',
      ),
      (
          ml_collections.ConfigDict({
              'init_from': ml_collections.ConfigDict({
                  'video_encoder': ml_collections.ConfigDict(
                      {'load_projection': True}
                  ),
                  'text_encoder': ml_collections.ConfigDict(
                      {'load_projection': True}
                  ),
              }),
              'model': ml_collections.ConfigDict({
                  'temporal_encoding_config': ml_collections.ConfigDict({
                      'kernel_init_method': 'central_frame_initializer',
                      'method': '3d_conv',
                  }),
                  'video_tower_config': ml_collections.ConfigDict({
                      'projection_size': 16,
                      'encoder_config': ml_collections.ConfigDict({
                          'classifier': 'token',
                          'image_encoder_config': ml_collections.ConfigDict(
                              {
                                  'classifier': 'token',
                              }
                          ),
                      }),
                  }),
                  'text_tower_config': ml_collections.ConfigDict(
                      {'projection_size': 16}
                  ),
              }),
          }),
          'video',
      ),
  )
  def test_initialize_from_clip_model(self, config, modality_name):
    params = {
        f'{modality_name}_encoder': {
            'conv1': {
                'kernel': np.zeros((4 * 4 * 3, 8)),
            },
            'class_embedding': np.zeros((8, 10)),
            'VisionTransformer': {
                'transformer': np.zeros((10,)),
                'ln_pre': np.zeros((10,)),
                'ln_post': np.zeros((10,)),
                'positional_embedding': np.zeros((17, 10)),
            },
        },
        'text_encoder': {
            'positional_embedding': np.zeros((15, 10)),
            'token_embedding': np.zeros((100, 10)),
            'transformer': np.zeros((10,)),
            'ln_final': np.zeros((10,)),
        },
        'text_projection': {
            'kernel': np.zeros((16, 16)),
        },
        f'{modality_name}_projection': {
            'kernel': np.zeros((16, 16)),
        },
    }
    clip_params = {
        'params': {
            'text': {
                'positional_embedding': np.ones((31, 10)),
                'transformer': np.ones((10,)),
                'ln_final': np.ones((10,)),
                'token_embedding': np.ones((100, 10)),
                'text_projection': {
                    'kernel': np.ones((16, 16)),
                },
            },
            'visual': {
                'conv1': {
                    'kernel': np.ones((4, 4, 3, 8)),
                },
                'class_embedding': np.ones((10,)),
                'ln_post': np.ones((10,)),
                'ln_pre': np.ones((10,)),
                'positional_embedding': np.ones((10, 10)),
                'proj': {
                    'kernel': np.ones((16, 16)),
                },
                'transformer': np.ones((10,)),
            },
        }
    }
    train_state = train_utils.TrainState(
        tx=struct.field(pytree_node=False), params=flax.core.freeze(params))
    expected_params = {
        f'{modality_name}_encoder': {
            'conv1': {
                'kernel': np.ones((4 * 4 * 3, 8)),
            },
            'class_embedding': jnp.ones((8, 10)),
            'VisionTransformer': {
                'transformer': np.ones((10,)),
                'ln_pre': np.ones((10,)),
                'ln_post': np.ones((10,)),
                'positional_embedding': jnp.ones((17, 10)),
            },
        },
        'text_encoder': {
            'positional_embedding': np.ones((15, 10)),
            'token_embedding': np.ones((100, 10)),
            'transformer': np.ones((10,)),
            'ln_final': np.ones((10,)),
        },
        'text_projection': {
            'kernel': np.ones((16, 16)),
        },
        f'{modality_name}_projection': {
            'kernel': np.ones((16, 16)),
        },
    }
    train_state = model_utils.initialize_from_clip_model(
        config,
        train_state,
        clip_params,
        load_image_tower=True,
        load_text_tower=True,
        video_modality_name=modality_name,
    )
    params = flax.core.unfreeze(train_state.params)
    self.assertDictEqualRecursive(params, expected_params)


if __name__ == '__main__':
  absltest.main()
