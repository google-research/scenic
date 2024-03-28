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

"""Tests for heads."""

from absl.testing import parameterized
from jax import random
import ml_collections
import numpy as np
from scenic.projects.unloc import heads
import tensorflow as tf


class HeadsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (np.ones((2, 10, 8), np.float32), (2, 10)),
      (np.ones((2, 8, 10, 8), np.float32), (2, 8, 10)),
  )
  def test_linear_head(self, inputs, expected_shape):
    rng = random.PRNGKey(0)
    output, _ = heads.LinearHead().init_with_output(
        rng, inputs, None, '', train=False
    )
    self.assertTupleEqual(output.shape, expected_shape)

  @parameterized.named_parameters(
      (
          'temporal_localization_weight_sharing',
          np.ones((2, 10, 4, 8)),
          np.ones((2, 10, 8)),
          None,
          True,
          True,
          (2, 4, 30),
          {'ClassificationHead', 'RegressionHead', 'scale_0', 'shift_0'},
      ),
      (
          'temporal_localization_w_fpn_weight_sharing',
          np.ones((2, 10, 6, 8)),
          np.ones((2, 10, 8)),
          ml_collections.ConfigDict({
              'num_features_level0': 4,
              'feature_pyramid_levels': [0, 1],
              'feature_pyramid_downsample_stride': 2,
          }),
          True,
          True,
          (2, 6, 30),
          {
              'ClassificationHead',
              'RegressionHead',
              'scale_0',
              'scale_1',
              'shift_0',
              'shift_1',
          },
      ),
      (
          'temporal_localization_w_fpn_weight_sharing_class_agnostic_displacements',
          np.ones((2, 10, 6, 8)),
          np.ones((2, 10, 8)),
          ml_collections.ConfigDict({
              'num_features_level0': 4,
              'feature_pyramid_levels': [0, 1],
              'feature_pyramid_downsample_stride': 2,
          }),
          True,
          False,
          (2, 6, 12),
          {
              'ClassificationHead',
              'RegressionHead',
              'scale_0',
              'scale_1',
              'shift_0',
              'shift_1',
          },
      ),
      (
          'temporal_localization_w_fpn_separate_decoders',
          np.ones((2, 10, 6, 8)),
          np.ones((2, 10, 8)),
          ml_collections.ConfigDict({
              'num_features_level0': 4,
              'feature_pyramid_levels': [0, 1],
              'feature_pyramid_downsample_stride': 2,
          }),
          False,
          True,
          (2, 6, 30),
          {
              'ClassificationHead_0',
              'ClassificationHead_1',
              'RegressionHead_0',
              'RegressionHead_1',
          },
      ),
      (
          'temporal_localization_w_fpn_separate_decoders_class_agnostic_displacements',
          np.ones((2, 10, 6, 8)),
          np.ones((2, 10, 8)),
          ml_collections.ConfigDict({
              'num_features_level0': 4,
              'feature_pyramid_levels': [0, 1],
              'feature_pyramid_downsample_stride': 2,
          }),
          False,
          False,
          (2, 6, 12),
          {
              'ClassificationHead_0',
              'ClassificationHead_1',
              'RegressionHead_0',
              'RegressionHead_1',
          },
      ),
  )
  def test_query_dependent_localization_head_tal(
      self,
      video_tokens,
      txt_emb,
      feature_pyramid_config,
      weight_sharing,
      output_per_class_displacements,
      expected_output_shape,
      expected_keys,
  ):
    rng = random.PRNGKey(0)
    output, params = heads.QueryDependentLocalizationHead(
        num_conv_layers=3,
        kernel_size=3,
        num_classes=-1,
        feature_pyramid_config=feature_pyramid_config,
        weight_sharing=weight_sharing,
        output_per_class_displacements=output_per_class_displacements,
    ).init_with_output(
        rng, video_tokens, txt_emb, 'temporal_localization', train=False
    )
    self.assertTupleEqual(output.shape, expected_output_shape)
    self.assertSetEqual(set(params['params'].keys()), expected_keys)

  @parameterized.named_parameters(
      (
          'weight_sharing_per_class_displacement',
          np.ones((2, 4, 8)),
          None,
          True,
          True,
          (2, 4, 30),
          {'ClassificationHead', 'RegressionHead', 'scale_0', 'shift_0'},
      ),
      (
          'weight_sharing',
          np.ones((2, 4, 8)),
          None,
          True,
          False,
          (2, 4, 12),
          {'ClassificationHead', 'RegressionHead', 'scale_0', 'shift_0'},
      ),
      (
          'fpn_weight_sharing_per_class_displacement',
          np.ones((2, 6, 8)),
          ml_collections.ConfigDict({
              'num_features_level0': 4,
              'feature_pyramid_levels': [0, 1],
              'feature_pyramid_downsample_stride': 2,
          }),
          True,
          True,
          (2, 6, 30),
          {
              'ClassificationHead',
              'RegressionHead',
              'scale_0',
              'scale_1',
              'shift_0',
              'shift_1',
          },
      ),
      (
          'fpn_separate_decoders_per_class_displacement',
          np.ones((2, 6, 8)),
          ml_collections.ConfigDict({
              'num_features_level0': 4,
              'feature_pyramid_levels': [0, 1],
              'feature_pyramid_downsample_stride': 2,
          }),
          False,
          True,
          (2, 6, 30),
          {
              'ClassificationHead_0',
              'ClassificationHead_1',
              'RegressionHead_0',
              'RegressionHead_1',
          },
      ),
      (
          'fpn_separate_decoders',
          np.ones((2, 6, 8)),
          ml_collections.ConfigDict({
              'num_features_level0': 4,
              'feature_pyramid_levels': [0, 1],
              'feature_pyramid_downsample_stride': 2,
          }),
          False,
          False,
          (2, 6, 12),
          {
              'ClassificationHead_0',
              'ClassificationHead_1',
              'RegressionHead_0',
              'RegressionHead_1',
          },
      ),
  )
  def test_localization_head_tal(
      self,
      x,
      feature_pyramid_config,
      weight_sharing,
      output_per_class_displacements,
      expected_output_shape,
      expected_keys,
  ):
    rng = random.PRNGKey(0)
    output, params = heads.LocalizationHead(
        num_conv_layers=3,
        kernel_size=3,
        num_classes=10,
        feature_pyramid_config=feature_pyramid_config,
        weight_sharing=weight_sharing,
        output_per_class_displacements=output_per_class_displacements,
    ).init_with_output(
        rng,
        x,
        None,
        'temporal_localization',
        train=False,
    )
    self.assertTupleEqual(output.shape, expected_output_shape)
    self.assertSetEqual(set(params['params'].keys()), expected_keys)

  @parameterized.named_parameters(
      (
          'moment_retrieval_weight_sharing',
          np.ones((2, 2, 8, 16)),
          np.ones((2, 2, 16)),
          None,
          True,
          (2, 2, 8, 3),
          {'ClassificationHead', 'RegressionHead', 'scale_0', 'shift_0'},
      ),
      (
          'moment_retrieval_w_fpn_weight_sharing',
          np.ones((2, 2, 12, 16)),
          np.ones((2, 2, 16)),
          ml_collections.ConfigDict({
              'num_features_level0': 8,
              'feature_pyramid_levels': [0, 1],
              'feature_pyramid_downsample_stride': 2,
          }),
          True,
          (2, 2, 12, 3),
          {
              'ClassificationHead',
              'RegressionHead',
              'scale_0',
              'scale_1',
              'shift_0',
              'shift_1',
          },
      ),
      (
          'moment_retrieval_w_fpn_separate_decoders',
          np.ones((2, 2, 12, 16)),
          np.ones((2, 2, 16)),
          ml_collections.ConfigDict({
              'num_features_level0': 8,
              'feature_pyramid_levels': [0, 1],
              'feature_pyramid_downsample_stride': 2,
          }),
          False,
          (2, 2, 12, 3),
          {
              'ClassificationHead_0',
              'ClassificationHead_1',
              'RegressionHead_0',
              'RegressionHead_1',
          },
      ),
  )
  def test_query_dependent_localization_head_mr(
      self,
      video_tokens,
      txt_emb,
      feature_pyramid_config,
      weight_sharing,
      expected_output_shape,
      expected_keys,
  ):
    rng = random.PRNGKey(0)
    output, params = heads.QueryDependentLocalizationHead(
        num_conv_layers=3,
        kernel_size=3,
        num_classes=-1,
        feature_pyramid_config=feature_pyramid_config,
        weight_sharing=weight_sharing,
    ).init_with_output(
        rng, video_tokens, txt_emb, 'moment_retrieval', train=False
    )
    self.assertTupleEqual(output.shape, expected_output_shape)
    self.assertSetEqual(set(params['params'].keys()), expected_keys)


if __name__ == '__main__':
  tf.test.main()
