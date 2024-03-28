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

"""Tests for backbone.py."""

import math

from absl.testing import absltest
from absl.testing import parameterized
from jax import random
import jax.numpy as jnp
from scenic.projects.baselines.deformable_detr.backbone import DeformableDETRBackbone
from scenic.projects.baselines.deformable_detr.backbone import mask_for_shape


class DeformableDETRBackboneTest(parameterized.TestCase):
  """Tests for DeformableDETRBackbone."""

  @parameterized.named_parameters(
      {
          'testcase_name': 'Single scale.',
          'num_feature_levels': 1,
          'shape': (2, 32, 64, 3),
      }, {
          'testcase_name': 'Multi scale.',
          'num_feature_levels': 3,
          'shape': (2, 32, 64, 3),
      }, {
          'testcase_name': 'Odd tiny shape.',
          'num_feature_levels': 3,
          'shape': (10, 9, 4, 3),
      })
  def test_backbone_output_shape(self, shape, num_feature_levels):
    """Test DeformableDETREncoderLayer output shape."""
    rng = random.PRNGKey(8877)
    x = jnp.ones(shape)

    backbone = DeformableDETRBackbone(
        embed_dim=4,
        num_filters=16,
        num_layers=18,
        num_feature_levels=num_feature_levels)
    (features, pad_masks, pos_embs), _ = backbone.init_with_output(rng, x)
    self.assertLen(features, num_feature_levels)
    for i in range(num_feature_levels):
      fshape = features[i].shape[1:3]
      self.assertSequenceEqual(fshape, pad_masks[i].shape[1:3])
      self.assertEqual(math.prod(fshape), pos_embs[i].shape[1])
      if i > 0:
        self.assertGreater(features[i].shape[-1], features[i - 1].shape[-1])

  @parameterized.named_parameters(
      {
          'testcase_name': 'Without mask.',
          'shape': (2, 32, 64, 3),
          'pad_mask_shape': None,
      }, {
          'testcase_name': 'Down-scale preserve ratio.',
          'shape': (2, 32, 64, 3),
          'pad_mask_shape': (2, 128, 256),
      }, {
          'testcase_name': 'Up-scale does not preserve ratio.',
          'shape': (2, 32, 64, 3),
          'pad_mask_shape': (2, 16, 16),
      })
  def test_mask_for_shape(self, shape, pad_mask_shape):
    """Test DeformableDETREncoderLayer output shape."""
    pad_mask = None
    if pad_mask_shape is not None:
      pad_mask = jnp.ones(pad_mask_shape, dtype=jnp.bool_)
    x = mask_for_shape(shape, pad_mask=pad_mask)
    self.assertSequenceEqual(x.shape, shape[:3])


if __name__ == '__main__':
  absltest.main()
