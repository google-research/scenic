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

"""Tests OWL-ViT models."""
import functools
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.projects.owl_vit import models

# Config for a tiny transformer for testing.
TINY_VIT_CONFIG = {'depth': 2, 'width': 64, 'mlp_dim': 256, 'num_heads': 2}


class TextZeroShotDetectionModuleTest(parameterized.TestCase):
  """Tests for TextZeroShotDetectionModule."""

  @parameterized.parameters((224, True), (224, False), (1333, True))
  def test_clip_zero_shot_detection_module(self, img_size, normalize):
    """Tests CLIP detection model construction and application."""
    batch_size, num_queries, seq_len, patch_size = 8, 10, 16, 32
    side_patches = int(np.ceil(img_size / patch_size))
    body_configs = ml_collections.ConfigDict(dict(
        type='clip',
        variant='vit_b32',
        merge_class_token='drop',
        text_stochastic_droplayer_rate=0.1,
        vision_stochastic_droplayer_rate=0.1,
        ))
    images = jnp.ones((batch_size, img_size, img_size, 3))
    texts = jnp.ones((batch_size, num_queries, seq_len), dtype=jnp.int32)

    model = models.TextZeroShotDetectionModule(
        body_configs,
        normalize=normalize)

    fn = functools.partial(model.init_with_output, train=False)
    out, variables = jax.eval_shape(fn, jax.random.PRNGKey(0), images, texts)

    self.assertCountEqual(variables.keys(), ['params'])
    expected_shapes = {
        'feature_map': (batch_size, side_patches, side_patches, 768),
        'pred_boxes': (batch_size, side_patches**2, 4),
        'pred_logits': (batch_size, side_patches**2, num_queries),
        'class_embeddings': (batch_size, side_patches**2, 512),
        'query_embeddings': (batch_size, num_queries, 512)
    }
    self.assertEqual(expected_shapes, jax.tree_util.tree_map(jnp.shape, out))


if __name__ == '__main__':
  absltest.main()
