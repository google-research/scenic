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

"""Tests OWL-ViT layers."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_collections
from scenic.projects.owl_vit import layers


class LayersTest(parameterized.TestCase):
  """Tests for ViT+ layers."""

  @parameterized.parameters((1, 2), (2, 4), (3, 6))
  def test_mlp_block(self, num_layers, expected_num_weights):
    batch_size, in_dim, out_dim = 8, 20, 30
    mlp_configs = {'out_dim': out_dim, 'num_layers': num_layers}
    mlp = layers.PredictorMLP(**mlp_configs)
    inputs = jnp.ones((batch_size, in_dim))
    out, variables = mlp.init_with_output(jax.random.PRNGKey(0), inputs)
    self.assertEqual(out.shape, (batch_size, out_dim))
    weights, _ = jax.tree_util.tree_flatten(variables)
    self.assertLen(weights, expected_num_weights)

  def test_class_predictor(self):
    """Tests class predictor."""
    batch_size, num_queries, img_size, patch_size = 8, 10, 224, 32
    side_patches = img_size // patch_size
    num_patches = side_patches ** 2
    img_emb_dim, out_dim = 64, 100
    img_emb = jnp.ones((batch_size, num_patches, img_emb_dim))
    txt_emb = jnp.ones((batch_size, num_queries, out_dim))
    label_mask = jnp.ones((batch_size, num_queries), dtype=jnp.int32)

    class_predictor = layers.ClassPredictor(out_dim=out_dim)
    outputs, variables = class_predictor.init_with_output(
        jax.random.PRNGKey(0), img_emb, txt_emb, label_mask)
    self.assertEqual(outputs['pred_logits'].shape,
                     (batch_size, num_patches, num_queries))
    num_weights = jax.tree_util.tree_flatten(variables)[0]
    self.assertLen(num_weights, 6)

  def test_clip_image_text_embedder(self):
    """Tests image and text embedding with a CLIP model."""
    batch_size, num_queries, seq_len, img_size, patch_size = 8, 10, 15, 224, 32
    side_patches = img_size // patch_size
    embed_configs = ml_collections.ConfigDict(dict(
        type='clip',
        variant='vit_b32',
        merge_class_token='drop',
        text_stochastic_droplayer_rate=0.1,
        vision_stochastic_droplayer_rate=0.1,
        ))
    images = jnp.ones((batch_size, img_size, img_size, 3))
    texts = jnp.ones((batch_size, num_queries, seq_len), dtype=jnp.int32)
    rng = jax.random.PRNGKey(0)

    embedder = layers.ClipImageTextEmbedder(embed_configs)

    with self.subTest(name='images_and_text'):
      (img, txt), _ = embedder.init_with_output(rng, images=images, texts=texts)
      self.assertEqual(img.shape, (batch_size, side_patches**2, 768))
      self.assertEqual(txt.shape, (batch_size, num_queries, 512))

    with self.subTest(name='only_images'):
      (img, _), _ = embedder.init_with_output(rng, images=images, texts=None)
      self.assertEqual(img.shape, (batch_size, side_patches**2, 768))

    with self.subTest(name='only_text'):
      (_, txt), _ = embedder.init_with_output(rng, images=None, texts=texts)
      self.assertEqual(txt.shape, (batch_size, num_queries, 512))


def _num_layers_and_params(params):
  """Returns number of weight layers and total number of params in params."""
  leaves = jax.tree_util.tree_leaves(params)
  return len(leaves), sum([jnp.prod(jnp.array(v.shape)) for v in leaves])


if __name__ == '__main__':
  absltest.main()
