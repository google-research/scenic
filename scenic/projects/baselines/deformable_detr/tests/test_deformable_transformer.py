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

"""Tests for deformable_transformer.py."""

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.projects.baselines.deformable_detr.deformable_transformer import BBoxCoordPredictor
from scenic.projects.baselines.deformable_detr.deformable_transformer import DeformableDETRDecoder
from scenic.projects.baselines.deformable_detr.deformable_transformer import DeformableDETRDecoderLayer
from scenic.projects.baselines.deformable_detr.deformable_transformer import DeformableDETREncoder
from scenic.projects.baselines.deformable_detr.deformable_transformer import DeformableDETREncoderLayer
from scenic.projects.baselines.deformable_detr.deformable_transformer import DeformableDETRTransformer
from scenic.projects.baselines.deformable_detr.deformable_transformer import get_encoder_reference_points
from scenic.projects.baselines.deformable_detr.deformable_transformer import get_mask_valid_ratio
from scenic.projects.baselines.deformable_detr.deformable_transformer import inverse_sigmoid

compiler_config = ml_collections.ConfigDict(
    dict(
        train_remat=True,
        eval_remat=False,
        attention_batching_mode='auto',
    ))


class DeformableDETREncoderLayerTest(parameterized.TestCase):
  """Tests for DeformableDETREncoderLayer."""

  @parameterized.named_parameters(
      {
          'testcase_name': 'Multi level',
          'shapes': ((10, 10), (2, 2)),
          'embed_dim': 8,
          'num_heads': 4,
      }, {
          'testcase_name': 'Single level',
          'shapes': ((2, 2),),
          'embed_dim': 8,
          'num_heads': 4,
      })
  def test_encoder_layer_output_shape(self, shapes, embed_dim, num_heads):
    """Test DeformableDETREncoderLayer output shape."""
    rng = random.PRNGKey(8877)

    # Setup size params.
    len_qkv = np.array(shapes).prod(axis=-1).sum()
    bs, num_levels, ref_dim = 2, len(shapes), 2
    src = jnp.array(np.random.normal(size=(bs, len_qkv, embed_dim)))
    ref_points = jnp.array(
        np.random.normal(size=(bs, len_qkv, num_levels, ref_dim)))
    pad_mask = np.zeros(src.shape[:-1], dtype=bool)
    pos_embed = jnp.array(np.random.normal(size=src.shape))

    # Compute.
    model = DeformableDETREncoderLayer(
        spatial_shapes=shapes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_reference_points=1,
        dropout=0.1,
        ffn_dim=16,
        compiler_config=compiler_config,
    )

    out, init_params = model.init_with_output(rng, src, pos_embed, ref_points,
                                              pad_mask, False)
    self.assertSequenceEqual(out.shape, (bs, len_qkv, embed_dim))

    # Can jit.
    run = jax.jit(model.apply, static_argnums=5)
    out2 = run(init_params, src, pos_embed, ref_points, pad_mask, False)
    self.assertSequenceEqual(out2.shape, (bs, len_qkv, embed_dim))


class DeformableDETREncoderTest(parameterized.TestCase):
  """Tests for DeformableDETREncoder."""

  @parameterized.named_parameters(
      {
          'testcase_name': 'Multi layer',
          'num_layers': 4,
          'shapes': ((10, 10), (2, 2)),
      }, {
          'testcase_name': 'Single layer',
          'num_layers': 1,
          'shapes': ((10, 10), (2, 2)),
      })
  def test_encoder_layer_output_shape(self, num_layers, shapes):
    """Test DeformableDETREncoderLayer output shape."""
    rng = random.PRNGKey(8877)

    # Setup size params.
    embed_dim, num_heads = 8, 2
    len_qkv = np.array(shapes).prod(axis=-1).sum()
    bs, num_levels, ref_dim = 2, len(shapes), 2
    src = jnp.array(np.random.normal(size=(bs, len_qkv, embed_dim)))
    ref_points = jnp.array(
        np.random.normal(size=(bs, len_qkv, num_levels, ref_dim)))
    pad_mask = np.zeros(src.shape[:-1], dtype=bool)
    pos_embed = jnp.array(np.random.normal(size=src.shape))

    # Compute.
    model = DeformableDETREncoder(
        spatial_shapes=shapes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_layers=num_layers,
        num_reference_points=1,
        ffn_dim=8,
        dropout=0.1,
        compiler_config=compiler_config,
    )

    out, init_params = model.init_with_output(rng, src, pos_embed, ref_points,
                                              pad_mask, False)
    self.assertSequenceEqual(out.shape, (bs, len_qkv, embed_dim))

    # Can jit.
    run = jax.jit(model.apply, static_argnums=5)
    out2 = run(init_params, src, pos_embed, ref_points, pad_mask, False)
    self.assertSequenceEqual(out2.shape, (bs, len_qkv, embed_dim))


class DeformableDETRDecoderLayerTest(parameterized.TestCase):
  """Tests for DeformableDETRDecoderLayer."""

  @parameterized.named_parameters(
      {
          'testcase_name': 'Multi level',
          'num_heads': 4,
          'shapes': ((10, 10), (2, 2)),
      }, {
          'testcase_name': 'Single level',
          'num_heads': 4,
          'shapes': ((2, 2),),
      })
  def test_decoder_layer_output_shape(self, shapes, num_heads):
    """Test DeformableDETRDecoderLayer output shape."""
    rng = random.PRNGKey(8877)
    bs, len_q, embed_dim = 4, 16, 32
    len_v = np.array(shapes).prod(1).sum()
    num_levels = len(shapes)

    query = jnp.ones((bs, len_q, embed_dim), dtype=jnp.float32)
    query_pos = jnp.ones_like(query)
    ref_points = jnp.zeros((bs, len_q, num_levels, 4))
    value = jnp.ones((bs, len_v, 8), dtype=jnp.float32)
    pad_mask = jnp.ones((bs, len_v), dtype=bool)

    model = DeformableDETRDecoderLayer(
        spatial_shapes=shapes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_reference_points=1,
        ffn_dim=32,
        dropout=0.1,
        compiler_config=compiler_config,
    )

    params = model.init(rng, query, query_pos, ref_points, value, pad_mask,
                        False)
    apply = jax.jit(model.apply, static_argnums=6)
    out = apply(params, query, query_pos, ref_points, value, pad_mask, False)
    self.assertEqual(out.shape, (bs, len_q, embed_dim))


class DeformableDETRDecoderTest(parameterized.TestCase):
  """Tests for DeformableDETRDecoder."""

  def test_inverse_sigmoid(self):
    x = jnp.array(np.random.normal(size=(2, 4, 8)))
    y = inverse_sigmoid(nn.sigmoid(x))
    self.assertSequenceAlmostEqual(x.reshape(-1), y.reshape(-1), delta=1e-5)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Multi level, ref as points',
          'ref_dim': 2,
          'shapes': ((10, 10), (2, 2)),
      }, {
          'testcase_name': 'Single level, ref as boxes',
          'ref_dim': 4,
          'shapes': ((2, 2),),
      })
  def test_decoder_output_shape(self, shapes, ref_dim):
    """Test DeformableDETRDecoder output shape."""
    rng = random.PRNGKey(8877)
    bs, len_q, embed_dim, num_heads = 3, 16, 32, 4
    num_layers = 5
    len_v = np.array(shapes).prod(1).sum()
    num_levels = len(shapes)

    query = jnp.ones((bs, len_q, embed_dim), dtype=jnp.float32)
    query_pos = jnp.ones_like(query)
    ref_points = jnp.zeros((bs, len_q, ref_dim))
    value = jnp.ones((bs, len_v, 8), dtype=jnp.float32)
    pad_mask = jnp.ones((bs, len_v), dtype=bool)
    valid_ratios = jnp.ones((bs, num_levels, 2), dtype=jnp.float32)

    bbox_embeds = [
        BBoxCoordPredictor(mlp_dim=embed_dim, num_layers=3, use_sigmoid=False)
        for _ in range(num_layers)
    ]

    model = DeformableDETRDecoder(
        spatial_shapes=shapes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_levels=num_levels,
        num_reference_points=1,
        bbox_embeds=bbox_embeds,
        ffn_dim=16,
        dropout=0.1,
        compiler_config=compiler_config,
    )

    dec_input = dict(
        query=query,
        query_pos=query_pos,
        ref_points=ref_points,
        value=value,
        valid_ratios=valid_ratios,
        pad_mask=pad_mask,
        train=False)

    params = model.init(rng, **dec_input)
    apply = jax.jit(model.apply, static_argnames='train')
    out, out_boxes = apply(params, **dec_input)
    self.assertEqual(out.shape, (num_layers, bs, len_q, embed_dim))
    self.assertEqual(out_boxes.shape, (num_layers, bs, len_q, 4))


class DeformableDETRTransformerTest(parameterized.TestCase):
  """Tests for DeformableDETRTransformer and utilities."""

  def test_get_mask_valid_ratio(self):
    """Test get_mask_valid_ratio."""
    shapes = ((10, 15), (6, 2), (30, 30))
    # Make masks of each shape, where only first pixel is not-padding.
    max_d = 30

    def create_mask(h, w):
      pad = ((0, max_d - h), (0, max_d - w))
      m = np.ones((h, w), dtype=bool)
      return np.pad(m, pad, 'constant', constant_values=False)

    masks = np.stack([create_mask(h, w) for w, h in shapes], 0)

    out = get_mask_valid_ratio(masks)
    exp_ratios = np.array(shapes, dtype=float) / max_d
    self.assertSequenceAlmostEqual(exp_ratios.flatten(), out.flatten())

  def test_get_encoder_reference_points(self):
    """Test get_encoder_reference_points."""
    shapes = ((10, 10), (2, 2), (30, 30))
    bs, nlevels = 3, len(shapes)
    valid_ratios = jnp.array([1., .5])[None, None, :] * jnp.ones(
        (bs, nlevels, 2), dtype=jnp.float32)
    ref_points = get_encoder_reference_points(shapes, valid_ratios)
    flat_shape = np.array(shapes).prod(1).sum()

    self.assertSequenceEqual(ref_points.shape, (bs, flat_shape, nlevels, 2))
    delta = 0.5
    self.assertSequenceAlmostEqual(ref_points[0, -1, 0], [1. - delta / 30.] * 2)
    self.assertSequenceAlmostEqual(ref_points[0, 0, 0], [delta / 10.] * 2)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Three decoder layers',
          'num_dec_layers': 3,
          'shapes': ((10, 10), (2, 2)),
      }, {
          'testcase_name': 'One decoder layer',
          'shapes': ((10, 10), (2, 2)),
          'num_dec_layers': 1,
      }, {
          'testcase_name': 'Single scale, single decoder layer',
          'shapes': ((2, 2),),
          'num_dec_layers': 1,
      })
  def test_transformer_output_shape(self, shapes, num_dec_layers):
    """Test DeformableDETRTransformer output shape."""
    rng = random.PRNGKey(8877)

    bs, num_heads, num_queries = 2, 3, 5
    embed_dim = 2 * num_heads
    inputs = []
    pad_masks = []
    pos_embeds = []
    for h, w in shapes:
      inputs.append(jnp.ones((bs, h, w, embed_dim), dtype=jnp.float32))
      pad_masks.append(jnp.ones((bs, h, w), dtype=jnp.bool_))
      pos_embeds.append(jnp.ones((bs, h * w, embed_dim), dtype=jnp.float32))

    bbox_embeds = [
        BBoxCoordPredictor(mlp_dim=embed_dim, num_layers=3, use_sigmoid=False)
        for _ in range(num_dec_layers)
    ]

    # Compute.
    model = DeformableDETRTransformer(
        enc_embed_dim=embed_dim,
        embed_dim=embed_dim,
        num_queries=num_queries,
        num_heads=num_heads,
        num_dec_layers=num_dec_layers,
        num_enc_layers=2,
        ffn_dim=8,
        bbox_embeds=bbox_embeds,
        num_enc_points=1,
        num_dec_points=2,
        dropout=0.1,
        compiler_config=compiler_config,
    )

    (out, ref_boxes, init_ref_points), init_params = model.init_with_output(
        rng, inputs, pad_masks, pos_embeds, False)

    self.assertSequenceEqual(out.shape,
                             (num_dec_layers, bs, num_queries, embed_dim))
    self.assertSequenceEqual(ref_boxes.shape,
                             (num_dec_layers, bs, num_queries, 4))
    self.assertSequenceEqual(init_ref_points.shape, (bs, num_queries, 2))

    # Can jit.
    run = jax.jit(model.apply, static_argnums=4)
    out, ref_boxes, init_ref_points = run(init_params, inputs, pad_masks,
                                          pos_embeds, False)
    self.assertSequenceEqual(out.shape,
                             (num_dec_layers, bs, num_queries, embed_dim))
    self.assertSequenceEqual(ref_boxes.shape,
                             (num_dec_layers, bs, num_queries, 4))
    self.assertSequenceEqual(init_ref_points.shape, (bs, num_queries, 2))


if __name__ == '__main__':
  absltest.main()
