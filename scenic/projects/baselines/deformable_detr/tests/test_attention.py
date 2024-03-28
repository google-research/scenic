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

"""Tests attention.py."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.projects.baselines.deformable_detr.attention import bilinear_interpolate
from scenic.projects.baselines.deformable_detr.attention import MultiScaleDeformableAttention


class DeformAttnSamplingFnTest(parameterized.TestCase):
  """Tests for deform_attn_sampling_fn."""

  def test_bilinear_interp_single_pixel(self):
    """Tests matches results from pytorch custom CUDA kernel."""
    im = np.ones((1, 1, 1))
    grid = np.array([[[0, 0], [0.5, 0.5], [1, 1]]])
    res = bilinear_interpolate(im, grid, 1, 1)
    self.assertSequenceEqual(res.shape, (1, 3, 1))
    res = res.reshape(-1)

    # (0, 0) maps to top-left, (0.5, 0.5) is center, (1, 1) is bottom-right.
    exp_res = [0.25, 1, 0.25]
    self.assertSequenceAlmostEqual(res, exp_res)

  def test_bilinear_interp_tiny_image(self):
    """Tests makes sense on tiny adhoc image."""
    im = np.arange(8).reshape(4, 2, 1)
    grid = np.array([[0.25, 0.25], [0.25, 0.5], [0.25, 0.75]])
    res = bilinear_interpolate(im, grid, 2, 4)
    self.assertSequenceEqual(res.shape, (3, 1))
    res = res.reshape(-1)

    # (0, 0) maps to top-left, (0.5, 0.5) is center, (1, 1) is bottom-right.
    exp_res = [1, 3, 5]
    self.assertSequenceAlmostEqual(res, exp_res)

  def test_bilinear_interp_out_of_bounds(self):
    """Tests handling of out-of-bounds with zero-padding."""
    im = np.ones((1, 10, 1)).astype(float)
    # Center rightmost pixel, edge rightmost pixel, way out of bounds.
    grid = np.array([[0.95, 0.5], [1.0, 0.5], [1.5, 0.5]])
    res = bilinear_interpolate(im, grid, 10, 1)
    self.assertSequenceEqual(res.shape, (3, 1))
    res = res.reshape(-1).tolist()
    self.assertSequenceEqual(res, [1., 0.5, 0.])


class MultiScaleDeformableAttentionTest(parameterized.TestCase):
  """Tests for MultiScaleDeformableAttention."""

  @parameterized.named_parameters(
      {
          'testcase_name': '2D Ref Points',
          'ref_dim': 2,
          'shapes': ((10, 10), (2, 2)),
          'embed_dim': 8,
          'num_heads': 4
      }, {
          'testcase_name': '4D Ref Points',
          'ref_dim': 4,
          'shapes': ((10, 10), (2, 2)),
          'embed_dim': 8,
          'num_heads': 4
      }, {
          'testcase_name': 'One level',
          'ref_dim': 2,
          'shapes': ((2, 2),),
          'embed_dim': 8,
          'num_heads': 4
      }, {
          'testcase_name': 'NumHeads == EmbedDim',
          'ref_dim': 2,
          'shapes': ((2, 2),),
          'embed_dim': 8,
          'num_heads': 8
      })
  def test_ms_deformable_attn_output_shape(self, ref_dim, shapes, embed_dim,
                                           num_heads):
    """Test MultiScaleDeformableAttention output shape."""
    rng = random.PRNGKey(8877)
    # Setup remaining size params.
    bs, len_q, num_points, num_levels = 2, 10, 1, len(shapes)
    len_v = np.array(shapes).prod(axis=-1).sum()

    # Set inputs.
    query = jnp.array(np.random.normal(size=(bs, len_q, embed_dim)))
    ref_points = jnp.array(
        np.random.normal(size=(bs, len_q, num_levels, ref_dim)))
    value = jnp.array(np.random.normal(size=(bs, len_v, embed_dim)))
    pad_mask = jnp.zeros(value.shape[:-1], dtype=bool)

    # Compute.
    model = MultiScaleDeformableAttention(
        embed_dim=embed_dim,
        num_levels=num_levels,
        num_heads=num_heads,
        num_points=num_points,
        spatial_shapes=shapes,
        compiler_config=ml_collections.ConfigDict(
            dict(
                train_remat=True,
                eval_remat=False,
                attention_batching_mode='auto',
            )),
        dtype=jnp.float32,
    )

    out, init_params = model.init_with_output(rng, query, ref_points, value,
                                              pad_mask, True)
    self.assertSequenceEqual(out.shape, (bs, len_q, embed_dim))

    # Can jit.
    run = jax.jit(model.apply, static_argnums=5)
    out2 = run(init_params, query, ref_points, value, pad_mask, True)
    self.assertSequenceEqual(out2.shape, (bs, len_q, embed_dim))


if __name__ == '__main__':
  absltest.main()
