"""Tests attention.py."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from scenic.projects.baselines.deformable_detr.attention import bilinear_interpolate
from scenic.projects.baselines.deformable_detr.attention import deform_attn_sampling_fn
from scenic.projects.baselines.deformable_detr.attention import MultiScaleDeformableAttention


class DeformAttnSamplingFnTest(parameterized.TestCase):
  """Tests for deform_attn_sampling_fn."""

  def test_deform_attn_sampling_fn_results(self):
    """Tests sampling function makes sense."""
    values = []
    im_shapes = []
    nheads = 1
    nlevels = 2

    # Add values for 3x3 at level 0.
    v = np.array([0., 0., 1.])[None, None, None, :]
    v = v.repeat(9, 1).repeat(nheads, 2)
    values.append(v)
    im_shapes.append((3, 3))

    # Add values for 1x1 at level 1.
    v = np.array([1., 0., 0.])[None, None, None, :]
    v = v.repeat(nheads, 2)
    values.append(v)
    im_shapes.append((1, 1))

    # 1 query per level each will sample in center of image.
    # [bs, len_q, nheads, nlevels, npoints, 2]
    sampling_locations = np.ones([1, 1, nheads, nlevels, 1, 2]) * 0.5

    # Attention weight is 2x for level0. [bs, len_q, nheads, nlevels, npoints]
    attn_weights = np.array([2., 1.])[None, None, None, :, None]
    attn_weights = attn_weights.repeat(nheads, 2)

    # [bs, len_v, nheads, nembed]
    values = np.concatenate(values, axis=1)

    # Shapes must be static.
    shapes = tuple([tuple(s) for s in im_shapes])
    out = deform_attn_sampling_fn(values, sampling_locations, attn_weights,
                                  shapes, True)
    self.assertSequenceEqual(out.shape, [1, 1, 3])
    self.assertSequenceAlmostEqual(out.reshape(-1).tolist(), [1, 0, 2])

  @parameterized.named_parameters(
      ('zero_center', True),
      ('no_zero_center', False),
  )
  def test_bilinear_interp_single_pixel(self, zero_center: bool):
    """Tests matches results from pytorch custom CUDA kernel."""
    im = np.ones((1, 1, 1))
    grid = np.array([[[0, 0], [0.5, 0.5], [1, 1]]])
    res = bilinear_interpolate(im, grid, zero_center=zero_center)
    self.assertSequenceEqual(res.shape, (1, 1, 3))
    res = res.reshape(-1)

    if zero_center:
      # (0, 0) maps to pixel center, (0.5, 0.5) is bottom-right, (1, 1) is out.
      exp_res = [1, 0.25, 0]
    else:
      # (0, 0) maps to top-left, (0.5, 0.5) is center, (1, 1) is bottom-right.
      exp_res = [0.25, 1, 0.25]
    self.assertSequenceAlmostEqual(res, exp_res)

  @parameterized.named_parameters(
      ('zero_center', True),
      ('no_zero_center', False),
  )
  def test_bilinear_interp_tiny_image(self, zero_center: bool):
    """Tests makes sense on tiny adhoc image."""
    im = np.arange(8).reshape(1, 4, 2)
    grid = np.array([[0.25, 0.25], [0.25, 0.5], [0.25, 0.75]])
    res = bilinear_interpolate(im, grid, zero_center=zero_center)
    self.assertSequenceEqual(res.shape, (1, 3))
    res = res.reshape(-1)

    if zero_center:
      # (0, 0) maps to pixel center, (0.5, 0.5) is bottom-right, (1, 1) is out.
      exp_res = [2.5, 4.5, 6.5]
    else:
      # (0, 0) maps to top-left, (0.5, 0.5) is center, (1, 1) is bottom-right.
      exp_res = [1, 3, 5]
    self.assertSequenceAlmostEqual(res, exp_res)

  def test_bilinear_interp_out_of_bounds(self):
    """Tests handling of out-of-bounds with zero-padding."""
    im = np.ones((1, 1, 10)).astype(float)
    # Center rightmost pixel, edge rightmost pixel, way out of bounds.
    grid = np.array([[0.95, 0.5], [1.0, 0.5], [1.5, 0.5]])
    res = bilinear_interpolate(im, grid, False)
    self.assertSequenceEqual(res.shape, (1, 3))
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
        spatial_shapes=shapes)

    out, init_params = model.init_with_output(rng, query, ref_points, value,
                                              pad_mask, True)
    self.assertSequenceEqual(out.shape, (bs, len_q, embed_dim))

    # Can jit.
    run = jax.jit(model.apply, static_argnums=5)
    out2 = run(init_params, query, ref_points, value, pad_mask, True)
    self.assertSequenceEqual(out2.shape, (bs, len_q, embed_dim))


if __name__ == '__main__':
  absltest.main()
