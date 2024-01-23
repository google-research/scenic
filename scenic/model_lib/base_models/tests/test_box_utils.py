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

"""Unit tests for functions in box_utils.py."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from scenic.model_lib.base_models import box_utils
from shapely import geometry


def sample_cxcywh_bbox(key, batch_shape):
  """Samples a bounding box in the [cx, cy, w, h] in [0, 1] range format."""
  frac = 0.8
  sample = jax.random.uniform(key, shape=(*batch_shape, 4)) * frac
  cx, cy, w, h = jnp.split(sample, indices_or_sections=4, axis=-1)
  # Make sure the bounding box doesn't cross the right and top image borders
  w = jnp.where(cx + w / 2. >= 1., frac * 2. * (1. - cx), w)
  h = jnp.where(cy + h / 2. >= 1., frac * 2. * (1. - cy), h)
  # Make sure the bounding box doesn't cross the left and bottom image borders
  w = jnp.where(cx - w / 2. <= 0., frac * 2. * cx, w)
  h = jnp.where(cy - h / 2. <= 0., frac * 2. * cy, h)

  bbox = jnp.concatenate([cx, cy, w, h], axis=-1)
  return bbox


class BoxUtilsTest(parameterized.TestCase):
  """Tests all the bounding box related utilities."""

  def test_box_cxcywh_to_xyxy(self):
    """Test for correctness of the box_cxcywh_to_xyxy operation."""
    cxcywh = jnp.array([[[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.2, 0.4]],
                        [[0.3, 0.2, 0.1, 0.4], [0.3, 0.2, 0.1, 0.4]]],
                       dtype=jnp.float32)
    expected = jnp.array([[[0.0, 0.1, 0.2, 0.5], [0.0, 0.1, 0.2, 0.5]],
                          [[0.25, 0.0, 0.35, 0.4], [0.25, 0.0, 0.35, 0.4]]],
                         dtype=jnp.float32)
    output = box_utils.box_cxcywh_to_xyxy(cxcywh)
    self.assertSequenceAlmostEqual(
        expected.flatten(), output.flatten(), places=5)

    # also test whether an exception is raised when a non-box input is provided
    with self.assertRaises(ValueError):
      cxcywh = jnp.array(np.random.uniform(size=(2, 3, 5)))
      _ = box_utils.box_cxcywh_to_xyxy(cxcywh)

  @parameterized.parameters([((3, 1, 4),), ((4, 6, 4),)])
  def test_box_cxcywh_to_xyxy_shape(self, input_shape):
    """Test whether the shape is correct for box_cxcywh_to_xyxy."""
    cxcywh = jnp.array(np.random.uniform(size=input_shape))
    xyxy = box_utils.box_cxcywh_to_xyxy(cxcywh)
    self.assertEqual(xyxy.shape, cxcywh.shape)

  @parameterized.parameters([((2, 5, 4),), ((1, 3, 4),)])
  def test_box_cxcy_to_xyxy_box_xyxy_to_cxcy(self, input_shape):
    """Test both box conversion functions as they are inverses of each other."""
    cxcywh = jnp.array(np.random.uniform(size=input_shape))
    xyxy = box_utils.box_cxcywh_to_xyxy(cxcywh)
    cxcywh_loop = box_utils.box_xyxy_to_cxcywh(xyxy)
    self.assertSequenceAlmostEqual(
        cxcywh_loop.flatten(), cxcywh.flatten(), places=5)


def sample_cxcywha(key, batch_shape):
  """Sample rotated bounding boxes [cx, cy, w, h, a (radians)]."""
  scale = jnp.array([0.3, 0.3, 0.5, 0.5, 1.0])
  offset = jnp.array([0.35, 0.35, 0, 0, 0])
  return jax.random.uniform(key, shape=(*batch_shape, 5)) * scale + offset


class RBoxUtilsTest(parameterized.TestCase):
  """Tests all the rotated bounding box related utilities."""

  def test_convert_cxcywha_to_corners(self):
    key = jax.random.PRNGKey(0)
    cxcywha = sample_cxcywha(key, batch_shape=(300, 200))
    self.assertEqual(cxcywha.shape, (300, 200, 5))

    corners = box_utils.cxcywha_to_corners(cxcywha)
    self.assertEqual(corners.shape, (300, 200, 4, 2))
    # This criteria depends on sample function sampling within unit square.
    self.assertTrue(jnp.all(corners >= 0))
    self.assertTrue(jnp.all(corners <= 1))

  def test_convert_corners_to_cxcywha(self):
    key = jax.random.PRNGKey(0)
    cxcywha = sample_cxcywha(key, batch_shape=(3, 2))
    self.assertEqual(cxcywha.shape, (3, 2, 5))

    corners = box_utils.cxcywha_to_corners(cxcywha)
    cxcywha2 = box_utils.corners_to_cxcywha(corners)
    np.testing.assert_allclose(cxcywha2, cxcywha, atol=1e-6)

  def test_convert_cxcywha_to_corners_single_rotated(self):
    cxcywha = jnp.array([1, 1, jnp.sqrt(2), jnp.sqrt(2), 45. * jnp.pi / 180.])
    corners = box_utils.cxcywha_to_corners(cxcywha)
    expected_corners = [[1, 0], [2, 1], [1, 2], [0, 1]]
    np.testing.assert_allclose(corners, expected_corners, atol=1e-7)

  def test_intersect_line_segments(self):
    """Test for correctness of the intersect_lines operation."""
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    lines1 = jax.random.uniform(subkey, (100, 2, 2))
    lines2 = jax.random.uniform(key, (100, 2, 2))
    intersect_line_segments = jax.jit(
        jax.vmap(box_utils.intersect_line_segments))
    intersections = intersect_line_segments(lines1, lines2)
    self.assertEqual(intersections.shape, (100, 2))

    expected_intersections = []
    for i in range(len(lines1)):
      line1 = geometry.LineString(lines1[i])
      line2 = geometry.LineString(lines2[i])
      it = line1.intersection(line2)
      it_coord = (
          it.coords[0]
          if isinstance(it, geometry.Point) else jnp.asarray([jnp.nan] * 2))
      expected_intersections.append(it_coord)

    np.testing.assert_allclose(intersections, expected_intersections, atol=1e-7)

  def test_intersect_rbox_edges_same_box(self):
    """Test for correctness of the intersect_rbox_edges operation."""
    rbox1 = jnp.array([0.5, 0.5, 1.0, 1.0, 0])
    rbox2 = rbox1
    corners1 = box_utils.cxcywha_to_corners(rbox1)
    corners2 = box_utils.cxcywha_to_corners(rbox2)
    it_points = box_utils.intersect_rbox_edges(corners1, corners2)
    self.assertEqual(it_points.shape, (4, 4, 2))
    it_points = it_points[~jnp.any(jnp.isnan(it_points), -1)]
    it_points = sorted([(x, y) for x, y in np.array(it_points)])
    expected_points = sorted([(0, 0), (0, 1), (1, 0), (1, 1)] * 2)
    self.assertSequenceEqual(it_points, expected_points)

  def test_intersect_rbox_edges_rotated_box(self):
    """Test rboxe inscribes the other with 45 degree angle."""
    rbox1 = jnp.array([1.0, 1.0, 1.0, 1.0, 0])
    rbox2 = jnp.array([1.0, 1.0, jnp.sqrt(2), jnp.sqrt(2), 45. * np.pi / 180.])
    corners1 = box_utils.cxcywha_to_corners(rbox1)
    corners2 = box_utils.cxcywha_to_corners(rbox2)
    it_points = box_utils.intersect_rbox_edges(corners1, corners2)
    it_points = jnp.round(
        it_points[~jnp.any(jnp.isnan(it_points), -1)], decimals=4)
    it_points = sorted([(x, y) for x, y in np.array(it_points)])
    # Expect intersection at unrotated box vertices.
    expected_pts = sorted([(1.5, 1.5), (1.5, 0.5), (0.5, 0.5), (0.5, 1.5)] * 2)
    self.assertSequenceEqual(it_points, expected_pts)


class IoUTest(parameterized.TestCase):
  """Test box_iou and generalized_box_iou functions."""

  def test_box_iou_values(self):
    """Tests if 0 <= IoU <= 1 and -1 <= gIoU <=1."""

    # Create fake predictions and targets
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    pred_bbox = sample_cxcywh_bbox(key, batch_shape=(4, 100))
    tgt_bbox = sample_cxcywh_bbox(subkey, batch_shape=(4, 63))

    pred_bbox = box_utils.box_cxcywh_to_xyxy(pred_bbox)
    tgt_bbox = box_utils.box_cxcywh_to_xyxy(tgt_bbox)

    iou, union = box_utils.box_iou(pred_bbox, tgt_bbox, all_pairs=True)
    self.assertTrue(jnp.all(iou >= 0))
    self.assertTrue(jnp.all(iou <= 1.))
    self.assertTrue(jnp.all(union >= 0.))

    giou = box_utils.generalized_box_iou(pred_bbox, tgt_bbox, all_pairs=True)
    self.assertTrue(jnp.all(giou >= -1.))
    self.assertTrue(jnp.all(giou <= 1.))

  def test_box_iou(self):
    """Test box_iou using hand designed targets."""
    in1 = jnp.array([
        [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.5, 1.0], [0.1, 0.2, 0.5, 0.8]],
        [[0.6, 0.2, 1.0, 1.0], [0.6, 0.2, 1.0, 0.8], [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0], [0.2, 0.1, 0.2, 0.1], [0.1, 0.1, 0.2, 0.2]],
    ],
                    dtype=jnp.float32)
    in2 = jnp.array([
        [[0.4, 0.4, 0.5, 0.8], [0.4, 0.4, 0.5, 0.8], [0.4, 0.4, 0.7, 0.8]],
        [[0.7, 0.4, 0.8, 0.6], [0.8, 0.6, 0.7, 0.4], [0.1, 0.1, 0.2, 0.2]],
        [[0.0, 0.0, 0.0, 0.0], [0.2, 0.1, 0.2, 0.1], [0.1, 0.1, 0.2, 0.2]],
    ],
                    dtype=jnp.float32)

    target = jnp.array(
        [[0.0, 0.125, 0.125], [0.0625, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float32)

    output, _ = box_utils.box_iou(in1, in2, all_pairs=False)

    self.assertSequenceAlmostEqual(output.flatten(), target.flatten(), places=3)

  @classmethod
  def _get_method_fn(cls, method):
    """Returns method_fn function corresponding to method str."""
    if method == 'iou':
      method_fn = lambda x, y, **kwargs: box_utils.box_iou(x, y, **kwargs)[0]
    elif method == 'giou':
      method_fn = box_utils.generalized_box_iou
    else:
      raise ValueError(f'Unknown method {method}')
    return method_fn

  @parameterized.parameters('iou', 'giou')
  def test_all_pairs_true_false(self, method):
    """Use *box_iou(..., all_pairs=False) to test the all_pairs=True case."""
    method_fn = self._get_method_fn(method)

    in1 = jnp.array(
        [  # [2, 2, 4] tensor.
            [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.5, 1.0]],
            [[0.6, 0.2, 1.0, 1.0], [0.6, 0.2, 1.0, 0.8]],
        ],
        dtype=jnp.float32)
    in2 = jnp.array([
        [[0.4, 0.4, 0.5, 0.8], [0.4, 0.4, 0.5, 0.8]],
        [[0.7, 0.4, 0.8, 0.6], [0.1, 0.5, 0.7, 0.7]],
    ],
                    dtype=jnp.float32)

    # we will simulate all_pairs=True by manually permuting in1
    in1_1 = jnp.array(
        [  # [2, 2, 4] tensor.
            [[0.1, 0.2, 0.5, 1.0], [0.1, 0.2, 0.3, 0.4]],
            [[0.6, 0.2, 1.0, 0.8], [0.6, 0.2, 1.0, 1.0]],
        ],
        dtype=jnp.float32)

    out = method_fn(in1, in2, all_pairs=False)  # [2, 2]
    out_1 = method_fn(in1_1, in2, all_pairs=False)  # [2, 2]

    # we can compare these against the output of all_pairs=True
    out_all = method_fn(in1, in2, all_pairs=True)  # [2, 2, 2]

    # assemble out_all_ using out and out_1. The comparisons are illustrated
    # below:
    # out     = [[0-0, 1-1], [2-2, 3-3]]
    # out_1   = [[1-0, 0-1], [3-2, 2-3]]
    # out_all = [[[0-0, 0-1], [1-0, 1-1]], [[2-2, 2-3], [3-2, 3-3]]]
    out_all_ = jnp.array([[[out[0, 0], out_1[0, 1]], [out_1[0, 0], out[0, 1]]],
                          [[out[1, 0], out_1[1, 1]], [out_1[1, 0], out[1, 1]]]],
                         dtype=jnp.float32)

    self.assertSequenceAlmostEqual(out_all.flatten(), out_all_.flatten())

  def test_generalized_box_iou(self):
    """Same as test_box_iou but for generalized_box_iou()."""
    in1 = jnp.array([
        [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.5, 1.0], [0.1, 0.2, 0.5, 0.8]],
        [[0.6, 0.2, 1.0, 1.0], [0.6, 0.2, 1.0, 0.8], [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0], [0.2, 0.1, 0.2, 0.1], [0.1, 0.1, 0.2, 0.2]],
    ],
                    dtype=jnp.float32)
    in2 = jnp.array([
        [[0.4, 0.4, 0.5, 0.8], [0.4, 0.4, 0.5, 0.8], [0.4, 0.4, 0.7, 0.8]],
        [[0.7, 0.4, 0.8, 0.6], [0.4, 0.4, 0.8, 0.6], [0.1, 0.1, 0.2, 0.2]],
        [[0.0, 0.0, 0.0, 0.0], [0.2, 0.1, 0.2, 0.1], [0.1, 0.1, 0.2, 0.2]],
    ],
                    dtype=jnp.float32)

    target_iou = jnp.array(
        [[0.0, 0.125, 0.125], [0.0625, 1. / 7., 0.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float32)
    target_extra = jnp.array(
        [[-2. / 3., 0.0, -1. / 9.], [0.0, -2. / 9., -3. / 4.], [0.0, 0.0, 0.0]],
        dtype=jnp.float32)
    target = target_iou + target_extra

    output = box_utils.generalized_box_iou(in1, in2, all_pairs=False)

    self.assertSequenceAlmostEqual(output.flatten(), target.flatten(), places=3)

    # if the boxes are invalid it should raise an AssertionError
    # TODO(b/166344282): uncomment these after enabling the assertions
    # in1 = jnp.array([[[0.1, 0.2, 0.3, 0.4],],], dtype=jnp.float32)
    # in2 = jnp.array([[[0.3, 0.4, 0.1, 0.2],],], dtype=jnp.float32)
    # with self.assertRaises(AssertionError):
    #   _ = box_utils.generalized_box_iou(in1, in2, all_pairs=False)

  @parameterized.parameters('iou', 'giou')
  def test_backward(self, method):
    """Test whether *box_iou methods have a grad."""
    method_fn = self._get_method_fn(method)

    def loss_fn(x, y, all_pairs):
      return method_fn(x, y, all_pairs=all_pairs).sum()

    grad_fn = jax.grad(loss_fn)

    in1 = jnp.array(
        [  # [2, 2, 4] tensor.
            [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.5, 1.0]],
            [[0.6, 0.2, 1.0, 1.0], [0.6, 0.2, 1.0, 0.8]],
        ],
        dtype=jnp.float32)
    in2 = jnp.array([
        [[0.4, 0.4, 0.5, 0.8], [0.4, 0.4, 0.5, 0.8]],
        [[0.7, 0.4, 0.8, 0.6], [0.1, 0.5, 0.7, 0.7]],
    ],
                    dtype=jnp.float32)

    grad_in1 = grad_fn(in1, in2, all_pairs=True)
    self.assertSequenceEqual(grad_in1.shape, in1.shape)

    grad_in1 = grad_fn(in1, in2, all_pairs=False)
    self.assertSequenceEqual(grad_in1.shape, in1.shape)


if __name__ == '__main__':
  absltest.main()
