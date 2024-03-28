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

"""Unit tests for transforms.py."""

import copy

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

from scenic.projects.baselines.detr import transforms

import tensorflow as tf

# TODO(aravindhm): Add unit tests for masks if doing panoptic segmentation.


def fake_boxes(n, h, w):
  """Sample boxes in [x, y, x, y] format in un-normalized image coordinates."""
  h, w = float(h), float(w)

  # samples boxes in [0, 1]
  boxes = tf.random.uniform((n, 4), minval=0.0, maxval=1.0, dtype=tf.float32)

  # ensures that x0 < x1, y0 < y1, and un-normalizes boxes
  x0 = tf.minimum(boxes[:, 0], boxes[:, 2]) * h
  x1 = tf.maximum(boxes[:, 0], boxes[:, 2]) * w
  y0 = tf.minimum(boxes[:, 1], boxes[:, 3]) * h
  y1 = tf.maximum(boxes[:, 1], boxes[:, 3]) * w

  area = (x1 - x0) * (y1 - y0)

  return tf.stack([x0, y0, x1, y1], axis=-1), area


def fake_decoded_features(n, h, w):
  """Create a fake features dictionary.

  Args:
    n: int; number of boxes/objects.
    h: int; height of fake image.
    w: int; width of fake image.

  Returns:
    Features dictionary with fields -
      'inputs': tf.Tensor; float32 [H x W x 3] decoded mean-subtracted image.
      'label': dict; target labels dict with keys -
        'boxes': tf.Tensor; float32 [n-objects x 4] in xyxy unnormalized format.
        'labels': tf.Tensor; int32 labels.
        'area': tf.Tensor; float32 [n-objects,] object box area.
        'is_crowd': tf.Tensor; bool [n-objects,] object is crowd type or not.
        'objects/id': tf.Tensor; int32 [n-objects,] object identifier.
  """
  inputs = tf.random.normal((h, w, 3), dtype=tf.float32)
  boxes, area = fake_boxes(n, h, w)

  is_crowd = [False for _ in range(n)]
  is_crowd[-1] = True

  label = {
      'boxes': boxes,
      'labels': tf.range(7),
      'area': area,
      'is_crowd': tf.constant(is_crowd, dtype=tf.bool),
      'objects/id': tf.random.uniform([n,], 0, 1024, dtype=tf.int32),
  }
  return {'inputs': inputs, 'label': label}


class TransformsTestBase(parameterized.TestCase):
  """A baseclass for testing transforms."""

  def _assert_tensors_equal(self, v1, v2, msg=''):
    """A wrapper that picks assertion based on tensor dtype."""
    self.assertEqual(v1.dtype, v2.dtype, msg=f'{msg}: Dtype mismatch')

    if v1.dtype == tf.bool or v1.dtype == tf.int32 or v1.dtype == tf.int64:
      # exact match for these data types
      self.assertTrue(np.all(np.equal(v1, v2)), msg=msg)
    elif v1.dtype == tf.float32 or v1.dtype == tf.float64:
      # approximate match for all others
      np.testing.assert_allclose(
          v1, v2, equal_nan=False, err_msg=msg, rtol=5e-5)
    else:
      raise NotImplementedError(f'{v1.dtype} type not supported.')

  def _assert_features_equal(self, f1, f2, msg=''):
    """Assert whether two feature dicts are equal."""
    self.assertSequenceEqual(list(f1.keys()), list(f2.keys()))
    for k, v in f1.items():
      if isinstance(v, tf.Tensor):
        self._assert_tensors_equal(v, f2[k], msg=f'{msg}[{k}]')
      elif isinstance(v, dict):
        self._assert_features_equal(v, f2[k], msg=f'{msg}[{k}]')
      else:
        # some other object type ... leave it to tensorflow and cross fingers
        self.assertEqual(v, f2[k], msg=f'{msg}[{k}]')


class RandomHorizontalFlipTest(TransformsTestBase):
  """Unit tests for RandomHorizontalFlip."""

  @parameterized.parameters([(7, 5, 4), (2, 4, 6)])
  def test_hflip_twice(self, n, h, w):
    """Tests hflip function by applying it twice and matching with original."""
    features = fake_decoded_features(n, h, w)

    features_copy = copy.deepcopy(features)
    features_flip = transforms.hflip(features_copy)
    features_recon = transforms.hflip(features_flip)

    self._assert_features_equal(features, features_recon,
                                msg='flip_twice mismatch at features')

  def test_hflip(self):
    """Tests hflip function by checking actual values."""
    features = fake_decoded_features(7, 5, 4)  # hardcoded `expected` for [5, 4]
    features['label']['boxes'] = tf.constant(
        [[1.0, 3.0, 3.0, 4.0],
         [2.0, 0.0, 3.0, 3.0]], dtype=tf.float32)
    expected = copy.deepcopy(features)
    # the following assumes cx-cy-h-w format to construct target.
    expected['inputs'] = features['inputs'][:, ::-1, :]
    expected['label']['boxes'] = tf.constant(
        [[1.0, 3.0, 3.0, 4.0],
         [1.0, 0.0, 2.0, 3.0]], dtype=tf.float32)

    features_flip = transforms.hflip(features)

    self._assert_features_equal(features_flip, expected,
                                msg='test_hflip')


class RandomResizeTest(TransformsTestBase):
  """Unit test RandomResize."""

  @parameterized.parameters([(8, None, [10, 8, 3]), (12, 10, [10, 8, 3])])
  def test_resize_shape(self, size, max_size, expected_shape):
    """Test whether resize produces the correct output shape."""
    features = fake_decoded_features(7, 5, 4)
    features_resized = transforms.resize(features, size, max_size=max_size)
    self.assertSequenceEqual(features_resized['inputs'].shape, expected_shape)

  def test_resize_transform(self):
    """Test Resize transform by checking shapes, box coordinates, area."""
    features = fake_decoded_features(7, 5, 4)
    features['label']['boxes'] = tf.constant(
        [[1.0, 3.0, 3.0, 4.0],
         [2.0, 0.0, 3.0, 3.0]], dtype=tf.float32)
    features['label']['area'] = tf.constant([2.0, 3.0], dtype=tf.float32)

    transform = transforms.Resize(12, max_size=10)
    features_resized = transform(features)

    # checks resized shape
    self.assertSequenceEqual(features_resized['inputs'].shape, [10, 8, 3])

    # checks resized boxes
    expected_boxes = np.array([[2.0, 6.0, 6.0, 8.0],
                               [4.0, 0.0, 6.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(features_resized['label']['boxes'].numpy(),
                               expected_boxes)

    # checks resized object area
    expected_areas = np.array([8.0, 12.0], dtype=np.float32)
    np.testing.assert_allclose(features_resized['label']['area'].numpy(),
                               expected_areas)

  def test_resize_randomresize(self):
    """Test consistency between Resize and RandomResize."""
    features = fake_decoded_features(7, 5, 4)
    transform_resize = transforms.Resize(10, max_size=11)
    transform_rnd_resize = transforms.RandomResize([10,], max_size=11)

    # resizes features using two different `Transforms` that are mathematically
    # identical
    features_resized = transform_resize(copy.deepcopy(features))
    features_rnd_resized = transform_rnd_resize(copy.deepcopy(features))

    # asserts that the outputs match
    self._assert_features_equal(features_resized, features_rnd_resized,
                                msg='resized vs random resized')

    # checks that they do not match had the parameter been different
    transform_rnd_resize = transforms.RandomResize([10,], max_size=None)
    features_rnd_resized = transform_rnd_resize(copy.deepcopy(features))
    with self.assertRaises(AssertionError):
      self._assert_features_equal(features_resized, features_rnd_resized,
                                  msg='resized vs random resized')


class NormalizeBoxesTest(TransformsTestBase):
  """Unit test NormalizeBoxes."""

  def test_normalize_boxes(self):
    """Numerically test whether boxes are being normalized correctly."""
    features = fake_decoded_features(2, 4, 8)
    features['label']['boxes'] = tf.constant(
        [[0.5, 0.5, 3.5, 2.5],
         [4.5, 1.5, 5.5, 3.5]], dtype=tf.float32)
    # area does not matter as area remains unnormalized
    del features['label']['area']

    expected = copy.deepcopy(features)
    expected['label']['boxes'] = tf.constant(
        [[0.25, 0.375, 0.375, 0.5],
         [0.625, 0.625, 0.125, 0.5]], dtype=tf.float32)

    features_normalized = transforms.NormalizeBoxes()(features)

    self._assert_features_equal(features_normalized, expected,
                                msg='Test normalization')


class InitPaddingMaskTest(TransformsTestBase):
  """Unit test for InitPaddingMask."""

  @parameterized.parameters([(4, 5, 3), (2, 4, 7)])
  def test_padding_mask(self, n, h, w):
    """Testing whether padding mask is currently initialized."""
    features = fake_decoded_features(n, h, w)

    expected = copy.deepcopy(features)
    expected['padding_mask'] = tf.ones((h, w), dtype=tf.float32)

    features_with_padding_mask = transforms.InitPaddingMask()(features)

    self._assert_features_equal(features_with_padding_mask, expected,
                                msg='Test padding mask')


class RandomSizeCrop(TransformsTestBase):
  """Unit test for RandomSizeCrop."""

  @parameterized.parameters([((1, 2, 3, 2), [3, 2, 3]),
                             ((2, 3, 1, 1), [1, 1, 3]),
                             ((0, 1, 5, 3), [5, 3, 3]),
                             ((1, 0, 4, 4), [4, 4, 3])])
  def test_crop_shape(self, region, expected_shape):
    """Test whether resize produces the correct output shape."""
    features = fake_decoded_features(7, 5, 4)
    features_resized = transforms.crop(features, region)
    self.assertSequenceEqual(features_resized['inputs'].shape, expected_shape)


if __name__ == '__main__':
  absltest.main()
