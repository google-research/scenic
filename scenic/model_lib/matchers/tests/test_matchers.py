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

"""Unit tests for functions in matchers."""


from typing import Optional

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from scenic.model_lib import matchers
from scenic.model_lib.base_models import box_utils
import scipy.optimize as sciopt


MATCHER_FUNCTIONS = {
    'hungarian': matchers.hungarian_matcher,
    'hungarian_tpu': matchers.hungarian_tpu_matcher,
    'hungarian_scan_tpu': matchers.hungarian_scan_tpu_matcher,
    'sinkhorn': matchers.sinkhorn_matcher,
    'greedy': matchers.greedy_matcher,
    'lazy': matchers.lazy_matcher,
    'hungarian_cover_tpu': matchers.hungarian_cover_tpu_matcher
}
EXACT_MATCHERS = ['hungarian', 'hungarian_tpu', 'hungarian_scan_tpu',
                  'hungarian_cover_tpu']
RECT_MATCHERS = ['hungarian', 'hungarian_tpu', 'hungarian_scan_tpu',
                 'hungarian_cover_tpu']
CPU_MATCHERS = ['hungarian']

EPS = 1e-4


def compute_cost(
    *,
    tgt_labels: jnp.ndarray,
    out_prob: jnp.ndarray,
    tgt_bbox: Optional[jnp.ndarray] = None,
    out_bbox: Optional[jnp.ndarray] = None,
    class_loss_coef: float,
    bbox_loss_coef: Optional[float] = None,
    giou_loss_coef: Optional[float] = None,
    target_is_onehot: bool,
) -> jnp.ndarray:
  """Computes cost matrices for a batch of predictions.

  Relevant code:
  https://github.com/facebookresearch/detr/blob/647917626d5017e63c1217b99537deb2dcb370d6/models/matcher.py#L35

  Args:
    tgt_labels: Class labels of shape [B, M]. If target_is_onehot then it is [B,
      M, C]. Note that the labels corresponding to empty bounding boxes are not
      yet supposed to be filtered out.
    out_prob: Classification probabilities of shape [B, N, C].
    tgt_bbox: Target box coordinates of shape [B, M, 4]. Note that the empty
      bounding boxes are not yet supposed to be filtered out.
    out_bbox: Predicted box coordinates of shape [B, N, 4]
    class_loss_coef: Relative weight of classification loss.
    bbox_loss_coef: Relative weight of bbox loss.
    giou_loss_coef: Relative weight of giou loss.
    target_is_onehot: boolean; Whether targets are one-hot encoded.

  Returns:
    A cost matrix [B, N, M].
  """
  if (tgt_bbox is None) != (out_bbox is None):
    raise ValueError('Both `tgt_bbox` and `out_bbox` must be set.')
  if (tgt_bbox is not None) and ((bbox_loss_coef is None) or
                                 (giou_loss_coef is None)):
    raise ValueError('For detection, both `bbox_loss_coef` and `giou_loss_coef`'
                     ' must be set.')

  batch_size, max_num_boxes = tgt_labels.shape[:2]
  num_queries = out_prob.shape[1]
  if target_is_onehot:
    mask = tgt_labels[..., 0] == 0  # [B, M]
  else:
    mask = tgt_labels != 0  # [B, M]

  # [B, N, M]
  cost_class = -out_prob  # DETR uses -prob for matching.
  max_cost_class = 0.0

  # [B, N, M]
  if target_is_onehot:
    cost_class = jnp.einsum('bnl,bml->bnm', cost_class, tgt_labels)
  else:
    cost_class = jax.vmap(jnp.take, (0, 0, None))(cost_class, tgt_labels, 1)

  cost = class_loss_coef * cost_class
  cost_upper_bound = max_cost_class

  if out_bbox is not None:
    # [B, N, M, 4]
    diff = jnp.abs(out_bbox[:, :, None] - tgt_bbox[:, None, :])
    cost_bbox = jnp.sum(diff, axis=-1)  # [B, N, M]
    cost = cost + bbox_loss_coef * cost_bbox

    # Cost_upper_bound is the approximate maximal possible total cost:
    cost_upper_bound = cost_upper_bound + bbox_loss_coef * 4.0  # cost_bbox <= 4

    # [B, N, M]
    cost_giou = -box_utils.generalized_box_iou(
        box_utils.box_cxcywh_to_xyxy(out_bbox),
        box_utils.box_cxcywh_to_xyxy(tgt_bbox),
        all_pairs=True)
    cost = cost + giou_loss_coef * cost_giou

    # cost_giou < 0, but can be a bit higher in the beginning of training:
    cost_upper_bound = cost_upper_bound + giou_loss_coef * 1.0

  # Don't make costs too large w.r.t. the rest to avoid numerical instability.
  mask = mask[:, None]
  cost = cost * mask + (1.0 - mask) * cost_upper_bound
  # Guard against NaNs and Infs.
  cost = jnp.nan_to_num(
      cost,
      nan=cost_upper_bound,
      posinf=cost_upper_bound,
      neginf=cost_upper_bound)

  assert cost.shape == (batch_size, num_queries, max_num_boxes)

  # Compute the number of unpadded columns for each batch element. It is assumed
  # that all padding is trailing padding.
  n_cols = jnp.where(
      jnp.max(mask, axis=1),
      jnp.expand_dims(jnp.arange(1, max_num_boxes + 1), axis=0), 0)
  n_cols = jnp.max(n_cols, axis=1)
  return cost, n_cols  # pytype: disable=bad-return-type  # jax-ndarray


# TODO(agritsenko): remove this copy-paste from
#  scenic.model_lib.base_models.tests.test_model_utils
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


class MatchingTest(parameterized.TestCase):
  """Test hungarian matcher."""

  def setUp(self):
    """Setup sample output predictions and target labels and bounding boxes."""
    super().setUp()

    self.batchsize = 4
    self.num_classes = 1000
    self.num_preds = 100
    # TODO(diwe): only N->N mapping is supported by greedy and sinkhorn.
    self.max_num_boxes = self.num_preds

    key = jax.random.PRNGKey(0)

    # Create fake predictions and targets
    key, subkey = jax.random.split(key)
    # set probabilities for class 0 higher than others
    p_logits = jnp.ones(self.num_classes).at[0].set(5.)
    p = jax.nn.softmax(p_logits)
    tgt_labels = jax.random.choice(
        subkey,
        self.num_classes,
        shape=(self.batchsize, self.max_num_boxes),
        replace=True,
        p=p)
    # Ensure last target is dummy empty target.
    tgt_labels = tgt_labels.at[:, -1].set(0)
    onehot_tgt_labels = jax.nn.one_hot(tgt_labels, self.num_classes)

    key, subkey = jax.random.split(key)
    pred_logits = jax.random.normal(
        subkey, shape=(self.batchsize, self.num_preds, self.num_classes))
    pred_probs = jax.nn.softmax(pred_logits, axis=-1)

    key, subkey = jax.random.split(key)
    pred_bbox = sample_cxcywh_bbox(
        subkey, batch_shape=(self.batchsize, self.num_preds))

    key, subkey = jax.random.split(key)
    tgt_bbox = sample_cxcywh_bbox(
        subkey, batch_shape=(self.batchsize, self.max_num_boxes))

    self.outputs = {'pred_probs': pred_probs, 'pred_boxes': pred_bbox}
    self.targets = {'labels': tgt_labels, 'boxes': tgt_bbox}
    self.onehot_targets = {'labels': onehot_tgt_labels, 'boxes': tgt_bbox}
    self.cost_matrix, self.cost_n_cols = compute_cost(
        tgt_bbox=tgt_bbox,
        tgt_labels=tgt_labels,
        out_bbox=pred_bbox,
        out_prob=pred_probs,
        bbox_loss_coef=1.,
        giou_loss_coef=1.,
        class_loss_coef=1.,
        target_is_onehot=False)
    self.cost_matrix_one_hot, self.cost_n_cols_one_hot = compute_cost(
        tgt_bbox=tgt_bbox,
        tgt_labels=onehot_tgt_labels,
        out_bbox=pred_bbox,
        out_prob=pred_probs,
        bbox_loss_coef=1.,
        giou_loss_coef=1.,
        class_loss_coef=1.,
        target_is_onehot=True)

  def test_cost_onehot_consistency(self):
    """Checks cost matrix consistency for one-hot and index representations."""
    diff = jnp.max(jnp.abs(self.cost_matrix - self.cost_matrix_one_hot))
    self.assertLess(diff, EPS)

  @parameterized.named_parameters(*(MATCHER_FUNCTIONS.items()))
  def test_matchers_identity(self, matcher_fn):
    """Tests if column==row indices for matching non-empty targets to itself."""

    # Note: you can only do this in the one hot case with targets
    # otherwise shapes don't match up.

    # Only use targets with non-empty boxes, otherwise
    # filtering messes up this test as it only filters the target labels
    # not the labels of the predictions.

    tgt_labels = []
    for i in range(self.batchsize):
      key = jax.random.PRNGKey(i)
      tgt_labels.append(jax.random.choice(
          key,
          jnp.arange(1, self.num_classes),
          shape=(self.max_num_boxes,),
          replace=False,
          p=None))
    tgt_labels = jnp.stack(tgt_labels)
    # Ensure last target is dummy empty target.
    tgt_labels = tgt_labels.at[:, -1].set(0)
    onehot_tgt_labels = jax.nn.one_hot(tgt_labels, self.num_classes)

    onehot_targets = self.onehot_targets.copy()
    onehot_targets['labels'] = onehot_tgt_labels

    outputs = {
        'pred_probs': onehot_tgt_labels,
        'pred_boxes': onehot_targets['boxes']
    }

    cost, _ = compute_cost(
        tgt_labels=tgt_labels,
        out_prob=outputs['pred_probs'],
        tgt_bbox=outputs['pred_boxes'],
        out_bbox=outputs['pred_boxes'],
        bbox_loss_coef=1.,
        giou_loss_coef=1.,
        class_loss_coef=1.,
        target_is_onehot=False)

    indices = matcher_fn(cost)
    self.assertEqual(indices.shape, (cost.shape[0], 2, cost.shape[1]))
    for row, col in indices:
      self.assertTrue(jnp.array_equal(row, col))

  @parameterized.named_parameters(
      *[(name, MATCHER_FUNCTIONS[name]) for name in EXACT_MATCHERS])
  def test_cost_matches_scipy(self, matcher_fn):
    """Can recover the matching returned by Scipy?"""
    sp_ind = np.array(list(map(lambda x: tuple(sciopt.linear_sum_assignment(x)),
                               self.cost_matrix)))
    ind = matcher_fn(self.cost_matrix)

    for i, ((sp_row, sp_col), (row, col)) in enumerate(zip(sp_ind, ind)):
      sp_cost = self.cost_matrix[i, sp_row, sp_col].sum()
      cost = self.cost_matrix[i, row, col].sum()
      self.assertAlmostEqual(sp_cost, cost, places=4)

  @parameterized.named_parameters(
      *[(name, MATCHER_FUNCTIONS[name]) for name in RECT_MATCHERS])
  def test_cost_matches_scipy_rect_n_bigger_m(self, matcher_fn):
    """Can recover the matching returned by Scipy for m > n matrices?"""
    # Test where n > m.
    cost_matrix = self.cost_matrix[:, :, self.cost_matrix.shape[2] // 2:]
    sp_ind = np.array(list(map(lambda x: tuple(sciopt.linear_sum_assignment(x)),
                               cost_matrix)))
    ind = matcher_fn(cost_matrix)

    for i, ((sp_row, sp_col), (row, col)) in enumerate(zip(sp_ind, ind)):
      sp_cost = cost_matrix[i, sp_row, sp_col].sum()
      cost = cost_matrix[i, row, col].sum()
      self.assertAlmostEqual(sp_cost, cost, places=4)

  @parameterized.named_parameters(
      *[(name, MATCHER_FUNCTIONS[name]) for name in RECT_MATCHERS])
  def test_cost_matches_scipy_rect_n_smaller_m(self, matcher_fn):
    """Can recover the matching returned by Scipy for n < m matrices?"""
    # Test where n < m.
    cost_matrix = self.cost_matrix[:, self.cost_matrix.shape[1] // 2:, :]
    sp_ind = np.array(list(map(lambda x: tuple(sciopt.linear_sum_assignment(x)),
                               cost_matrix)))
    ind = matcher_fn(cost_matrix)

    for i, ((sp_row, sp_col), (row, col)) in enumerate(zip(sp_ind, ind)):
      sp_cost = cost_matrix[i, sp_row, sp_col].sum()
      cost = cost_matrix[i, row, col].sum()
      self.assertAlmostEqual(sp_cost, cost, places=4)

  @parameterized.named_parameters(
      *[(name, MATCHER_FUNCTIONS[name]) for name in CPU_MATCHERS])
  def test_slicer_full(self, matcher_fn):
    """For a full matrix the slicer must return the same matching."""
    ind_full = matcher_fn(self.cost_matrix)
    ind_slicer = matchers.slicer(self.cost_matrix, self.cost_n_cols, matcher_fn)

    for i, ((full_row, full_col), (row, col)) in enumerate(
        zip(ind_full, ind_slicer)):
      full_cost = self.cost_matrix[i, full_row, full_col].sum()
      cost = self.cost_matrix[i, row, col].sum()
      self.assertAlmostEqual(full_cost, cost, places=4)

  @parameterized.named_parameters(
      *[(name, MATCHER_FUNCTIONS[name]) for name in CPU_MATCHERS])
  def test_slicer(self, matcher_fn):
    """Simulate padding and ensure that slicer can deal with it."""
    n_cols = self.cost_n_cols // 2
    mask = np.concatenate((np.ones((1, n_cols[0]), dtype=bool),
                           np.zeros(
                               (1, self.num_preds - n_cols[0]), dtype=bool)),
                          axis=1)
    cost = mask * self.cost_matrix + (1. - mask) * 5

    ind_full = matcher_fn(cost)
    ind_slicer = matchers.slicer(cost, n_cols, matcher_fn)

    for i, ((full_row, full_col), (slicer_row, slicer_col)) in enumerate(
        zip(ind_full, ind_slicer)):
      full_cost = cost[i, full_row, full_col].sum()
      slicer_cost = cost[i, slicer_row, slicer_col].sum()
      self.assertAlmostEqual(full_cost, slicer_cost, places=3)

  @parameterized.named_parameters(
      *[(name, MATCHER_FUNCTIONS[name]) for name in CPU_MATCHERS])
  def test_slicer_implicit(self, matcher_fn):
    """Ensure that implicit use of slicer works."""
    n_cols = self.cost_n_cols // 2
    mask = np.concatenate((np.ones((1, n_cols[0]), dtype=bool),
                           np.zeros(
                               (1, self.num_preds - n_cols[0]), dtype=bool)),
                          axis=1)
    cost = mask * self.cost_matrix + (1. - mask) * 5

    ind_slicer_impl = matcher_fn(cost, n_cols=n_cols)
    ind_slicer = matchers.slicer(cost, n_cols, matcher_fn)

    for i, ((impl_row, impl_col), (slicer_row, slicer_col)) in enumerate(
        zip(ind_slicer_impl, ind_slicer)):
      impl_cost = cost[i, impl_row, impl_col].sum()
      slicer_cost = cost[i, slicer_row, slicer_col].sum()
      self.assertAlmostEqual(impl_cost, slicer_cost, places=3)

  @parameterized.named_parameters(
      *[(name, MATCHER_FUNCTIONS[name]) for name in RECT_MATCHERS])
  def test_manual_cost_matrix(self, matcher_fn):
    """Test case from bencaine@ for repro."""
    cost_matrix = jnp.asarray([
        # We expect (0, 0) and (1, 1) to be matched.
        [[-100, 100],
         [100, -100],
         [100, 100]],
        # We expect (0, 0) and (2, 1) to be matched.
        [[-100, 100],
         [100, 100],
         [100, -100]]], dtype=jnp.float32)

    sp_ind = np.array(list(map(lambda x: tuple(sciopt.linear_sum_assignment(x)),
                               cost_matrix)))
    ind = matcher_fn(cost_matrix)
    for i, ((sp_row, sp_col), (row, col)) in enumerate(zip(sp_ind, ind)):
      sp_cost = cost_matrix[i, sp_row, sp_col].sum()
      cost = cost_matrix[i, row, col].sum()
      self.assertAlmostEqual(sp_cost, cost, places=4)

  class TestLazyMatcher(parameterized.TestCase):
    """Test lazy_matcher function."""

    @parameterized.named_parameters(('nbxy79', 7, 9), ('nbxy22', 2, 2))
    def test_lazy_matcher(self, nbx, nby):
      """Test across varying number of boxes."""

      cost_matrix = jnp.zeros((3, nbx, nby), dtype=jnp.float32)

      # Lazy matcher always returns jnp.array([0, 1, 2, ..., min-boxes]).
      expected_indices_per_row = jnp.array(list(range(min(nbx, nby))))

      indices = matchers.lazy_matcher(cost_matrix)
      self.assertEqual(indices.shape, (3, 2, min(nbx, nby)))
      for idx in indices:  # Iterate over elements in batch.
        src, tgt = idx
        self.assertTrue(jnp.array_equal(src, expected_indices_per_row))
        self.assertTrue(jnp.array_equal(tgt, expected_indices_per_row))


if __name__ == '__main__':
  absltest.main()
