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

"""Tests for metrics specific to ViViT (ie pairwise accuracy)."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from scenic.projects.vivit import evaluation_lib
from scenic.projects.vivit import model_utils


class EvaluationMetricsTester(absltest.TestCase):
  """Tests evaluation metrics specific to ViViT model."""

  def test_joint_accuracy(self):
    """Test pairwise accuracy calculation."""

    c_1 = 3
    c_2 = 2
    class_splits = jnp.array([c_1, c_1 + c_2])
    logits = jnp.array([
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ])
    one_hot_labels = jnp.array([[0, 1, 0, 1, 0], [0, 0, 1, 1, 0],
                                [0, 1, 0, 0, 1], [0, 0, 1, 0, 1]])

    accuracy = model_utils.joint_accuracy(logits, one_hot_labels, class_splits)
    expected_accuracy = jnp.array([0, 0, 0, 1]).astype(jnp.int32)

    np.testing.assert_almost_equal(
        np.array(expected_accuracy), np.array(accuracy))


class ConfusionMatrixTester(absltest.TestCase):
  """Tests confusion matrix metrics."""

  def test_confusion_matrix_metrics(self):
    """Test calculation of metrics given a confusion matrix."""

    confusion_matrices = [
        np.array([[[2, 0, 1], [1, 3, 0], [1, 2, 4]]]),
        np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]),
    ]
    metrics = evaluation_lib.compute_confusion_matrix_metrics(
        confusion_matrices, return_per_class_metrics=True)

    expected_keys = {'recall/mean', 'precision/mean', 'jaccard/mean',
                     'recall/0', 'recall/1', 'recall/2',
                     'precision/0', 'precision/1', 'precision/2',
                     'jaccard/0', 'jaccard/1', 'jaccard/2'}
    self.assertSameElements(expected_keys, metrics.keys())
    self.assertAlmostEqual(metrics['recall/mean'], np.mean([3/9, 8/19, 13/31]))
    self.assertAlmostEqual(metrics['recall/0'], 3 / 9)
    self.assertAlmostEqual(metrics['recall/1'], 8 / 19)
    self.assertAlmostEqual(metrics['recall/2'], 13 / 31)
    self.assertAlmostEqual(metrics['precision/mean'],
                           np.mean([3 / 16, 8 / 20, 13 / 23]))
    self.assertAlmostEqual(metrics['precision/0'], 3 / 16)
    self.assertAlmostEqual(metrics['precision/1'], 8 / 20)
    self.assertAlmostEqual(metrics['precision/2'], 13 / 23)
    self.assertAlmostEqual(metrics['jaccard/mean'],
                           np.mean([3 / 22, 8 / 31, 13 / 41]))
    self.assertAlmostEqual(metrics['jaccard/0'], 3 / 22)
    self.assertAlmostEqual(metrics['jaccard/1'], 8 / 31)
    self.assertAlmostEqual(metrics['jaccard/2'], 13 / 41)

  def test_with_nans(self):
    """Test metric calculation where one of the metrics is NaN."""

    confusion_matrices = [np.array([[[0, 1], [0, 2]]])]
    metrics = evaluation_lib.compute_confusion_matrix_metrics(
        confusion_matrices, return_per_class_metrics=True)

    self.assertAlmostEqual(metrics['recall/mean'], np.mean([0, 1]))
    self.assertAlmostEqual(metrics['recall/0'], 0)
    self.assertAlmostEqual(metrics['recall/1'], 1)
    self.assertAlmostEqual(metrics['precision/mean'],
                           2 / 3)  #  Should not not average over NaN metrics
    self.assertAlmostEqual(metrics['precision/0'], 0)  # Actually it's NaN
    self.assertAlmostEqual(metrics['precision/1'], 2 / 3)
    self.assertAlmostEqual(metrics['jaccard/mean'], np.mean([0, 2 / 3]))
    self.assertAlmostEqual(metrics['jaccard/0'], 0)
    self.assertAlmostEqual(metrics['jaccard/1'], 2 / 3)

if __name__ == '__main__':
  absltest.main()
