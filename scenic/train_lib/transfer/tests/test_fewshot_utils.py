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

"""Unit tests for few-shot utils."""

from absl.testing import absltest
from big_vision.evaluators import fewshot as bv_fewshot
import jax
from jax import random
from scenic.train_lib.transfer import fewshot_utils


def big_vision_linear_regression(x, y, x_test, y_test, l2_reg, num_classes):
  """Computes fewshot regression with eigenvalue solver in big_vision."""
  # pylint: disable=protected-access (testing a private function)
  cache = bv_fewshot._precompute_cache(x, y, num_classes)
  accuracy = bv_fewshot._eig_fewshot_acc_fn(cache, x_test, y_test, l2_reg)
  # pylint: enable=protected-access
  return accuracy


class LinearRegressionTest(absltest.TestCase):
  """Tests linear regression used in few-shot evaluation."""

  def test_linear_regression(self):
    """Test linear regression."""
    # Generate random data.
    num_points = 512
    dim = 16
    num_classes = 5
    l2_regs = [1.0, 2.0, 8.0, 0.0]
    rng = random.PRNGKey(0)

    x = random.normal(rng, shape=(num_points, dim))
    x_test = random.normal(rng, shape=(num_points, dim))
    y = random.randint(rng, shape=(num_points,), minval=0, maxval=num_classes)
    y_test = random.randint(
        rng, shape=(num_points,), minval=0, maxval=num_classes)

    for l2_reg in l2_regs:
      # Compute predictions.
      accuracy = fewshot_utils._fewshot_acc_fn(  # pylint: disable=protected-access (testing a private function)
          x,
          y,
          x_test,
          y_test,
          l2_reg,
          num_classes,
          target_is_one_hot=False)

      # Compare with big_vision.
      expected_accuracy = big_vision_linear_regression(x, y, x_test, y_test,
                                                       l2_reg, num_classes)
      self.assertGreater(accuracy, 0)
      self.assertLess(accuracy, 1)
      self.assertAlmostEqual(accuracy, expected_accuracy, delta=1e-6)

      # Check they are identical when labels are one-hot.
      y_one_hot = jax.nn.one_hot(y, num_classes)
      y_test_one_hot = jax.nn.one_hot(y_test, num_classes)

      accuracy_one_hot = fewshot_utils._fewshot_acc_fn(  # pylint: disable=protected-access (testing a private function)
          x,
          y_one_hot,
          x_test,
          y_test_one_hot,
          l2_reg,
          num_classes,
          target_is_one_hot=True)
      self.assertEqual(accuracy, accuracy_one_hot)


if __name__ == '__main__':
  absltest.main()
