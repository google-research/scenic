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

"""Unit tests for functions in metrics.py."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from scenic.projects.svvit import metrics


class MetricsTest(parameterized.TestCase):

  def setUp(self):
    self.one_hot_targets = jnp.array([[1, 0, 0], [0, 0, 1], [0, 1, 0],
                                      [0, 1, 0]])
    self.logits = jnp.array([[0.41, 0.39, 0.2], [0.4, 0.6, 0], [0.5, 0.1, 0.4],
                             [0.3, 0.5, 0.2]])
    super().setUp()

  def test_truvari_presicion(self):
    m = metrics.truvari_precision(self.logits, self.one_hot_targets)  # pytype: disable=wrong-arg-types  # jnp-type
    self.assertAlmostEqual(m['truvari_precision'], 0.5, places=5)

  def test_truvari_recall(self):
    m = metrics.truvari_recall(self.logits, self.one_hot_targets)  # pytype: disable=wrong-arg-types  # jnp-type
    self.assertAlmostEqual(m['truvari_recall'], 1.0 / 3.0, places=5)

  def test_truvari_presicion_events(self):
    m = metrics.truvari_precision_events(self.logits, self.one_hot_targets)  # pytype: disable=wrong-arg-types  # jnp-type
    self.assertAlmostEqual(m['truvari_precision_events'], 1.0, places=5)

  def test_truvari_recall_events(self):
    m = metrics.truvari_recall_events(self.logits, self.one_hot_targets)  # pytype: disable=wrong-arg-types  # jnp-type
    self.assertAlmostEqual(m['truvari_recall_events'], 2.0 / 3.0, places=5)


if __name__ == '__main__':
  absltest.main()
