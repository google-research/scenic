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

"""Tests for Scenic optax utils."""

from typing import Optional
from absl.testing import absltest
from absl.testing import parameterized
import ml_collections
from scenic.train_lib import optax
import tensorflow as tf


class OptaxTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for Scenic Optax utils."""

  @parameterized.parameters(True, False)
  def test_make_mask_trees(self, allow_unmatched):
    """Tests `make_mask_trees`."""
    tree = {'a': {'b': 1}, 'c': 2}
    patterns = [('a/.*', 'A', 0.1), ('b/.*', 'B', 0.3), ('c', 'C', 0.3)]
    masks = optax.make_mask_trees(
        tree, patterns, allow_unmatched=allow_unmatched)
    target_masks = [
        {'a': {'b': True}, 'c': False},
        {'a': {'b': False}, 'c': False},
        {'a': {'b': False}, 'c': True},
    ]
    self.assertAllEqual(masks, target_masks)

  def test_make_mask_trees_unmatched(self):
    """Tests `make_mask_trees` with allow_unmatched=False raises."""
    tree = {'a': {'b': 1}, 'c': 2, 'd': 3}
    patterns = [('a/.*', 'A'), ('b/.*', 'B'), ('c', 'C')]
    with self.assertRaises(ValueError):
      optax.make_mask_trees(tree, patterns, allow_unmatched=False)

  @parameterized.parameters(True, False)
  def test__make_mask_trees(self, allow_unmatched):
    """Tests `_make_mask_trees` with names and values."""
    tree = {'a': {'b': 1}, 'c': 2}
    patterns_names_values = [
        ('a/.*', 'A', 1), ('b/.*', 'B', 2), ('c', 'C', 3)]
    # pylint: disable=protected-access
    masks, names_values = optax._make_mask_trees(
        tree, patterns_names_values, allow_unmatched=allow_unmatched)
    # pylint: enable=protected-access
    target_masks = [
        {'a': {'b': True}, 'c': False},
        {'a': {'b': False}, 'c': False},
        {'a': {'b': False}, 'c': True},
    ]
    target_names_values = [('A', 1), ('B', 2), ('C', 3)]
    self.assertAllEqual(masks, target_masks)
    self.assertAllEqual(names_values, target_names_values)

  @parameterized.parameters(True, False)
  def test__make_mask_trees_with_values_only(self, allow_unmatched):
    """Tests `_make_mask_trees` with values only."""
    tree = {'a': {'b': 1}, 'c': 2}
    patterns_names_values = [
        ('a/.*', 1), ('b/.*', 2), ('c', 3)]
    # pylint: disable=protected-access
    masks, names_values = optax._make_mask_trees(
        tree, patterns_names_values, allow_unmatched=allow_unmatched)
    # pylint: enable=protected-access
    target_masks = [
        {'a': {'b': True}, 'c': False},
        {'a': {'b': False}, 'c': False},
        {'a': {'b': False}, 'c': True},
    ]
    target_names_values = [(None, 1), (None, 2), (None, 3)]
    self.assertAllEqual(masks, target_masks)
    self.assertAllEqual(names_values, target_names_values)

  def test__split_frozen_without_frozen(self):
    """Tests `_split_frozen` without a frozen schedule."""
    masks = [
        {'a': {'b': True, 'd': False}, 'c': False},
        {'a': {'b': False, 'd': True}, 'c': False},
    ]
    scheds = [1, 2]
    # pylint: disable=protected-access
    frozen_mask, masks, scheds = optax._split_frozen(masks, scheds)
    # pylint: enable=protected-access

    target_frozen_mask = {'a': {'b': False, 'd': False}, 'c': True}
    target_masks = [
        {'a': {'b': True, 'd': False}, 'c': False},
        {'a': {'b': False, 'd': True}, 'c': False},
    ]
    target_scheds = [1, 2]
    self.assertAllEqual(frozen_mask, target_frozen_mask)
    self.assertAllEqual(masks, target_masks)
    self.assertAllEqual(scheds, target_scheds)

  def test__split_frozen_with_frozen(self):
    """Tests `_split_frozen` with a frozen schedule."""
    masks = [
        {'a': {'b': True, 'd': False}, 'c': False},
        {'a': {'b': False, 'd': True}, 'c': False},
        {'a': {'b': False, 'd': False}, 'c': True},
    ]
    scheds = [1, 2, None]
    # pylint: disable=protected-access
    frozen_mask, masks, scheds = optax._split_frozen(masks, scheds)
    # pylint: enable=protected-access

    target_frozen_mask = {'a': {'b': False, 'd': False}, 'c': True}
    target_masks = [
        {'a': {'b': True, 'd': False}, 'c': False},
        {'a': {'b': False, 'd': True}, 'c': False},
    ]
    target_scheds = [1, 2]
    self.assertAllEqual(frozen_mask, target_frozen_mask)
    self.assertAllEqual(masks, target_masks)
    self.assertAllEqual(scheds, target_scheds)

  def test_replace_frozen_with_no_schedule(self):
    """Tests `replace_frozen` without a schedule."""
    tree = {'a': {'b': 1, 'd': 2}, 'c': 3}
    new_tree = optax.replace_frozen(None, tree, 0)
    self.assertAllEqual(tree, new_tree)

  def test_replace_frozen(self):
    """Tests `replace_frozen`."""
    tree = {'a': {'b': 1, 'd': 2}, 'c': 3}
    schedule = {
        'name': ml_collections.ConfigDict({'re': 'a/.*', 'lr_configs': None}),
        'rest': ml_collections.ConfigDict({'re': 'c', 'lr_configs': {}}),
    }
    new_tree = optax.replace_frozen(schedule, tree, 0)
    target_tree = {'a': {'b': 0, 'd': 0}, 'c': 3}
    self.assertAllEqual(target_tree, new_tree)

  def test_make_schedule(self):
    """Tests `make_schedule`."""
    def lr_fn(cfg: ml_collections.ConfigDict) -> float:
      """Dummy LR schedule creation function."""
      return float(cfg.lr_configs.lr)

    schedule = ml_collections.ConfigDict({
        'main': ml_collections.ConfigDict(
            {'re': 'a/.*',
             'lr_configs': ml_collections.ConfigDict({'lr': '1.0'})}
        ),
        'rest': ml_collections.ConfigDict(
            {'re': 'c',
             'lr_configs': ml_collections.ConfigDict(
                 {'lr': '2.0', 'base_learning_rate': 10.0})}
        ),
    })
    new_schedule = optax.make_schedule(schedule, lr_fn)
    target_schedule = [('a/.*', 'main', (1.0, 1.0)), ('c', 'rest', (2.0, 10.0))]
    self.assertAllEqual(target_schedule, new_schedule)

  def test_make_schedule_with_none(self):
    """Tests `make_schedule` with None schedule."""
    def lr_fn(cfg: ml_collections.ConfigDict) -> Optional[float]:
      """Dummy LR schedule creation function."""
      if cfg.lr_configs is None:
        return None
      return float(cfg.lr_configs.lr)

    new_schedule = optax.make_schedule(None, lr_fn)
    target_schedule = [('(.*)', 'all', (None, None))]
    self.assertAllEqual(target_schedule, new_schedule)

  def test_make(self):
    """Test that `make` runs."""
    cfg = ml_collections.ConfigDict({
        'max_grad_norm': 1.0,
        'per_example_clipping': True,
        'weight_decay_decouple': True,
        'decay_rules': [('.*c.*', 1.0)],
        'optax_name': 'scale_by_adam',
        'optax_grad_pmean': True,
        'optax_configs': ml_collections.ConfigDict({
            'b1': 0.9,
            'b2': 0.999
        })
    })
    params = {'a': {'b': 1, 'd': 3}, 'c': 2}
    _, _ = optax.make(cfg, [('.*', 'all', (None, 1.0))], params)


if __name__ == '__main__':
  absltest.main()
