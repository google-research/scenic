"""Tests for optimizer_utils."""

from absl.testing import absltest
import chex
import flax
import jax
import ml_collections
import numpy as np
import optax
from scenic.projects.unloc import optimizer_utils


class OptimizerUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.params = {
        'video_encoder': {
            'pos_embedding': np.zeros((8)),
            'block0': {
                'kernel': np.ones((10)),
                'bias': np.zeros((10)),
            },
        },
        'text_encoder': {
            'pos_embedding': np.zeros((8)),
            'block0': {
                'kernel': np.ones((10)),
                'bias': np.zeros((10)),
            },
        },
        'video_text_fusion': {
            'block0': {
                'kernel': np.ones((10)),
                'bias': np.zeros((10)),
            },
        },
    }
    self.params = flax.core.freeze(self.params)

  def test_optimizer_with_multi_lrs(self):
    config = ml_collections.ConfigDict({
        'optimizer': 'sgd',
        'optimizer_configs': dict(momentum=0.9),
        'lr_configs': dict(base_learning_rate=0.1),
        'layer_prefix_to_base_lrs': {
            'video_text_fusion': 1.0,
        },
    })
    expected_params = {
        'video_encoder': {
            'pos_embedding': np.zeros((8)) - 0.1,
            'block0': {
                'kernel': np.ones((10)) - 0.1,
                'bias': np.zeros((10)) - 0.1,
            },
        },
        'text_encoder': {
            'pos_embedding': np.zeros((8)) - 0.1,
            'block0': {
                'kernel': np.ones((10)) - 0.1,
                'bias': np.zeros((10)) - 0.1,
            },
        },
        'video_text_fusion': {
            'block0': {
                'kernel': np.ones((10)) - 1.0,
                'bias': np.zeros((10)) - 1.0,
            },
        },
    }
    tx = optimizer_utils.optimizer_with_multi_lrs(config, self.params)
    state = tx.init(self.params)
    gradients = jax.tree_util.tree_map(np.ones_like, self.params)
    updates, _ = tx.update(gradients, state, self.params)
    new_params = optax.apply_updates(self.params, updates)
    chex.assert_trees_all_close(flax.core.unfreeze(new_params), expected_params)


if __name__ == '__main__':
  absltest.main()
