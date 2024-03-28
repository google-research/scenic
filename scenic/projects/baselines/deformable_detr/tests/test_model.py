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

"""Tests for model.py."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from flax import jax_utils
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from scenic.projects.baselines.deformable_detr.configs import mini_config
from scenic.projects.baselines.deformable_detr.model import compute_cost
from scenic.projects.baselines.deformable_detr.model import DeformableDETRModel


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


class DeformableDETRModelLossTest(parameterized.TestCase):
  """Test DeformableDETRModel loss test."""

  def setUp(self):
    super().setUp()

    self.num_classes = 5
    self.input_shape = (3, 128, 128, 3)
    self.num_decoder_layers = 2
    config = mini_config.get_config()

    # Create and initialize the model.
    model_cls = DeformableDETRModel
    self.model = model_cls(
        config=config,
        dataset_meta_data={
            'num_classes': self.num_classes,
            'target_is_onehot': False,
        })

    rng = random.PRNGKey(0)
    initial_params = self.model.flax_model.init(
        rng, jnp.zeros(self.input_shape, jnp.float32), train=False)
    flax_model = functools.partial(self.model.flax_model.apply, initial_params)

    # A fake batch with 3 examples.
    self.batch = {
        'inputs':
            jnp.array(np.random.normal(size=self.input_shape)
                     ).astype(jnp.float32),
        'padding_mask':
            jnp.array(np.random.normal(size=self.input_shape[:-1])
                     ).astype(jnp.float32),
        'label': {
            'labels':
                jnp.array(
                    np.random.randint(
                        self.num_classes,
                        size=(3, self.model.config.num_queries))),
            'boxes':
                jnp.array(
                    np.random.uniform(
                        size=(3, self.model.config.num_queries, 4),
                        low=0.0,
                        high=1.0),
                    dtype=jnp.float32),
        }
    }
    self.outputs = flax_model(
        self.batch['inputs'],
        padding_mask=self.batch['padding_mask'],
        train=False)

    seq = np.arange(self.model.config.num_queries, dtype=np.int32)
    seq_rev = seq[::-1]
    seq_21 = np.concatenate([
        seq[self.model.config.num_queries // 2:],
        seq[:self.model.config.num_queries // 2]
    ])
    self.indices = jnp.array([(seq, seq_rev), (seq_rev, seq), (seq, seq_21)])

  def test_loss_function(self):
    """Test loss_function by checking its output's dictionary format."""

    # Test loss function in the pmapped setup.
    loss_function_pmapped = jax.pmap(
        self.model.loss_and_metrics_function, axis_name='batch')

    matches = jax_utils.replicate(
        # Fake matching for the  final output +  2 aux outputs.
        [self.indices] * 3)
    outputs_replicated, batch_replicated = (jax_utils.replicate(self.outputs),
                                            jax_utils.replicate(self.batch))
    total_loss, metrics_dict = loss_function_pmapped(
        outputs_replicated, batch_replicated, matches=matches)

    total_loss, metrics_dict = (jax_utils.unreplicate(total_loss),
                                jax_utils.unreplicate(metrics_dict))

    # Collect what keys we expect to find in the metrics_dict.
    expected_metrics_keys = [
        'loss_class', 'loss_bbox', 'loss_giou', 'total_loss'
    ]
    for i in range(self.num_decoder_layers - 1):
      expected_metrics_keys += [
          f'loss_class_aux{i}', f'loss_bbox_aux{i}', f'loss_giou_aux{i}'
      ]
    self.assertSameElements(expected_metrics_keys, metrics_dict.keys())

    # Since weight decay is not used, the following must hold.
    object_detection_loss = 0
    for k in metrics_dict.keys():
      b = k.split('_aux')[0]
      # If this loss going to be included in the total loss.
      if b in self.model.loss_terms_weights.keys():
        # Get the normalizer for this loss.
        object_detection_loss += (
            # Already scaled loss term / loss term normalizer.
            metrics_dict[k][0] / metrics_dict[k][1])
    self.assertAlmostEqual(total_loss, object_detection_loss, places=3)


class DeformableDETRModelCostTest(parameterized.TestCase):
  """Test DeformableDETRModel cost test."""

  def test_compute_cost(self):
    """Test compute_cost."""

    bs = 3
    num_classes = 7
    num_preds = 5
    # Includes padding.
    max_targets = 10

    key = jax.random.PRNGKey(0)

    # Create fake predictions and targets
    key, subkey = jax.random.split(key)
    # set probabilities for class 0 higher than others
    p_logits = jnp.ones(num_classes).at[0].set(5.)
    tgt_labels = jax.random.choice(
        subkey,
        np.arange(1, num_classes + 1),
        shape=(bs, max_targets),
        replace=True,
        p=jax.nn.softmax(p_logits))

    # Set padding for targets by index of batch
    for i in range(1, bs + 1):
      tgt_labels = tgt_labels.at[i, -i:].set(0)

    key, subkey = jax.random.split(key)
    pred_logits = jax.random.normal(subkey, shape=(bs, num_preds, num_classes))
    pred_probs = jax.nn.sigmoid(pred_logits)

    key, subkey = jax.random.split(key)
    pred_bbox = sample_cxcywh_bbox(subkey, batch_shape=(bs, num_preds))

    key, subkey = jax.random.split(key)
    tgt_bbox = sample_cxcywh_bbox(subkey, batch_shape=(bs, max_targets))

    cost, n_cols = compute_cost(
        tgt_bbox=tgt_bbox,
        tgt_labels=tgt_labels,
        out_bbox=pred_bbox,
        out_prob=pred_probs,
        bbox_loss_coef=1.,
        giou_loss_coef=1.,
        class_loss_coef=1.,
        target_is_onehot=False)
    self.assertSequenceEqual(cost.shape, (bs, num_preds, max_targets))
    exp_n_cols = range(max_targets, max_targets - bs, -1)
    self.assertSequenceEqual(n_cols.tolist(), exp_n_cols)


if __name__ == '__main__':
  absltest.main()
