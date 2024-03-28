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

"""Tests for the ViViT classification train script."""

import functools
import shutil
import os

from absl.testing import absltest
from absl.testing import parameterized
import flax
from flax import jax_utils
import flax.linen as nn
import jax.numpy as jnp
import jax.random
import ml_collections
import numpy as np
from scenic.model_lib.base_models import classification_model
from scenic.model_lib.base_models import multilabel_classification_model
from scenic.projects.vivit import evaluation_lib
from scenic.projects.vivit import train_utils as vivit_train_utils
from scenic.train_lib_deprecated import optimizers
from scenic.train_lib_deprecated import train_utils
import tensorflow as tf


class ViViTClassificationTrainerTest(parameterized.TestCase):
  """Tests the default trainer on single device setup."""

  def setUp(self):
    super(ViViTClassificationTrainerTest, self).setUp()
    self.test_dir = '/tmp/scenic_test'
    os.mkdir(self.test_dir)

    # Make sure Tensorflow does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super(ViViTClassificationTrainerTest, self).tearDown()

  def get_train_state(self, rng, fake_batch_logits):
    """Generates the initial training state."""
    config = ml_collections.ConfigDict({
        'lr_configs': {
            'base_learning_rate': 0.1,
        },
        'optimizer': 'sgd',
    })

    # Define a fake model that always outputs the same "fake_batch_logits".
    class FakeFlaxModel(nn.Module):
      """A fake flax model."""

      @nn.compact
      def __call__(self, x, train=False, debug=False):
        del x
        del train
        del debug
        # FakeFlaxModule always predicts class 2.
        return fake_batch_logits

    dummy_input = jnp.zeros((10, 10), jnp.float32)
    initial_params = FakeFlaxModel().init(rng, dummy_input).get(
        'params', flax.core.frozen_dict.FrozenDict({}))
    init_model_state = flax.core.frozen_dict.FrozenDict({})
    optimizer = optimizers.get_optimizer(config).create(initial_params)
    init_train_state = jax_utils.replicate(
        train_utils.TrainState(
            global_step=0,
            optimizer=optimizer,
            model_state=init_model_state,
            rng=jax.random.PRNGKey(0)))
    return FakeFlaxModel(), init_train_state

  def train_and_evaluation(self, model, train_state, fake_batches, metrics_fn,
                           return_confusion_matrix=False):
    """Given the train_state, trains the model on fake batches."""
    eval_metrics = []
    fake_batches_replicated = jax_utils.replicate(fake_batches)
    if return_confusion_matrix:
      confusion_matrices = []

    eval_step_pmapped = jax.pmap(
        functools.partial(
            vivit_train_utils.eval_step,
            flax_model=model,
            metrics_fn=metrics_fn,
            return_logits_and_labels=False,
            return_confusion_matrix=return_confusion_matrix,
            debug=False),
        axis_name='batch',
        donate_argnums=(1,),
    )
    for fake_batch in fake_batches_replicated:
      metric_data = eval_step_pmapped(train_state=train_state, batch=fake_batch)
      if return_confusion_matrix:
        metrics, confusion_matrix = metric_data
        confusion_matrices.append(vivit_train_utils.to_cpu(confusion_matrix))
      else:
        metrics = metric_data
      metrics = train_utils.unreplicate_and_get(metrics)
      eval_metrics.append(metrics)
    eval_metrics = train_utils.stack_forest(eval_metrics)
    eval_summary = jax.tree_util.tree_map(lambda x: x.sum(), eval_metrics)
    for key, val in eval_summary.items():
      eval_summary[key] = val[0] / val[1]

    if return_confusion_matrix:
      confusion_matrix_summary = (
          evaluation_lib.compute_confusion_matrix_metrics(
              confusion_matrices, return_per_class_metrics=True))
      return eval_summary, confusion_matrix_summary
    else:
      return eval_summary

  @parameterized.named_parameters(
      ('without confusion matrix summary', False),
      ('with confusion matrix summary', True),
  )
  def test_classifaction_model_evaluate(self, get_confusion_matrix):
    """Test trainer evaluate end to end with classification model metrics."""
    # Define a fixed output for the fake flax model.
    fake_batch_logits = np.tile([.5, .2, .7, 0.0], (4, 1))
    # 4 evaluation batches of size 4.
    fake_batches = [
        {
            'inputs': None,
            'label': np.array([3, 2, 1, 0]),
            'batch_mask': np.array([1, 1, 1, 1])
        },
        {
            'inputs': None,
            'label': np.array([0, 3, 2, 0]),
            'batch_mask': np.array([1, 1, 1, 1])
        },
        {
            'inputs': None,
            'label': np.array([0, 0, 0, 0]),
            'batch_mask': np.array([1, 1, 1, 1])
        },
        {
            'inputs': None,
            'label': np.array([1, 1, 1, 1]),
            'batch_mask': np.array([1, 1, 1, 1])
        },
    ]

    rng = jax.random.PRNGKey(0)
    model, train_state = self.get_train_state(rng, fake_batch_logits)
    eval_summary = self.train_and_evaluation(
        model, train_state, fake_batches,
        functools.partial(
            classification_model.classification_metrics_function,
            target_is_onehot=False),
        get_confusion_matrix)
    if get_confusion_matrix:
      eval_summary, confusion_matrix_summary = eval_summary

    def batch_loss(logits, targets):
      # Softmax cross-entropy loss.
      one_hot_targets = np.eye(4)[targets]
      loss = -np.sum(one_hot_targets * nn.log_softmax(logits), axis=-1)
      return loss

    expected_accuracy = 2.0 / 16.0  # FakeFlaxModule always predicts class 2.
    expected_loss = np.mean(
        [batch_loss(fake_batch_logits, b['label']) for b in fake_batches])

    self.assertEqual(expected_accuracy, eval_summary['accuracy'])
    np.testing.assert_allclose(expected_loss, eval_summary['loss'], atol=1e-6)

    if get_confusion_matrix:
      # As FakeFlaxModule always predicts class 2, this class has a recall of 1,
      # and the denominator for the precision is the total number of examples.
      self.assertAlmostEqual(confusion_matrix_summary['precision/0'], 0.0)
      self.assertAlmostEqual(confusion_matrix_summary['precision/1'], 0.0)
      self.assertAlmostEqual(confusion_matrix_summary['precision/2'], 2. / 16.)
      self.assertAlmostEqual(confusion_matrix_summary['precision/3'], 0.0)
      self.assertAlmostEqual(confusion_matrix_summary['recall/0'], 0.0)
      self.assertAlmostEqual(confusion_matrix_summary['recall/1'], 0.0)
      self.assertAlmostEqual(confusion_matrix_summary['recall/2'], 1.0)
      self.assertAlmostEqual(confusion_matrix_summary['recall/3'], 0.0)
      self.assertAlmostEqual(confusion_matrix_summary['jaccard/0'], 0.0)
      self.assertAlmostEqual(confusion_matrix_summary['jaccard/1'], 0.0)
      self.assertAlmostEqual(confusion_matrix_summary['jaccard/2'], 2. / 16.)
      self.assertAlmostEqual(confusion_matrix_summary['jaccard/3'], 0.0)

  def test_multi_label_classifaction_model_evaluate(self):
    """Test trainer evaluate with multi-label classification model metrics."""
    # Define a fixed output for the fake flax model.
    fake_batch_logits = np.tile([.5, .2, .7, 0.0], (4, 1))
    # 4 evaluation batches of size 4, with multihot labels.
    fake_batches = [
        {
            'inputs':
                None,
            'label':
                np.array([[1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0],
                          [1, 0, 0, 1]])
        },
        {
            'inputs':
                None,
            'label':
                np.array([[1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1],
                          [1, 0, 0, 1]])
        },
        {
            'inputs':
                None,
            'label':
                np.array([[1, 0, 0, 1], [1, 0, 0, 0], [1, 1, 0, 0],
                          [1, 0, 0, 0]])
        },
        {
            'inputs':
                None,
            'label':
                np.array([[0, 1, 0, 1], [0, 1, 0, 0], [1, 1, 0, 0],
                          [0, 1, 0, 0]])
        },
    ]

    rng = jax.random.PRNGKey(0)
    model, train_state = self.get_train_state(rng, fake_batch_logits)
    eval_summary = self.train_and_evaluation(
        model, train_state, fake_batches,
        functools.partial(
            multilabel_classification_model
            .multilabel_classification_metrics_function,
            target_is_multihot=True))

    def batch_loss(logits, multi_hot_targets):
      # Sigmoid cross-entropy loss.
      log_p = jax.nn.log_sigmoid(logits)
      log_not_p = jax.nn.log_sigmoid(-logits)
      loss = -np.sum(
          multi_hot_targets * log_p + (1. - multi_hot_targets) * log_not_p,
          axis=-1)
      return loss

    expected_prec_at_one = 2.0 / 16.0  # FakeFlaxModule always predicts class 2.
    expected_loss = np.mean(
        [batch_loss(fake_batch_logits, b['label']) for b in fake_batches])

    self.assertEqual(expected_prec_at_one, eval_summary['prec@1'])
    np.testing.assert_allclose(expected_loss, eval_summary['loss'], atol=1e-6)


if __name__ == '__main__':
  absltest.main()
