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

"""Tests for the classification train script."""

import functools
import shutil
import tempfile

from absl.testing import absltest
from clu import metric_writers
import flax
from flax import jax_utils
import flax.linen as nn
import jax.numpy as jnp
import jax.random
import ml_collections
import numpy as np
from scenic.dataset_lib import datasets
from scenic.model_lib import models
from scenic.model_lib.base_models import classification_model
from scenic.model_lib.base_models import multilabel_classification_model
from scenic.train_lib import classification_trainer
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils
import tensorflow as tf
import tensorflow_datasets as tfds


class ClassificationTrainerTest(absltest.TestCase):
  """Tests the default trainer on single device setup."""

  def setUp(self):
    super(ClassificationTrainerTest, self).setUp()
    self.test_dir = tempfile.mkdtemp()
    # make sure tf does not allocate gpu memory
    tf.config.experimental.set_visible_devices([], 'GPU')

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super(ClassificationTrainerTest, self).tearDown()

  def get_train_state(self, rng, fake_batch_logits):
    """Generates the initial training state."""
    config = ml_collections.ConfigDict({
        'lr_configs': {
            'base_learning_rate': 0.1,
        },
        'optimizer': 'sgd',
    })

    # define a fake model that always outputs the same logits: fake_batch_logits
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
    lr_fn = lr_schedules.get_learning_rate_fn(config)
    optimizer_config = optimizers.get_optax_optimizer_config(config)
    tx = optimizers.get_optimizer(optimizer_config, lr_fn)
    opt_state = jax.jit(tx.init, backend='cpu')(initial_params)
    init_train_state = jax_utils.replicate(
        train_utils.TrainState(
            global_step=0,
            params=initial_params,
            tx=tx, opt_state=opt_state,
            model_state=init_model_state,
            rng=jax.random.PRNGKey(0)))
    return FakeFlaxModel(), init_train_state

  def train_and_evaluation(self, model, train_state, fake_batches, metrics_fn):
    """Given the train_state, trains the model on fake batches."""
    eval_metrics = []
    fake_batches_replicated = jax_utils.replicate(fake_batches)

    eval_step_pmapped = jax.pmap(
        functools.partial(
            classification_trainer.eval_step,
            flax_model=model,
            metrics_fn=metrics_fn,
            debug=False),
        axis_name='batch',
        donate_argnums=(1,),
    )
    for fake_batch in fake_batches_replicated:
      metrics, _ = eval_step_pmapped(train_state, fake_batch)
      metrics = train_utils.unreplicate_and_get(metrics)
      eval_metrics.append(metrics)
    eval_metrics = train_utils.stack_forest(eval_metrics)
    eval_summary = jax.tree_util.tree_map(lambda x: x.sum(), eval_metrics)
    for key, val in eval_summary.items():
      eval_summary[key] = val[0] / val[1]
    return eval_summary

  def test_classifaction_model_evaluate(self):
    """Test trainer evaluate end to end with classification model metrics."""
    # define a fix output for the fake flax model
    fake_batch_logits = np.tile([.5, .2, .7, 0.0], (4, 1))
    # 4 evaluation batches of size 4.
    fake_batches = [
        {
            'inputs': None,
            'label': np.array([3, 2, 1, 0])
        },
        {
            'inputs': None,
            'label': np.array([0, 3, 2, 0])
        },
        {
            'inputs': None,
            'label': np.array([0, 0, 0, 0])
        },
        {
            'inputs': None,
            'label': np.array([1, 1, 1, 1])
        },
    ]

    rng = jax.random.PRNGKey(0)
    model, train_state = self.get_train_state(rng, fake_batch_logits)
    eval_summary = self.train_and_evaluation(
        model, train_state, fake_batches,
        functools.partial(
            classification_model.classification_metrics_function,
            target_is_onehot=False))

    def batch_loss(logits, targets):
      # softmax cross-entropy loss
      one_hot_targets = np.eye(4)[targets]
      loss = -np.sum(one_hot_targets * nn.log_softmax(logits), axis=-1)
      return loss

    expected_accuracy = 2.0 / 16.0  # FakeFlaxModule always predicts class 2.
    expected_loss = np.mean(
        [batch_loss(fake_batch_logits, b['label']) for b in fake_batches])

    self.assertEqual(expected_accuracy, eval_summary['accuracy'])
    np.testing.assert_allclose(expected_loss, eval_summary['loss'], atol=1e-6)

  def test_multi_label_classifaction_model_evaluate(self):
    """Test trainer evaluate with multi-label classification model metrics."""
    # define a fix output for the fake flax model
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
      # sigmoid cross-entropy loss
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

  def test_trainer(self):
    """Test training for two epochs on MNIST with a small model."""

    rng = jax.random.PRNGKey(0)
    np.random.seed(0)
    config = ml_collections.ConfigDict({
        'dataset_name': 'mnist',
        'data_dtype_str': 'float32',
        'rng_seed': 0,
        'lr_configs': {
            'learning_rate_schedule': 'compound',
            'factors': 'constant * cosine_decay',
            'steps_per_cycle': 100,
            'base_learning_rate': 0.1,
        },
        'hid_sizes': [20, 10],
        'model_dtype_str': 'float32',
        'optimizer': 'momentum',
        'optimizer_configs': {
            'momentum': 0.9
        },
        'batch_size': 128,
        'eval_batch_size': 64,
        'l2_decay_factor': .0005,
        'max_grad_norm': None,
        'label_smoothing': None,
        'write_summary': None,  # no summary writing
        'checkpoint': False,  # no checkpointing
        'debug_eval': False,
        'debug_train': False,
        'xprof': False,
    })

    with tfds.testing.mock_data(num_examples=1024):
      model_cls = models.get_model_cls('fully_connected_classification')
      dataset_builder = datasets.get_dataset('mnist')

      dataset = dataset_builder(
          batch_size=config.batch_size,
          eval_batch_size=config.eval_batch_size,
          num_shards=jax.local_device_count(),
          dtype_str=config.data_dtype_str)

      config.num_training_steps = 100
      config.log_eval_steps = 50
      config.num_training_epochs = None
      _, train_summary, eval_summary = classification_trainer.train(
          rng=rng,
          config=config,
          model_cls=model_cls,
          dataset=dataset,
          workdir=self.test_dir,
          writer=metric_writers.LoggingWriter())

    self.assertGreaterEqual(train_summary['accuracy'], 0.0)
    self.assertLess(train_summary['loss'], 5.0)
    self.assertGreaterEqual(eval_summary['accuracy'], 0.0)
    self.assertLess(eval_summary['loss'], 5.0)


if __name__ == '__main__':
  absltest.main()
