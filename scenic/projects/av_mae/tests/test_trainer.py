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

"""Tests the MAE trainers."""

import shutil
import tempfile

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from clu import metric_writers
import jax
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.projects.av_mae import trainer as avmae_trainer
from scenic.projects.av_mae import vit
import tensorflow as tf


def get_fake_config_mae(version='Test'):
  """Returns config for testing MAE."""

  patch = 16

  config = ml_collections.ConfigDict()
  config.model_name = 'vit_masked_autoencoder'
  config.model = ml_collections.ConfigDict()
  config.model.hidden_size = {'Test': 16,
                              'Ti': 192,
                              'S': 384,
                              'B': 768,
                              'L': 1024,
                              'H': 1280}[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [int(patch), int(patch)]
  config.model.num_heads = {'Test': 2,
                            'Ti': 3,
                            'S': 6,
                            'B': 12,
                            'L': 16,
                            'H': 16}[version]
  config.model.mlp_dim = {'Test': 64,
                          'Ti': 768,
                          'S': 1536,
                          'B': 3072,
                          'L': 4096,
                          'H': 5120}[version]
  config.model.num_layers = {'Test': 3,
                             'Ti': 12,
                             'S': 12,
                             'B': 12,
                             'L': 24,
                             'H': 32}[version]
  config.model.representation_size = None
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.1
  config.model_dtype_str = 'float32'
  config.model.classifier = 'token'

  config.model.decoder_config = ml_collections.ConfigDict()
  config.model.decoder_config.hidden_size = {
      'Test': 16,
      'Ti': 192,
      'S': 384,
      'B': 768,
      'L': 1024,
      'H': 1280
  }[version]
  config.model.decoder_config.num_heads = {
      'Test': 2,
      'Ti': 3,
      'S': 6,
      'B': 12,
      'L': 16,
      'H': 16
  }[version]
  config.model.decoder_config.mlp_dim = {
      'Test': 64,
      'Ti': 768,
      'S': 1536,
      'B': 3072,
      'L': 4096,
      'H': 5120
  }[version]
  config.model.decoder_config.num_layers = {
      'Test': 3,
      'Ti': 12,
      'S': 12,
      'B': 12,
      'L': 24,
      'H': 32
  }[version]
  config.model.decoder_config.attention_dropout_rate = 0.
  config.model.decoder_config.dropout_rate = 0.
  config.model.decoder_config.stochastic_depth = 0.

  config.dataset_configs = ml_collections.ConfigDict()

  # Masked loss
  config.masked_feature_loss = ml_collections.ConfigDict()
  config.masked_feature_loss.target = 'rgb'
  config.masked_feature_loss.token_mask_probability = 0.75

  # Training.
  config.trainer_name = 'feature_regression_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 0.05
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = 0.02  # In the MFP paper.
  config.label_smoothing = None
  config.num_training_epochs = 1
  config.log_eval_steps = 1000
  config.batch_size = 8
  config.rng_seed = 42
  config.init_head_bias = 0

  # Learning rate.
  steps_per_epoch = 2
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 1.6e-3
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*linear_decay'
  config.lr_configs.total_steps = total_steps
  config.lr_configs.end_learning_rate = 0
  config.lr_configs.base_learning_rate = base_lr

  config.checkpoint = False
  config.debug_train = False
  config.debug_eval = False

  return config


class FakeDataset():
  """Fake dataset for testing."""

  def __init__(self, batch_size=8, batch_size_eval=8,
               input_shape=(-1, 64, 64, 3), num_classes=5):
    logging.info('Construct test dataset with batch size %s', batch_size)
    self.batch_size = batch_size
    self.batch_size_eval = batch_size_eval
    self.num_classes = num_classes

    self.meta_data = {
        'input_shape': input_shape,
        'input_dtype': 'float32',
        'num_train_examples': batch_size * 2,
        'num_eval_examples': batch_size_eval * 2,
        'target_is_onehot': False,
        'num_classes': self.num_classes
    }

  def fake_batch(self):
    shape_inputs = [self.batch_size] + list(self.meta_data['input_shape'][1:])
    return {
        'inputs':
            np.random.uniform(size=tuple(shape_inputs)),
        'label':
            np.random.randint(
                low=0, high=self.num_classes, size=(self.batch_size))
    }

  def iter_data(self):
    while True:
      yield self.fake_batch()

  @property
  def train_iter(self):
    ds_iter = map(dataset_utils.shard, self.iter_data())
    yield from ds_iter

  @property
  def eval_iter(self):
    yield from self.train_iter

  @property
  def test_iter(self):
    yield from self.train_iter


def make_fake_dataset(batch_size, batch_size_eval, input_shape=(-1, 64, 64, 3)):
  ds = FakeDataset(batch_size, batch_size_eval, input_shape)
  return dataset_utils.Dataset(
      ds.train_iter,
      ds.eval_iter,
      ds.test_iter,
      ds.meta_data)


class TrainerTest(parameterized.TestCase):
  """Tests the default trainer on single device setup."""

  def setUp(self):
    super(TrainerTest, self).setUp()
    self.test_dir = tempfile.mkdtemp()
    # Make sure Tensorflow does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super(TrainerTest, self).tearDown()

  def test_trainer_mae(self):
    """Tests MAE trainer."""
    model_cls = vit.ViTMaskedAutoencoderModel
    trainer = avmae_trainer.train
    rng = jax.random.PRNGKey(0)
    config = get_fake_config_mae()
    workdir = self.test_dir
    dataset = make_fake_dataset(config.batch_size, config.batch_size)
    writer = metric_writers.LoggingWriter()

    _, train_summary, eval_summary = trainer(
        rng=rng,
        config=config,
        model_cls=model_cls,
        dataset=dataset,
        workdir=workdir,
        writer=writer)

    self.assertLess(train_summary['total_loss'], 1E5)
    self.assertLess(eval_summary['mean_squared_error'], 1E5)


if __name__ == '__main__':
  absltest.main()

