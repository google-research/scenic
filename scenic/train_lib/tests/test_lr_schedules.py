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

"""Unit tests for lr_schedules.py."""

from absl.testing import absltest
import ml_collections
from scenic.train_lib import lr_schedules
import tensorflow as tf


class LearningRateScchedulesTest(absltest.TestCase):
  """Tests different learning rate schedules ."""

  def test_constant(self):
    """Test constant schedule works correctly."""
    config = ml_collections.ConfigDict(
        dict(
            lr_configs={
                'learning_rate_schedule': 'compound',
                'factors': 'constant',
                'base_learning_rate': .1,
            }))
    lr_fn = lr_schedules.get_learning_rate_fn(config)
    config = config.lr_configs
    for step in range(400):
      expected_learning_rate = config.base_learning_rate
      self.assertAlmostEqual(lr_fn(step), expected_learning_rate)

  def test_constant_linear_warmup(self):
    """Test that linear warmup schedule works correctly."""
    warmup_steps = 100
    warmup_alpha = 0.1
    config = ml_collections.ConfigDict(
        dict(
            lr_configs={
                'learning_rate_schedule': 'compound',
                'factors': 'constant*linear_warmup',
                'base_learning_rate': 1.0,
                'warmup_steps': warmup_steps,
                'warmup_alpha': warmup_alpha
            }))
    lr_fn = lr_schedules.get_learning_rate_fn(config)
    for step in range(400):
      if step == 0:
        self.assertEqual(lr_fn(step), warmup_alpha)
      if step > 0 and step < warmup_steps:
        self.assertGreater(lr_fn(step), lr_fn(step - 1))
      if step >= warmup_steps:
        self.assertEqual(lr_fn(step), 1.0)

  def test_polynomial_decay(self):
    """Test polynomial schedule works correctly."""
    config = ml_collections.ConfigDict(
        dict(
            lr_configs={
                'learning_rate_schedule': 'compound',
                'factors': 'constant*polynomial',
                'decay_steps': 200,
                'power': 2.0,
                'base_learning_rate': .1,
                'end_factor': .01
            }))
    lr_fn = lr_schedules.get_learning_rate_fn(config)
    config = config.lr_configs
    tf_polynomial_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=config['base_learning_rate'],
        decay_steps=config['decay_steps'],
        end_learning_rate=config['end_factor'] * config['base_learning_rate'],
        power=config['power'])
    for step in range(400):
      expected_learning_rate = tf_polynomial_decay(step=step).numpy()
      self.assertAlmostEqual(lr_fn(step), expected_learning_rate)

  def test_exponential_decay(self):
    """Test exponential schedule works correctly."""
    for test_params in [
        {'decay_steps': 200, 'decay_rate': 0.99, 'staircase': True},
        {'decay_steps': 200, 'decay_rate': 0.99, 'staircase': False},
    ]:
      config = ml_collections.ConfigDict(
          dict(
              lr_configs={
                  'learning_rate_schedule': 'compound',
                  'factors': 'constant*exponential_decay',
                  'base_learning_rate': 0.1,
                  'decay_steps': test_params['decay_steps'],
                  'decay_rate': test_params['decay_rate'],
                  'staircase': test_params['staircase'],
              }
          )
      )
      lr_fn = lr_schedules.get_learning_rate_fn(config)
      config = config.lr_configs
      tf_exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
          initial_learning_rate=config['base_learning_rate'],
          decay_steps=config['decay_steps'],
          decay_rate=config['decay_rate'],
          staircase=config['staircase'],
      )
      for step in range(400):
        expected_learning_rate = tf_exponential_decay(step=step).numpy()
        self.assertAlmostEqual(lr_fn(step), expected_learning_rate)

  def test_cosine_decay(self):
    """Test cosine schedule works correctly."""
    config = ml_collections.ConfigDict(
        dict(
            lr_configs={
                'learning_rate_schedule': 'compound',
                'factors': 'cosine_decay',
                'steps_per_cycle': 100,
                't_mul': 2.,
                'm_mul': .5,
                'alpha': 0.3,
                'base_learning_rate': 1.,
            }))
    lr_fn = lr_schedules.get_learning_rate_fn(config)
    config = config.lr_configs
    tf_cosine_decay = tf.keras.experimental.CosineDecayRestarts(
        initial_learning_rate=config['base_learning_rate'],
        first_decay_steps=config['steps_per_cycle'],
        t_mul=config['t_mul'],
        m_mul=config['m_mul'],
        alpha=config['alpha'],
    )
    for step in range(400):
      expected_learning_rate = tf_cosine_decay(step=step).numpy()
      self.assertAlmostEqual(lr_fn(step), expected_learning_rate)


if __name__ == '__main__':
  absltest.main()
