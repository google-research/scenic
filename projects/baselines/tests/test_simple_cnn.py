"""Regression test for a simple cnn model on MNIST."""

import shutil
import tempfile

from absl.testing import absltest
from clu import metric_writers
import jax.random
import ml_collections
import numpy as np
from scenic.dataset_lib import datasets
from scenic.model_lib import models
from scenic.train_lib import trainers
import tensorflow as tf
import tensorflow_datasets as tfds


class SimpleCNNClassificationTest(absltest.TestCase):
  """Regression test for a simple CNN model on MNIST."""

  def setUp(self):
    super(SimpleCNNClassificationTest, self).setUp()
    self.test_dir = tempfile.mkdtemp()
    # Make sure tf does not allocate GPU memory.
    tf.config.experimental.set_visible_devices([], 'GPU')
    self.trainer = trainers.get_trainer('classification_trainer')

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super(SimpleCNNClassificationTest, self).tearDown()

  def test_simple_cnn_classification(self):
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
        'num_filters': [32, 64],
        'kernel_sizes': [3, 3],
        'use_bias': [False, True],
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

    model_cls = models.get_model_cls('simple_cnn_classification')
    with tfds.testing.mock_data(num_examples=1024):
      dataset_builder = datasets.get_dataset('mnist')
      dataset = dataset_builder(
          batch_size=config.batch_size,
          eval_batch_size=config.eval_batch_size,
          num_shards=jax.local_device_count(),
          dtype_str=config.data_dtype_str)

    config.num_training_steps = 100
    config.log_eval_steps = 50
    config.num_training_epochs = None
    _, train_summary, eval_summary = self.trainer(
        rng=rng,
        config=config,
        model_cls=model_cls,
        dataset=dataset,
        workdir=self.test_dir,
        writer=metric_writers.LoggingWriter()  # pytype: disable=attribute-error
    )

    self.assertGreaterEqual(train_summary['accuracy'], 0.0)
    self.assertLess(train_summary['loss'], 5.0)
    self.assertGreaterEqual(eval_summary['accuracy'], 0.0)
    self.assertLess(eval_summary['loss'], 5.0)


if __name__ == '__main__':
  absltest.main()
