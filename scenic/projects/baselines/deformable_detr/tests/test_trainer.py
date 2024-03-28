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

"""Test for the DeformableDETR train script."""

import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from clu import metric_writers
from jax import flatten_util
import jax.numpy as jnp
import jax.random
import numpy as np
from scenic.dataset_lib import datasets
from scenic.projects.baselines.deformable_detr import trainer
from scenic.projects.baselines.deformable_detr.configs import mini_config
from scenic.projects.baselines.deformable_detr.model import DeformableDETRModel
from scenic.projects.baselines.detr.tests import test_util
import tensorflow as tf
import tensorflow_datasets as tfds


class DETRTrainerTest(parameterized.TestCase):
  """Tests the DeformableDETR trainer on single device setup."""

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()
    # Make sure tf does not allocate GPU memory.
    tf.config.experimental.set_visible_devices([], 'GPU')

    self.config = mini_config.get_config()
    num_shards = jax.local_device_count()
    self.config.batch_size = num_shards * 2
    self.config.eval_batch_size = num_shards * 2

    self.model_cls = DeformableDETRModel

    with tfds.testing.mock_data(
        num_examples=50,
        as_dataset_fn=test_util.generate_fake_dataset(num_examples=50)):
      dataset_builder = datasets.get_dataset(self.config.dataset_name)
      self.dataset = dataset_builder(
          batch_size=self.config.batch_size,
          eval_batch_size=self.config.eval_batch_size,
          num_shards=num_shards,
          dtype_str=self.config.data_dtype_str,
          dataset_configs=self.config.dataset_configs)

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_deformable_detr_trainer(self):
    """Test training for a few steps of a mini DETR on a mini COCO dataset."""

    rng = jax.random.PRNGKey(0)
    np.random.seed(0)

    with tfds.testing.mock_data(
        num_examples=50,
        as_dataset_fn=test_util.generate_fake_dataset(num_examples=50)):
      _, train_summary, eval_summary = trainer.train_and_evaluate(
          rng=rng,
          config=self.config,
          model_cls=self.model_cls,
          dataset=self.dataset,
          workdir=self.test_dir,
          writer=metric_writers.LoggingWriter())

    expected_summary_keys = [
        'loss_bbox',
        'loss_class',
        'loss_giou',
        'total_loss',
        'object_detection_loss',
    ]
    for i in range(self.config.num_decoder_layers - 1):
      expected_summary_keys += [
          f'loss_class_aux{i}', f'loss_bbox_aux{i}', f'loss_giou_aux{i}'
      ]
    expected_global_summary_keys = [
        'AP', 'AP_50', 'AP_75', 'AP_small', 'AP_medium', 'AP_large', 'AR_max_1',
        'AR_max_10', 'AR_max_100', 'AR_small', 'AR_medium', 'AR_large'
    ]

    # Check summaries.
    self.assertSameElements(expected_summary_keys, train_summary.keys())
    self.assertSameElements(
        expected_summary_keys + expected_global_summary_keys,
        eval_summary.keys())

  @parameterized.named_parameters(('update_batch_stats', True),
                                  ('freeze_batch_stats', False))
  def test_deformable_detr_train_step(self, update_batch_stats):
    """Test DeformableDETR train_step directly."""
    rng = jax.random.PRNGKey(0)
    np.random.seed(0)

    model, tx, train_state, _, _, _ = trainer.get_model_and_tx_and_train_state(
        rng=rng,
        dataset=self.dataset,
        config=self.config,
        model_cls=self.model_cls,
        workdir='',
        input_spec=[(self.dataset.meta_data['input_shape'],
                     self.dataset.meta_data.get('input_dtype', jnp.float32))])

    train_step = trainer.get_train_step(
        apply_fn=model.flax_model.apply,
        loss_and_metrics_fn=model.loss_and_metrics_function,
        tx=tx,
        update_batch_stats=update_batch_stats,
        debug=False)

    train_step_pmapped = jax.pmap(
        train_step,
        axis_name='batch',
        donate_argnums=(0,),
    )

    with tfds.testing.mock_data(
        num_examples=50,
        as_dataset_fn=test_util.generate_fake_dataset(num_examples=50)):
      train_batch = next(self.dataset.train_iter)

    init_params, _ = flatten_util.ravel_pytree(train_state.params)
    init_model_state, _ = flatten_util.ravel_pytree(train_state.model_state)

    train_state, _, _ = train_step_pmapped(train_state, train_batch)

    self.assertFalse(jnp.array_equal(init_params, train_state.params))
    if update_batch_stats:
      # make sure that state were updated in the train step
      self.assertFalse(
          jnp.array_equal(init_model_state,
                          flatten_util.ravel_pytree(
                              train_state.model_state)[0]))

    else:
      # make sure the state were stayed frozen in the train step
      self.assertTrue(
          jnp.array_equal(init_model_state,
                          flatten_util.ravel_pytree(
                              train_state.model_state)[0]))


if __name__ == '__main__':
  absltest.main()
