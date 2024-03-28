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

"""Test for the DeformableDETR eval functions."""

import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from jax import flatten_util
import jax.numpy as jnp
import jax.random
import numpy as np
from scenic.dataset_lib import datasets
from scenic.projects.baselines.deformable_detr import evaluate as ddetr_eval
from scenic.projects.baselines.deformable_detr import input_pipeline_detection  # pylint: disable=unused-import
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

  def test_deformable_detr_eval_step(self):
    """Test DeformableDETR eval_step directly."""
    rng = jax.random.PRNGKey(0)
    np.random.seed(0)

    model, _, train_state, _, _, _ = trainer.get_model_and_tx_and_train_state(
        rng=rng,
        dataset=self.dataset,
        config=self.config,
        model_cls=self.model_cls,
        workdir='',
        input_spec=[(self.dataset.meta_data['input_shape'],
                     self.dataset.meta_data.get('input_dtype', jnp.float32))])

    eval_step = ddetr_eval.get_eval_step(
        flax_model=model.flax_model,
        loss_and_metrics_fn=model.loss_and_metrics_function,
        logits_to_probs_fn=model.logits_to_probs,
        debug=False)
    eval_step_pmapped = jax.pmap(
        eval_step, axis_name='batch', donate_argnums=(1,))

    with tfds.testing.mock_data(
        num_examples=50,
        as_dataset_fn=test_util.generate_fake_dataset(num_examples=50)):
      eval_batch = next(self.dataset.valid_iter)

    init_params, _ = flatten_util.ravel_pytree(train_state.params)
    _, predictions, metrics = eval_step_pmapped(train_state, eval_batch)
    after_params, _ = flatten_util.ravel_pytree(train_state.params)
    self.assertTrue(jnp.array_equal(init_params, after_params))

    exp_pred_keys = {'pred_logits', 'pred_boxes', 'pred_probs'}
    self.assertSameElements(predictions.keys(), exp_pred_keys)
    exp_metric_keys = ['loss_bbox', 'loss_class', 'loss_giou', 'total_loss']
    for i in range(self.config.num_decoder_layers - 1):
      aux = [f'loss_class_aux{i}', f'loss_bbox_aux{i}', f'loss_giou_aux{i}']
      exp_metric_keys += aux
    self.assertSameElements(exp_metric_keys, metrics.keys())


if __name__ == '__main__':
  absltest.main()
