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

"""Standalone eval binary."""

from absl import flags
from absl import logging
import chex
from clu import metric_writers
from flax import jax_utils
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app as scenic_app
from scenic.projects.boundary_attention import eval_manager
from scenic.projects.boundary_attention import trainer
from scenic.projects.boundary_attention.dataset_lib import dataloader
from scenic.projects.boundary_attention.models import all_models
from scenic.train_lib import train_utils as scenic_train_utils

flags.DEFINE_string('dataset_dir', '', 'Dataset directory.')
flags.DEFINE_string('checkpoint_path', '', 'Checkpoint path.')
flags.DEFINE_integer('checkpoint_step', -1, 'Checkpoint step.')
flags.DEFINE_string('weights_path', '', 'Pretrained weights path.')

FLAGS = flags.FLAGS
EVAL_ARTIFACT_DESCRIPTION = 'Last evaluated checkpoint'
FINAL_CKPT_ARTIFACT_DESCRIPTION = 'Final checkpoint'


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,  # pylint: disable=unused-argument
         writer: metric_writers.MetricWriter):
  """Main function for Boundary Attention evals."""

  # Update config if flags defined
  if len(FLAGS.dataset_dir) > 0:  # pylint: disable=g-explicit-length-test
    config.dataset.dataset_dir = FLAGS.dataset_dir
  if len(FLAGS.checkpoint_path) > 0:  # pylint: disable=g-explicit-length-test
    config.init_from.checkpoint_path = FLAGS.checkpoint_path
  if FLAGS.checkpoint_step != -1:
    config.init_from.checkpoint_step = FLAGS.checkpoint_step
  if len(FLAGS.weights_path) > 0:  # pylint: disable=g-explicit-length-test
    config.init_from.params_path = FLAGS.weights_path

  model_cls = all_models.get_model_cls(config.model.name)
  # Create the dataset for the primary eval, which we assume uses the same
  # dataset as the train job, but a different split. Other evals create their
  # own dataset instances.
  data_rng, model_rng = jax.random.split(rng)
  dataset = dataloader.get_dataloader(config, data_rng)
  if config.disable_pmap_and_jit:
    chex.fake_pmap_and_jit().start()

  dataset_metadata = dataset.meta_data

  # Build the loss_and_metrics_fn, metrics, and flax_model.
  model = model_cls(config, dataset_metadata)

  evaler = eval_manager.EvalManager(model, config, model_rng)

  checkpoint_dir = FLAGS.checkpoint_path
  latest_checkpoint = checkpoints.latest_checkpoint(checkpoint_dir)

   # Initialize model.
  rng, params_rng, dropout_rng = jax.random.split(key=rng, num=3)

  input_specs = []
  for input_shape in dataset_metadata['input_shape']:
    input_spec = (input_shape, dataset_metadata.get('input_dtype',
                                                    jnp.float32))
    input_specs.append(input_spec)

  (params, model_state, _,
   _) = scenic_train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=input_specs,
       config=config,
       rngs={'params': params_rng,
             'dropout': dropout_rng})

  _, train_rng = jax.random.split(rng)

  train_state = scenic_train_utils.TrainState(
      global_step=0,
      params=params,
      model_state=model_state,
      rng=train_rng)

  train_state, step = trainer.maybe_restore_model_or_params(model, train_state,
                                                            workdir, config)
  train_state = jax_utils.replicate(train_state)

  # Evaluate
  evaler.run_one_eval(
      train_state, step, dataset, writer, is_final=True)

  # Wait for all hosts to finish before proceeding to next checkpoint:
  logging.info('Reached barrier on host %s for checkpoint %s',
               jax.process_index(), latest_checkpoint)
  scenic_train_utils.barrier()

  # Wait for all hosts to finish before exiting:
  logging.info('Reached final barrier on host %s', jax.process_index())
  scenic_train_utils.barrier()

  # Shut down work unit and exit:
  logging.info('Exiting.')


if __name__ == '__main__':
  scenic_app.run(main=main)
