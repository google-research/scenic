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

"""Training script for Streaming DVC models."""

import functools
import time
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
from flax import jax_utils
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
from scenic.common_lib import debug_utils
from scenic.dataset_lib import dataset_utils
from scenic.projects.streaming_dvc import evaluate
from scenic.projects.streaming_dvc import partition_utils
from scenic.projects.streaming_dvc import train_utils as streaming_dvc_train_utils
from scenic.train_lib import lr_schedules
from scenic.train_lib import train_utils


def train_and_evaluate(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Any,
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter):
  """Main training loop lives in this function.

  Args:
    rng: JAX PRNGKey.
    config: Configurations of the experiment.
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    dataset: The dataset that has train_iter, eval_iter, meta_data, and
      optionally, test_iter.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    train_sate that has the state of training (including current
      global_step, model_state, rng, and the optimizer), train_summary
      and eval_summary which are dict of metrics. These outputs are used for
      regression testing.
  """
  is_host = jax.process_index() == 0
  logging.info('Training with config: %s', config)
  logging.info('Dataset metadata %s', dataset.meta_data)

  model = model_cls(config, dataset.meta_data)
  rng, init_rng = jax.random.split(rng)
  input_spec = [
      (dataset.meta_data['input_shape'],
       dataset.meta_data.get('input_dtype', jnp.float32))]
  if config.get('additional_input_spec', []):
    input_spec.extend(config.additional_input_spec)
  (params, model_state, num_trainable_params, gflops) = (
      train_utils.initialize_model(
          model_def=model.flax_model,
          input_spec=input_spec,
          config=config,
          rngs=init_rng,
      )
  )

  # Obtain the mapping of parameter names to frozen or not
  frozen_mapping = partition_utils.create_frozen_mask_from_regex(
      params, config.get('frozen_params')
  )

  lr_fn = lr_schedules.get_learning_rate_fn(config)
  _, train_rng = jax.random.split(rng)
  train_state, num_learnable_params, num_frozen_params = (
      partition_utils.create_partitioned_train_state(
          params, frozen_mapping, config, 0, model_state, train_rng, lr_fn))

  # Convert partitioned train state to a normal one for loading from pretrained
  # checkpoints, or from the saved one, without any changes.
  train_state = partition_utils.convert_to_train_state(train_state)

  # T5 models have a 'params_axes' model_state which is somehow not saved in the
  # checkpoint (being removed after a first train_step). Following Vid2Seq to
  # remove it when loading the checkpoint. It won't affect if the model does
  # have the 'params_axes' model_state.
  train_state, params_axes = streaming_dvc_train_utils.pop_axes_names(
      train_state, 'params_axes')
  train_state = checkpoints.restore_checkpoint(workdir, train_state)
  train_state = streaming_dvc_train_utils.re_add_axis_names(
      train_state, params_axes, 'params_axes')

  start_step = int(train_state.global_step)
  if start_step == 0:
    train_state, start_step = streaming_dvc_train_utils.load_weights(
        train_state, config)
    step0_log = {'num_params': num_trainable_params,
                 'num_learnable_params': num_learnable_params,
                 'num_frozen_params': num_frozen_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  # Now convert back to the partitioned train step.
  train_state = partition_utils.convert_to_partitioned_train_state(
      train_state, frozen_mapping)

  train_step_pmapped = jax.pmap(
      functools.partial(
          partition_utils.train_step_partitioned,
          flax_model=model.flax_model,
          loss_and_metrics_fn=model.loss_function,
          learning_rate_fn=lr_fn,
          debug=config.debug_train,
      ),
      axis_name='batch', donate_argnums=(0,),
  )

  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data)
  log_eval_steps = config.get('log_eval_steps', steps_per_epoch)
  log_summary_steps = config.get('log_summary_steps', 20)
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None
  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  chrono = train_utils.Chrono()
  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)

  def write_note(note):
    if is_host:
      platform.work_unit().set_notes(note)
    logging.info(note)

  logging.info('Starting training loop at step %d.', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer)
  hooks = []
  if is_host:
    hooks.append(report_progress)
  if config.get('xprof', True) and is_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      train_state, lr, train_predictions, metrics = train_step_pmapped(
          train_state, train_batch)
      train_metrics.append(metrics)
      extra_training_logs.append({'learning_rate': lr})
    for h in hooks:
      h(step)
    chrono.pause()
    del train_predictions

    if (step % log_summary_steps == 0) or (step == total_steps - 1):
      if is_host:
        chrono.tick(step, writer, write_note)
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(
              train_utils.unreplicate_and_get, train_metrics),
          extra_training_logs=jax.tree_util.tree_map(
              train_utils.unreplicate_and_get, extra_training_logs),
          writer=writer)
      train_metrics, extra_training_logs = [], []

    eval_first_step = config.get('eval_first_step', True) and step == 1
    do_eval = not config.get('not_eval', False)
    if ((step % log_eval_steps == 0) or (step == total_steps) or (
        eval_first_step)) and do_eval:
      logging.info('Starting evaluation ...')
      # Convert back to normal train state for doing evaluation without any
      # code changes.
      train_state = partition_utils.convert_to_train_state(train_state)
      start_time = time.time()
      with report_progress.timed('eval'):
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        last_eval_results, last_eval_metrics = evaluate.inference_on_dataset(
            model.flax_model,
            train_state, dataset,
            eval_batch_size=eval_batch_size,
            is_host=is_host,
            save_dir=workdir,
            step=step,
            config=config)
        last_eval_step = step
        train_utils.log_eval_summary(
            step=last_eval_step, eval_metrics=last_eval_metrics,
            extra_eval_summary=last_eval_results, writer=writer)
      duration = time.time() - start_time
      logging.info('Done with evaluation: %.4f sec.', duration)
      # Convert back to partitioned train state for training.
      train_state = partition_utils.convert_to_partitioned_train_state(
          train_state, frozen_mapping)
      writer.flush()

    if ((step % checkpoint_steps == 0 and step > 0) or
        (step == total_steps)):
      with report_progress.timed('checkpoint'):
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        if is_host:
          unrep_train_state = jax_utils.unreplicate(train_state)
          logging.info('Parameter summary after checkpoint:')
          debug_utils.log_param_shapes(
              unrep_train_state.params_learned,  # pytype: disable=attribute-error
              description='Learned params')
          if len(unrep_train_state.params_frozen):  # pylint: disable=g-explicit-length-test
            debug_utils.log_param_shapes(
                unrep_train_state.params_frozen,  # pytype: disable=attribute-error
                description='Frozen params')
          # Convert to unpartitioned train state for saving and loading without
          # needing any code changes.
          unrep_train_state = partition_utils.convert_to_train_state(
              unrep_train_state)
          train_utils.save_checkpoint(
              workdir, unrep_train_state,
              max_to_keep=config.get('checkpoint_max_to_keep', 1))
          del unrep_train_state
    chrono.resume()  # Un-pause now.

  train_utils.barrier()
  return train_state, train_summary, eval_summary
