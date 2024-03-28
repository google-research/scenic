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

"""Training Script."""

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Type

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
import flax
from flax import jax_utils
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import base_model
from scenic.projects.unloc import eval_utils as unloc_eval_utils
from scenic.projects.unloc import optimizer_utils
from scenic.projects.unloc import train_utils as unloc_train_utils
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils

# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]
LrFn = Callable[[jnp.ndarray], jnp.ndarray]


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Type[base_model.BaseModel],
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[
    train_utils.TrainState, Optional[Dict[str, Any]], Optional[Dict[str, Any]]
]:
  """Main training loop lives in this function.

  Given the model class and dataset, it prepares the items needed to run the
  training, including the TrainState.

  Args:
    rng: Jax rng key.
    config: Configurations of the experiment.
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    dataset: The dataset that has train_iter, eval_iter, meta_data, and
      optionally, test_iter.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    train_state that has the state of training (including current
      global_step, model_state, rng, and the optimizer), train_summary
      and eval_summary which are dict of metrics. These outputs are used for
      regression testing.
  """
  lead_host = jax.process_index() == 0
  # Build the loss_fn, metrics, and flax_model.
  model = model_cls(config, dataset.meta_data)

  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  (params, model_state, num_trainable_params, gflops) = (
      unloc_train_utils.initialize_model_with_pytree(
          model_def=model.flax_model,
          input_spec={
              'inputs': unloc_train_utils.create_input_spec(
                  dataset.meta_data['input_shape'],
                  dataset.meta_data['input_dtype'],
              )
          },
          config=config,
          rngs=init_rng,
      )
  )

  # Create optimizer.
  lr_fn = lr_schedules.get_learning_rate_fn(config)
  if config.get('layer_prefix_to_base_lrs') is not None:
    tx = optimizer_utils.optimizer_with_multi_lrs(config, params)
  else:
    optimizer_config = optimizers.get_optax_optimizer_config(config)
    # If the config is already an optax-compatible config, better call directly:
    #   optimizers.get_optimizer(config.optimizer_configs, lr_fn)
    tx = optimizers.get_optimizer(optimizer_config, lr_fn, params=params)
  # We jit this, such that the arrays that are created on the same device as the
  # input is, in this case the CPU. Else they'd be on device[0].
  opt_state = jax.jit(tx.init, backend='cpu')(params)

  _, train_rng = jax.random.split(rng)

  # Create chrono class to track and store training statistics and metadata:
  chrono = train_utils.Chrono()

  train_state = train_utils.TrainState(
      global_step=0,
      opt_state=opt_state,
      tx=tx,
      params=params,
      model_state=model_state,
      rng=train_rng,
      metadata={'chrono': chrono.save()})
  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state)
  chrono.load(train_state.metadata['chrono'])
  if (start_step == 0 and config.get('init_from') is not None):
    if config.init_from.get('load_from_unloc_checkpoint', False):
      train_state = unloc_train_utils.init_from_unloc_checkpoint(
          config, train_state
      )

    if config.init_from.get('load_image_tower', True):
      if config.init_from.get('video_encoder'):
        train_state = unloc_train_utils.VIDEO_ENCODER_INIT_FN[
            config.init_from.video_encoder.model_type
        ](config, train_state)
      else:
        for modality_name, init_config in config.init_from.get(
            'video_encoders', {}
        ).items():
          train_state = unloc_train_utils.VIDEO_ENCODER_INIT_FN[
              init_config.model_type
          ](config, train_state, modality_name)
    if config.init_from.get('load_text_tower', True):
      train_state = unloc_train_utils.TEXT_ENCODER_INIT_FN[
          config.init_from.text_encoder.model_type
      ](config, train_state)
  elif start_step == 0:
    logging.info('Training completely from scratch.'
                 'Not restoring from any checkpoint.')
  # Replicate the optimizer, state, and rng.
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data)

  train_step_pmapped = jax.pmap(
      functools.partial(
          unloc_train_utils.train_step,
          task=config.dataset_configs.get('task', 'classification'),
          dataset=config.dataset_configs.get('name', ''),
          flax_model=model.flax_model,
          loss_fn=model.loss_function,
          lr_fn=lr_fn,
          metrics_fn=model.get_metrics_fn('train'),
          config=config,
          debug=config.debug_train,
      ),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0, 1),
  )
  eval_step_pmapped = jax.pmap(
      functools.partial(
          unloc_eval_utils.eval_step,
          task=config.dataset_configs.get('task', 'classification'),
          dataset=config.dataset_configs.get('name', ''),
          flax_model=model.flax_model,
          metrics_fn=model.get_metrics_fn('validation'),
          debug=config.debug_eval,
          all_gather_loss=config.get('all_gather_loss', False),
      ),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1,),
  )
  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  if config.dataset_configs.get('do_multicrop_test'):
    log_test_steps = (
        int(steps_per_epoch * config.dataset_configs.log_test_epochs) or
        steps_per_epoch)
    task = config.dataset_configs.get('task', 'classification')
    if task == 'temporal_localization' or task == 'highlight_detection':
      test_step_pmapped = jax.pmap(
          functools.partial(
              unloc_eval_utils.temporal_localization_test_step,
              dataset=config.dataset_configs.get('name', ''),
              task=task,
              flax_model=model.flax_model,
              num_prompts=config.dataset_configs.get('num_prompts', 1),
              output_per_class_displacements=config.get(
                  'output_per_class_displacements', True
              ),
              debug=False,
          ),
          axis_name='batch',
      )
    elif task == 'moment_retrieval':
      test_step_pmapped = jax.pmap(
          functools.partial(
              unloc_eval_utils.moment_retrieval_test_step,
              dataset=config.dataset_configs.get('name', ''),
              flax_model=model.flax_model,
              debug=False,
          ),
          axis_name='batch',
      )
    elif task == 'action_segmentation':
      test_step_pmapped = jax.pmap(
          functools.partial(
              unloc_eval_utils.action_segmentation_test_step,
              dataset=config.dataset_configs.get('name', ''),
              flax_model=model.flax_model,
              n_clips=config.get('multicrop_clips_per_device', 2),
              num_prompts=config.dataset_configs.get('num_prompts', 1),
              prompt_index=config.dataset_configs.get('prompt_index', None),
              debug=False,
          ),
          axis_name='batch',
      )
    else:
      raise ValueError(f'test_step not supported for task: {task}.')
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps

  # Ceil rounding such that we include the last incomplete batch.
  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None

  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)
  logging.info('Starting training loop at step %d.', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer)

  def write_note(note):
    if lead_host:
      platform.work_unit().set_notes(note)

  hooks = []
  if lead_host:
    hooks.append(report_progress)
  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  if start_step == 0:
    step0_log = {'num_trainable_params': num_trainable_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)

  test_step_fn = {
      'classification': (
          unloc_eval_utils.run_classification_test_steps_and_save_eval_summary
      ),
      'temporal_localization': (
          unloc_eval_utils.run_temporal_localization_test_steps_and_save_eval_summary
      ),
      'highlight_detection': (
          unloc_eval_utils.run_temporal_localization_test_steps_and_save_eval_summary
      ),
      'moment_retrieval': (
          unloc_eval_utils.run_moment_retrieval_test_steps_and_save_eval_summary
      ),
      'action_segmentation': (
          unloc_eval_utils.run_action_segmentation_test_steps_and_save_eval_summary
      ),
  }
  write_note(f'First step compilations...\n{chrono.note}')
  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      train_state, t_metrics, t_logs = train_step_pmapped(
          train_state, train_batch)
      # This will accumulate metrics in TPU memory up to the point that we log
      # them. This is no problem for small metrics but may be a problem for
      # large (e.g. segmentation) metrics. An alternative is to set
      # `log_summary_steps` to a small number, or to use
      # `train_utils.unreplicate_and_get` here instead of right before writing
      # summaries, but that means in each step, we have data transfer between
      # tpu and host, which might slow down the training.
      train_metrics.append(t_metrics)
      # Additional training logs: learning rate:
      t_logs = jax.tree_util.tree_map(jax_utils.unreplicate, t_logs)
      extra_training_logs.append(t_logs)
    for h in hooks:
      h(step)
    # Below are once-in-a-while ops -> pause.
    ###################### LOG TRAIN SUMMARY ########################
    if ((step % log_summary_steps == 1) or (step == total_steps) or
        (lead_host and chrono.warmup)):
      chrono.pause(wait_for=(train_metrics))
      if lead_host:
        chrono.tick(step, writer, write_note)
      # train_metrics is list of a dictionaries of metrics, where the shape of
      # the metrics[key] is [n_local_devices]. However, because metric functions
      # have a psum, we have already summed across the whole sharded batch, and
      # what's returned is n_local_devices copies of the same summed metric.
      # So we do unreplicate and fetch them to host using `unreplicate_and_get`.
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(train_utils.unreplicate_and_get,
                                               train_metrics),
          extra_training_logs=jax.tree_util.tree_map(jax.device_get,
                                                     extra_training_logs),
          writer=writer)
      # Reset metric accumulation for next evaluation cycle.
      train_metrics, extra_training_logs = [], []
      chrono.resume()
    ################### EVALUATION #######################
    if (step % log_eval_steps == 1) or (step == total_steps):
      chrono.pause(wait_for=(train_state.params))
      with report_progress.timed('eval'):
        eval_metrics = []
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        for _ in range(steps_per_eval):
          eval_batch = next(dataset.valid_iter)
          e_metrics, _ = eval_step_pmapped(train_state, eval_batch)
          eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))
        eval_summary = train_utils.log_eval_summary(
            step=step, eval_metrics=eval_metrics, writer=writer)
      writer.flush()
      del eval_metrics
      chrono.resume()
    ################### TESTING #######################
    if (config.dataset_configs.get('do_multicrop_test') and
        (step % log_test_steps == 1 and step > 1 or step == total_steps)):
      chrono.pause(wait_for=(train_state.params))
      with report_progress.timed('test'):
        test_step_fn[config.dataset_configs.get('task', 'classification')](
            config, step, dataset, test_step_pmapped, train_state, writer)
      chrono.resume()
    ##################### CHECKPOINTING ###################
    if ((step % checkpoint_steps == 0 and step > 0) or
        (step == total_steps)) and config.checkpoint:
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))
      with report_progress.timed('checkpoint'):
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        if lead_host:
          # Take the first replica.
          unrep_train_state = jax_utils.unreplicate(train_state)
          metadata = unrep_train_state.metadata
          metadata['chrono'] = chrono.save()
          unrep_train_state.replace(metadata=metadata)  # pytype: disable=attribute-error
          flax.config.update('flax_use_orbax_checkpointing',
                             config.get('flax_use_orbax_checkpointing', False))
          train_utils.save_checkpoint(
              workdir,
              unrep_train_state,
              max_to_keep=config.get('max_checkpoints_to_keep', 3),
          )
          del unrep_train_state
      chrono.resume()  # Un-pause now.
  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary
