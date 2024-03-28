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

"""Boundary Attention Training Script."""

import functools
from typing import Any, Callable, Dict, Optional, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
from flax import jax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import optax
from scenic.dataset_lib import dataset_utils
from scenic.projects.boundary_attention import eval_manager
from scenic.projects.boundary_attention import train_utils
from scenic.projects.boundary_attention.helpers import viz_utils
from scenic.projects.boundary_attention.types import ArrayDict, LossFn, MetricFn  # pylint: disable=g-multiple-import, g-importing-member
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils as scenic_train_utils


# pylint: disable=unused-argument
def train_step(
    train_state: scenic_train_utils.TrainState,
    batch: ArrayDict,
    flax_model: nn.Module,
    grad_weight_schedule_fn: Callable[[int], float],
    loss_fn: LossFn,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    log_grad_info: bool = False
) -> Tuple[scenic_train_utils.TrainState, Dict[str, jnp.ndarray],
           ArrayDict]:
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Args:
    train_state: The current training state.
    batch: The batch of data that will be used for training.
    flax_model: The Flax model used in training.
    grad_weight_schedule_fn: A function that takes the current step and returns
      the current learning rate.
    loss_fn: The loss function used in training.
    metrics_fn: The metrics function used in training.
    config: The experiment config.
    log_grad_info: Whether to log gradient information.

  Returns:
    The new training state, the metrics, and the model outputs.
  """
  rng = train_state.rng

  # Bind the rng to the host/device we are on.
  # dropout_rng, rng = jax.random.split(rng)
  dropout_rng, params_rng, codebook_rng, rng = jax.random.split(key=rng, num=4)
  dropout_rng = scenic_train_utils.bind_rng_to_host_device(
      dropout_rng, axis_name='batch', bind_to='device')
  params_rng = scenic_train_utils.bind_rng_to_host_device(
      params_rng, axis_name='batch', bind_to='device')
  codebook_rng = scenic_train_utils.bind_rng_to_host_device(
      codebook_rng, axis_name='batch', bind_to='device')

  rngs = {'dropout': dropout_rng,
          'params': params_rng,
          'codebook': codebook_rng}

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    model_outputs, new_model_state = flax_model.apply(
        variables,
        batch['image'],
        mutable=['batch_stats'],
        rngs=rngs,
        train=True)
    loss = jnp.mean(loss_fn(model_outputs, batch))
    return loss, (new_model_state, model_outputs)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  (train_cost,
   (new_model_state,
    model_outputs)), grad = compute_gradient_fn(train_state.params)

  del train_cost

  # Clip gradients
  # grad = jax_optimizers.clip_grads(grad, config.max_grad_norm)

  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')

  if train_state.tx is None:
    raise ValueError('train_state.tx is None')

  updates, new_opt_state = train_state.tx.update(grad, train_state.opt_state,
                                                 train_state.params)
  new_params = optax.apply_updates(params=train_state.params, updates=updates)

  # Explicit weight decay, if necessary.
  # if config.get('explicit_weight_decay', None) is not None:
  #   new_optimizer = new_optimizer.replace(
  #       target=optimizers.tree_map_with_names(
  #           functools.partial(
  #               optimizers.decay_weight_fn,
  #               lr=lr,
  #               decay=config.explicit_weight_decay),
  #           new_optimizer.target,
  #           match_name_fn=lambda name: 'kernel' in name))

  metrics = metrics_fn(model_outputs, batch)
  new_rng, _ = jax.random.split(rng)
  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      params=new_params,
      model_state=new_model_state,
      rng=new_rng)

  return new_train_state, metrics, model_outputs


def maybe_restore_model_or_params(model: Any,
                                  train_state: scenic_train_utils.TrainState,
                                  workdir: str,
                                  config: ml_collections.ConfigDict):
  """Restores the model parameters from a checkpoint, if available."""
  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = scenic_train_utils.restore_checkpoint(
        workdir, train_state)

  if (start_step == 0  # Which means "no" checkpoint is restored!
      and len(config.init_from.get('checkpoint_path')) > 0):  # pylint: disable=g-explicit-length-test
    restored_model_cfg = config.init_from.get('model_config', config.model)
    init_checkpoint_path = config.init_from.get('checkpoint_path')
    if config.init_from.get('checkpoint_step', -1) != -1:
      restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
          init_checkpoint_path, train_state, assert_exist=True,
          step=config.init_from.checkpoint_step)
    else:
      restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
          init_checkpoint_path, train_state, assert_exist=True)
    # Load params from the init_model.
    train_state = model.init_from_train_state(  # pytype: disable=attribute-error
        train_state, restored_train_state, restored_model_cfg)
    del restored_train_state

  elif (start_step == 0) and len(config.init_from.get('params_path')) > 0:  # pylint: disable=g-explicit-length-test
    restored_model_cfg = config.init_from.get('model_config', config.model)
    init_checkpoint_path = config.init_from.get('params_path')
    restored_train_state = train_utils.restore_pretrained_params(
        init_checkpoint_path, train_state, assert_exist=True)
    # Load params from the init_model.
    train_state = model.init_from_train_state(  # pytype: disable=attribute-error
        train_state, restored_train_state, restored_model_cfg)

  return train_state, start_step


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Any,
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[scenic_train_utils.TrainState, Optional[Dict[str, Any]],
           Optional[Dict[str, Any]]]:
  """Main training loop lives in this function.

  Given the model class and dataset, it prepares the items needed to run the
  training, including the TrainState.

  Args:
    rng: Jax rng key.
    config: Configurations of the experiment.
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    dataset: A dataset object that contains train_iter, eval_iter, meta_data,
      and optionally, test_iter.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    train_sate that has the state of training (including current global_step,
    model_state, rng, and the optimizer), train_summary and eval_summary which
    are dict of metrics (from the last evaluation and train metric logging
    respectively). These outputs are used for regression testing.
  """
  lead_host = jax.process_index() == 0
  dataset_metadata = dataset.meta_data
  train_task = ''  # pylint: disable=unused-variable

  # Build the loss_and_metrics_fn, metrics, and flax_model.
  model = model_cls(config, dataset_metadata)

  # Initialize model.
  rng, params_rng, dropout_rng = jax.random.split(key=rng, num=3)

  input_specs = []
  for input_shape in dataset_metadata['input_shape']:
    input_spec = (input_shape, dataset_metadata.get('input_dtype',
                                                    jnp.float32))
    input_specs.append(input_spec)

  (params, model_state, num_trainable_params,
   gflops) = scenic_train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=input_specs,
       config=config,
       rngs={'params': params_rng,
             'dropout': dropout_rng})

  # Get learning rate scheduler.
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)

  # Create optimizer.
  optimizer = optimizers.get_optimizer(config.optimizer_configs,
                                       learning_rate_fn=learning_rate_fn,
                                       params=params)
  opt_state = jax.jit(optimizer.init, backend='cpu')(params)

  _, train_rng = jax.random.split(rng)
  # Creat chrono class to track and store training statistics and metadata:
  chrono = scenic_train_utils.Chrono()

  train_state = scenic_train_utils.TrainState(
      global_step=0,
      opt_state=opt_state,
      tx=optimizer,
      params=params,
      model_state=model_state,
      rng=train_rng,
      metadata={'chrono': chrono.save()})

  train_state, start_step = maybe_restore_model_or_params(model, train_state,
                                                          workdir, config)

  chrono.load(train_state.metadata['chrono'])

  # Replicate the optimizer, state, and rng.
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset_metadata)

  grad_weight_schedule_fn = train_utils.get_grad_weight_schedule_fn(config)

  train_step_pmapped = jax.pmap(
      functools.partial(
          train_step,
          flax_model=model.flax_model,
          grad_weight_schedule_fn=grad_weight_schedule_fn,
          loss_fn=model.loss_function,
          metrics_fn=model.get_metrics_fn('train'),
          config=config,
          log_grad_info=config.get('log_grad_info', False)),
      axis_name='batch',
      # We can donate the buffer of train_state. train_batch might be needed for
      # image summaries later.
      donate_argnums=(0,),
  )

  log_eval_steps = config.get('log_eval_steps')
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps

  train_metrics = []
  extra_training_logs = []
  train_summary, eval_summary = None, None

  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)

  logging.info('Starting training loop at step %d.', start_step)

  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer)

  if config.eval_during_train:
    evaler = eval_manager.EvalManager(model, config, rng, report_progress)

  hooks = [report_progress]
  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  if start_step == 0:
    step0_log = {'num_trainable_params': num_trainable_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)

  def write_note(note):
    if lead_host:
      platform.work_unit().set_notes(note)

  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)

      train_state, t_metrics, model_outputs = train_step_pmapped(
          train_state, train_batch)
      # This will accumulate metrics in TPU memory up to the point that we log
      # them. This is no problem for small metrics but may be a problem for
      # large (e.g. segmentation) metrics. An alternative is to set
      # `log_summary_steps` to a small number, or to use
      # `scenic_train_utils.unreplicate_and_get` here instead of right before
      # writing summaries, but that means in each step, we have data transfer
      # between tpu and host, which might slow down the training.
      train_metrics.append(t_metrics)
      # Additional training logs: learning rate:
      extra_training_logs.append({'learning_rate': learning_rate_fn(step)})

      # Quick indication that training is happening.
    logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
    for h in hooks:
      h(step)

    ############### LOG TRAIN SUMMARY ###############
    if (step % log_summary_steps == 1) or (step
                                           == total_steps) or chrono.warmup:
      chrono.pause()
      if lead_host:
        chrono.tick(step, writer, write_note)

      train_summary = {}
      prefix = 'train'
      train_summary.update(
          scenic_train_utils.log_train_summary(
              step=step,
              train_metrics=jax.tree_util.tree_map(
                  scenic_train_utils.unreplicate_and_get, train_metrics
              ),
              extra_training_logs=jax.tree_util.tree_map(
                  jax.device_get, extra_training_logs
              ),
              writer=writer,
              prefix=prefix,
          )
      )

      # ################### VISUALIZATION ###################

      if config.get('visualize', False):
        write_images = viz_utils.get_viz_dict_from_batch(train_batch,
                                                         model_outputs,
                                                         model,
                                                         'train')
        write_images = jax.tree_util.tree_map(
            scenic_train_utils.unreplicate_and_get, write_images
        )
        writer.write_images(step, write_images)

      # #########################################################

      writer.flush()
      # Reset metric accumulation for next evaluation cycle.
      train_metrics = []
      extra_training_logs = []
      chrono.resume()

    print('One step completed.')

    ################### EVALUATION #######################

    if config.eval_during_train:
      chrono.pause(wait_for=(train_state.params))
      evaler.run_one_eval(  # pylint: disable=undefined-variable
          train_state, step, dataset, writer, is_final=(step == total_steps))
      chrono.resume()

    ##################### CHECKPOINTING ############################
    if ((step % checkpoint_steps == 1 or step == total_steps) and
        config.checkpoint):
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))
      with report_progress.timed('checkpoint'):
        # Sync model state across replicas.
        train_state = scenic_train_utils.sync_model_state_across_replicas(
            train_state)
        if lead_host:
          # Take the first replica.
          unrep_train_state = jax_utils.unreplicate(train_state)
          metadata = unrep_train_state.metadata
          metadata['chrono'] = chrono.save()
          unrep_train_state.replace(metadata=metadata)  # pytype: disable=attribute-error
          scenic_train_utils.save_checkpoint(workdir,
                                             unrep_train_state,
                                             max_to_keep=100)
          del unrep_train_state
      chrono.resume()

  # Wait until computations are done before exiting.
  jax.random.normal(jax.random.key(0), ()).block_until_ready()
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary
