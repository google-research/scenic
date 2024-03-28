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

"""Training Script for knowledge-based models."""

import functools
from typing import Any, Dict, Optional, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
from flax import jax_utils
import flax.linen as nn
import jax
import jax.example_libraries.optimizers as jax_optimizers
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
import optax
from scenic.dataset_lib import dataset_utils
from scenic.xm import xm_utils
from scenic.model_lib.base_models import base_model
from scenic.projects.knowledge_visual_language import trainer_utils
from scenic.projects.knowledge_visual_language.models import constants
from scenic.projects.knowledge_visual_language.models import losses
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils


def init_state(
    model: base_model.BaseModel,
    dataset: dataset_utils.Dataset,
    config: ml_collections.ConfigDict,
    workdir: str,
    rng: jnp.ndarray,
    writer: metric_writers.MetricWriter,
):
  """Initialize the train state."""

  input_spec = {
      key[:-5]: dataset.meta_data[key]
      for key in dataset.meta_data
      if key[-5:] == '_spec'
  }
  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  params, model_state, num_params, gflops = (
      train_utils.initialize_model_with_pytree(
          model_def=model.flax_model,
          input_spec=input_spec,
          config=config,
          rngs=init_rng,
      )
  )
  logging.info('The model has %d params', num_params)

  if gflops:
    logging.info('uses %d gflops', gflops or -1)
  lr_fn = lr_schedules.get_learning_rate_fn(config)
  # Create the optimizer.
  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  tx = optimizers.get_optimizer(
      optimizer_config=optimizers.get_optax_optimizer_config(config),
      learning_rate_fn=lr_fn,
  )
  tx = trainer_utils.froze_param_optax(
      params=params,
      tx=tx,
      frozen_patterns=config.get('frozen_patterns', None),
      not_frozen_patterns=config.get('not_frozen_patterns', None),
  )
  opt_state = jax.jit(tx.init, backend='cpu')(params)
  # del params  # Do not keep a copy of the initial params.

  _, train_rng = jax.random.split(rng)
  chrono = train_utils.Chrono()
  train_state = train_utils.TrainState(
      global_step=0,
      opt_state=opt_state,
      tx=tx,
      params=params,
      model_state=model_state,
      rng=train_rng,
      metadata={'chrono': chrono.save()},
  )
  if config.checkpoint:
    logging.info('Continuing training from the checkpoint')
    logging.info('workdir= %s', workdir)
    train_state, params_axes = trainer_utils.pop_axes_names(
        train_state, axes_name='params_axes'
    )
    train_state, _ = train_utils.restore_checkpoint(workdir, train_state)
    train_state = trainer_utils.re_add_axis_names(
        train_state, params_axes, axes_name='params_axes'
    )

  start_step = int(train_state.global_step)
  chrono.load(train_state.metadata['chrono'])

  if start_step == 0:
    if config.get('init_from', False):
      if config.init_from.get('resume', False):
        workdir = config.init_from.get('resume')
        logging.info('Resuming training from the checkpoint')
        logging.info('workdir= %s', workdir)
        train_state, params_axes = trainer_utils.pop_axes_names(
            train_state, axes_name='params_axes'
        )
        train_state, _ = train_utils.restore_checkpoint(workdir, train_state)
        train_state = trainer_utils.re_add_axis_names(
            train_state, params_axes, axes_name='params_axes'
        )
        start_step = int(train_state.global_step)
        chrono.load(train_state.metadata['chrono'])

      elif config.init_from.get('xm', False):
        if config.init_from.load_key_encoder:
          params = trainer_utils.load_key_params(params, config)
          train_state = train_state.replace(  # pytype: disable=attribute-error
              params=params
          )
      else:
        logging.info('Loading T5 & ViT Parameter from Pre-Trained Model')
        params = trainer_utils.load_visual_params(params, config)
        params = trainer_utils.load_text_params(params, config)
        train_state = train_state.replace(  # pytype: disable=attribute-error
            params=params
        )
    step0_log = {'num_trainable_params': num_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)

  return train_state, start_step, chrono, lr_fn


def train_step(
    train_state: train_utils.TrainState,
    batch: constants.Batch,
    *,
    flax_model: nn.Module,
    loss_fn: constants.LossFn,
    metrics_fn: constants.MetricFn,
    model_config: ml_collections.ConfigDict,
    debug: Optional[bool] = False,
) -> Tuple[
    train_utils.TrainState, Dict[str, Tuple[float, int]], Dict[str, float]
]:
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Note that in this code, the buffers of the first (train_state) and second
  (batch) arguments are donated to the computation.

  Args:
    train_state: The state of training including the current global_step,
      model_state, rng, and optimizer. The buffer of this argument can be
      donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    flax_model: A Flax model.
    loss_fn: A loss function that given logits and batch of data, calculates the
      training loss.
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    model_config: Config for model architecture.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training, loss, computed metrics, and learning rate for
    logging.
  """
  new_rng, rng = jax.random.split(train_state.rng)

  # Bind the rng to the host/device we are on for dropout.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device'
  )

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    output_dict, new_model_state = flax_model.apply(
        variables,
        **batch,
        mutable=['batch_stats'],
        train=True,
        fuse_retrieval=model_config.fuse_retrieval,
        in_batch_neg=model_config.in_batch_neg,
        rngs={'dropout': dropout_rng},
        debug=debug,
    )
    if debug:
      logging.info(
          'Shape of token_logits in train step is: %s',
          output_dict['predicted_logits'].shape,
      )
    r = model_config.retrieval_ratio
    output_dict['supervised_retrieval'] = model_config.supervised_retrieval
    loss_dict = loss_fn(output_dict, batch)
    train_loss = loss_dict['gen_loss'] * (1 - r) + loss_dict['retr_loss'] * r
    return train_loss, (new_model_state, output_dict, loss_dict)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)

  (train_loss, (new_model_state, output_dict, loss_dict)), grad = (
      compute_gradient_fn(train_state.params)
  )
  grad = jax.lax.pmean(grad, axis_name='batch')
  grad_norm = jax_optimizers.l2_norm(grad).astype(float)
  updates, new_opt_state = train_state.tx.update(
      grad, train_state.opt_state, train_state.params
  )
  update_norm = jax_optimizers.l2_norm(updates).astype(float)
  new_params = optax.apply_updates(train_state.params, updates)
  param_norm = jax_optimizers.l2_norm(train_state.params).astype(float)

  metrics = metrics_fn(output_dict['predicted_logits'], batch)

  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      params=new_params,
      model_state=new_model_state,
      rng=new_rng,
  )

  logs = {
      'train/train_loss': train_loss,
      'train/gen_loss': loss_dict['gen_loss'],
      'train/retr_loss': loss_dict['retr_loss'],
      'train/retr_acc': loss_dict['retr_acc'],
      'train/s0': loss_dict['s0'],
      'train/s1': loss_dict['s1'],
      'grad_norm': grad_norm,
      'update_norm': update_norm,
      'param_norm': param_norm,
      'bias': train_state.params['att_transform']['bias'][0],
      'scale': train_state.params['att_transform']['scale'][0],
  }

  if 'retr_scores' in output_dict:
    logs['train/a0'] = output_dict['retr_scores'][0][0]
    logs['train/a1'] = output_dict['retr_scores'][0][1]

  return new_train_state, metrics, logs  # pytype: disable=bad-return-type  # jax-types


def eval_step(
    train_state: train_utils.TrainState,
    batch: constants.Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: constants.MetricFn,
    model_config: ml_collections.ConfigDict,
    debug: Optional[bool] = False,
) -> Tuple[Dict[str, Tuple[float, int]], Dict[str, float]]:
  """Runs a single step of evaluation, TODO(ziniu): Add beam search decoding.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Note that in this code, the buffers of the first (train_state) and second
  (batch) arguments are donated to the computation.

  Args:
    train_state: The state of training including the current global_step,
      model_state, rng, and optimizer. The buffer of this argument can be
      donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    flax_model: A Flax model.
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    model_config: Config for model architecture.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training, loss, computed metrics, and learning rate for
    logging.
  """
  variables = {'params': train_state.params, **train_state.model_state}
  output_dict = flax_model.apply(
      variables,
      **batch,
      train=False,
      fuse_retrieval=model_config.fuse_retrieval,
      in_batch_neg=model_config.in_batch_neg,
      debug=debug,
  )
  metrics = metrics_fn(output_dict['predicted_logits'], batch)

  retr_loss, (retr_acc, _, _) = losses.contrastive_loss(
      query_emb=output_dict['base_query'],
      key_emb=output_dict['retr_keys'],
      temperature=model_config.get('temperature'),
  )
  logs = {'eval/retr_loss': retr_loss, 'eval/retr_acc': retr_acc}

  return metrics, logs  # pytype: disable=bad-return-type  # jnp-type


def train_and_eval(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Any,
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[train_utils.TrainState, Dict[str, Any], Dict[str, Any]]:
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
    A tuple with:
      * the state that has the state of training (including current
        global_step, model_state, rng, and the optimizer)
      * a dictionary with the train_summary
      * a dictionary with the evaluation summary
  """
  host_id = jax.process_index()
  lead_host = host_id == 0
  # Build the loss_fn, metrics, and flax_model.
  model = model_cls(config, dataset.meta_data)
  train_state, start_step, chrono, lr_fn = init_state(
      model, dataset, config, workdir, rng, writer
  )

  logging.info('Complete initialization. Start Training.')
  train_state = jax_utils.replicate(train_state)
  logging.info('Number of processes is %s', jax.process_count())

  # Get the pmapped train and eval steps.
  train_step_pmapped = jax.pmap(
      functools.partial(
          train_step,
          flax_model=model.flax_model,
          loss_fn=model.loss_function,
          metrics_fn=model.get_metrics_fn('train'),
          debug=config.debug_train,
          model_config=config.model,
      ),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0, 1),
  )
  eval_step_pmapped = jax.pmap(
      functools.partial(
          eval_step,
          flax_model=model.flax_model,
          metrics_fn=model.get_metrics_fn('validation'),
          debug=config.debug_eval,
          model_config=config.model,
      ),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1,),
  )
  train_summary, eval_summary, e_metrics = {}, {}, {}
  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data
  )
  log_eval_steps = config.get('log_eval_steps', steps_per_epoch)
  checkpoint_steps = config.get('checkpoint_steps', steps_per_epoch)
  log_summary_steps = config.get('log_summary_steps', log_eval_steps)

  logging.info('Start training from step %d', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer
  )
  hooks = []
  if lead_host:
    hooks.append(report_progress)

  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  chrono.inform(
      int(start_step),
      int(total_steps),
      int(config.batch_size),
      int(steps_per_epoch),
  )

  summary_builder = trainer_utils.SummaryBuilder([], [])

  def write_note(note):
    if lead_host:
      platform.work_unit().set_notes(note)

  for step in range(start_step + 1, total_steps + 1):
    if lead_host:
      logging.info('training for step %d', step)
    ###################### Training ########################
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      train_state, t_metrics, logs = train_step_pmapped(
          train_state=train_state, batch=train_batch
      )
    for hook in hooks:
      hook(step)
    logs['learning_rate'] = lr_fn(step).reshape([-1, 1])
    summary_builder.update(metrics_update=t_metrics, extra_logs_update=logs)
    if lead_host:
      logging.info('finish training for step %d', step)
    ###################### LOG TRAIN SUMMARY ########################
    if (step % log_summary_steps == 10) or (step == total_steps):
      chrono.pause()
      if lead_host:
        logging.info('log training summary')
        chrono.tick(step, writer, write_note)
      train_summary = summary_builder.write(writer, step)
      chrono.resume()
    ################### EVALUATION ################################
    should_eval = (step % log_eval_steps == 10) or (step == total_steps)
    if should_eval:
      logging.info('Start validation!')
      chrono.pause(wait_for=(train_state.params))
      # Sync model state across replicas.
      train_state = train_utils.sync_model_state_across_replicas(train_state)
      for ds_name in dataset.valid_iter:
        logging.info('Validate on %s', ds_name)
        # Compute the number of evaluation steps per dataset.
        num_eval_examples = dataset.meta_data['num_eval_examples'][ds_name]
        total_eval_steps = int(
            np.ceil(num_eval_examples / (config.get('eval_batch_size')))
        )
        steps_per_eval = config.get('steps_per_eval', total_eval_steps)
        eval_metrics_all = []
        for _ in range(steps_per_eval):
          eval_batch = next(dataset.valid_iter[ds_name])
          e_metrics, logs = eval_step_pmapped(
              train_state=train_state, batch=eval_batch
          )
          eval_metrics_all.append(train_utils.unreplicate_and_get(e_metrics))
        logging.info(e_metrics)
        logging.info(logs)
        eval_summary = train_utils.log_eval_summary(
            step=step,
            eval_metrics=eval_metrics_all,
            writer=writer,
            prefix=ds_name,
            extra_eval_summary=jax.tree_util.tree_map(
                train_utils.unreplicate_and_get, logs
            ),
        )
      chrono.resume()

    ##################### CHECKPOINTING ###########################
    if not config.checkpoint:
      continue
    elif step % checkpoint_steps == 0 and step > 0 or (step == total_steps):
      logging.info('Save checkpoint!')
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))
      with report_progress.timed('checkpoint'):
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        if lead_host:
          logging.info('checkpointing (training step: %d)', step)
          # Take the first replica.
          unrep_train_state = jax_utils.unreplicate(train_state)
          metadata = unrep_train_state.metadata
          metadata['chrono'] = chrono.save()
          unrep_train_state.replace(metadata=metadata)  # pytype: disable=attribute-error
          train_utils.save_checkpoint(workdir, unrep_train_state)
          del unrep_train_state
      chrono.resume()
      logging.info('Checkpoint saved!')

  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train and eval summary after last step.
  return train_state, train_summary, eval_summary
