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

"""Training loop."""

import functools
import math
import pdb  # pylint: disable=unused-import

from typing import Any, Callable, Dict, Optional, Tuple, Type

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
from flax import jax_utils
import flax.linen as nn
import jax
from jax.example_libraries.optimizers import clip_grads
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
import optax
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import base_model
from scenic.projects.ncr import utils
from scenic.train_lib import train_utils


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
LossFn = Any
PyTree = Any


def train_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    use_ncr: bool,
    *,
    use_bootstrap: bool,
    flax_model: nn.Module,
    loss_fn: LossFn,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False
) -> Tuple[train_utils.TrainState, Dict[str, Tuple[float, int]]]:
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Note that in this code, the buffers of the first (train_state) and second
  (batch) arguments are donated to the computation.

  Args:
    train_state: The state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    use_ncr: Whether the NCR loss is used or not.
    use_bootstrap: Whether the bootstrap loss is used or not.
    flax_model: A Flax model.
    loss_fn: A loss function that given logits, a batch, and parameters of the
      model calculates the loss.
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    config: Configurations of the experiment.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training, computed metrics, and learning rate for logging.
  """
  new_rng, rng = jax.random.split(train_state.rng)

  if config.get('mixup') and config.mixup.alpha:
    rng, mixup_rng = jax.random.split(rng, 2)
    mixup_rng = train_utils.bind_rng_to_host_device(
        mixup_rng,
        axis_name='batch',
        bind_to=config.mixup.get('bind_to', 'device'))
    batch = utils.mixup(
        batch,
        config.mixup.alpha,
        config.mixup.get('image_format', 'NHWC'),
        rng=mixup_rng)

  # Bind the rng to the host/device we are on.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device')

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    model_outputs, new_model_state = flax_model.apply(
        variables,
        batch['inputs'],
        mutable=['batch_stats'],
        train=True,
        rngs={'dropout': dropout_rng},
        debug=debug)

    logits = model_outputs['logits']

    # For NCR models.
    if use_ncr:
      ncr_loss_weight = config.ncr.loss_weight
      features = model_outputs[config.ncr.ncr_feature]
      logits_all, features_all = utils.all_gather((logits, features))
      logging.info('logits.shape: %s. logits_all.shape %s', logits.shape,
                   logits_all.shape)

      loss, loss_metrics = loss_fn(logits, batch, use_ncr, use_bootstrap,
                                   features, logits_all,
                                   features_all, ncr_loss_weight,
                                   variables['params'])
    else:
      # For classification baseline model.
      bootstrap_loss_weight = config.get('bootstrap_weight', 0.0)

      loss, loss_metrics = loss_fn(
          logits=logits, batch=batch, use_ncr=use_ncr,
          use_bootstrap=use_bootstrap,
          loss_weight=bootstrap_loss_weight,
          model_params=variables['params'])

    return loss, (new_model_state, logits, loss_metrics)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  step = train_state.global_step

  (train_cost,
   (new_model_state, logits, loss_metrics)
   ), grad = compute_gradient_fn(train_state.params)

  del train_cost
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')

  if config.get('max_grad_norm', None) is not None:
    grad = clip_grads(grad, config.max_grad_norm)

  updates, new_opt_state = train_state.tx.update(
      grad,
      train_state.opt_state,
      train_state.params)

  new_params = optax.apply_updates(train_state.params, updates)

  metrics = metrics_fn(logits, batch)
  for metric_name, metric_value in loss_metrics.items():
    metrics[metric_name] = (metric_value, 1)  # Normalizer here is 1.

  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=step + 1,
      opt_state=new_opt_state,
      model_state=new_model_state,
      params=new_params,
      rng=new_rng)
  return new_train_state, metrics


def eval_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: MetricFn,
    debug: Optional[bool] = False
) -> Tuple[Dict[str, Tuple[float, int]], jnp.ndarray]:
  """Runs a single step of training.

  Note that in this code, the buffer of the second argument (batch) is donated
  to the computation.

  Assumed API of metrics_fn is:
  ```metrics = metrics_fn(logits, batch)
  where batch is yielded by the batch iterator, and metrics is a dictionary
  mapping metric name to a vector of per example measurements. eval_step will
  aggregate (by summing) all per example measurements and divide by the
  aggregated normalizers. For each given metric we compute:
  1/N sum_{b in batch_iter} metric(b), where  N is the sum of normalizer
  over all batches.

  Args:
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data. a metrics function, that given logits and
      batch of data, calculates the metrics as well as the loss.
    flax_model: A Flax model.
    metrics_fn: A metrics function, that given logits and batch of data,
      calculates the metrics as well as the loss.
    debug: Whether the debug mode is enabled during evaluation.
      `debug=True` enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Calculated metrics and Logits.
  """
  variables = {
      'params': train_state.params,
      **train_state.model_state
  }
  model_outputs = flax_model.apply(
      variables, batch['inputs'], train=False, mutable=False, debug=debug)
  if isinstance(model_outputs, dict):
    # For NCR models.
    logits = model_outputs['logits']
  else:
    # For baseline models.
    logits = model_outputs
  metrics = metrics_fn(logits, batch)
  return metrics, logits


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Type[base_model.BaseModel],
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
  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(dataset.meta_data['input_shape'],
                    dataset.meta_data.get('input_dtype', jnp.float32))],
       config=config,
       rngs=init_rng)

  # Create optimizer.
  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].

  lr_fn = optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=config.lr_configs.base_learning_rate,
      warmup_steps=config.lr_configs.warmup_steps,
      decay_steps=config.lr_configs.steps_per_cycle,
      end_value=0.0)

  tx = optax.chain(
      optax.add_decayed_weights(
          weight_decay=config.optimizer_configs.weight_decay),
      optax.sgd(lr_fn, momentum=config.optimizer_configs.momentum)
      )

  opt_state = jax.jit(tx.init, backend='cpu')(params)

  _, train_rng = jax.random.split(rng)

  chrono = train_utils.Chrono()

  train_state = train_utils.TrainState(
      global_step=0,
      opt_state=opt_state,
      tx=tx,
      model_state=model_state,
      params=params,
      rng=train_rng,
      metadata={'chrono': chrono.save()})
  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = utils.restore_checkpoint(workdir, train_state)
  # Replicate the optimzier, state, and rng.

  chrono.load(train_state.metadata['chrono'])
  train_state = train_state.replace(metadata={})

  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data)

  # Initial value for use_ncr
  use_bootstrap = False
  if config.loss_type == 'cross_entropy':
    use_ncr = False
    if config.get('use_bootstrap'):
      use_bootstrap = True
  elif config.loss_type == 'ncr' and config.ncr.starting_epoch > 0:
    use_ncr = False
  elif config.loss_type == 'ncr':
    use_ncr = True
  else:
    raise ValueError(f'Unknown loss type {config.loss_type}.')

  train_step_pmapped = jax.pmap(
      functools.partial(
          train_step,
          use_bootstrap=use_bootstrap,
          flax_model=model.flax_model,
          loss_fn=model.loss_function,
          metrics_fn=model.get_metrics_fn('train'),
          config=config,
          debug=config.debug_train),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0, 1),
      static_broadcasted_argnums=(2),
  )
  eval_step_pmapped = jax.pmap(
      functools.partial(
          eval_step,
          flax_model=model.flax_model,
          metrics_fn=model.get_metrics_fn('validation'),
          debug=config.debug_eval),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1,),
  )
  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps
  max_checkpoint_keep = config.get('max_checkpoint_keep', 3)

  # Ceil rounding such that we include the last incomplete batch.
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / config.batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None

  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)
  logging.info('Starting training loop at step %d.', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps,
      writer=writer,
      every_secs=None,
      every_steps=config.get('report_progress_step', log_summary_steps),
  )

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

  write_note(f'First step compilations...\n{chrono.note}')

  for step in range(start_step + 1, total_steps + 1):

    if (not use_ncr) and config.loss_type == 'ncr' and (
        math.floor(step/steps_per_epoch) >= config.ncr.starting_epoch):
      use_ncr = True

    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      train_state, t_metrics = train_step_pmapped(
          train_state, train_batch, use_ncr)
      lr = lr_fn(step)
      # This will accumulate metrics in TPU memory up to the point that we log
      # them. This is no problem for small metrics but may be a problem for
      # large (e.g. segmentation) metrics. An alternative is to set
      # `log_summary_steps` to a small number, or to use
      # `train_utils.unreplicate_and_get` here instead of right before writing
      # summaries, but that means in each step, we have data transfer between
      # tpu and host, which might slow down the training.
      train_metrics.append(t_metrics)
      # Additional training logs: learning rate:
      extra_training_logs.append({'learning_rate': lr})
    for h in hooks:
      h(step)

    ###################### LOG TRAIN SUMMARY ########################
    if (step % log_summary_steps == 1) or (step == total_steps):
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
          key_separator='/',
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
        train_state = utils.sync_model_state_across_replicas(train_state)
        for _ in range(steps_per_eval):
          eval_batch = next(dataset.valid_iter)
          e_metrics, _ = eval_step_pmapped(train_state, eval_batch)
          eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))
        eval_summary = train_utils.log_eval_summary(
            step=step,
            eval_metrics=eval_metrics,
            key_separator='/',
            writer=writer)
      writer.flush()
      chrono.resume()
      del eval_metrics
    ##################### CHECKPOINTING ###################
    if ((step % checkpoint_steps == 0 and step > 0) or
        (step == total_steps)) and config.checkpoint:
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))
      with report_progress.timed('checkpoint'):
        train_utils.handle_checkpointing(
            train_state, chrono, workdir, max_checkpoint_keep)
      chrono.resume()
  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train and eval summary after last step for regresesion testing.
  return train_state, train_summary, eval_summary
