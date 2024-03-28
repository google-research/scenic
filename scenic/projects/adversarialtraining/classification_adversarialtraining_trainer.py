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
from scenic.projects.adversarialtraining import train_utils as adv_train_utils
from scenic.projects.adversarialtraining.attacks import attack_compute
from scenic.projects.adversarialtraining.attacks import attack_metrics
from scenic.projects.adversarialtraining.models import models
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]
LrFn = Callable[[jnp.ndarray], jnp.ndarray]


def unnormalize_imgnet(input_tensors):
  return (input_tensors + 1.) / 2.


def train_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    loss_fn: LossFn,
    lr_fn: LrFn,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False,
) -> Tuple[train_utils.TrainState, Dict[str, Tuple[float, int]], Dict[
    str, jnp.ndarray], Dict[str, jnp.ndarray]]:
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
    flax_model: A Flax model.
    loss_fn: A loss function that given logits, a batch, and parameters of the
      model calculates the loss.
    lr_fn: The learning rate fn used for the logging the learning rate.
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

  images_to_log = {}
  images_to_log['input_image_before_mixup'] = unnormalize_imgnet(
      batch['inputs'])

  if config.get('mixup') and config.mixup.alpha:
    mixup_rng, rng = jax.random.split(rng, 2)
    mixup_rng = train_utils.bind_rng_to_host_device(
        mixup_rng,
        axis_name='batch',
        bind_to=config.mixup.get('bind_to', 'device'))
    batch = dataset_utils.mixup(
        batch,
        config.mixup.alpha,
        config.mixup.get('image_format', 'NHWC'),
        rng=mixup_rng)

  images_to_log['input_image_after_mixup'] = unnormalize_imgnet(batch['inputs'])

  # Bind the rng to the host/device we are on.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device')

  def training_loss_fn_single(params,
                              model_state,
                              batch,
                              use_aux_batchnorm,
                              use_aux_dropout,
                              train_var=True):
    variables = {'params': params, **model_state}
    apply_kwargs = models.get_kwargs(config, use_aux_batchnorm, use_aux_dropout)
    logits, new_model_state = flax_model.apply(
        variables,
        batch['inputs'],
        mutable=['batch_stats'],
        train=train_var,
        rngs={'dropout': dropout_rng},
        debug=debug,
        **apply_kwargs)
    loss = loss_fn(logits, batch, variables['params'])
    return loss, (new_model_state, logits)

  if not config.adversarial_augmentation_mode:

    def training_loss_fn(params, model_state):
      total_loss, (new_model_state, logits) = training_loss_fn_single(
          params,
          model_state,
          batch,
          use_aux_batchnorm=False,
          use_aux_dropout=False)
      misc_artifacts = {}
      return total_loss, (new_model_state, logits, misc_artifacts)
  elif config.adversarial_augmentation_mode == 'advprop_pyramid':

    def training_loss_fn(params, model_state):
      clean_inputs = batch['inputs']
      clean_loss, (clean_model_state, clean_logits) = training_loss_fn_single(
          params,
          model_state,
          batch,
          use_aux_batchnorm=False,
          use_aux_dropout=False)

      adv_image, adv_perturbation, misc_artifacts = attack_compute.get_adversarial_image_and_perturbation(
          batch=batch,
          clean_logits=clean_logits,
          config=config,
          training_loss_fn_single=training_loss_fn_single,
          train_state=train_state,
          dropout_rng=dropout_rng,
          lr_rng=rng)

      adv_batch = batch
      adv_batch['inputs'] = adv_image
      adv_loss, (adv_model_state, adv_logits) = training_loss_fn_single(
          params,
          clean_model_state,
          adv_batch,
          use_aux_batchnorm=True,
          use_aux_dropout=True)

      adv_loss_weight = config.advprop.adv_loss_weight
      total_loss = clean_loss + adv_loss_weight * adv_loss

      # set up misc_artifacts
      misc_artifacts['logits'] = clean_logits
      misc_artifacts['adv_logits'] = adv_logits
      misc_artifacts['image_diffs'] = (adv_image -
                                       clean_inputs).sum(axis=(1, 2, 3))
      misc_artifacts['adv_loss_weight'] = adv_loss_weight
      misc_artifacts['adv_logits'] = adv_logits
      misc_artifacts['loss'] = clean_loss
      misc_artifacts['adv_loss'] = adv_loss
      misc_artifacts['adv_image'] = adv_image
      misc_artifacts['adv_perturbation'] = adv_perturbation

      return total_loss, (adv_model_state, clean_logits, misc_artifacts)
  else:
    raise NotImplementedError('Unrecognized adversarial augmentation mode.')

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)

  computed_gradient = compute_gradient_fn(train_state.params,
                                          train_state.model_state)

  (train_cost, (new_model_state, logits,
                misc_artifacts)), grad = computed_gradient

  metrics = metrics_fn(logits, batch)
  if not config.adversarial_augmentation_mode:
    metrics, images_to_log = attack_metrics.get_metrics(misc_artifacts, batch,
                                                        metrics, images_to_log,
                                                        metrics_fn)

  del train_cost

  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')

  if config.get('max_grad_norm', None) is not None:
    grad = clip_grads(grad, config.max_grad_norm)

  updates, new_opt_state = train_state.tx.update(grad, train_state.opt_state,
                                                 train_state.params)
  new_params = optax.apply_updates(params=train_state.params, updates=updates)

  training_logs = {}
  training_logs['learning_rate'] = lr_fn(train_state.global_step)

  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      params=new_params,
      model_state=new_model_state,
      rng=new_rng)
  return new_train_state, metrics, training_logs, images_to_log


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
    Calculated metrics and logits.
  """
  variables = {
      'params': train_state.params,
      **train_state.model_state
  }
  logits = flax_model.apply(
      variables, batch['inputs'], train=False, mutable=False, debug=debug)
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

  # Get learning rate scheduler.
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)

  # Create optimizer.
  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  optimizer = optimizers.get_optimizer(
      config.optimizer_configs,
      learning_rate_fn=learning_rate_fn,
      params=params)
  opt_state = jax.jit(optimizer.init, backend='cpu')(params)

  # Create chrono class to track and store training statistics and metadata:
  chrono = train_utils.Chrono()

  rng, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
      global_step=0,
      opt_state=opt_state,
      tx=optimizer,
      params=params,
      model_state=model_state,
      rng=train_rng,
      metadata={'chrono': chrono.save()})
  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state)
  # Replicate the optimzier, state, and rng.
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data)

  train_step_pmapped = jax.pmap(
      functools.partial(
          train_step,
          flax_model=model.flax_model,
          loss_fn=model.loss_function,
          lr_fn=learning_rate_fn,
          metrics_fn=model.get_metrics_fn('train'),
          config=config,
          debug=config.debug_train),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0, 1),
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

  # Ceil rounding such that we include the last incomplete batch.
  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  train_metrics, train_images, extra_training_logs = [], [], []
  train_summary, eval_summary = None, None

  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)

  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer)
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

  def write_note(note):
    if lead_host:
      platform.work_unit().set_notes(note)

  write_note(f'First step compilations...\n{chrono.note}')
  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      train_state, t_metrics, t_logs, t_images = train_step_pmapped(
          train_state, train_batch)
      # This will accumulate metrics in TPU memory up to the point that we log
      # them. This is no problem for small metrics but may be a problem for
      # large (e.g. segmentation) metrics. An alternative is to set
      # `log_summary_steps` to a small number, or to use
      # `train_utils.unreplicate_and_get` here instead of right before writing
      # summaries, but that means in each step, we have data transfer between
      # tpu and host, which might slow down the training.
      train_metrics.append(t_metrics)
      t_logs = jax.tree_util.tree_map(jax_utils.unreplicate, t_logs)
      # Additional training logs: learning rate:
      extra_training_logs.append(t_logs['learning_rate'])
    for h in hooks:
      h(step)
    chrono.pause()  # Below are once-in-a-while ops -> pause.
    ###################### LOG TRAIN SUMMARY ########################
    if (step % log_summary_steps == 1) or (step == total_steps):
      if lead_host:
        chrono.tick(step, writer=writer, write_note=write_note)
      train_images.append(t_images)
      # train_metrics is list of a dictionaries of metrics, where the shape of
      # the metrics[key] is [n_local_devices]. However, because metric functions
      # have a psum, we have already summed across the whole sharded batch, and
      # what's returned is n_local_devices copies of the same summed metric.
      # So we do unreplicate and fetch them to host using `unreplicate_and_get`.
      train_summary = adv_train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(jax.device_get, train_metrics),
          train_images=jax.tree_util.tree_map(jax.device_get, train_images),
          extra_training_logs=jax.tree_util.tree_map(jax.device_get,
                                                     extra_training_logs),
          writer=writer)
      # Reset metric accumulation for next evaluation cycle.
      train_metrics, train_images, extra_training_logs = [], [], []
    ################### EVALUATION #######################
    if (step % log_eval_steps == 1) or (step == total_steps):
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
    ##################### CHECKPOINTING ###################
    if ((step % checkpoint_steps == 0 and step > 0) or
        (step == total_steps)) and config.checkpoint:
      with report_progress.timed('checkpoint'):
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        if lead_host:
          # Take the first replica.
          unrep_train_state = jax_utils.unreplicate(train_state)
          metadata = unrep_train_state.metadata
          metadata['chrono'] = chrono.save()
          unrep_train_state.replace(metadata=metadata)  # pytype: disable=attribute-error
          train_utils.save_checkpoint(workdir, unrep_train_state)
          del unrep_train_state
    chrono.resume()  # un-pause now
  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train and eval summary after last step for regresesion testing.
  return train_state, train_summary, eval_summary
