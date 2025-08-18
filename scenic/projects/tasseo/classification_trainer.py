# Copyright 2025 The Scenic Authors.
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

"""Tasseo training Script."""

import functools
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
from PIL import Image
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import base_model
from scenic.projects.tasseo import train_utils as tasseo_train_utils
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]
LrFn = Callable[[jnp.ndarray], jnp.ndarray]


def rotate_image(angle, image):
  """Returns a rotated image."""
  image_array = -np.ones((300, 300))
  image_array[50:249, 100:199] = -np.asarray(image)
  image = Image.fromarray(((image_array + 1) * 127.5).astype(np.uint8))
  rotated_image = image.rotate(angle, expand=True)
  rotated_image = -(np.asarray(rotated_image) / 127.5) + 1
  shape = np.shape(rotated_image)
  rotated_image = rotated_image[
      int(shape[0] / 2) - 100 : int(shape[0] / 2) + 99,
      int(shape[1] / 2) - 50 : int(shape[1] / 2) + 49,
  ]
  return rotated_image


def rotate_batch(train_batch_inputs, max_angle):
  """Rotate batch inputs."""
  rotated_batch = np.zeros_like(train_batch_inputs)
  shape = jnp.shape(train_batch_inputs)
  for i in range(shape[0]):
    for j in range(shape[1]):
      rotated_batch[i, j, :, :, 0] = rotate_image(
          np.random.random() * max_angle, train_batch_inputs[i, j, :, :, 0]
      )
  return jnp.asarray(rotated_batch)


def rescale_image(scale_factor, image):
  """Returns a rescaled image."""
  image = np.asarray(image)
  image = Image.fromarray(((image + 1) * 127.5).astype(np.uint8))
  rescaled_image = image.resize(
      (int(99 * scale_factor), int(199 * scale_factor))
  )
  rescaled_image = (np.asarray(rescaled_image) / 127.5) - 1
  shape = np.shape(rescaled_image)
  if scale_factor == 1 or shape == (199, 99):
    output_array = rescaled_image
  elif scale_factor < 1:
    output_array = np.ones((199, 99))
    output_array[
        100 - int(shape[0] / 2) : 100 - int(shape[0] / 2) + shape[0],
        50 - int(shape[1] / 2) : 50 - int(shape[1] / 2) + shape[1],
    ] = rescaled_image
  else:
    output_array = rescaled_image[
        int(shape[0] / 2) - 100 : int(shape[0] / 2) + 99,
        int(shape[1] / 2) - 50 : int(shape[1] / 2) + 49,
    ]
  return output_array


def rescale_batch(train_batch_inputs, min_scale_factor, max_scale_factor):
  """Rescale batch inputs."""
  rescaled_batch = np.zeros_like(train_batch_inputs)
  shape = jnp.shape(train_batch_inputs)
  for i in range(shape[0]):
    for j in range(shape[1]):
      scale_factor = min_scale_factor + np.random.random() * (
          max_scale_factor - min_scale_factor
      )
      rescaled_image = rescale_image(
          scale_factor, train_batch_inputs[i, j, :, :, 0]
      )
      while np.shape(rescaled_image) != (199, 99):  # catching weird edge cases
        scale_factor = min_scale_factor + np.random.random() * (
            max_scale_factor - min_scale_factor
        )
        rescaled_image = rescale_image(
            scale_factor, train_batch_inputs[i, j, :, :, 0]
        )
      rescaled_batch[i, j, :, :, 0] = rescaled_image
  return jnp.asarray(rescaled_batch)


def train_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    loss_fn: LossFn,
    lr_fn: LrFn,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False
) -> Tuple[train_utils.TrainState, Dict[str, Tuple[float, int]], Dict[str,
                                                                      Any]]:
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Note that in this code, the buffers of the first (train_state) and second
  (batch) arguments are donated to the computation.

  Args:
    train_state: The state of training including the current global_step,
      model_state, rng, params, and optimizer. The buffer of this argument can
      be donated to the computation.
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
    Updated state of training and computed metrics and some training logs.
  """
  training_logs = {}
  new_rng, rng = jax.random.split(train_state.rng)

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

  # Bind the rng to the host/device we are on.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device')

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    logits, new_model_state = flax_model.apply(
        variables,
        batch['inputs'],
        mutable=['batch_stats'],
        train=True,
        rngs={'dropout': dropout_rng},
        debug=debug)
    loss = loss_fn(logits, batch, variables['params'])
    return loss, (new_model_state, logits)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  (train_cost, (new_model_state,
                logits)), grad = compute_gradient_fn(train_state.params)

  del train_cost
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')

  if config.get('max_grad_norm') is not None:
    grad = clip_grads(grad, config.max_grad_norm)

  updates, new_opt_state = train_state.tx.update(grad, train_state.opt_state,
                                                 train_state.params)
  new_params = optax.apply_updates(train_state.params, updates)

  training_logs['l2_grads'] = jnp.sqrt(
      sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grad)]))
  ps = jax.tree_util.tree_leaves(new_params)
  training_logs['l2_params'] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
  us = jax.tree_util.tree_leaves(updates)
  training_logs['l2_updates'] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))
  # TODO(dehghani): Can we get this from the optimizer instead?
  training_logs['learning_rate'] = lr_fn(train_state.global_step)

  metrics = metrics_fn(logits, batch)
  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      params=new_params,
      model_state=new_model_state,
      rng=new_rng)

  return new_train_state, metrics, training_logs


def eval_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: MetricFn,
    all_gather: bool = False,
    debug: Optional[bool] = False
) -> Tuple[Dict[str, Tuple[float, int]], Optional[jnp.ndarray],
           Optional[jnp.ndarray]]:
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
    all_gather: If True, the function gather batch and output of model in from
      all hosts, using `jax.lax.all_gather` and return it, e.g., for computing
      global metrics on CPU.
    debug: Whether the debug mode is enabled during evaluation.
      `debug=True` enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Calculated metrics and optionally output, and batch after all_gather.
  """
  variables = {'params': train_state.params, **train_state.model_state}
  logits = flax_model.apply(
      variables,
      batch['inputs'],
      train=False,
      mutable=False,
      debug=debug,
  )
  metrics = metrics_fn(logits, batch)
  if all_gather:
    targets = {'label': batch['label'], 'batch_mask': batch['batch_mask']}
    logits = jax.lax.all_gather(logits, 'batch')
    targets = jax.lax.all_gather(targets, 'batch')
    return metrics, logits, targets
  else:
    return metrics, None, None


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
  lr_fn = lr_schedules.get_learning_rate_fn(config)
  optimizer_config = optimizers.get_optax_optimizer_config(config)
  # If the config is already an optax-compatible config, better call directly:
  #   optimizers.get_optimizer(config.optimizer_configs, lr_fn)
  tx = optimizers.get_optimizer(optimizer_config, lr_fn, params=params)
  # We jit this, such that the arrays that are created on the same device as the
  # input is, in this case the CPU. Else they'd be on device[0].
  opt_state = jax.jit(tx.init, backend='cpu')(params)

  rng, train_rng = jax.random.split(rng)

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
  train_state = train_state.replace(metadata={})
  # Replicate the optimizer, state, and rng.
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
          lr_fn=lr_fn,
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
          all_gather=config.get('global_metrics', False),
          debug=config.debug_eval),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1,),
  )
  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps
  max_checkpoint_keep = config.get('max_checkpoint_keep', 3)
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps

  # Ceil rounding such that we include the last incomplete batch.
  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  # If `global_metrics` are set in the config and we are the lead host
  compute_global_metrics = False
  if config.get('global_metrics', False) and lead_host:
    compute_global_metrics = True
  if compute_global_metrics:
    global_metrics_evaluator = tasseo_train_utils.TasseoGlobalEvaluator(
        config.global_metrics)

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
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      train_batch['label'] = jnp.asarray(
          train_batch['label'], dtype=jnp.float32
      )  # added because of dtype mismatch
      # train_batch['inputs'] = rotate_batch(train_batch['inputs'], 360)
      #  randomly rotate the inputs
      # train_batch['inputs'] = rescale_batch(train_batch['inputs'], 0.5, 1.1)
      # randomly rescale the inputs
      train_state, t_metrics, t_logs = train_step_pmapped(
          train_state, train_batch
      )
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
          eval_batch['label'] = jnp.asarray(
              eval_batch['label'], dtype=jnp.float32
          )
          # eval_batch['inputs'] = rotate_batch(eval_batch['inputs'], 360)
          #  randomly rotate the inputs
          # eval_batch['inputs'] = rescale_batch(eval_batch['inputs'], 0.4, 1.1)
          # randomly rescale the inputs
          e_metrics, e_output, e_batch = eval_step_pmapped(
              train_state, eval_batch
          )
          eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))
          if compute_global_metrics:
            # Unreplicate outputs of eval_step_pmapped that are coming from
            # `lax.all_gather`, fetch to the host and add to the Evaluator:
            e_batch_mask = train_utils.unreplicate_and_get(
                e_batch['batch_mask']
            ).astype(bool)
            global_metrics_evaluator.add_batch_of_examples(
                target=train_utils.unreplicate_and_get(e_batch['label'])[
                    e_batch_mask
                ],
                output=train_utils.unreplicate_and_get(e_output)[e_batch_mask],
            )
            del e_batch, e_output, e_batch_mask
        eval_global_metrics_summary = None
        if compute_global_metrics:
          eval_global_metrics_summary = (
              global_metrics_evaluator.compute_metrics(clear_annotations=True)
          )
        eval_summary = train_utils.log_eval_summary(
            step=step,
            eval_metrics=eval_metrics,
            extra_eval_summary=eval_global_metrics_summary,
            writer=writer,
        )
      writer.flush()
      del eval_metrics, eval_global_metrics_summary
      chrono.resume()
    ##################### CHECKPOINTING ###################
    if (
        (step % checkpoint_steps == 1 and step > 1) or (step == total_steps)
    ) and config.checkpoint:
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))
      with report_progress.timed('checkpoint'):
        train_utils.handle_checkpointing(
            train_state, chrono, workdir, max_checkpoint_keep
        )
      chrono.resume()

  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary
