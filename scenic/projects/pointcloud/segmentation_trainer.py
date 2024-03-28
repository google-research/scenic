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

"""Training script for semantic segmentation tasks."""

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Type

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
import flax.linen as nn
import jax
from jax.example_libraries.optimizers import clip_grads
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils
from scenic.projects.vivit import evaluation_lib
from scenic.train_lib_deprecated import lr_schedules
from scenic.train_lib_deprecated import optimizers
from scenic.train_lib_deprecated import train_utils


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]


def train_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    learning_rate_fn: Callable[[int], float],
    loss_fn: LossFn,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False
) -> Tuple[train_utils.TrainState, Dict[str, Tuple[float, int]], float,
           jnp.ndarray]:
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
    learning_rate_fn: learning rate scheduler which give the global_step
      generates the learning rate.
    loss_fn: A loss function that given logits, a batch, and parameters of the
      model calculates the loss.
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    config: Configurations of the experiment.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training, computed metrics, learning rate, and predictions
      for logging.
  """
  new_rng, rng = jax.random.split(train_state.rng)
  # Bind the rng to the host/device we are on.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device')

  def training_loss_fn(params):
    class_label = batch.get('class_label', None)
    variables = {'params': params, **train_state.model_state}
    logits, new_model_state = flax_model.apply(
        variables,
        batch['inputs'],
        class_label,
        mutable=['batch_stats'],
        train=True,
        rngs={'dropout': dropout_rng},
        debug=debug)
    loss = loss_fn(logits, batch, variables['params'])
    return loss, (new_model_state, logits)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  step = train_state.global_step
  lr = learning_rate_fn(step)
  (train_cost,
   (new_model_state,
    logits)), grad = compute_gradient_fn(train_state.optimizer.target)

  del train_cost
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')

  if config.get('max_grad_norm', None) is not None:
    grad = clip_grads(grad, config.max_grad_norm)

  new_optimizer = train_state.optimizer.apply_gradient(grad, learning_rate=lr)

  # Explicit weight decay, if necessary.
  if config.get('explicit_weight_decay', None) is not None:
    new_optimizer = new_optimizer.replace(
        target=optimizers.tree_map_with_names(
            functools.partial(
                optimizers.decay_weight_fn,
                lr=lr,
                decay=config.explicit_weight_decay),
            new_optimizer.target,
            match_name_fn=lambda name: 'kernel' in name))

  metrics = metrics_fn(logits, batch)
  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=step + 1,
      optimizer=new_optimizer,
      model_state=new_model_state,
      rng=new_rng)
  return new_train_state, metrics, lr, jnp.argmax(logits, axis=-1)


def eval_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: MetricFn,
    debug: Optional[bool] = False
) -> Tuple[Batch, jnp.ndarray, Dict[str, Tuple[float, int]], jnp.ndarray]:
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
    debug: Whether the debug mode is enabled during evaluation. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Batch, predictions and calculated metrics.
  """
  variables = {
      'params': train_state.optimizer.target,
      **train_state.model_state
  }
  class_label = batch.get('class_label', None)
  logits = flax_model.apply(
      variables,
      batch['inputs'],
      class_label,
      train=False,
      mutable=False,
      debug=debug)
  metrics = metrics_fn(logits, batch)

  confusion_matrix = get_confusion_matrix(
      labels=batch['label'], logits=logits, batch_mask=batch['batch_mask'])

  # Collect predictions and batches from all hosts.
  predictions = jnp.argmax(logits, axis=-1)
  predictions = jax.lax.all_gather(predictions, 'batch')
  batch = jax.lax.all_gather(batch, 'batch')
  confusion_matrix = jax.lax.all_gather(confusion_matrix, 'batch')

  return batch, predictions, metrics, confusion_matrix


def get_confusion_matrix(*, labels, logits, batch_mask):
  """Computes the confusion matrix that is necessary for global mIoU."""
  if labels.ndim == logits.ndim:  # One-hot targets.
    y_true = jnp.argmax(labels, axis=-1)
  else:
    y_true = labels
    # Set excluded pixels (label -1) to zero, because the confusion matrix
    # computation cannot deal with negative labels. They will be ignored due to
    # the batch_mask anyway:
    y_true = jnp.maximum(y_true, 0)
  y_pred = jnp.argmax(logits, axis=-1)

  # Prepare sample weights for confusion matrix:
  weights = batch_mask.astype(jnp.float32)
  # Normalize weights by number of samples to avoid having very large numbers in
  # the confusion matrix, which could lead to imprecise results (note that we
  # should not normalize by sum(weights) because that might differ between
  # devices/hosts):
  weights = weights / weights.size

  confusion_matrix = model_utils.confusion_matrix(
      y_true=y_true,
      y_pred=y_pred,
      num_classes=logits.shape[-1],
      weights=weights)
  confusion_matrix = confusion_matrix[jnp.newaxis, ...]  # Dummy batch dim.
  return confusion_matrix


def calculate_iou(predictions, labels, n_classes):
  """Calculates mean IoU of the entire test set."""
  all_intersection = np.zeros(n_classes)
  all_union = np.zeros(n_classes)
  for sem_idx in range(labels.shape[0]):
    for sem in range(n_classes):
      intersection = np.sum(
          np.logical_and(predictions[sem_idx] == sem, labels[sem_idx] == sem))
      union = jnp.sum(
          np.logical_or(predictions[sem_idx] == sem, labels[sem_idx] == sem))
      all_intersection[sem] += intersection
      all_union[sem] += union
  return np.mean(all_intersection / all_union)


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
  if dataset.meta_data.get('label_shape', None) is not None:
    input_specs = [(dataset.meta_data['input_shape'],
                    dataset.meta_data.get('input_dtype', jnp.float32)),
                   (dataset.meta_data['label_shape'],
                    dataset.meta_data.get('label_dtype', jnp.int64))]
  else:
    input_specs = [(dataset.meta_data['input_shape'],
                    dataset.meta_data.get('input_dtype', jnp.float32))]

  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=input_specs,
       config=config,
       rngs=init_rng)

  # Create optimizer.
  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  optimizer = jax.jit(
      optimizers.get_optimizer(config).create, backend='cpu')(
          params)
  rng, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
      global_step=0,
      optimizer=optimizer,
      model_state=model_state,
      rng=train_rng,
      accum_train_time=0)
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
  # Get learning rate scheduler.
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)

  train_step_pmapped = jax.pmap(
      functools.partial(
          train_step,
          flax_model=model.flax_model,
          learning_rate_fn=learning_rate_fn,
          loss_fn=model.loss_function,
          metrics_fn=model.get_metrics_fn('train'),
          config=config,
          debug=config.debug_train),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0, 1),
  )

  ############### EVALUATION CODE #################

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

  # Ceil rounding such that we include the last incomplete batch.
  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  def evaluate(train_state: train_utils.TrainState,
               step: int) -> Dict[str, Any]:
    eval_metrics = []
    eval_all_confusion_mats = []
    # Sync model state across replicas.
    train_state = train_utils.sync_model_state_across_replicas(train_state)

    # n_classes = dataset.meta_data['num_classes']

    def to_cpu(x):
      return jax.device_get(dataset_utils.unshard(jax_utils.unreplicate(x)))

    for _ in range(steps_per_eval):
      eval_batch = next(dataset.valid_iter)
      _, _, e_metrics, confusion_matrix = eval_step_pmapped(
          train_state, eval_batch)
      eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))
      # Evaluate global metrics on one of the hosts (lead_host), but given
      # intermediate values collected from all hosts.
      if lead_host and global_metrics_fn is not None:
        # Collect data to be sent for computing global metrics.
        eval_all_confusion_mats.append(to_cpu(confusion_matrix))

    eval_global_metrics_summary = {}
    if lead_host and global_metrics_fn is not None:
      # eval_global_metrics_summary = global_metrics_fn(eval_all_confusion_mats,
      #                                                 dataset.meta_data)
      eval_global_metrics_summary = evaluation_lib.compute_confusion_matrix_metrics(
          eval_all_confusion_mats, return_per_class_metrics=True)

    ############### LOG EVAL SUMMARY ###############
    eval_summary = train_utils.log_eval_summary(
        step=step,
        eval_metrics=eval_metrics,
        extra_eval_summary=eval_global_metrics_summary,
        writer=writer)

    test_summary = None
    if dataset.meta_data.get('num_test_examples', None) is not None:
      test_metrics = []
      test_all_confusion_mats = []

      total_test_steps = int(
          np.ceil(dataset.meta_data['num_test_examples'] / eval_batch_size))

      for _ in range(total_test_steps):
        test_batch = next(dataset.test_iter)
        _, _, e_metrics, confusion_matrix = eval_step_pmapped(
            train_state, test_batch)
        test_metrics.append(train_utils.unreplicate_and_get(e_metrics))
        # Evaluate global metrics on one of the hosts (lead_host), but given
        # intermediate values collected from all hosts.
        if lead_host and global_metrics_fn is not None:
          # Collect data to be sent for computing global metrics.
          test_all_confusion_mats.append(to_cpu(confusion_matrix))

      test_global_metrics_summary = {}
      if lead_host and global_metrics_fn is not None:
        test_global_metrics_summary = evaluation_lib.compute_confusion_matrix_metrics(
            test_all_confusion_mats, return_per_class_metrics=True)

      ############### LOG TEST SUMMARY ###############
      test_summary = train_utils.log_eval_summary(
          step=step,
          eval_metrics=test_metrics,
          extra_eval_summary=test_global_metrics_summary,
          writer=writer,
          prefix='test')

    writer.flush()
    del test_summary
    del eval_metrics
    return eval_summary

  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None
  global_metrics_fn = model.get_global_metrics_fn()  # pytype: disable=attribute-error

  chrono = train_utils.Chrono(
      first_step=start_step,
      total_steps=total_steps,
      steps_per_epoch=steps_per_epoch,
      global_bs=config.batch_size,
      accum_train_time=int(jax_utils.unreplicate(train_state.accum_train_time)))

  logging.info('Starting training loop at step %d.', start_step + 1)
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

  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', sfLtep_num=step):
      train_batch = next(dataset.train_iter)
      train_state, t_metrics, lr, _ = train_step_pmapped(
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
      extra_training_logs.append({'learning_rate': lr})

    for h in hooks:
      h(step)
    chrono.pause()  # Below are once-in-a-while ops -> pause.
    if step % log_summary_steps == 0 or (step == total_steps):
      ############### LOG TRAIN SUMMARY ###############
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(train_utils.unreplicate_and_get,
                                               train_metrics),
          extra_training_logs=jax.tree_util.tree_map(
              train_utils.unreplicate_and_get, extra_training_logs),
          writer=writer)
      # Reset metric accumulation for next evaluation cycle.
      train_metrics, extra_training_logs = [], []

    if (step % log_eval_steps == 0) or (step == total_steps):
      with report_progress.timed('eval'):
        # Sync model state across replicas (in case of having model state, e.g.
        # batch statistic when using batch norm).
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        eval_summary = evaluate(train_state, step)

    if ((step % checkpoint_steps == 0 and step > 0) or
        (step == total_steps)) and config.checkpoint:
      ################### CHECK POINTING ##########################
      with report_progress.timed('checkpoint'):
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        if lead_host:
          train_state.replace(  # pytype: disable=attribute-error
              accum_train_time=chrono.accum_train_time)
          train_utils.save_checkpoint(workdir, train_state)

    chrono.resume()  # Un-pause now.

  # Wait until computations are done before exiting.
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  # Return the train and eval summary after last step for regresesion testing.
  return train_state, train_summary, eval_summary
