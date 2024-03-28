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

"""Training script for the CenterNet."""

import functools
import time
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
from flax import jax_utils
import flax.linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import optax
from scenic.dataset_lib import dataset_utils
from scenic.projects.baselines.centernet import evaluate
from scenic.projects.baselines.centernet import optimizer_utils
from scenic.projects.baselines.centernet import train_utils as centernet_train_utils
from scenic.projects.baselines.centernet.modeling import centernet2
from scenic.train_lib import train_utils


def train_step(
    train_state,
    batch,
    *,
    flax_model: nn.Module,
    loss_and_metrics_fn: Any,
    learning_rate_fn: Any,
    debug: bool = False):
  """Run a single step of training.

  Args:
    train_state: learnable parameters and optimizer states.
    batch: a batch of data containing images ("inputs") and annotations.
    flax_model: the model definition.
    loss_and_metrics_fn: loss function.
    learning_rate_fn: Learning rate scheduler which given the global_step
      generates the learning rate.
    debug: enable debug mode or not.
  Returns:
    new_train_state: updated network parameters and optimizer states.
    lr: the learning rate of the current step (for visualization).
    predictions: the output of the network.
    metrics: losses and other metrics for visualization.
  """
  def loss_fn(params):
    new_rng, rng = jax.random.split(train_state.rng)
    model_rng = train_utils.bind_rng_to_host_device(
        rng, axis_name='batch', bind_to='device')
    variables = {'params': params, **train_state.model_state}
    kwargs = {}
    if isinstance(flax_model, centernet2.CenterNet2Detector):
      # Two-stage detectors adds gt boxes in RoI heads in training.
      kwargs['gt_boxes'] = batch['label']['boxes']
      kwargs['gt_classes'] = batch['label']['labels']
    predictions, new_model_state = flax_model.apply(
        variables,
        batch['inputs'],
        preprocess=True,
        padding_mask=batch['padding_mask'],
        mutable=['batch_stats'],
        train=True,
        rngs={'dropout': model_rng},
        debug=debug,
        **kwargs)
    loss, metrics = loss_and_metrics_fn(predictions, batch)
    # adapt to normalization API in log_train_summary
    metrics = {k: (v, 1.) for k, v in metrics.items()}
    return loss, (new_model_state, new_rng, metrics, predictions)

  compute_gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, aux), grad = compute_gradient_fn(train_state.params)
  new_model_state, new_rng, metrics, predictions = aux

  step = train_state.global_step
  lr = learning_rate_fn(step)
  grad = jax.lax.pmean(grad, axis_name='batch')
  updates, new_opt_state = train_state.tx.update(
      grad, train_state.opt_state, train_state.params)
  new_params = optax.apply_updates(train_state.params, updates)

  new_train_state = train_state.replace(
      global_step=step + 1,
      opt_state=new_opt_state,
      params=new_params,
      model_state=new_model_state,
      rng=new_rng)
  return new_train_state, lr, predictions, metrics


def train_and_evaluate(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Any,
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
):
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

  # Initialize model class (without parameters)
  model = model_cls(config, dataset.meta_data)

  # Create and initialize model parameters
  rng, init_rng = jax.random.split(rng)
  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(dataset.meta_data['input_shape'],
                    dataset.meta_data.get('input_dtype', jnp.float32))],
       config=config,
       rngs=init_rng)

  # Create and initialize optimizer parameters
  tx, lr_fn = optimizer_utils.create_optimizer_and_lr_fn(params, config)
  opt_state = tx.init(params)

  # Initialize "train_state" class, which contains all parameters.
  _, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
      global_step=0,
      opt_state=opt_state,
      tx=tx,
      params=params,
      model_state=model_state,
      rng=train_rng)

  # Resume (interrupted) training from the same workdir
  train_state = checkpoints.restore_checkpoint(workdir, train_state)
  start_step = int(train_state.global_step)

  # Load pretrained weights at the first step
  if start_step == 0:
    train_state, start_step = centernet_train_utils.load_weights(
        train_state, config)
    step0_log = {'num_trainable_params': num_trainable_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  # Define the function for each train step, and make it run on devices (pmap).
  train_step_pmapped = jax.pmap(
      functools.partial(
          train_step,
          flax_model=model.flax_model,
          loss_and_metrics_fn=model.loss_function,
          learning_rate_fn=lr_fn,
          debug=config.debug_train,
      ),
      axis_name='batch', donate_argnums=(0,),
  )

  # Define log options
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
  logging.info('Starting training loop at step %d.', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer)
  hooks = []
  if is_host:
    hooks.append(report_progress)
  if config.get('xprof', True) and is_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  # The actual train loop
  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      # Get a batch of training data
      train_batch = next(dataset.train_iter)

      # Actual training happens here.
      train_state, lr, train_predictions, metrics = train_step_pmapped(
          train_state, train_batch)

      train_metrics.append(metrics)
      extra_training_logs.append({'learning_rate': lr})
    for h in hooks:
      h(step)
    chrono.pause()
    del train_predictions

    # Print train log
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

    # Run evaluation
    if (step % log_eval_steps == 0) or (step == total_steps):
      start_time = time.time()
      with report_progress.timed('eval'):
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        last_eval_results, last_eval_metrics = evaluate.inference_on_dataset(
            model.flax_model,
            train_state, dataset,
            eval_batch_size=eval_batch_size,
            is_host=is_host,
            save_dir=workdir,
            config=config)
        last_eval_step = step
        train_utils.log_eval_summary(
            step=last_eval_step, eval_metrics=last_eval_metrics,
            extra_eval_summary=last_eval_results, writer=writer)
      duration = time.time() - start_time
      logging.info('Done with evaluation: %.4f sec.', duration)
      writer.flush()

    # Handle checkpointing
    if ((step % checkpoint_steps == 0 and step > 0) or
        (step == total_steps)):
      with report_progress.timed('checkpoint'):
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        if is_host:
          unrep_train_state = jax_utils.unreplicate(train_state)
          train_utils.save_checkpoint(workdir, unrep_train_state, max_to_keep=1)
          del unrep_train_state
    chrono.resume()  # Un-pause now.

  train_utils.barrier()
  return train_state, train_summary, eval_summary
