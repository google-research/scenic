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

"""Training script for GER models."""

import functools
import time
from typing import Any, Dict, Tuple

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
import numpy as np
import optax
from scenic.dataset_lib import dataset_utils
from scenic.projects.gerald import ger_eval
from scenic.projects.gerald import utils
from scenic.projects.gerald.models import ger_model
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils


def train_step(
    train_state,
    batch,
    *,
    flax_model: nn.Module,
    loss_and_metrics_fn: Any,
    learning_rate_fn: Any,
    entity2code: Any = None,):
  """Run a single step of training.

  Args:
    train_state: learnable parameters and optimizer states.
    batch: a batch of data containing images ("inputs") and annotations.
    flax_model: the model definition.
    loss_and_metrics_fn: loss function.
    learning_rate_fn: Learning rate scheduler which given the global_step
      generates the learning rate.
    entity2code: Entity id to its code.
  Returns:
    new_train_state: updated network parameters and optimizer states.
    lr: the learning rate of the current step (for visualization).
    predictions: the output of the network.
    metrics: losses and other metrics for visualization.
  """
  code_tokens = entity2code(batch['label']['entity/id'][..., 0, 0])
  def loss_fn(params):
    new_rng, rng = jax.random.split(train_state.rng)
    model_rng = train_utils.bind_rng_to_host_device(
        rng, axis_name='batch', bind_to='device')
    variables = {'params': params, **train_state.model_state}
    predictions, new_model_state = flax_model.apply(
        variables,
        batch['inputs'],
        code_tokens=code_tokens,
        context_text_tokens=batch.get('context', None),
        preprocess=True,
        mutable=['batch_stats'],
        train=True,
        rngs={'dropout': model_rng},
    )
    loss, metrics = loss_and_metrics_fn(
        predictions,
        {**{k: v for k, v in batch.items()}, 'code_tokens': code_tokens})
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
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[train_utils.TrainState, Dict[str, Any], Dict[str, Any]]:
  """Main training loop lives in this function.

  Args:
    rng: JAX PRNGKey.
    config: Configurations of the experiment.
    dataset: The dataset that has train_iter, eval_iter, meta_data, and
      optionally, test_iter.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    train_sate that has the state of training (including current
      global_step, model_state, rng, and the optimizer), train_summary
      and eval_summary which are dict of metrics.
  """
  is_host = jax.process_index() == 0

  model = ger_model.GERModel(config, dataset.meta_data)
  rng, init_rng = jax.random.split(rng)
  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(dataset.meta_data['input_shape'],
                    dataset.meta_data.get('input_dtype', jnp.float32))],
       config=config,
       rngs=init_rng)

  lr_fn = lr_schedules.get_learning_rate_fn(config)
  if config.optimizer.get('decoder_multiplier', -1.) >= 0.0:
    tx = utils.optimizer_with_decoder_multiplier(config, params=params)
  else:
    tx = optimizers.get_optimizer(config.optimizer, lr_fn, params=params)
  opt_state = jax.jit(tx.init, backend='cpu')(params)

  _, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
      global_step=0,
      opt_state=opt_state,
      tx=tx,
      params=params,
      model_state=model_state,
      rng=train_rng)

  if config.checkpoint:
    train_state = checkpoints.restore_checkpoint(workdir, train_state)
  start_step = int(train_state.global_step)
  if start_step == 0:
    train_state, start_step = utils.load_weights(train_state, config)
    step0_log = {'num_trainable_params': num_trainable_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  # Mapping from an entity id to its corresponding code.
  entity2code = utils.EntityIds2Code(config)
  code2entity = utils.get_code2id(np.array(entity2code.codes))

  train_step_pmapped = jax.pmap(
      functools.partial(
          train_step,
          flax_model=model.flax_model,
          loss_and_metrics_fn=model.loss_function,
          learning_rate_fn=lr_fn,
          entity2code=entity2code,
      ),
      axis_name='batch', donate_argnums=(0, 1),
  )

  eval_loss_and_accuracy_step_pmapped = jax.pmap(
      functools.partial(
          ger_eval.eval_loss_and_accuracy_step,
          flax_model=model.flax_model,
          loss_and_metrics_fn=model.loss_function,
          entity2code=entity2code,),
      axis_name='batch',
      donate_argnums=(1,),
  )
  eval_step_pmapped = jax.pmap(
      functools.partial(
          ger_eval.eval_step,
          flax_model=model.flax_model, config=config,
      ),
      axis_name='batch', donate_argnums=(1,),
  )

  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data)
  if config.get('early_stopping'):
    total_steps = config.early_stopping
  log_eval_steps = config.get('log_eval_steps', steps_per_epoch)
  log_summary_steps = config.get('log_summary_steps', 20)
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None
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

    if (step % log_eval_steps == 0) or (step == 1) or (step == total_steps):
      start_time = time.time()
      with report_progress.timed('eval'):
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        eval_bs = config.get('eval_batch_size', config.batch_size)
        total_eval_seen_steps = int(np.ceil(
            dataset.meta_data['num_eval_seen_examples'] / eval_bs))
        eval_results = ger_eval.evaluate_oven_ger(
            train_state, dataset.valid_iter, eval_step_pmapped, code2entity,
            total_eval_seen_steps,
            save_predictions=config.get('save_predictions', ''))
        unseen_results = ger_eval.evaluate_oven_ger(
            train_state, dataset.test_iter, eval_step_pmapped, code2entity,
            int(np.ceil(
                dataset.meta_data['num_eval_unseen_examples'] / eval_bs)))
        for k, v in unseen_results.items():
          eval_results['unseen_' + k] = v
          eval_results['hm_' + k] = 2. / (1. / eval_results[k] + 1. / v)

        last_eval_metrics = []
        for _ in range(total_eval_seen_steps):
          eval_batch = next(dataset.valid_iter)
          e_metrics = eval_loss_and_accuracy_step_pmapped(train_state,
                                                          eval_batch)
          last_eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))
        last_eval_step = step
        train_utils.log_eval_summary(
            step=last_eval_step, eval_metrics=last_eval_metrics,
            extra_eval_summary=eval_results, writer=writer)
      duration = time.time() - start_time
      logging.info('Done with evaluation: %.4f sec.', duration)
      writer.flush()

    if config.checkpoint and ((step % checkpoint_steps == 0 and step > 0) or
                              (step == total_steps)):
      with report_progress.timed('checkpoint'):
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        if is_host:
          unrep_train_state = jax_utils.unreplicate(train_state)
          train_utils.save_checkpoint(workdir, unrep_train_state, max_to_keep=1)
          del unrep_train_state
    chrono.resume()  # Un-pause now.

  train_utils.barrier()
  return train_state, train_summary, eval_summary  # pytype: disable=bad-return-type
