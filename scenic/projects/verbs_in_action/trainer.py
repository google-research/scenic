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

"""Video-Text training."""

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
import jax.profiler
import ml_collections
import numpy as np
import optax
from scenic.projects.verbs_in_action import utils
from scenic.train_lib import train_utils


def train_and_eval(
    rng: np.ndarray,
    config: ml_collections.ConfigDict,
    *,
    workdir: str,
    writer: Any,
    model_cls,
    dataset) -> Tuple[utils.OptaxTrainState, Any, Dict[str, Any]]:
  """Train (and occasionally evaluate) the model.

  Args:
    rng: JAX prng key.
    config: The configuration of the experiment.
    workdir: Where to checkpoint and write the summaries.
    writer: Summary writer object.
    model_cls: The model class used to instantiate the model.
    dataset: The dataset for training and evaluation.

  Returns:
    A tuple with:
      * the state that has the state of training (including current
        global_step, model_state, rng, and the optimizer)
      * a dictionary with the train_summary
      * a dictionary with the evaluation summary
  """
  lead_host = jax.host_id() == 0

  model = model_cls(config, dataset.meta_data)
  train_step_pmapped, eval_step_pmapped = pmapped_steps(model, config)
  train_state, start_step, chrono = utils.init_state(model, dataset, config,  # pytype: disable=wrong-arg-types  # jax-ndarray
                                                     workdir, rng)
  chrono.load(train_state.metadata['chrono'])
  train_state = jax_utils.replicate(train_state)

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data)
  log_eval_steps = config.get('log_eval_steps', steps_per_epoch)
  checkpoint_steps = config.get('checkpoint_steps', log_eval_steps)
  log_summary_steps = config.get('log_summary_steps', log_eval_steps)

  # And the number of evaluation steps.
  num_eval_examples = dataset.meta_data['num_eval_examples']
  logging.info('Number of processes is %s', jax.process_count())
  total_eval_steps = int(np.ceil(num_eval_examples /(config.get('batch_size'))))
  steps_per_eval = config.get('steps_per_eval', total_eval_steps)
  train_metrics = []
  train_summary = None


  eval_summary = {}
  logging.info('Start training from step %d', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer)
  def write_note(note):
    if lead_host:
      platform.work_unit().set_notes(note)
  hooks = []
  if lead_host:
    hooks.append(report_progress)
  if config.get('xprof', True) and jax.process_index() == 0:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))
  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)
  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      train_state, t_metrics = train_step_pmapped(train_state, train_batch)
      train_metrics.append(t_metrics)
    for hook in hooks:
      hook(step)
    # Log the train summary every `log_summary_steps`.
    if (step % log_summary_steps == 1) or (step == total_steps):
      chrono.pause()
      if lead_host:
        chrono.tick(step, writer, write_note)
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(
              train_utils.unreplicate_and_get, train_metrics),
          writer=writer)
      train_metrics = []
      chrono.resume()

    # Evaluate every `log_eval_steps`.
    do_eval = (step % log_eval_steps == 1) or (step == total_steps)
    if do_eval:
      eval_summary = eval_and_log_summary(
          train_state=train_state,
          iterator=dataset.valid_iter,
          eval_step_fn=eval_step_pmapped,
          eval_steps=steps_per_eval,
          writer=writer,
          train_iteration=step,
          num_eval_examples=num_eval_examples,
          compute_recall_metrics=True)
      writer.flush()

    # Checkpointing.
    if not config.checkpoint:
      continue
    elif do_eval or (step % checkpoint_steps == 0 and step > 0):
      chrono.pause(wait_for=(train_state.weights, train_state.opt_state))
      if lead_host:
        # Take the first replica.
        unrep_train_state = jax_utils.unreplicate(train_state)
        metadata = unrep_train_state.metadata
        metadata['chrono'] = chrono.save()
        unrep_train_state.replace(metadata=metadata)  # pytype: disable=attribute-error
        utils.save_checkpoint(workdir, unrep_train_state)
        del unrep_train_state
      chrono.resume()  # Un-pause now.
  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary


def train_step(
    train_state: utils.OptaxTrainState,
    batch: Any,
    *,
    flax_model: nn.Module,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False
) -> Tuple[utils.OptaxTrainState, Dict[str, Tuple[float, int]]]:
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
    config: Configuration of the experiment.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training, computed metrics, and learning rate for logging.
  """
  new_rng, rng = jax.random.split(train_state.rng)

  # Bind the rng to the host/device we are on for dropout.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device')

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    encoded_video, encoded_text = flax_model.apply(
        variables,
        batch['inputs'].get('rgb'),
        batch['text_indices'],
        train=True,
        rngs={'dropout': dropout_rng},
        debug=debug)
    encoded_verbs = None
    if config.get('verb_phrase_loss_weight'):
      # We pass None for video inputs so video encoder doesn't duplicate calc.
      _, encoded_verbs = flax_model.apply(
          variables,
          None,
          batch['verb_indices'],
          train=True,
          rngs={'dropout': dropout_rng},
          debug=debug)
    return flax_model.loss_function(
        encoded_video, encoded_text, batch, config, encoded_verbs=encoded_verbs)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=False)
  step = train_state.global_step
  train_loss, grad = compute_gradient_fn(train_state.weights)
  new_train_state = train_state
  metrics = {'loss': (train_loss, 1)}
  grad = jax.lax.pmean(grad, axis_name='batch')
  if config.get('max_grad_norm', None):
    grad = jax_optimizers.clip_grads(grad, config.max_grad_norm)
  if train_state.tx is not None:
    updates, new_opt_state = train_state.tx.update(
        grad, train_state.opt_state, train_state.weights)
    new_weights = optax.apply_updates(train_state.weights, updates)
    new_train_state = train_state.replace(  # pytype: disable=attribute-error
        global_step=step + 1,
        opt_state=new_opt_state,
        weights=new_weights,
        rng=new_rng)
  return new_train_state, metrics


def eval_step(train_state: utils.OptaxTrainState,
              batch: Any,
              *,
              flax_model: nn.Module,
              debug: Optional[bool] = False,) -> Any:
  """Runs a single step of evaluation."""
  variables = {'params': train_state.weights, **train_state.model_state}
  (encoded_video, encoded_text) = flax_model.apply(
      variables,
      batch['inputs'].get('rgb'),
      batch['text_indices'],
      train=False,
      mutable=False,
      debug=debug)
  # This function uses all_gather to fetch embeddings from all devices.
  _, encoded_video, encoded_text = utils.compute_inners(
      encoded_video, encoded_text, 'batch', return_embeddings=True)
  return encoded_video, encoded_text


def pmapped_steps(model, config):
  """Returns the pmapped train and eval steps."""
  train_step_pmapped = jax.pmap(
      functools.partial(
          train_step,
          flax_model=model.flax_model,
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
          debug=config.debug_eval),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1,),
  )
  return train_step_pmapped, eval_step_pmapped


def eval_and_log_summary(
    *,
    train_state: utils.OptaxTrainState,
    writer: metric_writers.MetricWriter,
    iterator,
    eval_step_fn,
    eval_steps,
    train_iteration,
    num_eval_examples,
    compute_recall_metrics=True,
    text_to_video_retrieval=True,):
  """Eval the model and write the summary."""
  output_dicts = {}
  logging.info('Total number of eval steps is %s', eval_steps)
  logging.info('Total number of eval examples is %s', num_eval_examples)
  # Do this to ensure we definitely cover the full test set
  eval_steps = int(np.ceil(1.3 * eval_steps))
  logging.info('The modified total number of eval steps is %s', eval_steps)
  for step in range(eval_steps):
    logging.info('Step %s/%s', step + 1, eval_steps)
    with jax.profiler.StepTraceAnnotation('eval', step_num=step):
      eval_batch = next(iterator)
      assert compute_recall_metrics
      assert 'key' in eval_batch, 'Keys must be added to batch'
      keys = utils.convert_strings_to_uint8_arrays(eval_batch['key'], 30)
      keys = utils.all_gather_and_unreplicate(keys)
      del eval_batch['key']
      batch_masks = utils.all_gather_and_unreplicate(
          eval_batch['batch_mask'])

      video_embeddings, text_embeddings = eval_step_fn(
          train_state, eval_batch)
      # Unreplicate the output of eval_step_pmapped (used `lax.all_gather`).
      video_embeddings = jax_utils.unreplicate(video_embeddings)
      text_embeddings = jax_utils.unreplicate(text_embeddings)
      batch_size = batch_masks.shape[0] * batch_masks.shape[1]
      text_embeddings = text_embeddings.reshape(
          batch_size, -1, text_embeddings.shape[-1])
      batch_masks = batch_masks.reshape(
          (batch_size,)).astype(bool)
      keys = keys.reshape((batch_size, -1))
      for i, mask in enumerate(batch_masks):
        if mask:
          key = utils.convert_uint8_array_to_string(keys[i])
          output_dicts[key] = {
              'text_emb': text_embeddings[i],
              'video_emb': video_embeddings[i]
          }
  logging.info('The number of the unique eval examples is %d',
               len(output_dicts))
  text_embeddings_array = np.stack(
      [v['text_emb'] for v in output_dicts.values()], axis=0)
  video_embeddings_array = np.stack(
      [v['video_emb'] for v in output_dicts.values()], axis=0)

  logging.info('Shape of text embedding array in val set is %s',
               text_embeddings_array.shape)
  additional_summary = utils.compute_recall_at_k(
      video_embeddings=video_embeddings_array,
      text_embeddings=text_embeddings_array,
      k_values={1, 5, 10},
      text_to_video_retrieval=text_to_video_retrieval)
  return train_utils.log_eval_summary(
      step=train_iteration,
      eval_metrics=[],
      extra_eval_summary=additional_summary,
      writer=writer,
      key_separator='/')
