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

"""Training script for the DETR."""

from concurrent import futures
import functools
import time
from typing import Any, Callable, Dict, Tuple, Optional

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from flax.training.checkpoints import restore_checkpoint as flax_restore_checkpoint
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
import optax

from scenic.projects.baselines.detr import train_utils as detr_train_utils
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils

PyTree = Any

def get_train_step(apply_fn: Callable[..., Tuple[PyTree, PyTree]],
                  loss_and_metrics_fn: Callable[..., Tuple[PyTree, PyTree]],
                  tx: optax.GradientTransformation,
                  update_batch_stats: bool = False,
                  debug: bool = False):
  """Runs a single step of training.
  
  Given the state of the training and a batch of data, computes the loss and
  updates the parameters of the model.

  Note that in this code, the buffers of the first (train_state) argument is
  donated to the computation.

  Args:
    apply_fn: Model application function.
    loss_and_metrics_fn: Function to calculate loss and metrics..
    tx: An optax.GradientTransformation
    update_batch_stats: Whether to update batch statistics in BatchNorm
      during training or freeze it.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific lossing/storing some values using
      jax.host_callback.
  
  Returns:
    Train step function that takes a train_state and batch and returns 
    new_train_state, metrics, predictions.
  """
  def train_step(train_state, batch):

    def loss_fn(params):
      new_rng, rng = jax.random.split(train_state.rng)
      # Bind the rng to the host/device we are on.
      model_rng = train_utils.bind_rng_to_host_device(
          rng, axis_name='batch', bind_to='device')
      variables = {'params': params, **train_state.model_state}
      predictions, mutated_variables = apply_fn(
        variables,
        batch['inputs'],
        padding_mask=batch['padding_mask'],
        update_batch_stats=update_batch_stats,
        mutable=train_state.model_state.keys(),
        train=True,
        rngs={'dropout': model_rng},
        debug=debug)
      loss, metrics = loss_and_metrics_fn(
        predictions, batch, model_params=variables['params'])
      return loss, (mutated_variables, metrics, predictions, new_rng)
  
    new_global_step = train_state.global_step + 1
    (_, (new_model_state, metrics, predictions, 
         new_rng)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
          train_state.params)
    
    # Re-use same axis_name as in the call to `pmap(...train_step...)`
    grads = jax.tree_map(lambda g: jnp.asarray(g, jnp.bfloat16), grads)
    grads = jax.lax.pmean(grads, axis_name='batch')

    updates, new_opt_state = tx.update(
        grads, train_state.opt_state, params=train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)
    train_state = train_state.replace(
        global_step=new_global_step,
        params=new_params,
        opt_state=new_opt_state,
        model_state=new_model_state,
        rng=new_rng)
    return train_state, metrics, predictions
  
  return train_step


def get_eval_step(flax_model,
                  loss_and_metrics_fn,
                  logits_to_probs_fn,
                  metrics_only=False,
                  debug=False):
  """Runs a single step of training.

  Note that in this code, the buffer of the second argument (batch) is donated
  to the computation.

  Args:
    flax_model: Flax model (an instance of nn.Module).
    loss_and_metrics_fn: A function that given model predictions, a batch, and
      parameters of the model calculates the loss as well as metrics.
    logits_to_probs_fn: Function that takes logits and converts them to probs.
    metrics_only: bool; Only return metrics.
    debug: bool; Whether the debug mode is enabled during evaluation.
      `debug=True` enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Eval step function which returns predictions and calculated metrics.
  """

  def metrics_fn(train_state, batch, predictions):
    _, metrics = loss_and_metrics_fn(
        predictions, batch, model_params=train_state.params)

    if metrics_only:
      return None, None, metrics

    pred_probs = logits_to_probs_fn(predictions['pred_logits'])
    # Collect necessary predictions and target information from all hosts.
    predictions_out = {
        'pred_probs': pred_probs,
        'pred_logits': predictions['pred_logits'],
        'pred_boxes': predictions['pred_boxes']
    }
    labels = {
        'image/id': batch['label']['image/id'],
        'size': batch['label']['size'],
        'orig_size': batch['label']['orig_size'],
    }
    to_copy = [
        'labels', 'boxes', 'not_exhaustive_category_ids', 'neg_category_ids'
    ]
    for name in to_copy:
      if name in batch['label']:
        labels[name] = batch['label'][name]

    targets = {'label': labels, 'batch_mask': batch['batch_mask']}

    predictions_out = jax.lax.all_gather(predictions_out, 'batch')
    targets = jax.lax.all_gather(targets, 'batch')
    return targets, predictions_out, metrics

  def eval_step(train_state, batch):
    variables = {
        'params': train_state.params,
        **train_state.model_state
    }
    predictions = flax_model.apply(
        variables,
        batch['inputs'],
        padding_mask=batch['padding_mask'],
        train=False,
        mutable=False,
        debug=debug)
    return metrics_fn(train_state, batch, predictions)

  return eval_step

def _handle_legacy_format(train_state):
  """Handle legacy format.

  To remove this function, make sure all checkpoints that are to be loaded are
  in the non-legacy format:
    * checkpoint['params'] is a PyTree of parameters;
    * checkpoint['model_state']['batch_stats'] is a PyTree of batch statistics.

  Args:
    train_state: A train state.
  """
  if 'params' not in train_state:
    train_state['params'] = train_state.pop('optimizer').pop(
        'target')
    if 'params' in train_state['params']:
      train_state['params'] = train_state['params']['params']
  if 'model_state' in train_state and 'batch_stats' not in train_state[
      'model_state']:
    bad_restored_batch_stats = train_state.pop('model_state')
    good_restored_batch_stats = {}
    for key, value in bad_restored_batch_stats.items():
      current = good_restored_batch_stats
      subkeys = [subkey for subkey in key.split('/') if subkey]
      for subkey in subkeys[:-1]:
        if subkey not in current:
          new_current = {}
          current[subkey] = new_current
          current = new_current
        else:
          current = current[subkey]
      current[subkeys[-1]] = value
    train_state['model_state'] = {}
    train_state['model_state']['batch_stats'] = good_restored_batch_stats

def train_and_evaluate(
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
  lead_host = jax.process_index() == 0

  # The pool is used to perform misc operations such as logging in async way.
  pool = futures.ThreadPoolExecutor(max_workers=2)

  # Build the loss_and_metrics_fn, metrics, and flax_model.
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
  # device as the input is, in this case the CPU. Else they'd be on device[0]
  tx = detr_train_utils.get_detr_optimizer(config, params)
  opt_state = jax.jit(tx.init, backend='cpu')(params)

  rng, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
      global_step=0,
      params=params,
      opt_state=opt_state,
      model_state=model_state,
      rng=train_rng)

  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state)

  if (start_step == 0  # Which means "no" checkpoint is restored!
      and config.get('init_from') is not None):
    init_checkpoint_path = config.init_from.get('checkpoint_path')
    restored_train_state = flax_restore_checkpoint(
        init_checkpoint_path, target=None)
    train_state = pretrain_utils.init_from_pretrain_state(
        train_state,
        restored_train_state,
        ckpt_prefix_path=config.init_from.get('ckpt_prefix_path'),
        model_prefix_path=config.init_from.get('model_prefix_path'),
        name_mapping=config.init_from.get('name_mapping'),
        skip_regex=config.init_from.get('skip_regex'))
    # Free unecessary memory.
    del restored_train_state
  elif start_step == 0 and config.get('load_pretrained_backbone', False):
    # Only load pretrained backbone if we are at the beginning of training.
    bb_checkpoint_path = config.pretrained_backbone_configs.get(
        'checkpoint_path')
    bb_train_state = flax_restore_checkpoint(bb_checkpoint_path, target=None)
    _handle_legacy_format(bb_train_state)
    train_state = pretrain_utils.init_from_pretrain_state(
        train_state, bb_train_state, model_prefix_path=['backbone'])

  update_model_state = not config.get('freeze_backbone_batch_stats', False)
  if not update_model_state:
    if not config.load_pretrained_backbone:
      raise ValueError('Freezing the batch statistics of the resnet backbone '
                       'is only possible when loading a pretrained resnet '
                       'backbone is enabled.')
  # Replicate the optimzier, state, and rng.
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data)

  update_batch_stats = not config.get('freeze_backbone_batch_stats', False)
  if not update_batch_stats:
    if not config.load_pretrained_backbone:
      raise ValueError('Freezing the batch statistics of the resnet backbone '
                       'is only possible when loading a pretrained resnet '
                       'backbone is enabled.')

  train_step = get_train_step(
    apply_fn=model.flax_model.apply,
    loss_and_metrics_fn=model.loss_function,
    tx=tx, 
    update_batch_stats=update_batch_stats,
    debug=config.debug_train
  )

  train_step_pmapped = jax.pmap(
      train_step, axis_name='batch', donate_argnums=(0,))

  ############### EVALUATION CODE #################
  eval_step = get_eval_step(
      flax_model=model.flax_model,
      loss_and_metrics_fn=model.loss_function,
      logits_to_probs_fn=model.logits_to_probs,
      debug=config.debug_eval)
  eval_step_pmapped = jax.pmap(
      eval_step, axis_name='batch', donate_argnums=(1,))

  # Ceil rounding such that we include the last incomplete batch.
  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  metrics_normalizer_fn = functools.partial(
      detr_train_utils.normalize_metrics_summary,
      object_detection_loss_keys=model.loss_terms_weights.keys())

  def evaluate(train_state, step):
    """Runs evaluation code."""
    future = None

    def _wait(future: Optional[futures.Future]) -> Any:  # pylint: disable=g-bare-generic
      if future is None:
        return None
      return future.result()

    def _add_examples(predictions, labels):
      for pred, label in zip(predictions, labels):
        global_metrics_evaluator.add_example(prediction=pred, target=label)

    eval_metrics = []
    if global_metrics_evaluator is not None:
      global_metrics_evaluator.clear()

    for eval_step in range(steps_per_eval):
      logging.info('Running eval step %d', eval_step)
      eval_batch = next(dataset.valid_iter)

      # Do the eval step given the matches.
      (eval_batch_all_hosts, eval_predictions_all_hosts,
       e_metrics) = eval_step_pmapped(train_state, eval_batch)

      # Variable aux_outputs is not needed anymore.
      eval_predictions_all_hosts.pop('aux_outputs', None)

      # Collect local metrics (returned by the loss function).
      eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))

      if global_metrics_evaluator is not None:
        # Unreplicate the output of eval_step_pmapped (used `lax.all_gather`).
        eval_batch_all_hosts = jax_utils.unreplicate(eval_batch_all_hosts)
        eval_predictions_all_hosts = jax_utils.unreplicate(
            eval_predictions_all_hosts)

        # Collect preds and labels to be sent for computing global metrics.
        predictions = detr_train_utils.process_and_fetch_to_host(
            eval_predictions_all_hosts, eval_batch_all_hosts['batch_mask'])
        predictions = jax.tree_util.tree_map(np.asarray, predictions)

        labels = detr_train_utils.process_and_fetch_to_host(
            eval_batch_all_hosts['label'], eval_batch_all_hosts['batch_mask'])
        labels = jax.tree_util.tree_map(np.asarray, labels)

        if eval_step == 0:
          logging.info('Pred keys: %s', list(predictions[0].keys()))
          logging.info('Labels keys: %s', list(labels[0].keys()))

        # Add to evaluator.
        _wait(future)
        future = pool.submit(_add_examples, predictions, labels)

        del predictions, labels

      del eval_batch, eval_batch_all_hosts, eval_predictions_all_hosts

    eval_global_metrics_summary_future = None
    if global_metrics_evaluator is not None:
      _wait(future)
      logging.info('Number of eval examples: %d', len(global_metrics_evaluator))
      if lead_host:
        eval_global_metrics_summary_future = pool.submit(
            global_metrics_evaluator.compute_metrics, clear_annotations=False)

    return (step, eval_metrics), eval_global_metrics_summary_future

  ###################################################

  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  log_summary_steps = config.get('log_summary_steps', 25)
  log_large_summary_steps = config.get('log_large_summary_steps', 0)
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps

  global_metrics_evaluator = None  # Only run eval on the lead_host node.
  if lead_host:
    global_metrics_evaluator = detr_train_utils.DetrGlobalEvaluator(
        config.dataset_name)

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None

  logging.info('Starting training loop at step %d.', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps,
      writer=writer,
      every_secs=None,
      every_steps=config.get('report_progress_step', log_summary_steps),
  )
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

  (last_eval_step, last_eval_metrics), last_eval_future = (None, None), None
  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      (train_state, t_metrics, train_predictions) = train_step_pmapped(
        train_state, train_batch)
      # This will accumulate metrics in TPU memory up to the point that we log
      # them. This is no problem for small metrics but may be a problem for
      # large (e.g. segmentation) metrics. An alternative is to set
      # `log_summary_steps` to a small number, or to use
      # `train_utils.unreplicate_and_get` here instead of right before writing
      # summaries, but that means in each step, we have data transfer between
      # tpu and host, which might slow down the training.
      train_metrics.append(train_utils.unreplicate_and_get(t_metrics))

    for h in hooks:
      h(step)

    if (log_large_summary_steps and step % log_large_summary_steps == 0 and
        lead_host):
      ############### LOG EXPENSIVE TRAIN SUMMARY ###############
      # Visualizes detections using side-by-side gt-pred images.
      # TODO(mjlm): Investigate this error when including `batch_mask`:
      # RuntimeError: Invalid argument: from_python argument must be an array.
      to_cpu = lambda x: jax.device_get(dataset_utils.unshard(x))
      del train_batch['batch_mask']
      train_pred_cpu = to_cpu(train_predictions)
      train_batch_cpu = to_cpu(train_batch)
      viz = detr_train_utils.draw_boxes_side_by_side(
          train_pred_cpu,
          train_batch_cpu,
          label_map=dataset.meta_data['label_to_name'])
      viz_detections = {
          f'sidebyside_{i}/detection': viz_[None, ...]
          for i, viz_ in enumerate(viz)
      }
      writer.write_images(step, viz_detections)

    del train_predictions

    if (step % log_summary_steps == 0) or (step == total_steps - 1):
      ############### LOG TRAIN SUMMARY ###############

      # Write summary:
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=train_metrics,
          extra_training_logs=jax.tree_util.tree_map(
              train_utils.unreplicate_and_get, extra_training_logs),
          writer=writer,
          metrics_normalizer_fn=metrics_normalizer_fn)
      # Reset metric accumulation for next evaluation cycle.
      train_metrics, extra_training_logs = [], []
      #################################################

    if (step % log_eval_steps == 0) or (step == total_steps):
      # First wait for the previous eval to finish & write summary.
      if last_eval_future is not None:
        train_utils.log_eval_summary(
            step=last_eval_step,
            eval_metrics=last_eval_metrics,
            extra_eval_summary=last_eval_future.result(),
            writer=writer,
            metrics_normalizer_fn=metrics_normalizer_fn)
        last_eval_future = None

      # Sync model state across replicas (in case of having model state, e.g.
      # batch statistic when using batch norm).
      start_time = time.time()
      with report_progress.timed('eval'):
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        (last_eval_step, last_eval_metrics), last_eval_future = evaluate(
            train_state, step)
      duration = time.time() - start_time
      logging.info('Done with async evaluation: %.4f sec.', duration)
      writer.flush()

    ##################### CHECKPOINTING ############################
    if ((step % checkpoint_steps == 0 and step > 0) or
        (step == total_steps)) and config.checkpoint:
      with report_progress.timed('checkpoint'):
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        if lead_host:
          train_state.replace(accum_train_time=chrono.accum_train_time)
          train_utils.save_checkpoint(workdir, train_state)

    chrono.resume()  # Un-pause now.

  # Wait until computations are done before exiting.
  pool.shutdown()
  if last_eval_future is not None:
    train_utils.log_eval_summary(
        step=last_eval_step,
        eval_metrics=last_eval_metrics,
        extra_eval_summary=last_eval_future.result(),
        writer=writer,
        metrics_normalizer_fn=metrics_normalizer_fn)
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  return train_state, train_summary, eval_summary
