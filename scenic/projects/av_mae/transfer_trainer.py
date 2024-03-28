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

import copy
import functools
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Type

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
import optax
from scenic.common_lib import debug_utils
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import base_model
from scenic.projects.av_mae import optimizer_utils
from scenic.projects.av_mae import train_utils as avmae_train_utils
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils
from scenic.train_lib.transfer import fewshot_utils

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
    train_state: The state of training including the current global_step,
      model_state, rng, and optimizer. The buffer of this argument can be
      donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
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
    mixup_rng, rng = jax.random.split(rng, 2)
    mixup_rng = train_utils.bind_rng_to_host_device(
        mixup_rng,
        axis_name='batch',
        bind_to=config.mixup.get('bind_to', 'device'))
    batch = avmae_train_utils.mixup_cutmix(
        batch,
        mixup_rng,
        config.mixup.alpha,
        cutmix_alpha=config.mixup.get('cutmix_alpha', 0.),
        switch_prob=config.mixup.get('cutmix_switch_prob', 0.5),
        label_smoothing=config.mixup.get('label_smoothing', 0.0))

  # Bind the dropout rng to the host/device we are on.
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
  (_, (new_model_state,
       logits)), grad = compute_gradient_fn(train_state.params)

  metrics = metrics_fn(logits, batch)

  if not config.get('grad_clip_after_pmean', True):
    metrics['max_grad_norm_preclip_before_pmean'] = (
        avmae_train_utils.compute_max_norm(grad), 1)
    if config.get('max_grad_norm', None) is not None:
      grad = clip_grads(grad, config.max_grad_norm)
      metrics['max_grad_norm_postclip_before_pmean'] = (
          avmae_train_utils.compute_max_norm(grad), 1)

  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')

  if config.get('grad_clip_after_pmean', True):
    metrics['max_grad_norm_preclip_after_pmean'] = (
        avmae_train_utils.compute_max_norm(grad), 1)
    if config.get('max_grad_norm', None) is not None:
      grad = clip_grads(grad, config.max_grad_norm)
      metrics['max_grad_norm_postclip_after_pmean'] = (
          avmae_train_utils.compute_max_norm(grad), 1)

  # We no longer perform explicit weight decay here. This can be added
  # as an Optax gradient transformation if necessary. Or one can also use
  # AdamW instead of Adam.
  updates, new_opt_state = train_state.tx.update(grad, train_state.opt_state,  # pytype: disable=attribute-error
                                                 train_state.params)
  new_params = optax.apply_updates(train_state.params, updates)

  # Log additional statistics. These are the L2 norms of the entire flattened
  # vector.
  metrics['l2_grads'] = (jnp.sqrt(
      sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grad)])), 1)
  metrics['l2_params'] = (jnp.sqrt(
      sum([jnp.vdot(p, p) for p in jax.tree_util.tree_leaves(new_params)])), 1)
  metrics['l2_updates'] = (jnp.sqrt(
      sum([jnp.vdot(u, u) for u in jax.tree_util.tree_leaves(updates)])), 1)

  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      params=new_params,
      model_state=new_model_state,
      rng=new_rng)
  return new_train_state, metrics  # pytype: disable=bad-return-type  # jax-types


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
    debug: Whether the debug mode is enabled during evaluation. `debug=True`
      enables model specific logging/storing some values using
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


def representation_fn(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    representation_layer: str,
    gather_to_host: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Feeds the inputs to the model and returns their representations.

  Args:
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data from the dataset.
    flax_model: A Flax model.
    representation_layer: The name of the layer to use as the representation.
    gather_to_host: Whether to gather results from all devices to the host,
      rather than leaving them distributed.

  Returns:
    Representation learned by the model for the given inputs and the labels and
    masks. If `gather_to_host` is True, these are collected from all hosts.
  """
  variables = {
      'params': train_state.params,
      **train_state.model_state
  }

  representation_layer_parts = representation_layer.split('/')
  filter_rep = lambda mdl, _: mdl.name == representation_layer_parts[-1]
  _, model_state = flax_model.apply(
      variables,
      batch['inputs'],
      train=False,
      capture_intermediates=filter_rep,
      mutable=['intermediates'],
      debug=False)
  if 'intermediates' not in model_state:
    raise ValueError(f'Layer with name "{representation_layer}"'
                     ' does not exist in your model.')

  representation = model_state['intermediates']
  for rep_layer in representation_layer_parts:
    if rep_layer:
      representation = representation[rep_layer]
  representation = representation['__call__'][0]
  if gather_to_host:
    representation = jax.lax.all_gather(representation, 'batch')
    batch = jax.lax.all_gather(batch, 'batch')
  return representation, batch['label'], batch['batch_mask']


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Type[base_model.BaseModel],
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter):
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
    train_sate that has the state of training (including current global_step,
    model_state, rng, and the optimizer), train_summary and eval_summary which
    are dict of metrics (from the last evaluation and train metric logging
    respectively). These outputs are used for regression testing.
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
  if 'layerwise_decay' in config.optimizer_configs:
    tx = optimizer_utils.optimizer_with_layerwise_decay(config, params)
  else:
    optimizer_config = optimizers.get_optax_optimizer_config(config)
    tx = optimizers.get_optimizer(optimizer_config, lr_fn, params)
  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  opt_state = jax.jit(tx.init, backend='cpu')(params)

  # Create Chrono ojbect to track and store training statistics and metadata.
  chrono = train_utils.Chrono()

  rng, train_rng = jax.random.split(rng)
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
    logging.info('Parameter summary after restoring checkpoint')
    debug_utils.log_param_shapes(train_state.params)

  chrono.load(train_state.metadata['chrono'])
  train_state = train_state.replace(metadata={})
  if (start_step == 0  # Which means "no" checkpoint is restored!
      and config.get('init_from') is not None):
    restored_model_cfg = config.init_from.get('model_config')
    init_checkpoint_path = config.init_from.get('checkpoint_path')

    if init_checkpoint_path is not None:
      checkpoint_format = config.init_from.get('checkpoint_format', 'scenic')
      if checkpoint_format == 'scenic':
        restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
            init_checkpoint_path, train_state, assert_exist=True)
      elif checkpoint_format == 'big_vision':
        restored_train_state = (
            pretrain_utils.convert_big_vision_to_scenic_checkpoint(  # pylint: disable=g-line-too-long
                init_checkpoint_path, train_state
            )
        )
        # Config dict in big_vision is not the same format as scenic.
        # Therefore, make sure config match the config of the loaded model!
        restored_model_cfg = copy.deepcopy(config)
        # The following is needed when the restored and target models used a
        # different classifier. As big_vision uses a different config dict, we
        # have to specify this manually.
        restored_model_cfg.model.classifier = config.init_from.get(
            'classifier_type', 'token')

      train_state = model.init_from_train_state(train_state,  # pytype: disable=attribute-error
                                                restored_train_state,
                                                restored_model_cfg)
      # Free unnecessary memory.
      del restored_train_state
      logging.info('Parameters after initialising weights from checkpoint.')
      debug_utils.log_param_shapes(train_state.params)

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

  if 'fewshot' in config:
    repr_fn = functools.partial(
        representation_fn,
        flax_model=model.flax_model,
        representation_layer=config.fewshot.representation_layer)

    if config.model_name.startswith('mvit'):
      is_2d_model = len(config.model.patch_size) == 2
    else:
      is_2d_model = len(config.model.patches.size) == 2

    if is_2d_model:
      fewshotter = fewshot_utils.FewShotEvaluator(repr_fn, config.fewshot)
    else:
      fewshotter = fewshot_utils.FewShotEvaluatorVideo(repr_fn, config.fewshot)

  if config.get('dataset_configs', dict()).get('do_multicrop_test', False):
    log_test_steps = int(
        steps_per_epoch * config.dataset_configs.log_test_epochs)

    test_step_pmapped = jax.pmap(
        functools.partial(
            avmae_train_utils.test_step,
            flax_model=model.flax_model,
            metrics_fn=model.get_metrics_fn('test'),
            n_clips=config.get('multicrop_clips_per_device', 2),
            debug=config.debug_eval),
        axis_name='batch',
        # We can donate the test_batch's buffer.
        donate_argnums=(1,),
    )

    if config.dataset_configs.test_batch_size != jax.local_device_count():
      raise ValueError(
          'The per-host batch size must be equal to the number of local devices'
          'This ensures that each TPU device is processing different views of'
          'the same original video. Got '
          f'{config.dataset_configs.test_batch_size} vs'
          f'{jax.local_device_count()}.')

    total_test_steps = int(
        np.ceil(dataset.meta_data['num_test_examples'] /
                (config.get('dataset_configs.test_batch_size') *
                 config.get('dataset_configs.num_test_clips') *
                 jax.process_count())))
    steps_per_test = config.get('steps_per_test') or total_test_steps

  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps

  def evaluate(train_state: train_utils.TrainState, step: int,
               valid_iter: Iterator[Batch],
               num_valid_ex: int) -> Dict[str, Any]:
    eval_summary = {}
    if not isinstance(valid_iter, dict):  # Only on validation set.
      valid_iter, num_valid_ex = {'valid': valid_iter}, {'valid': num_valid_ex}

    for val_name, val_iter in valid_iter.items():
      num_ex = num_valid_ex[val_name]
      # Ceil rounding such that we include the last incomplete batch.
      eval_batch_size = config.get('eval_batch_size', config.batch_size)
      total_eval_steps = int(np.ceil(num_ex / eval_batch_size))
      steps_per_eval = config.get('steps_per_eval') or total_eval_steps
      eval_metrics = []
      for _ in range(steps_per_eval):
        eval_batch = next(val_iter)
        if dataset.meta_data['target_is_onehot']:  # Which includes multi-hot.
          # Ignore the entries with all zero label for evaluation.
          eval_batch['batch_mask'] *= eval_batch['label'].max(axis=-1)
        e_metrics, _ = eval_step_pmapped(train_state, eval_batch)
        eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))
      eval_summary.update(
          train_utils.log_eval_summary(
              step=step,
              eval_metrics=eval_metrics,
              writer=writer,
              prefix=val_name))
    del eval_metrics
    writer.flush()
    return eval_summary

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None
  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)
  logging.info('Starting training loop at step %d.', start_step)

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
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      train_state, t_metrics = train_step_pmapped(train_state, train_batch)
      train_metrics.append(t_metrics)
      # Additional training logs: learning rate:
      extra_training_logs.append({'learning_rate': lr_fn(step)})

    # Quick indication that training is happening.
    logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
    for h in hooks:
      h(step)

    ############### LOG TRAIN SUMMARY ###############
    if ((step % log_summary_steps == 1) or (step == total_steps) or
        (chrono.warmup and lead_host)):
      chrono.pause(wait_for=(train_metrics))
      if lead_host:
        chrono.tick(step, writer, avmae_train_utils.log_note)
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(train_utils.unreplicate_and_get,
                                               train_metrics),
          extra_training_logs=extra_training_logs,
          writer=writer)
      # Reset metric accumulation for next evaluation cycle.
      train_metrics, extra_training_logs = [], []
      chrono.resume()

    ################### EVALUATION #######################
    if (step % log_eval_steps == 1) or (step == total_steps):
      chrono.pause(wait_for=(train_state.params))
      with report_progress.timed('eval'):
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        eval_summary = evaluate(train_state, step, dataset.valid_iter,
                                dataset.meta_data['num_eval_examples'])
      chrono.resume()

    ##################### CHECKPOINTING ############################
    if ((step % checkpoint_steps == 1 and step > 1) or
        (step == total_steps)) and config.checkpoint:
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))
      with report_progress.timed('checkpoint'):
        train_utils.handle_checkpointing(train_state, chrono, workdir)
      chrono.resume()

    ##################### FEWSHOT EVALUATION ############################
    if 'fewshot' in config:
      # Compute few-shot on-the-fly evaluation.
      if (step % config.fewshot.log_eval_steps == 1) or (step == total_steps):
        chrono.pause(wait_for=(train_state.params))
        with report_progress.timed('fewshot'):
          results = fewshotter.run_all(train_state, config.fewshot.datasets)
          fewshotter.log_fewshot_summary(
              writer=writer, step=step, results=results)
          del results
          writer.write_scalars(step, {'zz/epoch': step / steps_per_epoch})
        writer.flush()
        chrono.resume()

    ############# MULTICROP TESTING ############################
    if (config.get('dataset_configs', dict()).get('do_multicrop_test') and
        ((step % log_test_steps == 1) or step == total_steps)):

      chrono.pause(wait_for=(train_state.params))
      with report_progress.timed('checkpoint'):
        test_metrics = []
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)

        # At the end of training, evaluate on the whole test set.
        if step == total_steps:
          steps_per_test = total_test_steps

        logging.info('Starting multicrop test')
        for _ in range(steps_per_test):
          test_batch = next(dataset.test_iter)
          t_metrics = test_step_pmapped(train_state, test_batch)
          # Fetch t_metrics to host and store.
          test_metrics.append(train_utils.unreplicate_and_get(t_metrics))
        # Log eval summary.
        train_utils.log_eval_summary(
            step=step,
            eval_metrics=test_metrics,
            writer=writer,
            prefix='test')
        logging.info('Completed multicrop test')
        del test_metrics
        writer.flush()
        chrono.resume()

  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  logging.info('Parameter summary after completing training.')
  debug_utils.log_param_shapes(train_state.params)
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary
