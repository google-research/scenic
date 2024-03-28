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
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Type, Union

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
from scenic.projects.objectvivit import optimizer_utils
from scenic.projects.objectvivit import train_utils as custom_train_utils
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
MetricFnEval = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                        Dict[str, Tuple[float, int]]]


def train_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    loss_fn: Any,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False,
    learn_token_score: bool = False,
    add_boxes: bool = False,
) -> Tuple[train_utils.TrainState, Dict[str, Any]]:
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
    learn_token_score: if enable learning token score. The network will have
      additional outputs of each scores for each token.
    add_boxes: if add boxes in shape batch x time x num_objs x 4

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
    batch = custom_train_utils.mixup_cutmix(
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
    outputs, new_model_state = flax_model.apply(
        variables,
        batch['inputs'],
        boxes=batch['bboxes'] if add_boxes else None,
        mutable=['batch_stats'],
        train=True,
        rngs={'dropout': dropout_rng},
        debug=debug)
    if learn_token_score:
      logits = outputs['logits']
      cls_loss, token_loss = loss_fn(
          logits, batch, variables['params'], aux_outputs=outputs)
      loss = cls_loss + token_loss
      return loss, (new_model_state, outputs, cls_loss, token_loss)
    else:
      loss = loss_fn(outputs, batch, variables['params'])
      return loss, (new_model_state, outputs, None, None)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  (_, (new_model_state, outputs, cls_loss, token_loss
       )), grad = compute_gradient_fn(train_state.params)

  if learn_token_score:
    logits = outputs['logits']
    metrics = metrics_fn(logits, batch)
    metrics['cls_loss'] = (cls_loss, 1.)
    metrics['token_loss'] = (token_loss, 1.)
    token_scores = outputs['token_scores']
    gt_scores = outputs['gt_scores']
    n_tokens = token_scores.shape[-1]
    k = int(config.model.object_config.get('keep_token_ratio', 0.5) * n_tokens)
    topk_val, _ = jax.lax.top_k(token_scores, k)
    pred_mask = token_scores > topk_val[:, -1:]
    gt_topk_val, _ = jax.lax.top_k(gt_scores, k)
    gt_mask = gt_scores > gt_topk_val[:, -1:]
    acc = (pred_mask == gt_mask).sum() / pred_mask.size
    metrics['token_acc'] = (acc, 1.)
  else:
    metrics = metrics_fn(outputs, batch)

  if add_boxes:
    metrics['boxes_num'] = ((batch['bboxes'] > 0).any(axis=3).sum() / (
        batch['bboxes'].shape[0] * batch['bboxes'].shape[1]), 1.)

  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')

  metrics['max_grad_norm_preclip'] = (
      custom_train_utils.compute_max_norm(grad), 1)
  if config.get('max_grad_norm', None) is not None:
    grad = clip_grads(grad, config.max_grad_norm)
    metrics['max_grad_norm_postclip'] = (
        custom_train_utils.compute_max_norm(grad), 1)

  # We no longer perform explicit weight decay here. This can be added
  # as an Optax gradient transformation if necessary. Or one can also use
  # AdamW instead of Adam.
  updates, new_opt_state = train_state.tx.update(
      grad, train_state.opt_state, train_state.params
  )  # pytype: disable=attribute-error
  new_params = optax.apply_updates(train_state.params, updates)

  # Log additional statistics. These are the L2 norms of the entire flattened
  # vector.
  metrics['l2_grads'] = (
      jnp.sqrt(sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grad)])),
      1,
  )
  metrics['l2_params'] = (
      jnp.sqrt(
          sum([jnp.vdot(p, p) for p in jax.tree_util.tree_leaves(new_params)])
      ),
      1,
  )
  metrics['l2_updates'] = (
      jnp.sqrt(
          sum([jnp.vdot(u, u) for u in jax.tree_util.tree_leaves(updates)])
      ),
      1,
  )

  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      params=new_params,
      model_state=new_model_state,
      rng=new_rng)
  return new_train_state, metrics


def eval_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: MetricFn,
    debug: Optional[bool] = False,
    learn_token_score: bool = False,
    add_boxes: bool = False,
):
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
    learn_token_score: if enable learning token score. The network will have
      additional outputs of each scores for each token.
    add_boxes: if add boxes in shape batch x time x num_objs x 4

  Returns:
    Calculated metrics and logits.
  """
  variables = {
      'params': train_state.params,
      **train_state.model_state
  }
  outputs = flax_model.apply(
      variables, batch['inputs'],
      boxes=batch['bboxes'] if add_boxes else None,
      rngs={'dropout': train_state.rng},
      train=False, mutable=False, debug=debug)
  if learn_token_score:
    logits = outputs['logits']
  else:
    logits = outputs
  metrics = metrics_fn(logits, batch)
  return metrics, logits


def test_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: MetricFnEval,
    n_clips: int = 2,
    return_logits_and_labels: bool = False,
    softmax_logits: bool = False,
    debug: bool = False,
    learn_token_score: bool = False,
    add_boxes: bool = False,
) -> Union[
    Dict[str, Tuple[float, int]],
    Tuple[Dict[str, Tuple[float, int]], jnp.ndarray, jnp.ndarray],
]:
  """Runs a single step of testing.

  For multi-crop testing, we assume that num_crops consecutive entries in the
  batch are from the same example. And we average the logits over these examples

  We assume that the batch contains different crops of the same original
  example. Therefore, we can average all the logits of it.
  This assumption is true when local_batch_size = num_local_devices

  Args:
    train_state: The state of training including the current
      global_step, model_state, rng, and optimizer, and other metadata.
    batch: Dictionary with keys 'inputs', 'labels', 'batch_mask'. We assume that
      all the inputs correspond to the same original example in the test set.
      The input shapes to this function are batch['inputs'] = [num_crops, t, h,
      w, c] batch['labels'] = [num_crops, num_classes] However, for
      classification, the labels for all the crops are the same.
      batch['batch_mask'] = [num_crops]
    flax_model: A Flax model.
    metrics_fn: Metrics function for the model.
    n_clips: The number of clips to process at a time by each device. Set
      due to memory constraints.
    return_logits_and_labels: Whether return logits of the model or not.
    softmax_logits: Whether to softmax-normalise the logits before
      averaging
    debug: Whether the debug mode is enabled during evaluation.
      `debug=True` enables model specific logging/storing some values using
      jax.host_callback.
    learn_token_score: if enable learning token score. The network will have
      additional outputs of each scores for each token.
    add_boxes: if add boxes in shape batch x time x num_objs x 4

  Returns:
    Calculated metrics [and optionally averaged logits that are of
    shape `[1, num_classes]`].
  """
  all_logits = jnp.zeros(batch['label'].shape[1])
  num_crops = batch['inputs'].shape[0]
  variables = {
      'params': train_state.params,
      **train_state.model_state
  }

  for idx in range(0, num_crops, n_clips):
    temp_input = batch['inputs'][idx:idx + n_clips]
    outputs = flax_model.apply(
        variables, temp_input,
        boxes=batch['bboxes'][idx:idx + n_clips] if add_boxes else None,
        rngs={'dropout': train_state.rng},
        train=False, mutable=False, debug=debug)
    if learn_token_score:
      logits = outputs['logits']
    else:
      logits = outputs
    if softmax_logits:
      logits = nn.softmax(logits, axis=-1)
    logits = jnp.sum(logits, axis=0)
    all_logits = all_logits + logits

  all_logits = all_logits / num_crops
  all_logits = jnp.expand_dims(all_logits, axis=0)
  batch['label'] = jnp.expand_dims(batch['label'][0], axis=0)
  batch['batch_mask'] = jnp.expand_dims(batch['batch_mask'][0], axis=0)
  metrics = metrics_fn(all_logits, batch)
  if return_logits_and_labels:
    return metrics, all_logits, batch['label']
  return metrics


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

  learn_token_score = config.get(
      'learn_token_configs', {}).get('enabled', False)

  attach_objects = config.get('attach_configs', {}).get('enabled', False)
  token_score_from_dataloader = config.get('attach_configs', {}).get(
      'token_score_from_dataloader', False)
  add_boxes = config.dataset_configs.get('object_configs', {}).get(
      'return_boxes', -1) > 0

  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  init_rng = {'params': init_rng, 'dropout': init_rng}
  input_spec = [
      (dataset.meta_data['input_shape'],
       dataset.meta_data.get('input_dtype', jnp.float32))]
  if add_boxes:
    input_spec.append(
        (config.dataset_configs.object_configs.boxes_shape, jnp.float32))

  if attach_objects and not token_score_from_dataloader:
    input_spec.append((config.attach_configs.out_shape, jnp.float32))
  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=input_spec,
       config=config,
       rngs=init_rng)

  # Create optimizer.
  lr_fn = lr_schedules.get_learning_rate_fn(config)
  if 'layerwise_decay' in config.optimizer_configs:
    logging.info('Using layerwise decay optimizer.')
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
  del train_state.metadata['chrono']  # pytype: disable=unsupported-operands
  if (start_step == 0  # Which means "no" checkpoint is restored!
      and config.get('init_from') is not None):
    restored_model_cfg = config.init_from.get('model_config')
    init_checkpoint_path = config.init_from.get('checkpoint_path')

    if init_checkpoint_path is not None:
      restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
          init_checkpoint_path, train_state, assert_exist=True)
      logging.info(
          'restored_train_state.params.keys %s',
          restored_train_state.params.keys())
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
          debug=config.debug_train,
          learn_token_score=learn_token_score,
          add_boxes=add_boxes),
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
          learn_token_score=learn_token_score,
          add_boxes=add_boxes,
          ),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1,),
  )

  if config.dataset_configs.get('do_multicrop_test'):
    log_test_steps = int(
        steps_per_epoch * config.dataset_configs.log_test_epochs)

    test_step_pmapped = jax.pmap(
        functools.partial(
            test_step,
            flax_model=model.flax_model,
            metrics_fn=model.get_metrics_fn('test'),
            n_clips=config.get('multicrop_clips_per_device', 2),
            debug=config.debug_eval,
            learn_token_score=learn_token_score,
            add_boxes=add_boxes),
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
    additional_summary = None
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
              extra_eval_summary=additional_summary,
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
  hooks = [report_progress]

  if start_step == 0:
    step0_log = {'num_trainable_params': num_trainable_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)

  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      train_state, t_metrics = train_step_pmapped(train_state, train_batch)
      # This will accumulate metrics in TPU memory up to the point that we log
      # them. This is no problem for small metrics but may be a problem for
      # large (e.g. segmentation) metrics. An alternative is to set
      # `log_summary_steps` to a small number, or to use
      # `train_utils.unreplicate_and_get` here instead of right before writing
      # summaries, but that means in each step, we have data transfer between
      # tpu and host, which might slow down the training.
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
      chrono.pause()
      if lead_host:
        chrono.tick(step, writer, custom_train_utils.log_note)
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(
              train_utils.unreplicate_and_get, train_metrics
          ),
          extra_training_logs=extra_training_logs,
          writer=writer,
      )
      # Reset metric accumulation for next evaluation cycle.
      train_metrics, extra_training_logs = [], []
      chrono.resume()

    ################### EVALUATION #######################
    if (step % log_eval_steps == 0) or (step == total_steps):
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
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        if lead_host:
          unrep_train_state = jax_utils.unreplicate(train_state)
          metadata = unrep_train_state.metadata
          metadata['chrono'] = chrono.save()
          unrep_train_state.replace(metadata=metadata)
          train_utils.save_checkpoint(workdir, unrep_train_state, max_to_keep=1)
          del unrep_train_state
        chrono.resume()

    ############# MULTICROP TESTING ############################
    if (config.dataset_configs.get('do_multicrop_test') and
        ((step % log_test_steps == 0) or step == total_steps)):

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
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  logging.info('Parameter summary after completing training.')
  debug_utils.log_param_shapes(train_state.params)
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary
