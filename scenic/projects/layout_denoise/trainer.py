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

"""Training script for Layout Denoise model."""

import collections
from concurrent import futures
import functools
import random
import time
from typing import Any, Dict, Tuple, Optional

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
import flax.linen as nn
from flax.training.checkpoints import restore_checkpoint as flax_restore_checkpoint
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.projects.baselines.detr import train_utils as detr_train_utils
from scenic.projects.layout_denoise import train_utils as layout_train_utils
from scenic.projects.layout_denoise.datasets import dataset
from scenic.train_lib_deprecated import lr_schedules
from scenic.train_lib_deprecated import pretrain_utils
from scenic.train_lib_deprecated import train_utils


def _pred_probs(predictions):
  """Adds the predicted probabilities to prediction dict."""
  # Compute class probabilities on device, needed for Hungarian matching.
  # Does not remove class logits as they are needed for stable loss computation
  # with cross entropy losses.
  predictions['pred_probs'] = nn.softmax(predictions['pred_logits'], axis=-1)
  for aux_preds in predictions.get('aux_outputs', []):
    aux_preds['pred_probs'] = nn.softmax(aux_preds['pred_logits'], axis=-1)


def get_train_step(flax_model,
                   loss_and_metrics_fn,
                   learning_rate_fn,
                   backbone_learning_rate_fn,
                   max_grad_norm=None,
                   update_model_state=False,
                   debug=False):
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Note that in this code, the buffers of the first (train_state) and second
  (batch) arguments are donated to the computation.

  Args:
    flax_model: Flax model (an instance of nn.Module).
    loss_and_metrics_fn: A function that given model predictions, a batch, and
      parameters of the model calculates the loss as well as metrics.
    learning_rate_fn: Learning rate scheduler which given the global_step
      generates the learning rate.
    backbone_learning_rate_fn: Learning rate scheduler which given the
      global_step generates the learning rate used for updating  the parameters
      of the backbone in the model.
    max_grad_norm: float; Maximum gradient norm used for gradient clipping. If
      set to None, no gradient clipping happens.
    update_model_state: bool; whether to update the model_state (e.g. batch
      stats in BatchNorm) during training or freeze it.
    debug: bool; Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Train step function that takes a train_state and batch and returns
    new_train_state, metrics, lr, predictions.

  """
  backbone_learning_rate_fn = backbone_learning_rate_fn or learning_rate_fn

  def update_fn(train_state, new_model_state, grad, new_rng):
    step = train_state.global_step
    backbone_lr = backbone_learning_rate_fn(step)
    lr = learning_rate_fn(step)

    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grad = jax.lax.pmean(grad, axis_name='batch')

    if max_grad_norm is not None:
      grad = detr_train_utils.clip_grads(grad, max_grad_norm)

    (backbone_opt_hps,
     opt_hps) = train_state.optimizer.optimizer_def.hyper_params
    new_optimizer = train_state.optimizer.apply_gradient(
        grad,
        hyper_params=[
            backbone_opt_hps.replace(learning_rate=backbone_lr),
            opt_hps.replace(learning_rate=lr)
        ])
    new_train_state = train_state.replace(
        global_step=step + 1,
        optimizer=new_optimizer,
        model_state=new_model_state,
        rng=new_rng)
    return new_train_state, lr, backbone_lr

  def train_step(train_state, batch, task):

    def loss_fn(params):
      new_rng, rng = jax.random.split(train_state.rng)
      # Bind the rng to the host/device we are on.
      model_rng = train_utils.bind_rng_to_host_device(
          rng, axis_name='batch', bind_to='device')
      variables = {'params': params, **train_state.model_state}
      predictions, new_model_state = flax_model.apply(
          variables,
          image=batch['inputs'],
          obj_mask=batch['label']['obj_mask'],
          desc_id=batch['label']['desc_id'],
          resource_id=batch['label']['resource_id'],
          name_id=batch['label']['name_id'],
          boxes=batch['label']['boxes'],
          task=task,
          padding_mask=batch['padding_mask'],
          update_batch_stats=update_model_state,
          mutable=['batch_stats'],
          train=True,
          rngs={'dropout': model_rng},
          debug=debug)
      if task == 'layout_denoise':
        _pred_probs(predictions)

      loss, metrics = loss_and_metrics_fn(
          predictions, batch, task=task, model_params=variables['params'])
      return loss, (new_model_state, metrics, predictions, new_rng)

    compute_gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, aux), grad = compute_gradient_fn(train_state.optimizer.target)
    new_model_state, metrics, predictions, new_rng = aux
    new_train_state, lr, backbone_lr = update_fn(train_state, new_model_state,
                                                 grad, new_rng)
    return new_train_state, metrics, lr, backbone_lr, predictions

  return train_step


def get_eval_step(flax_model,
                  loss_and_metrics_fn,
                  metrics_only=False,
                  debug=False):
  """Runs a single step of training.

  Note that in this code, the buffer of the second argument (batch) is donated
  to the computation.

  Args:
    flax_model: Flax model (an instance of nn.Module).
    loss_and_metrics_fn: A function that given model predictions, a batch, and
      parameters of the model calculates the loss as well as metrics.
    metrics_only: bool; Only return metrics.
    debug: bool; Whether the debug mode is enabled during evaluation.
      `debug=True` enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Eval step function which returns predictions and calculated metrics.
  """

  def metrics_fn(train_state, batch, predictions, task):
    _, metrics = loss_and_metrics_fn(
        predictions,
        batch,
        task=task,
        model_params=train_state.optimizer.target)

    if metrics_only:
      return None, None, metrics

    # Collect necessary predictions and target information from all hosts.
    if task == 'layout_denoise':
      predictions_out = {
          'pred_logits': predictions['pred_logits'],
      }
      if 'binary_pred_logits' in predictions:
        predictions_out['binary_pred_logits'] = predictions[
            'binary_pred_logits']

    targets = {
        'label': {
            'size': batch['label']['size'],
            'orig_size': batch['label']['orig_size'],
            'labels': batch['label']['labels'],
            'binary_labels': batch['label']['binary_labels'],
        },
        'batch_mask': batch['batch_mask']
    }

    predictions_out = jax.lax.all_gather(predictions_out, 'batch')
    targets = jax.lax.all_gather(targets, 'batch')
    return targets, predictions_out, metrics

  def eval_step(train_state, batch, task):
    variables = {
        'params': train_state.optimizer.target,
        **train_state.model_state
    }
    predictions = flax_model.apply(
        variables,
        image=batch['inputs'],
        obj_mask=batch['label']['obj_mask'],
        desc_id=batch['label']['desc_id'],
        resource_id=batch['label']['resource_id'],
        name_id=batch['label']['name_id'],
        boxes=batch['label']['boxes'],
        # clickable=batch['label']['clickable'],
        task=task,
        padding_mask=batch['padding_mask'],
        train=False,
        mutable=False,
        debug=debug)
    if task == 'layout_denoise':
      _pred_probs(predictions)

    return metrics_fn(train_state, batch, predictions, task=task)

  return eval_step


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Any,
    dataset_dict: Dict[str, dataset_utils.Dataset],
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
    dataset_dict: A dict of datasets that each has train_iter, eval_iter,
      meta_data, and optionally, test_iter.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    train_sate that has the state of training (including current
      global_step, rng, and the optimizer), train_summary
      and eval_summary which are dict of metrics. These outputs are used for
      regression testing.
  """
  lead_host = jax.process_index() == 0
  datasets_metadata = {name: ds.meta_data for name, ds in dataset_dict.items()}
  # The pool is used to perform misc operations such as logging in async way.
  pool = futures.ThreadPoolExecutor(max_workers=2)

  # Set up lr configs based on the tasks/datasets:
  # config = layout_train_utils.set_lr_configs(config, datasets_metadata)

  # Build the loss_and_metrics_fn, metrics, and flax_model.
  model = model_cls(config, datasets_metadata)

  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  input_spec = {}
  for ds_name, metadata in datasets_metadata.items():
    input_spec[metadata['task_name']] = [
        (metadata['input_shape'], metadata.get('input_dtype', jnp.float32)),
        (metadata['obj_mask_shape'], metadata.get('obj_mask_dtype', jnp.int32)),
        (metadata['obj_desc_id_shape'],
         metadata.get('obj_desc_id_dtype', jnp.int32)),
        (metadata['obj_resource_id_shape'],
         metadata.get('obj_resource_id_dtype', jnp.int32)),
        (metadata['obj_name_id_shape'],
         metadata.get('obj_name_id_dtype', jnp.int32)),
        (metadata['obj_bbx_shape'], metadata.get('obj_bbx_dtype', jnp.float32)),
        # (metadata['obj_clickable_shape'],
        #  metadata.get('obj_clickable_dtype', jnp.int32)),
    ]

  (params, model_state, num_trainable_params,
   gflops) = layout_train_utils.initialize_multitask_model(
       model_def=model.flax_model,
       input_spec=input_spec,
       config=config,
       rngs=init_rng)

  # Create optimizer.
  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0]
  optimizer = jax.jit(
      detr_train_utils.get_detr_optimizer(config).create, backend='cpu')(
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
    # only load pretrained backbone if we are at the beginning of training
    bb_checkpoint_path = config.pretrained_backbone_configs.get(
        'checkpoint_path')
    bb_train_state = flax_restore_checkpoint(bb_checkpoint_path, target=None)

    if config.get('load_pretrained_vut', False):
      model_prefix_path = None
      skip_regex = config.get('param_skip_regex', None)
    else:
      model_prefix_path = ['backbone']
      skip_regex = None
    train_state = pretrain_utils.init_from_pretrain_state(
        train_state,
        bb_train_state,
        model_prefix_path=model_prefix_path,
        skip_regex=skip_regex)

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
  total_steps, steps_per_epoch = layout_train_utils.get_num_training_steps(
      config, datasets_metadata)
  total_steps = config.get('total_steps')

  # Get learning rate scheduler.
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)
  backbone_learning_rate_fn = None
  if config.get('backbone_training'):
    backbone_learning_rate_fn = lr_schedules.get_learning_rate_fn(
        config.backbone_training)

  train_step = get_train_step(
      flax_model=model.flax_model,
      loss_and_metrics_fn=model.loss_function,
      learning_rate_fn=learning_rate_fn,
      backbone_learning_rate_fn=backbone_learning_rate_fn,
      max_grad_norm=config.get('max_grad_norm', None),
      update_model_state=update_model_state,
      debug=config.debug_train)
  train_step_pmapped = jax.pmap(
      train_step,
      axis_name='batch',
      static_broadcasted_argnums=(2,),
      donate_argnums=(0, 1),
  )

  ############### EVALUATION CODE #################
  eval_step = get_eval_step(
      flax_model=model.flax_model,
      loss_and_metrics_fn=model.loss_function,
      debug=config.debug_eval)
  eval_step_pmapped = jax.pmap(
      eval_step,
      axis_name='batch',
      static_broadcasted_argnums=(2,),
      donate_argnums=(1,))

  steps_per_eval = {}
  for ds_name in datasets_metadata.keys():
    # Ceil rounding such that we include the last incomplete batch.
    eval_batch_size = config.get('eval_batch_size', config.batch_size)
    total_eval_steps = int(
        np.ceil(datasets_metadata[ds_name]['num_eval_examples'] /
                eval_batch_size))
    steps_per_eval[ds_name] = config.get('steps_per_eval') or total_eval_steps

  metrics_normalizer_fn = functools.partial(
      layout_train_utils.normalize_metrics_summary,
      object_detection_loss_keys=model.loss_terms_weights.keys())

  def evaluate(global_evaluator,
               train_state,
               ds_name,
               ds_task,
               step,
               eval_synchronously=False):
    """Runs evaluation code."""
    future = None

    def _wait(future: Optional[futures.Future]) -> Any:  # pylint: disable=g-bare-generic
      if future is None:
        return None
      return future.result()

    def _add_examples(predictions, labels, task):
      for pred, label in zip(predictions, labels):
        if task == 'layout_denoise':
          global_evaluator.add_example(
              prediction=pred, target=label, task_name=task, ds_name=ds_name)

    eval_metrics = []
    if global_evaluator is not None:
      global_evaluator.clear()

    for eval_step in range(steps_per_eval[ds_name]):
      task = dataset_dict[ds_name].meta_data['task_name']

      logging.info('Running eval step %d on %s-%s batch', eval_step, task,
                   ds_name)
      eval_batch = next(dataset_dict[ds_name].valid_iter)

      # Do the eval step given the matches.
      (eval_batch_all_hosts, eval_predictions_all_hosts,
       e_metrics) = eval_step_pmapped(train_state, eval_batch, task)

      # Add task/dataset name prefix to eval metrics.
      e_metrics = {f'{task}-{ds_name}/{k}': m for k, m in e_metrics.items()}

      # Variable aux_outputs is not needed anymore.
      eval_predictions_all_hosts.pop('aux_outputs', None)
      eval_batch_all_hosts['label'].pop('masks', None)

      # Collect local metrics (returned by the loss function).
      eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))

      if global_evaluator is not None:
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

        # Decompress masks.
        for pred in predictions:
          if 'pred_masks_compressed' in pred:
            pred['pred_masks'] = model.decompress_masks(
                [x[np.newaxis, ...] for x in pred['pred_masks_compressed']],
                config.num_queries)[0]
        for label in labels:
          if 'padding_mask_compressed' in label:
            label['padding_mask'] = np.logical_and(
                *label['padding_mask_compressed'])

        # Add to evaluator.
        _wait(future)
        if eval_synchronously:
          _add_examples(predictions, labels, task)
        else:
          future = pool.submit(_add_examples, predictions, labels, task)

        del predictions, labels

      del eval_batch, eval_batch_all_hosts, eval_predictions_all_hosts

    eval_summary_future = None
    eval_summary = None
    if global_evaluator is not None:
      _wait(future)
      if ds_task == 'layout':
        num_eval_examples = global_evaluator.get_num_examples_added()
        for (eval_task, eval_ds), cnt in num_eval_examples.items():
          logging.info('Number of %s-%s eval examples: %d', eval_task, eval_ds,
                       cnt)
      if lead_host:
        if eval_synchronously:
          eval_summary = global_evaluator.compute_metrics()
        else:
          eval_summary_future = pool.submit(global_evaluator.compute_metrics)

    ############### LOG EVAL SUMMARY ###############
    def log_fn_async(step, eval_metrics, future_eval_summary, writer,
                     metrics_normalizer_fn):
      eval_summary = _wait(future_eval_summary)

      # Merge dicts.
      all_scores = {}
      if eval_summary:
        all_scores.update(eval_summary)

      return train_utils.log_eval_summary(
          step=step,
          eval_metrics=eval_metrics,
          extra_eval_summary=all_scores,
          writer=writer,
          prefix='valid',
          metrics_normalizer_fn=metrics_normalizer_fn)

    def log_fn_sync(step, eval_metrics, eval_summary, writer,
                    metrics_normalizer_fn):

      # Merge dicts.
      all_scores = {}
      if eval_summary:
        all_scores.update(eval_summary)

      return train_utils.log_eval_summary(
          step=step,
          eval_metrics=eval_metrics,
          extra_eval_summary=all_scores,
          writer=writer,
          prefix='valid',
          metrics_normalizer_fn=metrics_normalizer_fn)

    # Note that we return a Future on a summary instead of the summary itself!
    if eval_synchronously:
      log_fn_sync(
          step=step,
          eval_metrics=eval_metrics,
          eval_summary=eval_summary,
          writer=writer,
          metrics_normalizer_fn=metrics_normalizer_fn)
    else:
      return pool.submit(
          log_fn_async,
          step=step,
          eval_metrics=eval_metrics,
          future_eval_summary=eval_summary_future,
          writer=writer,
          metrics_normalizer_fn=metrics_normalizer_fn)

  ###################################################

  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  log_summary_steps = config.get('log_summary_steps', 25)
  log_large_summary_steps = config.get('log_large_summary_steps', 0)
  checkpoint_steps = config.get('checkpoint_steps') or steps_per_epoch

  # These evaluators only run on the lead_host node.
  clean_metrics_evaluator = None

  if lead_host:
    metadata = list(datasets_metadata.values())
    tasks = set()
    for meta in metadata:
      tasks.add(meta['task_name'])

    if 'layout_denoise' in tasks:
      clean_metrics_evaluator = layout_train_utils.LayoutDenoiseGlobalEvaluator(
          config.get('num_classes', 26))

  train_metrics = collections.defaultdict(list)
  extra_training_logs = collections.defaultdict(list)
  train_summary, eval_summaries = None, None

  chrono = train_utils.Chrono(
      first_step=start_step,
      total_steps=total_steps,
      steps_per_epoch=steps_per_epoch,
      global_bs=config.batch_size,
      accum_train_time=int(jax_utils.unreplicate(train_state.accum_train_time)))

  all_datasets = config.get('dataset_names')

  def get_dataset_weights(crt_step):
    stages = config.get('dataset_weight_stages')
    stage_idx = 0
    for idx, stage_end in enumerate(stages):
      if stage_end > crt_step:
        stage_idx = idx
        break
    return config.get('dataset_weights')[stage_idx]

  def get_next_train_batch(step):
    strategy = config.get('multi_task_training_strategy', 'sampling')
    if strategy == 'sampling':
      dataset_name = random.Random(step).choices(all_datasets,
                                                 get_dataset_weights(step))[0]
    elif strategy == 'sequential':
      idx = step % len(all_datasets)
      dataset_name = all_datasets[idx]
    else:
      raise ValueError('Sampling strategy: %s' % strategy)
    ds = dataset_dict[dataset_name]
    return next(ds.train_iter), ds.meta_data['task_name'], dataset_name

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
    if writer:
      writer.write_scalars(1, step0_log)

  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch, train_task, train_ds = get_next_train_batch(step)
      (train_state, t_metrics, lr, backbone_lr,
       train_predictions) = train_step_pmapped(train_state, train_batch,
                                               train_task)
      # This will accumulate metrics in TPU memory up to the point that we log
      # them. This is no problem for small metrics but may be a problem for
      # large (e.g. segmentation) metrics. An alternative is to set
      # `log_summary_steps` to a small number, or to use
      # `train_utils.unreplicate_and_get` here instead of right before writing
      # summaries, but that means in each step, we have data transfer between
      # tpu and host, which might slow down the training.
      train_metrics[(train_task, train_ds)].append(t_metrics)

      # Additional training logs: learning rate, learning_rate_backbone:
      extra_training_logs[(train_task, train_ds)].append({
          'learning_rate': lr,
          'learning_rate_backbone': backbone_lr,
      })

    for h in hooks:
      h(step)

    chrono.pause()
    if (log_large_summary_steps and step % log_large_summary_steps == 0 and
        lead_host):
      ############### LOG EXPENSIVE TRAIN SUMMARY ###############
      # Visualizes detections using side-by-side gt-pred images.
      # TODO(mjlm): Investigate this error when including `batch_mask`:
      # RuntimeError: Invalid argument: from_python argument must be an array.
      to_cpu = lambda x: jax.device_get(dataset_utils.unshard(x))
      del train_batch['batch_mask']
      if ('pred_logits' in train_predictions and
          'pred_boxes' in train_predictions):
        # Only draw side-by-side when bbox/type predictions are present.
        train_pred_cpu = to_cpu(train_predictions)
        train_batch_cpu = to_cpu(train_batch)
        viz = detr_train_utils.draw_boxes_side_by_side(
            train_pred_cpu,
            train_batch_cpu,
            label_map=dataset.get_ui_label_map(config.num_classes))

        viz_detections = {
            f'sidebyside_{i}/detection': viz_[None, ...]
            for i, viz_ in enumerate(viz)
        }
        writer.write_images(step, viz_detections)

    del train_predictions

    if (step % log_summary_steps == 0) or (step == total_steps):
      ############### LOG TRAIN SUMMARY ###############
      if lead_host:
        chrono.tick(step, writer)
      train_summary = {}
      for train_task, train_ds in train_metrics.keys():
        train_metrics_cpu = jax.tree_util.tree_map(
            train_utils.unreplicate_and_get,
            train_metrics[(train_task, train_ds)])
        extra_training_logs_cpu = jax.tree_util.tree_map(
            train_utils.unreplicate_and_get,
            extra_training_logs[(train_task, train_ds)])
        train_summary.update(
            train_utils.log_train_summary(
                step=step,
                train_metrics=train_metrics_cpu,
                extra_training_logs=extra_training_logs_cpu,
                writer=writer,
                metrics_normalizer_fn=metrics_normalizer_fn,
                prefix=f'train-{train_task}-{train_ds}',
                key_separator='/'))
      # Reset metric accumulation for next evaluation cycle.
      train_metrics = collections.defaultdict(list)
      extra_training_logs = collections.defaultdict(list)
      #################################################

    if (step % log_eval_steps == 0) or (step == total_steps):
      # Sync model state across replicas (in case of having model state, e.g.
      # batch statistic when using batch norm).
      with report_progress.timed('eval'):
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        eval_summaries = []
        for ds_name in config.get('eval_dataset_names', all_datasets):
          ds_task = dataset_dict[ds_name].meta_data['task_name']
          start_time = time.time()
          if ds_task == 'layout_denoise':
            global_evaluator = clean_metrics_evaluator
          else:
            global_evaluator = None
          eval_summary = evaluate(global_evaluator, train_state, ds_name,
                                  ds_task, step,
                                  config.get('eval_synchronously', False))
          eval_summaries.append(eval_summary)
          duration = time.time() - start_time
          try:
            ex = eval_summary.exception(1) if eval_summary else None
            if ex is not None:
              logging.error('Failed with %s evaluation: %.4f sec.', ds_name,
                            duration)
              raise ex  # pylint: disable=raising-bad-type
            logging.info('Done with %s evaluation: %.4f sec.', ds_name,
                         duration)
          except futures.TimeoutError:
            pass

        if writer:
          writer.flush()
        if step != total_steps:
          del eval_summaries  # Free up space.

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

  # Aggregate eval summaries.
  eval_dict = {}
  if eval_summaries:
    for eval_summary in eval_summaries:
      eval_dict.update(eval_summary.result())

  pool.shutdown()
  # Wait until computations are done before exiting.
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  return train_state, train_summary, eval_dict
