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
from scenic.projects.av_mae import evaluation_lib
from scenic.projects.av_mae import optimizer_utils
from scenic.projects.av_mae import train_utils as avmae_train_utils
from scenic.projects.av_mae import trainer_multimodal
from scenic.projects.vivit import train_utils as vivit_train_utils
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
    image_format = config.mixup.get('image_format', 'NTHWC')
    batch_first = True
    if image_format.index('N') > 0:
      batch_first = False
    batch = avmae_train_utils.mixup_modalities(
        batch,
        config.mixup.alpha,
        batch_first,
        mixmod=config.get('mixmod', False),
        rng=mixup_rng)
  else:
    # No mixup is applied, all modalities share the same labels.
    if config.get('labels_as_dict', True):
      labels = batch['label']
      batch['label'] = {}  # pytype: disable=container-type-mismatch  # jax-ndarray
      for modality in batch['inputs']:
        batch['label'][modality] = labels
      batch['label']['all'] = labels

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

  if isinstance(logits, dict):
    # Necessary for MBT.
    # We use the first retrieved logits to report training metrics.
    modality = list(logits.keys())[0]
    batch['label'] = batch['label'][modality]
    metrics = metrics_fn(logits[modality], batch)
  else:
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
    return_logits_and_labels: bool = False,
    debug: Optional[bool] = False,
) -> Union[
    Tuple[Dict[str, Tuple[float, int]], jnp.ndarray, jnp.ndarray],
    Tuple[Dict[str, Tuple[float, int]], jnp.ndarray],
]:
  """Runs a single step of validation.

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
    return_logits_and_labels: If true, returns logits and labels. Can be used
      for calculating mean Average Precision for multi-label problems.
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
  # Here we have validation metrics computed on a single batch of data but
  # mAP is computed over the entire eval set. So we
  # 1) Gather & return logits and labels from all hosts for the sharded
  # global batch in this eval_step,
  # 2) Repeat for N global batches (eval_steps) needed to traverse the eval set,
  # 3) Once we gathered logits and labels for the entire eval set, compute mAP.
  if return_logits_and_labels:
    logits = jax.lax.all_gather(logits, 'batch')
    labels = jax.lax.all_gather(batch['label'], 'batch')
    return metrics, logits, labels
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

  is_multilabel_model = config.model_name in {
      'vit_multilabel_classification_mae', 'vit_multilabel_classification',
      'vivit_multimodal_multilabel_classification',
      'mbt_multilabel_classification'
  }
  logging.info('is_multilabel_model: %s', is_multilabel_model)

  # Initialize model.
  rng, params_init_rng, dropout_init_rng = jax.random.split(rng, num=3)
  init_rngs = {'params': params_init_rng, 'dropout': dropout_init_rng}
  input_spec_dict = {}
  for key in config.dataset_configs.modalities:
    if isinstance(dataset.meta_data['input_dtype'], dict):
      dtype = dataset.meta_data['input_dtype'][key]
    else:
      dtype = dataset.meta_data['input_dtype']
    input_spec = (dataset.meta_data['input_shape'][key], dtype)
    input_spec_dict[key] = input_spec

  (params, model_state, num_trainable_params,
   gflops) = trainer_multimodal.initialize_model(
       model_def=model.flax_model,
       input_spec_dict=input_spec_dict,
       config=config,
       rngs=init_rngs,
       is_train=True)

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
          return_logits_and_labels=is_multilabel_model,
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

  if config.dataset_configs.get('do_multicrop_test'):
    log_test_steps = int(
        steps_per_epoch * config.dataset_configs.log_test_epochs)

    test_step_pmapped = jax.pmap(
        functools.partial(
            avmae_train_utils.test_step_multimodal,
            flax_model=model.flax_model,
            metrics_fn=model.get_metrics_fn('test'),
            n_clips=config.get('multicrop_clips_per_device', 2),
            return_logits_and_labels=is_multilabel_model,
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
               num_valid_ex: int,
               compute_map: bool = False) -> Dict[str, Any]:
    """Perform validation and log results, possibly including mAP.
    """

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
      additional_summary = None
      if compute_map:
        eval_logits = []
        eval_labels = []
        n_classes = dataset.meta_data['num_classes']

      for _ in range(steps_per_eval):
        eval_batch = next(val_iter)
        if dataset.meta_data['target_is_onehot']:  # Which includes multi-hot.
          # Ignore the entries with all zero label for evaluation.
          eval_batch['batch_mask'] *= eval_batch['label'].max(axis=-1)

        # Compute validation metrics.
        if not compute_map:
          # only keep metrics
          e_metrics, _ = eval_step_pmapped(train_state, eval_batch)
        else:
          # return metrics, logits, labels
          e_metrics = eval_step_pmapped(train_state, eval_batch)
          e_metrics, logits_batch, labels_batch = e_metrics
          # Outcome of jax.lax.all_gather: all logits & labels from all hosts
          # for eval_batch in current evaluation step.
          # shape: (cores_per_host, n_devices, batch_size per device, n_classes)

          # Return a single instance of a replicated array, reshape to one
          # global batch, and transfer to host, where all global batches will be
          # concatenated for mAP computation.
          logits_batch_in_cpu = vivit_train_utils.to_cpu(logits_batch)
          labels_batch_in_cpu = vivit_train_utils.to_cpu(labels_batch)
          eval_logits.append(logits_batch_in_cpu)
          eval_labels.append(labels_batch_in_cpu)
        eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))

      if compute_map:
        # Once traversed the entire validation set, compute mAP.
        additional_summary = evaluation_lib.compute_mean_avg_precision_dprime(
            np.concatenate(eval_logits, axis=0),
            np.concatenate(eval_labels, axis=0),
            return_per_class_ap=n_classes < 10)

      eval_summary.update(
          train_utils.log_eval_summary(
              step=step,
              eval_metrics=eval_metrics,
              extra_eval_summary=additional_summary,
              writer=writer,
              key_separator='/',
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
        logging.info('Starting validation')
        eval_summary = evaluate(train_state, step, dataset.valid_iter,
                                dataset.meta_data['num_eval_examples'],
                                is_multilabel_model)
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
    if (config.dataset_configs.get('do_multicrop_test') and
        ((step % log_test_steps == 1) or step == total_steps)):

      chrono.pause(wait_for=(train_state.params))
      with report_progress.timed('test'):
        test_metrics = []
        additional_summary = None
        if is_multilabel_model:
          all_test_logits, all_test_labels = [], []
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)

        # At the end of training, evaluate on the whole test set.
        if step == total_steps:
          steps_per_test = total_test_steps

        logging.info('Starting multicrop test')
        for _ in range(steps_per_test):
          test_batch = next(dataset.test_iter)

          if not is_multilabel_model:
            # Only keep metrics.
            t_metrics = test_step_pmapped(train_state, test_batch)
          else:
            # Return metrics, logits, labels.
            t_metrics, t_logits, t_labels = test_step_pmapped(
                train_state, test_batch)
            # This should return n_classes logits and labels for each eval clip,
            # with N eval clips as we run in N devices in parallel.
            t_logits_in_cpu = vivit_train_utils.to_cpu(t_logits)
            t_labels_in_cpu = vivit_train_utils.to_cpu(t_labels)
            all_test_logits.append(t_logits_in_cpu)
            all_test_labels.append(t_labels_in_cpu)

          # Fetch t_metrics to host and store.
          test_metrics.append(train_utils.unreplicate_and_get(t_metrics))

        if is_multilabel_model:
          # Once traversed the entire eval set, compute mAP.
          all_test_logits_concat = np.concatenate(all_test_logits, axis=0)
          all_test_labels_concat = np.concatenate(all_test_labels, axis=0)
          additional_summary = evaluation_lib.compute_mean_avg_precision_dprime(
              all_test_logits_concat,
              all_test_labels_concat,
              return_per_class_ap=dataset.meta_data['num_classes'] < 10)

        # Log eval summary.
        train_utils.log_eval_summary(
            step=step,
            eval_metrics=test_metrics,
            extra_eval_summary=additional_summary,
            writer=writer,
            key_separator='/',
            prefix='test')
        logging.info('Completed multicrop test')
        del test_metrics
        writer.flush()
        chrono.resume()

  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  logging.info('Parameter summary after completing training.')
  # Object train_state is replicated for each TPU core.
  unrep_train_state = jax_utils.unreplicate(train_state)
  debug_utils.log_param_shapes(unrep_train_state.params)
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary
