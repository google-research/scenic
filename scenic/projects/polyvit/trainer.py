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

"""PolyVit Training Script."""

import collections
import copy
import functools
from typing import Any, Callable, Dict, Optional, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
from flax import jax_utils
import flax.linen as nn
import jax
from jax.example_libraries import optimizers as jax_optimizers
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
import optax
from scenic.dataset_lib import dataset_utils
from scenic.projects.mbt import trainer as mbt_trainer
from scenic.projects.polyvit import polyvit_base_model
from scenic.projects.polyvit import train_utils as polyvit_train_utils
from scenic.projects.vivit import evaluation_lib
from scenic.train_lib import optax as scenic_optax
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils
from scenic.train_lib.transfer import fewshot_utils


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray], str],
                    Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch, str, Optional[jnp.ndarray]], float]
LrFns = Dict[str, Callable[[jnp.ndarray], jnp.ndarray]]


def train_step(
    task: str,
    dataset: str,
    modality: str,
    train_state: train_utils.TrainState,
    batch: Batch,
    flax_model: nn.Module,
    loss_fn: LossFn,
    lr_fns: LrFns,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False,
) -> Tuple[
    train_utils.TrainState, Dict[str, Tuple[float, int]], Dict[str, Any]
]:
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Note that in this code, the buffers of the first (train_state) and second
  (batch) arguments are donated to the computation.

  Args:
    task: The task for which we are running the train_step.
    dataset: The name of the dataset used for the task.
    modality: The modality of the inputs in the batch.
    train_state: The state of training including the current global_step,
      model_state, rng, and optimizer. The buffer of this argument can be
      donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    flax_model: A Flax model.
    loss_fn: A loss function that given logits, a batch, and parameters of the
      model calculates the loss.
    lr_fns: The learning rate fns used for the optimizer in train_state.
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    config: Configurations of the experiment.
    debug: bool; Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training, computed metrics, and learning rate for logging.
  """
  training_logs = {}
  new_rng, rng = jax.random.split(train_state.rng)

  mixup_config = config.get('mixups', ml_collections.ConfigDict()).get(dataset)

  if mixup_config is not None:
    mixup_rng, rng = jax.random.split(rng, 2)
    mixup_rng = train_utils.bind_rng_to_host_device(
        mixup_rng,
        axis_name='batch',
        bind_to=mixup_config.get('bind_to', 'device'),
    )
    if modality == polyvit_base_model.Modality.AUDIO:
      batch = mbt_trainer.mixup_modalities(
          batch,
          mixup_config.alpha,
          True,
          mixmod=mixup_config.get('mixmod', False),
          rng=mixup_rng,
      )
      batch['label'] = batch['label']['all']
    elif modality == polyvit_base_model.Modality.VIDEO:
      batch = dataset_utils.mixup(
          batch,
          mixup_config.alpha,
          mixup_config.get('image_format', 'NTHWC'),
          rng=mixup_rng,
      )
    else:
      raise ValueError(f'Mixup not supported for modality {modality}')

  if modality == polyvit_base_model.Modality.AUDIO:
    batch['inputs'] = batch['inputs']['spectrogram']

  # Bind the rng to the host/device we are on.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device'
  )

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    logits, new_model_state = flax_model.apply(
        variables,
        x=batch['inputs'],
        targets=batch['label'],
        task=task,
        dataset=dataset,
        modality=modality,
        mutable=['batch_stats'],
        train=True,
        rngs={'dropout': dropout_rng},
        debug=debug,
    )
    loss = loss_fn(logits, batch, dataset, variables['params'])
    return loss, (new_model_state, logits)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  (train_cost, (new_model_state, logits)), grad = compute_gradient_fn(
      train_state.params
  )
  del train_cost

  # We clip gradients before pmean in ViViT and AViT and after in ViT,
  # following the original authors' code.
  if config.get('max_grad_norm', None) is not None and modality in [
      polyvit_base_model.Modality.VIDEO,
      polyvit_base_model.Modality.AUDIO,
  ]:
    grad = jax_optimizers.clip_grads(grad, config.max_grad_norm)

  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')

  if (
      config.get('max_grad_norm', None) is not None
      and modality == polyvit_base_model.Modality.IMAGE
  ):
    grad = jax_optimizers.clip_grads(grad, config.max_grad_norm)

  updates, new_opt_state = train_state.tx.update(
      grad, train_state.opt_state, train_state.params
  )
  new_params = optax.apply_updates(train_state.params, updates)
  training_logs['l2_grads'] = jnp.sqrt(
      sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grad)])
  )
  ps = jax.tree_util.tree_leaves(new_params)
  training_logs['l2_params'] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
  us = jax.tree_util.tree_leaves(updates)
  training_logs['l2_updates'] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))
  for name, lr_fn in lr_fns.items():
    lr_name = 'learning_rate' if name == 'all' else f'learning_rate_{name}'
    training_logs[lr_name] = lr_fn(train_state.global_step)

  metrics = metrics_fn(logits, batch, dataset)
  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      params=new_params,
      model_state=new_model_state,
      rng=new_rng,
  )

  return new_train_state, metrics, training_logs


def eval_step(
    task: str,
    dataset: str,
    modality: str,
    train_state: train_utils.TrainState,
    batch: Batch,
    flax_model: nn.Module,
    metrics_fn: MetricFn,
    debug: Optional[bool] = False,
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
    task: Task for which we are running the eval_step.
    dataset: The name of the dataset used for the task.
    modality: The modality for the train step.
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

  if modality == polyvit_base_model.Modality.AUDIO:
    batch['inputs'] = batch['inputs']['spectrogram']

  variables = {'params': train_state.params, **train_state.model_state}
  logits = flax_model.apply(
      variables,
      x=batch['inputs'],
      targets=batch['label'],
      task=task,
      dataset=dataset,
      modality=modality,
      train=False,
      mutable=False,
      debug=debug,
  )
  metrics = metrics_fn(logits, batch, dataset)

  return metrics, logits


def representation_fn(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    representation_layer: str,
    gather_to_host: bool = True,
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
  modality = 'image'
  task = polyvit_base_model.Task.FEWSHOT

  variables = {'params': train_state.params, **train_state.model_state}

  representation_layer_parts = representation_layer.split('/')
  filter_rep = lambda mdl, _: mdl.name == representation_layer_parts[-1]
  _, model_state = flax_model.apply(
      variables,
      x=batch['inputs'],
      targets=batch['label'],
      task=task,
      modality=modality,
      train=False,
      capture_intermediates=filter_rep,
      mutable=['intermediates'],
      debug=False,
  )
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
    model_cls: Any,
    dataset_dict: Dict[str, dataset_utils.Dataset],
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
    dataset_dict: A dict of datasets that each has train_iter, eval_iter,
      meta_data, and optionally, test_iter.
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
  datasets_metadata = {name: ds.meta_data for name, ds in dataset_dict.items()}

  # Build the loss_and_metrics_fn, metrics, and flax_model.
  model = model_cls(config, datasets_metadata)

  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  input_spec = {}
  for ds_name, meta_data in datasets_metadata.items():
    input_spec_key = (('dataset', ds_name), ('task', meta_data['task']),
                      ('modality', meta_data['modality']))

    if meta_data['modality'] == polyvit_base_model.Modality.AUDIO:
      input_shape = meta_data['input_shape']['spectrogram']
    else:
      input_shape = meta_data['input_shape']

    if meta_data['task'] in [
        polyvit_base_model.Task.LABEL,
        polyvit_base_model.Task.MULTILABEL,
        polyvit_base_model.Task.MULTIHEADLABEL,
    ]:
      input_spec[input_spec_key] = [(input_shape,
                                     meta_data.get('input_dtype', jnp.float32))]
    else:
      raise ValueError(
          f'Input specs for the task "{meta_data["task"]}" is not defined.'
      )

  (params, model_state, num_trainable_params, gflops) = (
      train_utils.initialize_multitask_model(
          model_def=model.flax_model,
          input_spec=input_spec,
          config=config,
          rngs=init_rng,
      )
  )

  # Multi-task training strategy of 'Weighted Task Sampling':
  all_datasets = []
  all_datasets_num_train_examples = []
  for name, metadata in datasets_metadata.items():
    all_datasets.append(name)
    all_datasets_num_train_examples.append(
        metadata.get('num_train_examples', 0)
    )
  ds_indices_per_step = []
  for index, ds_name in enumerate(all_datasets):
    n_steps = config.batch_sampling_strategy_steps.get(ds_name)
    ds_indices_per_step.append(jnp.full((n_steps,), index))
  ds_indices_per_step = jnp.concatenate(ds_indices_per_step)
  ds_indices_per_step = jax.random.permutation(
      jax.random.PRNGKey(0), ds_indices_per_step
  )
  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = polyvit_train_utils.get_num_training_steps(
      config, datasets_metadata
  )

  def get_dataset_at_step(step):
    return all_datasets[ds_indices_per_step[step]]  # pytype: disable=unsupported-operands  # jax-types

  # Create LR schedules and optimizer.
  schedule_fns = scenic_optax.make_schedule(config.get('schedule'))

  def update_schedule_fn(sfn):
    (re, name, (fn, base_lr)) = sfn
    updated_lr = []
    for step in range(1, total_steps + 1):
      dataset = get_dataset_at_step(step)
      updated_lr.append(
          fn(step) * config.get('lr_coefs', {dataset: 1.0})[dataset]
      )
    updated_lr = jnp.array(updated_lr)
    return (re, name, (lambda step: updated_lr[step], base_lr))

  schedule_fns = [update_schedule_fn(sfn) for sfn in schedule_fns]

  tx, _ = scenic_optax.make(config.optimizer, schedule_fns, params)
  opt_state = tx.init(params)

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
      metadata={'chrono': chrono.save()},
  )
  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state
    )
  chrono.load(train_state.metadata['chrono'])
  train_state = train_state.replace(metadata={})
  if (
      start_step == 0  # Which means "no" checkpoint is restored!
      and config.get('init_from') is not None
  ):
    restored_model_cfg = config.init_from.get('model_config')
    init_checkpoint_path = config.init_from.get('checkpoint_path')
    if init_checkpoint_path is not None:
      if config.init_from.get('init_from_vit', False):
        checkpoint_format = config.init_from.get('checkpoint_format', 'scenic')
        if checkpoint_format == 'scenic':
          restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
              init_checkpoint_path, None, assert_exist=True
          )
        elif checkpoint_format == 'big_vision':
          restored_train_state = (
              polyvit_train_utils.restore_pretrained_big_vision_checkpoint(
                  init_checkpoint_path
              )
          )
          # Config dict in big_vision is not the same format as scenic.
          # Therefore, make sure config match the config of the loaded model!
          restored_model_cfg = copy.deepcopy(config)
          # The following is needed when the restored and target models used a
          # different classifier. As big_vision uses a different config dict, we
          # have to specify this manually.

        train_state = model.init_from_vit_train_state(
            train_state, restored_train_state, restored_model_cfg
        )
      elif config.init_from.get('init_from_polyvit', False):
        restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
            init_checkpoint_path, train_state, assert_exist=True
        )
        # Load params from the init_model.
        train_state = model.init_from_polyvit_train_state(  # pytype: disable=attribute-error
            train_state,
            restored_train_state,
            tokenizer_to_use=config.init_from.get('tokenizer_to_use'),
            tokenizer_to_init=config.init_from.get('tokenizer_to_init'),
            resolution=config.init_from.get('resolution'),
        )
      elif config.init_from.get('init_from_mbt', False):
        restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
            init_checkpoint_path, None, assert_exist=True
        )
        # Load params from the init_model.
        train_state = model.init_from_mbt_train_state(  # pytype: disable=attribute-error
            train_state,
            restored_train_state,
            tokenizer_to_init=config.init_from.get(
                'tokenizer_to_init', 'tokenizer_spec'
            ),
            resolution=config.init_from.get('resolution'),
        )
      elif config.init_from.get('init_from_vivit', False):
        restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
            init_checkpoint_path, None, assert_exist=True
        )
        # Load params from the init_model.
        train_state = model.init_from_vivit_train_state(  # pytype: disable=attribute-error
            train_state,
            restored_train_state,
            tokenizer_to_init=config.init_from.get(
                'tokenizer_to_init', 'tokenizer3d'
            ),
            resolution=config.init_from.get('resolution'),
        )
      else:
        restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
            init_checkpoint_path, train_state, assert_exist=True
        )
        # Load params from the init_model.
        train_state = model.init_from_train_state(  # pytype: disable=attribute-error
            train_state, restored_train_state, restored_model_cfg
        )
      del restored_train_state

  # Replicate the optimzier, state, and rng.
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  train_step_pmapped = jax.pmap(
      functools.partial(
          train_step,
          flax_model=model.flax_model,
          loss_fn=model.loss_function,
          lr_fns={name: lr_fn for _, name, (lr_fn, _) in schedule_fns},
          metrics_fn=model.get_metrics_fn('train'),
          config=config,
          debug=config.debug_train,
      ),
      axis_name='batch',
      static_broadcasted_argnums=(0, 1, 2),  # Task, dataset and modality args.
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(3, 4),
  )
  eval_step_pmapped = jax.pmap(
      functools.partial(
          eval_step,
          flax_model=model.flax_model,
          metrics_fn=model.get_metrics_fn('validation'),
          debug=config.debug_eval,
      ),
      axis_name='batch',
      static_broadcasted_argnums=(0, 1, 2),  # Task, dataset and modality args.
      # We can donate the eval_batch's buffer.
      donate_argnums=(4,),
  )
  if 'fewshot' in config:
    representation_fn_fewshot = functools.partial(
        representation_fn,
        flax_model=model.flax_model,
        representation_layer=config.fewshot.representation_layer,
    )
    fewshotter = fewshot_utils.FewShotEvaluator(representation_fn_fewshot,
                                                config.fewshot)

  log_eval_steps = config.get('log_eval_steps')
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps

  def evaluate(
      train_state: train_utils.TrainState, step: int, dataset: str
  ) -> Dict[str, Any]:
    ds = dataset_dict[dataset]
    valid_iter = ds.valid_iter
    task = ds.meta_data['task']
    modality = ds.meta_data['modality']
    num_valid_ex = ds.meta_data['num_eval_examples']
    eval_summary = {}
    if not isinstance(valid_iter, dict):  # Only on validation set.
      valid_iter, num_valid_ex = {'valid': valid_iter}, {'valid': num_valid_ex}

    for val_name, val_iter in valid_iter.items():
      num_ex = num_valid_ex[val_name]
      is_one_hot = ds.meta_data['target_is_onehot']
      # Ceil rounding such that we include the last incomplete batch.
      if config.get('batch_sizes') is not None:
        batch_size = config.batch_sizes.get(dataset)
      else:
        batch_size = config.batch_size
      total_eval_steps = int(np.ceil(num_ex / batch_size))
      steps_per_eval = config.get('steps_per_eval') or total_eval_steps
      eval_metrics = []
      additional_summary = None
      if modality == polyvit_base_model.Modality.AUDIO:
        eval_logits = []
        eval_labels = []
        n_classes = ds.meta_data['num_classes']
      for _ in range(steps_per_eval):
        eval_batch = next(val_iter)
        if is_one_hot:  # Which includes multi-hot.
          # Ignore the entries with all zero label for evaluation.
          eval_batch['batch_mask'] *= eval_batch['label'].max(axis=-1)
        e_metrics, logits = eval_step_pmapped(task, dataset, modality,
                                              train_state, eval_batch)
        if modality == polyvit_base_model.Modality.AUDIO:
          eval_logits.append(
              jax.device_get(
                  logits.reshape(  # pytype: disable=attribute-error
                      [-1, n_classes]
                  )
              )
          )
          eval_labels.append(
              jax.device_get(
                  eval_batch['label'].reshape(  # pytype: disable=attribute-error
                      [-1, n_classes]
                  )
              )
          )
        eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))
      if modality == polyvit_base_model.Modality.AUDIO:
        # Note that this is the Mean AP computed from the examples processed
        # by a single host.
        additional_summary = evaluation_lib.compute_mean_average_precision(
            np.concatenate(eval_logits, axis=0),
            np.concatenate(eval_labels, axis=0),
            return_per_class_ap=n_classes < 10,
        )
      eval_summary.update(
          train_utils.log_eval_summary(
              step=step,
              eval_metrics=eval_metrics,
              extra_eval_summary=additional_summary,
              writer=writer,
              prefix=f'{task}/{dataset}/{val_name}',
          )
      )
    del eval_metrics
    writer.flush()
    return eval_summary

  train_metrics = collections.defaultdict(list)
  extra_training_logs = collections.defaultdict(list)
  train_summary, eval_summary = None, None

  chrono = train_utils.Chrono()
  logging.info('Starting training loop at step %d.', start_step)

  chrono.inform(
      start_step,
      total_steps,
      polyvit_train_utils.get_average_batch_size(config),
      steps_per_epoch,
  )
  logging.info('Starting training loop at step %d.', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer
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
      step0_log.update(gflops)
    writer.write_scalars(1, step0_log)

  def get_next_train_batch(step):
    dataset = get_dataset_at_step(step)
    ds = dataset_dict[dataset]
    return (
        next(ds.train_iter),
        ds.meta_data['task'],
        dataset,
        ds.meta_data['modality'],
    )

  write_note(f'First step compilations...\n{chrono.note}')
  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch, train_task, train_ds, train_modality = get_next_train_batch(
          step
      )
      train_state, t_metrics, t_logs = train_step_pmapped(
          train_task, train_ds, train_modality, train_state, train_batch
      )
      # This will accumulate metrics in accelerator memory up to the point that
      # we log them. This is no problem for small metrics but may be a problem
      # for large (e.g. segmentation) metrics. An alternative is to set
      # `log_summary_steps` to a small number, or to use
      # `train_utils.unreplicate_and_get` here instead of right before writing
      # summaries, but that means in each step, we have data transfer between
      # accelerator and host, which might slow down the training.
      train_metrics[(train_task, train_ds)].append(t_metrics)
      t_logs = jax.tree_util.tree_map(jax_utils.unreplicate, t_logs)
      extra_training_logs[(train_task, train_ds)].append(t_logs)

      # Quick indication that training is happening.
    logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
    for h in hooks:
      h(step)
    ############### LOG TRAIN SUMMARY ###############
    if (
        (step % log_summary_steps == 1)
        or (step == total_steps)
        or (lead_host and chrono.warmup)
    ):
      chrono.pause(wait_for=(train_metrics))
      if lead_host:
        chrono.tick(step, writer, write_note)
      train_summary = {}
      for train_task, train_ds in train_metrics.keys():
        train_summary.update(
            train_utils.log_train_summary(
                step=step,
                train_metrics=jax.tree_util.tree_map(
                    train_utils.unreplicate_and_get,
                    train_metrics[(train_task, train_ds)],
                ),
                extra_training_logs=jax.tree_util.tree_map(
                    jax.device_get,
                    extra_training_logs[(train_task, train_ds)],
                ),
                writer=writer,
                prefix=f'{train_task}/{train_ds}/train',
            )
        )
      # Reset metric accumulation for next evaluation cycle.
      train_metrics = collections.defaultdict(list)
      extra_training_logs = collections.defaultdict(list)
      chrono.resume()
    ################### EVALUATION #######################
    if (step % log_eval_steps == 1) or (step == total_steps):
      chrono.pause(wait_for=(train_state.params))
      with report_progress.timed('eval'):
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        eval_summaries = []
        for ds_name in dataset_dict.keys():
          eval_summaries.append(evaluate(train_state, step, ds_name))
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
              writer=writer, step=step, results=results
          )
          del results
          writer.write_scalars(step, {'zz/epoch': step / steps_per_epoch})
        writer.flush()
        chrono.resume()  # Un-pause now.

  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary
