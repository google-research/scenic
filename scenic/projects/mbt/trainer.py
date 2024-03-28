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

"""Training Script for MBT."""

import copy
import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union

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
from scenic.projects.mbt import train_utils as mbt_train_utils
from scenic.projects.vivit import evaluation_lib
from scenic.projects.vivit import train_utils as vivit_train_utils
from scenic.train_lib_deprecated import lr_schedules
from scenic.train_lib_deprecated import optimizers
from scenic.train_lib_deprecated import pretrain_utils
from scenic.train_lib_deprecated import train_utils

# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]


def mixup_modalities(batch: Dict['str', Any],
                     alpha: float = 1.0,
                     batch_first: bool = True,
                     mixmod: bool = False,
                     rng: Optional[Any] = None) -> Dict['str', jnp.ndarray]:
  """Mixes multimodal inputs and labels within a single batch.

  For more details, please see https://arxiv.org/abs/1710.09412.

  This function supports both using `numpy` to do mixup in the input-pipeline
  and `jax.numpy` to do mixup within a jitted/pmapped function (e.g. within
  a pmapped train step to apply mixup on device patch).

  Results in a batch with:
    mixed_inputs[idx] = weight * inputs[idx] + (1-weight) * inputs[-(idx+1)],
    where weight is sampled from a beta distribution with parameter alpha.

  Args:
    batch: dict; A batch of data with 'inputs' and 'label'. batch['inputs'] has
      field like 'rgb' or 'spectrogram'.
    alpha: float; Used to control the beta distribution that weight is sampled
      from.
    batch_first: bool; Batch is the first dimension or the last dimension.
    mixmod: bool; If True, applies mixup to each modality separately.
    rng: JAX rng key. If given, JAX numpy will be used as the backend, and if
      None (default value), normal numpy will be used.

  Returns:
    Tuple (mixed_images, mixed_labels).
  """
  inputs, labels = batch['inputs'], batch['label']
  batch['label'] = {}
  num_modalities = len(inputs)

  if labels.shape[-1] == 1:
    raise ValueError('Mixup requires one-hot targets.')

  batch_size = labels.shape[0]

  # Setup the the numpy backend and prepare mixup weights.
  if rng is None:
    np_backend = np  # Ordinary numpy
    if mixmod:
      weights = list(np_backend.random.beta(alpha, alpha, size=num_modalities))
    else:
      weights = [np_backend.random.beta(alpha, alpha)] * num_modalities
  else:
    np_backend = jnp  # JAX numpy
    if mixmod:
      weights = list(jax.random.beta(rng, alpha, alpha, shape=[num_modalities]))
    else:
      weights = [jax.random.beta(rng, alpha, alpha)] * num_modalities
  for i in range(num_modalities):
    weights[i] *= np_backend.ones((batch_size, 1))

  # Mixup inputs.
  # Shape calculations use np to avoid device memory fragmentation:
  for modality, values in inputs.items():
    weight = weights[len(batch['label'])]
    # Mixup labels.
    batch['label'][modality] = weight * labels + (1.0 - weight) * labels[::-1]
    weight_shape = np.ones((values.ndim))
    if batch_first:
      weight_shape[0] = batch_size
    else:
      weight_shape[-1] = batch_size
    weight = np_backend.reshape(weight,
                                weight_shape.astype(np_backend.int32))
    reverse = []
    for i in range(values.ndim):
      if (i == 0 and batch_first) or (i == values.ndim - 1 and not batch_first):
        reverse.append(slice(-1, None, -1))
      else:
        reverse.append(slice(values.shape[i]))
    batch['inputs'][modality] = (weight * values +
                                 (1.0 - weight) * values[tuple(reverse)])
  if num_modalities == 1 or not mixmod:
    batch['label']['all'] = weights[0] * labels + (1.0 -
                                                   weights[0]) * labels[::-1]

  return batch


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
) -> Tuple[train_utils.TrainState, Dict[str, Tuple[float, int]], float]:
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
    learning_rate_fn: learning rate scheduler which give the global_step
      generates the learning rate.
    loss_fn: A loss function that given logits, a batch, and parameters of the
      model calculates the loss.
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    config: Configuration of the experiment.
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
    batch = mixup_modalities(
        batch,
        config.mixup.alpha,
        batch_first,
        mixmod=config.get('mixmod', False),
        rng=mixup_rng)
  else:
    # No mixup is applied, all modalities share the same labels.
    labels = batch['label']
    batch['label'] = {}  # pytype: disable=container-type-mismatch  # jax-ndarray
    for modality in batch['inputs']:
      batch['label'][modality] = labels
    batch['label']['all'] = labels

  # Bind the rng to the host/device we are on for dropout.
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
  step = train_state.global_step
  lr = learning_rate_fn(step)
  (train_cost,
   (new_model_state,
    logits)), grad = compute_gradient_fn(train_state.optimizer.target)

  if config.get('max_grad_norm', None) is not None:
    grad = clip_grads(grad, config.max_grad_norm)

  del train_cost
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')
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

  if isinstance(logits, dict):
    # We use the first retrieved logits to report training metrics.
    modality = list(logits.keys())[0]
    batch['label'] = batch['label'][modality]
    metrics = metrics_fn(logits[modality], batch)
  else:
    metrics = metrics_fn(logits, batch)
  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=step + 1,
      optimizer=new_optimizer,
      model_state=new_model_state,
      rng=new_rng)
  return new_train_state, metrics, lr


def eval_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: MetricFn,
    return_logits_and_labels: bool = False,
    debug: Optional[bool] = False
) -> Union[Tuple[Dict[str, Tuple[float, int]], jnp.ndarray, jnp.ndarray], Dict[
    str, Tuple[float, int]]]:
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
    return_logits_and_labels: Whether to return logits and labels or not.
    debug: Whether the debug mode is enabled during evaluation.
      `debug=True` enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Calculated metrics [and optionally logits].
  """
  variables = {
      'params': train_state.optimizer.target,
      **train_state.model_state
  }
  logits = flax_model.apply(
      variables,
      batch['inputs'],
      train=False, mutable=False, debug=debug)

  metrics = metrics_fn(logits, batch)
  if return_logits_and_labels:
    logits = jax.lax.all_gather(logits, 'batch')
    labels = jax.lax.all_gather(batch['label'], 'batch')
    return metrics, logits, labels
  return metrics


def test_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: MetricFn,
    n_clips: int = 2,
    return_logits_and_labels: bool = False,
    softmax_logits: bool = False,
    debug: bool = False,
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

  Returns:
    Calculated metrics [and optionally averaged logits that are of
    shape `[1, num_classes]`].
  """

  all_logits = jnp.zeros(batch['label'].shape[1])
  assert len(batch['batch_mask'].shape) == 1, (
      'Spatial padding is not supported in multi-crop evaluation.')

  variables = {
      'params': train_state.optimizer.target,
      **train_state.model_state
  }
  for modality in batch['inputs']:
    num_crops = batch['inputs'][modality].shape[0]
  for idx in range(0, num_crops, n_clips):
    current_input = {}
    for modality in batch['inputs']:
      current_input[modality] = batch['inputs'][modality][idx:idx + n_clips]
    logits = flax_model.apply(
        variables, current_input, train=False, mutable=False, debug=debug)

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
    all_logits = jax.lax.all_gather(all_logits, 'batch')
    labels = jax.lax.all_gather(batch['label'], 'batch')
    return metrics, all_logits, labels
  return metrics


def train(
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
  is_multilabel_model = (config.model_name == 'mbt_multilabel_classification')

  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  input_shapes = dataset.meta_data['input_shape']
  input_dtype = dataset.meta_data.get('input_dtype', jnp.float32)
  if isinstance(input_shapes, dict):
    input_spec = {
        modality: (input_shapes[modality], input_dtype)
        for modality in input_shapes
    }
  else:
    input_spec = [(input_shapes, input_dtype)]
  (params, model_state, num_trainable_params,
   gflops) = mbt_train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=input_spec,
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

  if (start_step == 0  # Which means "no" checkpoint is restored!
      and config.get('init_from') is not None):
    restored_model_cfg = config.init_from.get('model_config')
    init_checkpoint_path = config.init_from.get('checkpoint_path')
    checkpoint_format = config.init_from.get('checkpoint_format', 'scenic')
    if checkpoint_format == 'scenic':
      restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
          init_checkpoint_path, train_state, assert_exist=True)
    elif checkpoint_format == 'big_vision':
      restored_train_state = pretrain_utils.convert_big_vision_to_scenic_checkpoint(
          init_checkpoint_path, train_state)
      # Config dict in big_vision is not the same format as scenic.
      # Therefore, make sure config match the config of the loaded model!
      restored_model_cfg = copy.deepcopy(config)
      # The following is needed when the restored and target models used a
      # different classifier. As big_vision uses a different config dict, we
      # have to specify this manually.
      restored_model_cfg.model.classifier = config.init_from.get(
          'classifier_type', 'token')

    train_state = model.init_from_train_state(
        train_state, restored_train_state, restored_model_cfg,
        restore_output_proj=config.init_from.get('restore_output_proj', False))
    # Free unnecessary memory.
    del restored_train_state
  elif start_step == 0:
    logging.info('Training completely from scratch.'
                 'Not restoring from any checkpoint.')

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
  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  log_test_steps = 0
  if config.dataset_configs.get('do_multicrop_test'):
    log_test_steps = int(steps_per_epoch *
                         config.dataset_configs.log_test_epochs)

    test_step_pmapped = jax.pmap(
        functools.partial(
            test_step,
            flax_model=model.flax_model,
            metrics_fn=model.get_metrics_fn('test'),
            n_clips=config.get('multicrop_clips_per_device', 2),
            return_logits_and_labels=is_multilabel_model,
            debug=config.debug_eval),
        axis_name='batch',
        # We can donate the test_batch's buffer.
        donate_argnums=(1,),
    )

    assert config.dataset_configs.test_batch_size == jax.local_device_count(), (
        'The per-host batch size must be equal to the number of local devices.'
        'This ensures that each TPU device is processing different views of'
        'the same original video.')

    total_test_steps = int(
        np.ceil(dataset.meta_data['num_test_examples'] /
                (config.get('dataset_configs.test_batch_size') *
                 config.get('dataset_configs.num_test_clips') *
                 jax.process_count())))
    steps_per_test = config.get('steps_per_test') or total_test_steps

  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps

  # Ceil rounding such that we include the last incomplete batch.
  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None

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
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      train_state, t_metrics, lr = train_step_pmapped(train_state, train_batch)
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
    ###################### LOG TRAIN SUMMARY ########################
    if (step % log_summary_steps == 1) or (step == total_steps):
      if lead_host:
        chrono.tick(step, writer=writer)
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(train_utils.unreplicate_and_get,
                                               train_metrics),
          extra_training_logs=jax.tree_util.tree_map(
              train_utils.unreplicate_and_get, extra_training_logs),
          writer=writer,
          key_separator='/')
      # Reset metric accumulation for next evaluation cycle.
      train_metrics, extra_training_logs = [], []

    ################### EVALUATION ################################
    if (step % log_eval_steps == 1) or (step == total_steps):
      with report_progress.timed('eval'):
        eval_metrics = []
        additional_summary = None
        if is_multilabel_model:
          eval_logits = []
          eval_labels = []
          n_classes = dataset.meta_data['num_classes']
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        for _ in range(steps_per_eval):
          eval_batch = next(dataset.valid_iter)
          e_metrics = eval_step_pmapped(train_state, eval_batch)
          if is_multilabel_model:
            e_metrics, logits_batch, labels_batch = e_metrics
            eval_logits.append(vivit_train_utils.to_cpu(logits_batch))
            eval_labels.append(vivit_train_utils.to_cpu(labels_batch))
          # Fetch e_metrics to host and store.
          eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))
        if is_multilabel_model:
          additional_summary = evaluation_lib.compute_mean_average_precision(
              np.concatenate(eval_logits, axis=0),
              np.concatenate(eval_labels, axis=0),
              return_per_class_ap=n_classes < 10)
        # Log eval summary.
        eval_summary = train_utils.log_eval_summary(
            step=step,
            eval_metrics=eval_metrics,
            extra_eval_summary=additional_summary,
            writer=writer,
            key_separator='/')
        writer.flush()
        del eval_metrics
    ##################### CHECKPOINTING ###########################
    if ((step % checkpoint_steps == 0 and step > 0) or (step == total_steps) or
        (step % log_eval_steps == 1)) and config.checkpoint:
      with report_progress.timed('checkpoint'):
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        if lead_host:
          train_state.replace(  # pytype: disable=attribute-error
              accum_train_time=chrono.accum_train_time)
          train_utils.save_checkpoint(workdir, train_state)

    ############# MULTICROP TESTING ############################
    if (config.dataset_configs.get('do_multicrop_test') and
        ((step % log_test_steps == 1 and step > 1) or step == total_steps)):
      with report_progress.timed('test'):
        test_metrics = []
        additional_summary = None
        if is_multilabel_model:
          test_logits = []
          test_labels = []
          n_classes = dataset.meta_data['num_classes']
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)

        # At the end of training, evaluate on the whole test set.
        if step == total_steps:
          steps_per_test = total_test_steps

        logging.info('Starting multicrop test')
        for _ in range(steps_per_test):
          test_batch = next(dataset.test_iter)
          t_metrics = test_step_pmapped(train_state, test_batch)
          if is_multilabel_model:
            t_metrics, logits_batch, labels_batch = t_metrics
            test_logits.append(vivit_train_utils.to_cpu(logits_batch))
            test_labels.append(vivit_train_utils.to_cpu(labels_batch))
          # Fetch t_metrics to host and store.
          test_metrics.append(train_utils.unreplicate_and_get(t_metrics))
        if is_multilabel_model:
          # Note that this is the Mean AP computed from the examples processed
          # by a single host.
          additional_summary = evaluation_lib.compute_mean_average_precision(
              np.concatenate(test_logits, axis=0),
              np.concatenate(test_labels, axis=0),
              return_per_class_ap=n_classes < 10)
        # Log eval summary.
        train_utils.log_eval_summary(
            step=step,
            eval_metrics=test_metrics,
            writer=writer,
            extra_eval_summary=additional_summary,
            prefix='test',
            key_separator='/')
        logging.info('Completed multicrop test')
        writer.flush()
        # Free up some space.
        del test_metrics

    chrono.resume()  # un-pause now
  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary
