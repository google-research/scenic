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
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple, Type, Union

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import jax_utils
import flax.linen as nn
import jax
from jax.example_libraries.optimizers import clip_grads
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.common_lib import debug_utils
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import base_model
from scenic.projects.av_mae import base_model as avmae_base_model
from scenic.projects.av_mae import train_utils as avmae_train_utils
from scenic.train_lib_deprecated import lr_schedules
from scenic.train_lib_deprecated import optimizers
from scenic.train_lib_deprecated import pretrain_utils
from scenic.train_lib_deprecated import train_utils


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, jnp.ndarray, Batch, Optional[jnp.ndarray]],
                  float]
# JAX team is working on type annotation for pytree:
# https://github.com/google/jax/issues/1555
PyTree = Any


def initialize_model(
    *,
    model_def: nn.Module,
    input_spec_dict: Dict[str, Tuple[Tuple[int, ...], jnp.dtype]],
    config: ml_collections.ConfigDict,
    rngs: Union[jnp.ndarray, Mapping[str, jnp.ndarray]],
    is_train: Optional[bool] = False
) -> Tuple[PyTree, PyTree, int, Optional[float]]:
  """Initializes parameters and model state.

  Args:
    model_def: Definition of a model.
    input_spec_dict: A dictionary of modality_name: (shape, dtype) specifying
      the shape and dtype of the inputs. If unspecified the dtype is float32.
    config: Configurations of the initialization.
    rngs: Jax rng keys.
    is_train: If the scenic model should be initialized in the train mode.

  Returns:
    Initial params, Init model_state, and number of trainable_params.
  """
  batch_size = (config.batch_size //
                jax.device_count()) if config.get('batch_size') else None
  dummy_input = {}

  for key, input_spec in input_spec_dict.items():
    in_st = debug_utils.input_spec_to_jax_shape_dtype_struct(
        input_spec, batch_size=batch_size)
    dummy_input[key] = jnp.zeros(in_st.shape, in_st.dtype)

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def _initialize_model(rngs):
    """Initialization function to be jitted."""
    init_model_state, init_params = flax.core.pop(model_def.init(
        rngs, dummy_input, train=is_train, debug=False), 'params')
    # Set bias in the head to low value, such that loss is small initially.
    if config.get('init_head_bias', None) is not None:
      init_params = flax.core.unfreeze(init_params)
      for key in input_spec_dict.keys():
        init_params[f'output_projection_{key}'] = optimizers.tree_map_with_names(  # pylint: disable=line-too-long
            lambda p: jnp.full_like(p, config.init_head_bias),
            init_params[f'output_projection_{key}'],
            match_name_fn=lambda name: 'bias' in name)
      init_params = flax.core.freeze(init_params)
    return init_params, init_model_state

  if not isinstance(rngs, dict):
    rngs = {'params': rngs}
  init_params, init_model_state = _initialize_model(rngs)
  # Pop out params rng:
  rngs.pop('params')

  # Count number of trainable parameters:
  num_trainable_params = debug_utils.log_param_shapes(init_params)
  gflops = None  # TODO(lgeorgescu): count the gflops.

  return init_params, init_model_state, num_trainable_params, gflops


def compute_max_norm(tensors: train_utils.PyTree) -> float:
  """Compute the maximum norm in a pytree of tensors."""
  leaves, _ = jax.tree_util.tree_flatten(tensors)
  norms = jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))
  max_norm = jnp.max(norms)
  return max_norm  # pytype: disable=bad-return-type  # jnp-type


def compute_feature_targets(batch: Batch, config: ml_collections.ConfigDict):
  """Compute the feature targets for feature regression.

  Args:
    batch: A single batch of data. This is updated with the feature target.
    config: The training configuration, used to generate the targets.
  """
  # TODO(lgeorgescu): some parameters should be modified only for one modality
  # such as patch size.

  batch['targets'] = {}  # pytype: disable=container-type-mismatch  # jax-ndarray
  feature_targets = config.masked_feature_loss.target
  for feature_target in feature_targets:
    if feature_target == avmae_base_model.FeatureTargets.RGB:
      batch['targets'][feature_target] = avmae_base_model.get_rgb_targets(
          batch['inputs'][feature_target],
          tuple(config.model.patches.size),
          config.masked_feature_loss.get('select_central_frame'),
          config.masked_feature_loss.get('reconstruct_grayscale', False),
          config.masked_feature_loss.get('standardise_per_patch', False),
          config.masked_feature_loss.get('standardise_per_patch_channels',
                                         False))
    elif feature_target == avmae_base_model.FeatureTargets.SPECTROGRAM:
      batch['targets'][feature_target] = avmae_base_model.get_spectogram_targets(  # pylint: disable=line-too-long
          batch['inputs'][feature_target], tuple(config.model.patches.size[:2]),
          config.masked_feature_loss.get('standardise_per_patch', False)
      )
    else:
      raise ValueError(f'Unsupported feature target: {feature_target}.')


def train_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    learning_rate_fn: Callable[[int], float],
    loss_fn: LossFn,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False):
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
    learning_rate_fn: Learning rate scheduler which given the global_step
      generates the learning rate.
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
    raise ValueError('mixup is not supported yet!')

  # Bind the dropout rng to the host/device we are on.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device')

  # Compute the targets for feature regression.
  compute_feature_targets(batch, config)

  def training_loss_fn(params, batch, dropout_rng):
    variables = {'params': params, **train_state.model_state}
    (logits, token_mask), new_model_state = flax_model.apply(
        variables,
        batch['inputs'],
        mutable=['batch_stats'],
        train=True,
        rngs={'dropout': dropout_rng},
        debug=debug)
    loss = loss_fn(logits, token_mask, batch, variables['params'])
    return loss, (new_model_state, logits, token_mask)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  (loss, (new_model_state, logits, token_mask)), grad = compute_gradient_fn(
      train_state.optimizer.target, batch, dropout_rng)  # pytype: disable=attribute-error
  metrics = metrics_fn(logits, token_mask, batch)
  metrics['total_loss'] = (loss, 1)

  for key, token_mask_  in token_mask.items():
    metrics[f'mask_ratio_{key}'] = (jnp.mean(token_mask_), 1)

  step = train_state.global_step
  lr = learning_rate_fn(step)
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')

  metrics['max_grad_norm_preclip'] = (compute_max_norm(grad), 1)
  if config.get('max_grad_norm', None) is not None:
    grad = clip_grads(grad, config.max_grad_norm)
    metrics['max_grad_norm_postclip'] = (compute_max_norm(grad), 1)

  new_optimizer = train_state.optimizer.apply_gradient(grad, learning_rate=lr)  # pytype: disable=attribute-error

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
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False
) -> Tuple[Dict[str, Tuple[float, int]], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    config: Configurations of the experiment.
    debug: Whether the debug mode is enabled during evaluation. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Calculated metrics and logits.
  """
  variables = {
      'params': train_state.optimizer.target,  # pytype: disable=attribute-error
      **train_state.model_state
  }

  # Compute the targets for feature regression.
  compute_feature_targets(batch, config)

  # We need an rng for masking at test time.
  # Note that we are using the same rng for the whole validation set (ie each
  # batch will have the same token mask).
  _, rng = jax.random.split(train_state.rng)
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device')

  logits, token_mask = flax_model.apply(
      variables, batch['inputs'], train=True, mutable=False, debug=debug,
      rngs={'dropout': dropout_rng})
  metrics = metrics_fn(logits, token_mask, batch)

  return metrics, logits, token_mask, batch['targets']


def get_image_grid(
    targets: jnp.ndarray, predictions: jnp.ndarray,
    token_mask: jnp.ndarray, config: ml_collections.ConfigDict,
    target_modality: str, input_size: Optional[
        Union[Tuple[int, int, int],
              Tuple[int, int, int, int]]] = None) -> Optional[jnp.ndarray]:
  """Returns an image grid for summary writing."""

  image_grid = None

  n_columns = config.masked_feature_loss.get('summary_num_columns',
                                             1)
  if len(config.model.patches.size) == 2 or target_modality == 'spectrogram':
    patch_size = tuple(config.model.patches.size[:2]) if (
        target_modality == 'spectrogram') else tuple(config.model.patches.size)

    image_grid = avmae_train_utils.generate_image_grid(
        targets, predictions, token_mask,
        patch_size, n_columns=n_columns, input_size=input_size,
        modality=target_modality)
  elif len(config.model.patches.size) == 3:
    num_img_in_column = config.masked_feature_loss.get(
        'number_of_img_in_column', 16)
    select_central_frame = config.masked_feature_loss.get(
        'select_central_frame', True)

    assert input_size is not None, 'Input size must be provided for video!'

    image_grid = avmae_train_utils.generate_image_grid_from_video(
        targets, predictions, token_mask,
        tuple(config.model.patches.size),
        input_size, num_img_in_column=num_img_in_column,
        select_central_frame=select_central_frame)
  else:
    raise ValueError(
        'The visualization is not implemented for'
        f'{config.model.patches} patches!')
  return image_grid


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
      'params': train_state.optimizer.target,  # pytype: disable=attribute-error
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

  if representation.ndim == 3:
    # Feature regression models return [batch, num_tokens, channels]
    logging.info('Representation shape before pooling tokens: %s',
                 representation.shape)
    representation = jnp.mean(representation, axis=1)
  logging.info('Representation shape: %s', representation.shape)

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
  rng, params_init_rng, dropout_init_rng = jax.random.split(rng, num=3)
  init_rngs = {'params': params_init_rng, 'dropout': dropout_init_rng}
  input_spec_dict = {}
  for key in config.dataset_configs.modalities:
    input_spec = (dataset.meta_data['input_shape'][key],
                  dataset.meta_data['input_dtype'][key])
    input_spec_dict[key] = input_spec

  (params, model_state, num_trainable_params,
   gflops) = initialize_model(
       model_def=model.flax_model,
       input_spec_dict=input_spec_dict,
       config=config,
       rngs=init_rngs,
       is_train=True)

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
    logging.info('Parameter summary after restoring checkpoint')
    debug_utils.log_param_shapes(train_state.optimizer.target)  # pytype: disable=attribute-error

  if (start_step == 0  # Which means "no" checkpoint is restored!
      and config.get('init_from') is not None):
    restored_model_cfg = config.init_from.get('model_config')
    init_checkpoint_path = config.init_from.get('checkpoint_path')
    restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
        init_checkpoint_path, train_state, assert_exist=True)
    # Load params from the init_model.
    train_state = model.init_from_train_state(  # pytype: disable=attribute-error
        train_state, restored_train_state, restored_model_cfg)
    logging.info('Parameter summary after adapting pretrained checkpoint.')
    debug_utils.log_param_shapes(train_state.optimizer.target)
    del restored_train_state

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
          config=config,
          debug=config.debug_eval),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1,),
  )

  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps

  def evaluate(train_state: train_utils.TrainState, step: int,
               valid_iter: Iterator[Batch],
               num_valid_ex: int) -> Dict[str, Any]:
    eval_summary = {}
    image_summary = {}
    if not isinstance(valid_iter, dict):  # Only on validation set.
      valid_iter, num_valid_ex = {'valid': valid_iter}, {'valid': num_valid_ex}

    for val_name, val_iter in valid_iter.items():
      num_ex = num_valid_ex[val_name]
      # Ceil rounding such that we include the last incomplete batch.
      total_eval_steps = int(np.ceil(num_ex / config.batch_size))
      steps_per_eval = config.get('steps_per_eval') or total_eval_steps
      eval_metrics = []
      for iteration in range(steps_per_eval):
        eval_batch = next(val_iter)
        if dataset.meta_data['target_is_onehot']:  # Which includes multi-hot.
          # Ignore the entries with all zero label for evaluation.
          eval_batch['batch_mask'] *= eval_batch['label'].max(axis=-1)
        e_metrics, predictions, token_mask, targets = eval_step_pmapped(
            train_state, eval_batch)
        eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))

        if iteration == 0:
          for key in eval_batch['inputs'].keys():  # pytype: disable=attribute-error  # jax-ndarray
            input_size = eval_batch['inputs'][key].shape[2:]
            unreplicate = jax_utils.unreplicate
            image_grid = get_image_grid(
                unreplicate(targets[key]), unreplicate(predictions[key]),
                unreplicate(token_mask[key]), config,
                target_modality=key, input_size=input_size)

            if image_grid is not None:
              image_summary[f'valid/reconstruction_{key}'] = (
                  jax.device_get(image_grid))

        del predictions
        del token_mask
        del targets
      eval_summary.update(
          train_utils.log_eval_summary(
              step=step,
              eval_metrics=eval_metrics,
              writer=writer,
              key_separator='/',
              prefix=val_name))
    del eval_metrics
    if image_summary:
      writer.write_images(step, image_summary)
      writer.flush()
    return eval_summary

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None

  chrono = train_utils.Chrono(
      first_step=start_step,
      total_steps=total_steps,
      steps_per_epoch=steps_per_epoch,
      global_bs=config.batch_size,
      accum_train_time=int(jax_utils.unreplicate(train_state.accum_train_time)))

  logging.info('Starting training loop at step %d for %d steps',
               start_step, total_steps)

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
      train_metrics.append(t_metrics)
      # Additional training logs: learning rate:
      extra_training_logs.append({'learning_rate': lr})

    # Quick indication that training is happening.
    logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
    for h in hooks:
      h(step)

    chrono.pause()  # Below are once-in-a-while ops -> pause.
    ############### LOG TRAIN SUMMARY ###############
    if (step % log_summary_steps == 1) or (step == total_steps):
      if lead_host:
        chrono.tick(step, writer=writer)
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(train_utils.unreplicate_and_get,
                                               train_metrics),
          extra_training_logs=jax.tree_util.tree_map(
              train_utils.unreplicate_and_get, extra_training_logs),
          key_separator='/',
          writer=writer)
      # Reset metric accumulation for next evaluation cycle.
      train_metrics, extra_training_logs = [], []

    ################### EVALUATION #######################
    if (step % log_eval_steps == 1) or (step == total_steps):
      # Sync model state across replicas.
      with report_progress.timed('eval'):
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        eval_summary = evaluate(train_state, step, dataset.valid_iter,
                                dataset.meta_data['num_eval_examples'])

    ##################### CHECKPOINTING ############################
    if ((step % checkpoint_steps == 1) or
        (step == total_steps)) and config.checkpoint:
      with report_progress.timed('checkpoint'):
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        if lead_host:
          train_state.replace(  # pytype: disable=attribute-error
              accum_train_time=chrono.accum_train_time)
          train_utils.save_checkpoint(workdir, train_state)

    chrono.resume()  # Un-pause now.

  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary
