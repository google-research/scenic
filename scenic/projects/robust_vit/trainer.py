"""Training Script."""

import functools
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Type

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import jax_utils
import flax.linen as nn
import jax
from jax.experimental.optimizers import clip_grads
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import base_model
from scenic.projects.robust_vit.train_lib import train_utils as robust_vit_train_utils
from scenic.projects.robust_vit.train_lib.optimizers_utils import get_partial_optimizer
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils
from scenic.train_lib.google.fewshot import fewshot_trainer
from scenic.train_lib.google.fewshot import fewshot_utils


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]


def get_customized_optimizer(config) -> Any:
  """Get Optimizer.

  Args:
    config: Configurations of the experiment.

  Returns:
    Selected optimizer
  """
  if config.optimizer in ['adam_vitonly', 'momentum_hp_vitonly']:
    opt_class = get_partial_optimizer(config)
  else:
    opt_class = optimizers.get_optimizer(config)
  return opt_class


def eval_step(
    *,
    flax_model: nn.Module,
    train_state: train_utils.TrainState,
    batch: Batch,
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
    flax_model: A Flax model.
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data. a metrics function, that given logits and
      batch of data, calculates the metrics as well as the loss.
    metrics_fn: A metrics function, that given logits and batch of data,
      calculates the metrics as well as the loss.
    debug: Whether the debug mode is enabled during evaluation.
      `debug=True` enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Logits and calculated metrics.
  """
  variables = {
      'params': train_state.optimizer.target,
      **train_state.model_state
  }
  logits = flax_model.apply(
      variables, batch['inputs'], train=False, mutable=False, debug=debug)
  metrics = metrics_fn(logits, batch)
  return metrics, logits


def train_step(
    *,
    flax_model: nn.Module,
    train_state: train_utils.TrainState,
    batch: Batch,
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
    flax_model: A Flax model.
    train_state: The state of training including the current global_step,
      model_state, rng, and optimizer. The buffer of this argument can be
      donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
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
    mixup_rng, rng = jax.random.split(rng, 2)
    mixup_rng = train_utils.bind_rng_to_host_device(
        mixup_rng,
        axis_name='batch',
        bind_to=config.mixup.get('bind_to', 'device'))
    batch = dataset_utils.mixup(
        batch,
        config.mixup.alpha,
        config.mixup.get('image_format', 'NHWC'),
        rng=mixup_rng)

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
  step = train_state.global_step
  lr = learning_rate_fn(step)
  (train_cost,
   (new_model_state,
    logits)), grad = compute_gradient_fn(train_state.optimizer.target)

  del train_cost
  grad = jax.lax.pmean(grad, axis_name='batch')

  if config.get('max_grad_norm', None) is not None:
    grad = clip_grads(grad, config.max_grad_norm)

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

  metrics = metrics_fn(logits, batch)
  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=step + 1,
      optimizer=new_optimizer,
      model_state=new_model_state,
      rng=new_rng)
  return new_train_state, metrics, lr


def representation_fn(
    *, flax_model: nn.Module, train_state: train_utils.TrainState, batch: Batch,
    config: ml_collections.ConfigDict
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Feeds the inputs to the model and returns their representations.

  Args:
    flax_model: A Flax model.
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data from the dataset.
    config: Configurations of the experiment.

  Returns:
    Representation learned by the model for the given inputs and the labels and
    masks collected from all hosts.
  """
  variables = {
      'params': train_state.optimizer.target,
      **train_state.model_state
  }

  representation_layer = config.fewshot.representation_layer.split('/')
  filter_rep = lambda mdl, _: mdl.name == representation_layer[-1]
  _, model_state = flax_model.apply(
      variables,
      batch['inputs'],
      train=False,
      capture_intermediates=filter_rep,
      mutable=['intermediates'],
      debug=False)
  if 'intermediates' not in model_state:
    raise ValueError(f'Layer with name "{config.fewshot.representation_layer}"'
                     ' does not exist in your model.')

  representation = model_state['intermediates']
  for rep_layer in representation_layer:
    if 'vit' in representation:
      # If this is combined model, then need special api
      representation = representation['vit'][rep_layer]
    elif rep_layer:
      representation = representation[rep_layer]
  representation = representation['__call__'][0]
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
   gflops) = robust_vit_train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(dataset.meta_data['input_shape'],
                    dataset.meta_data.get('input_dtype', jnp.float32))],
       config=config,
       rngs=init_rng)

  # Create optimizer.
  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  opt_class = get_customized_optimizer(config)
  optimizer = jax.jit(opt_class.create, backend='cpu')(params)
  rng, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
      global_step=0,
      optimizer=optimizer,
      model_state=model_state,
      rng=train_rng,
      accum_train_time=0)

  train_state = robust_vit_train_utils.init_from_vq_pretrain_state(
      train_state=train_state, config=config)

  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state)

  if (start_step == 0  # Which means "no" checkpoint is restored!
      and config.get('init_from') is not None):
    restored_model_cfg = config.init_from.get('model_config')
    init_checkpoint_path = config.init_from.get('checkpoint_path')
    restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
        init_checkpoint_path, train_state, assert_exist=True)

    # Load params from the init_model.
    train_state = model.init_from_train_state(  # pytype: disable=attribute-error
        train_state, restored_train_state, restored_model_cfg)

    optimizer_target = flax.core.unfreeze(train_state.optimizer.target)
    robust_vit_train_utils.dict_dfs_update_bias_scale(optimizer_target, '')
    optimizer = optimizer.replace(target=flax.core.freeze(optimizer_target))
    train_state = train_state.replace(optimizer=optimizer)

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
          fewshot_trainer.eval_step,
          flax_model=model.flax_model,
          metrics_fn=model.get_metrics_fn('validation'),
          debug=config.debug_eval),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1,),
  )
  if 'fewshot' in config:
    representation_fn_pmaped = jax.pmap(
        functools.partial(
            representation_fn, flax_model=model.flax_model, config=config),
        # We can donate the batch's buffer.
        donate_argnums=(1,),
        axis_name='batch')
    fewshotter = fewshot_utils.FewShotEvaluator(representation_fn_pmaped,
                                                config.fewshot)

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
      total_eval_steps = int(np.ceil(num_ex / config.batch_size))
      steps_per_eval = config.get('steps_per_eval') or total_eval_steps
      eval_metrics = []
      for _ in range(steps_per_eval):
        eval_batch = next(val_iter)
        if dataset.meta_data['target_is_onehot']:  # Which includes multi-hot.
          # Ignore the entries with all zero label for evaluation.
          eval_batch['batch_mask'] *= eval_batch['label'].max(axis=-1)
        e_metrics, _ = eval_step_pmapped(
            train_state=train_state, batch=eval_batch)
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

  chrono = train_utils.Chrono(
      first_step=start_step,
      total_steps=total_steps,
      steps_per_epoch=steps_per_epoch,
      global_bs=config.batch_size,
      accum_train_time=int(jax_utils.unreplicate(train_state.accum_train_time)))

  logging.info('Starting training loop at step %d.', start_step)

  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer)
  hooks = [report_progress]
  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  if start_step == 0:
    step0_log = {'num_trainable_params': num_trainable_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)

  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceContext('train', step_num=step):
      train_batch = next(dataset.train_iter)
      train_state, t_metrics, lr = train_step_pmapped(
          train_state=train_state, batch=train_batch)
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
          train_metrics=jax.tree_map(train_utils.unreplicate_and_get,
                                     train_metrics),
          extra_training_logs=jax.tree_map(train_utils.unreplicate_and_get,
                                           extra_training_logs),
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
    if ((step % checkpoint_steps == 1 and step > 1) or
        (step == total_steps)) and config.checkpoint:
      with report_progress.timed('checkpoint'):
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        if lead_host:
          train_state.replace(  # pytype: disable=attribute-error
              accum_train_time=chrono.accum_train_time)
          train_utils.save_checkpoint(workdir, train_state)

    ##################### FEWSHOT EVALUATION ############################
    if 'fewshot' in config:
      # Compute few-shot on-the-fly evaluation.
      if (step % config.fewshot.log_eval_steps == 1) or (step == total_steps):
        with report_progress.timed('fewshot'):
          results = fewshotter.run_all(train_state, config.fewshot.datasets)
          fewshotter.log_fewshot_summary(
              writer=writer, step=step, results=results)
          del results
          writer.write_scalars(step, {'zz/epoch': step / steps_per_epoch})
        writer.flush()

    chrono.resume()  # Un-pause now.

  # Wait until computations are done before exiting.
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary
