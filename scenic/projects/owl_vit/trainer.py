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

"""Training script for OWL-ViT."""

from concurrent import futures
import functools
import time
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
import flax
from flax import jax_utils
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
import optax
from scenic.dataset_lib import dataset_utils
from scenic.projects.owl_vit import utils
from scenic.train_lib import optax as scenic_optax
from scenic.train_lib import train_utils


def get_train_step(flax_model,
                   loss_and_metrics_fn,
                   config,
                   debug=False):
  """Runs a single step of training.

  Given the state of the training and a batch of data, the train step computes
  the loss and updates the parameters of the model.

  Args:
    flax_model: Flax model (an instance of nn.Module).
    loss_and_metrics_fn: A function that given model predictions, a batch, and
      parameters of the model calculates the loss as well as metrics.
    config: Experiment config dictionary.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Train step function that takes a train_state and batch and returns
    new_train_state and metrics.
  """
  # Get shorthands from the config.
  optax_grad_pmean = config.optimizer.get('optax_grad_pmean', False)
  per_example_clipping = config.optimizer.get('per_example_clipping', False)
  max_grad_norm = config.optimizer.get('max_grad_norm')

  def update_fn(train_state, grad, new_rng):
    step = train_state.global_step

    # In case of per-example gradients, we need to aggregate them after
    # clipping. This is implemented as an Optax tx.
    if not per_example_clipping:
      assert not optax_grad_pmean, ('Optax gradient aggregation should only be'
                                    ' used with per-example gradients.')

      # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
      grad = jax.lax.pmean(grad, axis_name='batch')

    updates, new_opt_state = train_state.tx.update(
        grad, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)

    new_train_state = train_state.replace(
        global_step=step + 1,
        opt_state=new_opt_state,
        params=new_params,
        rng=new_rng)
    return new_train_state, grad

  def train_step(train_state, batch):

    def grad_fn(inputs):
      batch, rng = inputs['batch'], inputs['rng']
      def loss_fn(params):
        # Bind the rng to the host/device we are on.
        model_rng = train_utils.bind_rng_to_host_device(
            rng, axis_name='batch', bind_to='device')
        kwargs = {'text_queries': batch['queries']}

        predictions = flax_model.apply(
            {'params': params, **train_state.model_state},
            batch['inputs'],
            train=True,
            debug=debug,
            rngs={'dropout': model_rng},
            **kwargs)

        return loss_and_metrics_fn(predictions, batch, model_params=params)

      compute_gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (_, metrics), grad = compute_gradient_fn(train_state.params)

      # Note: zero-ing out frozen gradients changes the L2 norm. Clipping is
      # done inside Optax before zero-inng out frozen weights.
      grad = scenic_optax.replace_frozen(config.schedule, grad, 0.)
      metrics['l2_grads_orig'] = (utils.l2_norm(grad), 1)
      return grad, metrics

    if per_example_clipping and max_grad_norm is not None:
      # For per-example clipping we produce per-example rngs.
      rngs = jax.random.split(train_state.rng, num=batch['inputs'].shape[0] + 1)
      new_rng, model_rng = rngs[0], rngs[1:]
      # We add an additional dimension which wil serve as the batch dimension
      # for single examples when applying scan or vmap.
      batch = jax.tree_util.tree_map(lambda x: x[:, jnp.newaxis], batch)
      inp = {'batch': batch, 'rng': model_rng}
      grad, metrics = jax.vmap(grad_fn, 0)(inp)
    else:
      # Without per example clipping we can just compute the gradient on the
      # entire batch.
      new_rng, model_rng = jax.random.split(train_state.rng)
      grad, metrics = grad_fn({'batch': batch, 'rng': model_rng})

    new_train_state, g = update_fn(train_state, grad, new_rng)
    metrics['l2_grads'] = (utils.l2_norm(g), 1)
    metrics['l2_params'] = (utils.l2_norm(new_train_state.params), 1)
    return new_train_state, metrics

  return train_step


def get_eval_step(flax_model,
                  loss_and_metrics_fn,
                  debug=False):
  """Runs a single step of training.

  Note that in this code, the buffer of the second argument (batch) is donated
  to the computation.

  Args:
    flax_model: Flax model (an instance of nn.Module).
    loss_and_metrics_fn: A function that given model predictions, a batch, and
      parameters of the model calculates the loss as well as metrics.
    debug: bool; Whether the debug mode is enabled during evaluation.
      `debug=True` enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Eval step function which returns validation metrics.
  """

  def metrics_fn(train_state, batch, predictions):
    _, metrics = loss_and_metrics_fn(
        predictions, batch, model_params=train_state.params)
    return metrics

  def eval_step(train_state, batch):
    predictions = flax_model.apply(
        {'params': train_state.params, **train_state.model_state},
        batch['inputs'],
        train=False,
        debug=debug,
        text_queries=batch['queries'])
    return metrics_fn(train_state, batch, predictions)

  return eval_step


def train(*, rng: jnp.ndarray, config: ml_collections.ConfigDict,
          model_cls: Any, dataset: dataset_utils.Dataset, workdir: str,
          writer: metric_writers.MetricWriter):
  """Main training loop.

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
      global_step, rng, and the optimizer), train_summary and eval_summary which
      are a dict of metrics.
  """
  lead_host = jax.process_index() == 0
  # The pool is used to perform async evaluation on the CPU.
  pool = futures.ThreadPoolExecutor(max_workers=2)

  # Build the loss_and_metrics_fn, metrics, and flax_model.
  model = model_cls(config, dataset.meta_data)

  # Initialize model.
  rng, init_rng = jax.random.split(rng)

  input_spec = [(dataset.meta_data['input_shape'],
                 dataset.meta_data.get('input_dtype', jnp.float32)),
                (dataset.meta_data['query_shape'], jnp.int32)]

  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=input_spec,
       config=config,
       rngs=init_rng)

  if config.prior_prob:
    params = flax.core.unfreeze(params)
    bias_init = utils.init_classification_bias(
        params['class_head']['logit_shift']['bias'], config.prior_prob)
    params['class_head']['logit_shift']['bias'] = bias_init
    params = flax.core.freeze(params)

  # Create optimizer & LR schedules.
  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  schedule = scenic_optax.make_schedule(config.get('schedule'))
  tx, sched_fns = scenic_optax.make(config.optimizer, schedule, params)
  opt_state = jax.jit(tx.init, backend='cpu')(params)
  sched_fns = [jax.jit(lr_fn, backend='cpu') for lr_fn in sched_fns]

  rng, train_rng = jax.random.split(rng)

  # Create chrono class to track and store training statistics and metadata:
  chrono = train_utils.Chrono()

  train_state = train_utils.TrainState(
      global_step=0,
      params=params,
      tx=tx,
      opt_state=opt_state,
      model_state=model_state,
      rng=train_rng,
      metadata={'chrono': chrono.save()})
  start_step = train_state.global_step

  # Decide how to initialize training. Four options will be tried in this order:
  # 1. Continue training run from an existing checkpoint in the workdir.
  # 2. Resume training from a previous checkpoint, i.e. load both params and
  #    optimizer state (e.g. a cooldown job).
  # 3. Initialize the model parameters from a checkpoint, but not the optimizer
  #    state (e.g. a fine-tuning job).
  # 4. Train from scratch.

  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state)
  chrono.load(train_state.metadata['chrono'])

  if start_step != 0:
    # Option 1:
    logging.info('Continuing from checkpoint in workdir: step=%s, workdir=%s',
                 start_step, workdir)
  else:
    # Option 2: Resume from previous training job:
    if config.get('resume_from') is not None:
      logging.info('Loading params and optimizer: %s', config.init_from)
      checkpoint_path = config.resume_from.get('checkpoint_path')
      train_state = checkpoints.restore_checkpoint(
          checkpoint_path, target=train_state)

    # Option 3: Load only the parameters, e.g. for fine-tuning:
    elif config.get('init_from') is not None:
      logging.info('Loading params: %s', config.init_from)
      init_config = config.init_from.copy_and_resolve_references()
      # Delegate the actual loading to the model. `module.bind()` is needed to
      # initialize submodules, which have their own `load` functions.
      params = model.flax_model.bind({}).load(
          train_state.params.unfreeze(), init_config)
      train_state = train_state.replace(params=flax.core.freeze(params))

    # Option 4: Train from scratch.
    else:
      logging.info('Training from scratch.')

  # Replicate the optimzier, state, and rng.
  train_state = jax_utils.replicate(train_state)
  del params  # do not keep a copy of the initial model

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data)

  train_step = get_train_step(
      flax_model=model.flax_model,
      loss_and_metrics_fn=model.loss_function,
      config=config,
      debug=config.debug_train)

  train_step_pmapped = jax.pmap(
      train_step, axis_name='batch', donate_argnums=(0,))

  ############### EVALUATION CODE #################

  eval_step = get_eval_step(
      flax_model=model.flax_model,
      loss_and_metrics_fn=model.loss_function,
      debug=config.debug_eval)
  eval_step_pmapped = jax.pmap(
      eval_step, axis_name='batch', donate_argnums=(1,))

  # Ceil rounding such that we include the last incomplete batch.
  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  metrics_normalizer_fn = functools.partial(
      utils.normalize_metrics_summary,
      object_detection_loss_keys=model.loss_terms_weights.keys())

  def evaluate(train_state, step, total_steps):
    """Runs evaluation code."""
    # For final evaluation, always run over the entire validation set.
    num_eval_steps = (
        total_eval_steps
        if step == total_steps and not config.get('light_eval', False)
        else steps_per_eval
    )
    eval_metrics = []

    for eval_step in range(num_eval_steps):
      logging.info('Running eval step %d', eval_step)
      eval_batch = next(dataset.valid_iter)

      with jax.profiler.TraceAnnotation('eval_step', step_num=step, _r=1):
        eval_metrics.append(
            train_utils.unreplicate_and_get(
                eval_step_pmapped(train_state, eval_batch)))

    ############### LOG EVAL SUMMARY ###############
    def log_fn(step, eval_metrics, writer,
               metrics_normalizer_fn):
      return train_utils.log_eval_summary(
          step=step,
          eval_metrics=eval_metrics,
          writer=writer,
          metrics_normalizer_fn=metrics_normalizer_fn)

    # Note that we return a Future on a summary instead of the summary itself!
    return pool.submit(
        log_fn,
        step=step,
        eval_metrics=eval_metrics,
        writer=writer,
        metrics_normalizer_fn=metrics_normalizer_fn)

  ###################################################

  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  log_summary_steps = config.get('log_summary_steps', 100)
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps

  train_metrics, extra_training_logs, cpu_training_logs = [], [], []
  train_summary, eval_summary = None, None
  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)

  logging.info('Start training from step %d to %d.', start_step + 1,
               total_steps + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps,
      writer=writer,
      every_secs=None,
      every_steps=log_summary_steps,
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
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)

  write_note(f'First step compilations...\n{chrono.note}')
  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      # Do the train step.
      with jax.profiler.TraceAnnotation('train_step', step_num=step, _r=1):
        train_state, t_metrics = train_step_pmapped(train_state, train_batch)
      # This will accumulate metrics in TPU memory up to the point that we log
      # them. This is no problem for small metrics but may be a problem for
      # large metrics. An alternative is to set `log_summary_steps` to a small
      # number, or to use `train_utils.unreplicate_and_get` here instead of
      # right before writing summaries, but that means in each step, we have
      # data transfer between tpu and host, which might slow down the training.
      train_metrics.append(t_metrics)

    for h in hooks:
      h(step)

    # Additional training logs: time, learning rate, num parameters.
    cpu_training_logs.append({
        f'learning_rate_{name}': lr_fn(step)
        for name, lr_fn in zip(config.schedule.keys(), sched_fns)})

    if ((step % log_summary_steps == 0) or (step == total_steps)
        or (lead_host and chrono.warmup)):
      ############### LOG TRAIN SUMMARY ###############
      chrono.pause(wait_for=(train_metrics))
      if lead_host:
        chrono.tick(step, writer, write_note)
      # Write summary:
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(train_utils.unreplicate_and_get,
                                               train_metrics),
          extra_training_logs=cpu_training_logs + jax.tree_util.tree_map(
              train_utils.unreplicate_and_get, extra_training_logs),
          writer=writer,
          metrics_normalizer_fn=metrics_normalizer_fn)
      # Reset metric accumulation for next evaluation cycle.
      train_metrics, extra_training_logs, cpu_training_logs = [], [], []
      chrono.resume()
      #################################################

    if (step % log_eval_steps == 0) or (step == total_steps):
      chrono.pause(wait_for=(train_state.params))
      start_time = time.time()
      with report_progress.timed('eval'):
        eval_summary = evaluate(train_state, step, total_steps)
      duration = time.time() - start_time
      try:
        ex = eval_summary.exception(1) if eval_summary else None
        if ex is not None:
          logging.error('Failed evaluation: %.4f sec.', duration)
          raise ex   # pylint: disable=raising-bad-type
        logging.info('Done with evaluation: %.4f sec.', duration)
      except futures.TimeoutError:
        pass
      writer.flush()
      if step != total_steps:
        eval_summary = None  # Free up space.
      chrono.resume()

    ##################### CHECKPOINTING ############################
    if ((step % checkpoint_steps == 0 and step > 0) or
        (step == total_steps)) and config.checkpoint:
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))
      with report_progress.timed('checkpoint'):
        if lead_host:
          # Take the first replica.
          unrep_train_state = jax_utils.unreplicate(train_state)
          metadata = unrep_train_state.metadata
          metadata['chrono'] = chrono.save()
          unrep_train_state = unrep_train_state.replace(metadata=metadata)
          train_utils.save_checkpoint(workdir, unrep_train_state)
          del unrep_train_state
      chrono.resume()

  # Wait until computations are done before exiting.
  if eval_summary is not None:
    eval_summary = eval_summary.result()
  pool.shutdown()
  train_utils.barrier()
  return train_state, train_summary, eval_summary
