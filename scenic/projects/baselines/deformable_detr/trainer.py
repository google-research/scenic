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

"""Training utilities for DeformableDETR."""

from collections.abc import Mapping
from concurrent import futures
import functools
import time
from typing import Any, Callable, Dict, Sequence, Tuple, Type, Union

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import jax_utils
from flax.training.checkpoints import restore_checkpoint as flax_restore_checkpoint
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
import optax
from scenic.dataset_lib import dataset_utils


from scenic.projects.baselines.deformable_detr import coco_eval
from scenic.projects.baselines.deformable_detr import evaluate as ddetr_eval
from scenic.projects.baselines.deformable_detr.model import DeformableDETRModel
from scenic.projects.baselines.detr import train_utils as detr_train_utils
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils
from scenic.train_lib.train_utils import PyTree
from scenic.train_lib.train_utils import TrainState

RngType = Union[jnp.ndarray, Mapping]
InputSpec = Sequence[Union[Tuple[Tuple[int, ...], jnp.dtype], Tuple[int, ...],
                           None]]


def get_optimizer(config: ml_collections.ConfigDict,
                  params: PyTree) -> optax.GradientTransformation:
  """Makes a Optax GradientTransformation for DeformableDETR."""

  # If we freeze the batch statistics for the backbone, the affine
  # transformation of a bn layer can be absorbed by its previous linear
  # transformation and therefore there is no need to train on the affine weights
  # of bn layers.
  def bn_and_freeze_batch_stats(path):
    if not config.freeze_backbone_batch_stats:
      return False
    names = ['/bn1/', '/bn2/', '/bn3/', '/init_bn/', '/proj_bn/']
    for s in names:
      if s in path:
        return True
    return False

  def early_and_load_pretrain(path):
    if not config.load_pretrained_backbone:
      return False
    names = [f'/ResidualBlock_{i}/' for i in range(3)
            ] + ['/init_bn/', '/stem_conv/']
    for s in names:
      if s in path:
        return True
    return False

  backbone_traversal = flax.traverse_util.ModelParamTraversal(
      lambda path, _: 'backbone' in path)
  ref_embed_traversal = flax.traverse_util.ModelParamTraversal(
      lambda path, _: 'ref_embed' in path or 'sampling_offsets' in path)
  bn_traversal = flax.traverse_util.ModelParamTraversal(
      lambda path, _: bn_and_freeze_batch_stats(path))
  early_layer_traversal = flax.traverse_util.ModelParamTraversal(
      lambda path, _: early_and_load_pretrain(path))

  all_false = jax.tree_util.tree_map(lambda _: False, params)

  def get_mask(traversal):
    return traversal.update(lambda _: True, all_false)

  backbone_mask = get_mask(backbone_traversal)
  ref_embed_mask = get_mask(ref_embed_traversal)
  bn_mask = get_mask(bn_traversal)
  early_layer_mask = get_mask(early_layer_traversal)

  oc = config.optimizer_config

  tx = optax.chain(
      optax.clip_by_global_norm(oc.max_grad_norm),
      optax.adamw(
          learning_rate=optax.piecewise_constant_schedule(
              oc.base_learning_rate,
              {oc.learning_rate_decay_event: oc.learning_rate_decay_rate}),
          b1=oc.beta1,
          b2=oc.beta2,
          weight_decay=oc.weight_decay),
      optax.masked(optax.scale(oc.learning_rate_reduction), backbone_mask),
      optax.masked(optax.scale(oc.learning_rate_reduction), ref_embed_mask),
      optax.masked(optax.scale(0), bn_mask),
      optax.masked(optax.scale(0), early_layer_mask))

  return tx


def get_train_step(apply_fn: Callable[..., Tuple[PyTree, PyTree]],
                   loss_and_metrics_fn: Callable[..., Tuple[PyTree, PyTree]],
                   tx: optax.GradientTransformation,
                   update_batch_stats: bool = False,
                   debug: bool = False):
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Note that in this code, the buffers of the first (train_state) argument is
  donated to the computation.

  Args:
    apply_fn: Model application function.
    loss_and_metrics_fn: Function to calculate loss and metrics.
    tx: An optax.GradientTransformation.
    update_batch_stats: Whether to update batch statistics in BatchNorm
      during training or freeze it.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
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
         new_rng)), grads = jax.value_and_grad(
             loss_fn, has_aux=True)(
                 train_state.params)

    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
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
    if 'param' in train_state['params']:
      train_state['params'] = train_state['params']['param']
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


def get_model_and_tx_and_train_state(rng: RngType,
                                     dataset: dataset_utils.Dataset,
                                     config: ml_collections.ConfigDict,
                                     model_cls: Type[DeformableDETRModel],
                                     workdir: str, input_spec: InputSpec):
  """Create model and train state."""
  # Build the loss_and_metrics_fn, metrics, and flax_model.
  model = model_cls(config, dataset.meta_data)

  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=input_spec,
       config=config,
       rngs=init_rng)

  tx = get_optimizer(config, params)

  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0]
  opt_state = jax.jit(tx.init, backend='cpu')(params)

  rng, train_rng = jax.random.split(rng)
  train_state = TrainState(
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
    logging.info('Init from checkpoint: %s', init_checkpoint_path)
    restored_train_state = flax_restore_checkpoint(
        init_checkpoint_path, target=None)
    _handle_legacy_format(restored_train_state)

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
        train_state, bb_train_state)

  # Replicate the optimzier, state, and rng.
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  return model, tx, train_state, num_trainable_params, gflops, start_step


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

  input_spec = [(dataset.meta_data['input_shape'],
                 dataset.meta_data.get('input_dtype', jnp.float32))]
  (model, tx, train_state, num_trainable_params, gflops,
   start_step) = get_model_and_tx_and_train_state(
       rng, dataset, config, model_cls, workdir, input_spec=input_spec)

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
      loss_and_metrics_fn=model.loss_and_metrics_function,
      tx=tx,
      update_batch_stats=update_batch_stats,
      debug=config.debug_train)

  train_step_pmapped = jax.pmap(
      train_step, axis_name='batch', donate_argnums=(0,))

  ############### EVALUATION CODE #################
  eval_step = ddetr_eval.get_eval_step(
      flax_model=model.flax_model,
      loss_and_metrics_fn=model.loss_and_metrics_function,
      logits_to_probs_fn=model.logits_to_probs,
      debug=config.debug_eval)
  eval_step_pmapped = jax.pmap(eval_step, axis_name='batch')

  # Ceil rounding such that we include the last incomplete batch.
  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  metrics_normalizer_fn = functools.partial(
      detr_train_utils.normalize_metrics_summary,
      object_detection_loss_keys=model.loss_terms_weights.keys())

  global_metrics_evaluator = None  # Only run eval on the lead_host node.
  if lead_host:
    global_metrics_evaluator = coco_eval.DeformableDetrGlobalEvaluator(
        config.dataset_name)

  ###################################################

  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  log_summary_steps = config.get('log_summary_steps', 25)
  log_large_summary_steps = config.get('log_large_summary_steps', 0)
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps

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

  # Do eval before first train step.
  if config.get('do_eval_first', False):
    # Sync model state across replicas (in case of having model state, e.g.
    # batch statistic when using batch norm).
    start_time = time.time()
    with report_progress.timed('eval'):
      train_state = train_utils.sync_model_state_across_replicas(train_state)
      (last_eval_step,
       last_eval_metrics), last_eval_future = ddetr_eval.run_eval(
           global_metrics_evaluator, dataset, train_state, eval_step_pmapped,
           pool, 0, steps_per_eval)
    duration = time.time() - start_time
    logging.info('Done with async evaluation: %.4f sec.', duration)
    writer.flush()

  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      train_state, t_metrics, train_predictions = train_step_pmapped(
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

    if (step % log_summary_steps == 0) or (step == total_steps):
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
        eval_summary = train_utils.log_eval_summary(
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
        (last_eval_step,
         last_eval_metrics), last_eval_future = ddetr_eval.run_eval(
             global_metrics_evaluator, dataset, train_state, eval_step_pmapped,
             pool, step, steps_per_eval)
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
          train_utils.save_checkpoint(workdir,
                                      jax_utils.unreplicate(train_state))

  # Wait until computations are done before exiting.
  pool.shutdown()
  if last_eval_future is not None:
    eval_summary = train_utils.log_eval_summary(
        step=last_eval_step,
        eval_metrics=last_eval_metrics,
        extra_eval_summary=last_eval_future.result(),
        writer=writer,
        metrics_normalizer_fn=metrics_normalizer_fn)
  train_utils.barrier()
  return train_state, train_summary, eval_summary


def evaluate(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Any,
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Dict[str, Any]:
  """Eval only loop.

  Given the model class and dataset, it prepares the items needed to run the
  evaluation without training.

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
    Eval summary.
  """
  lead_host = jax.process_index() == 0

  # The pool is used to perform misc operations such as logging in async way.
  pool = futures.ThreadPoolExecutor(max_workers=2)

  input_spec = [(dataset.meta_data['input_shape'],
                 dataset.meta_data.get('input_dtype', jnp.float32))]
  (model, _, train_state, num_trainable_params, gflops,
   _) = get_model_and_tx_and_train_state(
       rng, dataset, config, model_cls, workdir, input_spec=input_spec)

  step0_log = {'num_trainable_params': num_trainable_params}
  if gflops:
    step0_log['gflops'] = gflops
  writer.write_scalars(1, step0_log)

  report_progress = periodic_actions.ReportProgress(
      writer=writer,
      every_secs=None,
      every_steps=config.get('report_progress_step', 10),
  )
  hooks = []
  if lead_host:
    hooks.append(report_progress)
  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  eval_step = ddetr_eval.get_eval_step(
      flax_model=model.flax_model,
      loss_and_metrics_fn=model.loss_and_metrics_function,
      logits_to_probs_fn=model.logits_to_probs,
      debug=config.debug_eval)
  eval_step_pmapped = jax.pmap(
      eval_step, axis_name='batch', donate_argnums=(1,))

  # Ceil rounding such that we include the last incomplete batch.
  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps
  logging.info('steps_per_eval: %s', steps_per_eval)

  metrics_normalizer_fn = functools.partial(
      detr_train_utils.normalize_metrics_summary,
      object_detection_loss_keys=model.loss_terms_weights.keys())

  global_metrics_evaluator = None  # Only run eval on the lead_host node.
  if lead_host:
    global_metrics_evaluator = coco_eval.DeformableDetrGlobalEvaluator(
        config.dataset_name)

  # Sync model state across replicas (in case of having model state, e.g.
  # batch statistic when using batch norm).
  start_time = time.time()
  with report_progress.timed('eval'):
    train_state = train_utils.sync_model_state_across_replicas(train_state)
    (last_eval_step, last_eval_metrics), last_eval_future = ddetr_eval.run_eval(
        global_metrics_evaluator=global_metrics_evaluator,
        dataset=dataset,
        train_state=train_state,
        eval_step_pmapped=eval_step_pmapped,
        pool=pool,
        step=0,
        steps_per_eval=steps_per_eval)
  duration = time.time() - start_time
  logging.info('Done with async evaluation: %.4f sec.', duration)
  writer.flush()

  # Wait until computations are done before exiting.
  pool.shutdown()

  assert last_eval_future is not None
  eval_summary = train_utils.log_eval_summary(
      step=last_eval_step,
      eval_metrics=last_eval_metrics,
      extra_eval_summary=last_eval_future.result(),
      writer=writer,
      metrics_normalizer_fn=metrics_normalizer_fn)
  train_utils.barrier()

  logging.info('Eval Summary: %s.', eval_summary)
  return eval_summary
