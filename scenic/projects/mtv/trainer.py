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

"""Training Script for MTV."""

import copy
import functools
import os
from typing import Any, Dict, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
from flax import jax_utils
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.projects.mtv import model as model_lib
from scenic.projects.mtv import train_utils as mtv_train_utils
from scenic.projects.vivit import evaluation_lib
from scenic.projects.vivit import train_utils as vivit_train_utils
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils
import tensorflow as tf


def init_from_mtv_checkpoint(
    config: ml_collections.ConfigDict,
    model: model_lib.MTVClassificationModel,
    train_state: train_utils.TrainState,
) -> train_utils.TrainState:
  """Initialize train state from a MTV checkpoint."""
  if config.init_from.get('model_cfg') is None:
    logging.info('model_cfg is empty. Using current model_cfg.')
    restored_model_cfg = copy.deepcopy(config)
  else:
    restored_model_cfg = config.init_from.model_cfg
  init_checkpoint_path = config.init_from.get('checkpoint_path')
  restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
      init_checkpoint_path, train_state, assert_exist=True)
  return model.init_from_train_state(
      train_state,
      restored_train_state,
      restored_model_cfg,
      restore_output_proj=config.init_from.get('restore_output_projection',
                                               False))


def init_from_vit_checkpoints(
    config: ml_collections.ConfigDict,
    model: model_lib.MTVClassificationModel,
    train_state: train_utils.TrainState,
) -> train_utils.TrainState:
  """Initialize train state from ViT checkpoints."""
  if config.init_from.get('model_cfg') is None:
    logging.info('model_cfg is empty. Use current model\'s classifier.')
    restored_model_cfgs = []
    for _ in range(len(config.model.view_configs)):
      restored_model_cfgs.append(
          ml_collections.ConfigDict(
              {'model': {
                  'classifier': config.model.classifier
              }}))
  else:
    restored_model_cfgs = config.init_from.model_cfg
  init_checkpoint_paths = config.init_from.get('checkpoint_path')
  assert len(init_checkpoint_paths) == len(
      config.model.view_configs
  ), ('Number of initial checkpoint paths must match with the number of view '
      'configs.')
  checkpoint_formats = config.init_from.get('checkpoint_formats')
  checkpoint_formats = (['scenic'] * len(init_checkpoint_paths)
                        if checkpoint_formats is None else checkpoint_formats)
  assert len(checkpoint_formats) == len(
      init_checkpoint_paths
  ), 'The lengths of checkpoint_formats and init_checkpoint_paths must match.'
  assert set(checkpoint_formats).issubset(
      {'scenic',
       'big_vision'}), 'Only scenic and big_vision formats are supported.'
  restored_train_states = []
  for path, checkpoint_format in zip(init_checkpoint_paths, checkpoint_formats):
    if checkpoint_format == 'big_vision':
      restored_train_states.append(
          pretrain_utils.convert_big_vision_to_scenic_checkpoint(
              path, train_state))
    else:
      restored_train_states.append(
          pretrain_utils.restore_pretrained_checkpoint(
              path, train_state, assert_exist=True))
  return model.init_from_vit_train_states(train_state, restored_train_states,
                                          restored_model_cfgs,
                                          checkpoint_formats)


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
  is_multilabel_model = 'multilabel_classification' in config.model_name
  compute_map = is_multilabel_model and config.get('compute_map', False)
  get_confusion_matrix = (config.get('confusion_matrix_metrics', False)
                          and not is_multilabel_model)

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
  optimizer_config = optimizers.get_optax_optimizer_config(config)
  # If the config is already an optax-compatible config, better call directly:
  #   optimizers.get_optimizer(config.optimizer_configs, lr_fn)
  tx = optimizers.get_optimizer(optimizer_config, lr_fn, params=params)
  # We jit this, such that the arrays that are created on the same device as the
  # input is, in this case the CPU. Else they'd be on device[0].
  opt_state = jax.jit(tx.init, backend='cpu')(params)

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
        workdir, train_state)
  chrono.load(train_state.metadata['chrono'])
  del train_state.metadata['chrono']
  if (start_step == 0 and config.get('init_from') is not None):
    model_type = config.init_from.get('model_type', 'mtv')
    if model_type == 'mtv':
      train_state = init_from_mtv_checkpoint(config, model, train_state)
    elif model_type == 'vit':
      train_state = init_from_vit_checkpoints(config, model, train_state)
    else:
      raise ValueError(f'Unknown model type: {model_type}.')
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
          mtv_train_utils.train_step,
          flax_model=model.flax_model,
          lr_fn=learning_rate_fn,
          loss_fn=model.loss_function,
          metrics_fn=model.get_metrics_fn('train'),
          config=config,
          debug=config.debug_train,
      ),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0, 1),
  )
  eval_step_pmapped = jax.pmap(
      functools.partial(
          mtv_train_utils.eval_step,
          flax_model=model.flax_model,
          metrics_fn=model.get_metrics_fn('validation'),
          return_logits_and_labels=compute_map,
          return_confusion_matrix=get_confusion_matrix,
          debug=config.debug_eval,
      ),
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
            mtv_train_utils.test_step,
            flax_model=model.flax_model,
            metrics_fn=model.get_metrics_fn('test'),
            n_clips=config.get('multicrop_clips_per_device', 2),
            debug=config.debug_eval,
        ),
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
  max_checkpoint_keep = config.get('max_checkpoint_keep', 3)
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps

  # Ceil rounding such that we include the last incomplete batch.
  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None

  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)
  logging.info('Starting training loop at step %d.', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps,
      writer=writer,
      every_secs=None,
      every_steps=config.get('report_progress_step', log_summary_steps),
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

  # Manually defragment memory before starting training, if we are using the
  # tfrt runtime.
  do_memory_defrag = False
  if config.get('do_memory_defrag', False):
    client = jax.lib.xla_bridge.get_backend()
    try:
      logging.info('Defragmenting memory')
      client.defragment()
      do_memory_defrag = True
    except RuntimeError:
      logging.warn('Memory defragmentation not possible, use the tfrt runtime')

  write_note(f'First step compilations...\n{chrono.note}')

  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      train_state, t_metrics, t_logs = train_step_pmapped(
          train_state, train_batch
      )
      # This will accumulate metrics in TPU memory up to the point that we log
      # them. This is no problem for small metrics but may be a problem for
      # large (e.g. segmentation) metrics. An alternative is to set
      # `log_summary_steps` to a small number, or to use
      # `train_utils.unreplicate_and_get` here instead of right before writing
      # summaries, but that means in each step, we have data transfer between
      # tpu and host, which might slow down the training.
      train_metrics.append(t_metrics)
      # Additional training logs: learning rate, l2 grads, etc.
      t_logs = jax.tree_util.tree_map(jax_utils.unreplicate, t_logs)
      extra_training_logs.append(t_logs)

    for h in hooks:
      # Catch exception in case XProf fails.
      try:
        h(step)
      except ValueError as error:
        logging.exception('Hook failed: %r', error)

    # Save a pprof after the first step.
    if step == start_step + 1 and lead_host:
      profile = jax.profiler.device_memory_profile()
      with tf.io.gfile.GFile(os.path.join(workdir, 'memory.pprof'), 'wb') as fp:
        fp.write(profile)
    ###################### LOG TRAIN SUMMARY ########################
    if (
        (step % log_summary_steps == 1)
        or (step == total_steps)
        or (lead_host and chrono.warmup)
    ):
      chrono.pause(wait_for=(train_metrics))
      if lead_host:
        chrono.tick(step, writer, write_note)
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(
              train_utils.unreplicate_and_get, train_metrics
          ),
          extra_training_logs=jax.tree_util.tree_map(
              jax.device_get, extra_training_logs
          ),
          writer=writer,
          key_separator='/',
      )
      # Reset metric accumulation for next evaluation cycle.
      train_metrics, extra_training_logs = [], []
      if do_memory_defrag:
        logging.info('Defragmenting memory')
        client.defragment()
      chrono.resume()

    ################### EVALUATION ################################
    if (step % log_eval_steps == 1) or (step == total_steps):
      chrono.pause(wait_for=(train_state.params))
      with report_progress.timed('eval'):
        if do_memory_defrag:
          logging.info('Defragmenting memory')
          client.defragment()

        eval_metrics = []
        additional_summary = None
        if compute_map:
          eval_logits = []
          eval_labels = []
          n_classes = dataset.meta_data['num_classes']
        if get_confusion_matrix:
          confusion_matrices = []
          n_classes = dataset.meta_data['num_classes']

        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        for _ in range(steps_per_eval):
          eval_batch = next(dataset.valid_iter)
          e_metrics = eval_step_pmapped(train_state, eval_batch)
          if compute_map:
            e_metrics, logits_batch, labels_batch = e_metrics
            # TODO(dehghani, lucic): Fetching from the device in each step might
            #  be an unnecessary penalty. Consider updating to async fetching
            #  as in CL/378384754.
            eval_logits.append(vivit_train_utils.to_cpu(logits_batch))
            eval_labels.append(vivit_train_utils.to_cpu(labels_batch))
          if get_confusion_matrix:
            e_metrics, conf_matrix = e_metrics
            confusion_matrices.append(vivit_train_utils.to_cpu(conf_matrix))
          # Fetch e_metrics to host and store.
          eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))

        # Compute global metrics if applicable from all the batches.
        if compute_map:
          additional_summary = evaluation_lib.compute_mean_average_precision(
              np.concatenate(eval_logits, axis=0),
              np.concatenate(eval_labels, axis=0),
              return_per_class_ap=n_classes < 10)
        if get_confusion_matrix:
          additional_summary = evaluation_lib.compute_confusion_matrix_metrics(
              confusion_matrices, return_per_class_metrics=n_classes < 10)
          if lead_host:
            conf_matrix_image = vivit_train_utils.render_confusion_matrices(
                confusion_matrices, normalization_method='rows')
            conf_matrix_unnorm = vivit_train_utils.render_confusion_matrices(
                confusion_matrices, normalization_method='none')

            writer.write_images(
                step, {'valid/conf_matrix': conf_matrix_image,
                       'valid/conf_matrix_unnormalized': conf_matrix_unnorm})

        # Log eval summary.
        eval_summary = train_utils.log_eval_summary(
            step=step,
            eval_metrics=eval_metrics,
            extra_eval_summary=additional_summary,
            writer=writer,
            key_separator='/')
        writer.flush()
        del eval_metrics
        if do_memory_defrag:
          logging.info('Defragmenting memory')
          client.defragment()
      chrono.resume()

    ##################### CHECKPOINTING ###########################
    if ((step % checkpoint_steps == 1 and step > 1) or
        (step == total_steps)) and config.checkpoint:
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))
      with report_progress.timed('checkpoint'):
        train_utils.handle_checkpointing(
            train_state, chrono, workdir, max_checkpoint_keep
        )
      chrono.resume()

    ############# MULTICROP TESTING ############################
    if (config.dataset_configs.get('do_multicrop_test') and
        ((step % log_test_steps == 1 and step > 1) or step == total_steps)):
      chrono.pause(wait_for=(train_state.params))
      with report_progress.timed('test'):
        if do_memory_defrag:
          logging.info('Defragmenting memory')
          client.defragment()

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
            prefix='test',
            key_separator='/')
        logging.info('Completed multicrop test')
        writer.flush()
        # Free up some space.
        del test_metrics
        if do_memory_defrag:
          logging.info('Defragmenting memory')
          client.defragment()
      chrono.resume()

  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary
