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

"""Training Script for ViViT."""

import copy
import functools
from typing import Any, Dict, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import jax_utils
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.projects.vivit import evaluation_lib
from scenic.projects.vivit import train_utils as vivit_train_utils
from scenic.train_lib_deprecated import lr_schedules
from scenic.train_lib_deprecated import optimizers
from scenic.train_lib_deprecated import pretrain_utils
from scenic.train_lib_deprecated import train_utils


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
  flax.config.update('flax_return_frozendict', True)
  lead_host = jax.process_index() == 0
  # Build the loss_fn, metrics, and flax_model.
  model = model_cls(config, dataset.meta_data)
  is_multilabel_model = (config.model_name == 'vivit_multilabel_classification')
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
      restored_train_state = (
          pretrain_utils.convert_big_vision_to_scenic_checkpoint(
              init_checkpoint_path, train_state))
      # Config dict in big_vision is not the same format as scenic.
      # Therefore, make sure config match the config of the loaded model!
      restored_model_cfg = copy.deepcopy(config)
      # The following is needed when the restored and target models used a
      # different classifier. As big_vision uses a different config dict, we
      # have to specify this manually.
      restored_model_cfg.model.classifier = config.init_from.get(
          'classifier_type', 'token')

    train_state = model.init_from_train_state(train_state, restored_train_state,
                                              restored_model_cfg)
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
          vivit_train_utils.train_step,
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
          vivit_train_utils.eval_step,
          flax_model=model.flax_model,
          metrics_fn=model.get_metrics_fn('validation'),
          return_logits_and_labels=is_multilabel_model,
          return_confusion_matrix=get_confusion_matrix,
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
            vivit_train_utils.test_step,
            flax_model=model.flax_model,
            metrics_fn=model.get_metrics_fn('test'),
            n_clips=config.get('multicrop_clips_per_device', 2),
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
      # Catch exception in case XProf fails.
      try:
        h(step)
      except ValueError as error:
        logging.exception('Hook failed: %r', error)

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
      if do_memory_defrag:
        logging.info('Defragmenting memory')
        client.defragment()

    ################### EVALUATION ################################
    if (step % log_eval_steps == 1) or (step == total_steps):
      with report_progress.timed('eval'):
        if do_memory_defrag:
          logging.info('Defragmenting memory')
          client.defragment()

        eval_metrics = []
        additional_summary = None
        if is_multilabel_model:
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
          if is_multilabel_model:
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
        if is_multilabel_model:
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

    ##################### CHECKPOINTING ###########################
    if ((step % checkpoint_steps == 0 and step > 0) or
        (step == total_steps)) and config.checkpoint:
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

    chrono.resume()  # un-pause now
  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary
