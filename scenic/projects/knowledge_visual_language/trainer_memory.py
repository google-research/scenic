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

"""Training Script for knowledge-based models."""

import functools
import gc
from typing import Any, Dict, Optional, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
from flax import jax_utils
import flax.linen as nn
from flax.training import common_utils
import jax
import jax.example_libraries.optimizers as jax_optimizers
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
import optax
from scenic.dataset_lib import dataset_utils
from scenic.xm import xm_utils
from scenic.model_lib.base_models import base_model
from scenic.projects.knowledge_visual_language import trainer_utils
from scenic.projects.knowledge_visual_language.models import constants
from scenic.projects.knowledge_visual_language.models import local_memory
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils

local_kb = local_memory.kb


def init_state(
    model: base_model.BaseModel,
    dataset: dataset_utils.Dataset,
    config: ml_collections.ConfigDict,
    workdir: str,
    rng: jnp.ndarray,
    writer: metric_writers.MetricWriter,
):
  """Initialize the train state."""

  input_spec = {
      key[:-5]: dataset.meta_data[key]
      for key in dataset.meta_data
      if key[-5:] == '_spec'
  }
  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  params, model_state, num_params, gflops = (
      train_utils.initialize_model_with_pytree(
          model_def=model.flax_model,
          input_spec=input_spec,
          config=config,
          rngs=init_rng,
      )
  )
  logging.info('The model has %d params', num_params)

  if gflops:
    logging.info('uses %d gflops', gflops or -1)
  lr_fn = lr_schedules.get_learning_rate_fn(config)
  # Create the optimizer.
  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  tx = optimizers.get_optimizer(
      optimizer_config=optimizers.get_optax_optimizer_config(config),
      learning_rate_fn=lr_fn,
  )
  tx = trainer_utils.froze_param_optax(
      params=params,
      tx=tx,
      frozen_patterns=config.get('frozen_patterns', None),
      not_frozen_patterns=config.get('not_frozen_patterns', None),
  )
  opt_state = jax.jit(tx.init, backend='cpu')(params)

  _, train_rng = jax.random.split(rng)
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
  if config.checkpoint:
    logging.info('Continuing training from the checkpoint')
    logging.info('workdir= %s', workdir)
    train_state, params_axes = trainer_utils.pop_axes_names(
        train_state, axes_name='params_axes'
    )
    train_state, memory_axes = trainer_utils.pop_axes_names(
        train_state, axes_name='memory'
    )
    train_state, _ = train_utils.restore_checkpoint(workdir, train_state)
    train_state = trainer_utils.re_add_axis_names(
        train_state, memory_axes, axes_name='memory'
    )
    train_state = trainer_utils.re_add_axis_names(
        train_state, params_axes, axes_name='params_axes'
    )

  start_step = int(train_state.global_step)
  chrono.load(train_state.metadata['chrono'])

  if start_step == 0:
    if config.get('init_from', False):
      if config.init_from.get('resume', False):
        xid, wid = config.init_from.get('resume')
        (_, workdir) = xm_utils.get_info_from_xmanager(xid, wid)
        logging.info('Resuming training from the checkpoint')
        logging.info('workdir= %s', workdir)
        train_state, params_axes = trainer_utils.pop_axes_names(
            train_state, axes_name='params_axes'
        )
        train_state, memory_axes = trainer_utils.pop_axes_names(
            train_state, axes_name='memory'
        )
        train_state, _ = train_utils.restore_checkpoint(workdir, train_state)
        train_state = trainer_utils.re_add_axis_names(
            train_state, memory_axes, axes_name='memory'
        )
        train_state = trainer_utils.re_add_axis_names(
            train_state, params_axes, axes_name='params_axes'
        )
        start_step = int(train_state.global_step)
        chrono.load(train_state.metadata['chrono'])

      elif config.init_from.get('xm', False):
        if config.init_from.load_key_encoder:
          params = trainer_utils.load_key_params(params, config)
          train_state = train_state.replace(  # pytype: disable=attribute-error
              params=params
          )
      else:
        logging.info('Loading T5 & ViT Parameter from Pre-Trained Model')
        params = trainer_utils.load_visual_params(params, config)
        params = trainer_utils.load_text_params(params, config)
        train_state = train_state.replace(  # pytype: disable=attribute-error
            params=params
        )
    step0_log = {'num_trainable_params': num_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)
  train_state, _ = trainer_utils.pop_axes_names(
      train_state, axes_name='params_axes'
  )
  return train_state, start_step, chrono, lr_fn


def train_step(
    train_state: train_utils.TrainState,
    batch: constants.Batch,
    *,
    flax_model: nn.Module,
    loss_fn: constants.LossFn,
    metrics_fn: constants.MetricFn,
    model_config: ml_collections.ConfigDict,
    debug: Optional[bool] = False,
) -> Tuple[
    train_utils.TrainState,
    Dict[str, Tuple[float, int]],
    Dict[str, float],
    constants.JTensor,
]:
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
    loss_fn: A loss function that given logits and batch of data, calculates the
      training loss.
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    model_config: Config for model architecture.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training, loss, computed metrics, and learning rate for
    logging.
  """
  new_rng, rng = jax.random.split(train_state.rng)

  # Bind the rng to the host/device we are on for dropout.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device'
  )

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    logging.info(variables.keys())
    output_dict = flax_model.apply(
        variables,
        **batch,
        train=True,
        rngs={'dropout': dropout_rng},
        debug=debug,
        use_memory=True,
        use_psudo_retr=model_config.use_psudo_retr,
        retrieve_local=model_config.retrieve_local,
        frozen_base=model_config.t5_frozen_base,
    )
    if debug:
      logging.info(
          'Shape of token_logits in train step is: %s',
          output_dict['predicted_logits'].shape,
      )
    loss_dict = loss_fn(output_dict, batch)
    return loss_dict['total_loss'], (output_dict, loss_dict)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)

  (train_loss, (output_dict, loss_dict)), grad = compute_gradient_fn(
      train_state.params
  )
  grad = jax.lax.pmean(grad, axis_name='batch')
  grad_norm = jax_optimizers.l2_norm(grad).astype(float)
  updates, new_opt_state = train_state.tx.update(
      grad, train_state.opt_state, train_state.params
  )
  update_norm = jax_optimizers.l2_norm(updates).astype(float)
  new_params = optax.apply_updates(train_state.params, updates)
  param_norm = jax_optimizers.l2_norm(train_state.params).astype(float)

  logging.info(output_dict['predicted_logits'].shape)
  logging.info(batch)
  metrics = metrics_fn(output_dict['predicted_logits'], batch)

  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      params=new_params,
      rng=new_rng,
  )

  logs = {
      'train/train_loss': train_loss,
      'train/gen_loss': loss_dict['gen_loss'],
      'train/contra_loss': loss_dict['contra_loss'],
      'train/contra_accs': loss_dict['contra_accs'],
      'grad_norm': grad_norm,
      'update_norm': update_norm,
      'param_norm': param_norm,
      'bias': train_state.params['att_transform']['bias'][0],
      'scale': train_state.params['att_transform']['scale'][0],
  }

  if 'retr_scores' in output_dict:
    logs['train/a0'] = output_dict['retr_scores'][0][0]
    logs['train/a1'] = output_dict['retr_scores'][0][1]
    logs['train/a2'] = output_dict['retr_scores'][0][2]
    logs['train/a3'] = output_dict['retr_scores'][0][3]

  if 'topk_scores' in output_dict:
    logs['train/s0'] = output_dict['topk_scores'][0][0]
    logs['train/s1'] = output_dict['topk_scores'][0][1]

  if 'inbatch_sim' in output_dict:
    logs['train/i0'] = output_dict['inbatch_sim'][0][0]
    logs['train/i1'] = output_dict['inbatch_sim'][0][1]

  if 'base_norm' in output_dict:
    logs['train/base_norm'] = output_dict['base_norm']
  if 'data_norm' in output_dict:
    logs['train/data_norm'] = output_dict['data_norm']
  if 'vals_norm' in output_dict:
    logs['train/memory_norm'] = output_dict['vals_norm']
  if 'disentangle_reg' in output_dict:
    logs['train/disentangle'] = output_dict['disentangle_reg']
  if 'gap' in output_dict:
    logs['train/gap'] = output_dict['gap']
  retr_top_image = output_dict['retr_data']['image'][:, 0]
  return new_train_state, metrics, logs, retr_top_image  # pytype: disable=bad-return-type  # jax-types


def eval_step(
    train_state: train_utils.TrainState,
    batch: constants.Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: constants.MetricFn,
    model_config: ml_collections.ConfigDict,
    debug: Optional[bool] = False,
) -> Tuple[Dict[str, Tuple[float, int]], constants.JTensor, Any]:
  """Runs a single step of evaluation.

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
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    model_config: Config for model architecture.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training, loss, computed metrics, and learning rate for
    logging.
  """
  variables = {'params': train_state.params, **train_state.model_state}
  output_dict = flax_model.apply(
      variables,
      **batch,
      train=False,
      debug=debug,
      use_memory=True,
      retrieve_local=model_config.retrieve_local,
  )
  retr_top_image = output_dict['retr_data']['image'][:, 0]
  metrics = metrics_fn(output_dict['predicted_logits'], batch)
  return metrics, retr_top_image, output_dict['predicted_logits']


def eval_step_autoregressive_decoding(
    train_state: train_utils.TrainState,
    batch: constants.Batch,
    *,
    model: Any,
    metrics_fn: Any,
    model_config: ml_collections.ConfigDict,
    vocab_size: int,
    num_decodes: int,
    beam_search: bool = True,
    debug: Optional[bool] = False,
) -> Tuple[Dict[str, Tuple[float, int]], constants.JTensor, Any]:
  """Evaluate autoregressive generation.

  Args:
    train_state: The state of training including the current global_step,
      model_state, rng, and optimizer. The buffer of this argument are donated
      to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    model: The scenic model.
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    model_config: Config for model architecture.
    vocab_size: size of the vocabulary.
    num_decodes: number of decoding attempts. A larger number means longer
      inference time but better performance.
    beam_search: if True, perform beam search. If False, perform temperature
      sampling.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    metrics: dictionary mapping metrics to values.
    logits: predicted logits of the model.
  """
  variables = {'params': train_state.params, **train_state.model_state}

  # Loop per example since decoding only works for single examples.
  predicted_tokens, _, retr_top_image = (
      model.apply_with_autoregressive_decoding(
          variables,
          **batch,
          num_decodes=num_decodes,
          beam_search=beam_search,
          debug=debug,
          use_memory=True,
          retrieve_local=model_config.retrieve_local,
      )
  )

  # The autoregressive decoder yields tokens. However, the metrics want
  # logits. So make the predictions into one-hot predictions.
  logits = common_utils.onehot(predicted_tokens, vocab_size)
  metrics = metrics_fn(logits, batch)

  return metrics, retr_top_image, logits


def train_and_eval(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Any,
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
    kb_datasets: Dict[str, dataset_utils.Dataset],
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
    kb_datasets: dictionary of datasets served as knowledge base.

  Returns:
    A tuple with:
      * the state that has the state of training (including current
        global_step, model_state, rng, and the optimizer)
      * a dictionary with the train_summary
      * a dictionary with the evaluation summary
  """
  host_id = jax.process_index()
  lead_host = host_id == 0
  # Build the loss_fn, metrics, and flax_model.

  model = model_cls(config, dataset.meta_data, kb_datasets=kb_datasets)
  train_state, start_step, chrono, lr_fn = init_state(
      model, dataset, config, workdir, rng, writer
  )

  logging.info('Complete initialization for %s. Start Training.', host_id)
  train_state = jax_utils.replicate(train_state)
  local_kb.set_encode_fn(flax_model=model.flax_model)
  train_state = local_kb.update_memory(
      train_state,
      bsz=config.batch_size,
      data_k=model.data_k,
      retr_k=model.retr_k,
      axis_index_groups=model.axis_index_groups,
  )
  logging.info('Number of processes is %s', jax.process_count())

  # Get the pmapped train and eval steps.
  train_step_pmapped = jax.pmap(
      functools.partial(
          train_step,
          flax_model=model.flax_model,
          loss_fn=model.loss_function_dict,
          metrics_fn=model.get_metrics_fn('train'),
          debug=config.debug_train,
          model_config=config.model,
      ),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0, 1),
  )
  eval_step_pmapped = jax.pmap(
      functools.partial(
          eval_step,
          flax_model=model.flax_model,
          metrics_fn=model.get_metrics_fn('validation'),
          debug=config.debug_eval,
          model_config=config.model,
      ),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1),
  )
  train_summary, eval_summary, e_metrics = {}, {}, {}
  eval_step_autoregressive_decoding_pmapped = jax.pmap(
      functools.partial(
          eval_step_autoregressive_decoding,
          model=model,
          metrics_fn=model.get_metrics_fn('validation'),
          model_config=config.model,
          vocab_size=config.vocab_size,
          num_decodes=config.autoregressive_decoding.num_decodes,
          beam_search=config.autoregressive_decoding.beam_search,
          debug=config.debug_eval,
      ),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1),
  )

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data
  )
  log_eval_steps = config.get('log_eval_steps', steps_per_epoch)
  checkpoint_steps = config.get('checkpoint_steps', steps_per_epoch)
  log_summary_steps = config.get('log_summary_steps', log_eval_steps)
  frozen_memory = config.get('frozen_memory', False)

  logging.info('Start training from step %d', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer
  )
  hooks = [report_progress]

  # if config.get('xprof', True) and lead_host:
  #  hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  chrono.inform(
      int(start_step),
      int(total_steps),
      int(config.batch_size),
      int(steps_per_epoch),
  )

  summary_builder = trainer_utils.SummaryBuilder([], [])

  def write_note(note):
    if lead_host:
      platform.work_unit().set_notes(note)

  for step in range(start_step + 1, total_steps + 1):
    if lead_host:
      logging.info('training for step %d', step)
    ###################### Training ########################
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      inp_image_sample = np.asarray(train_batch['encoder_input_image'][0][0])
      train_batch['device_id'] = jnp.arange(
          train_batch['encoder_input_image'].shape[0]
      )
      train_state, t_metrics, logs, retr_top_image = train_step_pmapped(
          train_state, train_batch
      )
      retr_top_image = np.asarray(retr_top_image[0][0])
    for hook in hooks:
      hook(step)
    logs['learning_rate'] = lr_fn(step).reshape([-1, 1])
    summary_builder.update(metrics_update=t_metrics, extra_logs_update=logs)
    jax.tree_util.tree_map(lambda h: h.delete(), train_batch)
    if lead_host:
      logging.info('finish training for step %d', step)
    ###################### LOG TRAIN SUMMARY ########################
    if (step % log_summary_steps == 2) or (step == total_steps):
      chrono.pause()
      if lead_host:
        logging.info('log training summary for step %d', step)
        chrono.tick(step, writer, write_note)
        writer.write_images(
            step,
            {
                'train/retr_image': np.expand_dims(retr_top_image, axis=0),
                'train/query_image': np.expand_dims(inp_image_sample, axis=0),
            },
        )
      writer.flush()
      train_summary = summary_builder.write(writer, step)
      chrono.resume()
    del inp_image_sample, retr_top_image
    ################### EVALUATION ################################
    should_eval = (step % log_eval_steps == 2) or (step == total_steps)
    if should_eval:
      # update the KB memory
      if not frozen_memory:
        train_state = local_kb.update_memory(
            train_state,
            bsz=config.batch_size,
            data_k=model.data_k,
            retr_k=model.retr_k,
            axis_index_groups=model.axis_index_groups,
        )
      logging.info('Start validation!')
      chrono.pause(wait_for=(train_state.params))
      # Sync model state across replicas.
      for ds_name in dataset.valid_iter:
        logging.info('Validate on %s', ds_name)
        # Compute the number of evaluation steps per dataset.
        num_eval_examples = dataset.meta_data['num_eval_examples'][ds_name]
        total_eval_steps = int(
            np.ceil(num_eval_examples / (config.get('eval_batch_size')))
        )
        steps_per_eval = config.get('steps_per_eval', total_eval_steps)
        for test_mode in range(2):
          eval_metrics_all = []
          eval_vqa_metrics_all = {}
          for step_id in range(steps_per_eval):
            eval_batch = next(dataset.valid_iter[ds_name])
            inp_image_sample = np.asarray(
                eval_batch['encoder_input_image'][0][0]
            )
            if test_mode == 0:
              eval_batch['device_id'] = jnp.arange(
                  eval_batch['encoder_input_image'].shape[0]
              )
            e_metrics, retr_top_image, predicted_logits = eval_step_pmapped(
                train_state=train_state, batch=eval_batch
            )
            eval_metrics_all.append(train_utils.unreplicate_and_get(e_metrics))
            # add vqa accuracy metric if it's a vqa dataset
            if config.model.get('qa', False):
              eval_vqa_metric_fn = model.get_vqa_metrics(
                  predicted_logits, eval_batch
              )
              eval_vqa_metrics = eval_vqa_metric_fn.compute()
              for key in eval_vqa_metrics:
                eval_vqa_metrics_all.setdefault(key, 0.0)
                eval_vqa_metrics_all[key] += eval_vqa_metrics[key]

            if test_mode == 0 and step_id == 0 and lead_host:
              logging.info(e_metrics)
              if ds_name == 'val_cc':
                retr_top_image = np.asarray(retr_top_image[0][0])
                writer.write_images(
                    step,
                    {
                        'eval/retr_image': np.expand_dims(
                            retr_top_image, axis=0
                        ),
                        'eval/query_image': np.expand_dims(
                            inp_image_sample, axis=0
                        ),
                    },
                )
                writer.flush()
                del retr_top_image
            jax.tree_util.tree_map(lambda h: h.delete(), eval_batch)
            del inp_image_sample
          for key in eval_vqa_metrics_all:
            eval_vqa_metrics_all[key] /= float(steps_per_eval)
          if test_mode == 0:
            eval_summary = train_utils.log_eval_summary(
                step=step,
                eval_metrics=eval_metrics_all,
                extra_eval_summary=eval_vqa_metrics_all,
                writer=writer,
                prefix=ds_name,
            )
          else:
            eval_summary = train_utils.log_eval_summary(
                step=step,
                eval_metrics=eval_metrics_all,
                extra_eval_summary=eval_vqa_metrics_all,
                writer=writer,
                prefix='random_retrieve_' + ds_name,
            )
        # autoregressive eval
        eval_metrics_all = []
        eval_vqa_metrics_all = {}
        for step_id in range(steps_per_eval):
          eval_batch = next(dataset.valid_iter[ds_name])
          eval_batch['device_id'] = jnp.arange(
              eval_batch['encoder_input_image'].shape[0]
          )
          logging.log_first_n(
              logging.INFO, 'Peforming eval with autoregressive decode', 3
          )
          e_metrics, retr_top_image, predicted_logits = (
              eval_step_autoregressive_decoding_pmapped(
                  train_state=train_state, batch=eval_batch
              )
          )
          eval_metrics_all.append(train_utils.unreplicate_and_get(e_metrics))
          # add vqa accuracy metric if it's a vqa dataset
          if config.model.get('qa', False):
            eval_vqa_metric_fn = model.get_vqa_metrics(
                predicted_logits, eval_batch
            )
            eval_vqa_metrics = eval_vqa_metric_fn.compute()
            for key in eval_vqa_metrics:
              eval_vqa_metrics_all.setdefault(key, 0.0)
              eval_vqa_metrics_all[key] += eval_vqa_metrics[key]
        for key in eval_vqa_metrics_all:
          eval_vqa_metrics_all[key] /= float(steps_per_eval)
        eval_summary = train_utils.log_eval_summary(
            step=step,
            eval_metrics=eval_metrics_all,
            extra_eval_summary=eval_vqa_metrics_all,
            writer=writer,
            prefix=ds_name + '_autoregressive',
        )

      chrono.resume()
      gc.collect()
    ##################### CHECKPOINTING ###########################
    if not config.checkpoint:
      continue
    elif step % checkpoint_steps == 0 and step > 0 or (step == total_steps):
      logging.info('Save checkpoint!')
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))
      with report_progress.timed('checkpoint'):
        if lead_host:
          logging.info('checkpointing (training step: %d)', step)
          # Take the first replica.
          unrep_train_state = jax_utils.unreplicate(train_state)
          unrep_train_state, mem = trainer_utils.pop_axes_names(
              unrep_train_state, axes_name='memory'
          )
          metadata = unrep_train_state.metadata
          metadata['chrono'] = chrono.save()
          unrep_train_state.replace(metadata=metadata)  # pytype: disable=attribute-error
          train_utils.save_checkpoint(workdir, unrep_train_state)
          del unrep_train_state, mem
      chrono.resume()
      logging.info('Checkpoint saved!')

  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()

  # Return the train and eval summary after last step.
  return train_state, train_summary, eval_summary
