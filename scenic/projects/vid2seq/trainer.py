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

"""Dense Video Captioning training."""

import copy
import dataclasses
import functools
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from dmvr import tokenizers
from flax import jax_utils
from flax.core import unfreeze
import flax.linen as nn
import jax
import jax.example_libraries.optimizers as jax_optimizers
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import base_model
from scenic.projects.t5 import model as t5_model
from scenic.projects.t5 import tokenizer as t5_tokenizer
from scenic.projects.vid2seq import load_utils
from scenic.projects.vid2seq import models
from scenic.projects.vid2seq import train_utils as vid2seq_train_utils
from scenic.projects.vid2seq import dvc_eval
from scenic.train_lib_deprecated import lr_schedules
from scenic.train_lib_deprecated import optimizers
from scenic.train_lib_deprecated import pretrain_utils
from scenic.train_lib_deprecated import train_utils

# Note this list must be in the exact order of the inputs required by the model.
MAX_CAPTION_STR_LEN = 200
MAX_KEY_STR_LEN = 400

# Aliases for custom types:
PyTree = Any
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Batch], Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch], float]


def remove_nonascii(text):
  return ''.join([i if ord(i) < 128 else ' ' for i in text])


def decode_tokens(seq, tokenizer, vocabulary_size):
  seq = [x for x in seq if x < vocabulary_size]
  text = tokenizer.indices_to_string(seq)
  text = remove_nonascii(text).strip()
  return text


def decode_time(time, duration, fmt):
  if fmt == 'cd':
    time = [
        time[0] - time[1] // 2,
        time[0] + time[1] // 2
        ]
  time[0] = max(time[0], 0)
  time[1] = min(time[1], duration)
  return time


def train_step(
    dataset: str,  # pylint: disable=unused-argument
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    learning_rate_fn: Callable[[int], float],
    loss_fn: LossFn,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False  # pylint: disable=unused-argument
) -> Tuple[train_utils.TrainState, float, Dict[str, Tuple[float, int]], float]:
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Note that in this code, the buffers of the first (train_state) and second
  (batch) arguments are donated to the computation.

  Args:
    dataset: dataset name
    train_state: The state of training including the current global_step,
      model_state, rng, and optimizer. The buffer of this argument can be
      donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    flax_model: A Flax model.
    learning_rate_fn: learning rate scheduler which give the global_step
      generates the learning rate.
    loss_fn: A loss function that given logits and batch of data, calculates the
      training loss.
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    config: Configuration of the experiment.
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
      rng, axis_name='batch', bind_to='device')

  token_loss_coef = config.get('token_loss_coef')
  corrupt_coef = config.dataset_configs.get(
      'corrupt_coef') if config.dataset_configs.corrupt else 0.
  return_as_dict = config.dataset_configs.return_as_dict
  modalities = config.dataset_configs.modalities

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    assert return_as_dict
    encoder_inputs = batch['encoder_inputs']

    # encode video
    if 'features' in modalities:
      enc_video, _ = flax_model.apply(
          variables, {'features': encoder_inputs['features']},
          mutable=['batch_stats'],
          train=True,
          rngs={'dropout': dropout_rng},
          method=flax_model.encode)
    # encode speech
    if 'text' in modalities and 'text' in encoder_inputs:
      enc_text, _ = flax_model.apply(
          variables, {'text': encoder_inputs['text']},
          mutable=['batch_stats'],
          train=True,
          rngs={'dropout': dropout_rng},
          method=flax_model.encode)

    # concat video and text encodings
    if 'features' in modalities and 'text' in modalities and (
        'text'
        in encoder_inputs):
      encoded = jnp.concatenate([enc_video['encoded'], enc_text['encoded']], -2)
      mask = jnp.concatenate([enc_video['mask'], enc_text['mask']], -1)
      encoded = {'encoded': encoded, 'mask': mask}
    elif 'features' in modalities:
      encoded = enc_video
    elif 'text' in modalities and 'text' in encoder_inputs:
      encoded = enc_text

    loss = 0.
    aux = {}

    if token_loss_coef:
      token_encoded = encoded
      token_decoder_inputs = {
          'encoder_input_tokens': token_encoded['mask'],
          'decoder_input_tokens': batch['text_indices'][..., :-1],
          'decoder_target_tokens': batch['text_indices'][..., 1:]
      }
      token_logits, new_model_state = flax_model.apply(
          variables,
          token_encoded['encoded'],
          token_decoder_inputs,
          mutable=['batch_stats'],
          train=True,
          rngs={'dropout': dropout_rng},
          method=flax_model.decode)

      loss_token = loss_fn(token_logits,  # pytype: disable=wrong-arg-types  # jax-ndarray
                           {'decoder_inputs': token_decoder_inputs})
      aux['token_logits'] = token_logits
      aux['token_loss'] = loss_token
      loss += loss_token * token_loss_coef

    if corrupt_coef:
      corrupt_encoder_inputs = {'text': batch['text_indices_corrupt_inputs']}

      # encode text
      out_text, _ = flax_model.apply(
          variables,
          corrupt_encoder_inputs,
          mutable=['batch_stats'],
          train=True,
          rngs={'dropout': dropout_rng},
          method=flax_model.encode)
      if 'features' in modalities:
        corrupt_encoded = jnp.concatenate([
            encoded['encoded'][..., :batch['features'].shape[1], :],
            out_text['encoded']
        ], 1)
        corrupt_mask = jnp.concatenate([
            encoded['mask'][..., :batch['features'].shape[1]], out_text['mask']
        ], 1)
        corrupt_encoded = {'encoded': corrupt_encoded, 'mask': corrupt_mask}
      else:
        corrupt_encoded = out_text

      corrupt_decoder_inputs = {
          'encoder_input_tokens':
              corrupt_encoded['mask'],
          'decoder_input_tokens':
              batch['text_indices_corrupt_outputs'][..., :-1],
          'decoder_target_tokens':
              batch['text_indices_corrupt_outputs'][..., 1:]
      }

      # decode
      corrupt_logits, new_model_state = flax_model.apply(
          variables,
          corrupt_encoded['encoded'],
          corrupt_decoder_inputs,
          mutable=['batch_stats'],
          train=True,
          rngs={'dropout': dropout_rng},
          method=flax_model.decode)
      loss_corrupt = loss_fn(corrupt_logits,  # pytype: disable=wrong-arg-types  # jax-ndarray
                             {'decoder_inputs': corrupt_decoder_inputs})
      aux['corrupt_logits'] = corrupt_logits
      aux['corrupt_loss'] = loss_corrupt
      loss += corrupt_coef * loss_corrupt

    aux['state'] = new_model_state

    return loss, aux

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  step = train_state.global_step
  lr = learning_rate_fn(step)
  (train_loss, aux), grad = compute_gradient_fn(train_state.optimizer.target)
  new_model_state = aux['state']

  grad = jax.lax.pmean(grad, axis_name='batch')

  if config.get('max_grad_norm', None):
    grad = jax_optimizers.clip_grads(grad, config.max_grad_norm)

  new_optimizer = train_state.optimizer.apply_gradient(grad, learning_rate=lr)

  metrics = {}
  if token_loss_coef:
    token_logits = aux['token_logits']
    x = metrics_fn(
        token_logits, {
            'decoder_inputs': {
                'decoder_input_tokens': batch['text_indices'][..., :-1],
                'decoder_target_tokens': batch['text_indices'][..., 1:]
            },
            'batch_mask': batch['batch_mask']
        })
    metrics.update({'token_accuracy': x['token_accuracy']})
  if corrupt_coef:
    corrupt_logits = aux['corrupt_logits']
    x = metrics_fn(
        corrupt_logits, {
            'decoder_inputs': {
                'decoder_input_tokens':
                    batch['text_indices_corrupt_outputs'][..., :-1],
                'decoder_target_tokens':
                    batch['text_indices_corrupt_outputs'][..., 1:]
            },
            'batch_mask': batch['batch_mask']
        })
    metrics.update({'corrupt_token_accuracy': x['token_accuracy']})

  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=step + 1,
      optimizer=new_optimizer,
      model_state=new_model_state,
      rng=new_rng)
  return new_train_state, train_loss, metrics, lr


def eval_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    model: models.EncoderWithT5DecoderModel,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False,
) -> Any:
  """Runs a single step of evaluation.

  Note: The buffer of the provided batch is donated to the computation.

  Args:
    train_state: TrainState, the state of training including the current
    batch: A single batch of data. a metrics function, that given logits and
      batch of data, calculates the metrics as well as the loss.
    model: An EncoderWithT5DecoderModel. global_step, model_state, rng, and
      optimizer. The buffer of this argument can be donated to the computation.
    metrics_fn: A metrics function, that given logits and batch of data,
      calculates the metrics as well as the loss.
    config: Configuration of the experiment.
    debug: Whether the debug mode is enabled during evaluation. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Calculated metrics and predicted tokens.
  """
  variables = {
      'params': train_state.optimizer.target,
      **train_state.model_state
  }
  if config.dataset_configs.return_as_dict:
    encoder_inputs = batch['encoder_inputs']
  else:
    raise NotImplementedError

  decoding_method = config.decoding.get('decoding_method', 'beamsearch')
  if decoding_method == 'beamsearch':
    decode_fn = models.beam_search
  elif decoding_method == 'temperature_sample':
    decode_fn = models.temperature_sample
  else:
    raise ValueError('Unrecognized decoding method.')
  batch['decoder_inputs'] = {  # pytype: disable=container-type-mismatch  # jax-ndarray
      'decoder_input_tokens': batch['text_indices'][..., :-1],
      'decoder_target_tokens': batch['text_indices'][..., 1:]
  }

  decoded, _ = model.predict_batch_with_aux(
      variables, {
          'encoder_inputs': encoder_inputs,
          'decoder_inputs': batch['decoder_inputs']
      },
      decode_fn,
      num_decodes=config.decoding.get('num_decodes'),
      alpha=config.decoding.get('alpha'),
      decoding_method=decoding_method,
      temperature=config.decoding.get('temperature'),
      eos_id=1,
      vocabulary_size=32128)

  if debug:
    logging.info('Shape of decoded in eval step is: %s', decoded.shape)

  return metrics_fn(decoded, batch), decoded


def pmapped_steps(model, config):
  """Returns the pmapped train and eval steps."""
  # Learning rate scheduler.
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
      donate_argnums=(1, 2),
      static_broadcasted_argnums=(0,),  # dataset arg.
  )
  eval_step_pmapped = jax.pmap(
      functools.partial(
          eval_step,
          model=model,
          metrics_fn=model.get_metrics_fn('validation'),
          config=config,
          debug=config.debug_eval),
      axis_name='batch',
  )
  return train_step_pmapped, eval_step_pmapped


def load_decoder_params(train_state: train_utils.TrainState,
                        config: ml_collections.ConfigDict):
  """Load T5 decoder params from a checkpoint."""
  init_config = config.init_from
  t5_params = {}
  load = False
  if init_config.get('decoder') and init_config.decoder.get(
      'load_pretrained_weights', True):
    model_name = config.model.decoder.t5_decoder.pretrained_config
    t5_params = t5_model.load_pretrained_weights(model_name)
    load = True
  logging.info('T5 params are:')
  logging.info(jax.tree_util.tree_map(lambda x: x.shape, t5_params))
  if not load:
    return train_state
  train_state = load_utils.init_from_pretrain_weights(
      train_state,
      restored_params=t5_params,
      ckpt_prefix_path=['params', 't5_module', 'token_embedder'],
      model_prefix_path=['shared_decoder_token_embedder'],
  )
  return load_utils.init_from_pretrain_weights(
      train_state,
      restored_params=t5_params,
      ckpt_prefix_path=['params', 't5_module', 'decoder'],
      model_prefix_path=[
          'text_decoder', 'decoder_module'
      ],
  )


def load_encoder_params(train_state: train_utils.TrainState,
                        config: ml_collections.ConfigDict):
  """Load encoder parameters."""

  if config.init_from.encoder.get('load_pretrained_weights', True):
    model_name = config.model.encoder.cat_encoder.pretrained_config
    t5_params = t5_model.load_pretrained_weights(model_name)
    model_prefix_path = [
        'video_encoder', 'encoder_module'
    ]
    if config.init_from.encoder.load_pretrained_weights:
      model_prefix_path = ['encoder'] + model_prefix_path
    train_state = load_utils.init_from_pretrain_weights(
        train_state,
        restored_params=t5_params,
        ckpt_prefix_path=['params', 't5_module', 'encoder'],
        model_prefix_path=model_prefix_path,
    )

  return train_state


def init_state(model: base_model.BaseModel, dataset: dataset_utils.Dataset,
               config: ml_collections.ConfigDict, workdir: str,
               rng: jnp.ndarray):
  """Initialize the train state."""

  encoder_input_shape = dataset.meta_data['encoder_input_shape']
  encoder_input_dtype = dataset.meta_data.get('encoder_input_dtype',
                                              jnp.float32)
  encoder_input_text_dtype = dataset.meta_data.get('encoder_input_text_dtype',
                                                   jnp.int32)
  decoder_input_shape = dataset.meta_data['decoder_input_shape']
  decoder_input_dtype = dataset.meta_data.get('decoder_input_dtype', jnp.int32)

  encoder_input_spec = {}
  if isinstance(encoder_input_shape, dict):
    for mod in config.dataset_configs.modalities:
      mod_spec = None
      if mod in encoder_input_shape:
        logging.info('Modality %s is present for this dataset', mod)
        local_encoder_input_dtype = encoder_input_dtype
        if mod == 'text':
          local_encoder_input_dtype = encoder_input_text_dtype
        mod_spec = (encoder_input_shape[mod], local_encoder_input_dtype)
      encoder_input_spec[mod] = mod_spec
  else:
    encoder_input_spec = (encoder_input_shape, encoder_input_dtype)

  decoder_input_spec = {
      k: (v, decoder_input_dtype) for k, v in decoder_input_shape.items()
  }

  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  (params, model_state, num_params,
   gflops) = train_utils.initialize_model_with_pytree(
       model_def=model.flax_model,
       input_spec=(encoder_input_spec, decoder_input_spec),
       config=config,
       rngs=init_rng)
  logging.info('The model has %d params, uses %d gflops', num_params, gflops or
               -1)

  # Create the optimizer.
  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  optimizer = jax.jit(
      optimizers.get_optimizer(config).create, backend='cpu')(
          params)
  del params  # Do not keep a copy of the initial params.

  rng, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
      global_step=0,
      optimizer=optimizer,
      model_state=model_state,
      rng=train_rng,
      accum_train_time=0)
  start_step = train_state.global_step
  if config.checkpoint:
    logging.info('Continuing training from the checkpoint')
    train_state, params_axes = vid2seq_train_utils.pop_axes_names(
        train_state, 'params_axes')
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state
    )
    train_state = vid2seq_train_utils.re_add_axis_names(
        train_state, params_axes, 'params_axes'
    )

  if start_step == 0 and config.get('init_from'):
    if config.init_from.get('checkpoint_path'):
      restored_model_cfg = config.init_from.get('model_config')
      init_checkpoint_path = config.init_from.get('checkpoint_path')
      step = config.init_from.get('step')
      restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
          init_checkpoint_path, train_state, assert_exist=True, step=step)

      # init T5 encoder if in model but not in ckpt
      if config.model.encoder.encoder_type in [
          't5_encoder', 'cat_encoder'
      ] and 'text' in config.dataset_configs.modalities:
        if 'video_encoder' not in restored_train_state.optimizer.target[
            'encoder'] and 'video_encoder' in train_state.optimizer.target[
                'encoder']:
          train_state = load_encoder_params(train_state, config)
          x = unfreeze(restored_train_state.optimizer.target)
          x['encoder'] = copy.deepcopy(train_state.optimizer.target['encoder'])
          optimizer = restored_train_state.optimizer.replace(target=x)
          restored_train_state = restored_train_state.replace(
              optimizer=optimizer)

          y = unfreeze(restored_train_state.optimizer.state.param_states)
          y['encoder']['video_encoder'] = copy.deepcopy(
              train_state.optimizer.state.param_states['encoder']
              ['video_encoder'])
          state = restored_train_state.optimizer.state.replace(param_states=y)
          optimizer = restored_train_state.optimizer.replace(state=state)
          restored_train_state = restored_train_state.replace(
              optimizer=optimizer)

      # throw away T5 encoder if in checkpoint but not in model
      if config.model.encoder.encoder_type in [
          't5_encoder', 'cat_encoder'
      ] and 'text' not in config.dataset_configs.modalities:
        if 'video_encoder' in restored_train_state.optimizer.target[
            'encoder'] and 'video_encoder' not in train_state.optimizer.target[
                'encoder']:
          x = unfreeze(restored_train_state.optimizer.target)
          if 'encoder' in x and 'video_encoder' in x['encoder']:
            del x['encoder']['video_encoder']
          optimizer = restored_train_state.optimizer.replace(target=x)
          restored_train_state = restored_train_state.replace(
              optimizer=optimizer)

          y = unfreeze(restored_train_state.optimizer.state.param_states)
          if 'encoder' in y and 'video_encoder' in y['encoder']:
            del y['encoder']['video_encoder']
          state = restored_train_state.optimizer.state.replace(param_states=y)
          optimizer = restored_train_state.optimizer.replace(state=state)
          restored_train_state = restored_train_state.replace(
              optimizer=optimizer)

      train_state = load_utils.initialise_from_train_state(
          config,
          train_state,
          restored_train_state,
          restored_model_cfg,
          restore_output_proj=False)
    else:
      # Seperately intiialize encoders and decoders
      if config.model.encoder.encoder_type in [
          't5_encoder', 'cat_encoder'
      ]:
        train_state = load_encoder_params(train_state, config)
      train_state = load_decoder_params(train_state, config)

  elif start_step == 0:
    logging.info('Training completely from scratch. '
                 'Not restoring from any checkpoint.')
  return train_state, start_step


@dataclasses.dataclass
class SummaryBuilder:
  """A helper class to build the summary over the training iterations."""
  metrics: List[Dict[str, Tuple[float, int]]]
  extra_logs: List[Dict[str, Any]]

  def update(self, metrics_update, extra_logs_update):
    """Update with the given per-step metrics."""
    self.metrics.append(metrics_update)
    self.extra_logs.append(extra_logs_update)

  def write(self, writer: metric_writers.MetricWriter, step: int):
    """Write to the given writer and training step.

    After writing, the state gets reset.

    Args:
      writer: The summary will be written with this writer.
      step: The current training step.

    Returns:
      The summary since the last write.
    """
    summary = train_utils.log_train_summary(
        step=step,
        train_metrics=jax.tree_util.tree_map(train_utils.unreplicate_and_get,
                                             self.metrics),
        extra_training_logs=jax.tree_util.tree_map(
            train_utils.unreplicate_and_get, self.extra_logs),
        writer=writer,
        key_separator='/')
    self.metrics = []
    self.extra_logs = []
    return summary


def eval_and_log_summary(
    *,
    train_state: train_utils.TrainState,
    writer: metric_writers.MetricWriter,
    iterator,
    eval_step_fn,
    eval_steps,
    train_iteration,
    tokenizer,
    dataset_name,
    num_bins,
    vocabulary_size,
    abs_time_token,
    time_format,
    tmp_only,
    runlocal,  # pylint: disable=unused-argument
    order,
    workdir,
    soda,
    eval_batch_size,  # pylint: disable=unused-argument
    max_events,
    para,
    t,
    is_split
    ):
  """Eval the model and write the summary."""
  # Sync model state across replicas.
  train_state = train_utils.sync_model_state_across_replicas(train_state)
  eval_packs = {}
  logging.info('Total number of eval steps is %s', eval_steps)
  # This ensures that all eval batchs are covered.
  eval_steps = int(eval_steps * 1.3)
  keys = []
  for step in range(eval_steps):
    with jax.profiler.StepTraceAnnotation('eval', step_num=step):
      eval_batch = next(iterator)

      # Put the string inputs to a separate lists and delete them before passing
      # to the pmapped function.
      eval_pack = {
          'gts':
              dvc_eval.convert_strings_to_uint8_arrays(
                  eval_batch['caption_strings'], MAX_CAPTION_STR_LEN),
          'key':
              dvc_eval.convert_strings_to_uint8_arrays(
                  eval_batch['videoid'], MAX_KEY_STR_LEN),
          'batch_mask':
              eval_batch['batch_mask'],
          'duration':
              eval_batch['duration'],
          'gts_start':
              eval_batch['timestamp_start'],
          'gts_end':
              eval_batch['timestamp_end'],
          'split':
              eval_batch['split'] if 'split' in eval_batch else
              np.ones_like(eval_batch['timestamp_start']),
      }
      to_del = ['caption_strings', 'key', 'videoid', 'timestamp_start',
                'timestamp_end', 'split']  # 'duration',
      for x in to_del:
        if x in eval_batch:
          del eval_batch[x]

      eval_metrics, preds = eval_step_fn(train_state, eval_batch)

      # Do not gather at this stage to run dvc_eval before gathering
      eval_pack['pred'] = preds
      eval_pack = jax.tree_util.tree_map(
          lambda x: x.reshape((np.prod(x.shape[:2]),) + x.shape[2:]), eval_pack
      )
      logging.info(
          'eval_pack %d shapes: %s',
          step,
          jax.tree_util.tree_map(lambda x: x.shape, eval_pack),
      )

      gts_timestamps = [[
          [s, e] for s, e in zip(ls, le)
      ] for ls, le in zip(eval_pack['gts_start'], eval_pack['gts_end'])]
      gts_timestamps = [[x for x in y if x[0] != -1] for y in gts_timestamps
                       ]  # unpad GT

      gts = [[remove_nonascii(dvc_eval.convert_uint8_array_to_string(x))
              for x in y]
             for y in eval_pack['gts']]
      gts = [[x for x in y if x] for y in gts]  # unpad GT

      splits = [[k for m, k in enumerate(eval_pack['split'][i])
                 if m < len(gts[i])] for i in range(len(gts))]

      for i, valid in enumerate(eval_pack['batch_mask']):
        if valid:
          key = dvc_eval.convert_uint8_array_to_string(eval_pack['key'][i])
          if key in eval_packs:  # redundant video
            continue
          keys.append(key)

          pred, pred_timestamps = [], []
          # get indexes in the predicted seq that delimit the pred segments
          indexes = [
              j for j in range(len(eval_pack['pred'][i]) - 1)
              if eval_pack['pred'][i][j] >= vocabulary_size and
              eval_pack['pred'][i][j + 1] >= vocabulary_size
          ]  # pylint: disable=g-complex-comprehension

          last_processed = -2

          # iterate over predicted segments and decode them
          for j in range(len(indexes)):
            if indexes[j] == last_processed + 1:  # 3 timestamps != 2 events
              continue

            # get predicted tokens and transform to string
            if order == 'ld':
              start_idx = indexes[j] + 2
              end_idx = indexes[j + 1] if j < len(indexes) - 1 else len(
                  eval_pack['pred'][i])
            else:
              start_idx = indexes[j - 1] + 2 if j > 0 else 0
              end_idx = indexes[j]
            pred_seq = [int(eval_pack['pred'][i][k])
                        for k in range(start_idx, end_idx)]
            pred_text = decode_tokens(pred_seq, tokenizer, vocabulary_size)
            if (not pred_text) and (not tmp_only):  # remove empty string
              continue

            # get start and end
            if not abs_time_token:
              max_offset = num_bins - 1
              pred_time = [
                  (int(eval_pack['pred'][i][indexes[j]])
                   - vocabulary_size) *
                  eval_pack['duration'][i] / max_offset,
                  (int(eval_pack['pred'][i][indexes[j] + 1]) -
                   vocabulary_size) *
                  eval_pack['duration'][i] / max_offset
                  ]
            else:
              pred_time = [
                  (int(eval_pack['pred'][i][indexes[j]])
                   - vocabulary_size) * t,
                  (int(eval_pack['pred'][i][indexes[j] + 1]) -
                   vocabulary_size) * t
                  ]
              pred_time = decode_time(pred_time, eval_pack['duration'][i],
                                      time_format)
            if pred_time[1] <= pred_time[0]:  # remove end < start
              continue
            last_processed = indexes[j]

            pred.append(pred_text)
            pred_timestamps.append(pred_time)

          eval_packs[key] = {
              'pred': pred,
              'gts': gts[i],
              'pred_timestamps': pred_timestamps,
              'gts_timestamps': gts_timestamps[i],  # unpad GT timestamp
              'split': splits[i],
          }

      to_del = [
          'batch_mask', 'gts', 'pred', 'duration', 'gts_start', 'gts_end',
          'key', 'split'
      ]
      for x in to_del:
        del eval_pack[x]
      logging.info('Finished %d decoding', step)

  predicted_captions = [eval_packs[x]['pred'] for x in keys]
  predicted_segments = [
      np.array(eval_packs[x]['pred_timestamps']) for x in keys
  ]
  gt_captions = [eval_packs[x]['gts'] for x in keys]
  gt_segments = [np.array(eval_packs[x]['gts_timestamps']) for x in keys]
  splits = [eval_packs[x]['split'] for x in keys]

  if para:
    logging.info('Gathering predictions')
    # Fill fixed shape arrays
    pad_len = eval_steps * (eval_batch_size // jax.process_count())
    res = {
        'pred':
            np.zeros([pad_len,
                      MAX_CAPTION_STR_LEN * max_events]).astype(np.uint8),
        'gt':
            np.zeros([pad_len, 2,
                      MAX_CAPTION_STR_LEN * max_events]).astype(np.uint8),
        'mask':
            np.zeros([pad_len])
    }
    res['mask'][:len(splits)] = 1
    for i in range(len(splits)):
      if predicted_captions[i]:
        pred = ' '.join(predicted_captions[i])
        pred = dvc_eval.convert_strings_to_uint8_arrays(
            np.array([pred]), MAX_CAPTION_STR_LEN * max_events)
        res['pred'][i] = pred[0]
      split = splits[i]
      unique_splits = set(split)
      for j, s in enumerate(unique_splits):
        indexes = np.where(split == s)[0]
        gt = ' '.join([gt_captions[i][idx] for idx in indexes])
        gt = dvc_eval.convert_strings_to_uint8_arrays(
            np.array([gt]), MAX_CAPTION_STR_LEN * max_events)
        res['gt'][i][j] = gt[0]
    ndevh = jax.device_count() // jax.process_count()
    for x in res:
      res[x] = res[x].reshape((ndevh, res[x].shape[0] // ndevh) +
                              res[x].shape[1:])

    # Gather and filter by mask
    res = train_utils.unreplicate_and_get(
        jax.pmap(lambda x: jax.lax.all_gather(x, 'batch'), 'batch')(res))
    res = jax.tree_util.tree_map(
        lambda x: x.reshape((np.prod(x.shape[:2]),) + x.shape[2:]), res
    )
    mask = res['mask'].astype(bool)
    pred = res['pred'][mask]
    pred = [dvc_eval.convert_uint8_array_to_string(x) for x in pred]
    gt = res['gt'][mask]
    gt = [[dvc_eval.convert_uint8_array_to_string(y) for y in x] for x in gt]
    gt = [[y for y in x if y] for x in gt]  # unpad splits

    logging.info('Computing paragraph metrics...')
    logging.info(pred[0])
    logging.info(gt[0])
    para_res = dvc_eval.evaluate_para(pred, gt)
    logging.info('Done')

    return train_utils.log_eval_summary(
        step=train_iteration,
        eval_metrics=[train_utils.unreplicate_and_get(eval_metrics)],
        extra_eval_summary=para_res,
        writer=writer,
        key_separator='/',
        prefix=dataset_name)
  logging.info('The size of eval_packs for %s: %d', dataset_name,
               len(eval_packs))
  eval_res = dvc_eval.evaluate_dense_captions(
      predicted_segments=predicted_segments,
      gt_segments=gt_segments,
      predicted_captions=predicted_captions,
      gt_captions=gt_captions,
      splits=splits,
      iou_thresholds=(0.3, 0.5, 0.7, 0.9),
      soda=soda,
      keys=keys,
      tmponly=tmp_only)
  logging.info('Finished per-host evaluation')

  # fill a fixed shape array
  full_res = {
      x: np.zeros([eval_steps * (eval_batch_size // jax.process_count())])
      for x in eval_res.keys() if x != 'key'
  }
  full_res['mask'] = np.zeros(
      [eval_steps * (eval_batch_size // jax.process_count())])
  for x in eval_res:
    if x != 'key':
      full_res[x][:len(eval_res[x])] = np.array(eval_res[x])
    full_res['mask'][:len(eval_res[x])] = 1

  # gather results on all hosts
  for x in full_res:
    # number of devices per host
    ndevh = jax.device_count() // jax.process_count()
    full_res[x] = full_res[x].reshape((ndevh, full_res[x].shape[0] // ndevh))
  full_res = train_utils.unreplicate_and_get(
      jax.pmap(lambda x: jax.lax.all_gather(x, 'batch'), 'batch')(full_res))
  full_res = jax.tree_util.tree_map(
      lambda x: x.reshape((np.prod(x.shape[:2]),) + x.shape[2:]), full_res
  )
  logging.info(full_res[list(full_res)[0]].shape)

  # compute averaged statistics
  avg_res = {}
  mask = full_res['mask'].astype(bool)
  for x in full_res:
    if x == 'SODA_c_1' or x == 'SODA_c_2':
      mask2 = jnp.logical_and(mask, full_res[x] != -1)
      avg_res[x] = float(np.mean(full_res[x][mask2]))
    elif x != 'mask':
      avg_res[x] = float(np.mean(full_res[x][mask]))
  if is_split:
    avg_res['SODA_c'] = (avg_res['SODA_c_2'] +
                         avg_res['SODA_c_1']) / 2
  else:
    avg_res['SODA_c'] = avg_res['SODA_c_1']
  del avg_res['SODA_c_2'], avg_res['SODA_c_1']
  logging.info('Finished gathering eval metrics for %d samples', sum(mask))

  return train_utils.log_eval_summary(
      step=train_iteration,
      eval_metrics=[train_utils.unreplicate_and_get(eval_metrics)],
      extra_eval_summary=avg_res,
      writer=writer,
      key_separator='/',
      prefix=dataset_name)


def get_tokenizer(
    config: ml_collections.ConfigDict) -> tokenizers.TextTokenizer:
  """Get tokenizer to decode strings for eval."""
  tokenizer_config = config.dataset_configs.get('tokenizer',
                                                ml_collections.ConfigDict())
  tokenizer_type = tokenizer_config.get('tokenizer_type', None)
  tokenizer_model = tokenizer_config.get('tokenizer_model', None)

  if tokenizer_type == 'sentence_piece':
    if tokenizer_model is not None:
      tokenizer = t5_tokenizer.build_dmvr_sp_model(tokenizer_model)
    else:
      tokenizer = t5_tokenizer.build_dmvr_sp_model()
  else:
    raise ValueError('Unsupported tokenizer.')

  return tokenizer


def train_and_eval(
    rng: np.ndarray, config: ml_collections.ConfigDict, *, workdir: str,
    writer: Any, model_cls, dataset_dict
) -> Tuple[train_utils.TrainState, Dict[str, Any], Dict[str, Any]]:
  """Train (and occasionally evaluate) the model.

  Args:
    rng: JAX prng key.
    config: The configuration of the experiment.
    workdir: Where to checkpoint and write the summaries.
    writer: Summary writer object.
    model_cls: The model class used to instantiate the model.
    dataset_dict: The dataset for training and evaluation.

  Returns:
    A tuple with:
      * the state that has the state of training (including current
        global_step, model_state, rng, and the optimizer)
      * a dictionary with the train_summary
      * a dictionary with the evaluation summary
  """

  lead_host = jax.host_id() == 0

  datasets_metadata = {name: ds.meta_data for name, ds in dataset_dict.items()}
  all_datasets = []
  all_datasets_num_train_examples = []
  for name, metadata in datasets_metadata.items():
    all_datasets.append(name)
    all_datasets_num_train_examples.append(
        metadata.get('num_train_examples', 0))
  model = model_cls(config, datasets_metadata)
  train_step_pmapped, eval_step_pmapped = pmapped_steps(model, config)

  train_state, start_step = init_state(model, dataset_dict[all_datasets[0]],  # pytype: disable=wrong-arg-types  # jax-ndarray
                                       config, workdir, rng)
  train_state = jax_utils.replicate(train_state)
  logging.info('Number of processes is %s', jax.process_count())

  del rng  # So that we don't mistakenly re-use it.

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = (
      vid2seq_train_utils.get_num_training_steps_multi(
          config, datasets_metadata
      )
  )
  log_eval_steps = config.get('log_eval_steps', steps_per_epoch)
  checkpoint_steps = config.get('checkpoint_steps', log_eval_steps)
  log_summary_steps = config.get('log_summary_steps', log_eval_steps)

  # Build a tokenizer for the evaluation
  tokenizer = get_tokenizer(config)

  chrono = train_utils.Chrono(
      first_step=start_step,
      total_steps=total_steps,
      steps_per_epoch=steps_per_epoch,
      global_bs=config.batch_size,
      accum_train_time=int(jax_utils.unreplicate(train_state.accum_train_time)))

  logging.info('Start training from step %d', start_step + 1)
  hooks = []
  if config.get('xprof', True) and jax.process_index() == 0:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  summary_builder = {x: SummaryBuilder([], []) for x in all_datasets}
  train_summary, eval_summary = None, None

  def get_next_train_batch(all_datasets, step):
    dataset = random.Random(step).choices(
        all_datasets,
        config.get('probs', [1. / len(all_datasets)] * len(all_datasets)))[0]
    ds = dataset_dict[dataset]

    return next(ds.train_iter), dataset

  for step in range(start_step + 1, total_steps + 1):
    chrono.resume()
    train_batch, train_ds = get_next_train_batch(all_datasets, step)
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_state, train_loss, t_metrics, lr = train_step_pmapped(
          train_ds, train_state, train_batch)
    for hook in hooks:
      hook(step)
    summary_builder[train_ds].update(t_metrics, {
        'lr': lr,
        'train/loss': train_loss
    })
    chrono.pause()

    # Log the train summary every `log_summary_steps`.
    if (step % log_summary_steps == 1) or (step == total_steps):
      if lead_host:
        chrono.tick(step, writer)
      train_summary = {x: {} for x in all_datasets}
      for x in all_datasets:
        if len(summary_builder[x].metrics):  # pylint: disable=g-explicit-length-test
          train_summary[x].update(summary_builder[x].write(writer, step))

    # Evaluate every `log_eval_steps`.
    should_eval = (step % log_eval_steps == 1) or (step == total_steps)
    if should_eval:
      for ds_name in dataset_dict[all_datasets[0]].valid_iter:
        # Compute the number of evaluation steps per dataset.
        num_eval_examples = dataset_dict[
            all_datasets[0]].meta_data['num_eval_examples'][ds_name]
        total_eval_steps = int(
            np.ceil(num_eval_examples / (config.get('eval_batch_size'))))
        steps_per_eval = config.get('steps_per_eval', total_eval_steps)

        eval_summary = eval_and_log_summary(
            train_state=train_state,
            iterator=dataset_dict[all_datasets[0]].valid_iter[ds_name],
            eval_step_fn=eval_step_pmapped,
            eval_steps=steps_per_eval,
            writer=writer,
            train_iteration=step,
            tokenizer=tokenizer,
            dataset_name=ds_name,
            num_bins=config.dataset_configs.num_bins,
            vocabulary_size=config.dataset_configs.vocabulary_size,
            abs_time_token=config.dataset_configs.abs_time_token,
            time_format=config.dataset_configs.time_format,
            tmp_only=config.dataset_configs.tmp_only,
            runlocal=config.runlocal,
            order=config.dataset_configs.order,
            workdir=workdir,
            soda='soda' in config.eval_metrics,
            eval_batch_size=config.eval_batch_size,
            max_events=config.dataset_configs.max_events,
            para='para' in ds_name,
            t=1000000.,  # 1 FPS
            is_split=config.dataset_configs.split)

    # Checkpoint.
    if not config.checkpoint:
      continue
    elif step % checkpoint_steps == 0 and step > 0:
      logging.info('checkpointing starts (training step: %d)', step)
      # Sync model state across replicas.
      train_state = train_utils.sync_model_state_across_replicas(train_state)
      logging.info('checkpointing (training step: %d)', step)
      train_state.replace(  # pytype: disable=attribute-error
          accum_train_time=chrono.accum_train_time)
      train_utils.save_checkpoint(workdir, train_state, max_to_keep=100)

  # Return the train and eval summary after last step.
  return train_state, train_summary, eval_summary


def eval_only(
    rng: np.ndarray, config: ml_collections.ConfigDict, *, workdir: str,
    writer: Any, model_cls, dataset_dict
) -> Tuple[train_utils.TrainState, Dict[str, Any], Dict[str, Any]]:
  """Evaluate the model.

  Args:
    rng: JAX prng key.
    config: The configuration of the experiment.
    workdir: Where to checkpoint and write the summaries.
    writer: Summary writer object.
    model_cls: The model class used to instantiate the model.
    dataset_dict: The dataset for training and evaluation.

  Returns:
    A tuple with:
      * the state that has the state of training (including current
        global_step, model_state, rng, and the optimizer)
      * a dictionary with the train_summary
      * a dictionary with the evaluation summary
  """

  datasets_metadata = {name: ds.meta_data for name, ds in dataset_dict.items()}
  all_datasets = []
  all_datasets_num_train_examples = []
  for name, metadata in datasets_metadata.items():
    all_datasets.append(name)
    all_datasets_num_train_examples.append(
        metadata.get('num_train_examples', 0))
  dataset = dataset_dict[all_datasets[0]]

  model = model_cls(config, dataset.meta_data)
  _, eval_step_pmapped = pmapped_steps(model, config)

  train_state, start_step = init_state(model, dataset, config, workdir, rng)  # pytype: disable=wrong-arg-types  # jax-ndarray
  assert start_step == 0
  train_state = jax_utils.replicate(train_state)
  logging.info('Number of processes is %s', jax.process_count())

  del rng  # So that we don't mistakenly re-use it.

  # Build a tokenizer for the evaluation
  tokenizer = get_tokenizer(config)

  hooks = []
  if config.get('xprof', True) and jax.process_index() == 0:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  # Evaluate every `log_eval_steps`.
  for ds_name in dataset.valid_iter:
    # Compute the number of evaluation steps per dataset.
    num_eval_examples = dataset.meta_data['num_eval_examples'][ds_name]
    total_eval_steps = int(
        np.ceil(num_eval_examples / (config.get('eval_batch_size'))))
    steps_per_eval = config.get('steps_per_eval', total_eval_steps)

    eval_summary = eval_and_log_summary(
        train_state=train_state,
        iterator=dataset.valid_iter[ds_name],
        eval_step_fn=eval_step_pmapped,
        eval_steps=steps_per_eval,
        writer=writer,
        train_iteration=0,
        tokenizer=tokenizer,
        dataset_name=ds_name,
        num_bins=config.dataset_configs.num_bins,
        vocabulary_size=config.dataset_configs.vocabulary_size,
        abs_time_token=config.dataset_configs.abs_time_token,
        time_format=config.dataset_configs.time_format,
        tmp_only=config.dataset_configs.tmp_only,
        runlocal=config.runlocal,
        order=config.dataset_configs.order,
        workdir=workdir,
        soda='soda' in config.eval_metrics,
        eval_batch_size=config.eval_batch_size,
        max_events=config.dataset_configs.max_events,
        para='para' in ds_name,
        t=1000000.,  # 1 FPS
        is_split=config.dataset_configs.split)

  # Return the train and eval summary after last step.
  return train_state, {}, eval_summary
