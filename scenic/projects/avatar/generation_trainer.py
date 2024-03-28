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

"""Token generation training.

Template from third_party/py/scenic/projects/vtretrieval/trainer.py
Auto-regressive generation from third_party/py/flax/examples/wmt/train.py
"""

import copy
import dataclasses
import functools
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from dmvr import tokenizers
from flax import jax_utils
import flax.linen as nn
import jax
import jax.example_libraries.optimizers as jax_optimizers
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import base_model
from scenic.projects.avatar import decode
from scenic.projects.avatar import metrics_utils
from scenic.projects.avatar import model_utils
from scenic.projects.avatar.datasets import dataset_utils as ds_utils
from scenic.train_lib_deprecated import lr_schedules
from scenic.train_lib_deprecated import optimizers
from scenic.train_lib_deprecated import pretrain_utils
from scenic.train_lib_deprecated import train_utils

from tensorflow.io import gfile


# Note this list must be in the exact order of the inputs required by the model.
SUPPORTED_MODALITIES = ['rgb', 'flow', 'spectrogram', 'waveform', 'text']

# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricNormalizerFnDict = base_model.MetricNormalizerFnDict
MetricFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                    Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], float]
PyTree = Any


def tohost(x):
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


def to_cpu(pytree: PyTree) -> PyTree:
  """Transfers arrays (replicated on multiple hosts) to a single host.

  Args:
    pytree: PyTree of replicated arrays of [num_hosts, num_devices,
     local_batch_size, ...]

  Returns:
    PyTree of arrays of shape [global_batch_size, ...] where
      global_batch_size = num_devices * local_batch_size
  """
  return jax.device_get(dataset_utils.unshard(jax_utils.unreplicate(pytree)))


def decode_tokens(tokenizer, toks):
  # Decode a sequence of tokens into text
  eos_id = tokenizer.eos_token
  toks = toks.astype(np.int32)
  if eos_id in toks:
    toks = toks[:np.argmax(toks == eos_id) + 1]
  s = tokenizer.indices_to_string(toks)
  # Remove spaces around apostrophe
  s = s.replace(' \' ', '\'')
  return s


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
    train_state: The state of training including the current global_step,
      model_state, rng, and optimizer. The buffer of this argument can be
      donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    flax_model: A Flax model.
    learning_rate_fn: learning rate scheduler which give the global_step
      generates the learning rate.
    loss_fn: A loss function that given logits, targets, weights, and parameters
      of the model calculates the loss.
    metrics_fn: A metrics function that given logits, targets and weights
      calculates the metrics as well as the loss.
    config: Configuration of the experiment.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training, computed metrics, and learning rate for logging.
  """
  new_rng, rng = jax.random.split(train_state.rng)

  # Bind the rng to the host/device we are on for dropout.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device')

  if config.model.encoder_model == 'vivit':
    video_inputs = batch['inputs']
  elif config.dataset_configs.return_as_dict:
    video_inputs = [
        batch['inputs'].get('rgb', None),  # pytype: disable=attribute-error  # jax-ndarray
        batch['inputs'].get('flow', None),  # pytype: disable=attribute-error  # jax-ndarray
        batch['inputs'].get('spectrogram', None),  # pytype: disable=attribute-error  # jax-ndarray
        batch['inputs'].get('waveform', None),  # pytype: disable=attribute-error  # jax-ndarray
        batch['inputs'].get('text', None)  # pytype: disable=attribute-error  # jax-ndarray
    ]
  # When using DMVR datasets which only have RGB
  else:
    video_inputs = [batch['inputs'], None, None, None, None]
  targets = batch['targets']
  # Remove the "num_captions" dimension
  targets = jnp.squeeze(targets, axis=-2)

  weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)

  other_inputs = {}
  if config.get('predict_masked_word', False):
    other_inputs['masked_token_idxs'] = batch['masked_input_token_indices']
    other_inputs['masked_token_idx_masks'] = batch[
        'valid_input_token_index_mask']
    other_inputs['masked_word_targets'] = batch['masked_targets']

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}

    logits, new_model_state = flax_model.apply(
        variables,
        *video_inputs,
        targets,
        **other_inputs,
        mutable=['batch_stats'],
        train=True,
        rngs={'dropout': dropout_rng},
        debug=debug)

    if config.get('predict_masked_word', False):
      b, m, t = batch['masked_targets'].shape
      masked_word_targets = jnp.reshape(batch['masked_targets'], [b * m, t])
      masked_word_weights = jnp.where(masked_word_targets > 0, 1,
                                      0).astype(jnp.float32)
      logits, masked_word_logits = logits

      loss = loss_fn((logits, masked_word_logits),  # pytype: disable=wrong-arg-types  # jax-ndarray
                     (targets, masked_word_targets),
                     (weights, masked_word_weights), variables['params'])
    else:
      loss = loss_fn(logits, targets, weights, variables['params'])

    return loss, (new_model_state, logits)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  step = train_state.global_step
  lr = learning_rate_fn(step)
  (train_cost,
   (new_model_state,
    logits)), grad = compute_gradient_fn(train_state.optimizer.target)
  del train_cost

  if config.get('max_grad_norm', None):
    grad = jax_optimizers.clip_grads(grad, config.max_grad_norm)

  grad = jax.lax.pmean(grad, axis_name='batch')
  new_optimizer = train_state.optimizer.apply_gradient(grad, learning_rate=lr)

  # Explicit weight decay, if necessary.
  if config.get('explicit_weight_decay', None):
    new_optimizer = new_optimizer.replace(
        target=optimizers.tree_map_with_names(
            functools.partial(
                optimizers.decay_weight_fn,
                lr=lr,
                decay=config.explicit_weight_decay),
            new_optimizer.target,
            match_name_fn=lambda name: 'kernel' in name))

  metrics = metrics_fn(logits, targets, weights)
  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=step + 1,
      optimizer=new_optimizer,
      model_state=new_model_state,
      rng=new_rng)
  return new_train_state, metrics, lr


def eval_step(train_state: train_utils.TrainState,
              batch: Batch,
              *,
              flax_model: nn.Module,
              metrics_fn: MetricFn,
              config: ml_collections.ConfigDict,
              debug: Optional[bool] = False) -> Any:
  """Runs a single step of evaluation.

  Note: The buffer of the provided batch is donated to the computation.

  Args:
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data.
    flax_model: A Flax model.
    metrics_fn: A metrics function that given logits, targets and weights
      calculates the metrics as well as the loss.
    config: Configuration of the experiment.
    debug: Whether the debug mode is enabled during evaluation. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Calculated metrics.
  """
  variables = {
      'params': train_state.optimizer.target,
      **train_state.model_state
  }
  if config.model.encoder_model == 'vivit':
    video_inputs = batch['inputs']
  elif config.dataset_configs.return_as_dict:
    video_inputs = [
        batch['inputs'].get('rgb', None), batch['inputs'].get('flow', None),  # pytype: disable=attribute-error  # jax-ndarray
        batch['inputs'].get('spectrogram',  # pytype: disable=attribute-error  # jax-ndarray
                            None), batch['inputs'].get('waveform', None),  # pytype: disable=attribute-error  # jax-ndarray
        batch['inputs'].get('text', None)  # pytype: disable=attribute-error  # jax-ndarray
    ]
  # When using DMVR datasets which only have RGB
  else:
    video_inputs = [batch['inputs'], None, None, None, None]

  targets = batch['targets']
  # Remove the "num_captions" dimension
  targets = jnp.squeeze(targets, axis=-2)

  weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)

  logits = flax_model.apply(
      variables,
      *video_inputs,
      targets=targets,
      mutable=False,
      train=False,
      debug=debug)

  return metrics_fn(logits, targets, weights)


def test_step(*,
              train_state: train_utils.TrainState,
              batch: Batch,
              flax_model: nn.Module,
              cache,
              config: ml_collections.ConfigDict,
              debug: Optional[bool] = False) -> Any:
  """Runs a single step of test."""
  variables = {
      'params': train_state.optimizer.target,
      **train_state.model_state
  }
  if config.model.encoder_model == 'vivit':
    video_inputs = batch['inputs']
  elif config.dataset_configs.return_as_dict:
    video_inputs = [
        batch['inputs'].get('rgb', None),  # pytype: disable=attribute-error  # jax-ndarray
        batch['inputs'].get('flow', None),  # pytype: disable=attribute-error  # jax-ndarray
        batch['inputs'].get('spectrogram', None),  # pytype: disable=attribute-error  # jax-ndarray
        batch['inputs'].get('waveform', None),  # pytype: disable=attribute-error  # jax-ndarray
        batch['inputs'].get('text', None)  # pytype: disable=attribute-error  # jax-ndarray
    ]
  # When using DMVR datasets which only have RGB
  else:
    video_inputs = [batch['inputs'], None, None, None, None]

  beam_size = config.get('beam_size', 4)
  # Prepare transformer fast-decoder call for beam search: for beam search, we
  # need to set up our decoder model to handle a batch size equal to
  # batch_size * beam_size, where each batch item"s data is expanded in-place
  # rather than tiled.
  # i.e. if we denote each batch element subtensor as el[n]:
  # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]

  # As the second return value of flax_model.encode is None in test time, we
  # simply use the first return value here.
  encoded_inputs = decode.flat_batch_beam_expand(
      flax_model.apply(
          variables, *video_inputs, train=False, method=flax_model.encode)[0],
      beam_size)

  def tokens_ids_to_logits(flat_ids, flat_cache):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, 1, vocab]
    flat_logits, new_vars = flax_model.apply(
        {
            'params': train_state.optimizer.target,
            'cache': flat_cache,
            **train_state.model_state,
        },
        encoded_inputs,
        flat_ids,
        decode=True,
        train=False,
        mutable=['cache'],
        method=flax_model.decode,
        debug=debug)
    new_flat_cache = new_vars['cache']
    # Remove singleton sequence-length dimension:
    # [batch * beam, 1, vocab] --> [batch * beam, vocab]
    flat_logits = flat_logits.squeeze(axis=1)
    return flat_logits, new_flat_cache

  # Get the first modality
  mod = list(batch['inputs'].keys())[0]  # pytype: disable=attribute-error  # jax-ndarray
  batch_size = batch['inputs'][mod].shape[0]
  dummy_inputs = jnp.ones((batch_size), jnp.int32)

  brevity_penalty = config.get('brevity_penalty', 0.6)
  # Using the above-defined single-step decoder function, run a
  # beam search over possible sequences given input encoding.
  beam_seqs, _ = decode.beam_search(
      dummy_inputs,  # Only used to obtain the batch size
      cache,
      tokens_ids_to_logits,
      beam_size=beam_size,
      alpha=brevity_penalty,
      eos_id=config.eos_id,
      max_decode_len=config.max_decode_len)

  # Beam search returns [device_batch_size, n_beam, n_length + 1] with beam
  # dimension sorted in increasing order of log-probability.
  # Keep the highest scoring beam sequence, drop first dummy 0 token.
  # Gather those beam sequences across all devices across replicas
  predicted = beam_seqs[:, -1, 1:]
  outputs = {
      'key': batch['key'],
      'pred': predicted,
      'ref': batch['raw_caption'],
      'batch_mask': batch['batch_mask']
  }

  return outputs


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
  test_step_pmapped = jax.pmap(
      functools.partial(
          test_step,
          flax_model=model.flax_model,
          config=config,
          debug=config.debug_eval),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1,),
  )
  init_cache_pmapped = jax.pmap(
      functools.partial(
          initialize_cache, flax_model=model.flax_model, config=config
      ),
      axis_name='batch',
  )
  return (
      train_step_pmapped,
      eval_step_pmapped,
      test_step_pmapped,
      init_cache_pmapped,
  )


def init_state(
    model: base_model.BaseModel,
    dataset: dataset_utils.Dataset,
    config: ml_collections.ConfigDict,
    workdir: str,
    rng: jnp.ndarray,
):
  """Initialize the model state."""

  input_shapes = dataset.meta_data['input_shape']
  input_dtype = dataset.meta_data.get('input_dtype', jnp.float32)
  target_spec = (dataset.meta_data['target_shape'],
                 dataset.meta_data['target_dtype'])
  encoder_model = config.model.get('encoder_model', 'mbt')

  if isinstance(input_shapes, dict):
    final_spec_list = []
    for mod in SUPPORTED_MODALITIES:
      if mod in input_shapes:
        logging.info('Modality %s is present for this dataset', mod)
        final_spec_list.append((input_shapes[mod], input_dtype))
      else:
        final_spec_list.append(None)
    final_spec_list.append(target_spec)
  # Using MBT model with DMVR datasets that only return RGB
  elif encoder_model == 'mbt':
    final_spec_list = []
    for mod in SUPPORTED_MODALITIES:
      if mod == 'rgb':
        final_spec_list.append((input_shapes, input_dtype))
      else:
        final_spec_list.append(None)
    final_spec_list.append(target_spec)
  else:
    final_spec_list = [(input_shapes, input_dtype)]
    final_spec_list.append(target_spec)
  if config.get('predict_masked_word', False):
    final_spec_list.append((dataset.meta_data['masked_token_idxs_shape'],
                            dataset.meta_data['masked_token_idxs_dtype']))
    final_spec_list.append((dataset.meta_data['masked_token_idx_masks_shape'],
                            dataset.meta_data['masked_token_idx_masks_dtype']))
    final_spec_list.append((dataset.meta_data['masked_word_targets_shape'],
                            dataset.meta_data['masked_word_targets_dtype']))

  # Initialize model.
  logging.debug('Initializing model...')
  rng, init_rng = jax.random.split(rng)
  params, model_state, num_params, gflops = train_utils.initialize_model(
      model_def=model.flax_model,
      input_spec=final_spec_list,
      config=config,
      rngs=init_rng)
  logging.info('The model has %d params, uses %d gflops', num_params, gflops)

  # Create the optimizer.
  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  optimizer = jax.jit(
      optimizers.get_optimizer(config).create, backend='cpu')(
          params)
  del params  # Do not keep a copy of the initial params.

  rng, train_rng = jax.random.split(rng)
  # The variable global_step indicates the last completed step.
  # Because the step number is incremented in the training loop and we start
  # with step=0 (zero-shot evaluation), we set global_step=-1 here.
  global_step = -1
  train_state = train_utils.TrainState(
      global_step=global_step,
      optimizer=optimizer,
      model_state=model_state,
      rng=train_rng,
      accum_train_time=0,
  )
  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state
    )

  if start_step == -1 and config.get('checkpoint_path', None):
    restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
        config.checkpoint_path, train_state, assert_exist=True
    )
    restored_model_cfg = copy.deepcopy(config)
    train_state = model_utils.initialise_from_train_state(
        config,
        train_state,
        restored_train_state,
        restored_model_cfg,
        restore_output_proj=config.init_from.get('restore_output_proj', False),
        restore_from_old_format=config.get('restore_from_old_format', True),
    )
    else:
      # TODO(valgab): Seperately intialise encoder and decoder
      pass
  elif start_step == -1:
    logging.info('Training completely from scratch. '
                 'Not restoring from any checkpoint.')
  return train_state, start_step


def initialize_cache(flax_model, batch, config):
  """Initialize a cache for a given input shape and max decode length."""

  if config.model.encoder_model == 'vivit':
    video_inputs = batch['inputs']
  elif config.dataset_configs.return_as_dict:
    video_inputs = [
        batch['inputs'].get('rgb', None), batch['inputs'].get('flow', None),
        batch['inputs'].get('spectrogram',
                            None), batch['inputs'].get('waveform', None),
        batch['inputs'].get('text', None)
    ]
  # When using DMVR datasets which only have RGB
  else:
    video_inputs = [batch['inputs'], None, None, None, None]

  target_shape = tuple(batch['targets'].shape[:-1]) + (config.max_decode_len,)
  target_dtype = jnp.int32
  dummy_target = jnp.ones(target_shape, target_dtype)
  # Remove the "num_captions" dimension
  dummy_target = jnp.squeeze(dummy_target, axis=-2)

  initial_variables = flax_model.init(
      jax.random.PRNGKey(0),
      *video_inputs,
      dummy_target,
      decode=True,
      train=False)
  return initial_variables['cache']


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
    prefix,
):
  """Evaluate the model and write the summary."""

  eval_metrics = []
  # Sync model state across replicas.
  train_state = train_utils.sync_model_state_across_replicas(train_state)

  logging.info('Total number of eval steps is %s', eval_steps)
  for step in range(eval_steps):
    with jax.profiler.StepTraceAnnotation('eval', step_num=step):
      eval_batch = next(iterator)
      metrics = eval_step_fn(train_state, eval_batch)
      # Fetch metrics to host and store.
      eval_metrics.append(train_utils.unreplicate_and_get(metrics))

  return train_utils.log_eval_summary(
      step=train_iteration,
      eval_metrics=eval_metrics,
      extra_eval_summary=None,
      prefix=prefix,
      writer=writer,
      key_separator='/')


def decode_ints_to_string(ints):
  """Decode a sequence of ASCII values into a string."""
  char_list = [chr(char_int) for char_int in ints if char_int]
  return ''.join(char_list)


def test_and_log_summary(*, train_state: train_utils.TrainState,
                         writer: metric_writers.MetricWriter, iterator,
                         eval_step_fn, init_cache_fn, eval_steps,
                         train_iteration, tokenizer, workdir, prefix):
  """Eval the model and write the summary."""
  logging.info('Generating tokens for the test set.')
  output_dicts = {}
  # Sync model state across replicas.
  train_state = train_utils.sync_model_state_across_replicas(train_state)

  logging.info('Total number of test steps is %s', eval_steps)
  for step in range(eval_steps):
    with jax.profiler.StepTraceAnnotation('test', step_num=step):
      eval_batch = next(iterator)
      cache = init_cache_fn(batch=eval_batch)
      outputs = eval_step_fn(
          train_state=train_state, batch=eval_batch, cache=cache)

      if 'noise_word_mask' in eval_batch and eval_batch['noise_word_mask'].size:
        outputs['noise_word_mask'] = eval_batch['noise_word_mask']

      outputs = to_cpu(
          jax.pmap(lambda x: jax.lax.all_gather(x, 'batch'), 'batch')(outputs))

      for i, valid in enumerate(outputs['batch_mask']):
        if valid:
          k = decode_ints_to_string(outputs['key'][i])

          if k not in output_dicts:
            output_dicts[k] = {
                'ref': decode_ints_to_string(outputs['ref'][i]),
                'hyp': decode_tokens(tokenizer, outputs['pred'][i]),
            }
            if 'noise_word_mask' in outputs:
              output_dicts[k]['word_mask'] = outputs['noise_word_mask'][i]

  logging.info('%s: %d outputs.', prefix, len(output_dicts))

  # TODO(phseo): Currently, we are iterating over 1.x times the total eval
  #   dataset size to handle different number of batches per host. This should
  #   be changed later to properly iterate over the dataset exactly one time.
  keys = []
  refs = []
  preds = []
  for k, v in output_dicts.items():
    keys.append(k)
    refs.append(v['ref'])
    preds.append(v['hyp'])

  wer, rates = metrics_utils.word_error_rate(refs, preds)
  # Save decoded samples for tensorboard.
  exemplars = ''
  for n in np.random.choice(np.arange(len(preds)), 8):
    exemplars += f'{keys[n]}\n\nGT: {refs[n]}\n\nHY: {preds[n]}\n\n'
  logging.info(f'{prefix}: ' + exemplars.replace('%', '').replace('\n', ' '))

  if jax.host_id() == 0:
    prefix_local = re.sub(r'[\[\]]', '_', prefix)
    write_examples_to_disk(workdir, prefix_local, train_iteration, keys, refs,
                           preds)

  writer.write_texts(train_iteration, {f'{prefix}_samples': exemplars})

  (del_rate, ins_rate, sub_rate, cor_rate) = rates
  eval_dict = {
      'wer': (wer, 1),
      'del_rate': (del_rate, 1),
      'ins_rate': (ins_rate, 1),
      'sub_rate': (sub_rate, 1),
      'cor_rate': (cor_rate, 1),
  }

  return train_utils.log_eval_summary(
      step=train_iteration,
      eval_metrics=[eval_dict],
      extra_eval_summary=None,
      prefix=prefix,
      writer=writer,
      key_separator='/')


def write_examples_to_disk(workdir, prefix, train_iteration, keys, refs, preds):
  """Convert examples to dict and write to json file."""
  res = {}
  for i, key in enumerate(keys):
    res[key] = {}
    res[key]['groundtruth'] = refs[i]
    res[key]['predictions'] = preds[i]
    res[key]['dataset'] = prefix

  out_path = os.path.join(workdir, prefix, f'{train_iteration:012d}.json')
  logging.info('Writing results to file %s', out_path)
  gfile.makedirs(os.path.dirname(out_path))
  with gfile.GFile(out_path, 'w') as f:
    f.write(json.dumps(res, indent=4, sort_keys=True))


def set_tokenizer(tokenizer_config):
  """Set the tokenizer."""
  tokenizer_type = tokenizer_config.get('tokenizer_type', 'bert')
  tokenizer_vocab = tokenizer_config.get('tokenizer_vocab', None)
  if tokenizer_type == 'bert':
    assert tokenizer_vocab
    tokenizer = tokenizers.BertTokenizer(tokenizer_vocab)
  else:
    raise ValueError('Tokenizer not supported')
  vocab_size = int(tokenizer.vocab_size)
  logging.info('vocab_size %d', vocab_size)
  logging.info('EOS token: %d', tokenizer.eos_token)
  # Init the TF models of the tokenizer.
  tokenizer.initialize()
  return tokenizer


def get_num_training_steps(
    config: ml_collections.ConfigDict,
    dataset_metadata: Dict[str, Any]) -> Tuple[int, Optional[int]]:
  """Calculates the total number of training step and possibly steps_per_epoch.

  The main raining loop is based on number of training steps. Thus, for datasets
  that we want to train based on number of epochs, we need to calculate the
  total number of training steps. This function looks for `num_training_steps`
  in config, if it exists it returns that as the total step and `None` as
  `steps_per_epoch`. If num_training_steps doesn't exist, then it looks for
  `num_training_epochs` and given the size of training data calculates the total
  steps and steps_per_epoch. In this computation, we assume that
  drop_remainder=True.

  Args:
    config: Configuration of the experiment.
    dataset_metadata: Meta-data that is generated by the dataset_builder.

  Returns:
    total_steps: Total number of training steps.
    steps_per_epoch: Number of steps in every epoch.
  """
  # We either use num_training_epochs or num_training_steps.
  steps_per_epoch = dataset_metadata.get('num_train_examples',
                                         0) // config.batch_size

  if config.get('num_training_steps', None) is not None:
    assert not config.get('num_training_epochs')
    return config.num_training_steps, steps_per_epoch or None
  else:
    assert config.num_training_epochs and not config.get('num_training_steps')
    return int(steps_per_epoch * config.num_training_epochs), steps_per_epoch


def get_noise_info(configs_dict):
  """Make noise config strings to show metrics in XM."""
  noise_info = ['[clean]']
  for noise_types, configs in configs_dict.items():
    noise_types = noise_types.split(',')
    for nt in noise_types:
      assert nt in ds_utils.VALID_EVAL_NOISE_TYPES
    for config in configs:
      noise_str = []
      if 'environment_noise' in noise_types:
        en_config = config['environment_noise_configs']
        noise_str += ['EnvN:%0.1f' % en_config['snr']]
      if 'packet_loss_noise' in noise_types:
        pln_config = config['packet_loss_noise_configs']
        noise_str += ['PacN:%d_%0.1f' % (pln_config['max_num_bursts'],
                                         pln_config['max_length_rate'])]
      noise_str = '[' + ';'.join(noise_str) + ']'
      noise_info.append(noise_str)
  return noise_info


def train_and_eval(
    rng: np.ndarray, config: ml_collections.ConfigDict, *, workdir: str,
    writer: Any, model_cls,
    dataset) -> Tuple[train_utils.TrainState, Any, Dict[str, Any]]:
  """Train (and occasionally evaluate) the model.

  Args:
    rng: JAX prng key.
    config: The configuration of the experiment.
    workdir: Where to checkpoint and write the summaries.
    writer: Summary writer object.
    model_cls: The model class used to instantiate the model.
    dataset: The dataset for training and evaluation.

  Returns:
    A tuple with:
      * the state that has the state of training (including current
        global_step, model_state, rng, and the optimizer)
      * a dictionary with the train_summary
      * a dictionary with the evaluation summary
  """
  logging.info('Starting train and eval')

  lead_host = jax.host_id() == 0
  logging.info('Number of processes is %s', jax.process_count())

  # Tokenizer
  tokenizer = set_tokenizer(config.dataset_configs.get('tokenizer'))

  model = model_cls(config, dataset.meta_data)
  (
      train_step_pmapped,
      eval_step_pmapped,
      test_step_pmapped,
      init_cache_pmapped,
  ) = pmapped_steps(model, config)

  train_state, start_step = init_state(model, dataset, config, workdir, rng)  # pytype: disable=wrong-arg-types  # jax-ndarray
  train_state = jax_utils.replicate(train_state)

  del rng  # So that we don't mistakenly re-use it.

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = get_num_training_steps(config,
                                                        dataset.meta_data)
  logging.info('Total number of training steps %d', total_steps)
  logging.info('Steps per epoch %d', steps_per_epoch)
  log_eval_steps = config.get('log_eval_steps', steps_per_epoch)
  log_summary_steps = config.get('log_summary_steps', log_eval_steps)

  # Calculate the number of evaluation steps.
  num_eval_examples = dataset.meta_data['num_eval_examples']
  total_eval_steps = int(
      np.ceil(num_eval_examples / (config.get('eval_batch_size'))))
  steps_per_eval = config.get('steps_per_eval', total_eval_steps)
  logging.info('Total number of eval steps %d', total_eval_steps)
  logging.info('Steps per eval %d', steps_per_eval)

  # Calculate the number of test steps.
  total_test_steps = int(
      np.ceil(dataset.meta_data['num_test_examples'] /
              (config.get('eval_batch_size'))))
  steps_per_test = config.get('steps_per_test', total_test_steps)
  logging.info('Total number of test steps %d', total_test_steps)
  logging.info('Steps per test %d', steps_per_test)

  chrono = train_utils.Chrono(
      first_step=start_step + 1,
      total_steps=total_steps,
      steps_per_epoch=steps_per_epoch,
      global_bs=config.batch_size,
      accum_train_time=int(jax_utils.unreplicate(train_state.accum_train_time)))

  logging.info('Start training from step %d', start_step + 1)
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps, writer=writer)
  if jax.process_index() == 0:
    hooks.append(report_progress)
    if config.get('xprof', True):
      hooks.append(
          periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  summary_builder = SummaryBuilder([], [])
  for step in range(start_step + 1, total_steps + 1):
    # Step 0 only consists in a zero-shot evaluation.
    if step > 0:
      chrono.resume()
      train_batch = next(dataset.train_iter)
      with jax.profiler.StepTraceAnnotation('train', step_num=step):
        train_state, t_metrics, lr = train_step_pmapped(train_state,
                                                        train_batch)
      for h in hooks:
        # Catch exception in case XProf fails.
        try:
          h(step)
        except ValueError as error:
          logging.exception('Hook failed: %r', error)
      summary_builder.update(t_metrics, {'lr': lr})
      chrono.pause()

      # Log the train summary every `log_summary_steps`.
      if (step % log_summary_steps == 0) or (step == total_steps):
        if lead_host:
          chrono.tick(step, writer)
        train_summary = summary_builder.write(writer, step)
    else:
      train_summary = None

    # Evaluate every `log_eval_steps`.
    should_eval = (step % log_eval_steps == 0) or (step == total_steps)

    if should_eval:
      # TODO(valgab): Make the evaluation on a single host. Because of the way
      # the shards are split between hosts, evaluating on a
      # multi host is wrong because some eval examples are missed or repeated.
      splits = {
          config.dataset_configs.tables.val.name:
              (dataset.valid_iter, steps_per_eval),
          config.dataset_configs.tables.test.name:
              (dataset.test_iter, steps_per_test),
      }
      for split, (iterators, nb_steps) in splits.items():
        if isinstance(iterators, list):
          noise_descs = get_noise_info(
              config.dataset_configs.spec_from_wave_eval_noise_configs)
        else:
          iterators = [iterators]
          noise_descs = ['[clean]']
        assert len(iterators) == len(noise_descs)
        with report_progress.timed('eval'):
          for iterator, noise_desc in zip(iterators, noise_descs):
            eval_summary = eval_and_log_summary(
                train_state=train_state,
                iterator=iterator,
                eval_step_fn=eval_step_pmapped,
                eval_steps=nb_steps,
                writer=writer,
                train_iteration=step,
                prefix=split + noise_desc)

        with report_progress.timed('test'):
          logging.info('Starting testing')
          for iterator, noise_desc in zip(iterators, noise_descs):
            test_summary = test_and_log_summary(
                train_state=train_state,
                iterator=iterator,
                eval_step_fn=test_step_pmapped,
                init_cache_fn=init_cache_pmapped,
                eval_steps=nb_steps,
                writer=writer,
                train_iteration=step,
                tokenizer=tokenizer,
                workdir=workdir,
                prefix=split + noise_desc)
        # Free up some space.
        del test_summary
        writer.flush()

    # Checkpoint.
    if not config.checkpoint:
      continue
    elif should_eval and step > 0:
      with report_progress.timed('checkpoint'):
        # Sync model state across replicas.
        train_state = train_utils.sync_model_state_across_replicas(train_state)
        if lead_host:
          train_state.replace(  # pytype: disable=attribute-error
              accum_train_time=chrono.accum_train_time)
          train_utils.save_checkpoint(workdir, train_state)

  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()

  logging.info('Training completed in %d steps', step)

  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary
