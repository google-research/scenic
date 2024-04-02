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

"""Evaluation script for the COCO Caption."""

import functools
import time
from typing import Any, Optional

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from dmvr import tokenizers
import flax
from flax import jax_utils
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np

from scenic.dataset_lib import dataset_utils
from scenic.projects.streaming_dvc import caption_evaluator
from scenic.projects.streaming_dvc import densecap_evaluator
from scenic.projects.streaming_dvc import post_processing_utils
from scenic.projects.streaming_dvc.modeling import auto_regressive_decode
from scenic.projects.t5 import tokenizer as t5_tokenizer
from scenic.train_lib import train_utils

FrozenDict = flax.core.FrozenDict


def eval_step(
    train_state, batch, *,
    flax_model, debug=False):
  """Runs a single step of inference.

  eval_step is used for evaluating the dataset metrics on the whole dataset. It
  does not need the ground truth data and does not compute the losses.

  Args:
    train_state: TrainState containing the model parameters.
    batch: The validation data.
    flax_model: model definition.
    debug: bool.
  Returns:
    targets: ground truth.
    predictions: model predictions.
    metrics: dict.
  """
  variables = {
      'params': train_state.params,
  }
  kwargs = {}
  if 'context_tokens' in batch['label']:
    # Prompts or questions in QA.
    kwargs['context_tokens'] = batch['label']['context_tokens']
  if 'image_features' in batch:
    kwargs['image_features'] = batch['image_features']
  predictions = flax_model.apply(
      variables,
      batch['inputs'],
      preprocess=True,
      train=False,
      mutable=False,
      debug=debug,
      **kwargs)
  predictions = auto_regressive_decode.autoregressive_predict(
      flax_model, variables['params'],
      predictions,
      method=flax_model.decode_method,
      beam_size=flax_model.decode_beam_size,
      brevity_penalty_alpha=flax_model.decode_brevity_penalty_alpha,
      feature_key=flax_model.decode_feature_key,
  )
  metrics = {}
  if 'batch_mask' in batch:
    batch_mask = batch['batch_mask']
  else:
    batch_mask = jnp.ones((predictions['text_tokens'].shape[0],))
  targets = {'label': batch['label'], 'batch_mask': batch_mask}
  predictions = jax.lax.all_gather(predictions, 'batch')
  targets = jax.lax.all_gather(targets, 'batch')
  return targets, predictions, metrics


def streaming_dense_eval_step(
    train_state, batch, *,
    flax_model, debug=False):
  """Runs a single step of inference with intermediate outputs.

  Compared to the regular eval_step, the main difference here is we forward the
  language decoder num_dense_outputs times, each time with different visual
  features and with context from previous steps. The visual_feature here after
  the visual backbone should be in shape (batch_size, num_dense_outputs,
  num_tokens, hidden_size), instead of (batch_size, num_tokens, hidden_size).

  Args:
    train_state: TrainState containing the model parameters.
    batch: The validation data.
    flax_model: model definition.
    debug: bool.
  Returns:
    targets: ground truth.
    predictions: model predictions.
    metrics: dict.
  """
  variables = {
      'params': train_state.params,
  }
  kwargs = {}
  if 'image_features' in batch:
    kwargs['image_features'] = batch['image_features']
  predictions = flax_model.apply(
      variables,
      batch['inputs'],
      preprocess=True,
      train=False,
      mutable=False,
      debug=debug,
      **kwargs)
  num_dense_outputs = flax_model.num_dense_outputs_test if (
      flax_model.num_dense_outputs_test > 0) else flax_model.num_dense_outputs
  batch_size, max_cap_len = predictions['begin_tokens'].shape
  if flax_model.early_segments_as_context:
    context = jnp.concatenate([
        jnp.broadcast_to(jnp.asarray(
            [flax_model.begin_token_id, flax_model.end_token_id],
            dtype=jnp.int32)[None], (batch_size, 2)),
        jnp.zeros([batch_size, max_cap_len - 2], dtype=jnp.int32)], axis=1)
  else:
    context = None
  context_without_timestamp = context
  raw_streaming_feature = predictions['raw_streaming_feature'] if (
      'raw_streaming_feature' in predictions) else None
  all_visual_features = predictions['visual_features']
  # (batch_size, num_dense_outputs, num_tokens, hidden_size)
  assert all_visual_features.shape[1] == num_dense_outputs

  misc_predictions = {}
  for i in range(num_dense_outputs):
    context_input = context if not flax_model.no_timestamp_in_context else (
        context_without_timestamp)
    predictions = auto_regressive_decode.autoregressive_predict(
        flax_model, variables['params'],
        {'visual_features': all_visual_features[:, i],
         'begin_tokens': predictions['begin_tokens'],
         'context_tokens': context_input},
        method=flax_model.decode_method,
        beam_size=flax_model.decode_beam_size,
        brevity_penalty_alpha=flax_model.decode_brevity_penalty_alpha,
        feature_key=flax_model.decode_feature_key,
    )
    misc_predictions[f'text_tokens_{i}'] = predictions['text_tokens']
    misc_predictions[f'context_{i}'] = context  # for debug
    text_tokens = predictions['text_tokens']  # (batch_size, max_cap_len)
    if flax_model.remove_segments_from_wrong_checkpoint:
      checkpoint_size = (flax_model.num_bins - 1) // num_dense_outputs + 1
      text_tokens = post_processing_utils.remove_segments_from_wrong_checkpoint(
          text_tokens,
          max_end_time=checkpoint_size * (i + 1),
          ori_vocab_size=flax_model.vocab_size - flax_model.num_bins,
          bos_id=flax_model.begin_token_id,
          eos_id=flax_model.end_token_id)
    if flax_model.early_segments_as_context:
      if flax_model.copy_context:
        context = text_tokens
      else:
        context = (
            post_processing_utils.remove_padding_and_concate_and_pad_tokens(
                [context, text_tokens],
                flax_model.begin_token_id, flax_model.end_token_id,
                flax_model.max_caption_length))  # (batch_size, max_cap_len)
      if flax_model.normalize_early_timestamps and (i < num_dense_outputs - 1):
        # We don't need to rescale the context for the last checkpoint, as this
        # is directly our outputs.
        ori_vocab_size = flax_model.vocab_size - flax_model.num_bins
        is_time_token = context >= ori_vocab_size
        context = jnp.where(
            is_time_token,
            (context - ori_vocab_size) * (i + 1) // (i + 2) + ori_vocab_size,
            context,
        )
      context_without_timestamp = post_processing_utils.remove_timestamps(
          context, ori_vocab_size=flax_model.vocab_size - flax_model.num_bins)

  # context is now the concatenated outputs of all outputs.
  if flax_model.early_segments_as_context:
    predictions['text_tokens'] = context
  if debug:
    predictions.update(misc_predictions)
  if raw_streaming_feature is not None:
    predictions['raw_streaming_feature'] = raw_streaming_feature
  metrics = {}
  if 'batch_mask' in batch:
    batch_mask = batch['batch_mask']
  else:
    # FlexIO does not currently return a batch mask.
    batch_mask = jnp.ones(
        (predictions['text_tokens'].shape[0],))  # pytype: disable=attribute-error
  targets = {'label': batch['label'], 'batch_mask': batch_mask}
  predictions = jax.lax.all_gather(predictions, 'batch')
  targets = jax.lax.all_gather(targets, 'batch')
  return targets, predictions, metrics


def process_and_fetch_to_host(pred_or_tgt, batch_mask):
  """Used to collect predictions and targets of the whole valid/test set.

  Forked from scenic/projects/baselines/detr/train_utils.py

  Args:
    pred_or_tgt: pytree; A pytree of jnp-arrays where leaves are of shape
      `[num_devices, bs, X,...,Y]`.
    batch_mask: A nd-array of shape `[num_devices, bs]`, where zero values
      indicate padded examples.

  Returns:
    A list of length num_devices * bs of items, where each item is a tree with
    the same structure as `pred_or_tgt` and each leaf contains a single example.
  """
  # Fetch to host in a single call.
  pred_or_tgt, batch_mask = jax.device_get((pred_or_tgt, batch_mask))
  batch_mask = np.array(batch_mask).astype(bool)

  def _split_mini_batches(x):
    # Filter out padded examples.
    x = x[batch_mask]
    # Split minibatch of examples into a list of examples.
    x_list = np.split(x, x.shape[0], axis=0)
    # Squeeze out the dummy dimension.
    return jax.tree_util.tree_map(lambda x: np.squeeze(x, axis=0), x_list)

  leaves, treedef = jax.tree_util.tree_flatten(pred_or_tgt)

  batch_shape = batch_mask.shape
  assert all([leaf.shape[:2] == batch_shape for leaf in leaves]), (
      'Inconsistent batch shapes.')

  # Split batched leaves into lists of examples:
  leaves = list(map(_split_mini_batches, leaves))

  # Go from leaf-lists to list of trees:
  out = []
  if leaves:
    num_examples = np.sum(batch_mask, dtype=np.int32)
    for example_ind in range(num_examples):
      out.append(treedef.unflatten([leaf[example_ind] for leaf in leaves]))
  return out


def inference_on_dataset(
    flax_model: Any,
    train_state: train_utils.TrainState,
    dataset: dataset_utils.Dataset,
    eval_batch_size: int = 1,
    is_host: bool = False,
    save_dir: str = '',
    step: Optional[int] = None,
    config: ml_collections.ConfigDict = ml_collections.ConfigDict(),
    ) -> Any:
  """The main evaluation loop. Run evaluation on the whole validation set.

  Args:
    flax_model: Flax model (an instance of nn.Module).
    train_state: train_state that contains the model parameters.
    dataset: The dataset that has valid_iter and meta_data.
    eval_batch_size: integer. Batch size per-device in evaluation.
    is_host: bool: whether its the host machine. During multi-machine training,
      we only hold the evaluating data in one of the machines. The machine with
      `jax.process_index() == 0` sets `is_host` to True and will gather data
      from other machines and do the evaluation. Other machines set `is_host`
      as False.
    save_dir: string: where to save the json prediction
    step: Optional integer of the training step. The step is appended to the
      serialised results if provided.
    config: config dict
  Returns:
    evaluation results.
  """
  global_metrics_evaluator = None  # Only run eval on the is_host node.
  tokenizer = None
  evaluator_name = config.get('evaluator', 'caption')
  if is_host:
    annotations_loc = config.get('dataset_configs', {}).get(
        'test_annotation_path', '')
    tokenizer_weight_path = config.get('dataset_configs', {}).get(
        'tokenizer_weight_path')
    logging.info('tokenizer_weight_path: %s', tokenizer_weight_path)
    if tokenizer_weight_path == 't5':
      tokenizer = t5_tokenizer.build_dmvr_sp_model()
    else:
      tokenizer = tokenizers.BertTokenizer(tokenizer_weight_path)
    tokenizer.initialize()

    if evaluator_name == 'caption':
      eval_meteor_spice = config.get('eval_meteor_spice', False)
      global_metrics_evaluator = caption_evaluator.CaptionEvaluator(
          annotations_loc, eval_meteor_spice=eval_meteor_spice, step=step)
    elif evaluator_name == 'densecap':
      global_metrics_evaluator = densecap_evaluator.DenseCapEvaluator(
          annotations_loc=annotations_loc, tokenizer=tokenizer,
          num_bins=config.model.num_bins, step=step)
    else:
      raise NotImplementedError(evaluator_name)
    global_metrics_evaluator.clear()

  eval_step_fn = eval_step if config.model.model_name not in [
      'streaming_dense_model',
      'streaming_vid2seq'] else streaming_dense_eval_step
  eval_step_pmapped = jax.pmap(
      functools.partial(
          eval_step_fn,
          flax_model=flax_model,
          debug=config.get('debug_eval', False),
      ),
      axis_name='batch', donate_argnums=(1,),
  )

  eval_metrics = []
  eval_step_multiplier = config.get('eval_step_multiplier', 1.)
  total_eval_steps = int(np.ceil(eval_step_multiplier * dataset.meta_data[
      'num_eval_examples'] / eval_batch_size))
  for eval_step_i in range(total_eval_steps):
    if eval_step_i % 100 == 0:
      logging.info('Running eval step %d', eval_step_i)
    eval_batch = next(dataset.valid_iter)
    eval_batch_all_hosts, predictions_all_hosts, metrics = eval_step_pmapped(
        train_state, eval_batch)
    eval_metrics.append(train_utils.unreplicate_and_get(metrics))

    if is_host:
      eval_batch_all_hosts = jax_utils.unreplicate(eval_batch_all_hosts)
      predictions_all_hosts = jax_utils.unreplicate(
          predictions_all_hosts)
      # Collect preds and labels to be sent for computing global metrics.
      labels = process_and_fetch_to_host(
          eval_batch_all_hosts['label'], eval_batch_all_hosts['batch_mask'])
      results = process_and_fetch_to_host(
          predictions_all_hosts, eval_batch_all_hosts['batch_mask'])

      for pred, label in zip(results, labels):
        if evaluator_name != 'densecap':
          assert evaluator_name == 'caption'
          pred = tokenizer.indices_to_string(pred['text_tokens'][1:].tolist())
          label['captions'] = [
              tokenizer.indices_to_string(x[1:].tolist())
              for x in label['text_tokens']]
        global_metrics_evaluator.add_example(  # pytype: disable=attribute-error
            prediction=pred, target=label)

  results = None
  if is_host:
    logging.info('Number of eval examples: %d', len(global_metrics_evaluator))
    results = global_metrics_evaluator.compute_metrics(  # pytype: disable=attribute-error
        save_dir=save_dir, clear_annotations=False,
        skip_evaluate=config.get('skip_evaluate', False))
  return results, eval_metrics


def evaluate(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Any,
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
):
  """Prepares the items needed to run the evaluation.

  Args:
    rng: JAX PRNGKey.
    config: Configurations of the experiment.
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    dataset: The dataset that has train_iter, eval_iter, meta_data, and
      optionally, test_iter.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.
  """
  is_host = jax.process_index() == 0

  checkpoint_config = config
  checkpoint_path = config.weights
  model = model_cls(checkpoint_config, dataset.meta_data)

  checkpoint_data = checkpoints.restore_checkpoint(checkpoint_path, None)
  if 'params' in checkpoint_data:
    params = checkpoint_data['params']
  else:
    # Old Scenic train state format.
    params = checkpoint_data['optimizer']['target']
  train_state = train_utils.TrainState(
      global_step=0,
      params=FrozenDict(params),
      rng=rng)
  train_state = jax_utils.replicate(train_state)
  del checkpoint_data, params

  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=0, writer=writer)

  hooks = []
  if is_host:
    hooks.append(report_progress)
  if config.get('xprof', True) and is_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  start_time = time.time()
  with report_progress.timed('eval'):
    train_state = train_utils.sync_model_state_across_replicas(train_state)
    eval_results, eval_metrics = inference_on_dataset(
        model.flax_model,
        train_state,
        dataset,
        eval_batch_size=eval_batch_size,
        is_host=is_host,
        save_dir=workdir,
        config=config,
    )
    train_utils.log_eval_summary(
        step=0,
        eval_metrics=eval_metrics,
        extra_eval_summary=eval_results,
        writer=writer,
    )
  duration = time.time() - start_time
  logging.info('Done with evaluation: %.4f sec.', duration)
  writer.flush()
  train_utils.barrier()
