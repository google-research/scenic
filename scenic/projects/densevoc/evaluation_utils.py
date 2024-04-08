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

"""Utilities for evaluation."""

import functools
from typing import Any

from absl import logging

from dmvr import tokenizers
from flax import jax_utils
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.projects.baselines.centernet import train_utils as centernet_train_utils
from scenic.projects.densevoc import densevoc_evaluator
from scenic.projects.densevoc import vidstg_evaluator
from scenic.train_lib import train_utils


def eval_step(
    train_state, batch, *,
    model, debug=False):
  """Runs a single step of inference."""
  variables = {
      'params': train_state.params,
      **train_state.model_state,
  }
  # get detection outputs and features for text decoder.
  predictions = model.flax_model.apply(
      variables,
      batch['inputs'],
      preprocess=True,
      # padding_mask=batch['padding_mask'],
      padding_mask=jnp.ones((1, 1, 1), dtype=jnp.float32),
      train=False,
      mutable=False,
      debug=debug)
  # Run text decoder and get text outputs.
  predictions = model.autoregressive_predict(
      variables['params'], predictions)
  metrics = {}
  targets = {'label': batch['label'], 'batch_mask': batch['batch_mask']}
  predictions = jax.lax.all_gather(predictions, 'batch')
  targets = jax.lax.all_gather(targets, 'batch')
  return targets, predictions, metrics


def inference_on_image_dataset(
    model: Any,
    train_state: train_utils.TrainState,
    dataset: dataset_utils.Dataset,
    config: ml_collections.ConfigDict,
    eval_batch_size: int = 1,
    is_host: bool = False,
    save_dir: str = '') -> Any:
  """The main evaluation loop. Run evaluation on the whole validation set.

  Args:
    model: Scenic basemodel (an instance of nn.Module).
    train_state: train_state that contains the model parameters.
    dataset: The dataset that has valid_iter and meta_data.
    config: config dict.
    eval_batch_size: integer. Batch size per-device in evaluation.
    is_host: bool: whether its the host machine. During multi-machine training,
      we only hold the evaluating data in one of the machines. The machine with
      `jax.process_index() == 0` sets `is_host` to True and will gather data
      from other machines and do the evaluation. Other machines set `is_host`
      as False.
    save_dir: string: where to save the json prediction
  Returns:
    evaluation results.
  """
  eval_meteor = config.get('eval_meteor', True)
  annotations_loc = config.get('dataset_configs', {}).get(
      'test_annotation_path', None)
  debug = config.get('debug_eval', False)
  eval_cap_switch = config.get('eval_cap_switch', False)
  eval_chota = config.get('eval_chota', False)
  chota_caption_metric = config.get(
      'chota_caption_metric', 'cider')
  tokenizer_weight_path = config.get('dataset_configs', {}).get(
      'tokenizer_weight_path')

  global_metrics_evaluator = None  # Only run eval on the is_host node.
  tokenizer = None
  if is_host:
    global_metrics_evaluator = densevoc_evaluator.DensecapGlobalEvaluator(
        annotations_loc=annotations_loc,
        ignore_empty_string=True,
        eval_cap_switch=eval_cap_switch,
        eval_meteor=eval_meteor,
        eval_chota=eval_chota,
        chota_caption_metric=chota_caption_metric)
    global_metrics_evaluator.clear()
    tokenizer = tokenizers.BertTokenizer(tokenizer_weight_path)
    tokenizer.initialize()

  eval_step_pmapped = jax.pmap(
      functools.partial(
          eval_step,
          model=model,
          debug=debug,
      ),
      axis_name='batch', donate_argnums=(1,),
  )

  eval_metrics = []
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
  for eval_step_i in range(total_eval_steps):
    if eval_step_i % 100 == 0:
      logging.info('Running eval step %d', eval_step_i)
    if is_host:
      xm_utils.get_xm_note_writer()(
          f'Running eval step {eval_step_i} / {total_eval_steps}')
    eval_batch = next(dataset.valid_iter)

    eval_batch_all_hosts, predictions_all_hosts, metrics = eval_step_pmapped(
        train_state, eval_batch)
    eval_metrics.append(train_utils.unreplicate_and_get(metrics))

    if global_metrics_evaluator is not None:
      eval_batch_all_hosts = jax_utils.unreplicate(eval_batch_all_hosts)
      predictions_all_hosts = jax_utils.unreplicate(predictions_all_hosts)

      # Collect preds and labels to be sent for computing global metrics.
      labels = centernet_train_utils.split_batch_and_fetch_to_host(
          eval_batch_all_hosts['label'], eval_batch_all_hosts['batch_mask'])
      labels = jax.tree_util.tree_map(np.asarray, labels)
      results = centernet_train_utils.split_batch_and_fetch_to_host(
          predictions_all_hosts, eval_batch_all_hosts['batch_mask'])

      for pred, label in zip(results, labels):
        texts = [tokenizer.indices_to_string(
            x[1:].tolist()) for x in pred['text_tokens']]
        detection_pred = (
            pred['detection_boxes'], pred['detection_scores'],
            pred['detection_classes'], texts)
        global_metrics_evaluator.add_example(
            prediction=detection_pred, target=label)

  results = None
  if global_metrics_evaluator is not None:
    logging.info('Number of eval examples: %d', len(global_metrics_evaluator))
    if save_dir:
      global_metrics_evaluator.write_pred_annotations_to_file(save_dir)
    results = global_metrics_evaluator.compute_metrics()
  return results, eval_metrics


def grounding_step(
    train_state, batch, model, debug=False):
  """Runs a single step of inference."""
  del debug
  variables = {
      'params': train_state.params,
      **train_state.model_state,
  }
  # Get detection outputs and features for text decoder.
  inputs = batch['inputs']  # (B, T, H, W, 3)
  caption_tokens = batch['caption_tokens']  # (B, max_size)
  b, t, h, w, _ = inputs.shape
  inputs = inputs.reshape(b * t, h, w, 3)  # (B, T, H, W, 3) -> (BT, H, W, 3)
  predictions = model.flax_model.apply(
      variables,
      inputs,
      preprocess=True,
      padding_mask=jnp.ones((1, 1, 1), dtype=jnp.float32),
      train=False,
      mutable=False,
      )
  caption_tokens = jnp.broadcast_to(
      caption_tokens, (b * t, caption_tokens.shape[1]))
  predictions = model.compute_sentence_likelihood(
      variables['params'], predictions, caption_tokens)
  del predictions['begin_tokens']
  del predictions['object_features']
  predictions = jax.tree_util.tree_map(
      lambda x: x.reshape((b, t) + x.shape[1:]), predictions)
  targets = {'label': batch['label'], 'batch_mask': batch['batch_mask']}
  predictions = jax.lax.all_gather(predictions, 'batch')
  targets = jax.lax.all_gather(targets, 'batch')
  return targets, predictions


def densecap_step(train_state, batch, model, debug=False):
  """Runs a single step of dense caption."""
  del debug
  variables = {
      'params': train_state.params,
      **train_state.model_state,
  }
  # Get detection outputs and features for text decoder.
  inputs = batch['inputs']  # (B, T, H, W, 3)
  b, t, h, w, _ = inputs.shape
  inputs = inputs.reshape(b * t, h, w, 3)  # (B, T, H, W, 3) -> (BT, H, W, 3)
  kwargs = {}

  predictions = model.flax_model.apply(
      variables,
      inputs,
      preprocess=True,
      padding_mask=jnp.ones((1, 1, 1), dtype=jnp.float32),
      train=False,
      mutable=False,
      **kwargs,
      )

  predictions['object_features'], mask = (
      model.flax_model.update_object_feature_with_track(predictions, t))

  # Run text decoder and get text outputs.
  predictions = model.autoregressive_predict(
      variables['params'], predictions, mask=mask)
  keys_to_delete = [
      'begin_tokens', 'object_features', 'tracked_object_features',
      'asso_scores', 'track_features', 'track_feature_mask']
  for key in keys_to_delete:
    if key in predictions:
      del predictions[key]
  predictions = jax.tree_util.tree_map(
      lambda x: x.reshape((b, t) + x.shape[1:]), predictions)
  targets = {'label': batch['label'], 'batch_mask': batch['batch_mask']}
  predictions = jax.lax.all_gather(predictions, 'batch')
  targets = jax.lax.all_gather(targets, 'batch')
  return targets, predictions


def convert_to_vidstg_grounding_format(video_id, pred, label):
  """Convert outputs to VidSTG evaluator's format."""
  # Predictions are assumed to be sorted by score.
  ret = {}
  num_frames = len(label['frame_ids'])
  for i in range(num_frames):
    frame_id = label['frame_ids'][i]
    image_id = f'{video_id}_{frame_id}'
    boxes, scores = pred['detection_boxes'][i], pred['detection_scores'][i]
    num_valid_objs = (scores >= 0).sum()
    if num_valid_objs > 0:
      boxes = boxes[:num_valid_objs]
      unused_scores = scores[:num_valid_objs]
      likelihood = pred['likelihood'][i, :num_valid_objs]
      box_idx = likelihood.argmax()
      ret[image_id] = {'boxes': [boxes[box_idx].tolist()]}
  return ret


def greedy_start_end_times(scores, p=1.0):
  l, r = 0, len(scores) - 1
  while scores[l] < p and l + 1 <= r and sum(
      scores[l + 1:]) / (r - l) > sum(scores[l:]) / (r - l + 1):
    l = l + 1
  while scores[r] < p and l <= r - 1 and sum(
      scores[:r]) / (r) > sum(scores[:r + 1]) / (r + 1):
    r = r - 1
  return (l, r)


def temporal_grounding(label, pred, model_config):
  """Run temporal grounding."""
  simple_temporal_localization = model_config.get(
      'simple_temporal_localization', -1.0)
  valid_frame_inds = label['frame_ids'][label['frame_ids'] >= 0]
  sted = [int(valid_frame_inds.min()), int(valid_frame_inds.max())]
  if simple_temporal_localization >= 0:
    tl_scores = []
    num_frames = len(label['frame_ids'])
    for i in range(num_frames):
      num_valid_objs = (pred['detection_scores'][i] >= 0).sum()
      if num_valid_objs > 0:
        tl_scores.append(pred['likelihood'][i, :num_valid_objs].max())
      else:
        tl_scores.append(-1.)
    st, ed = greedy_start_end_times(tl_scores, p=simple_temporal_localization)
    sted = [
        int(label['frame_ids'][st: ed + 1].min()),
        int(label['frame_ids'][st: ed + 1].max())]
  return sted


def inference_on_video_dataset(
    model: Any,
    train_state: train_utils.TrainState,
    dataset: dataset_utils.Dataset,
    config: ml_collections.ConfigDict,
    eval_batch_size: int = 1,
    is_host: bool = False,
    save_dir: str = '',
    ):
  """The main evaluation loop. Run evaluation on the whole validation set.

  Args:
    model: Scenic basemodel (an instance of nn.Module).
    train_state: train_state that contains the model parameters.
    dataset: The dataset that has valid_iter and meta_data.
    config: config dict.
    eval_batch_size: integer. Batch size per-device in evaluation.
    is_host: bool: whether its the host machine. During multi-machine training,
      we only hold the evaluating data in one of the machines. The machine with
      `jax.process_index() == 0` sets `is_host` to True and will gather data
      from other machines and do the evaluation. Other machines set `is_host`
      as False.
    save_dir: string: where to save the json prediction
  Returns:
    evaluation results.
  """
  annotations_loc = config.get('dataset_configs', {}).get(
      'test_annotation_path', None)
  debug = config.get('debug_eval', False)
  eval_cap_switch = config.get('eval_cap_switch', False)
  eval_chota = config.get('eval_chota', False)
  chota_caption_metric = config.get(
      'chota_caption_metric', 'cider')
  tokenizer_weight_path = config.get('dataset_configs', {}).get(
      'tokenizer_weight_path')
  task = config.get('video_eval_task', 'detection')

  evaluator = None
  tokenizer = None
  if is_host:
    if task == 'grounding':
      evaluator = vidstg_evaluator.VidSTGEvaluator(annotations_loc)
    elif task == 'densecap':
      evaluator = densevoc_evaluator.DensecapGlobalEvaluator(
          annotations_loc=annotations_loc,
          ignore_empty_string=True,
          eval_cap_switch=eval_cap_switch,
          eval_meteor=True,
          eval_chota=eval_chota,
          chota_caption_metric=chota_caption_metric,
      )
      tokenizer = tokenizers.BertTokenizer(tokenizer_weight_path)
      tokenizer.initialize()
    else:
      raise NotImplementedError(task)
  kwargs = {}
  if task == 'grounding':
    step_fn = grounding_step
  elif task == 'densecap':
    step_fn = densecap_step
  else:
    raise NotImplementedError(task)
  inference_step_pmapped = jax.pmap(
      functools.partial(
          step_fn,
          model=model,
          debug=debug,
          **kwargs,
      ),
      axis_name='batch', donate_argnums=(1,),
  )

  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
  for eval_step_i in range(total_eval_steps):
    if eval_step_i % 100 == 0:
      logging.info('Running eval step %d', eval_step_i)
    if is_host:
      xm_utils.get_xm_note_writer()(
          f'Running eval step {eval_step_i} / {total_eval_steps}')
    eval_batch = next(dataset.valid_iter)

    eval_batch_all_hosts, predictions_all_hosts = (
        inference_step_pmapped(train_state, eval_batch))

    if is_host:
      eval_batch_all_hosts = jax_utils.unreplicate(eval_batch_all_hosts)
      predictions_all_hosts = jax_utils.unreplicate(predictions_all_hosts)
      labels = centernet_train_utils.split_batch_and_fetch_to_host(
          eval_batch_all_hosts['label'], eval_batch_all_hosts['batch_mask'])
      results = centernet_train_utils.split_batch_and_fetch_to_host(
          predictions_all_hosts, eval_batch_all_hosts['batch_mask'])
      for pred, label in zip(results, labels):
        h, w = np.asarray(label['orig_size'])
        input_h, input_w = np.asarray(label['size'])
        scale_factor = np.array([w, h, w, h]) / np.array(
            [input_w, input_h, input_w, input_h])
        if task == 'grounding':
          pred['detection_boxes'] *= scale_factor[None, None]
          video_id = int(label['video_id'])
          evaluator.update(  # pytype: disable=attribute-error
              convert_to_vidstg_grounding_format(video_id, pred, label))
          sted = temporal_grounding(
              label, pred, model_config=model.config.model)
          evaluator.video_update(  # pytype: disable=attribute-error
              {video_id: {'qtype': 'declarative', 'sted': sted}})
        elif task == 'densecap':
          for i in range(len(label['image_ids'])):
            if label['image_ids'][i] <= 0:  # padded frames in a video.
              break
            label['image/id'] = label['image_ids'][i]
            texts = [tokenizer.indices_to_string(
                x[1:]) for x in pred['text_tokens'][i]]
            detection_pred = (
                pred['detection_boxes'][i], pred['detection_scores'][i],
                pred['detection_classes'][i], texts)
            if 'track_ids' in pred:
              detection_pred = detection_pred + (pred['track_ids'][i],)
            evaluator.add_example(  # pytype: disable=attribute-error
                prediction=detection_pred, target=label)

  results = None
  if is_host:
    if task in ['grounding', 'densecap']:
      results = evaluator.compute_metrics()  # pytype: disable=attribute-error
      evaluator.write_pred_annotations_to_file(  # pytype: disable=attribute-error
          save_dir)
  return results, []


def override_train_model_config(train_config, config):
  """Override test-time configs."""
  for k, v in config.model.items():
    train_config.model[k] = v
  return train_config
