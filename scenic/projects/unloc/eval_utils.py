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

"""Contains evaluation utilities."""

import collections
import pickle
from typing import Any, Callable, Dict, Optional, Tuple

from absl import logging
from clu import metric_writers
from flax import jax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.projects.unloc import activity_net_eval
from scenic.projects.unloc import metrics as unloc_metrics
from scenic.projects.unloc import postprocessing_utils
from scenic.projects.vivit import evaluation_lib as vivit_evaluation_lib
from scenic.train_lib import train_utils
import sklearn.metrics
import tensorflow as tf

Batch = Dict[str, Any]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
_MICROSECONDS_PER_SECOND = 1e6


def all_gather_metrics_inputs(batch: Batch) -> Dict[str, Any]:
  """Gathers inputs for moment retrieval metrics from all devices."""

  label = batch['label']
  gt_displacements = batch['displacements']
  batch_mask = batch['batch_mask']
  caption_mask = batch['inputs']['caption_mask']
  frame_mask = batch['inputs']['input_mask']
  (label, gt_displacements,
   batch_mask, caption_mask, frame_mask) = jax.tree_util.tree_map(
       gather_flatten,
       (label, gt_displacements, batch_mask, caption_mask, frame_mask))
  return {
      'label': label,
      'displacements': gt_displacements,
      'batch_mask': batch_mask,
      'inputs': {
          'input_mask': frame_mask,
          'caption_mask': caption_mask,
      }
  }


def run_model_all_gather_results(variables: Dict[str, Any],
                                 batch: Batch,
                                 task: str,
                                 flax_model: nn.Module,
                                 train: bool,
                                 dropout_rng: Any,
                                 debug: Optional[bool] = False) -> jnp.ndarray:
  """Run models and gather results from all devices."""

  video_tokens = flax_model.apply(
      variables,
      batch['inputs'],
      train=train,
      debug=debug,
      rngs={'dropout': dropout_rng} if train else None,
      method=flax_model.encode_video)
  text_tokens = flax_model.apply(
      variables,
      batch['inputs'],
      task=task,
      train=train,
      debug=debug,
      rngs={'dropout': dropout_rng} if train else None,
      method=flax_model.encode_text)

  text_key = 'caption' if task == 'moment_retrieval' else 'class_names'
  input_word_ids = batch['inputs'][text_key]['input_word_ids']
  # Merge all captions into batch.
  input_word_ids = input_word_ids.reshape((-1, input_word_ids.shape[-1]))
  text_input_mask = batch['inputs'][text_key]['input_mask']
  text_input_mask = text_input_mask.reshape((-1, text_input_mask.shape[-1]))
  text_tokens, input_word_ids, text_input_mask = jax.tree_util.tree_map(
      gather_flatten, (text_tokens, input_word_ids, text_input_mask))
  logits = flax_model.apply(
      variables,
      video_tokens,
      text_tokens,
      task=task,
      input_word_ids=input_word_ids,
      text_input_mask=text_input_mask,
      video_input_mask=batch['inputs'].get('input_mask'),
      train=train,
      rngs={'dropout': dropout_rng} if train else None,
      method=flax_model.fuse_video_text)
  logits = gather_flatten(logits)

  return logits  # pytype: disable=bad-return-type  # jax-ndarray


def _eval_step_all_gather(
    variables: Dict[str, Any],
    batch: Batch,
    task: str,
    dataset: str,
    flax_model: nn.Module,
    metrics_fn: MetricFn,
    debug: Optional[bool] = False
) -> Tuple[Dict[str, Tuple[float, int]], jnp.ndarray]:
  """Runs a single step of moment retrieval evaluation."""

  del dataset
  logits = run_model_all_gather_results(
      variables,
      batch,
      task,
      flax_model,
      train=False,
      dropout_rng=None,
      debug=debug)
  gathered_batch = all_gather_metrics_inputs(batch)
  metrics = metrics_fn(logits, gathered_batch)
  return metrics, logits


def eval_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    task: str,
    dataset: str,
    flax_model: nn.Module,
    metrics_fn: MetricFn,
    debug: Optional[bool] = False,
    all_gather_loss: bool = False
) -> Tuple[Dict[str, Tuple[float, int]], jnp.ndarray]:
  """Runs a single step of evaluation.

  This function is branched from scenic/train_lib/classification_trainer.py.
  Here, the model function takes two additional args, `task` and `dataset`.

  Args:
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, params and optimizer state. The buffer of
      this argument can be donated to the computation.
    batch: A single batch of data. a metrics function, that given logits and
      batch of data, calculates the metrics as well as the loss.
    task: The task name, 'action_segmentation', 'highlight_detection',
      'moment_retrieval', or 'temporal_localization'.
    dataset: The dataset name.
    flax_model: A Flax model.
    metrics_fn: A metrics function, that given logits and batch of data,
      calculates the metrics as well as the loss.
    debug: Whether the debug mode is enabled during evaluation. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.
    all_gather_loss: Wether or not to gather results from all devices before
      computing metrics and loss.

  Returns:
    Calculated metrics and logits.
  """

  variables = {'params': train_state.params, **train_state.model_state}
  if all_gather_loss:
    assert task == 'moment_retrieval'
    metrics, logits = _eval_step_all_gather(variables, batch, task, dataset,
                                            flax_model, metrics_fn, debug)
  else:
    logits = flax_model.apply(
        variables,
        batch['inputs'],
        task=task,
        dataset=dataset,
        train=False,
        mutable=False,
        debug=debug)
    metrics = metrics_fn(logits, batch)
  return metrics, logits


def _get_input_batch_from_one_prompt(
    batch: Dict[str, Any],
    num_classes: int,
    prompt_index: int,
    crop_index: int,
    n_clips: int,
) -> Dict[str, Any]:
  """Prepares input data from one prompt.

  Args:
    batch: A batch of input data.
    num_classes: Number of classes.
    prompt_index: The index of prompts to process.
    crop_index: The starting index of the clip to process.
    n_clips: The number of clips to process at a time by each device. Set due to
      memory constraints.

  Returns:
    A batch of input corresponding to one prompt.
  """

  def _get_words_from_one_prompt(x: np.ndarray) -> np.ndarray:
    num_prompts = x.shape[1] // num_classes
    y = x.reshape((x.shape[0], num_classes, num_prompts, -1))
    return y[:, :, prompt_index]

  temp_input = jax.tree_util.tree_map(
      lambda x, idx=crop_index: x[idx:idx + n_clips], batch['inputs'])
  if 'class_names' in temp_input:
    temp_input['class_names'] = jax.tree_util.tree_map(
        _get_words_from_one_prompt, temp_input['class_names'])
  return temp_input


def _average_multicrop_multiprompts(train_state: train_utils.TrainState,
                                    batch: Batch,
                                    task: str,
                                    dataset: str,
                                    flax_model: nn.Module,
                                    n_clips: int = 2,
                                    num_prompts: int = 1,
                                    prompt_index: Optional[int] = None,
                                    softmax_logits: bool = False,
                                    debug: bool = False):
  """Averages prediction from different crops and prompts."""

  num_classes = batch['label'].shape[-1]
  all_logits = jnp.zeros(num_classes)

  assert len(batch['batch_mask'].shape) == 1, (
      'Spatial padding is not supported in multi-crop evaluation.')

  num_crops = batch['label'].shape[0]
  variables = {'params': train_state.params, **train_state.model_state}
  if prompt_index is not None:
    prompt_indices = [prompt_index]
  else:
    prompt_indices = range(num_prompts)
  for idx in range(0, num_crops, n_clips):
    for prompt_index in prompt_indices:
      temp_input = _get_input_batch_from_one_prompt(
          batch,
          num_classes=num_classes,
          prompt_index=prompt_index,
          crop_index=idx,
          n_clips=n_clips)
      logits = flax_model.apply(
          variables,
          temp_input,
          task=task,
          dataset=dataset,
          train=False,
          mutable=False,
          debug=debug)
      if softmax_logits:
        logits = nn.softmax(logits, axis=-1)
      logits = jnp.sum(logits, axis=0)
      all_logits = all_logits + logits
  return all_logits / (num_crops * num_prompts)


def action_segmentation_test_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    dataset: str,
    flax_model: nn.Module,
    n_clips: int = 2,
    num_prompts: int = 1,
    prompt_index: Optional[int] = None,
    softmax_logits: bool = False,
    debug: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
           Optional[jnp.ndarray]]:
  r"""Runs a single test step of the action segmentation task.

  This function is branched from third_party/py/scenic/projects/vivit/google/\
  train_utils.py. The input batch is different from ViViT and supports
  prompting.

  Args:
    train_state: The state of training including the current global_step,
      model_state, rng, and optimizer, and other metadata.
    batch: Dictionary with keys 'inputs', 'label', 'batch_mask'. We assume that
      all the inputs correspond to the same original example in the test set.
      The input shapes to this function are: batch['inputs']['rgb'] = [
      num_crops, t, h, w, c], batch['labels'] = [num_crops, t, num_classes],
      batch['batch_mask'] = [num_crops], batch['inputs']['input_mask'] = [
      num_crops, t].
    dataset: The dataset name.
    flax_model: A Flax model.
    n_clips: The number of clips to process at a time by each device. Set due to
      memory constraints.
    num_prompts: Number of text prompts.
    prompt_index: If set, this particular prompt will be used for testing.
      Otherwise, all prompts are used and the final output is the average of
      them.
    softmax_logits: Whether to softmax-normalise the logits before averaging
    debug: Whether the debug mode is enabled during evaluation. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    all_logits: Predicted logits of shape (global_batch_size, t, num_classes)
      gathered from all devices.
    label: Ground truth labels of shape (global_batch_size, t, num_classes)
      gathered from all devices.
    batch_mask: Batch masks of shape (global_batch_size,) gathered from all
      devices.
    frame_mask: Frame masks of shape (global_batch_size, num_frames) gathered
      from all devices.
    vids: Video ids of shape (global_batch_size,) gathered from all devices.
  """
  all_logits = _average_multicrop_multiprompts(train_state, batch,
                                               'action_segmentation', dataset,
                                               flax_model, n_clips, num_prompts,
                                               prompt_index, softmax_logits,
                                               debug)
  all_logits = jnp.expand_dims(all_logits, axis=0)
  vids = jnp.expand_dims(batch['vid'][0], axis=0) if 'vid' in batch else None
  label = jnp.expand_dims(batch['label'][0], axis=0)
  batch_mask = jnp.expand_dims(batch['batch_mask'][0], axis=0)
  frame_mask = jnp.expand_dims(batch['inputs']['input_mask'][0], axis=0)

  all_logits, label, batch_mask, frame_mask, vids = jax.tree_util.tree_map(
      gather_flatten, (all_logits, label, batch_mask, frame_mask, vids))
  return all_logits, label, batch_mask, frame_mask, vids


def gather_flatten(x: Optional[jnp.ndarray],
                   axis_name: str = 'batch') -> Optional[jnp.ndarray]:
  """Flatten leading two dims, e.g. to get global batch after all_gather."""
  if x is None:
    return x
  return jnp.concatenate(jax.lax.all_gather(x, axis_name), axis=0)


def run_classification_test_steps_and_save_eval_summary(
    config: ml_collections.ConfigDict, cur_step: int,
    dataset: dataset_utils.Dataset, test_step_pmapped: Any,
    train_state: train_utils.TrainState, writer: metric_writers.MetricWriter):
  """Runs test iterations and save evaluation summary."""

  num_spatial_crops = (3 if config.dataset_configs.get('do_three_spatial_crops',
                                                       False) else 1)
  num_crops = (
      config.dataset_configs.get('num_test_clips', 1) * num_spatial_crops)
  num_eval_examples = dataset.meta_data['num_test_examples']
  # For one host, we can set total_eval_epochs to 1. For multihost, we may need
  # to increase this number because the shards may not be balanced.
  total_eval_epochs = config.dataset_configs.get('total_eval_epochs', 1.0)
  total_eval_steps = int(
      np.ceil(total_eval_epochs * num_eval_examples /
              (config.dataset_configs.test_batch_size * num_crops *
               jax.process_count())))

  all_logits, all_labels, all_batch_masks, all_vids = [], [], [], []
  for step in range(total_eval_steps):
    test_batch = next(dataset.test_iter)
    logits, label, batch_mask, vids = test_step_pmapped(train_state, test_batch)
    (logits, label, batch_mask, vids) = jax.tree_util.tree_map(
        jax_utils.unreplicate, (logits, label, batch_mask, vids)
    )
    all_logits.append(logits)
    all_labels.append(label)
    all_batch_masks.append(batch_mask)
    all_vids.append(vids)
  all_logits = jnp.concatenate(all_logits, axis=0)
  all_labels = jnp.concatenate(all_labels, axis=0)
  all_batch_masks = jnp.concatenate(all_batch_masks, axis=0)
  if all_vids[0] is not None:
    all_vids = jnp.concatenate(all_vids, axis=0)
    (all_logits, all_labels,
     all_batch_masks, all_vids) = jax.tree_util.tree_map(
         jax.device_get, (all_logits, all_labels, all_batch_masks, all_vids))
    all_logits, all_labels, deduped_vids = postprocessing_utils.dedup_by_vid(
        all_logits, all_labels, all_batch_masks, all_vids)
    duplicates = len(all_vids) - len(deduped_vids)
    logging.info(
        '%d unique samples encountered during test and found %d duplicates.',
        len(deduped_vids), duplicates)
    if len(deduped_vids) < num_eval_examples:
      logging.warning(
          'Total number of eval sample: %d but only seen %d samples. You may '
          'increase the number of test steps.', num_eval_examples,
          len(deduped_vids))
  top1 = sklearn.metrics.top_k_accuracy_score(
      np.argmax(all_labels, axis=-1), all_logits, k=1)
  top5 = sklearn.metrics.top_k_accuracy_score(
      np.argmax(all_labels, axis=-1), all_logits, k=5)
  test_summary = {
      'test/top_1_accuracy': top1,
      'test/top_5_accuracy': top5,
  }
  writer.write_scalars(cur_step, test_summary)
  writer.flush()
  return test_summary


def run_action_segmentation_test_steps_and_save_eval_summary(
    config: ml_collections.ConfigDict, cur_step: int,
    dataset: dataset_utils.Dataset, test_step_pmapped: Any,
    train_state: train_utils.TrainState, writer: metric_writers.MetricWriter):
  """Runs test iterations and save evaluation summary."""
  num_spatial_crops = (3 if config.dataset_configs.get('do_three_spatial_crops',
                                                       False) else 1)
  num_crops = (
      config.dataset_configs.get('num_test_clips', 1) * num_spatial_crops)
  num_eval_examples = dataset.meta_data['num_test_examples']
  # For one host, we can set total_eval_epochs to 1. For multihost, we may need
  # to increase this number because the shards may not be balanced.
  total_eval_epochs = config.dataset_configs.get('total_eval_epochs', 1.0)
  total_eval_steps = int(
      np.ceil(
          total_eval_epochs
          * num_eval_examples
          / (
              config.dataset_configs.test_batch_size
              * num_crops
              * jax.process_count()
          )
      )
  )

  all_logits, all_labels, all_batch_masks, all_frame_masks, all_vids = (
      [], [], [], [], [],
      )
  for step in range(total_eval_steps):
    test_batch = next(dataset.test_iter)
    logits, label, batch_mask, frame_mask, vids = test_step_pmapped(
        train_state, test_batch)
    (logits, label, batch_mask, frame_mask, vids) = jax.tree_util.tree_map(
        train_utils.unreplicate_and_get,
        (logits, label, batch_mask, frame_mask, vids),
    )
    all_logits.append(logits)
    all_labels.append(label)
    all_batch_masks.append(batch_mask)
    all_frame_masks.append(frame_mask)
    all_vids.append(vids)
  all_logits = np.concatenate(all_logits, axis=0)
  all_labels = np.concatenate(all_labels, axis=0)
  all_batch_masks = np.concatenate(all_batch_masks, axis=0)
  all_frame_masks = np.concatenate(all_frame_masks, axis=0)
  if all_vids[0] is not None:
    all_vids = np.concatenate(all_vids, axis=0)
    all_logits, all_labels, deduped_vids = postprocessing_utils.dedup_by_vid(
        all_logits, all_labels, all_batch_masks, all_vids, all_frame_masks)
    duplicates = len(all_vids) - len(deduped_vids)
    logging.info(
        '%d unique samples encountered during test and found %d duplicates.',
        len(deduped_vids), duplicates)
    if len(deduped_vids) < num_eval_examples:
      logging.warning(
          'Total number of eval sample: %d but only seen %d samples. You may '
          'increase the number of test steps.', num_eval_examples,
          len(deduped_vids))

  test_summary = {
      'test/frame_accuracy': unloc_metrics.frame_accuracy(
          all_logits,
          all_labels,
          background_logit_threshold=config.get(
              'background_logit_threshold', 0.0
          ),
      ),
  }
  test_summary.update(
      vivit_evaluation_lib.compute_mean_average_precision(
          all_logits, all_labels, suffix='test'))
  writer.write_scalars(cur_step, test_summary)
  writer.flush()
  return test_summary


def temporal_localization_test_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    dataset: str,
    task: str,
    flax_model: nn.Module,
    num_prompts: int = 1,
    output_per_class_displacements: bool = True,
    debug: Optional[bool] = False,
) -> Tuple[jnp.ndarray, ...]:
  """Runs a single temporal localization testing step.

  Args:
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, params and optimizer state.
    batch: A single batch of data.
    dataset: The dataset name.
    task: The task name, `temporal_localization` or `highlight_detection`.
    flax_model: A Flax model.
    num_prompts: Number of text prompts.
    output_per_class_displacements: Whether or not the model predict start/end
      time displacements for each class.
    debug: Whether the debug mode is enabled during evaluation. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Class probabilities in shape (batch, num_frames, num_classes).
    Start/end time displacements in shape (batch, num_frames, num_classes, 2).
    Batch mask in shape (batch,).
    Input mask in shape (batch, num_frames).
    Ground truth segments (start/end timestamps) in shape (batch, max_segments,
      2).
    Segment label indices in shape (batch, max_segments).
    Total frames in shape (batch,).
    Video Ids in shape (batch,).
  """

  bs, num_frames, num_classes = batch['label'].shape
  if output_per_class_displacements:
    logits = jnp.zeros((num_frames, num_classes * 3), dtype=jnp.float32)
  else:
    logits = jnp.zeros((num_frames, num_classes + 2), dtype=jnp.float32)
  variables = {'params': train_state.params, **train_state.model_state}
  for prompt_index in range(num_prompts):
    temp_input = _get_input_batch_from_one_prompt(
        batch,
        num_classes=num_classes,
        prompt_index=prompt_index,
        crop_index=0,
        n_clips=1)
    temp_logits = flax_model.apply(
        variables,
        temp_input,
        task=task,
        dataset=dataset,
        train=False,
        mutable=False,
        debug=debug,
    )
    logits = logits + temp_logits
  logits = logits[None, ...] / num_prompts
  if output_per_class_displacements:
    logits = logits.reshape((bs, num_frames, num_classes, 3))
    class_probs = nn.sigmoid(logits[..., 0])
    displacements = logits[..., 1:]
  else:
    logits = logits.reshape((bs, num_frames, num_classes + 2))
    class_probs = nn.sigmoid(logits[..., :num_classes])
    displacements = logits[..., num_classes:]
    displacements = jnp.tile(
        jnp.expand_dims(displacements, axis=2),
        [1, 1, num_classes, 1],
    )
  gt_segments = jnp.stack(
      [batch['segment_start_timestamp'], batch['segment_end_timestamp']],
      axis=-1) / _MICROSECONDS_PER_SECOND
  return jax.tree_util.tree_map(
      gather_flatten,
      (
          class_probs,
          displacements,
          batch['batch_mask'],
          batch['inputs']['input_mask'],
          gt_segments,
          batch['segment_label_index'],
          batch['total_frames'],
          batch['vid'],
      ),
  )


def _run_temporal_localization_test_steps(
    config: ml_collections.ConfigDict, dataset: dataset_utils.Dataset,
    test_step_pmapped: Any,
    train_state: train_utils.TrainState) -> Dict[str, Any]:
  """Run temporal localization test steps and save results into a dict."""

  num_eval_examples = dataset.meta_data['num_test_examples']
  # For one host, we can set total_eval_epochs to 1. For multihost, we may need
  # to increase this number because the shards may not be balanced.
  total_eval_epochs = config.dataset_configs.get('total_eval_epochs', 1.0)
  total_eval_steps = int(
      np.ceil(total_eval_epochs * num_eval_examples /
              (config.dataset_configs.test_batch_size * jax.process_count())))
  all_pred_and_labels = {}
  duplicates = 0
  nms_fn = (
      postprocessing_utils.non_max_suppression_multiclass if config.get(
          'multiclass_nms', False) else
      postprocessing_utils.non_max_suppression)
  for step in range(total_eval_steps):
    test_batch = next(dataset.test_iter)
    output = test_step_pmapped(train_state, test_batch)
    output = jax.tree_util.tree_map(train_utils.unreplicate_and_get, output)
    (
        pred_class_probs,
        pred_displacements,
        batch_mask,
        input_mask,
        gt_segments,
        gt_segment_labels,
        total_frames,
        vids,
    ) = output
    batch_mask = batch_mask.astype(bool)
    (pred_class_probs, pred_displacements, input_mask, gt_segments,
     gt_segment_labels, total_frames, vids) = jax.tree_util.tree_map(
         lambda x, mask=batch_mask: x[mask],
         (pred_class_probs, pred_displacements, input_mask, gt_segments,
          gt_segment_labels, total_frames, vids))
    for idx in range(pred_class_probs.shape[0]):
      vid = vids[idx]
      if vid in all_pred_and_labels:
        duplicates += 1
      else:
        cur_gt_segments = gt_segments[idx]
        cur_gt_segment_labels = gt_segment_labels[idx]
        # Labels are padded with -1s.
        mask = cur_gt_segment_labels > -1
        cur_gt_segment_labels = cur_gt_segment_labels[mask]
        cur_gt_segments = cur_gt_segments[mask]

        (cur_pred_class_indices, cur_pred_class_probs, cur_pred_segments) = (
            postprocessing_utils.get_segments_from_frame_predictions(
                class_probs=pred_class_probs[idx],
                displacements=pred_displacements[idx],
                input_mask=input_mask[idx],
                total_frames=total_frames[idx],
                stride=config.dataset_configs.stride,
                sampling_strategy=config.dataset_configs.get(
                    'sampling_strategy', 'random'
                ),
                displacement_normalizer=config.dataset_configs.get(
                    'displacement_normalizer'
                ),
                secs_per_timestep=config.dataset_configs.get(
                    'secs_per_timestep'
                ),
                score_threshold=config.get('score_threshold', float('-inf')),
                feature_pyramid_config=config.dataset_configs.get(
                    'feature_pyramid_config'
                ),
            )
        )

        cur_pred_class_indices, cur_pred_class_probs, cur_pred_segments = (
            nms_fn(
                cur_pred_class_indices,
                cur_pred_class_probs,
                cur_pred_segments,
                config,
            )
        )

        all_pred_and_labels[vid] = (
            cur_pred_class_indices,
            cur_pred_class_probs,
            cur_pred_segments,
            cur_gt_segments,
            cur_gt_segment_labels,
        )
  logging.info(
      '%d unique samples encountered during test and found %d duplicates.',
      len(all_pred_and_labels), duplicates)
  if len(all_pred_and_labels) < num_eval_examples:
    logging.warning(
        'Total number of eval sample: %d but only seen %d samples. You may '
        'increase the number of test steps.', num_eval_examples,
        len(all_pred_and_labels))
  return all_pred_and_labels


def run_temporal_localization_test_steps_and_save_eval_summary(
    config: ml_collections.ConfigDict, cur_step: int,
    dataset: dataset_utils.Dataset, test_step_pmapped: Any,
    train_state: train_utils.TrainState,
    writer: metric_writers.MetricWriter) -> Dict[str, float]:
  """Runs test iterations and save evaluation summary."""

  all_pred_and_labels = _run_temporal_localization_test_steps(
      config, dataset, test_step_pmapped, train_state)
  results = collections.defaultdict(list)
  # Convert results to ActivityNet evaluator format.
  for vid, pred_and_labels in all_pred_and_labels.items():
    (pred_classes, pred_class_probs, pred_segments, gt_segments,
     gt_segment_label) = pred_and_labels
    results['video_id'].append(vid)
    results['detection_segments'].append(pred_segments)
    results['detection_scores'].append(pred_class_probs)
    results['detection_classes'].append(pred_classes)
    results['groundtruth_segments'].append(gt_segments)
    results['groundtruth_classes'].append(gt_segment_label)

  if config.get('result_pickle'):
    with tf.io.gfile.GFile(config.result_pickle, 'wb') as fp:
      pickle.dump(results, fp)
  test_summary = activity_net_eval.evaluate_detection_results_anet(
      results, num_classes=config.dataset_configs.num_classes)
  writer.write_scalars(cur_step, test_summary)
  writer.flush()
  return test_summary


def moment_retrieval_test_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    dataset: str,
    flax_model: nn.Module,
    debug: Optional[bool] = False) -> Tuple[jnp.ndarray, ...]:
  """Runs a single moment retrieval testing step.

  Args:
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, params and optimizer state.
    batch: A single batch of data.
    dataset: The dataset name.
    flax_model: A Flax model.
    debug: Whether the debug mode is enabled during evaluation. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Class probabilities in shape (batch, max_num_captions, num_frames).
    Start/end time displacements in shape (batch, max_num_captions,
      num_frames, 2).
    Batch mask in shape (batch,).
    Input mask in shape (batch, num_frames).
    Caption mask in shape (batch, max_num_captions).
    Ground truth segments (start/end timestamps) in shape (batch,
      max_num_captions, 2).
    Total frames in shape (batch,).
    Video Ids in shape (batch,).
  """
  assert (
      batch['label'].shape[0] == 1
  )  # For multicrop to work the local batch size needs to be 1
  bs, num_captions, num_frames, _ = batch['label'].shape
  variables = {'params': train_state.params, **train_state.model_state}
  logits = flax_model.apply(
      variables,
      batch['inputs'],
      task='moment_retrieval',
      dataset=dataset,
      train=False,
      mutable=False,
      debug=debug)
  assert logits.shape[0] == 1 and logits.shape[1] == num_captions
  logits = logits[None, ...]
  logits = logits.reshape((bs, bs, num_captions, num_frames, 3))
  logits = logits[jnp.arange(bs), jnp.arange(bs)]
  class_probs = nn.sigmoid(logits[..., 0])
  displacements = logits[..., 1:]
  gt_segments = jnp.stack(
      [batch['segment_start_timestamp'], batch['segment_end_timestamp']],
      axis=-1) / _MICROSECONDS_PER_SECOND
  return jax.tree_util.tree_map(
      gather_flatten,
      (class_probs, displacements, batch['batch_mask'],
       batch['inputs']['input_mask'], batch['inputs']['caption_mask'],
       gt_segments, batch['total_frames'], batch['vid']))


def _run_moment_retrieval_test_steps(
    config: ml_collections.ConfigDict, dataset: dataset_utils.Dataset,
    test_step_pmapped: Any,
    train_state: train_utils.TrainState) -> Dict[str, Any]:
  """Run moment retrieval test steps and save results into a dict."""

  num_eval_examples = dataset.meta_data['num_test_examples']
  # For one host, we can set total_eval_epochs to 1. For multihost, we may need
  # to increase this number because the shards may not be balanced.
  total_eval_epochs = config.dataset_configs.get('total_eval_epochs', 1.0)
  total_eval_steps = int(
      np.ceil(total_eval_epochs * num_eval_examples /
              (config.dataset_configs.test_batch_size * jax.process_count())))
  all_pred_and_labels = {}
  duplicates = 0
  nms_fn = postprocessing_utils.non_max_suppression_mr
  for step in range(total_eval_steps):
    test_batch = next(dataset.test_iter)
    output = test_step_pmapped(train_state, test_batch)
    output = jax.tree_util.tree_map(train_utils.unreplicate_and_get, output)
    (pred_class_probs, pred_displacements, batch_mask, input_mask, caption_mask,
     gt_segments, total_frames, vids) = output

    batch_mask = batch_mask.astype(bool)
    (pred_class_probs, pred_displacements, input_mask, caption_mask,
     gt_segments, total_frames, vids) = jax.tree_util.tree_map(
         lambda x, mask=batch_mask: x[mask],
         (pred_class_probs, pred_displacements, input_mask, caption_mask,
          gt_segments, total_frames, vids))
    for idx in range(pred_class_probs.shape[0]):
      vid = vids[idx]
      if vid in all_pred_and_labels:
        duplicates += 1
      else:
        cur_gt_segments = gt_segments[idx]
        cur_gt_segments = cur_gt_segments[caption_mask[idx].astype(bool)]
        (cur_pred_scores, cur_pred_segments) = (
            postprocessing_utils.get_segments_from_frame_predictions_mr(
                class_probs=pred_class_probs[idx],
                displacements=pred_displacements[idx],
                input_mask=input_mask[idx],
                caption_mask=caption_mask[idx],
                total_frames=total_frames[idx],
                stride=config.dataset_configs.stride,
                sampling_strategy=config.dataset_configs.get(
                    'sampling_strategy', 'random'
                ),
                displacement_normalizer=config.dataset_configs.get(
                    'displacement_normalizer'
                ),
                secs_per_timestep=config.dataset_configs.get(
                    'secs_per_timestep'
                ),
                feature_pyramid_config=config.dataset_configs.get(
                    'feature_pyramid_config'
                ),
            )
        )
        cur_pred_scores, cur_pred_segments = nms_fn(
            cur_pred_scores,
            cur_pred_segments,
            config,
        )

        all_pred_and_labels[vid] = (
            cur_pred_scores,
            cur_pred_segments,
            cur_gt_segments,
        )
  logging.info(
      '%d unique samples encountered during test and found %d duplicates.',
      len(all_pred_and_labels), duplicates)
  if len(all_pred_and_labels) < num_eval_examples:
    logging.warning(
        'Total number of eval sample: %d but only seen %d samples. You may '
        'increase the number of test steps.', num_eval_examples,
        len(all_pred_and_labels))
  return all_pred_and_labels


def run_moment_retrieval_test_steps_and_save_eval_summary(
    config: ml_collections.ConfigDict, cur_step: int,
    dataset: dataset_utils.Dataset, test_step_pmapped: Any,
    train_state: train_utils.TrainState,
    writer: metric_writers.MetricWriter) -> Dict[str, Any]:
  """Runs test iterations and save evaluation summary."""

  all_pred_and_labels = _run_moment_retrieval_test_steps(
      config, dataset, test_step_pmapped, train_state)
  video_id = []
  all_pred_scores = []
  all_pred_segments = []
  all_gt_segments = []
  for vid, pred_and_labels in all_pred_and_labels.items():
    pred_scores, pred_segments, gt_segments = pred_and_labels
    video_id.append(vid)
    for cap_id in range(gt_segments.shape[0]):
      all_pred_scores.append(pred_scores[cap_id])
      all_pred_segments.append(pred_segments[cap_id])
      all_gt_segments.append(gt_segments[cap_id])

  test_summary = unloc_metrics.compute_recall_at_k(
      all_gt_segments,
      all_pred_segments,
      all_pred_scores,
      ranks=[1, 5],
      iou_thresholds=[0.5, 0.7],
  )
  writer.write_scalars(cur_step, test_summary)
  writer.flush()
  return test_summary
