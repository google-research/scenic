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

"""Utilities for Layout Denoise trainer."""

import collections
import copy
import functools
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

import numpy as np
from scenic.common_lib import debug_utils
from scenic.train_lib_deprecated import optimizers
from scenic.train_lib_deprecated import train_utils

# JAX team is working on type annotation for pytree:
# https://github.com/google/jax/issues/1555
PyTree = Union[Mapping[str, Mapping], Any]


def initialize_multitask_model(
    *,
    model_def: nn.Module,
    input_spec: Dict[str, Sequence[Union[Tuple[Tuple[int, ...], jnp.dtype],
                                         Tuple[int, ...]]]],
    config: ml_collections.ConfigDict,
    rngs: Union[jnp.ndarray, Mapping[str, jnp.ndarray]],
) -> Tuple[PyTree, PyTree, int, Optional[float]]:
  """Initializes parameters and model state.

  Args:
    model_def: Definition of a model.
    input_spec: A dictionary from task names to an iterable of (shape, dtype)
      pairs specifying the shape and dtype of the inputs. If unspecified the
      dtype is float32.
    config: Configurations of the initialization.
    rngs: Jax rng keys.

  Returns:
    Initial params, Init model_state, and number of trainable_params.
  """
  batch_size = (config.batch_size //
                jax.device_count()) if config.get('batch_size') else None

  def init_fn(model_def):
    for task, in_spec in input_spec.items():
      input_shapetype = [
          debug_utils.input_spec_to_jax_shape_dtype_struct(
              spec, batch_size=batch_size) for spec in in_spec
      ]
      dummy_input = []
      for in_st in input_shapetype:
        dummy_input.append(jnp.zeros(in_st.shape, in_st.dtype))
      model_def(*dummy_input, task=task, train=False, debug=False)

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def _initialize_model(rngs):
    """Initialization function to be jitted."""
    init_model_state, init_params = nn.init(
        fn=init_fn, module=model_def)(rngs).pop('params')
    # Set bias in the head to low value, such that loss is small initially.
    if (config.get('init_head_bias', None) is not None and
        'output_projection' in init_params):
      init_params = flax.core.unfreeze(init_params)
      init_params['output_projection'] = optimizers.tree_map_with_names(
          lambda p: jnp.full_like(p, config.init_head_bias),
          init_params['output_projection'],
          match_name_fn=lambda name: 'bias' in name)
      init_params = flax.core.freeze(init_params)
    return init_params, init_model_state

  if not isinstance(rngs, dict):
    rngs = {'params': rngs}
  init_params, init_model_state = _initialize_model(rngs)
  # Pop out params rng:
  rngs.pop('params')

  # Count number of trainable parameters:
  num_trainable_params = debug_utils.log_param_shapes(init_params)

  # Count gflops:
  count_flops = config.get('count_flops',
                           ml_collections.ConfigDict({'count_flops': True}))
  if count_flops:
    variables = {'params': init_params, **init_model_state}
    flops = 0
    for task, in_spec in input_spec.items():
      flops += debug_utils.compute_flops(
          flax_model_apply_fn=functools.partial(
              model_def.apply,
              variables,
              train=False,
              debug=False,
              rngs=rngs,
              task=task),
          input_spec=count_flops.get('input_spec', in_spec),
          fuse_multiply_add=count_flops.get('fuse_multiply_add', True))
    gflops = flops / (10**9)
  else:
    gflops = None

  return init_params, init_model_state, num_trainable_params, gflops


def multi_class_scores(y_true, y_pred, mask):
  """Computes precision/recall/f1 scores using one-hot labels."""
  tp = np.count_nonzero(y_pred * y_true * mask)
  fp = np.count_nonzero(y_pred * (y_true - 1) * mask)
  fn = np.count_nonzero((y_pred - 1) * y_true * mask)
  precision = 0
  recall = 0
  f1 = 0
  if tp + fp:
    precision = np.divide(tp, tp + fp)
  if tp + fn:
    recall = np.divide(tp, tp + fn)
  if precision + recall:
    f1 = np.divide(2 * precision * recall, precision + recall)
  return precision, recall, f1


class LayoutDenoiseGlobalEvaluator():
  """An interface between the Scenic DETR implementation and COCO evaluators."""

  def __init__(self, num_classes):
    self._num_classes = num_classes
    self._num_examples_added = collections.defaultdict(int)

    # Maps from (task, dataset) to results.
    self._preds = {}
    self._labels = {}

  def _get_result_lists(self, task_name, ds_name):
    """Gets result lists for the task and dataset."""
    key = (task_name, ds_name)
    if key not in self._preds:
      self._preds[key] = []
    if key not in self._labels:
      self._labels[key] = []
    return self._labels[key], self._preds[key]

  def add_example(self, task_name: str, ds_name: str,
                  prediction: Dict[str, np.ndarray], target: Dict[str,
                                                                  np.ndarray]):
    """Add a single example to the evaluator.

    Args:
      task_name: Name of the task, e.g., layout.
      ds_name: Name of the dataset, e.g., rico.
      prediction: Model prediction dictionary with key 'pred_logits', in shape
        of `[num_objects, num_classes]`.
      target: Target dictionary with key 'labels'.
    """
    # Get result lists for the (task, dataset) pair.
    labels, preds = self._get_result_lists(task_name, ds_name)

    # Get valid (non-padding) target size.
    valid_index = [i for i, l in enumerate(target['labels']) if l > 0]

    target_size = len(valid_index)
    if target_size == 0:
      # Ignore empty examples.
      return

    target_labels = target['labels'][valid_index]
    labels += target_labels.tolist()

    pred_labels = np.argmax(prediction['pred_logits'], axis=-1)
    preds += pred_labels[valid_index].tolist()
    self._num_examples_added[(task_name, ds_name)] += 1

    if 'binary_pred_logits' in prediction:
      labels, preds = self._get_result_lists(f'binary_{task_name}', ds_name)
      target_labels = target['binary_labels'][valid_index]
      labels += target_labels.tolist()
      pred_labels = np.argmax(prediction['binary_pred_logits'], axis=-1)
      preds += pred_labels[valid_index].tolist()
      self._num_examples_added[(f'binary_{task_name}', ds_name)] += 1

  def compute_metrics(self) -> Dict[str, Any]:
    """Computes the metrics for all added predictions."""
    results = {}

    keys = list(self._labels.keys())
    for key in keys:
      labels = np.array(self._labels[key])
      preds = np.array(self._preds[key])

      correct = np.sum(labels == preds)
      accuracy = 0.0
      if self._labels[key]:
        accuracy = correct / len(labels)

      labels_onehot = np.zeros((len(labels), self._num_classes))
      labels_onehot[np.arange(labels.size), labels] = 1
      # Remove padding/non-target as we only compute P/R/F for target classes.
      labels_onehot = labels_onehot[:, 2:]

      preds_onehot = np.zeros((len(preds), self._num_classes))
      preds_onehot[np.arange(preds.size), preds] = 1
      preds_onehot = preds_onehot[:, 2:]

      # We calculate P/R/F for target classes.
      mask = labels > 1
      mask = np.expand_dims(mask, axis=-1)
      p, r, f = multi_class_scores(labels_onehot, preds_onehot, mask)

      task, ds = key
      results.update({
          f'{task}-{ds}/accuracy': accuracy,
          f'{task}-{ds}/precision': p,
          f'{task}-{ds}/recall': r,
          f'{task}-{ds}/f1': f,
          f'{task}-{ds}/num_instances': len(labels),
      })
    return results

  def clear(self):
    """Clears predictions/labels for previous run."""
    self._num_examples_added = collections.defaultdict(int)
    self._preds.clear()
    self._labels.clear()

  def get_num_examples_added(self):
    return self._num_examples_added


def set_lr_configs(
    config: ml_collections.ConfigDict,
    datasets_metadata: Dict[str, Dict[str, Any]]) -> ml_collections.ConfigDict:
  """Sets learning rate configurations."""

  num_total_train_examples = 0
  for ds_metadata in datasets_metadata.values():
    num_total_train_examples += ds_metadata.get('num_train_examples', 0)

  with config.unlocked():
    decay_events = {500: 400}
    steps_per_epoch = num_total_train_examples // config.batch_size
    config.lr_configs.learning_rate_schedule = 'compound'
    config.lr_configs.factors = 'constant*piecewise_constant'
    config.lr_configs.decay_events = [
        decay_events.get(config.num_training_epochs,
                         config.num_training_epochs * 2 // 3) * steps_per_epoch,
    ]
    # Note: this is absolute (not relative):
    config.lr_configs.decay_factors = [.1]
    config.lr_configs.base_learning_rate = 1e-4

    # Also set backbone lr configs
    config.backbone_training.lr_configs = copy.deepcopy(config.lr_configs)
    config.backbone_training.lr_configs.base_learning_rate = 1e-5
  return config


def get_num_training_steps(
    config: ml_collections.ConfigDict,
    datasets_metadata: Dict[str, Dict[str, Any]]) -> Tuple[int, Optional[int]]:
  """Calculates the total number of training step and possibly steps_per_epoch.

  The main training loop is based on number of training steps. Thus, for
  datasets
  that we want to train based on number of epochs, we need to calculate the
  total number of training steps. This function looks for `num_training_steps`
  in config, if it exists it returns that as the total step and `None` as
  `steps_per_epoch`. If num_training_steps doesn't exist, then it looks for
  `num_training_epochs` and given the size of training data calculates the total
  steps and steps_per_epoch. In this computation, we assume that
  drop_remainder=True.

  Args:
    config: Configuration of the experiment.
    datasets_metadata: Meta-data that is generated by the dataset_builder.

  Returns:
    total_steps: Total number of training steps.
    steps_per_epoch: Number of steps in every epoch.
  """
  num_total_train_examples = 0
  for ds_metadata in datasets_metadata.values():
    num_total_train_examples += ds_metadata.get('num_train_examples', 0)

  # We either use num_training_epochs or num_training_steps.
  steps_per_epoch = num_total_train_examples // config.batch_size
  if config.get('num_training_steps'):
    assert not config.get('num_training_epochs')
    return config.num_training_steps, steps_per_epoch or None
  else:
    assert config.num_training_epochs and not config.get('num_training_steps')
    return (steps_per_epoch * config.num_training_epochs), steps_per_epoch


def normalize_metrics_summary(metrics_summary, split,
                              object_detection_loss_keys):
  """Normalizes the metrics in the given metrics summary.

  Note that currently we only support metrics of the form 1/N sum f(x_i).

  Args:
    metrics_summary: dict; Each value is a sum of a calculated metric over all
      examples.
    split: str; Split for which we normalize the metrics. Used for logging.
    object_detection_loss_keys: list; A loss key used for computing the object
      detection loss.

  Returns:
    Normalized metrics summary.

  Raises:
    TrainingDivergedError: Due to observing a NaN in the metrics.
  """
  for key, val in metrics_summary.items():
    metrics_summary[key] = val[0] / val[1]
    if np.isnan(metrics_summary[key]):
      logging.error('%s metrics %s is NaN', split, key)
      raise train_utils.TrainingDivergedError(
          'NaN detected in {}'.format(f'{split}_{key}'))

  # compute and add object_detection_loss using globally normalize terms
  object_detection_losses = []
  for loss_term_key in object_detection_loss_keys:
    if loss_term_key in metrics_summary:
      object_detection_losses.append(metrics_summary[loss_term_key])
  if object_detection_losses:
    # Object detection loss is present only for layout task.
    metrics_summary['object_detection_loss'] = sum(object_detection_losses)

  return metrics_summary
