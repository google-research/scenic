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

"""Functions for evaluation."""

import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from absl import logging
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
from scenic.train_lib_deprecated import pretrain_utils
from scenic.train_lib_deprecated import train_utils
from sklearn.metrics import average_precision_score
from tensorflow.io import gfile


# Aliases for custom types:
Array = Union[jnp.ndarray, np.ndarray]


def restore_checkpoint(checkpoint_path: str,
                       train_state: Optional[train_utils.TrainState] = None,
                       assert_exist: bool = False,
                       step: int = None) -> Tuple[train_utils.TrainState, int]:
  """Restores the last checkpoint.

  Supports checkpoints saved either with old Scenic (flax.deprecated.nn) or
  current Scenic (flax.Linen). Therefore, this function can be used for
  evaluation of old or current models.

  First restores the checkpoint, which is an instance of TrainState that holds
  the state of training, and then replicates it.

  Args:
    checkpoint_path: Directory for saving the checkpoint.
    train_state: An instance of TrainState that holds the state of training.
    assert_exist: bool; Assert that there is at least one checkpoint exists in
      the given path.
    step: Step number to load or None to load latest. If specified,
      checkpoint_path must be a directory.

  Returns:
    Training state and an int which is the current step.
  """
  if assert_exist:
    glob_path = os.path.join(checkpoint_path, 'checkpoint_*')
    if not gfile.glob(glob_path):
      raise ValueError('No checkpoint for the pretrained model is found in: '
                       f'{checkpoint_path}')
  restored_train_state = checkpoints.restore_checkpoint(checkpoint_path, None,
                                                        step)

  if restored_train_state:
    (restored_params, restored_model_state
    ) = pretrain_utils.get_params_and_model_state_dict(restored_train_state)
    restored_params = flax.core.freeze(restored_params)
    restored_model_state = flax.core.freeze(restored_model_state)
    train_state = train_state or train_utils.TrainState()
    if train_state.optimizer:
      new_optimizer = train_state.optimizer.replace(target=restored_params)
    else:
      new_optimizer = {'target': restored_params}
    train_state = train_state.replace(  # pytype: disable=attribute-error
        optimizer=new_optimizer,
        model_state=restored_model_state,
        global_step=int(restored_train_state['global_step']),
        rng=restored_train_state['rng'],
        accum_train_time=restored_train_state.get('accum_train_time', 0))
  else:
    train_state = train_state or train_utils.TrainState()

  return train_state, int(train_state.global_step)


def compute_mean_average_precision(logits, labels, suffix='',
                                   suffix_separator='_',
                                   return_per_class_ap=False):
  """Computes mean average precision for multi-label classification.

  Args:
    logits: Numpy array of shape [num_examples, num_classes]
    labels: Numpy array of shape [num_examples, num_classes]
    suffix: Suffix to add to the summary
    suffix_separator: Separator before adding the suffix
    return_per_class_ap: If True, return results for each class in the summary.

  Returns:
    summary: Dictionary containing the mean average precision, and also the
      average precision per class.
  """

  assert logits.shape == labels.shape, 'Logits and labels have different shapes'
  n_classes = logits.shape[1]
  average_precisions = []
  if suffix:
    suffix = suffix_separator + suffix
  summary = {}

  for i in range(n_classes):
    ave_precision = average_precision_score(labels[:, i], logits[:, i])
    if np.isnan(ave_precision):
      logging.warning('AP for class %d is NaN', i)

    if return_per_class_ap:
      summary_key = f'per_class_average_precision_{i}{suffix}'
      summary[summary_key] = ave_precision
    average_precisions.append(ave_precision)

  mean_ap = np.nanmean(average_precisions)
  summary[f'mean_average_precision{suffix}'] = mean_ap
  logging.info('Mean AP is %0.3f', mean_ap)

  return summary


def compute_confusion_matrix_metrics(
    confusion_matrices: Sequence[Array],
    return_per_class_metrics: bool) -> Dict[str, float]:
  """Computes classification metrics from a confusion matrix.

  Computes the recall, precision and jaccard index (IoU) from the input
  confusion matrices. The confusion matrices are assumed to be of the form
  [ground_truth, predictions]. In other words, ground truth classes along the
  rows, and predicted classes along the columns.

  Args:
    confusion_matrices: Sequence of [n_batch, n_class, n_class] confusion
      matrices. The first two dimensions will be summed over to get an
      [n_class, n_class] matrix for further metrics.
    return_per_class_metrics: If true, return per-class metrics.

  Returns:
    A dictionary of metrics (recall, precision and jaccard index).
  """

  conf_matrix = np.sum(confusion_matrices, axis=0)  # Sum over eval batches.
  if conf_matrix.ndim != 3:
    raise ValueError(
        'Expecting confusion matrix to have shape '
        f'[batch_size, num_classes, num_classes], got {conf_matrix.shape}.')
  conf_matrix = np.sum(conf_matrix, axis=0)  # Sum over batch dimension.
  n_classes = conf_matrix.shape[0]
  metrics_dict = {}

  # We assume that the confusion matrix is [ground_truth x predictions].
  true_positives = np.diag(conf_matrix)
  sum_rows = np.sum(conf_matrix, axis=0)
  sum_cols = np.sum(conf_matrix, axis=1)

  recall_per_class = true_positives / sum_cols
  precision_per_class = true_positives / sum_rows
  jaccard_index_per_class = (
      true_positives / (sum_rows + sum_cols - true_positives))

  metrics_dict['recall/mean'] = np.nanmean(recall_per_class)
  metrics_dict['precision/mean'] = np.nanmean(precision_per_class)
  metrics_dict['jaccard/mean'] = np.nanmean(jaccard_index_per_class)

  def add_per_class_results(metric: Array, name: str) -> None:
    for i in range(n_classes):
      # We set NaN values (from dividing by 0) to 0, to not cause problems with
      # logging.
      metrics_dict[f'{name}/{i}'] = np.nan_to_num(metric[i])

  if return_per_class_metrics:
    add_per_class_results(recall_per_class, 'recall')
    add_per_class_results(precision_per_class, 'precision')
    add_per_class_results(jaccard_index_per_class, 'jaccard')

  return metrics_dict


def prune_summary(summary, prefixes_to_remove):
  """Removes keys starting with provided prefixes from the dict."""
  ret = {}
  for key in summary.keys():
    report_key = True
    for prefix in prefixes_to_remove:
      if key.startswith(prefix):
        report_key = False
        break
    if report_key:
      ret[key] = summary[key]
  return ret




def log_eval_summary(step: int,
                     eval_metrics: Sequence[Dict[str, Tuple[float, int]]],
                     extra_eval_summary: Optional[Dict[str, Any]] = None,
                     summary_writer: Optional[Any] = None,
                     metrics_normalizer_fn: Optional[
                         Callable[[Dict[str, Tuple[float, int]], str],
                                  Dict[str, float]]] = None,
                     prefix: str = 'valid',
                     key_separator: str = '_') -> Dict[str, float]:
  """Computes and logs eval metrics.

  Args:
    step: Current step.
    eval_metrics: Sequence of dictionaries of calculated metrics.
    extra_eval_summary: A dict containing summaries that are already ready to be
      logged, e.g. global metrics from eval set, like precision/recall.
    summary_writer: Summary writer object.
    metrics_normalizer_fn: Used for normalizing metrics. The api for
      this function is: `new_metrics_dict = metrics_normalizer_fn( metrics_dict,
        split)`. If set to None, we use the normalize_metrics_summary which uses
        the normalizer paired with each metric to normalize it.
    prefix: str; Prefix added to the name of the summaries writen by this
      function.
    key_separator: Separator added between the prefix and key.

  Returns:
    eval summary: A dictionary of metrics.
  """
  eval_metrics = train_utils.stack_forest(eval_metrics)

  # Compute the sum over all examples in all batches.
  eval_metrics_summary = jax.tree_util.tree_map(lambda x: x.sum(), eval_metrics)
  # Normalize metrics by the total number of exampels.
  metrics_normalizer_fn = (
      metrics_normalizer_fn or train_utils.normalize_metrics_summary)
  eval_metrics_summary = metrics_normalizer_fn(eval_metrics_summary, 'eval')
  # If None, set to an empty dictionary.
  extra_eval_summary = extra_eval_summary or {}

  if jax.process_index() == 0:
    message = ''
    for key, val in eval_metrics_summary.items():
      message += f'{key}: {val} | '
    for key, val in extra_eval_summary.items():
      message += f'{key}: {val} | '
    logging.info('step: %d -- %s -- {%s}', step, prefix, message)

    if summary_writer is not None:
      for key, val in eval_metrics_summary.items():
        summary_writer.scalar(f'{prefix}{key_separator}{key}', val, step)
      for key, val in extra_eval_summary.items():
        summary_writer.scalar(f'{prefix}{key_separator}{key}', val, step)
      summary_writer.flush()

  # Add extra_eval_summary to the returned eval_summary.
  eval_metrics_summary.update(extra_eval_summary)
  return eval_metrics_summary
