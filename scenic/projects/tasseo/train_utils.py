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

"""Utilities for tasseo trainer."""

from typing import Any, Dict, List, Optional

from absl import logging
import numpy as np
import sklearn.metrics


def chrom_auc_pr_score(
    target: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
  """Compute Area Under the PR Curve for abnormal.

  Args:
    target: Numpy array of targets of shape (n_samples, n_classes). Since this
      metric is only used for anomaly detection, we assume the n_classes is
      equal to 2.
    prediction: Numpy array of model predictions of shape (n_samples,
      n_classes). Here also we assume the n_classes is equal to 2.

  Returns:
    AUC PR score.
  """
  target = np.argmax(target, axis=-1)

  # In the prediction array, label 0 is for normal and 1 for abnormal examples.
  pr_curve_precisions, pr_curve_recalls, _ = sklearn.metrics.precision_recall_curve(
      target, prediction[:, 1])
  return {
      'chrom_auc_pr_score':
          sklearn.metrics.auc(pr_curve_recalls, pr_curve_precisions)
  }


def chrom_roc_auc_score(
    target: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
  """Compute Area Under the ROC Curve for abnormal.

  Args:
    target: Numpy array of targets of shape (n_samples, n_classes). Since this
      metric is only used for anomaly detection, we assume the n_classes is
      equal to 2.
    prediction: Numpy array of model predictions of shape (n_samples,
      n_classes). Here also we assume the n_classes is equal to 2.

  Returns:
    ROC AUC score.
  """
  target = np.argmax(target, axis=-1)

  # In the prediction array, label 0 is for normal and 1 for abnormal examples.
  return {
      'chrom_roc_auc_score':
          sklearn.metrics.roc_auc_score(target, prediction[:, 1])
  }


def chrom_f1_score(
    target: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
  """Compute F1 score.

  Args:
    target: Numpy array of targets of shape (n_samples, n_classes).
    prediction: Numpy array of model predictions of shape (n_samples,
      n_classes). Here also we assume the n_classes is equal to 2.

  Returns:
    F1 score.
  """
  target = np.argmax(target, axis=-1)
  prediction = np.argmax(prediction, axis=-1)
  return {'chrom_f1': sklearn.metrics.f1_score(target, prediction)}


def chrom_recall(
    target: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
  """Compute recall.

  Args:
    target: Numpy array of targets of shape (n_samples, n_classes).
    prediction: Numpy array of model predictions of shape (n_samples,
      n_classes). Here also we assume the n_classes is equal to 2.

  Returns:
    Recall score.
  """
  target = np.argmax(target, axis=-1)
  prediction = np.argmax(prediction, axis=-1)
  return {'chrom_recall': sklearn.metrics.recall_score(target, prediction)}


def chrom_specificity(
    target: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
  """Compute recall.

  Args:
    target: Numpy array of targets of shape (n_samples, n_classes).
    prediction: Numpy array of model predictions of shape (n_samples,
      n_classes). Here also we assume the n_classes is equal to 2.

  Returns:
    Specificity score.
  """
  target = np.argmax(target, axis=-1)
  prediction = np.argmax(prediction, axis=-1)
  # In binary classification, recall of the negative class is specificity.
  return {
      'chrom_specificity':
          sklearn.metrics.recall_score(target, prediction, average=None)[0]
  }


def chrom_precision(
    target: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
  """Compute precision.

  Args:
    target: Numpy array of targets of shape (n_samples, n_classes).
    prediction: Numpy array of model predictions of shape (n_samples,
      n_classes). Here also we assume the n_classes is equal to 2.

  Returns:
    Precision score.
  """
  target = np.argmax(target, axis=-1)
  prediction = np.argmax(prediction, axis=-1)
  return {
      'chrom_precision': sklearn.metrics.precision_score(target, prediction)
  }


class TasseoGlobalEvaluator():
  """Evaluator used for tasseo global metrics evaluation."""

  def __init__(self, global_metrics: List[str]):
    self.global_metrics = global_metrics
    self.batches = None
    self._num_examples_added = 0

  def add_batch_of_examples(self, target: np.ndarray, output: np.ndarray):
    """Add a batch of examples to the evaluator.

    Args:
      target: Target to be predicted as a Numpy array.
      output: Output from the model as a Numpy array.
    """
    self._num_examples_added += output.shape[0]
    if self.batches is None:
      self.batches = (target, output)
    else:  # Append targets and outputs for the new examples.
      self.batches = (np.append(self.batches[0], target, axis=0),
                      np.append(self.batches[1], output, axis=0))

  def compute_metrics(self,
                      clear_annotations: Optional[bool] = True
                     ) -> Dict[str, Any]:
    """Computes the relevant metrics for all added <target, output> pairs."""
    # To handle the case where the batch contains only a single class, fall back
    # to an empty metric value for that point rather than raising an exception.
    def try_with_default(func, default_retval=None):
      try:
        return func()
      except ValueError as e:
        logging.warn('Failed to compute metrics: %r', e)
      return default_retval if default_retval is not None else {}

    metrics = {}
    # pylint: disable=g-long-lambda
    if 'chrom_f1' in self.global_metrics:
      metrics.update(
          try_with_default(lambda: chrom_f1_score(
              target=self.batches[0], prediction=self.batches[1])))
    if 'chrom_recall' in self.global_metrics:
      metrics.update(
          try_with_default(lambda: chrom_recall(
              target=self.batches[0], prediction=self.batches[1])))
    if 'chrom_precision' in self.global_metrics:
      metrics.update(
          try_with_default(lambda: chrom_precision(
              target=self.batches[0], prediction=self.batches[1])))
    if 'chrom_roc_auc_score' in self.global_metrics:
      metrics.update(
          try_with_default(lambda: chrom_roc_auc_score(
              target=self.batches[0], prediction=self.batches[1])))
    if 'chrom_specificity' in self.global_metrics:
      metrics.update(
          try_with_default(lambda: chrom_specificity(
              target=self.batches[0], prediction=self.batches[1])))
    if 'chrom_auc_pr_score' in self.global_metrics:
      metrics.update(
          try_with_default(lambda: chrom_auc_pr_score(
              target=self.batches[0], prediction=self.batches[1])))
    if clear_annotations:
      self.clear()
    # pylint: enable=g-long-lambda
    return metrics

  def clear(self):
    self.batches = None
    self._num_examples_added = 0

  def __len__(self):
    return self._num_examples_added
