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

"""Functions to compute Truvari metrics using jax.numpy.

https://github.com/spiralgenetics/truvari/wiki/bench
"""

from typing import Optional, List, Dict

import numpy as np

_EPSILON = 1e-5


def truvari_precision(
    logits: np.ndarray,
    one_hot_targets: np.ndarray,
) -> Dict[str, float]:
  """Computes genotype-level precision.

  Reference events are not considered.

  Args:
    logits: float array; Output of model in shape [batch, ..., num_classes].
    one_hot_targets: Multi hot vector of shape [batch, ..., num_classes].

  Returns:
    The value of categorical precision.
  """
  preds = np.argmax(logits, axis=-1)
  targets = np.argmax(one_hot_targets, axis=-1)
  correct = np.equal(preds, targets)
  incorrect = np.not_equal(preds, targets)

  non_ref_targets = np.not_equal(targets, 0)
  non_ref_preds = np.not_equal(preds, 0)

  tp = np.sum(non_ref_targets & correct)
  fp = np.sum(non_ref_preds & incorrect)

  return {'truvari_precision': np.divide(tp, tp + fp + _EPSILON)}


def truvari_recall(
    logits: np.ndarray,
    one_hot_targets: np.ndarray,
) -> Dict[str, float]:
  """Computes genotype-level recall.

  Reference events are not considered.

  Args:
    logits: float array; Output of model in shape [batch, ..., num_classes].
    one_hot_targets: Multi hot vector of shape [batch, ..., num_classes].

  Returns:
    The value of categorical recall.
  """
  preds = np.argmax(logits, axis=-1)
  targets = np.argmax(one_hot_targets, axis=-1)
  correct = np.equal(preds, targets)
  incorrect = np.not_equal(preds, targets)

  non_ref_targets = np.not_equal(targets, 0)

  tp = np.sum(non_ref_targets & correct)
  fn = np.sum(non_ref_targets & incorrect)

  return {'truvari_recall': np.divide(tp, tp + fn + _EPSILON)}


def truvari_precision_events(
    logits: np.ndarray,
    one_hot_targets: np.ndarray,
) -> Dict[str, float]:
  """Computes event-level precision, regardless of the genotype match.

  Args:
    logits: float array; Output of model in shape [batch, ..., num_classes].
    one_hot_targets: Multi hot vector of shape [batch, ..., num_classes].

  Returns:
    The value of precision events.
  """
  preds = np.not_equal(np.argmax(logits, axis=-1), 0)
  targets = np.not_equal(np.argmax(one_hot_targets, axis=-1), 0)
  correct = np.equal(preds, targets)
  incorrect = np.not_equal(preds, targets)
  tp = np.sum(targets & correct)
  fp = np.sum(preds & incorrect)

  return {'truvari_precision_events': np.divide(tp, tp + fp + _EPSILON)}


def truvari_recall_events(
    logits: np.ndarray,
    one_hot_targets: np.ndarray,
) -> Dict[str, float]:
  """Computes event-level recall, regardless of the genotype match.

  Args:
    logits: float array; Output of model in shape [batch, ..., num_classes].
    one_hot_targets: Multi hot vector of shape [batch, ..., num_classes].

  Returns:
    The value of recall events.
  """

  preds = np.not_equal(np.argmax(logits, axis=-1), 0)
  targets = np.not_equal(np.argmax(one_hot_targets, axis=-1), 0)
  correct = np.equal(preds, targets)
  incorrect = np.not_equal(preds, targets)

  tp = np.sum(targets & correct)
  fn = np.sum(targets & incorrect)

  return {'truvari_recall_events': np.divide(tp, tp + fn + _EPSILON)}


def gt_concordance(
    logits: np.ndarray,
    one_hot_targets: np.ndarray,
) -> Dict[str, float]:
  """Computes the genotype concordance.

  Genotype concordance is the fraction of predicted genotypes that exactly
  match the call set genotype.

  Args:
    logits: float array; Output of model in shape [batch, ..., num_classes].
    one_hot_targets: Multi hot vector of shape [batch, ..., num_classes].

  Returns:
    The value of genotype concordance.
  """
  preds = np.argmax(logits, axis=-1)
  targets = np.argmax(one_hot_targets, axis=-1)
  correct = np.equal(preds, targets)

  return {'gt_concordance': np.divide(np.sum(correct), len(preds))}


def nonref_concordance(
    logits: np.ndarray,
    one_hot_targets: np.ndarray,
) -> Dict[str, float]:
  """Computes non-reference concordance.

  Non reference concordance treats heterozygous and homozygous alternate
  genotypes as equivalent.

  Args:
    logits: float array; Output of model in shape [batch, ..., num_classes].
    one_hot_targets: Multi hot vector of shape [batch, ..., num_classes].

  Returns:
    The value of non reference concordance.
  """

  preds = np.not_equal(np.argmax(logits, axis=-1), 0)
  targets = np.not_equal(np.argmax(one_hot_targets, axis=-1), 0)
  correct = np.equal(preds, targets)

  return {'nonref_concordance': np.divide(np.sum(correct), len(preds))}


class TruvariGlobalEvaluator():
  """Evaluator used for global metrics evaluation."""

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
                     ) -> Dict[str, float]:
    """Computes the relevant metrics for all added <target, output> pairs."""
    metrics = {}
    if 'truvari_recall_events' in self.global_metrics:
      metrics.update(
          truvari_recall_events(
              one_hot_targets=self.batches[0], logits=self.batches[1]))
    if 'truvari_precision_events' in self.global_metrics:
      metrics.update(
          truvari_precision_events(
              one_hot_targets=self.batches[0], logits=self.batches[1]))
    if 'truvari_recall' in self.global_metrics:
      metrics.update(
          truvari_recall(
              one_hot_targets=self.batches[0], logits=self.batches[1]))
    if 'truvari_precision' in self.global_metrics:
      metrics.update(
          truvari_precision(
              one_hot_targets=self.batches[0], logits=self.batches[1]))
    if 'gt_concordance' in self.global_metrics:
      metrics.update(
          gt_concordance(
              one_hot_targets=self.batches[0], logits=self.batches[1]))
    if 'nonref_concordance' in self.global_metrics:
      metrics.update(
          nonref_concordance(
              one_hot_targets=self.batches[0], logits=self.batches[1]))

    if clear_annotations:
      self.clear()
    return metrics

  def clear(self):
    self.batches = None
    self._num_examples_added = 0

  def __len__(self):
    return self._num_examples_added
