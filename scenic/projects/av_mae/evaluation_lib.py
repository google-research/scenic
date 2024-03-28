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

from absl import logging
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


def get_d_prime(auc_roc_value):
  d_prime = stats.norm().ppf(auc_roc_value) * np.sqrt(2.0)
  return d_prime


def compute_mean_avg_precision_dprime(logits,
                                      labels,
                                      suffix='',
                                      suffix_separator='_',
                                      return_per_class_ap=False):
  """Computes mean average precision and d-prime for multi-label classification.

  Args:
    logits: Numpy array of shape [num_examples, num_classes]
    labels: Numpy array of shape [num_examples, num_classes]
    suffix: Suffix to add to the summary
    suffix_separator: Separator before adding the suffix
    return_per_class_ap: If True, return results for each class in the summary.

  Returns:
    summary: Dictionary containing the mean average precision, ROC AUC, d-prime,
      and maybe the average precision per class.
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

  nanmean_ap = np.nanmean(average_precisions)
  summary[f'nanmean_average_precision{suffix}'] = nanmean_ap
  logging.info('NanMean AP is %0.5f', nanmean_ap)
  logging.info('Shape of logits for computing mAP: %s', logits.shape)
  logging.info('Shape of labels for computing mAP: %s', labels.shape)

  # Compute overall mAP, ROC AUC, d-prime. With average=None, scores for each
  # class are returned.
  auc_pc = roc_auc_score(labels, logits, average=None)

  mean_average_precision = np.mean(average_precisions)
  mean_auc = np.mean(auc_pc)
  balanced_d_prime = get_d_prime(mean_auc)

  logging.info('====Reporting overall multi-label evaluation metrics:')
  logging.info('Mean AP is %0.5f', mean_average_precision)
  logging.info('Mean AUC is %0.5f', mean_auc)
  logging.info('Mean d-prime is %1.4f', balanced_d_prime)
  summary[f'mAP{suffix}'] = mean_average_precision
  summary[f'AUC{suffix}'] = mean_auc
  summary[f'd-prime{suffix}'] = balanced_d_prime

  return summary
