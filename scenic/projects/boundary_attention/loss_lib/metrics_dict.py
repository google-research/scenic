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

"""Metrics functions for the boundary attention model."""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from scenic.model_lib.base_models import base_model as scenic_base_model
from scenic.projects.boundary_attention.loss_lib import metrics


def get_number(number):

  def helper(batch, target, weights=None):
    del batch, target, weights
    return number

  return helper


FASTFOJ_METRICS = {
    'gt_train_loss': (metrics.gt_train_loss, get_number(1)),
    'gt_standard_metric': (metrics.gt_standard_metric, get_number(1)),
}


def metric_function_noop(
    model_output: Dict[str, Any],
    batch: scenic_base_model.Batch,
) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
  del model_output, batch
  return {}


def metric_function(  # pylint: disable=dangerous-default-value
    model_output: Dict[str, Any],
    batch: scenic_base_model.Batch,
    metrics_dict=FASTFOJ_METRICS,
    loss_fn: Optional[Callable] = None  # pylint: disable=g-bare-generic
) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
  """Computes metrics for the given model output and batch.

  Args:
    model_output: The output of the model.
    batch: The batch of data.
    metrics_dict: A dictionary of metrics to compute.
    loss_fn: A loss function to use.

  Returns:
    A dictionary of metrics.
  """
  psum_metric_norm = psum_metric_normalizer
  evaluated_metrics = {}

  weights = 1
  for key, val in metrics_dict.items():
    metric_val = val[0](model_output, batch, weights, loss_fn=loss_fn)
    metric_count = val[1](model_output, batch, weights)
    if isinstance(metric_val, dict):
      for k, v in metric_val.items():
        evaluated_metrics[key + '_' + k] = psum_metric_norm((v, metric_count))
    else:
      evaluated_metrics[key] = psum_metric_norm((metric_val, metric_count))
  return evaluated_metrics


# Helper Functions
def psum_metric_normalizer(
    metrics: Tuple[jnp.ndarray, jnp.ndarray]  # pylint: disable=redefined-outer-name
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Applies psum over the given tuple of (metric, normalizer)."""
  psumed_metric = jnp.sum(jax.lax.psum(metrics[0], axis_name='batch'))
  psumed_normalizer = jnp.sum(jax.lax.psum(metrics[1], axis_name='batch'))
  return (psumed_metric, psumed_normalizer)
