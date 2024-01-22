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

"""Base class for all regression models."""

import functools
from typing import Dict, Optional, Tuple, Union

from immutabledict import immutabledict
import jax.numpy as jnp
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils


_REGRESSION_METRICS = immutabledict({
    'mean_squared_error':
        (model_utils.weighted_squared_error, model_utils.num_examples)
})


def regression_metrics_function(
    predictions: jnp.ndarray,
    batch: base_model.Batch,
    metrics: base_model.MetricNormalizerFnDict = _REGRESSION_METRICS,
    axis_name: Union[str, Tuple[str, ...]] = 'batch',
) -> Dict[str, Tuple[float, int]]:
  """Calculate metrics for the regression task.

  Currently we assume each metric_fn has the API:
    ```metric_fn(predictions, targets, weights)```
  and returns an array of shape [batch,]. We also assume that to compute
  the aggregate metric, one should sum across all batches, then divide by the
  total samples seen. In this way we currently only support metrics of the 1/N
  sum f(inputs, targets). Note, the caller is responsible for dividing by
  the normalizer when computing the mean of each metric.

  Args:
   predictions: Output of model in shape [batch, length].
   batch: Batch (dict) with keys 'targets' and optionally 'batch_mask'.
   metrics: The regression metrics to evaluate. The key is the
     name of the  metric, and the value is the metrics function.
   axis_name: List of axes on which we run the pmsum.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """
  targets = batch['targets']
  weights = batch.get('batch_mask')
  evaluated_metrics = {}
  for key, val in metrics.items():
    evaluated_metrics[key] = model_utils.psum_metric_normalizer(
        (val[0](predictions, targets, weights), val[1](predictions, targets,  # pytype: disable=wrong-arg-types  # jax-ndarray
                                                       weights)),
        axis_name=axis_name)
  return evaluated_metrics  # pytype: disable=bad-return-type  # jax-ndarray


class RegressionModel(base_model.BaseModel):
  """Defines commonalities between all regression models.

  A model is class with three members: get_metrics_fn, loss_fn, and a
  flax_model.

  `get_metrics_fn` returns a callable function, metric_fn, that calculates the
  metrics and returns a dictionary. The metric function computes f(x_i, y_i) on
  a minibatch, it has API:
    ```metric_fn(predictions, targets, weights).```

  The trainer will then aggregate and compute the mean across all samples
  evaluated. By default, both the metric and loss are the mean squared error.

  loss_fn is a function of API
    loss = loss_fn(predictions, batch, model_params=None).

  flax_model is returned from the build_flax_model function. A typical
  usage pattern will be:
    ```
    model_cls = model_lib.models.get_model_cls(
        'my_regression_model')
    model = model_cls(config, dataset.meta_data)
    flax_model = model.build_flax_model
    dummy_input = jnp.zeros(input_shape, model_input_dtype)
    model_state, params = flax_model.init(
        rng, dummy_input, train=False).pop('params')
    ```
  The model can then be applied by:
    variables = {'params': params, **model_state}
    predictions, new_model_state = flax_model.apply(variables, inputs, ...)
    ```
  """

  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    By default, we return the same metric for each split.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API:
    ```metrics_fn(predictions, batch)```
    """

    del split  # Same function for all splits.
    return functools.partial(
        regression_metrics_function, metrics=_REGRESSION_METRICS)

  def loss_function(self,
                    predictions: jnp.ndarray,
                    batch: base_model.Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Returns the (weighted) mean squared error.

    Args:
      predictions: Output of model in shape [batch, length].
      batch: Batch (dict) with keys 'targets' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      The (weighted) mean squared error.
    """
    weights = batch.get('batch_mask')
    targets = batch['targets']

    total_loss = model_utils.weighted_mean_squared_error(
        predictions, targets, weights)
    if self.config.get('l2_decay_factor'):
      l2_loss = model_utils.l2_regularization(model_params)
      total_loss += 0.5 * self.config.l2_decay_factor * l2_loss
    return total_loss  # pytype: disable=bad-return-type  # jax-ndarray

  def build_flax_model(self):
    raise NotImplementedError('Subclasses must implement build_flax_model().')

  def default_flax_model_config(self):
    """Default config for the flax model that is built in `build_flax_model`.

    This function in particular serves the testing functions and supposed to
    provide config tha are passed to the flax_model when it's build in
    `build_flax_model` function, e.g., `model_dtype_str`.
    """
    raise NotImplementedError(
        'Subclasses must implement default_flax_model_config().')
