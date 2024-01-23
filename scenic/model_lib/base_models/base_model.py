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

"""Base class models."""

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from absl import logging
import flax.linen as nn
from flax.training import common_utils
import jax.numpy as jnp
import ml_collections

# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricNormalizerFnDict = Mapping[
    str, Tuple[Callable[[jnp.ndarray, bool, Optional[jnp.ndarray]], float],
               Callable[[jnp.ndarray, bool, Optional[jnp.ndarray]], float]]]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]


def metrics_function_jit(
    logits: jnp.ndarray,
    batch: Batch,
    target_is_one_or_multihot: bool,
    metrics,
) -> Dict[str, Tuple[float, int]]:
  """Calculates metrics for the multi-label classification task for jit.

  Currently we assume each metric_fn has the API:
    ```metric_fn(logits, targets, weights)```
  and returns an array of shape [batch_size].

  Pmap-based trainers assume that to compute the aggregate metric, one should
  sum across all batches, then divide by the total samples seen.

  We follow the same API here, but note that summing should no longer use
  lax.psum, but rather a jnp.sum suffices as jit uses global arrays.

  Args:
   logits: Output of model in shape [batch, length, num_classes].
   batch: Batch of data that has 'label' and optionally 'batch_mask'.
   target_is_one_or_multihot: If the target is a one-hot or multi-hot vector.
   metrics: The metrics to evaluate. The key is the name of the metric,
   and the value is the metrics function.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """
  if target_is_one_or_multihot:
    one_or_multihot_target = batch['label']
  else:
    # This is to support running a multi-label classification model on
    # single-label classification tasks:
    one_or_multihot_target = common_utils.onehot(batch['label'],
                                                 logits.shape[-1])
  weights = batch.get('batch_mask')  # batch_mask might not be defined

  evaluated_metrics = {}
  for key, metric in metrics.items():
    fn, normaliser = metric
    metric_value = fn(logits, one_or_multihot_target, weights)
    norm_value = normaliser(logits, one_or_multihot_target, weights)
    evaluated_metrics[key] = (jnp.sum(metric_value), jnp.sum(norm_value))

  return evaluated_metrics  # pytype: disable=bad-return-type  # jax-types


class BaseModel:
  """Defines commonalities between all models.

  A model is class with three members: get_metrics_fn, loss_fn, and a
  flax_model.

  get_metrics_fn returns a callable function, metric_fn, that calculates the
  metrics and returns a dictionary. The metric function computes f(x_i, y_i) on
  a minibatch, it has API:
    ```metric_fn(logits, label, weights).```

  The trainer will then aggregate and compute the mean across all samples
  evaluated.

  loss_fn is a function of API
    loss = loss_fn(logits, batch, model_params=None).

  This model class defines a cross_entropy_loss with weight decay, where the
  weight decay factor is determined by config.l2_decay_factor.

  flax_model is returned from the build_flax_model function. A typical
  usage pattern will be:
    ```
    model_cls = model_lib.models.get_model_cls('fully_connected_classification')
    model = model_cls(config, dataset.meta_data)
    flax_model = model.build_flax_model
    dummy_input = jnp.zeros(input_shape, model_input_dtype)
    model_state, params = flax_model.init(
        rng, dummy_input, train=False).pop('params')
    ```
  And this is how to call the model:
    variables = {'params': params, **model_state}
    logits, new_model_state = flax_model.apply(variables, inputs, ...)
    ```
  """

  def __init__(
      self,
      config: Optional[ml_collections.ConfigDict],
      dataset_meta_data: Dict[str, Any],
  ) -> None:
    if config is None:
      logging.warning('You are creating the model with default config.')
      config = self.default_flax_model_config()
    self.config = config
    self.dataset_meta_data = dataset_meta_data
    self.flax_model = self.build_flax_model()

  def get_metrics_fn(self, split: Optional[str] = None) -> MetricFn:
    """Returns a callable metric function for the model.

    The metrics function is for pmap-based models, where we need to normalise
    by doing p-sums over other devices.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].

    Returns:
      A metric function with the following API: ```metrics_fn(logits,
      batch)```
    """
    raise NotImplementedError('Subclasses must implement get_metrics_fn.')

  def get_metrics_fn_jit(self, split: Optional[str] = None) -> MetricFn:
    """Returns a callable metric function for the model.

    The metrics function is for jit-based models, where we normalise by doing
    sums over global arrays.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].

    Returns:
      A metric function with the following API: ```metrics_fn(logits,
      batch)```
    """
    raise NotImplementedError('Subclasses must implement get_metrics_fn_jit.')

  def loss_function(self,
                    logits: jnp.ndarray,
                    batch: Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Returns the loss.

    Args:
      logits: Output of model in shape [batch, length, num_classes].
      batch: Batch of data that has 'label' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """
    raise NotImplementedError('Subclasses must implement loss_function.')

  def build_flax_model(self) -> nn.Module:
    raise NotImplementedError('Subclasses must implement build_flax_model().')

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    """Default config for the flax model that is built in `build_flax_model`.

    This function in particular serves the testing functions and supposed to
    provide config that are passed to the flax_model when it's built in
    `build_flax_model` function, e.g., `model_dtype_str`.
    """
    raise NotImplementedError(
        'Subclasses must implement default_flax_model_config().')
