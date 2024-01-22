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

"""Base class for all multi-label classification models."""

import functools
from typing import Dict, Optional, Tuple, Union

from flax.training import common_utils
from immutabledict import immutabledict
import jax.numpy as jnp
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils


# Standard default metrics for the multi-label classification models.
_MULTI_LABEL_CLASSIFICATION_METRICS = immutabledict({
    'prec@1': (model_utils.weighted_top_one_correctly_classified,
               model_utils.num_examples),
    'loss': (model_utils.weighted_unnormalized_sigmoid_cross_entropy,
             model_utils.num_examples)
})


def multilabel_classification_metrics_function(
    logits: jnp.ndarray,
    batch: base_model.Batch,
    target_is_multihot: bool = False,
    metrics: base_model.MetricNormalizerFnDict = _MULTI_LABEL_CLASSIFICATION_METRICS,
    axis_name: Union[str, Tuple[str, ...]] = 'batch',
) -> Dict[str, Tuple[float, int]]:
  """Calculates metrics for the multi-label classification task.

  Currently we assume each metric_fn has the API:
    ```metric_fn(logits, targets, weights)```
  and returns an array of shape [batch_size]. We also assume that to compute
  the aggregate metric, one should sum across all batches, then divide by the
  total samples seen. In this way we currently only support metrics of the 1/N
  sum f(inputs, targets). Note, the caller is responsible for dividing by
  the normalizer when computing the mean of each metric.

  Args:
   logits: Output of model in shape [batch, length, num_classes].
   batch: Batch of data that has 'label' and optionally 'batch_mask'.
   target_is_multihot: If the target is a multi-hot vector.
   metrics: The multi-label classification metrics to evaluate. The key is the
     name of the  metric, and the value is the metrics function.
   axis_name: List of axes on which we run the pmsum.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """
  if target_is_multihot:
    multihot_target = batch['label']
  else:
    # This is to support running a multi-label classification model on
    # single-label classification tasks:
    multihot_target = common_utils.onehot(batch['label'], logits.shape[-1])
  weights = batch.get('batch_mask')  # batch_mask might not be defined

  # This psum is required to correctly evaluate with multihost. Only host 0
  # will report the metrics, so we must aggregate across all hosts. The psum
  # will map an array of shape [n_global_devices, batch_size] -> [batch_size]
  # by summing across the devices dimension. The outer sum then sums across the
  # batch dim. The result is then we have summed across all samples in the
  # sharded batch.
  evaluated_metrics = {}
  for key, val in metrics.items():
    evaluated_metrics[key] = model_utils.psum_metric_normalizer(  # pytype: disable=wrong-arg-types  # jax-ndarray
        (val[0](logits, multihot_target, weights), val[1](  # pytype: disable=wrong-arg-types  # jax-types
            logits, multihot_target, weights)),
        axis_name=axis_name)
  return evaluated_metrics  # pytype: disable=bad-return-type  # jax-types


class MultiLabelClassificationModel(base_model.BaseModel):
  """Defines commonalities between all multi-label classification models.

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

  This model class defines a sigmoid_cross_entropy_loss with weight decay, where
  the weight decay factor is determined by config.l2_decay_factor.

  flax_model is returned from the build_flax_model function. A typical
  usage pattern will be:
    ```
    model_cls = model_lib.models.get_model_cls(
        'fully_connected_multilabel_classification')
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

  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      batch)```
    """
    del split  # For all splits, we return the same metric functions.
    return functools.partial(
        multilabel_classification_metrics_function,
        target_is_multihot=self.dataset_meta_data.get('target_is_onehot',
                                                      False),
        metrics=_MULTI_LABEL_CLASSIFICATION_METRICS)

  def get_metrics_fn_jit(self,
                         split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      batch)```
    """
    del split  # For all splits, we return the same metric functions.
    return functools.partial(
        base_model.metrics_function_jit,
        target_is_multihot=self.dataset_meta_data.get('target_is_onehot',
                                                      False),
        metrics=_MULTI_LABEL_CLASSIFICATION_METRICS)

  def loss_function(
      self,
      logits: jnp.ndarray,
      batch: base_model.Batch,
      model_params: Optional[jnp.ndarray] = None,
  ) -> float:
    """Returns sigmoid cross entropy loss with an L2 penalty on the weights.

    Args:
      logits: Output of model in shape [batch, length, num_classes].
      batch: Batch of data that has 'label' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """
    weights = batch.get('batch_mask')

    if self.dataset_meta_data.get('target_is_onehot', False):
      multihot_target = batch['label']
    else:
      # this is to support running a multi-label classification model on
      # single-label classification tasks
      multihot_target = common_utils.onehot(batch['label'], logits.shape[-1])

    sig_ce_loss = model_utils.weighted_sigmoid_cross_entropy(
        logits,
        multihot_target,
        weights,
        label_smoothing=self.config.get('label_smoothing'))
    if self.config.get('l2_decay_factor') is None:
      total_loss = sig_ce_loss
    else:
      l2_loss = model_utils.l2_regularization(model_params)
      total_loss = sig_ce_loss + 0.5 * self.config.l2_decay_factor * l2_loss
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
