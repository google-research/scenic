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

"""Base class for all semantic segmentation models."""

import functools
from typing import Any, Callable, List, Dict, Optional, Tuple, Union

from flax.training import common_utils
from immutabledict import immutabledict
import jax.numpy as jnp
import numpy as np
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils


GlobalMetricFn = Callable[[List[jnp.ndarray], Dict[str, Any]], Dict[str, float]]


def num_pixels(logits: jnp.ndarray,
               one_hot_targets: jnp.ndarray,
               weights: Optional[jnp.ndarray] = None) -> float:
  """Computes number of pixels in the target to be used for normalization.

  It needs to have the same API as other defined metrics.

  Args:
    logits: Unused.
    one_hot_targets: Targets, in form of one-hot vectors.
    weights: Input weights (can be used for accounting the padding in the
      input).

  Returns:
    Number of (non-padded) pixels in the input.
  """
  del logits
  if weights is None:
    return np.prod(one_hot_targets.shape[:3])
  assert weights.ndim == 3, (
      'For segmentation task, the weights should be a pixel level mask.')
  return weights.sum()  # pytype: disable=bad-return-type  # jax-ndarray


# Standard default metrics for the semantic segmentation models.
_SEMANTIC_SEGMENTATION_METRICS = immutabledict({
    'accuracy': (model_utils.weighted_correctly_classified, num_pixels),

    # The loss is already normalized, so we set num_pixels to 1.0:
    'loss': (model_utils.weighted_softmax_cross_entropy, lambda *a, **kw: 1.0)
})


def semantic_segmentation_metrics_function(
    logits: jnp.ndarray,
    batch: base_model.Batch,
    target_is_onehot: bool = False,
    metrics: base_model.MetricNormalizerFnDict = _SEMANTIC_SEGMENTATION_METRICS,
    axis_name: Union[str, Tuple[str, ...]] = 'batch',
) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
  """Calculates metrics for the semantic segmentation task.

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
   target_is_onehot: If the target is a one-hot vector.
   metrics: The semantic segmentation metrics to evaluate. The key is the name
    of the metric, and the value is the metrics function.
   axis_name: List of axes on which we run the pmsum.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """
  if target_is_onehot:
    one_hot_targets = batch['label']
  else:
    one_hot_targets = common_utils.onehot(batch['label'], logits.shape[-1])
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
        (val[0](logits, one_hot_targets, weights), val[1](  # pytype: disable=wrong-arg-types  # jax-types
            logits, one_hot_targets, weights)),
        axis_name=axis_name)
  return evaluated_metrics


class SegmentationModel(base_model.BaseModel):
  """Defines commonalities between all semantic segmentation models.

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

  This model class defines a softmax_cross_entropy_loss with weight decay,
  where the weight decay factor is determined by config.l2_decay_factor.

  flax_model is returned from the build_flax_model function. A typical
  usage pattern will be:
    ```
    model_cls = model_lib.models.get_model_cls('simple_cnn_segmentation')
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
        semantic_segmentation_metrics_function,
        target_is_onehot=self.dataset_meta_data.get('target_is_onehot', False),
        metrics=_SEMANTIC_SEGMENTATION_METRICS)

  def loss_function(self,
                    logits: jnp.ndarray,
                    batch: base_model.Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Returns softmax cross entropy loss with an L2 penalty on the weights.

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
      one_hot_targets = batch['label']
    else:
      one_hot_targets = common_utils.onehot(batch['label'], logits.shape[-1])

    sof_ce_loss = model_utils.weighted_softmax_cross_entropy(
        logits,
        one_hot_targets,
        weights,
        label_smoothing=self.config.get('label_smoothing'),
        label_weights=self.get_label_weights())
    if self.config.get('l2_decay_factor') is None:
      total_loss = sof_ce_loss
    else:
      l2_loss = model_utils.l2_regularization(model_params)
      total_loss = sof_ce_loss + 0.5 * self.config.l2_decay_factor * l2_loss
    return total_loss  # pytype: disable=bad-return-type  # jax-ndarray

  def get_label_weights(self) -> jnp.ndarray:
    """Returns labels' weights to be used for computing weighted loss.

    This can used for weighting the loss terms based on the amount of available
    data for each class, when we have un-balances data for different classes.
    """
    if not self.config.get('class_rebalancing_factor'):
      return None  # pytype: disable=bad-return-type  # jax-ndarray
    if 'class_proportions' not in self.dataset_meta_data:
      raise ValueError(
          'When `class_rebalancing_factor` is nonzero, `class_proportions` must'
          ' be provided in `dataset_meta_data`.')
    w = self.config.get('class_rebalancing_factor')
    assert 0.0 <= w <= 1.0, '`class_rebalancing_factor` must be in [0.0, 1.0]'
    proportions = self.dataset_meta_data['class_proportions']
    proportions = np.maximum(proportions / np.sum(proportions), 1e-8)
    # Interpolate between no rebalancing (w==0.0) and full reweighting (w==1.0):
    proportions = w * proportions + (1.0 - w)
    weights = 1.0 / proportions
    weights /= np.sum(weights)  # Normalize so weights sum to 1.
    weights *= len(weights)  # Scale so weights sum to num_classes.
    return weights

  def get_global_metrics_fn(self) -> GlobalMetricFn:
    """Returns a callable metric function for global metrics.

      The return function implements metrics that require the prediction for the
      entire test/validation dataset in one place and has the following API:
        ```global_metrics_fn(all_confusion_mats, dataset_metadata)```
      If return None, no global metrics will be computed.
    """
    return global_metrics_fn

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


def global_metrics_fn(all_confusion_mats: List[jnp.ndarray],
                      dataset_metadata: Dict[str, Any]) -> Dict[str, float]:
  """Returns a dict with global (whole-dataset) metrics."""
  # Compute mIoU from list of confusion matrices:
  assert isinstance(all_confusion_mats, list)  # List of eval batches.
  cm = np.sum(all_confusion_mats, axis=0)  # Sum over eval batches.
  assert cm.ndim == 3, ('Expecting confusion matrix to have shape '
                        '[batch_size, num_classes, num_classes], got '
                        f'{cm.shape}.')
  cm = np.sum(cm, axis=0)  # Sum over batch dimension.
  mean_iou, iou_per_class = model_utils.mean_iou(cm)
  metrics_dict = {'mean_iou': float(mean_iou)}
  for label, iou in enumerate(iou_per_class):
    tag = f'iou_per_class/{label:02.0f}'
    if 'class_names' in dataset_metadata:
      tag = f"{tag}_{dataset_metadata['class_names'][label]}"
    metrics_dict[tag] = float(iou)
  return metrics_dict
