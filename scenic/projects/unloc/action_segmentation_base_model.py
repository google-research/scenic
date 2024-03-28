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

"""Base class for action segmentation models."""

import functools
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import immutabledict
import jax.numpy as jnp
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils
from scenic.projects.unloc import temporal_localization_base_model

Batch = Dict[str, Any]


_ACTION_SEGMENTATION_METRICS = immutabledict.immutabledict({
    'frame_accuracy': (
        temporal_localization_base_model.weighted_top_one_correctly_classified,
        model_utils.num_examples,
    ),
    'sigmoid_classification_loss': (
        model_utils.weighted_unnormalized_sigmoid_cross_entropy,
        model_utils.num_examples,
    ),
})


def action_segmentation_metrics_function(
    logits: jnp.ndarray,
    batch: Batch,
    metrics: Mapping[str, Any] = _ACTION_SEGMENTATION_METRICS,
    axis_name: Union[str, Tuple[str, ...]] = 'batch',
) -> Dict[str, Tuple[float, float]]:
  """Calculates metrics for the action segmentation task.

  Args:
   logits: Output of model in shape [batch, num_frames, num_classes].
   batch: Batch of data that has 'label', 'displacements', and optionally
     'batch_mask'.
   metrics: Mapping from classification metric names to metric functions.
   axis_name: List of axes on which we run the pmsum.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """
  if batch.get('batch_mask') is None:
    batch_mask = jnp.ones((logits.shape[0],), dtype=jnp.float32)
  else:
    batch_mask = batch.get('batch_mask')
  frame_mask = batch['inputs']['input_mask']
  weights = batch_mask[:, None] * frame_mask
  evaluated_metrics = {}
  class_label = batch['label']
  if len(logits.shape) == 2:
    assert class_label.shape[-1] == 1
    logits = logits.reshape(class_label.shape)
  for key, val in metrics.items():
    evaluated_metrics[key] = model_utils.psum_metric_normalizer(
        (val[0](logits, class_label, weights), val[1](logits, class_label,
                                                      weights)),
        axis_name=axis_name)

  return evaluated_metrics  # pytype: disable=bad-return-type  # jax-ndarray


class ActionSegmentationModel(base_model.BaseModel):
  """Defines metrics/loss among all action segmentation models.

  A model is class with three members: get_metrics_fn, loss_fn, & a flax_model.

  get_metrics_fn returns a callable function, metric_fn, that calculates the
  metrics and returns a dictionary. The metric function computes f(logits_i,
  batch_i) on a minibatch, it has API:
    ```metric_fn(logits, batch).```

  The trainer will then aggregate and compute the mean across all samples
  evaluated.

  loss_fn is a function of API
    loss = loss_fn(logits, batch, model_params=None).
  """

  def get_metrics_fn(self, split: Optional[str] = None):
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      batch)```
    """
    del split  # For all splits, we return the same metric functions.
    return functools.partial(
        action_segmentation_metrics_function,
        metrics=_ACTION_SEGMENTATION_METRICS)

  def loss_function(self,
                    logits: jnp.ndarray,
                    batch: Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Returns the sum of classification loss.

    Args:
      logits: (batch_size, num_frames, num_classes).
      batch: Batch of data.
      model_params: Parameters of the model, not used. Regularization is
        performed inside the optimizer.

    Returns:
      Total loss.
    """
    label = batch['label']
    batch_mask = batch['batch_mask']
    frame_mask = batch['inputs']['input_mask']
    weights = batch_mask[:, None] * frame_mask
    if len(logits.shape) == 2:
      assert label.shape[-1] == 1
      logits = logits.reshape(label.shape)
    return model_utils.weighted_sigmoid_cross_entropy(  # pytype: disable=bad-return-type  # jax-ndarray
        logits,
        label,
        weights,
        label_smoothing=self.config.get('label_smoothing'))

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
