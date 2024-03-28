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

"""Base class for temporal localization models."""

import functools
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

import immutabledict
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils
from scenic.projects.unloc import metrics as unloc_metrics

Batch = Dict[str, Any]
MetricFn = Callable[[jnp.ndarray, Dict[str, Any]], Dict[str, Tuple[float,
                                                                   float]]]


_BOX_REGRESSION_LOSS_FNS = {
    'iou': lambda x, y: 1.0 - unloc_metrics.temporal_iou(x, y),
    'l1': unloc_metrics.normalized_l1,
    'center_offset_squared': unloc_metrics.center_offset_squared,
}


def weighted_top_one_correctly_classified(
    logits: jnp.ndarray,
    multihot_targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    background_logit_threshold: float = 0.0,
) -> jnp.ndarray:
  """Computes weighted number of correctly classified.

  We assume there are background samples where the labels are all zeros.

  Args:
    logits: Class logits in shape (batch_size, num_frames, num_classes).
    multihot_targets: Multihot class labels in shape (batch_size, num_frames,
      num_classes).
    weights: None or weights in shape (batch_size, num_frames).
    background_logit_threshold: If the max logit is lower than this score, we
      predict this example as background.

  Returns:
    Weighted numbers of correctly classified samples.
  """
  if logits.shape[-1] == 1:
    preds = logits >= 0.0
    correct = jnp.equal(preds, multihot_targets)
  else:
    top1_idx = jnp.argmax(logits, axis=-1)
    background_label = jnp.sum(multihot_targets, axis=-1) == 0
    background_pred = (
        jnp.max(logits, axis=-1) <= background_logit_threshold
    ).astype(np.int32)

    # Extracts the label at the highest logit index for each input.
    top1_correct = jnp.take_along_axis(
        multihot_targets, top1_idx[..., None], axis=-1)
    top1_correct = jnp.squeeze(top1_correct)
    foreground_correct = ~background_pred.astype(bool) * top1_correct

    # Count correctly classified background samples.
    background_correct = background_pred * background_label
    correct = foreground_correct + background_correct

  if weights is not None:
    return model_utils.apply_weights(correct, weights)
  return correct


def weighted_unnormalized_box_regression_loss(
    displacements: jnp.ndarray,
    gt_displacements: jnp.ndarray,
    label: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    loss_type: str = 'l1+iou',
) -> jnp.ndarray:
  """Computes weighted box regression losses.

  Box regression losses are only computed at frames within a positive segment.

  Args:
    displacements: Predicted start/end time displacements in shape (batch_size,
      num_frames, num_classes, 2) for TAL or (batch_size, num_captions,
      num_frames, 2) for MR.
    gt_displacements: Ground truth start/end time displacements in shape
      (batch_size, num_frames, num_classes, 2) for TAL or (batch_size,
      num_captions, num_frames, 2) for MR.
    label: Multihot vector of shape (batch_size, num_frames, num_classes) for
      TAL or (batch_size, num_captions, num_frames) for MR.
    weights: None or weights array of shape (batch_size, num_frames) for TAL or
      (batch_size, num_captions, num_frames) for MR.
    loss_type: Box regression loss type. Multiple losses can be used
      simultaneously connected by a `+` and one can also specify the weight for
      each loss, e.g., `0.5*l1+1.0*iou`.

  Returns:
    The weighted DIoU losses in shape (batch_size, num_frames, num_classes) for
    TAL or (batch_size, num_captions, num_frames) for MR.
  """

  box_regression_loss = 0.0
  loss_types = loss_type.split('+')
  for weight_and_type in loss_types:
    wt = weight_and_type.split('*')
    (w, t) = (1.0, wt[0]) if len(wt) == 1 else (float(wt[0]), wt[1])
    box_regression_loss += w * _BOX_REGRESSION_LOSS_FNS[t](
        displacements, gt_displacements
    )
  # Only compute the losses on the positive segments.
  box_regression_loss = box_regression_loss * label
  if weights is None:
    return box_regression_loss
  return model_utils.apply_weights(box_regression_loss, weights)


def weighted_unnormalized_iou(
    displacements: jnp.ndarray,
    gt_displacements: jnp.ndarray,
    label: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Computes weighted IoUs.

  IoUs are only computed at frames within a positive segment.

  Args:
    displacements: Predicted start/end time displacements in shape (batch_size,
      num_frames, num_classes, 2) for TAL or (batch_size, num_captions,
      num_frames, 2) for MR.
    gt_displacements: Ground truth start/end time displacements in shape
      (batch_size, num_frames, num_classes, 2) for TAL or (batch_size,
      num_captions, num_frames, 2) for MR.
    label: Multihot vector of shape (batch_size, num_frames, num_classes) for
      TAL or (batch_size, num_captions, num_frames) for MR.
    weights: None or weights array of shape (batch_size, num_frames) for TAL or
      (batch_size, num_captions, num_frames) for MR.

  Returns:
    The weighted IoUs in a batch.
  """

  iou = unloc_metrics.temporal_iou(displacements, gt_displacements)
  # Only compute the losses on the positive segments.
  iou = iou * label
  return model_utils.apply_weights(iou, weights)


def num_positive_frames(
    label: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None) -> Union[jnp.ndarray, float]:
  """Returns number of frames within positive segments."""
  normalizer = jnp.sum(label, axis=-1).astype(bool).astype(np.float32)
  if weights is None:
    return normalizer.sum()
  return (normalizer * weights).sum()


_TEMPORAL_LOCALIZATION_SIGMOID_LOSS_CLASSIFICATION_METRICS = (
    immutabledict.immutabledict({
        'precision@1': (
            weighted_top_one_correctly_classified,
            model_utils.num_examples,
        ),
        'sigmoid_classification_loss': (
            model_utils.weighted_unnormalized_sigmoid_cross_entropy,
            model_utils.num_examples,
        ),
    })
)
_TEMPORAL_LOCALIZATION_FOCAL_LOSS_CLASSIFICATION_METRICS = (
    immutabledict.immutabledict({
        'precision@1': (
            weighted_top_one_correctly_classified,
            model_utils.num_examples,
        ),
        'focal_classification_loss': (
            model_utils.focal_sigmoid_cross_entropy,
            model_utils.num_examples,
        ),
    })
)
_TEMPORAL_LOCALIZATION_BOX_REGRESSION_METRICS = immutabledict.immutabledict({
    'mean_iou': (weighted_unnormalized_iou, num_positive_frames),
})


def weighted_box_regression_loss(
    displacements: jnp.ndarray,
    gt_displacements: jnp.ndarray,
    label: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    loss_type: str = 'l1+iou',
) -> jnp.ndarray:
  """Computes weighted box regression loss.

  Args:
    displacements: Predicted start/end time displacements in shape (batch_size,
      num_frames, num_classes, 2) for TAL or (batch_size, num_captions,
      num_frames, 2) for MR.
    gt_displacements: Ground truth start/end time displacements in shape
      (batch_size, num_frames, num_classes, 2) for TAL or (batch_size,
      num_captions, num_frames, 2) for MR.
    label: Multihot vector of shape (batch_size, num_frames, num_classes) for
      TAL or (batch_size, num_captions, num_frames) for MR.
    weights: None or weights array of shape (batch_size, num_frames) for TAL or
      (batch_size, num_captions, num_frames) for MR.
    loss_type: Box regression loss type. Multiple losses can be used
      simultaneously connected by a `+` and one can also specify the weight for
      each loss, e.g., `0.5*l1+1.0*iou`.

  Returns:
    The mean box regression loss of the examples in a batch as a scalar.
  """

  if weights is not None:
    normalization = model_utils.apply_weights(label, weights).sum()
  else:
    normalization = np.prod(label.shape[:-1])
  box_loss = weighted_unnormalized_box_regression_loss(
      displacements, gt_displacements, label, weights, loss_type
  )
  return jnp.sum(box_loss) / (normalization + 1e-8)


def weighted_focal_sigmoid_cross_entropy(
    logits: jnp.ndarray,
    multi_hot_targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_weights: Optional[jnp.ndarray] = None,
    label_smoothing: Optional[float] = None,
    alpha: Optional[float] = 0.5,
    gamma: Optional[float] = 2.0) -> jnp.ndarray:
  """Computes weighted focal sigmoid cross entropy given logits and targets.

  Args:
    logits: Output of model in shape [batch, ..., num_classes].
    multi_hot_targets: Multi-hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch, ...] (rank of one_hot_targets -1).
    label_weights: None or array of shape broadcastable to the shape of logits.
      Typically this would be [num_classes] and is the weight to apply to each
      label.
    label_smoothing: Scalar to use to smooth the one-hot labels.
    alpha: Focal loss parameter alpha.
    gamma: Focal loss parameter gamma.

  Returns:
    The mean focal loss of the examples in the given batch as a scalar.
  """
  if weights is not None:
    normalization = weights.sum()
  else:
    normalization = np.prod(multi_hot_targets.shape[:-1])

  unnormalized_sigmoid_ce = model_utils.focal_sigmoid_cross_entropy(
      logits,
      multi_hot_targets,
      weights=weights,
      label_weights=label_weights,
      label_smoothing=label_smoothing,
      alpha=alpha,
      gamma=gamma)
  return jnp.sum(unnormalized_sigmoid_ce) / (normalization + 1e-8)


def temporal_localization_metrics_function(
    logits: jnp.ndarray,
    batch: Batch,
    config: ml_collections.ConfigDict,
    classification_metrics: Mapping[
        str, Any
    ] = _TEMPORAL_LOCALIZATION_SIGMOID_LOSS_CLASSIFICATION_METRICS,
    box_regression_metrics: Mapping[
        str, Any
    ] = _TEMPORAL_LOCALIZATION_BOX_REGRESSION_METRICS,
    axis_name: Union[str, Tuple[str, ...]] = 'batch',
) -> Dict[str, Tuple[float, float]]:
  """Calculates metrics for the temporal localization task.

  Args:
   logits: Output of model in shape [batch, num_frames, num_classes * 3] if
     config.output_per_class_displacements = True, otherwise in shape (batch,
     num_frames, num_classes + 2).
   batch: Batch of data that has 'label', 'displacements', and optionally
     'batch_mask'.
   config: Loss config.
   classification_metrics: Mapping from classification metric names to metric
     functions.
   box_regression_metrics: Mapping from box regression metric names to metric
     functions.
   axis_name: List of axes on which we run the pmsum.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """
  if batch.get('batch_mask') is None:
    batch_mask = jnp.ones((logits.shape[0],), dtype=jnp.float32)
  else:
    batch_mask = batch.get('batch_mask')
  weights = batch_mask[:, None] * batch['inputs']['input_mask'].astype(
      jnp.float32)

  # This psum is required to correctly evaluate with multihost. Only host 0
  # will report the metrics, so we must aggregate across all hosts. The psum
  # will map an array of shape [n_global_devices, batch_size] -> [batch_size]
  # by summing across the devices dimension. The outer sum then sums across the
  # batch dim. The result is then we have summed across all samples in the
  # sharded batch.
  evaluated_metrics = {}
  bs, num_frames, _ = logits.shape
  if config.get('output_per_class_displacements', True):
    num_classes = logits.shape[-1] // 3
    reshaped_logits = logits.reshape((bs, num_frames, num_classes, 3))
    class_logits = reshaped_logits[..., 0]
    pred_displacements = reshaped_logits[..., 1:]
    gt_displacements = batch['displacements']
  else:
    num_classes = logits.shape[-1] - 2
    class_logits = logits[..., :num_classes]
    pred_displacements = jnp.expand_dims(logits[..., num_classes:], axis=2)
    gt_displacements = jnp.expand_dims(batch['displacements'], axis=2)

  class_label = batch['label']
  for key, val in classification_metrics.items():
    if key == 'focal_classification_loss':
      evaluated_metrics[key] = model_utils.psum_metric_normalizer(
          (val[0](
              class_logits,
              class_label,
              weights,
              alpha=config.get('focal_loss_alpha', 0.5),
              gamma=config.get('focal_loss_gamma', 2.0)), val[1](
                  class_logits, class_label, weights)),
          axis_name=axis_name)
    else:
      evaluated_metrics[key] = model_utils.psum_metric_normalizer(
          (val[0](class_logits, class_label, weights), val[1](
              class_logits, class_label, weights)),
          axis_name=axis_name)

  for key, val in box_regression_metrics.items():
    evaluated_metrics[key] = model_utils.psum_metric_normalizer(
        (
            val[0](
                pred_displacements,
                gt_displacements,
                class_label,
                weights,
            ),
            val[1](class_label, weights),
        ),
        axis_name=axis_name,
    )
  return evaluated_metrics  # pytype: disable=bad-return-type  # jax-ndarray


class TemporalLocalizationModel(base_model.BaseModel):
  """Defines metrics/loss among all temporal localization models.

  A model is class with three members: get_metrics_fn, loss_fn, & a flax_model.

  get_metrics_fn returns a callable function, metric_fn, that calculates the
  metrics and returns a dictionary. The metric function computes f(logits_i,
  batch_i) on a minibatch, it has API:
    ```metric_fn(logits, batch).```

  The trainer will then aggregate and compute the mean across all samples
  evaluated.

  loss_fn is a function of API
    loss = loss_fn(logits, batch, model_params=None).

  This model class defines two losses, sigmoid cross entropy for classification
  and IoU for boundary regression.
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
    cls_loss_type = self.config.get('classification_loss_type', 'sigmoid')
    box_loss_type = self.config.get('box_loss_type', 'l1+iou')
    box_loss_types = box_loss_type.split('+')
    box_regression_metrics = dict(_TEMPORAL_LOCALIZATION_BOX_REGRESSION_METRICS)
    for weight_and_type in box_loss_types:
      loss_type = weight_and_type.split('*')[-1]
      box_regression_metrics[f'{loss_type}_loss'] = (
          functools.partial(
              weighted_unnormalized_box_regression_loss, loss_type=loss_type
          ),
          num_positive_frames,
      )
    cls_metrics = (
        _TEMPORAL_LOCALIZATION_FOCAL_LOSS_CLASSIFICATION_METRICS
        if cls_loss_type == 'focal' else
        _TEMPORAL_LOCALIZATION_SIGMOID_LOSS_CLASSIFICATION_METRICS)
    return functools.partial(
        temporal_localization_metrics_function,
        config=self.config,
        classification_metrics=cls_metrics,
        box_regression_metrics=box_regression_metrics,
    )

  def loss_function(self,
                    logits: jnp.ndarray,
                    batch: Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Returns the sum of classification and IoU loss.

    Args:
      logits: class logits and predicted start/end time displacements in shape
        (batch_size, num_frames, num_classes * 3) if
        output_per_class_displacements = True, otherwise in shape (batch_size,
        num_frames, num_classes + 2).
      batch: Batch of data.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """
    bs, num_frames, _ = logits.shape
    if self.config.get('output_per_class_displacements', True):
      num_classes = logits.shape[-1] // 3
      reshaped_logits = logits.reshape((bs, num_frames, num_classes, 3))
      class_logits = reshaped_logits[..., 0]
      displacements = reshaped_logits[..., 1:]
      gt_displacements = batch['displacements']
    else:
      num_classes = logits.shape[-1] - 2
      class_logits = logits[..., :num_classes]
      displacements = jnp.expand_dims(logits[..., num_classes:], axis=2)
      gt_displacements = jnp.expand_dims(batch['displacements'], axis=2)
    batch_mask = batch['batch_mask']
    weights = batch_mask[:, None] * batch['inputs']['input_mask'].astype(
        jnp.float32)
    box_loss_type = self.config.get('box_loss_type', 'l1+iou')
    box_loss = weighted_box_regression_loss(
        displacements,
        gt_displacements,
        batch['label'],
        weights=weights,
        loss_type=box_loss_type,
    )
    classification_loss_type = self.config.get('classification_loss_type',
                                               'sigmoid')
    if classification_loss_type == 'focal':
      classification_loss = weighted_focal_sigmoid_cross_entropy(
          class_logits,
          batch['label'],
          weights=weights,
          label_smoothing=self.config.get('label_smoothing'),
          alpha=self.config.get('focal_loss_alpha', 0.5),
          gamma=self.config.get('focal_loss_gamma', 2.0))
    elif classification_loss_type == 'sigmoid':
      classification_loss = model_utils.weighted_sigmoid_cross_entropy(
          class_logits,
          batch['label'],
          weights=weights,
          label_smoothing=self.config.get('label_smoothing'))
    else:
      raise ValueError(f'Unknown loss type: {classification_loss_type}.')
    return (
        self.config.get('classification_loss_alpha', 1.0) * classification_loss
        + box_loss
    )

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
