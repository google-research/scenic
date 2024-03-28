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

"""Base class for moment retrieval models."""

import functools
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import immutabledict
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils
from scenic.projects.unloc import temporal_localization_base_model

Batch = Dict[str, Any]


def _adjust_classification_inputs(
    logits: jnp.ndarray, targets: jnp.ndarray, all_gather_loss: bool
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Adjusts logits and label to have the same shape.

  Args:
    logits: Output of model in shape [batch, batch * max_num_captions,
      num_frames, 1].
    targets: Binary array of shape [batch, max_num_captions, num_frames, 1].
    all_gather_loss: Wether or not to gather results from all devices before
      computing metrics and loss.

  Returns:
    Reshaped logits and targets in shape (batch, batch, max_num_captions,
      num_frames, 1) when all_gather_loss = True. Otherwise, it is
      (batch, max_num_captions, num_frames, 1).
  """
  bs, num_cap, num_frames, _ = targets.shape
  reshaped_logits = logits.reshape((bs, bs, num_cap, num_frames, 1))
  if all_gather_loss:
    expanded_targets = jnp.zeros((bs,) + targets.shape, dtype=targets.dtype)
    expanded_targets = expanded_targets.at[jnp.arange(bs), jnp.arange(bs)].set(
        targets
    )
    return reshaped_logits, expanded_targets
  reshaped_logits = reshaped_logits[jnp.arange(bs), jnp.arange(bs)]
  return reshaped_logits, targets


def _adjust_regression_inputs(displacements: jnp.ndarray) -> jnp.ndarray:
  """Adjusts prediction to have the same shape with ground truth.

  Regression loss is only computed at positive frames.

  Args:
    displacements: Predicted displacements in shape [batch, batch *
      max_num_captions, num_frames, 2].

  Returns:
    Reshaped prediction in shape (batch, max_num_captions, num_frames, 2)
  """
  bs, num_caps, num_frames, _ = displacements.shape
  num_caps_per_vid = num_caps // bs
  reshaped_displacements = displacements.reshape(
      (bs, bs, num_caps_per_vid, num_frames, 2))
  return reshaped_displacements[jnp.arange(bs), jnp.arange(bs)]


def _cls_loss_weights(
    batch_mask: jnp.ndarray,
    caption_mask: jnp.ndarray,
    frame_mask: jnp.ndarray,
    all_gather_loss: bool,
) -> jnp.ndarray:
  """This function is to compute the weights for classification."""
  if all_gather_loss:
    return (batch_mask[:, jnp.newaxis, jnp.newaxis, jnp.newaxis] *
            batch_mask[jnp.newaxis, :, jnp.newaxis, jnp.newaxis] *
            caption_mask[jnp.newaxis, :, :, jnp.newaxis] *
            frame_mask[jnp.newaxis, :, jnp.newaxis, :].astype(jnp.float32))
  return (batch_mask[:, jnp.newaxis, jnp.newaxis] *
          caption_mask[..., jnp.newaxis] *
          frame_mask[:, jnp.newaxis, :].astype(jnp.float32))


def _box_loss_weights(
    batch_mask: jnp.ndarray, caption_mask: jnp.ndarray, frame_mask: jnp.ndarray
) -> jnp.ndarray:
  """This function is to compute the weights for box loss."""
  return (batch_mask[:, jnp.newaxis, jnp.newaxis] *
          caption_mask[..., jnp.newaxis] *
          frame_mask[:, jnp.newaxis, :].astype(jnp.float32))


def weighted_correctly_classified(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Computes weighted number of correctly classified over the given batch.

  Args:
   logits: Output of model in shape [batch, ..., 1].
   targets: Binary array of shape [batch, ..., 1].
   weights: None or array of shape [batch, ...] (rank of targets -1).

  Returns:
    The number of correctly classified examples in the given batch.
  """
  if logits.ndim != targets.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s one_hot_targets' %
        (str(logits.shape), str(targets.shape)))
  preds = logits >= 0.0
  correct = jnp.equal(preds, targets)

  if weights is not None:
    correct = model_utils.apply_weights(correct, weights)

  return correct.astype(jnp.int32)


_MOMENT_RETRIEVAL_SIGMOID_LOSS_CLASSIFICATION_METRICS = (
    immutabledict.immutabledict({
        'accuracy': (weighted_correctly_classified, model_utils.num_examples),
        'sigmoid_classification_loss': (
            model_utils.weighted_unnormalized_sigmoid_cross_entropy,
            model_utils.num_examples,
        ),
    })
)
_MOMENT_RETRIEVAL_FOCAL_LOSS_CLASSIFICATION_METRICS = (
    immutabledict.immutabledict({
        'accuracy': (weighted_correctly_classified, model_utils.num_examples),
        'focal_classification_loss': (
            model_utils.focal_sigmoid_cross_entropy,
            model_utils.num_examples,
        ),
    })
)
_MOMENT_RETRIEVAL_BOX_REGRESSION_METRICS = immutabledict.immutabledict({
    'mean_iou': (
        temporal_localization_base_model.weighted_unnormalized_iou,
        temporal_localization_base_model.num_positive_frames,
    ),
})


def moment_retrieval_metrics_function(
    logits: jnp.ndarray,
    batch: Batch,
    config: ml_collections.ConfigDict,
    classification_metrics: Mapping[
        str, Any
    ] = _MOMENT_RETRIEVAL_FOCAL_LOSS_CLASSIFICATION_METRICS,
    box_regression_metrics: Mapping[
        str, Any
    ] = _MOMENT_RETRIEVAL_BOX_REGRESSION_METRICS,
    axis_name: Union[str, Tuple[str, ...]] = 'batch',
) -> Dict[str, Tuple[float, float]]:
  """Calculates metrics for the moment retrieval task.

  Args:
   logits: Output of model in shape [batch, batch * max_num_captions,
     num_frames, 3].
   batch: Batch of data that has 'label', 'displacements', 'inputs' and
     optionally 'batch_mask'.
   config: Loss config.
   classification_metrics: Mapping from classification metric names to metric
     functions.
   box_regression_metrics: Mapping from box metric names to metric functions.
   axis_name: List of axes on which we run the pmsum.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """
  if batch.get('batch_mask') is None:
    batch_mask = jnp.ones((logits.shape[0],), dtype=jnp.float32)
  else:
    batch_mask = batch.get('batch_mask')
  caption_mask = batch['inputs']['caption_mask']
  frame_mask = batch['inputs']['input_mask']

  # This psum is required to correctly evaluate with multihost. Only host 0
  # will report the metrics, so we must aggregate across all hosts. The psum
  # will map an array of shape [n_global_devices, batch_size] -> [batch_size]
  # by summing across the devices dimension. The outer sum then sums across the
  # batch dim. The result is then we have summed across all samples in the
  # sharded batch.
  cls_weights = _cls_loss_weights(
      batch_mask,
      caption_mask,
      frame_mask,
      all_gather_loss=config.get('all_gather_loss', True),
  )
  evaluated_metrics = {}
  class_logits = logits[..., :1]
  class_label = batch['label']
  class_logits, class_label = _adjust_classification_inputs(
      class_logits,
      class_label,
      all_gather_loss=config.get('all_gather_loss', True),
  )
  for key, val in classification_metrics.items():
    if key == 'focal_classification_loss':
      evaluated_metrics[key] = model_utils.psum_metric_normalizer(
          (val[0](
              class_logits,
              class_label,
              cls_weights,
              alpha=config.get('focal_loss_alpha', 0.5),
              gamma=config.get('focal_loss_gamma', 2.0)), val[1](
                  class_logits, class_label, cls_weights)),
          axis_name=axis_name)
    else:
      evaluated_metrics[key] = model_utils.psum_metric_normalizer(
          (val[0](class_logits, class_label, cls_weights), val[1](
              class_logits, class_label, cls_weights)),
          axis_name=axis_name)

  pred_displacements = logits[..., 1:]
  gt_displacements = batch['displacements']
  pred_displacements = _adjust_regression_inputs(pred_displacements)
  iou_weights = _box_loss_weights(batch_mask, caption_mask, frame_mask)
  for key, val in box_regression_metrics.items():
    evaluated_metrics[key] = model_utils.psum_metric_normalizer(
        (val[0](pred_displacements, gt_displacements, batch['label'][..., 0],
                iou_weights), val[1](batch['label'], iou_weights)),
        axis_name=axis_name)
  return evaluated_metrics  # pytype: disable=bad-return-type  # jax-ndarray


class MomentRetrievalModel(base_model.BaseModel):
  """Defines metrics/loss among all moment retrieval models.

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
    cls_metrics = (
        _MOMENT_RETRIEVAL_FOCAL_LOSS_CLASSIFICATION_METRICS if cls_loss_type
        == 'focal' else _MOMENT_RETRIEVAL_SIGMOID_LOSS_CLASSIFICATION_METRICS)
    box_regression_metrics = dict(_MOMENT_RETRIEVAL_BOX_REGRESSION_METRICS)
    for weight_and_type in box_loss_types:
      loss_type = weight_and_type.split('*')[-1]
      box_regression_metrics[f'{loss_type}_loss'] = (
          functools.partial(
              temporal_localization_base_model.weighted_unnormalized_box_regression_loss,
              loss_type=loss_type,
          ),
          temporal_localization_base_model.num_positive_frames,
      )
    return functools.partial(
        moment_retrieval_metrics_function,
        config=self.config,
        classification_metrics=cls_metrics,
        box_regression_metrics=box_regression_metrics,
    )

  def _box_loss(
      self,
      batch_mask: jnp.ndarray,
      caption_mask: jnp.ndarray,
      frame_mask: jnp.ndarray,
      pred_displacements: jnp.ndarray,
      gt_displacements: jnp.ndarray,
      label: jnp.ndarray,
  ) -> jnp.ndarray:
    """Computes box regression loss."""
    weights = _box_loss_weights(batch_mask, caption_mask, frame_mask)
    pred_displacements = _adjust_regression_inputs(pred_displacements)
    box_loss_type = self.config.get('box_loss_type', 'l1+iou')
    box_loss = temporal_localization_base_model.weighted_box_regression_loss(
        pred_displacements,
        gt_displacements,
        label[..., 0],
        weights=weights,
        loss_type=box_loss_type,
    )
    return box_loss

  def _cls_loss(self, batch_mask: jnp.ndarray, caption_mask: jnp.ndarray,
                frame_mask: jnp.ndarray, class_logits: jnp.ndarray,
                label: jnp.ndarray) -> jnp.ndarray:
    """Computes classification loss."""
    classification_loss_type = self.config.get(
        'classification_loss_type', 'sigmoid'
    )
    cls_loss_weights = _cls_loss_weights(
        batch_mask,
        caption_mask,
        frame_mask,
        all_gather_loss=self.config.get('all_gather_loss', True),
    )
    class_logits, label = _adjust_classification_inputs(
        class_logits,
        label,
        all_gather_loss=self.config.get('all_gather_loss', True),
    )
    if classification_loss_type == 'focal':
      classification_loss = (
          temporal_localization_base_model.weighted_focal_sigmoid_cross_entropy(
              class_logits,
              label,
              weights=cls_loss_weights,
              label_smoothing=self.config.get('label_smoothing'),
              alpha=self.config.get('focal_loss_alpha', 0.5),
              gamma=self.config.get('focal_loss_gamma', 2.0),
          )
      )
    elif classification_loss_type == 'sigmoid':
      classification_loss = model_utils.weighted_sigmoid_cross_entropy(
          class_logits,
          label,
          weights=cls_loss_weights,
          label_smoothing=self.config.get('label_smoothing'))
    else:
      raise ValueError(f'Unknown loss type: {classification_loss_type}.')
    return classification_loss

  def loss_function(self,
                    logits: jnp.ndarray,
                    batch: Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Returns the sum of classification and box regression losses.

    Args:
      logits: (batch_size, batch_size * num_max_captions, num_frames, 3).
      batch: Batch of data.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """
    label = batch['label']
    gt_displacements = batch['displacements']
    class_logits = logits[..., :1]
    pred_displacements = logits[..., 1:]
    batch_mask = batch['batch_mask']
    caption_mask = batch['inputs']['caption_mask']
    frame_mask = batch['inputs']['input_mask']
    box_loss = self._box_loss(
        batch_mask,
        caption_mask,
        frame_mask,
        pred_displacements,
        gt_displacements,
        label,
    )
    cls_loss = self._cls_loss(batch_mask, caption_mask, frame_mask,
                              class_logits, label)
    return (
        self.config.get('classification_loss_alpha', 1.0) * cls_loss + box_loss
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
