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

"""Base classes for object detection with matching."""
import abc
import functools
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib import matchers
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import box_utils
from scenic.model_lib.base_models import model_utils
from scenic.projects.owl_vit import losses as losses_lib

ArrayDict = Dict[str, jnp.ndarray]
MetricsDict = Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]
PyTree = Any


class BaseModelWithMatching(base_model.BaseModel, metaclass=abc.ABCMeta):
  """Base model for object detection with matching."""

  def __init__(self, config: Optional[ml_collections.ConfigDict],
               dataset_meta_data: Dict[str, Any]):
    """Initialize model.

    Args:
      config: Configurations of the model.
      dataset_meta_data: Dataset meta data specifies `target_is_onehot`, which
        must be True. The padding-objects have label 0. The first legitimate
        object has label 1, and so on.
    """
    if not dataset_meta_data.get('target_is_onehot', True):
      raise ValueError('Targets must be in one-hot/multi-hot format.')
    self.losses_and_metrics = ['labels']
    if config is not None:
      self.loss_terms_weights = {'loss_class': config.class_loss_coef}
    super().__init__(config, dataset_meta_data)

  @property
  @abc.abstractmethod
  def loss_and_metrics_map(
      self) -> Dict[str, Callable[..., Tuple[ArrayDict, MetricsDict]]]:
    """Returns a dict that lists all losses for this model."""
    return {'labels': self.labels_losses_and_metrics}

  @abc.abstractmethod
  def compute_cost_matrix(self, predictions: ArrayDict,
                          targets: ArrayDict) -> jnp.ndarray:
    """Computes the matching cost matrix.

    Args:
      predictions: Dictionary of outputs from a model.
      targets: Dictionary of ground truth targets.
    Returns:
      The matching cost matrix of shape [B, N, M].
      Number of unpadded columns per batch element [B].
    """
    ...

  def matcher(
      self, cost: jnp.ndarray, n_cols: Optional[jnp.ndarray] = None
  ) -> jnp.ndarray:
    """Implements a matching function.

    Matching functions match predicted detections against ground truth
    detections and return match indices.

    Args:
      cost: Matching cost matrix [B, N, M].
      n_cols: Number of non-padded columns in each cost matrix.

    Returns:
      Matched indices in the form of a list of tuples (src, dst), where
      `src` and `dst` are indices of corresponding predicted and ground truth
      detections. [B, 2, min(N, M)].
    """
    if self.config.matcher == 'hungarian':
      matcher_fn = functools.partial(matchers.hungarian_matcher, n_cols=n_cols)
    elif self.config.matcher == 'hungarian_cover_tpu':
      matcher_fn = matchers.hungarian_cover_tpu_matcher
    else:
      raise ValueError('Unknown matcher (%s).' % self.config.matcher)

    return jax.lax.stop_gradient(matcher_fn(cost))

  def labels_losses_and_metrics(
      self,
      outputs: ArrayDict,
      batch: ArrayDict,
      indices: jnp.ndarray,
      log: bool = True) -> Tuple[ArrayDict, MetricsDict]:
    """Classification loss.

    Args:
      outputs: Model predictions. For the purpose of this loss, outputs must
        have key 'pred_logits'. outputs['pred_logits'] is a nd-array of the
        predicted logits of shape [batch-size, num-objects, num-classes].
      batch: Dict that has 'inputs', 'batch_mask' and, 'label' (ground truth).
        batch['label'] is a dict. For the purpose of this loss, label dict must
        have key 'labels', which the value is an int nd-array of labels with
        shape [batch_size, num_boxes, num_classes + 1]. Since the number of
        boxes (objects) in each example in the batch could be different, the
        input pipeline might add padding boxes to some examples. These padding
        boxes are identified based on their class labels. So if the class label
        is `0`, i.e., a one-hot vector of [1, 0, 0, ..., 0], the box/object is a
        padding object and the loss computation will take that into account. The
        input pipeline also pads the partial batches (last batch of eval/test
        set with num_example < batch_size). batch['batch_mask'] is used to
        identify padding examples which is incorporated to set the weight of
        these examples to zero in the loss computations.
      indices: Matcher output of shape [batch-size, 2, num-objects] which
        conveys source to target pairing of objects.
      log: If true, return classification accuracy as well.

    Returns:
      loss: Dict with 'loss_class' and other model specific losses.
      metrics: Dict with 'loss_class' and other model specific metrics.
    """
    assert 'pred_logits' in outputs
    assert 'label' in batch

    batch_weights = batch.get('batch_mask')
    losses, metrics = {}, {}
    targets = batch['label']
    if isinstance(targets, dict):
      targets = targets['labels']

    src_logits = outputs['pred_logits']

    # Apply the permutation communicated by indices.
    src_logits = model_utils.simple_gather(src_logits, indices[:, 0])
    tgt_labels = model_utils.simple_gather(targets, indices[:, 1])

    unnormalized_loss_class, denom = self._compute_per_example_class_loss(
        tgt_labels=tgt_labels,
        src_logits=src_logits,
        batch_weights=batch_weights,
    )

    metrics['loss_class'] = (unnormalized_loss_class.sum(), denom.sum())

    if self.config.normalization == 'global':
      denom = jax.lax.pmean(denom.sum(), axis_name='batch')
      denom = jnp.maximum(denom, 1.)
      normalized_loss_class = unnormalized_loss_class.sum() / denom
    elif self.config.normalization == 'per_example':
      normalized_loss_class = unnormalized_loss_class.sum(axis=1)
      denom = jnp.maximum(denom, 1.)
      normalized_loss_class = (normalized_loss_class / denom).mean()
    else:
      raise ValueError(f'Unknown normalization {self.config.normalization}.')

    losses['loss_class'] = normalized_loss_class

    if log:
      # Class accuracy for non-padded (label != 0) labels
      not_padded = tgt_labels[:, :, 0] == 0
      if batch_weights is not None:
        not_padded = not_padded * jnp.expand_dims(batch_weights, axis=1)
      num_correct_no_pad = model_utils.weighted_correctly_classified(
          src_logits[..., 1:], tgt_labels[..., 1:], weights=not_padded)
      metrics['class_accuracy_not_pad'] = (num_correct_no_pad, not_padded.sum())

    # Sum metrics and normalizers over all replicas.
    for k, v in metrics.items():
      metrics[k] = model_utils.psum_metric_normalizer(v)
    return losses, metrics

  def _compute_per_example_class_loss(
      self,
      *,
      tgt_labels: jnp.ndarray,
      src_logits: jnp.ndarray,
      batch_weights: Optional[jnp.ndarray],
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the unnormalized per-example classification loss and denom."""
    loss_kwargs = {
        'weights': batch_weights,
    }
    if self.config.focal_loss:
      loss_kwargs['gamma'] = self.config.focal_gamma
      loss_kwargs['alpha'] = self.config.focal_alpha
      loss_fn = model_utils.focal_sigmoid_cross_entropy
    else:
      loss_fn = model_utils.weighted_unnormalized_sigmoid_cross_entropy

    # Don't compute loss for the padding index.
    unnormalized_loss_class = loss_fn(
        src_logits[..., 1:], tgt_labels[..., 1:], **loss_kwargs)
    # Sum losses over all classes. The unnormalized_loss_class is of shape
    # [bs, 1 + max_num_boxes, num_classes], and after the next line, it becomes
    # [bs, 1 + max_num_boxes].
    unnormalized_loss_class = jnp.sum(unnormalized_loss_class, axis=-1)
    # Normalize by number of "true" labels after removing padding label.
    denom = tgt_labels[..., 1:].sum(axis=[1, 2])  # pytype: disable=wrong-arg-types  # jax-ndarray

    if batch_weights is not None:
      denom *= batch_weights

    return unnormalized_loss_class, denom

  def get_losses_and_metrics(
      self, loss: str, outputs: ArrayDict,
      batch: ArrayDict, indices: jnp.ndarray,
      **kwargs: Any) -> Tuple[ArrayDict, MetricsDict]:
    """A convenience wrapper to all the loss_* functions in this class."""
    assert loss in self.loss_and_metrics_map, f'Unknown loss {loss}.'
    return self.loss_and_metrics_map[loss](outputs, batch, indices, **kwargs)

  def loss_function(  # pytype: disable=signature-mismatch
      self,
      outputs: ArrayDict,
      batch: ArrayDict,
      matches: Optional[jnp.ndarray] = None,
      model_params: Optional[PyTree] = None
  ) -> Tuple[jnp.ndarray, MetricsDict]:
    """Loss and metrics function for matching models.

    Args:
      outputs: Model prediction. The exact fields depend on the losses used.
        Please see labels_losses_and_metrics and boxes_losses_and_metrics for
        details.
      batch: Dict that has 'inputs', 'batch_mask' and, 'label' (ground truth).
        batch['label'] is a dict where the keys and values depend on the losses
        used. Please see labels_losses_and_metrics and boxes_losses_and_metrics
        member methods.
      matches: Output of a matcher [B, 2, M]. If not provided, will be computed.
      model_params: pytree (optional); Parameters of the model.

    Returns:
      total_loss: Total loss weighted appropriately using
        self.loss_terms_weights.
      metrics_dict: Individual loss terms with and without weighting for
        logging purposes.
    """
    batch = batch.copy()
    batch['label'] = batch['label'].copy()

    # Append an instance with "padding" label (i.e., "0" as the class label).
    # Shape is [batch, num_instances, num_classes]. This is necessary because
    # the matching code requires at least one padding instance, to which
    # unmatched instances will be assigned.
    label_shape = batch['label']['labels'].shape
    num_classes = label_shape[-1]
    instance = jax.nn.one_hot(0, num_classes)
    reshape_shape = (1,) * (len(label_shape) - 1) + (num_classes,)
    broadcast_shape = label_shape[:-2] + (1, num_classes)
    instance = jnp.broadcast_to(
        jnp.reshape(instance, reshape_shape), broadcast_shape)
    batch['label']['labels'] = jnp.concatenate(
        [batch['label']['labels'], instance], axis=-2)
    if 'boxes' in batch['label']:
      instance = jnp.zeros_like(batch['label']['boxes'][..., :1, :])
      batch['label']['boxes'] = jnp.concatenate(
          [batch['label']['boxes'], instance], axis=-2)

    # Compute matches if not provided.
    if matches is None:
      if 'cost' not in outputs:
        cost, n_cols = self.compute_cost_matrix(outputs, batch['label'])  # pytype: disable=wrong-arg-types  # jax-ndarray
      else:
        cost, n_cols = outputs['cost'], outputs.get('cost_n_cols')
      matches = self.matcher(cost, n_cols)

    if not isinstance(matches, (list, tuple)):
      # Ensure matches come as a sequence.
      matches = [matches]

    # Pad matches if the matching is not complete (i.e. the number of
    # predicted instances is larger than the number of gt instances).
    num_pred = outputs['pred_logits'].shape[-2]

    def pad_matches(match):
      batch_size, _, num_matched = match.shape  # [B, 2, M]
      if num_pred > num_matched:

        def get_unmatched_indices(row, ind):
          return jax.lax.top_k(jnp.logical_not(row.at[ind].set(1)),
                               k=num_pred - num_matched)

        get_unmatched_indices = jax.vmap(get_unmatched_indices)

        indices = jnp.zeros((batch_size, num_pred), dtype=jnp.bool_)
        _, indices = get_unmatched_indices(indices, match[:, 0, :])
        indices = jnp.expand_dims(indices, axis=1)

        padding = jnp.concatenate(
            [indices, jnp.full(indices.shape, fill_value=num_matched - 1)],
            axis=1)
        return jnp.concatenate([match, padding], axis=-1)
      return match

    matches = [pad_matches(match) for match in matches]

    indices = matches[0]

    # Compute all the requested losses and metrics.
    loss_dict = {}
    metrics_dict = {}
    for loss_name in self.losses_and_metrics:
      loss, metrics = self.get_losses_and_metrics(loss_name, outputs, batch,
                                                  indices)
      loss_dict.update(loss)
      metrics_dict.update(metrics)

    # Compute the total loss by combining loss_dict with loss_terms_weights.
    total_loss = []
    for k, v in loss_dict.items():
      if k in self.loss_terms_weights:
        total_loss.append(self.loss_terms_weights[k] * v)
    total_loss = sum(total_loss)

    if self.config.get('l2_decay_factor') is not None:
      l2_loss = model_utils.l2_regularization(model_params)
      metrics_dict['l2_loss'] = (l2_loss, 1)
      total_loss = total_loss + 0.5 * self.config.l2_decay_factor * l2_loss

    # Process metrics dictionary to generate final unnormalized metrics.
    metrics = self.get_metrics(metrics_dict)
    metrics['total_loss'] = (total_loss, 1)
    return total_loss, metrics  # pytype: disable=bad-return-type  # jax-ndarray

  def get_metrics(self, metrics_dict: MetricsDict) -> MetricsDict:
    """Arrange loss dictionary into a metrics dictionary."""
    metrics = {}
    # Some metrics don't get scaled, so no need to keep their unscaled version,
    # i.e. those that are not in self.loss_terms_weights.keys()
    for k, v in metrics_dict.items():
      loss_term = self.loss_terms_weights.get(k)
      if loss_term is not None:
        metrics[f'{k}_unscaled'] = v
        metrics[k] = (loss_term * v[0], v[1])
      else:
        metrics[k] = v

    return metrics


class ObjectDetectionModel(BaseModelWithMatching):
  """Base model for object detection with matching."""

  def __init__(self, config: Optional[ml_collections.ConfigDict],
               dataset_meta_data: Dict[str, Any]):
    """Initializes detection model.

    Args:
      config: Hyper-parameter dictionary.
      dataset_meta_data: Dataset meta data specifies `target_is_onehot`, which
        must be True. The padding-objects have label 0. The first legitimate
        object has label 1, and so on.
    """
    super().__init__(config, dataset_meta_data)
    self.losses_and_metrics.append('boxes')
    if config is not None:
      self.loss_terms_weights['loss_bbox'] = config.bbox_loss_coef
      self.loss_terms_weights['loss_giou'] = config.giou_loss_coef

  @property
  def loss_and_metrics_map(
      self) -> Dict[str, Callable[..., Tuple[ArrayDict, MetricsDict]]]:
    """Returns a dict that lists all losses for this model."""
    return {
        **super().loss_and_metrics_map,
        'boxes': self.boxes_losses_and_metrics,
    }

  def compute_cost_matrix(self, predictions: ArrayDict,
                          targets: ArrayDict) -> jnp.ndarray:
    """Implements the matching cost matrix computations.

    Args:
      predictions: Dictionary of outputs from a model. Must contain 'pred_boxes'
        and 'pred_probs' keys with shapes [B, N, 4] and [B, N, L] respectively.
      targets: Dictionary of ground truth targets. Must contain 'boxes' and
        'labels' keys of shapes [B, M, 4] and [B, M, L] respectively.

    Returns:
      The matching cost matrix of shape [B, N, M].
      Number of unpadded columns per batch element [B].
    """
    return losses_lib.compute_cost(
        tgt_labels=targets['labels'],
        out_logits=predictions['pred_logits'],
        tgt_bbox=targets['boxes'],
        out_bbox=predictions['pred_boxes'],
        class_loss_coef=self.config.class_loss_coef,
        bbox_loss_coef=self.config.bbox_loss_coef,
        giou_loss_coef=self.config.giou_loss_coef,
        focal_loss=self.config.focal_loss,
        focal_alpha=self.config.get('focal_alpha'),
        focal_gamma=self.config.get('focal_gamma'),
        )

  def boxes_losses_and_metrics(
      self,
      outputs: ArrayDict,
      batch: ArrayDict,
      indices: jnp.ndarray) -> Tuple[ArrayDict, MetricsDict]:
    """Bounding box losses: L1 regression loss and GIoU loss.

    Args:
      outputs: dict; Model predictions. For the purpose of this loss, outputs
        must have key 'pred_boxes'. outputs['pred_boxes'] is a nd-array of the
        predicted box coordinates in (cx, cy, w, h) format. This nd-array has
        shape [batch-size, num-boxes, 4].
      batch: dict; that has 'inputs', 'batch_mask' and, 'label' (ground truth).
        batch['label'] is a dict. For the purpose of this loss, batch['label']
        must have key 'boxes', which the value has the same format as
        outputs['pred_boxes']. Additionally in batch['label'], key 'labels' is
        required that should match the specs defined in the member function
        `labels_losses_and_metrics`. This is to decide which boxes are invalid
        and need to be ignored. Invalid boxes have class label 0. If
        batch['batch_mask'] is provided it is used to weight the loss for
        different images in the current batch of examples.
      indices: list[tuple[nd-array, nd-array]]; Matcher output which conveys
        source to target pairing of objects.

    Returns:
      loss: dict with keys 'loss_bbox', 'loss_giou'. These are
        losses averaged over the batch. Therefore they have shape [].
      metrics: dict with keys 'loss_bbox' and 'loss_giou`.
        These are metrics psumed over the batch. Therefore they have shape [].
    """
    assert 'pred_boxes' in outputs
    assert 'label' in batch

    targets = batch['label']
    assert 'boxes' in targets
    assert 'labels' in targets
    losses, metrics = {}, {}
    batch_weights = batch.get('batch_mask')

    src_boxes = model_utils.simple_gather(outputs['pred_boxes'], indices[:, 0])
    tgt_boxes = model_utils.simple_gather(targets['boxes'], indices[:, 1])
    tgt_labels = targets['labels']

    # Some of the boxes are padding. We want to discount them from the loss.
    n_labels_per_instance = jnp.sum(tgt_labels[..., 1:], axis=-1)
    tgt_not_padding = n_labels_per_instance > 0  # [B, M]

    # tgt_is_padding has shape [batch-size, num-boxes].
    # Align this with the model predictions using simple_gather.
    tgt_not_padding = model_utils.simple_gather(tgt_not_padding, indices[:, 1])

    src_boxes_xyxy = box_utils.box_cxcywh_to_xyxy(src_boxes)
    tgt_boxes_xyxy = box_utils.box_cxcywh_to_xyxy(tgt_boxes)
    unnormalized_loss_giou = 1 - box_utils.generalized_box_iou(
        src_boxes_xyxy, tgt_boxes_xyxy, all_pairs=False)

    unnormalized_loss_bbox = model_utils.weighted_box_l1_loss(
        src_boxes_xyxy,
        tgt_boxes_xyxy,
        weights=batch_weights,
    ).sum(axis=2)

    denom = tgt_not_padding.sum(axis=1)
    if batch_weights is not None:
      denom *= batch_weights
      unnormalized_loss_giou = model_utils.apply_weights(
          unnormalized_loss_giou, batch_weights)

    unnormalized_loss_bbox *= tgt_not_padding
    unnormalized_loss_giou *= tgt_not_padding

    if self.config.normalization != 'per_example':
      # Normalize by number of boxes in batch.
      denom = jnp.maximum(jax.lax.pmean(denom.sum(), axis_name='batch'), 1)
      normalized_loss_bbox = unnormalized_loss_bbox.sum() / denom
      normalized_loss_giou = unnormalized_loss_giou.sum() / denom
    else:  # Normalize by number of boxes in image.
      denom = jnp.maximum(denom, 1.)
      normalized_loss_bbox = (unnormalized_loss_bbox.sum(axis=1) / denom).mean()
      normalized_loss_giou = (unnormalized_loss_giou.sum(axis=1) / denom).mean()

    losses['loss_bbox'] = normalized_loss_bbox
    metrics['loss_bbox'] = (normalized_loss_bbox, 1.)
    losses['loss_giou'] = normalized_loss_giou
    metrics['loss_giou'] = (normalized_loss_giou, 1.)

    # Sum metrics and normalizers over all replicas.
    for k, v in metrics.items():
      metrics[k] = model_utils.psum_metric_normalizer(v)  # pytype: disable=wrong-arg-types  # jax-ndarray
    return losses, metrics  # pytype: disable=bad-return-type  # jax-ndarray


class ObjectDetectionModelWithMasks(ObjectDetectionModel):
  """Base model for object detection with matching including a mask loss.

  The masks are predicted one for each instance and must be in direct
  correspondence with the bounding boxes (both predicted and ground truth).

  In practice this is typically used to predict cropped masks within the
  predicted bounding boxes, similar to what has been done with, for example,
  mask RCNN. Right now the cropping must occur outside this class.
  """

  def __init__(self, config: Optional[ml_collections.ConfigDict],
               dataset_meta_data: Dict[str, Any]):
    """Initializes detection model.

    Args:
      config: Hyper-parameter dictionary.
      dataset_meta_data: Dataset meta data specifies `target_is_onehot`, which
        must be True. The padded objects have label 0. The first
        legitimate object has label 1, and so on.
    """
    super().__init__(config, dataset_meta_data)
    self.losses_and_metrics.append('masks')
    if config is not None:
      self.loss_terms_weights['loss_mask'] = config.mask_loss_coef

  @property
  def loss_and_metrics_map(
      self) -> Dict[str, Callable[..., Tuple[ArrayDict, MetricsDict]]]:
    """Returns a dict that lists all losses for this model."""
    return {
        **super().loss_and_metrics_map,
        'boxes': self.boxes_losses_and_metrics,
        'masks': self.masks_losses_and_metrics,
    }

  def masks_losses_and_metrics(
      self,
      outputs: ArrayDict,
      batch: ArrayDict,
      indices: jnp.ndarray) -> Tuple[ArrayDict, MetricsDict]:
    """Mask losses - pixelwise cross entropy a la mask-RCNN.

    Note that it is assumed all masks, both predicted and ground truth, have
    been resized to a fixed height and width (i.e., allowing distortion of the
    aspect ratio).

    Args:
      outputs: Model predictions. For the purpose of this loss, outputs
        must have key 'pred_masks'. outputs['pred_masks'] is a nd-array of the
        predicted masks with (batch, num-instances, h, w) shape, type float32
        in range [0, 1].
      batch: Has keys 'inputs', 'batch_mask' and, 'label' (ground truth).
        batch['label'] is a dict. For the purpose of this loss, batch['label']
        must have key 'masks', which the value has the same format
        as outputs['pred_masks']. Additionally in batch['label'], key
        'valid_masks'. This is to decide which masks are invalid
        and need to be ignored. Invalid masks have label 0. If
        batch['batch_mask'] is provided it is used to weight the loss for
        different images in the current batch of examples.
      indices: Matcher output which conveys source to target pairing of objects.

    Returns:
      loss: Has keys 'loss_mask'. This is mask losses averaged over the batch.
        Therefore it has shape [].
      metrics: Has keys 'loss_mask'. This is the metrics psumed over the batch.
        Therefore it has shape [].
    """
    assert 'pred_masks' in outputs
    assert 'label' in batch

    targets = batch['label']
    assert 'masks' in targets
    assert 'valid_masks' in targets
    assert targets['valid_masks'].shape[1] == targets['masks'].shape[1]
    assert targets['valid_masks'].shape[1] == targets['boxes'].shape[1]
    losses, metrics = {}, {}
    batch_weights = batch.get('batch_mask')

    pred_masks = model_utils.simple_gather(outputs['pred_masks'], indices[:, 0])
    tgt_masks = model_utils.simple_gather(targets['masks'], indices[:, 1])
    tgt_not_padding = model_utils.simple_gather(targets['valid_masks'],
                                                indices[:, 1])

    src_boxes = model_utils.simple_gather(outputs['pred_boxes'], indices[:, 0])
    tgt_boxes = model_utils.simple_gather(targets['boxes'], indices[:, 1])

    if self.config.weight_mask_loss_by_iou:
      src_boxes_xyxy = box_utils.box_cxcywh_to_xyxy(src_boxes)
      tgt_boxes_xyxy = box_utils.box_cxcywh_to_xyxy(tgt_boxes)
      box_iou, _ = box_utils.box_iou(
          src_boxes_xyxy, tgt_boxes_xyxy, all_pairs=False)
      loss_weight = jax.lax.stop_gradient(box_iou)
    else:
      loss_weight = 1.0

    if self.config.move_true_to_pred_mask:
      tgt_masks = jax.vmap(jax.vmap(move_true_to_pred_mask))(
          true_mask=tgt_masks,
          true_box=tgt_boxes,
          pred_box=jax.lax.stop_gradient(src_boxes))

    unnormalized_ce_loss_mask = (
        model_utils.weighted_unnormalized_sigmoid_cross_entropy(
            logits=pred_masks, multi_hot_targets=tgt_masks
        )
    )
    # Take the mean across the spatial height, width of the masks.
    unnormalized_ce_loss_mask = jnp.mean(
        unnormalized_ce_loss_mask, axis=(-1, -2))
    unnormalized_ce_loss_mask *= loss_weight  # Weigh loss by box overlap.
    denom = tgt_not_padding.sum(axis=1)
    if batch_weights is not None:
      denom *= batch_weights
      unnormalized_ce_loss_mask = model_utils.apply_weights(
          unnormalized_ce_loss_mask, batch_weights)
    unnormalized_ce_loss_mask *= tgt_not_padding
    norm_type = self.config.get('normalization')

    if norm_type != 'per_example':
      denom = jnp.maximum(jax.lax.pmean(denom.sum(), axis_name='batch'), 1)
      normalized_ce_loss_mask = unnormalized_ce_loss_mask.sum() / denom
    else:
      denom = jnp.maximum(denom, 1.)
      normalized_ce_loss_mask = (unnormalized_ce_loss_mask.sum(axis=1) /
                                 denom).mean()

    losses['loss_mask'] = normalized_ce_loss_mask.sum()
    metrics['loss_mask'] = (normalized_ce_loss_mask.sum(), 1.)

    # Sum metrics and normalizers over all replicas.
    for k, v in metrics.items():
      metrics[k] = model_utils.psum_metric_normalizer(v)  # pytype: disable=wrong-arg-types  # jax-ndarray
    return losses, metrics  # pytype: disable=bad-return-type  # jax-ndarray


def move_true_to_pred_mask(*, true_mask, true_box, pred_box):
  """Scales and translates the true mask into the pred box reference frame.

  If predicted and true boxes are not identical, then the segmentation masks
  within the boxes will be misaligned. This function scales and translates the
  true mask so that it aligns with the predicted mask, based on the positions
  of the true and predicted bounding boxes.

  This decouples the mask loss from the bounding box locations. In other words,
  the mask loss will be exactly as if we were using full-image masks, rather
  than within-box masks.

  Args:
    true_mask: Array of shape (n, n) containing the true mask for the area
      inside the true bounding box.
    true_box: True (cx, cy, w, h) bounding box.
    pred_box: Predicted (cx, cy, w, h) bounding box.

  Returns:
    New array of shape (n, n) containing the true mask for the area inside the
    predicted box.

  Raises:
    ValueError if true_mask is not a single square 2D mask.
  """
  if true_mask.ndim != 2:
    raise ValueError(f'Expected a single 2D masks, got shape {true_mask.shape}')
  if true_mask.shape[0] != true_mask.shape[1]:
    raise ValueError(f'Expected mask to be square, got shape {true_mask.shape}')

  cx_true, cy_true, w_true, h_true = jnp.split(true_box, 4, axis=-1)
  cx_pred, cy_pred, w_pred, h_pred = jnp.split(pred_box, 4, axis=-1)
  w_pred = jnp.maximum(w_pred, 1e-6)
  h_pred = jnp.maximum(h_pred, 1e-6)

  # Get top-left corner coordinates:
  x0_true = cx_true - w_true / 2
  y0_true = cy_true - h_true / 2
  x0_pred = cx_pred - w_pred / 2
  y0_pred = cy_pred - h_pred / 2

  # Get coordinates of the true box in the reference frame of the pred box:
  x0_true_wrt_pred = (x0_true - x0_pred) / w_pred
  y0_true_wrt_pred = (y0_true - y0_pred) / h_pred
  w_true_wrt_pred = w_true / w_pred
  h_true_wrt_pred = h_true / h_pred

  # Scale and translate true masks:
  width = true_mask.shape[0]
  return jax.image.scale_and_translate(
      true_mask,
      shape=true_mask.shape,
      spatial_dims=(0, 1),
      scale=jnp.concatenate((h_true_wrt_pred, w_true_wrt_pred)),
      translation=width * jnp.concatenate((y0_true_wrt_pred, x0_true_wrt_pred)),
      method='linear')
