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

"""Code to implement a cascade ROI head in two stage detectors.

The original FasterRCNN head can be a special case of CascadeROIHead by
setting a single matching_threshold.

Modified from
https://github.com/google-research/google-research/blob/master/fvlm/
modeling/heads.py

"""

import math
from typing import Any, Dict, List, Optional

from flax import linen as nn
import jax
import jax.numpy as jnp
import optax
from scenic.projects.baselines.centernet.modeling import box_head as roi_box_head
from scenic.projects.baselines.centernet.modeling import iou_assignment
from scenic.projects.baselines.centernet.modeling import roi_align
from scenic.projects.baselines.centernet.modeling import roi_head_utils

Assignment = iou_assignment.Assignment
ArrayDict = Dict[str, jnp.ndarray]
ArrayList = List[jnp.ndarray]
Array = jnp.ndarray


class CascadeROIHeads(nn.Module):
  """Module that performs per-region computation in an R-CNN."""

  input_strides: dict[str, int]
  """Stride of each input feature.

  The keys can be a subset of what will be passed to __call__. In that case
  the extra input features are not used.
  """
  num_classes: int = 80
  conv_dims: Any = ()
  conv_norm: Optional[str] = None
  fc_dims: Any = (1024, 1024)

  samples_per_image: int = 512
  """Number of RoI samples to train on."""

  positive_fraction: float = 0.25
  """The desired fraction of positive RoIs selected for training."""

  matching_threshold: Any = (0.6, 0.7, 0.8)
  """IoU threshold to match proposals with groundtruth.
  Proposals are assigned foreground or background based on this threshold.
  """
  cascade_box_weights: Any = (
      (10.0, 10.0, 5.0, 5.0),
      (20.0, 20.0, 10.0, 10.0),
      (30.0, 30.0, 15.0, 15.0),)

  nms_threshold: float = 0.7  # Final NMS threshold
  class_box_regression: bool = False  # class specific box head
  mult_proposal_score: bool = True
  scale_cascade_gradient: bool = False
  use_sigmoid_ce: bool = False
  use_zeroshot_cls: bool = False
  add_box_pred_layers: bool = False
  zs_weight_dim: int = 512
  zs_weight: Optional[jnp.ndarray] = None
  one_class_per_proposal: bool = False
  return_last_proposal: bool = False
  append_gt_boxes: bool = True
  score_threshold: float = 0.05
  post_nms_num_detections: int = 100
  return_detection_in_training: bool = False

  @property
  def _levels(self) -> list[int]:
    """Sorted list of input feature levels."""
    return [int(math.log2(s)) for s in sorted(self.input_strides.values())]

  def label_and_sample_proposals(
      self, proposals: jnp.ndarray, gt_boxes: jnp.ndarray,
      gt_classes: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Match proposals with ground truth, and sample a subset for training.

    Args:
      proposals: (B, T, 4) proposal boxes.
      gt_boxes: (B, M, 4) ground truth boxes.
      gt_classes: (B, M) integer labels in range [1, #classes].

    Returns:
      proposals: (B, T', 4) sampled proposals, sorted by their foreground /
        background status. Foregrounds are placed at the beginning.
      matched_idxs: (B, T') integer id of matched GT for each sampled proposal.
        If the proposal is background, its corresponding item could point to any
        GT.
      matched_classes: (B, T') integer labels for each sampled proposals. Label
        is 0 if the proposal is considered background.
    """

    def _impl(proposals: jnp.ndarray, gt_boxes: jnp.ndarray, gt_classes,
              key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
      iou = roi_head_utils.pairwise_iou(gt_boxes, proposals)
      matched_idxs, assignments = iou_assignment.label_assignment(
          iou, [self.matching_threshold[0]],
          [Assignment.NEGATIVE, Assignment.POSITIVE])
      num_samples = self.samples_per_image
      # Invalid proposals will have IoU=0 but should be ignored.
      valid = (
          (proposals >= 0) | (proposals != proposals[..., :1])).any(axis=-1)
      assignments = jnp.where(valid, assignments, Assignment.IGNORE)
      assignments = iou_assignment.subsample_assignments(
          assignments, num_samples, self.positive_fraction, key)
      # Select the final sampled proposals, with foreground first.
      assert Assignment.IGNORE < Assignment.NEGATIVE < Assignment.POSITIVE
      sampled_idxs = jnp.argsort(-assignments)[:num_samples]

      assignments = assignments[sampled_idxs]
      proposals = proposals[sampled_idxs]
      matched_idxs = matched_idxs[sampled_idxs]

      # Note: invalid box will have an IoU of 0 with any boxes. As long as
      # threshold > 0, invalid GT will not match any proposals.
      matched_classes = gt_classes[matched_idxs]
      matched_classes = jnp.where(assignments != Assignment.POSITIVE, 0,
                                  matched_classes)
      return proposals, matched_idxs, matched_classes  # pytype: disable=bad-return-type  # jax-ndarray

    key = self.make_rng("dropout")
    return jax.vmap(
        _impl, in_axes=0)(proposals, gt_boxes, gt_classes,
                          jax.random.split(key, gt_boxes.shape[0]))

  @nn.compact
  def __call__(
      self, features: dict[str, jnp.ndarray], image_shape: jnp.ndarray,
      gt_boxes: jnp.ndarray, gt_classes: jnp.ndarray,
      proposal_boxes: jnp.ndarray, proposal_scores: jnp.ndarray, *,
      training: bool, postprocess: bool = True, debug: bool = False,
      ):
    """Forward computation.

    Args:
      features: Features to crop RoIs from.
      image_shape: array in shape B x 2.
      gt_boxes: (B, max_gt_boxes, 4)
      gt_classes: (B, max_gt_boxes)
      proposal_boxes: (B, num_boxes, 4) boxes of region proposals.
      proposal_scores: (B, num_boxes) scores of region proposals.
      training: Training mode.
      postprocess: if False, directly return box head outputs.
      debug: if output more logs.

    Returns:
      In training, a dict[str, jnp.ndarray] of losses.
      In eval, a dict of final predictions. See return value of
      `flax/ops/generate_detections.py:process_and_generate_detections()`
    """
    strides = sorted(self.input_strides.items(), key=lambda x: x[1])
    features = [features[s[0]] for s in strides]  # Sorted features
    proposals = proposal_boxes

    if training:
      # Proposals should have no gradients when training RoI heads.
      proposals = jax.lax.stop_gradient(proposals)
      if self.append_gt_boxes:
        proposals = jnp.concatenate([proposals, gt_boxes], axis=1)

      proposals, matched_idxs, matched_classes = (
          self.label_and_sample_proposals(proposals, gt_boxes, gt_classes))
      matched_boxes = jnp.take_along_axis(
          gt_boxes,
          matched_idxs[..., None],
          axis=1,
          mode="promise_in_bounds")

      detections, losses = self._forward_box(  # pytype: disable=wrong-arg-types  # jax-ndarray
          features,
          proposals,
          image_shape,
          matched_boxes=matched_boxes,
          matched_classes=matched_classes,
          gt_boxes=gt_boxes,
          gt_classes=gt_classes,
          training=training,
          debug=debug)

      return detections, losses
    else:
      detections, _ = self._forward_box(
          features, proposals, image_shape,
          proposal_scores=proposal_scores, training=training,
          postprocess=postprocess, debug=debug)
      return detections, {}

  def roi_align(self,
                features: list[jnp.ndarray],
                boxes: jnp.ndarray,
                output_size: int,
                sampling_ratio: int = 2) -> jnp.ndarray:
    """RoIAlign on multilevel features.

    Args:
      features: A sorted list of (B, Hi, Wi, C) features.
      boxes: (B, T, 4) boxes in XYXY format, where T is the max number of boxes.
      output_size: Output resolution.
      sampling_ratio: Over-sampling ratio of each output value.

    Returns:
      Cropped and resized features of shape (B, T, output_size, output_size, C).
      Invalid boxes result in undefined, finite feature values.
    """
    if len(features) != len(self.input_strides):
      raise ValueError(
          "features in roi_align does not match self.input_strides!")
    min_level, max_level = min(self._levels), max(self._levels)
    if len(self.input_strides) != max_level - min_level + 1:
      raise ValueError("Features levels in ROIHeads must be contiguous! "
                       f"Got {self.input_strides}.")
    area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    sqrt_area = jnp.sqrt(jnp.maximum(area, 0))

    # Eqn.(1) in the FPN paper. An input with area 224**2 (typical ImageNet
    # pretraining size) is assigned to feature level 4 (stride 16). We can
    # make these numbers configurable if needed.
    level_assignment = jnp.floor(4 + jnp.log2(sqrt_area / 224 + 1e-8))
    level_assignment = jnp.clip(
        level_assignment, a_min=min_level, a_max=max_level)
    scale = jnp.float_power(2.0, level_assignment)[:, :, None]

    return roi_align.multilevel_roi_align(
        features,
        boxes / scale,
        level_assignment.astype(jnp.int32) - min_level,
        output_size=output_size,
        sampling_ratio=sampling_ratio)

  def _forward_box(
      self,
      features: list[jnp.ndarray],
      proposals: jnp.ndarray,
      image_shape: jnp.ndarray,
      proposal_scores: Optional[jnp.ndarray] = None,
      matched_boxes: Optional[jnp.ndarray] = None,
      matched_classes: Optional[jnp.ndarray] = None,
      gt_boxes: Optional[jnp.ndarray] = None,
      gt_classes: Optional[jnp.ndarray] = None,
      *,
      training: bool,
      postprocess: bool = True,
      debug: bool = False,
  ):
    """Forward box head and get losses or post-processed predictions.

    Args:
      features: A sorted list of (B, Hi, Wi, C) features.
      proposals: array in shape (B, samples_per_image, 4) in XYXY.
      image_shape: array in shape B x 2
      proposal_scores: array in shape (B, samples_per_image): scores of
        proposals. Used in inference only.
      matched_boxes: shape (B, samples_per_image, 4). Used in training only.
      matched_classes: shape (B, samples_per_image). Used in training only.
      gt_boxes: array (B, max_gt_boxes, 4). Used in training only.
      gt_classes: (B, max_gt_boxes). Used in training only.
      training: bool.
      postprocess: mostly should be True. If False (for debugging), return the
        raw outputs of bounding box regression and classification.
      debug: if return more info.
    Returns:
      detection_results: dict, only return if training == False. Otherwise {}.
        detection_boxes: (B, samples_per_image, 4)
        detection_scores: (B, samples_per_image)
        detection_classes: (B, samples_per_image)
      metrics: dict. only return if training == True. Owtherwise {}.
        roi_cls_loss: float scalar.
        roi_reg_loss: float scalar.
    """
    head_outputs = []
    class_outputs = None
    for k, (thresh, box_weights) in enumerate(
        zip(self.matching_threshold, self.cascade_box_weights)):
      if k > 0 and training:
        matched_boxes, matched_classes = self._match_and_label_boxes(
            proposals, gt_boxes, gt_classes, thresh)
      roi_features = self.roi_align(features, proposals, 7)
      # TODO(zhouxy): scale gradient here. This is an approximite version.
      # Fix this with actual gradient scale.
      if self.scale_cascade_gradient:
        roi_features = roi_features / len(self.matching_threshold)
      box_head = roi_box_head.ROIBoxHead(
          num_classes=self.num_classes,
          conv_dims=self.conv_dims,
          conv_norm=self.conv_norm,
          fc_dims=self.fc_dims,
          class_box_regression=self.class_box_regression,
          add_box_pred_layers=self.add_box_pred_layers,
          use_zeroshot_cls=self.use_zeroshot_cls,
          zs_weight_dim=self.zs_weight_dim,
          zs_weight=self.zs_weight,
          bias_init_prob=0.01 if self.use_sigmoid_ce else None,
          name=f"box_head.{k}",
      )
      class_outputs, box_outputs = box_head(
          roi_features, training=training)
      head_outputs.append(
          (proposals, box_outputs, class_outputs,
           matched_boxes, matched_classes))
      proposals = roi_head_utils.decode_boxes(
          box_outputs, proposals, weights=box_weights)
      # Do not backpropgate to previous stage proposals. Otherwise gives nan.
      proposals = jax.lax.stop_gradient(proposals)
      proposals = roi_head_utils.clip_boxes(
          proposals, image_shape[:, None, [1, 0]])

    if not training:
      # Use box predictions from the last cascade stage.
      proposals, box_outputs = head_outputs[-1][0], head_outputs[-1][1]
      if not postprocess:
        return {"class_outputs": class_outputs, "box_outputs": box_outputs}, {}
      # Convert from XYXY to YXYX to be used with existing ops.
      proposals_yxyx = proposals[..., [1, 0, 3, 2]]
      box_outputs = box_outputs[..., [1, 0, 3, 2]]
      # use class predictions as the average of all cascade stages
      if self.use_sigmoid_ce:
        class_outputs = sum([jax.nn.sigmoid(x[2]) for x in head_outputs]
                            ) / len(head_outputs)  # B x N x (C + 1)
      else:
        class_outputs = sum(
            [jax.nn.softmax(x[2], axis=-1) for x in head_outputs]
            ) / len(head_outputs)  # B x N x (C + 1)
      if self.mult_proposal_score:
        class_outputs = (class_outputs * proposal_scores[..., None]) ** 0.5
      if self.one_class_per_proposal:
        class_outputs = class_outputs * (
            class_outputs == class_outputs[..., 1:].max(axis=-1)[..., None])
      detection_results = roi_head_utils.process_and_generate_detections(
          box_outputs, class_outputs, proposals_yxyx, image_shape,
          nms_threshold=self.nms_threshold,
          score_threshold=self.score_threshold,
          post_nms_num_detections=self.post_nms_num_detections,
          class_box_regression=self.class_box_regression,
          box_weights=self.cascade_box_weights[len(self.matching_threshold)-1],
          use_vmap=True)
      # YXYX to XYXY
      detection_results["detection_boxes"] = detection_results[
          "detection_boxes"][..., [1, 0, 3, 2]]
      return detection_results, {}
    else:
      outputs, metrics = {}, {}
      for k, (head_output, box_weights) in enumerate(
          zip(head_outputs, self.cascade_box_weights)):
        (proposals, box_outputs, class_outputs,
         matched_boxes, matched_classes) = head_output
        metrics[f"stage{k}_num_proposals"] = proposals.shape[1]

        batch_size = proposals.shape[0]
        area = (proposals[..., 2] - proposals[..., 0]) * (
            proposals[..., 3] - proposals[..., 1])
        valid_mask = area > 0
        fg_mask = (matched_classes > 0) & valid_mask  # Foreground proposals.
        metrics[f"stage{k}_num_valid_proposals"] = valid_mask.sum() / batch_size
        metrics[f"stage{k}_num_positives_per_img_scalar"] = fg_mask.sum(
            ) / batch_size

        eps = 1e-4
        pred_classes = class_outputs.argmax(axis=-1)
        metrics[f"stage{k}_accuracy_scalar"] = (
            (pred_classes == matched_classes) & valid_mask) / (
                valid_mask.sum() + eps)
        metrics[f"stage{k}_foreground_accuracy_scalar"] = (
            ((pred_classes == matched_classes) & fg_mask).sum() / (
                fg_mask.sum() + eps))
        if self.use_sigmoid_ce:
          gt = jax.lax.stop_gradient(jax.nn.one_hot(
              matched_classes, self.num_classes + 1))  # B x N x (C + 1)
          cls_loss = optax.sigmoid_binary_cross_entropy(class_outputs, gt)
          cls_loss = (
              cls_loss * valid_mask[..., None]).sum() / jnp.size(valid_mask)
        else:
          cls_loss = (optax.softmax_cross_entropy_with_integer_labels(
              class_outputs, matched_classes) * valid_mask).sum() / (
                  valid_mask.sum() + eps)

        box_targets = roi_head_utils.encode_boxes(
            matched_boxes, proposals, weights=box_weights)
        # Invalid GT boxes could encode to infinite values. Reset them to 0.
        box_targets = jnp.where(jnp.isfinite(box_targets), box_targets, 0)

        assert not self.class_box_regression, (
            "class-specific box is not supported for cascade rcnn.")
        reg_loss = jnp.abs(box_outputs - box_targets)
        reg_loss = (reg_loss * fg_mask[..., None]).sum() / jnp.size(fg_mask)
        metrics[f"stage{k}_roi_cls_loss"] = cls_loss
        metrics[f"stage{k}_roi_reg_loss"] = reg_loss
        if debug:
          metrics[f"stage{k}_box_outputs"] = box_outputs
          metrics[f"stage{k}_class_outputs"] = class_outputs
          metrics[f"stage{k}_proposals"] = proposals
          metrics[f"stage{k}_matched_boxes"] = matched_boxes
          metrics[f"stage{k}_matched_classes"] = matched_classes
          metrics[f"stage{k}_box_targets"] = box_targets
      if self.return_last_proposal:
        outputs["last_proposals"] = head_outputs[-1][0]
      if self.return_detection_in_training:
        proposals, box_outputs = head_outputs[-1][0], head_outputs[-1][1]
        # Convert from XYXY to YXYX to be used with existing ops.
        proposals_yxyx = proposals[..., [1, 0, 3, 2]]
        box_outputs = box_outputs[..., [1, 0, 3, 2]]
        # use class predictions as the average of all cascade stages
        if self.use_sigmoid_ce:
          class_outputs = sum([jax.nn.sigmoid(x[2]) for x in head_outputs]
                              ) / len(head_outputs)  # B x N x (C + 1)
        else:
          class_outputs = sum(
              [jax.nn.softmax(x[2], axis=-1) for x in head_outputs]
              ) / len(head_outputs)  # B x N x (C + 1)
        detection_results = roi_head_utils.process_and_generate_detections(
            box_outputs, class_outputs, proposals_yxyx, image_shape,
            nms_threshold=self.nms_threshold,
            score_threshold=self.score_threshold,
            post_nms_num_detections=self.post_nms_num_detections,
            class_box_regression=self.class_box_regression,
            box_weights=self.cascade_box_weights[
                len(self.matching_threshold)-1])
        # YXYX to XYXY
        detection_results["detection_boxes"] = detection_results[
            "detection_boxes"][..., [1, 0, 3, 2]]
        outputs.update(detection_results)
      return outputs, metrics  # pytype: disable=bad-return-type  # jax-ndarray

  def _match_and_label_boxes(self, proposals, gt_boxes, gt_classes, thresh):
    """Match and label boxes in each cascade stages.

    Args:
      proposals: array (B, samples_per_image, 4).
      gt_boxes: array (B, max_gt_boxes, 4). Used in training only.
      gt_classes: (B, max_gt_boxes). Used in training only.
      thresh: float.
    Returns:
      matched_boxes: shape (B, samples_per_image, 4).
      matched_classes: shape (B, samples_per_image).
    """
    def _impl(proposals, gt_boxes, gt_classes):
      iou = roi_head_utils.pairwise_iou(gt_boxes, proposals)
      matched_idxs, assignments = iou_assignment.label_assignment(
          iou, [thresh], [Assignment.NEGATIVE, Assignment.POSITIVE])
      matched_classes = gt_classes[matched_idxs]
      matched_classes = jnp.where(assignments != Assignment.POSITIVE, 0,
                                  matched_classes)
      return matched_idxs, matched_classes
    matched_idxs, matched_classes = jax.vmap(_impl, in_axes=0)(
        proposals, gt_boxes, gt_classes)
    matched_boxes = jnp.take_along_axis(
        gt_boxes,
        matched_idxs[..., None],
        axis=1,
        mode="promise_in_bounds")
    return matched_boxes, matched_classes
