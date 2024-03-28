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

"""Implementation of CenterNet with RoIHeads."""

# pylint: disable=not-callable

import dataclasses
import math
from typing import Any, Dict, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from scenic.projects.baselines.centernet.modeling import centernet
from scenic.projects.baselines.centernet.modeling import roi_heads
from tensorflow.io import gfile

ArrayDict = Dict[str, jnp.ndarray]
MetricsDict = Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]


class CenterNet2Detector(centernet.CenterNetDetector):
  """Two-stage detector with Centernet as proposal network."""
  roi_matching_threshold: Any = (0.6,)
  pre_nms_topk_train: int = 2000
  post_nms_topk_train: int = 1000
  pre_nms_topk_test: int = 1000
  post_nms_topk_test: int = 256
  roi_nms_threshold: float = 0.7
  roi_num_classes: int = 80
  roi_conv_dims: Any = ()
  roi_conv_norm: Optional[str] = None
  roi_fc_dims: Any = (1024, 1024)
  roi_samples_per_image: int = 512
  roi_positive_fraction: float = 0.25
  roi_mult_proposal_score: bool = True
  roi_class_box_regression: bool = False
  roi_scale_cascade_gradient: bool = False
  roi_use_sigmoid_ce: bool = False
  roi_add_box_pred_layers: bool = False
  roi_use_zeroshot_cls: bool = False
  roi_zs_weight_dim: int = 512
  roi_zs_weight_path: Optional[str] = None
  roi_one_class_per_proposal: bool = False
  roi_score_threshold: float = 0.05
  roi_post_nms_num_detections: int = 100
  roi_append_gt_boxes: bool = True
  custom_classifier: Optional[jnp.ndarray] = None

  def setup(self):
    super().setup()
    if isinstance(self.roi_matching_threshold, float):
      roi_matching_threshold = [self.roi_matching_threshold]
    else:
      roi_matching_threshold = self.roi_matching_threshold
    if self.roi_zs_weight_path:
      assert self.custom_classifier is None
      zs_weight = jnp.asarray(
          np.load(gfile.GFile(self.roi_zs_weight_path, 'rb'))
          ).transpose(1, 0)  # roi_zs_weight_dim x roi_num_classes
      zs_weight = jnp.concatenate(
          [jnp.zeros((zs_weight.shape[0], 1), jnp.float32), zs_weight],
          axis=1)  # roi_zs_weight_dim x (roi_num_classes + 1)
      zs_weight = zs_weight / (
          (zs_weight ** 2).sum(axis=0)[None, :] ** 0.5 + 1e-8)  # L2 normalize
    elif self.custom_classifier is not None:
      zs_weight = self.custom_classifier
      zs_weight = jnp.concatenate(
          [jnp.zeros((zs_weight.shape[0], 1), jnp.float32), zs_weight],
          axis=1)  # roi_zs_weight_dim x (roi_num_classes + 1)
      zs_weight = zs_weight / (
          (zs_weight ** 2).sum(axis=0)[None, :] ** 0.5 + 1e-8)  # L2 normalize
    else:
      zs_weight = None
    self.roi_heads = roi_heads.CascadeROIHeads(
        input_strides={str(int(math.log2(s))): s for s in self.strides},
        num_classes=self.roi_num_classes,
        conv_dims=self.roi_conv_dims,
        conv_norm=self.roi_conv_norm,
        fc_dims=self.roi_fc_dims,
        samples_per_image=self.roi_samples_per_image,
        positive_fraction=self.roi_positive_fraction,
        matching_threshold=roi_matching_threshold,
        nms_threshold=self.roi_nms_threshold,
        class_box_regression=self.roi_class_box_regression,
        mult_proposal_score=self.roi_mult_proposal_score,
        scale_cascade_gradient=self.roi_scale_cascade_gradient,
        use_sigmoid_ce=self.roi_use_sigmoid_ce,
        add_box_pred_layers=self.roi_add_box_pred_layers,
        use_zeroshot_cls=self.roi_use_zeroshot_cls,
        zs_weight_dim=self.roi_zs_weight_dim,
        zs_weight=zs_weight,
        one_class_per_proposal=self.roi_one_class_per_proposal,
        score_threshold=self.roi_score_threshold,
        post_nms_num_detections=self.roi_post_nms_num_detections,
        append_gt_boxes=self.roi_append_gt_boxes,
    )

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               gt_boxes: Optional[jnp.ndarray] = None,
               gt_classes: Optional[jnp.ndarray] = None,
               train: bool = False,
               preprocess: bool = False,
               postprocess: bool = True,
               *,
               padding_mask: Optional[jnp.ndarray] = None,
               debug: bool = False) -> Any:
    """Applies CenterNet2 model on the input.

    Args:
      inputs: array of the preprocessed input images, in shape B x H x W x 3.
      gt_boxes: B x N x 4. Only used in training.
      gt_classes: B x N. Only used in training.
      train: Whether it is training.
      preprocess: If using the build-in preprocessing functions on inputs.
      postprocess: If true, return post-processed boxes withe scores and
        classes; If false, return raw network outputs from FPN: heatmaps,
        regressions, RoI regression and classification.
      padding_mask: Binary matrix with 0 at padded image regions.
      debug: Whether the debug mode is enabled. debug=True enables model
        specific logging/storing some values using jax.host_callback.

    Returns:
      If postprocess == False, return a dict of outputs. See the output of
        CenterNetDetector for details. In addition of outputs from CenterNet,
        return outputs from the RoI heads. That includes losses (if train==True)
        or raw outputs from the RoI heads (if train==False).
      If postprocess == True, return a list of tuples. Each tuple is
        three arrays (boxes, scores, classes). boxes is in shape of
        n x 4, scores and classes are both in shape n.
    """
    if preprocess:
      inputs = self.preprocess(inputs, padding_mask)

    backbone_features = self.backbone(inputs, train=train)
    outputs = self.proposal_generator(backbone_features, train=train)

    pre_nms_topk = self.pre_nms_topk_train if train else self.pre_nms_topk_test
    post_nms_topk = (
        self.post_nms_topk_train if train else self.post_nms_topk_test)
    boxes, scores, classes = self.extract_peaks(
        outputs, pre_nms_topk=pre_nms_topk)
    proposals = self.nms(
        boxes, scores, classes, post_nms_topk=post_nms_topk)
    proposal_boxes = jnp.stack(
        [x[0] for x in proposals], axis=0)  # B x num_prop x 4
    proposal_boxes = jnp.maximum(proposal_boxes, 0)
    proposal_boxes = jnp.minimum(
        proposal_boxes, max(inputs.shape[1], inputs.shape[2]))
    proposal_scores = jnp.stack(
        [x[1] for x in proposals], axis=0)  # B x num_propq
    rpn_features = {str(int(math.log2(s))): v for s, v in zip(
        self.strides, backbone_features)}
    # scenic dataloader loads classes in range [0, num_class - 1], and
    #  dpax RoI heads assume gt_classes in range [1, num_class]. Add 1 to valid
    #  gt objects (indicated by any box axis > 0).
    if gt_classes is not None and gt_boxes is not None:
      gt_classes = gt_classes + (gt_boxes.max(axis=2) > 0)
    image_shape = jnp.concatenate([
        jnp.ones((inputs.shape[0], 1), jnp.float32) * inputs.shape[1],
        jnp.ones((inputs.shape[0], 1), jnp.float32) * inputs.shape[2],
    ], axis=1)  # B x 2, in order (height, width)
    detections, metrics = self.roi_heads(
        rpn_features, image_shape,
        gt_boxes, gt_classes,
        proposal_boxes, proposal_scores,
        training=train, postprocess=postprocess, debug=debug)
    if not train:
      if postprocess:
        # Return a list for batch and convert 1-based class id to 0-based.
        per_batch_detection = [
            (d, s, c) for d, s, c in zip(
                detections['detection_boxes'], detections['detection_scores'],
                (detections['detection_classes'] - 1).astype(jnp.int32))]
        return per_batch_detection
      else:
        # Return raw network output.
        outputs.update(detections)
        return outputs
    else:
      # Return training losses computed in the RoI heads.
      outputs['metrics'] = metrics
      return outputs

  def loss_function(
      self,
      outputs: Any,
      batch: Any,
  ):
    """loss function of CenterNet.

    Args:
      outputs: dict of 'heatmaps' and `box_regs`. Both are list of arrays from
        different FPN levels, in shape L x [B, hl, wl, C']. L is the number
        of FPN levels, hl, wl are the shape in FPN level l.
      batch: dict that has 'inputs', 'batch_mask' and, 'label' (ground truth).
        batch['label'] is a dict with the following keys and shape:
          'boxes': B x max_boxes x 4
          'labels': B x max_boxes
    Returns:
      total_loss: Total loss weighted appropriately.
      metrics: auxiliary metrics for debugging and visualization.
    """
    proposal_loss, metrics = super().loss_function(outputs, batch)
    roi_metrics = outputs['metrics']
    metrics.update(roi_metrics)
    loss = proposal_loss
    for k in range(len(self.roi_matching_threshold)):
      loss += (roi_metrics[f'stage{k}_roi_cls_loss'] + roi_metrics[
          f'stage{k}_roi_reg_loss']) / len(self.roi_matching_threshold)
    metrics['total_loss'] = loss
    return loss, metrics


class CenterNet2Model(centernet.CenterNetModel):
  """Scenic Model Wrapper."""

  def build_flax_model(self):
    fields = set(x.name for x in dataclasses.fields(CenterNet2Detector))
    config_dict = {
        k: v for k, v in self.config.model.items() if k in fields}
    return CenterNet2Detector(**config_dict)
