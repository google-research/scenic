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

"""Box decoder based on CenterNet2."""

import functools
import math
from typing import Any, Dict, Optional
import flax.linen as nn
import jax
import jax.numpy as jnp
from scenic.projects.baselines.centernet.modeling import centernet2
from scenic.projects.baselines.centernet.modeling import centernet_head
from scenic.projects.baselines.centernet.modeling import fpn
from scenic.projects.baselines.centernet.modeling import iou_assignment
from scenic.projects.baselines.centernet.modeling import roi_head_utils
from scenic.projects.baselines.centernet.modeling import roi_heads

Assignment = iou_assignment.Assignment


class SimpleFeaturePyramid(nn.Module):
  """This module implements SimpleFeaturePyramid in paper:`vitdet`.

  It creates pyramid features built on top of the input feature map.

  Modified: remove backbone args to take feature map as input

  Attributes:
    out_channels (int): number of channels in the output feature maps.
    scale_factors (list[float]): list of scaling factors to upsample or
      downsample the input features for creating pyramid features.
    num_top_blocks (int): top level downsample block
    norm (str): the normalization to use.
  """

  in_dim: int = 768
  out_channels: int = 256
  scale_factors: Any = (2.0, 1.0, 0.5)
  num_top_blocks: int = 2
  num_additional_convs: int = 0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, features: jnp.ndarray, train: bool = False):
    results = []
    # dim = features.shape[-1]
    dim = self.in_dim
    conv_transpose = functools.partial(
        nn.ConvTranspose, kernel_size=(2, 2), strides=(2, 2), dtype=self.dtype
    )
    ln = functools.partial(nn.LayerNorm, epsilon=1e-6)
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    for scale in self.scale_factors:
      x = features
      if scale == 4.0:
        stage, idx_base = 2, 4
        x = conv_transpose(dim // 2, name='simfp_2.0')(x)
        x = ln(name='simfp_2.1')(x)
        x = nn.gelu(x, approximate=False)
        x = conv_transpose(dim // 4, name='simfp_2.3')(x)
      elif scale == 2.0:
        stage, idx_base = 3, 1
        x = conv_transpose(dim // 2, name='simfp_3.0')(x)
      elif scale == 1.0:
        stage, idx_base = 4, 0
      elif scale == 0.5:
        stage, idx_base = 5, 1
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
      else:
        raise NotImplementedError(f'scale_factor={scale} is not supported yet.')
      x = conv(
          self.out_channels,
          kernel_size=(1, 1),
          name=f'simfp_{stage}.{idx_base}',
      )(x)
      x = ln(name=f'simfp_{stage}.{idx_base}.norm')(x)
      x = conv(
          self.out_channels,
          kernel_size=(3, 3),
          padding=[(1, 1), (1, 1)],
          name=f'simfp_{stage}.{idx_base + 1}',
      )(x)
      x = ln(name=f'simfp_{stage}.{idx_base + 1}.norm')(x)
      if self.num_additional_convs > 0:
        for i in range(self.num_additional_convs):
          x = conv(
              self.out_channels,
              kernel_size=(3, 3),
              padding=[(1, 1), (1, 1)],
              name=f'simfp_{stage}.{idx_base + 2 + i}',
          )(x)
          x = ln(name=f'simfp_{stage}.{idx_base + 2 + i}.norm')(x)
      results.append(x)

    if self.num_top_blocks == 1:
      x = nn.max_pool(
          results[-1], (1, 1), strides=(2, 2), padding=[(0, 0), (0, 0)]
      )
      results.append(x)
    elif self.num_top_blocks == 2:
      top_block = fpn.TwiceDownsampleBlock(
          out_channels=self.out_channels, dtype=self.dtype, name='top_block'
      )
      p6, p7 = top_block(results[-1])
      results.extend([p6, p7])
    else:
      if self.num_top_blocks != 0:
        raise NotImplementedError(
            f'num_top_blocks={self.num_top_blocks} is not supported yet.'
        )
    return results


class FpnCenterNet2(centernet2.CenterNet2Detector):
  """CenterNet2 with SimpleFPN without backbone."""

  match_gt_thresh: float = 0.8
  use_roi_box_in_training: bool = False

  def setup(self):
    self.fpn = SimpleFeaturePyramid(**self.fpn_args, name='fpn')
    self.proposal_generator = centernet_head.CenterNetHead(
        num_classes=self.num_classes,
        dtype=self.dtype,
        num_levels=len(self.strides),
        name='proposal_generator',
    )

    self.roi_heads = roi_heads.CascadeROIHeads(
        input_strides={str(int(math.log2(s))): s for s in self.strides},
        num_classes=self.roi_num_classes,
        conv_dims=self.roi_conv_dims,
        conv_norm=self.roi_conv_norm,
        fc_dims=self.roi_fc_dims,
        samples_per_image=self.roi_samples_per_image,
        positive_fraction=self.roi_positive_fraction,
        matching_threshold=self.roi_matching_threshold,
        nms_threshold=self.roi_nms_threshold,
        class_box_regression=self.roi_class_box_regression,
        mult_proposal_score=self.roi_mult_proposal_score,
        scale_cascade_gradient=self.roi_scale_cascade_gradient,
        use_sigmoid_ce=self.roi_use_sigmoid_ce,
        add_box_pred_layers=self.roi_add_box_pred_layers,
        return_last_proposal=True,
        return_detection_in_training=self.use_roi_box_in_training,
        score_threshold=self.roi_score_threshold,
        post_nms_num_detections=self.roi_post_nms_num_detections,
    )

  @nn.compact
  def __call__(  # pytype: disable=signature-mismatch
      self,
      image_embeddings: jnp.ndarray,
      image_shape: jnp.ndarray,
      gt_boxes: Optional[jnp.ndarray] = None,
      gt_classes: Optional[jnp.ndarray] = None,
      train: bool = False,
      postprocess: bool = True,
      debug: bool = False,
  ) -> Any:
    """Applies CenterNet2 model on the image embedding.

    Args:
      image_embeddings: B x H' x W' x D
      image_shape: Bx2
      gt_boxes: B x N x 4. Only used in training.
      gt_classes: B x N. Only used in training.
      train: Whether it is training.
      postprocess: If true, return post-processed boxes withe scores and
        classes; If false, return raw network outputs from FPN: heatmaps,
        regressions, RoI regression and classification.
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
    backbone_features = self.fpn(
        image_embeddings, train=train
    )
    outputs = self.proposal_generator(backbone_features, train=train)

    pre_nms_topk = self.pre_nms_topk_train if train else self.pre_nms_topk_test
    post_nms_topk = (
        self.post_nms_topk_train if train else self.post_nms_topk_test
    )
    boxes, scores, classes = self.extract_peaks(
        outputs, pre_nms_topk=pre_nms_topk
    )
    proposals = self.nms(boxes, scores, classes, post_nms_topk=post_nms_topk)
    proposal_boxes = jnp.stack(
        [x[0] for x in proposals], axis=0
    )  # B x num_prop x 4
    proposal_boxes = roi_head_utils.clip_boxes(
        proposal_boxes, image_shape[:, None, [1, 0]])
    proposal_scores = jnp.stack(
        [x[1] for x in proposals], axis=0
    )  # B x num_propq
    rpn_features = {
        str(int(math.log2(s))): v
        for s, v in zip(self.strides, backbone_features)
    }
    # unlike project/centernet impl, we use 1-based class labels by default, so
    # we commented out below conversion
    # if gt_classes is not None and gt_boxes is not None:
    #   gt_classes = gt_classes + (gt_boxes.max(axis=2) > 0)
    # NOTE: convert gt_class to class agnostic
    if self.roi_num_classes == 1 and gt_classes is not None:
      gt_classes = jnp.where(gt_classes > 0, 1, 0)

    detections, metrics = self.roi_heads(
        rpn_features,
        image_shape,
        gt_boxes,
        gt_classes,
        proposal_boxes,
        proposal_scores,
        training=train,
        postprocess=postprocess,
        debug=debug,
    )
    detections.update(outputs)
    return detections, metrics

  def match_gt(self, proposals, gt_boxes, gt_classes):
    """Match proposals and their texts based on bounding box IoU.

    Args:
      proposals: Boxes with array (B, samples_per_image, 4).
      gt_boxes: Boxes with array (B, max_gt_boxes, 4).
      gt_classes: (B, max_gt_boxes). This is needed for background padding.

    Returns:
      matched_idxs: shape (B, samples_per_image).
      matched: shape (B, samples_per_image): 0 or 1.
    """

    def _impl(proposals, gt_boxes, gt_classes):
      iou = roi_head_utils.pairwise_iou(gt_boxes, proposals)
      matched_idxs, assignments = iou_assignment.label_assignment(
          iou,
          [self.match_gt_thresh],
          [Assignment.NEGATIVE, Assignment.POSITIVE],
      )
      matched_classes = gt_classes[matched_idxs]
      matched_classes = jnp.where(
          assignments != Assignment.POSITIVE, 0, matched_classes
      )
      return matched_idxs, matched_classes

    matched_idxs, matched_classes = jax.vmap(_impl, in_axes=0)(
        proposals, gt_boxes, gt_classes
    )
    return matched_idxs, matched_classes

  def loss_function(
      self,
      detections: Dict[str, jnp.ndarray],
      metrics: Dict[str, jnp.ndarray],
      gt_boxes: jnp.ndarray,
      gt_classes: jnp.ndarray,
  ):
    # NOTE: proposal_generator use 0 based gt_classes
    gt_classes = jnp.maximum(gt_classes - 1, 0)
    loss, metrics = super().loss_function(
        {**detections, 'metrics': metrics},
        {'label': {'boxes': gt_boxes, 'labels': gt_classes}},
    )
    metrics['det_loss'] = metrics.pop('total_loss')
    return loss, metrics
