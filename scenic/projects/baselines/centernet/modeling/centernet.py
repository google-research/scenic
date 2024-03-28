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

"""Implementation of CenterNet architecture."""

# pylint: disable=not-callable

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models import base_model
from scenic.projects.baselines.centernet.modeling import centernet_head
from scenic.projects.baselines.centernet.modeling import centernet_utils
from scenic.projects.baselines.centernet.modeling import fpn
from scenic.projects.baselines.centernet.modeling import nms
from scenic.projects.baselines.centernet.modeling import vitdet


ArrayDict = Dict[str, jnp.ndarray]
MetricsDict = Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]

INF = centernet_utils.INF
# ImageNet mean and std from detectron2:
#   https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/
#   defaults.py#L42
IMAGENET_PIXEL_MEAN = (103.530, 116.280, 123.675)
IMAGENET_PIXEL_STD = (57.375, 57.120, 58.395)


class CenterNetDetector(nn.Module):
  """One-stage centernet detector.

  Attributes:
    num_classes: Number of object classes. num_classes = 0 will merge the
      classification feature layers and regression layers. This is
      class-agnostic detection. This is used for proposal-only mode.
    backbone_name: string of the backbone name.
    score_thresh: output score threshold.
    pre_nms_topk: extracting top K pixels and run NMS on it. A high value will
      slow down inference.
    strides: strides of all FPN levels.
    dtype: Data type of the computation (default: float32).
  """

  num_classes: int
  backbone_name: str = 'convnext'
  score_thresh: float = 0.05
  pre_nms_topk: int = 1000
  post_nms_topk: int = 100
  iou_thresh: float = 0.5
  strides: Any = (8, 16, 32, 64, 128)
  fpn_range: Any = ((0, 80), (64, 160), (128, 320), (256, 640), (512, 100000))
  head_norm: str = 'GN'
  hm_min_overlap: float = 0.8
  min_radius: int = 4
  sigmoid_eps: float = 1e-4
  focal_alpha: float = 0.25
  focal_beta: float = 4.
  focal_gamma: float = 2.
  hm_weight: float = 1.
  reg_weight: float = 2.
  sqrt_score: bool = False
  pixel_mean: Any = IMAGENET_PIXEL_MEAN
  pixel_std: Any = IMAGENET_PIXEL_STD
  sync_device_norm: bool = True
  dtype: jnp.dtype = jnp.float32
  vitdet_scale_factors: Any = (2.0, 1.0, 0.5)
  vitdet_num_top_blocks: int = 2
  backbone_args: ml_collections.ConfigDict = dataclasses.field(
      default_factory=ml_collections.ConfigDict)
  fpn_args: ml_collections.ConfigDict = dataclasses.field(
      default_factory=ml_collections.ConfigDict)

  def setup(self):
    if self.backbone_name == 'vitdet':
      self.backbone = vitdet.SimpleFeaturePyramid(
          backbone_args=self.backbone_args,
          scale_factors=self.vitdet_scale_factors,
          num_top_blocks=self.vitdet_num_top_blocks,
          dtype=self.dtype,
          name='backbone')
    else:
      self.backbone = fpn.FPN(
          backbone_name=self.backbone_name,
          in_features=['stage_2', 'stage_3', 'stage_4'],
          backbone_args=self.backbone_args,
          dtype=self.dtype,
          name='backbone')
    self.proposal_generator = centernet_head.CenterNetHead(
        num_classes=self.num_classes, dtype=self.dtype,
        num_levels=len(self.strides),
        norm=self.head_norm,
        name='proposal_generator')

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool = False,
      preprocess: bool = False,
      postprocess: bool = False,
      *,
      padding_mask: Optional[jnp.ndarray] = None,
      debug: bool = False,
      ):
    """Applies CenterNet model on the input.

    Args:
      inputs: array of the preprocessed input images, in shape B x H x W x 3.
      train: Whether it is training.
      preprocess: If using the build-in preprocessing functions on inputs.
      postprocess: If true, return post-processed boxes withe scores and
        classes; If false, return raw network outputs from FPN: heatmaps and
        regressions.
      padding_mask: Binary matrix with 0 at padded image regions.
      debug: Whether the debug mode is enabled. debug=True enables model
        specific logging/storing some values using jax.host_callback.

    Returns:
      If postprocess == False, return a dict of outputs.
        That might be different across detection heads. For example,
        for CenterNet, it will be: 'heatmaps' and `box_regs`. The value of the
        dict should be list of arrays from different FPN levels,
        each in shape B x H' x W' x C'.
      If postprocess == True, return a list of tuples. Each tuple is
        three arrays (boxes, scores, classes). boxes is in shape of
        n x 4, scores and classes are both in shape n.
    """
    if preprocess:
      inputs = self.preprocess(inputs, padding_mask)

    backbone_features = self.backbone(inputs, train=train)
    output = self.proposal_generator(backbone_features, train=train)
    if postprocess:
      output = self.inference(output)
    return output

  def loss_function(
      self,
      outputs: ArrayDict,
      batch: ArrayDict,
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
    # Generate ground truth.
    # grids: # L x [hl*wl, 2]
    # gt_heatmaps: B x m x C, where m = sum_l hl * wl, C is num of classes
    # gt_regs: B x m x 4.
    grids = centernet_utils.get_grid(
        [(x.shape[1], x.shape[2]) for x in outputs['heatmaps']], self.strides)
    gt_heatmaps, gt_regs = self.get_ground_truth(
        batch['label']['boxes'], batch['label']['labels'], grids)

    # Convert output shape to match ground gruth:
    #   L x [B, hl, wl, C] --> B x m x C, where m = sum_l hl * wl.
    heatmaps = centernet_utils.level_first_to_batch_first(outputs['heatmaps'])
    box_regs = centernet_utils.level_first_to_batch_first(outputs['box_regs'])

    # Compute losses.
    reg_loss, _, reg_norm = self.reg_loss(box_regs, gt_regs)
    pos_loss, neg_loss, hm_norm = self.heatmap_focal_loss(heatmaps, gt_heatmaps)
    total_loss = self.hm_weight * (
        pos_loss + neg_loss) + self.reg_weight * reg_loss
    metrics = {'total_loss': total_loss, 'reg_loss': reg_loss,
               'pos_loss': pos_loss, 'neg_loss': neg_loss,
               'hm_norm': hm_norm, 'reg_norm': reg_norm}
    return total_loss, metrics

  def heatmap_focal_loss(self, heatmaps, gt_heatmaps):
    """Compute heatmap loss.

    Args:
      heatmaps: a single array in shape B x m x C. m = sum_l hl * wl. B is
        the batch size, C is the number of classes.
      gt_heatmaps: a single array in shape B x m x C. Pixels with value 1. are
        positive. All other pixels are negative.
    Returns:
      pos_loss: a scalar for the positive loss.
      neg_loss: a scalar for the negative loss.
      norm: a scalar, the normalization factor, which is the number of positive
        pixels. This is for visualization/ debugging only.
    """
    pred = jnp.clip(
        nn.sigmoid(heatmaps),
        self.sigmoid_eps, 1. - self.sigmoid_eps)  # B x m x C
    neg_w = jnp.power(1. - gt_heatmaps, self.focal_beta)  # B x m x C
    pos_w = (gt_heatmaps == 1.).astype(jnp.float32)  # B x m x C
    pos_loss = jnp.log(pred) * jnp.power(1 - pred, self.focal_gamma) * pos_w
    neg_loss = jnp.log(1. - pred) * jnp.power(pred, self.focal_gamma) * neg_w
    norm = jnp.maximum(pos_w.sum(), 1.)  # scalar
    if self.sync_device_norm:  # sync across GPUs. Helpful for small batch size.
      norm = jax.lax.pmean(norm, axis_name='batch')
    pos_loss = pos_loss.sum() / norm  # scalar
    neg_loss = neg_loss.sum() / norm  # scalar
    if self.focal_alpha >= 0:
      pos_loss = self.focal_alpha * pos_loss
      neg_loss = (1. - self.focal_alpha) * neg_loss
    return - pos_loss, - neg_loss, norm / heatmaps.shape[0]

  def reg_loss(self, box_regs, gt_regs):
    """Compute regression loss.

    Args:
      box_regs: a single array in shape B x m x 4. m = sum_l hl * wl. B is
        the batch size.
      gt_regs: a single array in shape B x m x 4. Invalid/ padded pixels have
        gt of - INF.
    Returns:
      reg_loss: a single scalar for the regression loss.
      gious: the per-pixel losses.
      norm: a scalar, the normalization factor, which is the number of pixels
        that have been applied the loss. Note this is different from the norm
        in heatmap loss, where here we apply regression loss to the 3x3 region
        near the center, and the heatmap loss is only applied to the peaks.
    """
    reg_inds = gt_regs.max(axis=2) >= 0  # B x m: find valid pixels.
    gious = centernet_utils.giou_loss(box_regs, gt_regs)  # B x m
    norm = jnp.maximum(reg_inds.sum(), 1.)  # scalar
    if self.sync_device_norm:
      norm = jax.lax.pmean(norm, axis_name='batch')
    reg_loss = (gious * reg_inds).sum() / norm  # scalar
    return reg_loss, gious, norm / reg_inds.shape[0]

  def _get_bbox_ltrb(self, grids, boxes, m, n):
    """generate FCOS style regression targets.

    Args:
      grids: array in shape m x 2: all output pixel coordinates.
      boxes: array in shape n x 4: ground truth boxes.
      m: number of output pixels.
      n: number of objects.
    Returns:
      reg_target: array in shape m x n x 4: the left, top, right, bottom pixel
        distance between each pixel and object center pairs.
    """
    l = grids[:, 0].reshape(m, 1) - boxes[:, 0].reshape(1, n)  # m x n
    t = grids[:, 1].reshape(m, 1) - boxes[:, 1].reshape(1, n)  # m x n
    r = boxes[:, 2].reshape(1, n) - grids[:, 0].reshape(m, 1)  # m x n
    b = boxes[:, 3].reshape(1, n) - grids[:, 1].reshape(m, 1)  # m x n
    reg_target = jnp.stack([l, t, r, b], axis=2)  # m x n x 4
    return reg_target

  def _get_centers_and_expand(self, grids, boxes, stride_per_pixel, m, n):
    """pad arrays which will be used to generate heatmaps.

    Args:
      grids: array in shape m x 2: all output pixel coordinates.
      boxes: array in shape n x 4: ground truth boxes.
      stride_per_pixel: array in shape m: the stride of the FPN level that
        the pixel is from. This will be used for determining if a center is
        within a pixel grid and for normalizing regression target.
      m: number of output pixels.
      n: number of objects.
    Returns:
      centers_expanded: array in shape m x n x 2: expended n object centers.
      centers_discret: array in shape m x n x 2: snap each of the n centers to
        its closest pixels.
      grid_expanded: array in shape m x n x 2: expanded m pixels.
      strides_expanded: array in shape m x n x 2: expanded m pixel strides.
    """
    centers = (boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2  # n x 2
    centers_expanded = jnp.broadcast_to(centers.reshape(1, n, 2), (m, n, 2))
    strides_expanded = jnp.broadcast_to(
        stride_per_pixel.reshape((m, 1, 1)), (m, n, 2))  # m x n x 2
    grid_expanded = jnp.broadcast_to(grids.reshape((m, 1, 2)), (m, n, 2))
    centers_discret = ((centers_expanded / strides_expanded).astype(
        jnp.int32) * strides_expanded).astype(
            jnp.float32) + strides_expanded / 2  # m x n x 2
    return centers_expanded, centers_discret, grid_expanded, strides_expanded

  def _get_positive_masks(
      self, centers_discret, grid_expanded, strides_expanded,
      is_valid, reg_target, fpn_range):
    """define positive/ negative pixels for regression and heatmap.

    Args:
      centers_discret: array in shape m x n x 2: snap each of the n centers to
        its closest pixels.
      grid_expanded: array in shape m x n x 2: expanded m pixels.
      strides_expanded: array in shape m x n x 2: expanded m pixel strides.
      is_valid: array in shape m x n: if the object is padded for batching.
      reg_target: array in shape m x n x 4: pairwise pixel-to-center distance.
      fpn_range: array in shape m x 2: the FPN assign threshold for the FPN
        level that each pixel is from.
    Returns:
      reg_pos_mask: array in shape m x n: if we will apply regression loss to
        a pixel-object pair.
      heatmap_pos_mask: array in shape m x n: if we will apply a positive
        heatmap loss to a pixel-object pair.
    """
    is_peak = ((grid_expanded - centers_discret) ** 2).sum(
        axis=2) == 0  # m x n
    is_in_boxes = reg_target.min(axis=2) > 0  # m x n
    is_center3x3 = centernet_utils.get_center3x3(
        grid_expanded, centers_discret, strides_expanded)  # m x n
    is_in_fpn_level = centernet_utils.assign_fpn_level(
        reg_target, fpn_range)  # m x n
    # pixels to apply regression losses
    reg_pos_mask = is_center3x3 & is_in_boxes & is_in_fpn_level  # m x n
    reg_pos_mask = reg_pos_mask & is_valid  # m x n, remove padded boxes
    # positive pixels are defined here as (is_peak & is_in_fpn_level)
    heatmap_pos_mask = is_peak & is_in_fpn_level
    return reg_pos_mask, heatmap_pos_mask

  def _get_regression_and_heatmaps(
      self, area, centers_expanded, grid_expanded, stride_per_pixel,
      is_valid, reg_target, reg_pos_mask, heatmap_pos_mask, labels, m, n):
    """generate heatmaps and normalized regression targets.

    Args:
      area: array in shape n: area if each ground truth box.
      centers_expanded: array in shape m x n x 2: expended n object centers.
      grid_expanded: array in shape m x n x 2: expanded m pixels.
      stride_per_pixel: array in shape m: the stride of the FPN level that
        the pixel is from. This will be used for determining if a center is
        within a pixel grid and for normalizing regression target.
      is_valid: array in shape m x n: if the object is padded for batching.
      reg_target: array in shape m x n x 4: pairwise pixel-to-center distance.
      reg_pos_mask: array in shape m x n: if we will apply regression loss to
        a pixel-object pair.
      heatmap_pos_mask: array in shape m x n: if we will apply a positive
        heatmap loss to a pixel-object pair.
      labels: int array in shape n: the class label of each object.
      m: number of output pixels.
      n: number of objects.
    Returns:
      reg: array in shape m x 4: the regression target of each pixel.
      heatmap: array in shape m x C: the heatmap to apply loss in Eq.1 of the
        CenterNet paper (https://arxiv.org/pdf/1904.07850.pdf).
    """
    dist2 = ((grid_expanded - centers_expanded) ** 2).sum(axis=2)  # m x n
    delta = (1. - self.hm_min_overlap) / (1. + self.hm_min_overlap)  # scalar
    radius2 = delta ** 2 * 2 * area  # n
    radius2 = jnp.maximum(radius2, self.min_radius ** 2)  # n
    weighted_dist2 = dist2 / jnp.broadcast_to(radius2.reshape(1, n), (m, n))
    weighted_dist2 = jnp.maximum(weighted_dist2, 1e-6)  # ensure neg gt<1
    weighted_dist2 = weighted_dist2 * (1. - heatmap_pos_mask)  # m x n
    # remove padded boxes
    weighted_dist2 = weighted_dist2 * is_valid + (1. - is_valid) * INF
    # render dense heatmaps and regression maps.
    heatmap = centernet_utils.create_heatmaps(
        weighted_dist2, labels, self.num_classes)  # m x C
    reg = centernet_utils.get_reg_targets(
        reg_target, weighted_dist2, reg_pos_mask)  # m x 4
    reg = reg / stride_per_pixel[:, None]
    return reg, heatmap

  def get_ground_truth(self, gt_boxes, gt_labels, grids):
    """Generate ground truth heatmaps and regression maps.

    Args:
      gt_boxes: array in shape (B x max_boxes x 4)
      gt_labels: array in shape (B x max_boxes)
      grids: List of arrays, in shape L x [hl * wl, 2]. L is the number of
        FPN levels. hl, wl are the size in FPN level l.
    Returns:
      gt_heatmaps: array in shape B x m x C. m = sum_l hl * wl. B is
        the batch size, C is the number of classes.
      gt_regs: array in shape B x m x 4.
    """
    num_locs_per_level = [len(x) for x in grids]  # L
    stride_per_pixel = jnp.concatenate([
        jnp.ones(x, dtype=jnp.float32) * self.strides[l]
        for l, x in enumerate(num_locs_per_level)], axis=0)  # m = sum_l wl * hl
    fpn_range = jnp.concatenate(
        [jnp.broadcast_to(jnp.asarray(
            r, dtype=jnp.float32).reshape(1, 2), (x, 2))
         for x, r in zip(num_locs_per_level, self.fpn_range)], axis=0)  # m x 2
    grids = jnp.concatenate(grids, axis=0)  # m x 2
    m = len(grids)
    gt_heatmaps, gt_regs = [], []
    batch_size = gt_boxes.shape[0]
    for i in range(batch_size):
      n = gt_boxes[i].shape[0]  # n = max_boxes, including padded ones.
      boxes = gt_boxes[i]  # n * 4 in order of [l, t, r, b]
      labels = jnp.minimum(gt_labels[i], max(self.num_classes - 1, 0))  # n
      area = jnp.prod(boxes[:, 2:] - boxes[:, :2], axis=1)  # n

      reg_target = self._get_bbox_ltrb(grids, boxes, m, n)  # m x n x 4

      centers_expanded, centers_discret, grid_expanded, strides_expanded = (
          self._get_centers_and_expand(grids, boxes, stride_per_pixel, m, n)
      )  # all arrays are m x n x 2

      is_valid = jnp.broadcast_to((area > 0).reshape(1, n), (m, n))  # m x n
      reg_pos_mask, heatmap_pos_mask = self._get_positive_masks(
          centers_discret, grid_expanded, strides_expanded,
          is_valid, reg_target, fpn_range
      )  # all arrays are m x n

      reg, heatmap = self._get_regression_and_heatmaps(
          area, centers_expanded, grid_expanded, stride_per_pixel,
          is_valid, reg_target, reg_pos_mask, heatmap_pos_mask, labels, m, n)

      gt_heatmaps.append(heatmap)
      gt_regs.append(reg)

    gt_heatmaps = jnp.stack(gt_heatmaps, axis=0)  # B x m x C
    gt_regs = jnp.stack(gt_regs, axis=0)  # B x m x 4
    return gt_heatmaps, gt_regs

  def extract_peaks(self, outputs, pre_nms_topk):
    """Concert dense outputs from the network to objects.

    Args:
      outputs: dict of list of arrays. The keys should be 'heatmaps' and
        'box_regs'. Both should be a list of arrays in FPN levels.
        'heatmaps' has a shape of B x Hl x Wl x C, 'box_regs' has a shape
        of B x Hl x Wl x 4.
      pre_nms_topk: int: number of peaks to extract

    Returns:
      boxes: float arrays in shape B x n x 4. B is the batch size, n is the
        number of detected objects, which is limited by pre_nms_topk.
        Boxes are in absolute coordinate in order of (l, t, r, b).
      scores: float arrays in shape B x n in range [0, 1].
      classes: int arrays in shape B x n: classes of each object in range [0, C)
    """
    grids = centernet_utils.get_grid(
        [(x.shape[1], x.shape[2]) for x in outputs['heatmaps']],
        self.strides)  # L x [hl*wl, 2]
    grids = jnp.concatenate(grids, axis=0)  # m x 2
    box_regs = [x * self.strides[l] for l, x in enumerate(outputs['box_regs'])]
    # Convert output shape
    #   L x [bs, hl, wl, C] --> bs x m x C, where m = sum_l hl * wl.
    box_regs = centernet_utils.level_first_to_batch_first(box_regs)
    heatmaps = centernet_utils.level_first_to_batch_first(outputs['heatmaps'])
    heatmaps = nn.sigmoid(heatmaps)
    bs, m, c = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2]
    k = min(pre_nms_topk, m * c - 1)
    scores, inds = jax.lax.top_k(heatmaps.reshape(bs, -1), k)  # bs x k
    if self.sqrt_score:
      scores = jnp.sqrt(scores)
    loc, classes = inds // c, inds % c  # bs x k
    grids = jnp.broadcast_to(grids[None], (bs, m, 2))
    # Equivalence of torch.gather(grid, 2, loc[..., None].expand(bs, m, 2))
    gather_inds = jnp.arange(bs * k) // k * m + loc.reshape(bs * k)
    peaks = grids.reshape(bs * m, 2)[gather_inds].reshape(bs, k, 2)
    regs = box_regs.reshape(bs * m, 4)[gather_inds].reshape(bs, k, 4)
    boxes = jnp.stack([
        peaks[:, :, 0] - regs[:, :, 0],
        peaks[:, :, 1] - regs[:, :, 1],
        peaks[:, :, 0] + regs[:, :, 2],
        peaks[:, :, 1] + regs[:, :, 3],
    ], axis=2)  # bs x k x 4
    return boxes, scores, classes

  def nms(self, boxes, scores, classes, post_nms_topk):
    """Running NMS on batched objects.

    Args:
      boxes: float arrays in shape B x n x 4. Boxes are in absolute coordinate
        in order of (l, t, r, b).
      scores: float arrays in shape B x n in range [0, 1].
      classes: int arrays in shape B x n: classes of each object in range [0, C)
      post_nms_topk: int; number of boxes after NMS.
    Returns:
      detection results in batched list of tuples. Each tuple is
        three arrays (boxes, scores, classes). boxes is in shape of
        n_i x 4, scores and classes are both in shape n_i. Different batches
        can have different number of objects.
    """
    results = []
    for box_i, sc_i, cls_i in zip(boxes, scores, classes):
      box_i, sc_i, cls_i = nms.batched_nms_jax(
          box_i, sc_i, cls_i, post_nms_topk, self.iou_thresh)
      results.append((box_i, sc_i, cls_i))
    return results

  def inference(
      self,
      outputs,
  ) -> List[Tuple[Any, Any, Any]]:
    """Generate detections from model outputs.

    Args:
      outputs: dict of list of arrays. The keys should be 'heatmaps' and
        'box_regs'. Both should be a list of arrays in FPN levels.
        'heatmaps' has a shape of B x Hl x Wl x C, 'box_regs' has a shape
        of B x Hl x Wl x 4.

    Returns:
      detection results in batched list of tuples. Each tuple is
        three arrays (boxes, scores, classes). boxes is in shape of
        n x 4, scores and classes are both in shape n.
    """
    boxes, scores, classes = self.extract_peaks(
        outputs, pre_nms_topk=self.pre_nms_topk)
    results = self.nms(
        boxes, scores, classes, post_nms_topk=self.post_nms_topk)
    return results

  def preprocess(self, inputs, padding_mask=None):
    """Proprocess images. Normalize pixels for non-padded pixels."""
    mean = jnp.asarray(self.pixel_mean, dtype=self.dtype).reshape(1, 1, 1, 3)
    std = jnp.asarray(self.pixel_std, dtype=self.dtype).reshape(1, 1, 1, 3)
    inputs = (inputs.astype(self.dtype) - mean) / std
    if padding_mask is not None:
      inputs = inputs * padding_mask[..., None]  # Padded pixels remain 0
    return inputs


class CenterNetModel(base_model.BaseModel):
  """Scenic Model Wrapper."""

  def build_flax_model(self):
    fields = set(x.name for x in dataclasses.fields(CenterNetDetector))
    config_dict = {
        k: v for k, v in self.config.model.items() if k in fields}
    return CenterNetDetector(**config_dict)

  def loss_function(self, outputs, batch):
    return self.flax_model.loss_function(outputs, batch)
