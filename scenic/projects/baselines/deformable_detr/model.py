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

"""DeformableDETR model."""

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from absl import logging
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib import matchers
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import box_utils
from scenic.model_lib.base_models import model_utils
from scenic.projects.baselines.deformable_detr.backbone import DeformableDETRBackbone
from scenic.projects.baselines.deformable_detr.backbone import InputPosEmbeddingSine
from scenic.projects.baselines.deformable_detr.backbone import mask_for_shape
from scenic.projects.baselines.deformable_detr.deformable_transformer import BBoxCoordPredictor
from scenic.projects.baselines.deformable_detr.deformable_transformer import DeformableDETRTransformer
from scenic.projects.baselines.deformable_detr.deformable_transformer import inverse_sigmoid
from scenic.projects.baselines.deformable_detr.deformable_transformer import pytorch_kernel_init

ArrayDict = Dict[str, jnp.ndarray]
MetricsDict = Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]


def compute_cost(
    *,
    tgt_labels: jnp.ndarray,
    out_prob: jnp.ndarray,
    tgt_bbox: jnp.ndarray,
    out_bbox: jnp.ndarray,
    alpha: float = 0.25,
    gamma: float = 2.0,
    class_loss_coef: float = 1.0,
    bbox_loss_coef: float = 1.0,
    giou_loss_coef: float = 1.0,
    target_is_onehot: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes cost matrices for DeformableDETR predictions.

  Relevant code:
  https://github.com/fundamentalvision/Deformable-DETR/blob/11169a60c33333af00a4849f1808023eba96a931/models/matcher.py#L45

  Args:
    tgt_labels: Class labels of shape [bs, ntargets]. If target_is_onehot then
      it is [bs, ntargets, nclasses]. Note that the labels corresponding to
      empty bounding boxes are not yet supposed to be filtered out.
    out_prob: Classification probabilities of shape [bs, nout, nclasses].
    tgt_bbox: Target box coordinates of shape [bs, ntargets, 4]. Note that the
      empty bounding boxes are not yet supposed to be filtered out.
    out_bbox: Predicted box coordinates of shape [bs, nout, 4]
    alpha: Focal loss alpha for class classification loss.
    gamma: Focal loss gamma for class classification loss.
    class_loss_coef: Relative weight of classification loss.
    bbox_loss_coef: Relative weight of bbox loss.
    giou_loss_coef: Relative weight of giou loss.
    target_is_onehot: Whether targets are one-hot encoded.

  Returns:
    All pairs cost matrix [bs, nout, ntargets].
  """
  # Calculate cost using pred_prob [bs, npreds].
  logfn = lambda x: jnp.log(jnp.clip(x, a_min=1e-8))
  neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-logfn(1 - out_prob))
  pos_cost_class = alpha * ((1 - out_prob)**gamma) * (-logfn(out_prob))
  cost_class = pos_cost_class - neg_cost_class

  # Select class cost for target class [bs, npreds, ntargets].
  if target_is_onehot:
    cost_class = jnp.einsum('bnl,bml->bnm', cost_class, tgt_labels)
  else:
    cost_class = jax.vmap(jnp.take, (0, 0, None))(cost_class, tgt_labels, 1)

  # Pairwise box l1 [bs, npreds, ntargets, 4].
  diff = jnp.abs(out_bbox[:, :, None] - tgt_bbox[:, None, :])
  # [bs, npreds, ntargets]
  cost_bbox = jnp.sum(diff, axis=-1)

  # [bs, npreds, ntargets]
  cost_giou = -box_utils.generalized_box_iou(
      box_utils.box_cxcywh_to_xyxy(out_bbox),
      box_utils.box_cxcywh_to_xyxy(tgt_bbox),
      all_pairs=True,
      eps=1e-8)

  total_cost = (
      bbox_loss_coef * cost_bbox + class_loss_coef * cost_class +
      giou_loss_coef * cost_giou)

  # Compute the number of unpadded columns for each batch element. It is assumed
  # that all padding is trailing padding.
  if target_is_onehot:
    tgt_not_padding = tgt_labels[..., 0] == 0
  else:
    tgt_not_padding = tgt_labels != 0
  n_cols = jnp.sum(tgt_not_padding, axis=-1)
  return total_cost, n_cols


def loss_labels(*,
                pred_logits: jnp.ndarray,
                tgt_labels: jnp.ndarray,
                indices: jnp.ndarray,
                alpha: float = 0.25,
                gamma: float = 2.0,
                class_loss_coef: float = 1.0,
                target_is_onehot: bool = False) -> ArrayDict:
  """Calculate DeformableDETR classification loss.

  Args:
    pred_logits: [bs, n_preds, n_classes].
    tgt_labels: [bs, n_max_targets].
    indices: [bs, 2, min(n_preds, n_max_targets)].
    alpha: Focal loss alpha.
    gamma: Focal loss gamma.
    class_loss_coef: Classification loss coefficient.
    target_is_onehot: Tgt is [bs, n_max_targets, n_classes]

  Returns:
    `loss_class`: Classification loss with coefficient applied.
  """
  # Apply the permutation communicated by indices.
  pred_logits = model_utils.simple_gather(pred_logits, indices[..., 0, :])
  tgt_labels = model_utils.simple_gather(tgt_labels, indices[..., 1, :])

  if target_is_onehot:
    tgt_labels_onehot = tgt_labels
  else:
    nclasses = pred_logits.shape[-1]
    tgt_labels_onehot = jnp.where(tgt_labels == 0, nclasses, tgt_labels)
    tgt_labels_onehot = jax.nn.one_hot(tgt_labels_onehot, nclasses)

  loss = model_utils.focal_sigmoid_cross_entropy(
      pred_logits, tgt_labels_onehot, alpha=alpha, gamma=gamma)
  loss = loss.mean(1).sum() * pred_logits.shape[1]
  loss = class_loss_coef * loss
  return {'loss_class': loss}


def loss_boxes(*,
               src_boxes: jnp.ndarray,
               tgt_labels: jnp.ndarray,
               tgt_boxes: jnp.ndarray,
               indices: jnp.ndarray,
               bbox_loss_coef: float = 1.0,
               giou_loss_coef: float = 1.0,
               target_is_onehot: bool = False) -> ArrayDict:
  """Calculate DeformableDETR bounding box losses.

  Args:
    src_boxes: [bs, n_preds, 4].
    tgt_labels: [bs, n_max_targets].
    tgt_boxes: [bs, n_max_targets, 4].
    indices: [bs, 2, min(n_preds, n_max_targets)].
    bbox_loss_coef: L1 box coordinate loss coefficient.
    giou_loss_coef: Generalized IOU (GIOU) loss coefficient.
    target_is_onehot: Tgt is [bs, n_max_targets, n_classes]

  Returns:
    `loss_class`: Classification loss with coefficient applied.
  """
  src_indices = indices[..., 0, :]
  tgt_indices = indices[..., 1, :]

  src_boxes = model_utils.simple_gather(src_boxes, src_indices)
  tgt_boxes = model_utils.simple_gather(tgt_boxes, tgt_indices)

  # Some of the boxes are padding. We want to discount them from the loss.
  if target_is_onehot:
    tgt_not_padding = 1 - tgt_labels[..., 0]
  else:
    tgt_not_padding = tgt_labels != 0
  # Align this with the permuted target indices.
  tgt_not_padding = model_utils.simple_gather(tgt_not_padding, tgt_indices)
  tgt_not_padding = jnp.asarray(tgt_not_padding, dtype=jnp.float32)

  # To match official repo we do L1 loss on the cxcywh format.
  loss_bbox = model_utils.weighted_box_l1_loss(src_boxes, tgt_boxes)
  loss_bbox *= tgt_not_padding[..., None]
  loss_bbox = bbox_loss_coef * loss_bbox.sum()

  loss_giou = 1 - box_utils.generalized_box_iou(
      box_utils.box_cxcywh_to_xyxy(src_boxes),
      box_utils.box_cxcywh_to_xyxy(tgt_boxes),
      all_pairs=False,
      eps=1e-8)
  loss_giou *= tgt_not_padding
  loss_giou = giou_loss_coef * loss_giou.sum()

  losses = {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}
  return losses


def _targets_from_batch(
    batch: ArrayDict,
    target_is_onehot: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Get target labels and boxes with additional non-object appended."""
  # Append the no-object class label so we are always guaranteed one.
  tgt_labels = batch['label']['labels']
  tgt_boxes = batch['label']['boxes']

  # Append a class label.
  if target_is_onehot:
    # Shape is [batch, num_instances, num_classes]
    label_shape = tgt_labels.shape
    num_classes = label_shape[-1]
    instance = jax.nn.one_hot(0, num_classes)
    reshape_shape = (1,) * (len(label_shape) - 1) + (num_classes,)
    broadcast_shape = label_shape[:-2] + (1, num_classes)
    instance = jnp.broadcast_to(
        jnp.reshape(instance, reshape_shape), broadcast_shape)
  else:
    instance = jnp.zeros_like(tgt_labels[..., :1])
  tgt_labels = jnp.concatenate([tgt_labels, instance], axis=1)

  # Same for boxes.
  instance = jnp.zeros_like(tgt_boxes[..., :1, :])
  tgt_boxes = jnp.concatenate([tgt_boxes, instance], axis=1)
  return tgt_labels, tgt_boxes


class InputProj(nn.Module):
  """Simple input projection layer.

  Attributes:
    embed_dim: Size of the output embedding dimension.
    num_groups: Number channel groups for group norm.
    kernel_size: Convolution kernel size.
    stride: Stride in all dimensions.
    padding: Same as nn.Conv padding type.
  """
  embed_dim: int
  num_groups: int = 32
  kernel_size: int = 1
  stride: int = 1
  padding: Union[str, int] = 'SAME'

  @nn.compact
  def __call__(self, x: jnp.ndarray):
    """Use conv kernel to project into embed_dim."""
    if isinstance(self.padding, str):
      padding = self.padding
    else:
      padding = [self.padding] * 2
    x = nn.Conv(
        features=self.embed_dim,
        kernel_size=[self.kernel_size] * 2,
        strides=self.stride,
        padding=padding,
        kernel_init=nn.initializers.glorot_uniform(),
        bias_init=nn.initializers.zeros,
    )(
        x)
    x = nn.GroupNorm(num_groups=self.num_groups)(x)
    return x


class ObjectClassPredictor(nn.Module):
  """Linear Projection block for predicting classification."""
  num_classes: int
  prior_prob: float = 0.01
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies Linear Projection to inputs.

    Args:
      inputs: Input data.

    Returns:
      Output of Linear Projection block.
    """
    bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
    return nn.Dense(
        self.num_classes,
        kernel_init=pytorch_kernel_init(dtype=self.dtype),
        bias_init=nn.initializers.constant(bias_init_value),
        dtype=self.dtype)(
            inputs)


class DeformableDETR(nn.Module):
  """DeformableDETR.

  Attributes:
    num_classes: Number of classes to predict.
    embed_dim: Size of the hidden embedding dimension.
    embed_dim: Size of the hidden embedding dimension for encoder.
    num_heads: Number of heads.
    num_queries: Number of object queries.
    num_enc_layers: Number of encoder layers.
    num_dec_layers: Number of decoder layers.
    num_feature_levels: Number of feature levels/scales.
    num_enc_points: Number of encoder points in deformable attention.
    num_dec_points: Number of decoder points in deformable attention.
    transformer_ffn_dim: Transformers feed-forward/MLP dimension.
    backbone_num_filters: Number of filters for Resnet.
    backbone_num_layers: Number of layers for Resnet.
    dropout: Dropout rate.
    compiler_config: Compiler configuration.
    dtype: Data type of the computation (default: float32).
  """
  num_classes: int
  embed_dim: int
  enc_embed_dim: int
  num_heads: int
  num_queries: int
  num_enc_layers: int
  num_dec_layers: int
  num_feature_levels: int
  num_enc_points: int
  num_dec_points: int
  transformer_ffn_dim: int
  backbone_num_filters: int
  backbone_num_layers: int
  dropout: float
  compiler_config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               train: bool = False,
               *,
               padding_mask: Optional[jnp.ndarray] = None,
               update_batch_stats: bool = False,
               debug: bool = False) -> Dict[str, jnp.ndarray]:
    """Perform multi-scale DeformableDETR.

    Args:
      inputs: [bs, h, w, c] input data.
      train: Whether it is training.
      padding_mask: [bs, h, w] of bools with 0 at padded image regions.
      update_batch_stats: Whether update the batch statistics for the BatchNorms
        in the backbone. if None, the value of `train` flag will be used, i.e.
        we update the batch stat if we are in the train mode.
      debug: Necessary for scenic api.

    Returns:
      Output:
        'pred_logits' - [bs, num_queries, num_classes] logits.
        'pred_boxes' - [bs, num_queries, 4] boxes in cxcywh format.
    """
    del debug
    if padding_mask is not None:
      padding_mask = padding_mask.astype(bool)

    features, pad_masks, pos_embeds = DeformableDETRBackbone(
        embed_dim=self.enc_embed_dim,
        num_layers=self.backbone_num_layers,
        num_filters=self.backbone_num_filters,
        num_feature_levels=min(self.num_feature_levels, 3),
        dtype=self.dtype,
        name='backbone')(
            inputs, train=update_batch_stats, padding_mask=padding_mask)

    projs = [
        InputProj(embed_dim=self.enc_embed_dim, name=f'input_proj{idx}')(x)
        for idx, x in enumerate(features)
    ]

    # Add any additional feature scales beyond the last features from backbone.
    x = features[-1]
    for i in range(len(features), self.num_feature_levels):
      x = InputProj(
          embed_dim=self.enc_embed_dim,
          kernel_size=3,
          stride=2,
          padding=1,
          name=f'input_proj{i}')(
              x)
      projs.append(x)
      pad_masks.append(mask_for_shape(x.shape, pad_mask=padding_mask))
      pos_embeds.append(
          InputPosEmbeddingSine(hidden_dim=self.enc_embed_dim)(pad_masks[-1]))

    # Create shared bbox predictors.
    bbox_embeds = []
    for layer_idx in range(self.num_dec_layers):
      bbox = BBoxCoordPredictor(
          mlp_dim=self.embed_dim,
          num_layers=3,
          use_sigmoid=False,
          name=f'bbox_embed{layer_idx}')
      bbox_embeds.append(bbox)

    x, ref_points, dec_init_ref_points = DeformableDETRTransformer(
        enc_embed_dim=self.enc_embed_dim,
        embed_dim=self.embed_dim,
        num_heads=self.num_heads,
        num_queries=self.num_queries,
        num_enc_layers=self.num_enc_layers,
        num_dec_layers=self.num_dec_layers,
        ffn_dim=self.transformer_ffn_dim,
        num_enc_points=self.num_enc_points,
        num_dec_points=self.num_dec_points,
        bbox_embeds=bbox_embeds,
        name='transformer',
        dropout=self.dropout,
        compiler_config=self.compiler_config,
        dtype=self.dtype)(
            inputs=projs,
            pad_masks=pad_masks,
            pos_embeds=pos_embeds,
            train=train)

    assert len(x) == self.num_dec_layers

    # Classes and Box coordinates prediction heads.
    pred_logits_by_layer = []
    pred_boxes_by_layer = []
    for layer_idx in range(self.num_dec_layers):
      # Logits.
      logits = ObjectClassPredictor(
          num_classes=self.num_classes, name=f'class_embed{layer_idx}')(
              x[layer_idx])
      pred_logits_by_layer.append(logits)

      # Predict box coordinates using intermediate decoder outputs.
      if layer_idx == 0:
        level_ref = dec_init_ref_points
      else:
        level_ref = ref_points[layer_idx - 1]
      level_ref = inverse_sigmoid(level_ref)

      bbox = bbox_embeds[layer_idx](x[layer_idx])
      if level_ref.shape[-1] == 4:
        bbox += level_ref
      else:
        xy = bbox[..., :2] + level_ref
        bbox = jnp.concatenate([xy, bbox[..., 2:]], -1)
      bbox = nn.sigmoid(bbox)
      pred_boxes_by_layer.append(bbox)

    *prev_layers_out, out = jax.tree_util.tree_map(
        lambda logits, boxes: dict(pred_logits=logits, pred_boxes=boxes),
        pred_logits_by_layer, pred_boxes_by_layer)
    out['aux_outputs'] = prev_layers_out
    return out


class DeformableDETRModel(base_model.BaseModel):
  """DeformableDETR model for object detection task."""

  def __init__(self, config: Optional[ml_collections.ConfigDict],
               dataset_meta_data: Dict[str, Any]):
    """Initialize DeformableDETR Detection model.

    Args:
      config: Configurations of the model.
      dataset_meta_data: Dataset meta data specifies `target_is_onehot`, which
        is False by default. The padded objects have label 0. The first
        legitimate object has label 1, and so on.
    """
    if config is not None:
      self.loss_terms_weights = {
          'loss_class': config.class_loss_coef,
          'loss_bbox': config.bbox_loss_coef,
          'loss_giou': config.giou_loss_coef
      }
    super().__init__(config, dataset_meta_data)

  def build_flax_model(self):
    return DeformableDETR(
        num_classes=self.config.num_classes,
        embed_dim=self.config.embed_dim,
        enc_embed_dim=self.config.enc_embed_dim,
        num_queries=self.config.num_queries,
        num_heads=self.config.num_heads,
        num_enc_layers=self.config.num_encoder_layers,
        num_dec_layers=self.config.num_decoder_layers,
        num_feature_levels=self.config.num_feature_levels,
        num_enc_points=self.config.num_enc_points,
        num_dec_points=self.config.num_dec_points,
        backbone_num_filters=self.config.backbone_num_filters,
        backbone_num_layers=self.config.backbone_num_layers,
        transformer_ffn_dim=self.config.transformer_ffn_dim,
        dropout=self.config.dropout_rate,
        compiler_config=self.config.compiler_config,
        dtype=jnp.float32)

  def compute_loss_for_layer(
      self,
      tgt_labels: jnp.ndarray,
      pred_logits: jnp.ndarray,
      tgt_boxes: jnp.ndarray,
      pred_boxes: jnp.ndarray,
      indices: Optional[jnp.ndarray] = None) -> ArrayDict:
    """Loss and metrics function for single prediction layer."""
    target_is_onehot = self.dataset_meta_data.get('target_is_onehot', False)
    if indices is None:
      pred_prob = self.logits_to_probs(pred_logits)
      cost, n_cols = compute_cost(
          tgt_labels=tgt_labels,
          out_prob=pred_prob,
          tgt_bbox=tgt_boxes,
          out_bbox=pred_boxes,
          alpha=self.config.focal_loss_alpha,
          gamma=self.config.focal_loss_gamma,
          class_loss_coef=self.config.class_loss_coef,
          bbox_loss_coef=self.config.bbox_loss_coef,
          giou_loss_coef=self.config.giou_loss_coef,
          target_is_onehot=target_is_onehot)
      indices = matchers.hungarian_matcher(cost, n_cols=n_cols)

    losses = {}
    # Class loss.
    losses.update(
        loss_labels(
            pred_logits=pred_logits,
            tgt_labels=tgt_labels,
            indices=indices,
            alpha=self.config.focal_loss_alpha,
            gamma=self.config.focal_loss_gamma,
            class_loss_coef=self.config.class_loss_coef,
            target_is_onehot=target_is_onehot))
    # Boxes loss.
    losses.update(
        loss_boxes(
            src_boxes=pred_boxes,
            tgt_labels=tgt_labels,
            tgt_boxes=tgt_boxes,
            indices=indices,
            bbox_loss_coef=self.config.bbox_loss_coef,
            giou_loss_coef=self.config.giou_loss_coef,
            target_is_onehot=target_is_onehot))
    return losses

  def logits_to_probs(self,
                      logits: jnp.ndarray,
                      log_p: bool = False) -> jnp.ndarray:
    is_sigmoid = self.config.get('sigmoid_loss', True)
    # We can overwrite logit normalization explicitly if we wanted to, so we
    # can normalize logits using softmax but using sigmoid loss.
    is_sigmoid = self.config.get('sigmoid_logit_norm', is_sigmoid)
    if not is_sigmoid:
      return jax.nn.log_softmax(logits) if log_p else jax.nn.softmax(logits)
    else:
      return jax.nn.log_sigmoid(logits) if log_p else jax.nn.sigmoid(logits)

  def loss_and_metrics_function(
      self,
      outputs: ArrayDict,
      batch: ArrayDict,
      matches: Optional[Sequence[jnp.ndarray]] = None,
      model_params: Optional[jnp.ndarray] = None
  ) -> Tuple[jnp.ndarray, MetricsDict]:
    """Loss and metrics function for DeformableDETR.

    Args:
      outputs: Model prediction. The exact fields depend on the losses used.
        Please see labels_losses_and_metrics and boxes_losses_and_metrics for
        details.
      batch: Dict that has 'inputs', 'batch_mask' and, 'label' (ground truth).
        batch['label'] is a dict where the keys and values depend on the losses
        used. Please see labels_losses_and_metrics and boxes_losses_and_metrics
        member methods.
      matches: Possibly pass in matches if already done.
      model_params: Pass in model params if we are doing L2 regularization.

    Returns:
      total_loss: Total loss weighted appropriately.
      metrics_dict: Individual loss terms for logging purposes.
    """
    tgt_labels, tgt_boxes = _targets_from_batch(
        batch, self.config.get('target_is_onehot', False))
    if matches is None:
      indices, aux_indices = None, None
    else:
      indices, *aux_indices = matches
    losses = self.compute_loss_for_layer(
        pred_logits=outputs['pred_logits'],
        tgt_labels=tgt_labels,
        pred_boxes=outputs['pred_boxes'],
        tgt_boxes=tgt_boxes,
        indices=indices)

    if 'aux_outputs' in outputs:
      for i, aux_outputs in enumerate(outputs['aux_outputs']):
        aux_losses = self.compute_loss_for_layer(
            pred_logits=aux_outputs['pred_logits'],
            tgt_labels=tgt_labels,
            pred_boxes=aux_outputs['pred_boxes'],
            tgt_boxes=tgt_boxes,
            indices=aux_indices[i] if aux_indices is not None else None)
        aux_losses = {f'{k}_aux{i}': v for k, v in aux_losses.items()}
        losses.update(aux_losses)

    ntargets = jnp.sum(tgt_labels > 0, axis=1)
    norm_type = self.config.get('normalization', 'detr')
    logging.info('Normalization type: %s', norm_type)
    if norm_type == 'detr':
      ntargets = jnp.maximum(ntargets.sum(), 1.)
    elif norm_type == 'global':
      ntargets = jax.lax.pmean(ntargets.sum(), axis_name='batch')
      ntargets = jnp.maximum(ntargets, 1.)
    else:
      raise ValueError(f'Unknown normalization {norm_type}.')

    # Normalize losses by num_boxes.
    losses = jax.tree_util.tree_map(lambda x: x / ntargets, losses)

    if self.config.get('l2_decay_factor', 0) > 0:
      l2_loss = model_utils.l2_regularization(model_params)
      losses['l2_loss'] = 0.5 * self.config.l2_decay_factor * l2_loss

    # Sum total loss.
    losses['total_loss'] = jax.tree_util.tree_reduce(jnp.add, losses, 0)  # pytype: disable=wrong-arg-types  # numpy-scalars

    # Store metrics for logging.
    metrics = {k: (v, 1.) for k, v in losses.items()}
    for k, v in metrics.items():
      metrics[k] = model_utils.psum_metric_normalizer(v)  # pytype: disable=wrong-arg-types  # jax-ndarray

    return losses['total_loss'], metrics  # pytype: disable=bad-return-type  # jax-ndarray
