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

"""Losses."""
from typing import Optional, Union

import jax
import jax.numpy as jnp
from scenic.model_lib.base_models import box_utils

EPS = 1e-6


def sigmoid_cost(
    logit: Union[jnp.ndarray, float],
    *,
    focal_loss: bool = False,
    focal_alpha: Optional[float] = None,
    focal_gamma: Optional[float] = None
) -> Union[jnp.ndarray, float]:
  """Computes the classification cost.

  Relevant code:
  https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/matcher.py#L76

  Args:
    logit: Sigmoid classification logit(s).
    focal_loss: Whether to apply focal loss for classification cost.
    focal_alpha: Alpha scaling factor for focal loss.
    focal_gamma: Gamma scaling factor for focal loss.

  Returns:
    Classification cost.
  """
  neg_cost_class = -jax.nn.log_sigmoid(-logit)
  pos_cost_class = -jax.nn.log_sigmoid(logit)
  if focal_loss:
    neg_cost_class *= (1 - focal_alpha) * jax.nn.sigmoid(logit)**focal_gamma
    pos_cost_class *= focal_alpha * jax.nn.sigmoid(-logit)**focal_gamma
  return pos_cost_class - neg_cost_class  # [B, N, C]


def compute_cost(
    *,
    tgt_labels: jnp.ndarray,
    out_logits: jnp.ndarray,
    tgt_bbox: jnp.ndarray,
    out_bbox: jnp.ndarray,
    class_loss_coef: float,
    bbox_loss_coef: jnp.ndarray,
    giou_loss_coef: jnp.ndarray,
    focal_loss: bool = False,
    focal_alpha: Optional[float] = None,
    focal_gamma: Optional[float] = None,
) -> jnp.ndarray:
  """Computes cost matrices for a batch of predictions.

  Relevant code:
  https://github.com/facebookresearch/detr/blob/647917626d5017e63c1217b99537deb2dcb370d6/models/matcher.py#L35

  Args:
    tgt_labels: Class labels of shape [B, M, C] (one/multi-hot). Note that the
      labels corresponding to empty bounding boxes are not yet supposed to be
      filtered out.
    out_logits: Classification sigmoid logits of shape [B, N, C].
    tgt_bbox: Target box coordinates of shape [B, M, 4]. Note that the empty
      bounding boxes are not yet supposed to be filtered out.
    out_bbox: Predicted box coordinates of shape [B, N, 4]
    class_loss_coef: Relative weight of classification loss.
    bbox_loss_coef: Relative weight of bbox loss.
    giou_loss_coef: Relative weight of giou loss.
    focal_loss: Whether to apply focal loss for classification cost.
    focal_alpha: Alpha scaling factor for focal loss.
    focal_gamma: Gamma scaling factor for focal loss.

  Returns:
    A cost matrix [B, N, M].
    Number of unpadded columns per batch element [B].
  """
  if focal_loss and (focal_alpha is None or focal_gamma is None):
    raise ValueError('For focal loss, focal_alpha and focal_gamma must be set.')

  # Number of non-padding labels for each of the target instances.
  n_labels_per_instance = jnp.sum(tgt_labels[..., 1:], axis=-1)
  mask = n_labels_per_instance > 0  # [B, M]

  # Make sure padding target is 0 for instances with other labels.
  tgt_labels = jnp.concatenate(
      [jnp.expand_dims(~mask, -1), tgt_labels[..., 1:]], axis=-1)

  cost_class = sigmoid_cost(  # [B, N, C]
      out_logits,
      focal_loss=focal_loss,
      focal_alpha=focal_alpha,
      focal_gamma=focal_gamma)

  # Resulting shape is [B, N, M].
  # Note that we do *not* normalize by the number of per-target instances.
  cost_class = jnp.einsum('bnl,bml->bnm', cost_class, tgt_labels)

  cost = class_loss_coef * cost_class

  diff = jnp.abs(out_bbox[:, :, None] - tgt_bbox[:, None, :])  # [B, N, M, 4]
  cost_bbox = jnp.sum(diff, axis=-1)  # [B, N, M]
  cost = cost + bbox_loss_coef * cost_bbox

  cost_giou = -box_utils.generalized_box_iou(
      box_utils.box_cxcywh_to_xyxy(out_bbox),
      box_utils.box_cxcywh_to_xyxy(tgt_bbox),
      all_pairs=True)
  cost = cost + giou_loss_coef * cost_giou

  mask = mask[:, None]

  # Determine mask value dynamically.
  cost_mask_value = jnp.max(jnp.where(mask, cost, -1e10), axis=(1, 2))
  # Special case.
  all_masked = jnp.all(~mask, axis=(1, 2))
  cost_mask_value = jnp.where(~all_masked, cost_mask_value, 1.0)
  cost_mask_value = cost_mask_value[:, None, None] * 1.1 + 10.0

  cost = cost * mask + (1.0 - mask) * cost_mask_value
  # Guard against NaNs and Infs.
  cost = jnp.nan_to_num(
      cost,
      nan=cost_mask_value,
      posinf=cost_mask_value,
      neginf=cost_mask_value)

  # Compute the number of unpadded columns for each batch element. It is assumed
  # that all padding is trailing padding.
  max_num_boxes = tgt_labels.shape[1]
  n_cols = jnp.where(
      jnp.max(mask, axis=1),
      jnp.expand_dims(jnp.arange(1, max_num_boxes + 1), axis=0), 0)
  n_cols = jnp.max(n_cols, axis=1)
  return cost, n_cols  # pytype: disable=bad-return-type  # jax-ndarray
