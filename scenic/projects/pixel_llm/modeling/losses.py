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

"""Loss functions for PixelLLM."""

import jax
import jax.numpy as jnp
import optax
from scenic.model_lib.base_models import model_utils


def text_loss(
    text_outputs,
    gt_text,
    mask=None,
    label_smooth=0.1,
    end_token_id: int = 102,  # tokenizer.sep_token_id == 102
    vocab_size: int = 30522,  # size of BertTokenizer
):
  """Text loss with label smoothing.

  Args:
    text_outputs: (text_batch_size, max_caption_length, vocab_size)
    gt_text: (text_batch_size, max_caption_length)
    mask: (text_batch_size,)
    label_smooth: float
    end_token_id: int
    vocab_size: int

  Returns:
    loss: float
  """
  if text_outputs.ndim == 4:
    text_outputs = text_outputs.reshape(
        (text_outputs.shape[0] * text_outputs.shape[1],)
        + text_outputs.shape[2:]
    )
  gt_text = gt_text.reshape(
      gt_text.shape[0] * gt_text.shape[1],
      gt_text.shape[2],
  )  # (batch_size * num_caps_per_image, max_cap_len)
  if mask is None:
    mask = jnp.ones((gt_text.shape[0],))
  else:
    mask = mask.reshape((gt_text.shape[0],))
  text_outputs = text_outputs[:, :-1]  # Move gt 1 word to the right.
  gt_text = gt_text[:, 1:]  # No need to predict BOS
  # valid: (text_batch_size, max_caption_length - 1)
  valid = ((gt_text > 0) & (mask[:, None] > 0))
  # Ignore samples with empty ground truth (from padding).
  valid = valid & (gt_text[:, 0] != end_token_id)[:, None]
  valid = valid.astype(jnp.float32)
  # gt: (text_batch_size, max_caption_length - 1, vocab_size)
  gt = jax.nn.one_hot(gt_text, vocab_size)
  # customized label smoothing following GRiT
  #   https://github.com/JialianW/GRiT/blob/master/grit/modeling/text/
  #   text_decoder.py#L668
  gt = gt * (1. - label_smooth) + (1. - gt) * label_smooth / (vocab_size - 1)
  # loss:  (text_batch_size, max_caption_length - 1)
  gt = jax.lax.stop_gradient(gt)
  loss = optax.softmax_cross_entropy(text_outputs, gt)
  loss = (loss * valid[:, :]).sum() / (valid.sum() + 1e-8)
  return {'text_loss': loss}


def point_loss(
    pred_points,
    pred_valid_mask,
    gt_points,
    gt_valid_mask,
    loss_type='l1',
):
  """L1 point loss.

  Args:
    pred_points: (batch_size, num_caps_per_image, max_cap_len,
      num_points, 2)
    pred_valid_mask: (batch_size, num_caps_per_image, max_cap_len)
    gt_points: (batch_size, num_caps_per_image, max_cap_len, num_points, 2)
      or (batch_size, num_caps_per_image, num_points, 2)
    gt_valid_mask: (batch_size, num_caps_per_image, max_cap_len)
      or (batch_size, num_caps_per_image)
    loss_type: l1 or l2

  Returns:
    loss: dict[str, float]
  """

  batch_size, num_caps_per_image, max_cap_len = pred_points.shape[:3]
  num_points = pred_points.shape[-2]

  if gt_points.ndim == 4:
    # [batch_size, num_caps_per_image, max_cap_len, num_points, 2]
    gt_points = jnp.broadcast_to(
        jnp.expand_dims(gt_points, axis=2), pred_points.shape
    )

  if gt_valid_mask.ndim == 2:
    # [batch_size, num_caps_per_image, max_cap_len]
    gt_valid_mask = jnp.broadcast_to(
        jnp.expand_dims(gt_valid_mask, axis=2), pred_points.shape[:-2]
    )

  assert pred_points.shape == gt_points.shape
  assert pred_valid_mask.shape == gt_valid_mask.shape

  pred_valid_mask = pred_valid_mask.astype(jnp.int32)
  gt_valid_mask = gt_valid_mask.astype(jnp.int32)

  total_batch = batch_size * num_caps_per_image * max_cap_len

  pred_points = jnp.reshape(pred_points, (total_batch, num_points, 2))
  gt_points = jnp.reshape(gt_points, (total_batch, num_points, 2))

  pred_valid_mask = jnp.reshape(pred_valid_mask, (total_batch,))
  gt_valid_mask = jnp.reshape(gt_valid_mask, (total_batch,))

  denom = jnp.maximum(jnp.sum(gt_valid_mask * pred_valid_mask), 1.0)

  # [total_batch, ]
  if loss_type == 'l1':
    loss_point = jnp.abs(pred_points - gt_points).mean(axis=(-2, -1))
  elif loss_type == 'l2':
    loss_point = jnp.square(pred_points - gt_points).mean(axis=(-2, -1))
  elif loss_type == 'l1_nonzero':
    loss_point = jnp.abs(pred_points - gt_points).mean(axis=(-2, -1))
    gt_valid_mask = gt_valid_mask * (
        (gt_points ** 2).sum(axis=(-2, -1)) > 0).astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(gt_valid_mask * pred_valid_mask), 1.0)
  else:
    raise ValueError(f'Unknown loss type: {loss_type}')
  loss_point = loss_point * gt_valid_mask * pred_valid_mask
  loss_point = loss_point.sum() / denom

  metrics = {'point_loss': loss_point}

  return metrics


def sam_mask_loss(
    pred_masks,
    pred_ious,
    gt_masks,
    gt_valid,
    padding_mask,
    focal_loss_weight: float = 20.0,
    dice_loss_weight: float = 1.0,
    iou_pred_loss_weight: float = 1.0,
):
  """Mask loss following SAM.

  Args:
    pred_masks: (batch_size, num_masks_per_image, num_masks_per_prompt, H, W)
    pred_ious: (batch_size, num_masks_per_image, num_masks_per_prompt)
    gt_masks: (batch_size, num_masks_per_image, H, W, 1)
    gt_valid: (batch_size, num_caps_per_image)
    padding_mask: (batch_size, H, W)
    focal_loss_weight:
    dice_loss_weight:
    iou_pred_loss_weight:

  Returns:
    loss: dict[str, float]
  """
  gt_masks = (gt_masks > 0).astype(jnp.float32)

  metrics = {}
  num_masks_per_image, num_masks_per_prompt = pred_masks.shape[1:3]
  assert pred_ious.shape == pred_masks.shape[:3]

  batch_size = pred_masks.shape[0]
  num_masks = num_masks_per_image * num_masks_per_prompt

  # [batch_size, num_masks_per_image, 1, 1, 1]
  mask_valid = jnp.reshape(
      gt_valid, (gt_valid.shape[0], gt_valid.shape[1], 1, 1, 1)
  )
  # [batch_size, num_masks_per_image, num_masks_per_prompt, 1, 1]
  mask_valid = jnp.tile(mask_valid, (1, 1, num_masks_per_prompt, 1, 1))
  # [batch_size, num_masks, 1, 1]
  mask_valid = jnp.reshape(mask_valid, (batch_size, num_masks, 1, 1))
  # [batch_size, num_masks, H, W]
  mask_valid = padding_mask[:, None] * mask_valid

  # resize all masks into the same shape
  height = padding_mask.shape[1]
  width = padding_mask.shape[2]
  # [batch_size, num_masks, H, W]
  pred_masks = jnp.reshape(
      pred_masks, (batch_size, num_masks) + pred_masks.shape[3:]
  )
  assert gt_masks.shape[2:] == (height, width, 1), gt_masks.shape
  # [batch_size, num_masks, H, W]
  pred_masks = jax.image.resize(
      pred_masks, pred_masks.shape[:2] + (height, width), method='bilinear'
  )
  # [batch_size, num_masks, H, W, 1]
  pred_masks = jnp.expand_dims(pred_masks, axis=-1)

  # [batch_size, num_masks_per_image, 1, H, W, 1]
  gt_masks = jnp.expand_dims(gt_masks, axis=2)
  # [batch_size, num_masks_per_image, num_masks_per_prompt, H, W, 1]
  gt_masks = jnp.tile(gt_masks, (1, 1, num_masks_per_prompt, 1, 1, 1))
  # [batch_size, num_masks, H, W, 1]
  gt_masks = jnp.reshape(gt_masks, (batch_size, num_masks, height, width, 1))

  # [batch_size, num_masks, H, W, 1]
  focal_loss = model_utils.focal_sigmoid_cross_entropy(
      pred_masks, gt_masks, weights=mask_valid
  )
  # [batch_size, num_masks]
  focal_loss = focal_loss.sum(axis=(2, 3, 4)) / jnp.maximum(
      mask_valid.sum(axis=(2, 3)), 1.0
  )
  # [batch_size, num_masks_perimage, num_masks_per_prompt]
  focal_loss = jnp.reshape(
      focal_loss, (batch_size, num_masks_per_image, num_masks_per_prompt)
  )
  # [batch_size, num_masks]
  dice_loss = model_utils.dice_loss(
      jnp.squeeze(pred_masks, axis=-1),
      jnp.squeeze(gt_masks, axis=-1),
      weights=jnp.max(mask_valid, axis=(-2, -1)) > 0,
      all_pairs=False,
  )
  # [batch_size, num_masks_perimage, num_masks_per_prompt]
  dice_loss = jnp.reshape(
      dice_loss, (batch_size, num_masks_per_image, num_masks_per_prompt)
  )

  # [batch_size, num_masks_perimage, num_masks_per_prompt]
  mask_loss = focal_loss * focal_loss_weight + dice_loss * dice_loss_weight
  # [batch_size, num_masks_perimage]
  min_loss_ind = jnp.argmin(mask_loss, axis=-1)
  # [batch_size, num_masks_perimage]
  mask_loss = jnp.take_along_axis(mask_loss, min_loss_ind[..., None], axis=-1)[
      ..., 0
  ]

  mask_loss = (mask_loss * gt_valid).sum() / jnp.maximum(gt_valid.sum(), 1.0)

  focal_loss = jnp.take_along_axis(
      focal_loss, min_loss_ind[..., None], axis=-1
  )[..., 0]
  focal_loss = (focal_loss * gt_valid).sum() / jnp.maximum(gt_valid.sum(), 1.0)
  metrics['mask_focal_loss'] = focal_loss * focal_loss_weight
  dice_loss = jnp.take_along_axis(dice_loss, min_loss_ind[..., None], axis=-1)[
      ..., 0
  ]
  dice_loss = (dice_loss * gt_valid).sum() / jnp.maximum(gt_valid.sum(), 1.0)
  metrics['mask_dice_loss'] = dice_loss * dice_loss_weight

  # [batch_size, num_masks]
  gt_inter = ((pred_masks > 0) * (gt_masks > 0)).sum(axis=(-3, -2, -1))
  gt_union = (((pred_masks > 0) + (gt_masks > 0)) > 0).sum(axis=(-3, -2, -1))
  gt_ious = gt_inter / jnp.maximum(gt_union, 1.0)
  # [batch_size, num_masks_per_image, num_masks_per_prompt]
  gt_ious = jnp.reshape(
      gt_ious, (batch_size, num_masks_per_image, num_masks_per_prompt)
  )

  iou_pred_loss = (gt_ious - pred_ious) ** 2
  iou_pred_loss = (iou_pred_loss * gt_valid[..., None]).sum() / jnp.maximum(
      jnp.broadcast_to(gt_valid[..., None], gt_ious.shape).sum(), 1.0
  )

  metrics['mask_iou_pred_loss'] = iou_pred_loss * iou_pred_loss_weight

  metrics['mask_loss'] = mask_loss + iou_pred_loss

  return metrics
