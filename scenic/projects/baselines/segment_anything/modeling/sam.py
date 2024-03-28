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

r"""Segment Anything Model.

Pytorch reference:

https://github.com/facebookresearch/segment-anything/blob/HEAD/\
segment_anything/modeling/sam.py

"""
import dataclasses
from typing import Any, Optional

from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.projects.baselines.segment_anything.modeling import image_encoder
from scenic.projects.baselines.segment_anything.modeling import mask_decoder
from scenic.projects.baselines.segment_anything.modeling import prompt_encoder
from scenic.projects.baselines.segment_anything.modeling import utils

PIXEL_MEAN = (123.675, 116.28, 103.53)
PIXEL_STD = (58.395, 57.12, 57.375)

SIZE_CONFIGS = {
    'B': (768, 12, 12, 0.1, (0, 1, 3, 4, 6, 7, 9, 10)),
    'L': (1024, 24, 16, 0.4, (
        0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22)),
    'H': (1280, 32, 16, 0.5, (
        0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21,
        22, 24, 25, 26, 27, 28, 29, 30)),
}


class Sam(nn.Module):
  """Segment anything model.

  Default parameters following
  https://github.com/facebookresearch/segment-anything/blob/main/
  segment_anything/automatic_mask_generator.py#L35

  Attributes:
    mask_threshold: threshold to convert output logits to binary masks.
    pixel_mean: used in preprocessing inputs.
    pixel_std: used in preprocessing inputs.
    max_objects: number of output objects in "segment anything" mode.
    points_per_side: number of point anchors perside in "segment anything" mode.
    points_per_batch: batch size for processing point anchors.
    pred_iou_thresh: score threshold in "segment anything" mode.
    box_nms_thresh: NMS threshold
    stability_score_thresh: threshold for filtering with a stability metric.
    stability_score_offset: used in computing the stability metric.
    pre_nms_topk: new hyper-parameter in this implementation. Used for keeping a
      fixed shape after filtering mask predictions.
    image_encoder_args: args for image backbone.
    prompt_encoder_args: args for prompt encoder.
    mask_decoder_args: args for mask decoder.
  """
  mask_threshold: float = 0.0
  pixel_mean: Any = PIXEL_MEAN
  pixel_std: Any = PIXEL_STD
  max_objects: int = 100
  points_per_side: Optional[int] = 32
  points_per_batch: int = 64
  pred_iou_thresh: float = 0.88
  box_nms_thresh: float = 0.7
  stability_score_thresh: float = 0.95
  stability_score_offset: float = 1.0
  pre_nms_topk: int = 1536
  image_encoder_args: ml_collections.ConfigDict = dataclasses.field(
      default_factory=ml_collections.ConfigDict)
  prompt_encoder_args: ml_collections.ConfigDict = dataclasses.field(
      default_factory=ml_collections.ConfigDict)
  mask_decoder_args: ml_collections.ConfigDict = dataclasses.field(
      default_factory=ml_collections.ConfigDict)

  def setup(self):
    # pylint: disable=not-a-mapping
    self.image_encoder = image_encoder.ImageEncoderViT(
        **self.image_encoder_args, name='image_encoder')
    self.prompt_encoder = prompt_encoder.PromptEncoder(
        **self.prompt_encoder_args, name='prompt_encoder')
    self.mask_decoder = mask_decoder.MaskDecoder(
        **self.mask_decoder_args, name='mask_decoder')
    # pylint: enable=not-a-mapping

  @nn.compact
  def __call__(
      self, image, point_coords, point_labels, padding_mask=None,
      image_embeddings=None, boxes=None, mask_inputs=None,
      multimask_output: bool = True, return_image_embedding: bool = False,
      upsample_mask: bool = True, return_batch_as_list: bool = True,
      train: bool = False, debug: bool = False):
    """Forward Sam model.

    Args:
      image: (batch_size, H, W, 3). Input pixels in RGB values [0, 255].
      point_coords: (batch_size, num_prompts, num_points, 2). Input point
        prompts. In absolute range [0, image.shape[1 or 2]].
      point_labels: (batch_size, num_prompts, num_points). 1: positive points;
        0: negative points. -1: padded/ ignored points.
      padding_mask: (batch_size, H, W). Indicate which pixels in the input are
        padded. 1: not padded; 0: padded. This is used to match the pytorch
        preprocessing process: normalize then pad, while in Jax we need to pad
        first.
      image_embeddings: cached image embeddings if they are provided.
        (batch_size, H', W', D). If not provided, image must be not None.
      boxes: (batch_size, num_prompts, 4); box prompts;
      mask_inputs: (batch_size, num_prompts, 1, H, W); mask prompts.
      multimask_output: bool. If false, C = 1, otherwise,
        C = self.mask_decoder_args.num_multimask_outputs
      return_image_embedding: bool
      upsample_mask: bool; If False, only return the 4x downsampled masks. This
        saves memory.
      return_batch_as_list: If True, return a list where each item is the
        results of a single image; If False, return a dict with batched results.
      train: bool
      debug: bool
    Returns:
      ret: a list (batch) of dicts, each with the following keys:
        'masks': (num_prompts, C, H, W). C is the num of masks (see above).
        'iou_predictions': (num_prompts, C). Predicted mask quality scores.
        'low_res_logits': (num_prompts, C, H', W'). The output mask of the
          mask decoder. The final masks are resized from this.
    """
    del debug
    msg = 'One of "image" or "image_embedding" should be provided!'
    assert image is not None or image_embeddings is not None, msg
    assert image is None or image_embeddings is None, msg
    if image_embeddings is None:
      assert image is not None
      image_embeddings = self.get_image_embeddings(
          image, padding_mask=padding_mask,
          train=train)  # (batch_size, H', W', D)
    image_size = image.shape[1:3] if image is not None else (
        (image_embeddings.shape[1] * 16, image_embeddings.shape[2] * 16))
    ret = []
    for b, curr_embedding in enumerate(image_embeddings):
      curr_point_coords = point_coords[b] if point_coords is not None else None
      curr_point_labels = point_labels[b] if point_labels is not None else None
      box_prompt = boxes[b] if boxes is not None else None
      mask_prompt = mask_inputs[b] if mask_inputs is not None else None
      sparse_embeddings, dense_embeddings = self.prompt_encoder(
          curr_point_coords, curr_point_labels,
          boxes=box_prompt, masks=mask_prompt,
          image_size=image_size,
          image_embedding_size=curr_embedding.shape[:2])
      low_res_masks, iou_predictions = self.mask_decoder(
          image_embeddings=curr_embedding,
          image_pe=self.prompt_encoder.get_dense_pe(curr_embedding.shape[:2]),
          sparse_prompt_embeddings=sparse_embeddings,
          dense_prompt_embeddings=dense_embeddings,
          multimask_output=multimask_output,
      )
      out = {
          'iou_predictions': iou_predictions,
          'low_res_logits': low_res_masks,
      }
      if upsample_mask:
        masks = (
            self.postprocess_masks(
                low_res_masks, image_size[0], image_size[1]
            )
            > self.mask_threshold
        )
        out['masks'] = masks
      ret.append(out)
    if return_image_embedding:
      for batch_i, image_embedding in enumerate(image_embeddings):
        ret[batch_i]['image_embedding'] = image_embedding
    if not return_batch_as_list:
      ret = {k: jnp.stack([ret[i][k] for i in range(len(ret))], axis=0)
             for k in ret[0].keys()}
    return ret

  def get_image_embeddings(self, image, padding_mask=None, train=False):
    image = self.preprocess(image, padding_mask)  # (batch_size, H, W, 3)
    image_embeddings = self.image_encoder(
        image, train=train)  # (batch_size, H', W', D)
    return image_embeddings

  @staticmethod
  def postprocess_masks(masks, h, w):
    """Resize masks to input resolution."""
    masks = jax.image.resize(
        masks, (masks.shape[0], masks.shape[1], h, w),
        method='bilinear', antialias=False)
    return masks

  @staticmethod
  def postprocess_to_orig(
      lowres_masks, unpad_size, orig_size, mask_threshold=0.0):
    """Resize masks to input resolution."""
    lowres_h, lowres_w = lowres_masks.shape[1:]
    unpad_h, unpad_w = unpad_size
    down_ratio = max(lowres_h, lowres_w) / max(unpad_h, unpad_w)
    h, w = int(unpad_h * down_ratio), int(unpad_w * down_ratio)
    orig_h, orig_w = orig_size

    masks = (
        jax.image.resize(
            jax.device_put(
                lowres_masks[:, :h, :w],
                device=jax.local_devices(backend='cpu')[0],
            ),
            (lowres_masks.shape[0], orig_h, orig_w),
            method='bilinear',
            antialias=False,
        )
        > mask_threshold
    )
    boxes = utils.batched_mask_to_box_np(np.asarray(masks))
    return masks, boxes

  def preprocess(self, inputs, padding_mask=None):
    """Proprocess images. Normalize pixels for non-padded pixels."""
    mean = jnp.asarray(self.pixel_mean, dtype=jnp.float32).reshape(1, 1, 1, 3)
    std = jnp.asarray(self.pixel_std, dtype=jnp.float32).reshape(1, 1, 1, 3)
    inputs = (inputs - mean) / std
    if padding_mask is not None:
      inputs = inputs * padding_mask[..., None]  # Padded pixels remain 0
    return inputs

  def generate(
      self, image=None, padding_mask=None, upsample_mask=True,
      image_embedding=None, return_image_embedding=False):
    """Automatically generate masks for all objects.

    This function is from the original SamAutomaticMaskGenerator at
    https://github.com/facebookresearch/segment-anything/blob/HEAD/
    segment_anything/automatic_mask_generator.py.

    Here we merge it inside the Sam flax model, as we don't use a separate
    predictor class.

    Here are a few key differences compared to the original implementation:

      - The original implementation did filtering inside each prompt-batch. We
        can't do this in jax as the filtering changes the data shape. Instead,
        we do a filtering after concatenating the raw outputs from all batches,
        and use an additional parameter "pre_nms_topk" to control the output
        shape. By default "pre_nms_topk" is half of all prompts.

      - We move mask upsampling (i.e., "postprocess_masks") to the very end of
        the process (after NMS), to save peak memory. This means the box-NMS and
        the stability_score are computed on the 4x-downsampled masks. This
        introduces small errors compared to the original implementation.

      - We don't support the multi-crop testing in the original code as this is
        not enabled in the default config.

    Args:
      image: a single image, (H x W x 3)
      padding_mask: (H x W)
      upsample_mask: bool; If False, only return the 4x downsampled masks. This
        saves memory.
      image_embedding: image embeddings if they are provided. (H', W', D). If
        not provided, image must be not None.
      return_image_embedding: bool
    Returns:
      Result dict of that image, with keys:
        'masks': (self.max_objects H, W).
        'iou_predictions': (self.max_objects,). Predicted mask quality scores.
        'low_res_logits': (self.max_objects, H', W'). The output mask of the
          mask decoder. The final masks are resized from this.
        'boxes': (self.max_objects, 4). Box from the masks.
        'stability_score': (stability_score,). A measurement of how stable the
          mask is when self.mask_threshold changes.
    """
    msg = 'One of "image" or "image_embedding" should be provided!'
    assert image is not None or image_embedding is not None, msg
    assert image is None or image_embedding is None, msg
    if image_embedding is None:
      padding_mask = padding_mask if padding_mask is not None else (
          jnp.ones((image.shape[0], image.shape[1]), dtype=jnp.float32))
      image_embedding = self.get_image_embeddings(
          image[None], padding_mask=padding_mask[None])[0]  # (H', W', D)
    else:
      nopadding_msg = 'Padding_mask should be provided if using image_embedding'
      assert padding_mask is not None, nopadding_msg

    point_grid = utils.build_point_grid(
        self.points_per_side)[:, None]  # (points_per_side ** 2, 1, 2)
    # Ignore padded region in creating grid.
    valid_h = padding_mask.max(axis=1).sum()
    valid_w = padding_mask.max(axis=0).sum()
    point_grid = point_grid * jnp.asarray(
        [valid_w, valid_h], dtype=jnp.float32).reshape(1, 1, 2)
    point_labels = jnp.ones(
        (point_grid.shape[0], point_grid.shape[1]),
        dtype=jnp.int32)  # (points_per_side ** 2, 1)

    num_prompts = point_grid.shape[0]
    bs = self.points_per_batch
    assert num_prompts % bs == 0, num_prompts
    num_batches = num_prompts // bs
    low_res_masks, iou_predictions = [], []
    for b in range(num_batches):
      in_points = point_grid[b * bs: (b + 1) * bs]
      in_labels = point_labels[b * bs: (b + 1) * bs]
      sparse_embeddings_cur, dense_embeddings_cur = self.prompt_encoder(
          in_points, in_labels,
          image_size=image.shape[:2],
          image_embedding_size=image_embedding.shape[:2])
      low_res_masks_cur, iou_predictions_cur = self.mask_decoder(
          image_embeddings=image_embedding,
          image_pe=self.prompt_encoder.get_dense_pe(image_embedding.shape[:2]),
          sparse_prompt_embeddings=sparse_embeddings_cur,
          dense_prompt_embeddings=dense_embeddings_cur,
          multimask_output=True,
      )  # low_res_masks: (bs, 3, h', w')
      low_res_masks.append(low_res_masks_cur)
      iou_predictions.append(iou_predictions_cur)
    ret = {}
    if return_image_embedding:
      ret['image_embedding'] = image_embedding
    del image_embedding

    low_res_masks = jnp.concatenate(
        low_res_masks, axis=0)
    iou_predictions = jnp.concatenate(iou_predictions, axis=0)
    low_res_masks = low_res_masks.reshape(
        (-1,) + low_res_masks.shape[-2:])  # (points_per_side ** 2 * 3, h', w')
    iou_predictions = iou_predictions.reshape(-1)  # (points_per_side ** 2 * 3,)
    keep_mask = iou_predictions > self.pred_iou_thresh

    # Note: the original code computes stability_score on upsampled masks.
    stability_score = utils.calculate_stability_score(
        low_res_masks,
        self.mask_threshold, self.stability_score_offset)
    if self.stability_score_thresh > 0.0:
      keep_mask = keep_mask & (stability_score > self.stability_score_thresh)

    iou_predictions = iou_predictions * keep_mask

    _, inds = jax.lax.top_k(iou_predictions, k=self.pre_nms_topk)
    iou_predictions = jnp.take_along_axis(iou_predictions, inds, axis=0)
    low_res_masks = jnp.take_along_axis(
        low_res_masks, inds[:, None, None], axis=0)

    # Note: the original code run NMS on upsampled masks.
    low_res_boxes = utils.batched_mask_to_box(
        low_res_masks > self.mask_threshold)
    keep_inds = utils.nms(
        low_res_boxes, iou_predictions,
        iou_threshold=self.box_nms_thresh,
        num_outputs=self.max_objects)  # (max_objects,)
    low_res_masks = jnp.take_along_axis(
        low_res_masks, keep_inds[:, None, None], axis=0)
    ret.update({
        'iou_predictions': jnp.take_along_axis(
            iou_predictions, keep_inds, axis=0),
        'low_res_logits': low_res_masks,
        'low_res_boxes': jnp.take_along_axis(
            low_res_boxes, keep_inds[:, None], axis=0),
        'stability_score': jnp.take_along_axis(
            stability_score, keep_inds, axis=0),
    })
    if upsample_mask:
      masks = (
          self.postprocess_masks(
              low_res_masks[None], image.shape[0], image.shape[1]
          )[0]
          > self.mask_threshold
      )
      boxes = utils.batched_mask_to_box(masks)
      ret['masks'] = masks
      ret['boxes'] = boxes
    return ret

  def batch_generate(self, image, padding_mask, upsample_mask=True):
    return jax.vmap(lambda x, y: self.generate(x, y, upsample_mask))(
        image, padding_mask)

