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

"""Layers / Flax modules for OWL-ViT."""

import abc
import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union, Sequence

from absl import logging
from big_vision.models import bit
from big_vision.models import vit
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models import box_utils
from scenic.projects.owl_vit import utils
from scenic.projects.owl_vit.clip import layers as clip_layers
from scenic.projects.owl_vit.clip import model as clip_model

Params = Dict[Any, Any]


class ResNet(nn.Module):
  """ResNetV1 based on big_vision/models/bit.py.

  This variant makes the root_block optional.

  Attributes:
    num_classes: Number of output channels for final projection. If set to zero,
      no final projection will be done.
    width: Width multiplier for the ResNet.
    depth: Sequence of ints specifying depth of each stage, or int specifying
      one of the standard ResNet depths.
    root_block: Whether to apply the root block or not.
  """
  num_classes: int
  width: float = 1
  depth: Union[int, Sequence[int]] = 50
  root_block: bool = True

  @nn.compact
  def __call__(self, inputs, *, train=False):
    del train  # Unused
    blocks = bit.get_block_desc(self.depth)
    width = int(64 * self.width)

    outs = {}

    # Root block.
    if self.root_block:
      convolved = bit.StdConv(
          width, (7, 7), (2, 2), use_bias=False, name='conv_root')(inputs)
      normed = nn.GroupNorm(name='gn_root')(convolved)
      rectified = nn.relu(normed)
      pooled = nn.max_pool(rectified, (3, 3), strides=(2, 2), padding='SAME')
      body_in = outs['stem'] = pooled
    else:
      body_in = inputs

    # Stages.
    activation = bit.ResNetStage(blocks[0], nmid=width, name='block1')(body_in)
    outs['stage1'] = activation
    for i, block_size in enumerate(blocks[1:], 1):
      activation = bit.ResNetStage(
          block_size, nmid=width * 2**i,
          first_stride=(2, 2),
          name=f'block{i + 1}')(activation)
      outs[f'stage{i + 1}'] = activation
    outs['pre_logits_2d'] = activation

    # Head.
    main_out = outs['pre_logits'] = jnp.mean(outs['pre_logits_2d'], axis=(1, 2))

    if self.num_classes:
      head = nn.Dense(
          self.num_classes, name='head', kernel_init=nn.initializers.zeros)
      outs['logits_2d'] = head(outs['pre_logits_2d'])
      main_out = outs['logits'] = head(outs['pre_logits'])

    return main_out, outs


class HourglassNetwork(nn.Module):
  """Hourglass-like network.

  Similar to https://arxiv.org/pdf/2104.00613.pdf, but based on the BiT ResNet
  instead of the standard ResNet.

  Attributes:
    num_classes: Number of output channels for final projection. If set to zero,
      no final projection will be done.
    width: Width multiplier for the ResNet.
    depth: Sequence of ints specifying depth of each stage, or int specifying
      one of the standard ResNet depths.
  """
  num_classes: int
  width: float = 1
  depth: Union[int, Sequence[int]] = 50

  @nn.compact
  def __call__(self, inputs, *, train=False):
    del train  # Unused
    blocks = list(bit.get_block_desc(self.depth))
    resnet_stage = functools.partial(bit.ResNetStage, first_stride=(1, 1))

    # Encoder:
    _, outs = ResNet(
        num_classes=0, width=self.width, depth=blocks, root_block=False,
        name='encoder')(inputs)
    encoded = [v for k, v in outs.items() if k.startswith('stage')]

    # Bottleneck:
    activation = resnet_stage(
        blocks.pop(), name='bottleneck_block')(encoded.pop())
    bottleneck_width = activation.shape[-1]

    # Decoder:
    for skip, block_size in reversed(list(zip(encoded, blocks))):
      b, h, w, c = skip.shape
      i = bottleneck_width // c
      activation = resnet_stage(
          block_size, nmid=c // 4, name=f'decoder_block{i}')(activation)
      activation = jax.image.resize(
          activation, (b, h, w, activation.shape[-1]), method='bilinear')
      activation = activation + skip
      outs[f'decoder_stage{i}'] = activation
    outs['pre_logits'] = activation

    # Head:
    if self.num_classes:
      main_out = nn.Dense(
          self.num_classes, name='head', kernel_init=nn.initializers.zeros)(
              activation)
    else:
      main_out = outs['pre_logits']

    return main_out, outs


class PredictorMLP(nn.Module):
  """FFN block for predicting continuous outputs, e.g. bounding box coordinates.

  Attributes:
    out_dim: Size of output of this mlp.
    num_layers: Number of layers.
    mlp_dim: Size of hidden dimension of dense layers.
    hidden_activation: Activation function of hidden layers.
    out_activation: Activation of the output.
    dtype: Data type, e.g. jnp.float32.
  """
  out_dim: int
  num_layers: int = 1
  mlp_dim: Optional[int] = None
  hidden_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = nn.gelu
  out_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies FFN MLP block to inputs for prediction."""
    x = inputs
    mlp_dim = self.mlp_dim or x.shape[-1]
    for _ in range(self.num_layers-1):
      x = nn.Dense(mlp_dim, dtype=self.dtype)(x)
      if self.hidden_activation is not None:
        x = self.hidden_activation(x)

    x = nn.Dense(self.out_dim, kernel_init=nn.zeros)(x)
    if self.out_activation is not None:
      x = self.out_activation(x)  # pylint: disable=not-callable
    return x


class ClassPredictor(nn.Module):
  """Open-vocabulary instance class predictor."""
  normalize: bool = False
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      query_embeddings: Optional[jnp.ndarray] = None,
      query_mask: Optional[jnp.ndarray] = None,
  ) -> Dict[str, jnp.ndarray]:
    """Computes class prediction logits.

    Query embeddings from a text encoder define the classification label space.

    Args:
      x: Image features [batch_size, num_patches, emb_dim].
      query_embeddings: The embeddings to classify against of shape [batch_size,
        num_queries, out_dim]. If not specified, only the image class embeddings
        will be returned.
      query_mask: Mask indicating whether query is real (1) or padding (0), of
        shape [batch_size, num_queries].
    Returns:
      Dict with keys 'class_embeddings' and, if query embeddings were provided,
      'pred_logits'.
    """
    if self.out_dim is not None:
      out_dim = self.out_dim
    elif query_embeddings is not None:
      out_dim = query_embeddings.shape[-1]
    else:
      raise ValueError('Unable to infer class head shape. Please pass out_dim.')

    image_class_emb = nn.Dense(
        out_dim, kernel_init=nn.initializers.normal(1e-6))(x)
    if query_embeddings is None:
      return {'class_embeddings': image_class_emb}
    assert out_dim == query_embeddings.shape[-1]

    if self.normalize:
      image_class_emb /= jnp.linalg.norm(
          image_class_emb, axis=-1, keepdims=True) + 1e-6
      query_embeddings /= jnp.linalg.norm(
          query_embeddings, axis=-1, keepdims=True) + 1e-6

    assert query_embeddings.ndim > 2, ('Expects shape (batch, query, out_dim). '
                                       f'Got {query_embeddings.shape}')
    pred_logits = jnp.einsum(
        '...pd,...qd->...pq', image_class_emb, query_embeddings)

    # Apply a learnable shift and scale to logits:
    logit_shift = nn.Dense(1, name='logit_shift')(x)
    logit_scale = nn.Dense(1, use_bias=True, name='logit_scale')(x)
    logit_scale = nn.elu(logit_scale) + 1
    pred_logits = (pred_logits + logit_shift) * logit_scale

    if query_mask is not None:
      if query_mask.ndim > 1:
        query_mask = jnp.expand_dims(query_mask, axis=-2)
      pred_logits = jnp.where(query_mask == 0, -1e6, pred_logits)

    return {'pred_logits': pred_logits, 'class_embeddings': image_class_emb}


class ImageTextEmbedderBase(nn.Module, metaclass=abc.ABCMeta):
  """Embeds images and texts into a shared space."""
  embed_configs: ml_collections.ConfigDict

  @nn.compact
  @abc.abstractmethod
  def __call__(
      self,
      *,
      images: Optional[jnp.ndarray] = None,
      texts: Optional[jnp.ndarray] = None,
      train: bool = False
  ) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    pass

  @abc.abstractmethod
  def load_backbone(self, params: Params,
                    backbone_checkpoint_path: Optional[str]) -> Params:
    """Loads backbone parameters for this model from a checkpoint."""
    pass


class ClipImageTextEmbedder(ImageTextEmbedderBase):
  """Embeds images and texts using the CLIP image-text model."""
  embed_configs: ml_collections.ConfigDict

  @nn.compact
  def __call__(
      self,
      *,
      images: Optional[jnp.ndarray] = None,
      texts: Optional[jnp.ndarray] = None,
      train: bool = False
  ) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """Embeds images and texts using the CLIP image-text model."""
    texts_shape = None
    if texts is not None:
      texts_shape = texts.shape
      if len(texts_shape) > 2:
        texts = texts.reshape(-1, texts_shape[-1])

    model_config = clip_model.CONFIGS[self.embed_configs['variant']]
    model_config['vision_return_map'] = True
    # Copy over required CLIP config settings:
    for name in [
        'text_stochastic_droplayer_rate',
        'vision_stochastic_droplayer_rate',
    ]:
      if self.embed_configs.get(name) is not None:
        model_config[name] = self.embed_configs[name]
    # Copy over optional CLIP config settings:
    model_config['vision_native_grid_size'] = self.embed_configs.get(
        'native_image_grid_size'
    )
    model = clip_layers.CLIP(**model_config, name='clip')
    # Input images should have range (0.0, 1.0). Shift them to CLIP range:
    if images is not None:
      images = clip_model.normalize_image(images)
    # Don't normalize image and text embeddings.
    img_emb, txt_emb = model(
        images, texts, normalize=False, deterministic=not train)
    # Drop or merge class embedding token.
    if img_emb is not None:
      merge_class_token = self.embed_configs.get('merge_class_token', 'drop')
      if merge_class_token == 'drop':
        img_emb = img_emb[:, 1:, :]   # [B, P, emb_dim]
      elif merge_class_token == 'mul-ln':
        class_token_out = jnp.broadcast_to(
            img_emb[:, :1, :],
            np.array(img_emb.shape) - (0, 1, 0))
        img_emb = img_emb[:, 1:, :] * class_token_out   # [B, P, emb_dim]
        img_emb = nn.LayerNorm(name='merged_class_token')(img_emb)
      else:
        raise ValueError(f'Unknown merge_class_token: {merge_class_token}')

    if txt_emb is not None and len(texts_shape) > 2:
      txt_emb = txt_emb.reshape(texts_shape[:-1] + (-1,))
    return img_emb, txt_emb

  def load_backbone(self, params: Params,
                    backbone_checkpoint_path: Optional[str]) -> Params:
    """Loads backbone parameters for this model from a checkpoint."""
    del backbone_checkpoint_path  #  Redundant since we use the model variant.

    # This loads only the CLIP-backbone parameters, not the additional
    # parameters added by the LayerNorm above. This function is only intended
    # for initialization from pretrained CLIP checkpoints, not for loading the
    # whole model, which can be done easily in the trainer.
    loaded = clip_model.load_model_vars(self.embed_configs.variant)['params']
    loaded = flax.core.unfreeze(loaded)

    # Remove unused parameters:
    del loaded['visual']['proj']

    # Resize positional embeddings if necessary for visual tower.
    target_size = params['clip']['visual']['positional_embedding'].shape[0]
    loaded_size = loaded['visual']['positional_embedding'].shape[0]
    if target_size != loaded_size:
      loaded['visual']['positional_embedding'] = utils.resize_posemb(
          loaded['visual']['positional_embedding'], target_size)

    # Truncate positional embeddings if necessary for text tower.
    target_size = params['clip']['text']['positional_embedding'].shape[0]
    loaded_size = loaded['text']['positional_embedding'].shape[0]
    if target_size != loaded_size:
      logging.info('Truncating text positional embeddigns from %s to %s',
                   loaded_size, target_size)
      loaded['text']['positional_embedding'] = (
          loaded['text']['positional_embedding'][:target_size])

    # Cast to float32:
    loaded = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32) if x.dtype == jnp.float16 else x,
        loaded)

    params['clip'] = loaded
    return params


class BoxMaskHead(nn.Module):
  """Head for predicting masks inside bounding boxes.

  The architecture is informed by https://arxiv.org/abs/2104.00613.

  The head takes the following inputs:
   * Predicted boxes and image features (output tokens of the image backbone)
     for each box as inputs.
   * The input image, for extracting additional low-level image features.
   * During training, the ground-truth boxes, to select which predicted boxes
     to predict masks for.

  The head performs the following steps:
    1. Apply a small ResNet to the input image to extract low-level features.
    2. Apply ROIAlign to the ResNet features to get per-box low-level features.
    3. Merge per-box low-level features with the image features coming from the
       main image backbone.
    4. Apply an Hourglass network to the per-box features, to merge low- and
       high-level features. Applying a relatively large/deep per-box network was
       found to be useful especially for novel classes in
       https://arxiv.org/abs/2104.00613. The outputs of the Hourglass network
       are the final segmentation masks.

  Attributes:
    mask_size: Integer specifying the width and height of the predicted masks.
    roi_align_num_parallel: Number of boxes to call roi_align on in parallel.
      Larger values are faster but consume more memory.
    stop_box_gradients: Whether to stop the box gradients from flowing back to
      the main model.
    stop_image_gradients: Whether to stop the image feature gradients from
      flowing back to the main model.
    num_training_boxes: If set, only the top predicted boxes by IoU with ground-
      truth boxes will be used during training. This speeds up training because
      most predicted boxes will not be matched to true boxes during training,
      and predicting masks for these unmatched boxes takes time but provides no
      training signal.
    num_mlp_layers_backbone_features: How many MLP layers to apply to the
      backbone image features before merging them with the low-level image
      features.
    image_resnet_width: Width multiplier of the low-level image ResNet.
    image_resnet_depth: Depth spec of the low-level image ResNet.
    mask_resnet_width: Width multiplier of the per-box Hourglass network.
    mask_resnet_depth: Depth spec of the per-box Hourglass network.
    add_image_coords: Whether to add image-centric x/y-coordinate maps to the
      low-level features.
    add_mask_coords: Whether to add mask-centric x/y-coordinate maps to the
      per-box features.
    resnet_out_width_mult: Width multiplier for the low-level features.
    backbone_out_width_mult: Width multiplier for the image backbone features.
  """
  mask_size: int
  roi_align_num_parallel: int
  stop_box_gradients: bool
  stop_image_gradients: bool
  num_training_boxes: Optional[int] = None
  num_mlp_layers_backbone_features: int = 0
  image_resnet_width: float = 0.5
  image_resnet_depth: Union[int, Tuple[int, ...]] = (1, 1, 1, 1)
  mask_resnet_width: float = 1.0
  mask_resnet_depth: Union[int, Tuple[int, ...]] = (1, 1, 1, 1)
  add_image_coords: bool = False
  add_mask_coords: bool = False
  resnet_out_width_mult: int = 1
  backbone_out_width_mult: int = 1

  def _gather_top_boxes(self, boxes, image_backbone_features, true_boxes):
    """Gathers the top instances based on IoU with ground-truth boxes."""
    top_k_indices = None
    if self.num_training_boxes is not None and true_boxes is not None:
      iou_mat, _ = box_utils.box_iou(
          boxes1=box_utils.box_cxcywh_to_xyxy(boxes),
          boxes2=box_utils.box_cxcywh_to_xyxy(true_boxes),
          all_pairs=True)
      max_iou = jnp.max(iou_mat, axis=-1)
      _, top_k_indices = jax.lax.top_k(
          max_iou, self.num_training_boxes or true_boxes.shape[-2])
      gather = jax.vmap(lambda arr, idx: arr[idx])
      boxes = gather(boxes, top_k_indices)
      image_backbone_features = gather(image_backbone_features, top_k_indices)
    return boxes, image_backbone_features, top_k_indices

  def _scatter_top_masks(self, top_pred_masks, top_k_indices, orig_num_boxes):
    """Scatters the top masks back to a full-sized array of all-zero masks."""
    b, _, h, w = top_pred_masks.shape
    pred_masks = jnp.zeros_like(
        top_pred_masks, shape=(b, orig_num_boxes, h, w))
    scatter = lambda arr, idx, update: arr.at[idx].set(update)
    return jax.vmap(scatter)(pred_masks, top_k_indices, top_pred_masks)

  def _roi_align_image_features(self, image_features, boxes):
    # To reduce peak memory consumption, some boxes are processed serially.
    b, num_instances, _ = boxes.shape
    if num_instances % self.roi_align_num_parallel:
      raise ValueError('roi_align_num_parallel must evenly divide '
                       f'num_instances ({num_instances}).')
    serial_batch_size = num_instances // self.roi_align_num_parallel
    boxes = jnp.reshape(
        boxes, (b, self.roi_align_num_parallel, serial_batch_size, 4))
    roi_align_image = jax.vmap(roi_align_batch_serial, in_axes=[None, 0, None])
    roi_align_batch = jax.vmap(roi_align_image, in_axes=[0, 0, None])
    roi_features = roi_align_batch(image_features, boxes, self.mask_size)
    return jnp.reshape(
        roi_features, (b, num_instances, self.mask_size, self.mask_size, -1))

  def _add_coord_channels(self, features):
    b, h, w, _ = features.shape
    xg, yg = jnp.meshgrid(jnp.linspace(0, 1, w), jnp.linspace(0, 1, h))
    xg = jnp.broadcast_to(xg[None, :, :, None], (b, h, w, 1))
    yg = jnp.broadcast_to(yg[None, :, :, None], (b, h, w, 1))
    return jnp.concatenate([features, xg, yg], axis=-1)

  @nn.compact
  def __call__(self,
               image: jnp.ndarray,
               image_backbone_features: jnp.ndarray,
               boxes: jnp.ndarray,
               *,
               true_boxes: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Forward pass of the mask head.

    Args:
      image: [B, H, W, C] input image.
      image_backbone_features: [B, num_boxes, D] image backbone output tokens.
      boxes: [B, num_boxes, 4] predicted boxes.
      true_boxes: [B, num_true_boxes, 4] ground-truth bounding boxes. Only used
        during training.

    Returns:
      [B, num_masks, mask_width, mask_width] array of segmentation masks.

    Raises:
      ValueError if true boxes are provided but gradients are not stopped.
      ValueError if roi_align_num_parallel does not evenly divide the number of
        boxes.
    """
    if not self.stop_box_gradients and true_boxes is not None:
      raise ValueError('stop_box_gradients must be true when using true boxes.')

    if self.stop_box_gradients:
      boxes = jax.lax.stop_gradient(boxes)
    if self.stop_image_gradients:
      image_backbone_features = jax.lax.stop_gradient(image_backbone_features)

    # If true boxes are available, only compute masks for the predictions that
    # overlap most with the true boxes (the others won't get matched anyway).
    orig_num_boxes = boxes.shape[-2]
    top_k_indices = None
    if true_boxes is not None:
      boxes, image_backbone_features, top_k_indices = self._gather_top_boxes(
          boxes, image_backbone_features, true_boxes)

    # Apply a ResNet to get low-level image features:
    _, out = ResNet(
        num_classes=0,
        width=self.image_resnet_width,
        depth=self.image_resnet_depth,
        name='image_resnet')(
            image)
    resnet_features = {
        k: v for k, v in out.items() if k == 'stem' or k.startswith('stage')
    }

    # Reduce number of channels for each stage and resize:
    b, h, w, c = resnet_features['stem'].shape
    for name, feature in resnet_features.items():
      if name.startswith('stage'):
        feature = nn.Dense(c)(feature)
        resnet_features[name] = jax.image.resize(
            feature, (b, h, w, c), method='linear')

    # Concatenate and project to manageable size:
    resnet_features = jnp.concatenate(list(resnet_features.values()), axis=-1)
    resnet_features = nn.Dense(c * self.resnet_out_width_mult)(resnet_features)

    if self.add_image_coords:
      resnet_features = self._add_coord_channels(resnet_features)

    # Get feature map for each box using RoIAlign:
    roi_resnet_features = self._roi_align_image_features(resnet_features, boxes)

    # Process and project backbone features:
    for _ in range(self.num_mlp_layers_backbone_features):
      image_backbone_features = vit.MlpBlock()(image_backbone_features)
    image_backbone_features = nn.Dense(c * self.backbone_out_width_mult)(
        image_backbone_features)

    # Concatenate image features with maps created by replicating the
    # backbone output features in space:
    b, num_instances, h, w, _ = roi_resnet_features.shape
    backbone_feature_map = jnp.broadcast_to(
        image_backbone_features[:, :, None, None, :],
        (b, num_instances, self.mask_size, self.mask_size,
         c * self.backbone_out_width_mult))
    roi_features = jnp.concatenate(
        [roi_resnet_features, backbone_feature_map], axis=-1)

    # Apply per-mask Hourglass network:
    roi_features = jnp.reshape(
        roi_features, (b * num_instances, h, w, roi_features.shape[-1]))
    if self.add_mask_coords:
      roi_features = self._add_coord_channels(roi_features)
    pred_masks_batch, _ = HourglassNetwork(
        num_classes=1,
        width=self.mask_resnet_width,
        depth=self.mask_resnet_depth,
        name='mask_hourglass')(roi_features)
    pred_masks = jnp.reshape(pred_masks_batch, (b, num_instances, h, w))

    # If we're only predicting masks for the best boxes, scatter them back to
    # full size to align with the predicted boxes:
    if top_k_indices is not None:
      pred_masks = self._scatter_top_masks(pred_masks, top_k_indices,
                                           orig_num_boxes)

    return pred_masks

  def load(
      self, params: Params, init_config: ml_collections.ConfigDict
  ) -> Params:
    """Loads backbone parameters for this model from a checkpoint."""
    params = params.copy()
    params['image_resnet'] = bit.load(
        params['image_resnet'],
        init_config.image_resnet,
        None,
        dont_load=('head/.*',),
    )
    return params


def roi_align_batch_serial(feature_map: jnp.ndarray, boxes: jnp.ndarray,
                           output_width: int) -> jnp.ndarray:
  """Applies RoIAlign serially to a batch of boxes."""
  roi_align_single = functools.partial(
      roi_align, feature_map, output_width=output_width)
  return jax.lax.map(roi_align_single, boxes)


def roi_align(feature_map: jnp.ndarray, box: jnp.ndarray,
              output_width: int) -> jnp.ndarray:
  """Extracts a fixed-size feature map for an ROI from a larger feature map.

  See the Mask-RCNN paper (https://arxiv.org/abs/1703.06870) for details on
  ROIAlign.

  Args:
    feature_map: [H, W, C] map of features from which to crop a region of
      interest.
    box: [cx, cy, w, h] bounding box defining the region of interest.
    output_width: The output region will be resized to [width, width].

  Returns:
    Crop of size [width, width] taken from feature_map.
  """
  input_height, input_width, c = feature_map.shape
  output_height = output_width

  cx, cy, w, h = jnp.split(box, 4, axis=-1)
  x0 = cx - w / 2
  y0 = cy - h / 2
  w = jnp.maximum(w, 1e-6)
  h = jnp.maximum(h, 1e-6)
  x_scale = output_width / (w * input_width)
  y_scale = output_height / (h * input_height)

  return jax.image.scale_and_translate(
      feature_map,
      shape=(output_height, output_width, c),
      spatial_dims=(0, 1),
      scale=jnp.concatenate((y_scale, x_scale)),
      translation=jnp.concatenate(
          (-y0 * output_height / h, -x0 * output_width / w)),
      method='linear',
      precision=jax.lax.Precision('fastest'))
