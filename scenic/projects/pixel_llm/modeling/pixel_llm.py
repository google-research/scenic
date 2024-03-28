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

"""PixelLLM model."""

import dataclasses
from typing import Any

from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models import base_model
from scenic.projects.pixel_llm import auto_regressive_decode
from scenic.projects.pixel_llm.modeling import builder
from scenic.projects.pixel_llm.modeling import losses as losses_lib
from scenic.projects.pixel_llm.modeling import utils

ConfigDict = ml_collections.ConfigDict
GIT_PIXEL_MEAN = (0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255)
GIT_PIXEL_STD = (0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255)
GIT_IMAGE_SIZE = (384, 384)
DET_PIXEL_MEAN = (123.675, 116.28, 103.53)
DET_PIXEL_STD = (58.395, 57.12, 57.375)
DET_IMAGE_SIZE = (1024, 1024)
SAM_PIXEL_MEAN = (123.675, 116.28, 103.53)
SAM_PIXEL_STD = (58.395, 57.12, 57.375)
SAM_IMAGE_SIZE = (1024, 1024)


class PixelLlmFlaxModel(nn.Module):
  """Universal Model for Det/Sam/Git training inference."""

  git_backbone_name: str = 'git_vit'
  git_backbone_args: ConfigDict = dataclasses.field(default_factory=ConfigDict)
  git_backbone_param_name: str = 'git_backbone'
  git_preprocess_args: ConfigDict = dataclasses.field(
      default_factory=ConfigDict
  )
  det_backbone_name: str = 'centernet_vit'
  det_backbone_args: ConfigDict = dataclasses.field(default_factory=ConfigDict)
  det_backbone_param_name: str = 'det_backbone'
  det_preprocess_args: ConfigDict = dataclasses.field(
      default_factory=ConfigDict
  )
  sam_backbone_name: str = 'sam_vit'
  sam_backbone_args: ConfigDict = dataclasses.field(default_factory=ConfigDict)
  sam_backbone_param_name: str = 'sam_backbone'
  sam_preprocess_args: ConfigDict = dataclasses.field(
      default_factory=ConfigDict
  )
  frozen_feature_keys: str = ''
  text_decoder_name: str = 'git'
  text_decoder_args: ConfigDict = dataclasses.field(default_factory=ConfigDict)
  text_decoder_param_name: str = 'textual'
  text_decoder_feature_key: str = 'git_visual_features'
  box_decoder_name: str = 'centernet2_det_decoder'
  box_decoder_args: ConfigDict = dataclasses.field(default_factory=ConfigDict)
  box_decoder_param_name: str = 'box_decoder'
  box_decoder_feature_key: str = 'det_visual_features'
  mask_decoder_name: str = 'sam_mask_decoder'
  mask_decoder_args: ConfigDict = dataclasses.field(default_factory=ConfigDict)
  mask_decoder_param_name: str = 'mask_decoder'
  mask_decoder_feature_key: str = 'sam_visual_features'
  prompt_encoder_name: str = 'sam_prompt_encoder'
  prompt_encoder_args: ConfigDict = dataclasses.field(
      default_factory=ConfigDict
  )
  prompt_encoder_param_name: str = 'prompt_encoder'
  prompt_adapter_name: str = 'sam_prompt_adapter'
  prompt_adapter_args: ConfigDict = dataclasses.field(
      default_factory=ConfigDict
  )
  prompt_adapter_param_name: str = 'prompt_adapter'
  prompt_use_box_rate: float = 0.0
  point_predictor_name: str = 'mlp_point_predictor'
  point_predictor_args: ConfigDict = dataclasses.field(
      default_factory=ConfigDict
  )
  point_predictor_param_name: str = 'point_predictor'
  visual_project_layers_name: str = 'none'
  visual_project_layers_args: ConfigDict = dataclasses.field(
      default_factory=ConfigDict
  )
  visual_project_layers_param_name: str = 'visual_project_layers'
  mask_adapter_name: str = 'none'
  mask_adapter_args: ConfigDict = dataclasses.field(
      default_factory=ConfigDict
  )
  mask_adapter_param_name: str = 'mask_adapter'

  max_caption_length: int = 40
  begin_token_id: int = 101  # tokenizer.cls_token_id == 101
  end_token_id: int = 102  # tokenizer.sep_token_id == 102
  vocab_size: int = 30522  # size of BertTokenizer
  label_smooth: float = 0.1
  text_loss_weight: float = 1.0
  det_loss_weight: float = 1.0
  point_loss_weight: float = 1.0
  mask_loss_weight: float = 1.0
  prompt_fuse_fn: str = 'sparse'
  point_output_ignore: str = ''
  trace_point_output_ignore: str = ''
  prompt_drop_rate: float = 0.0
  num_detections: int = 100
  use_roi_box_in_training: bool = False
  num_text_proposals: int = 128
  box_points_per_side: int = 2
  box_point_with_offset: bool = False
  gt_box_points_per_side: int = 2
  gt_box_point_with_offset: bool = False
  point_loss_type: str = 'l1_nonzero'

  def setup(self):
    self.git_backbone = builder.get_image_encoder(
        self.git_backbone_name,
        self.git_backbone_args,
        self.git_backbone_param_name,
    )
    self.det_backbone = builder.get_image_encoder(
        self.det_backbone_name,
        self.det_backbone_args,
        self.det_backbone_param_name,
    )
    self.sam_backbone = builder.get_image_encoder(
        self.sam_backbone_name,
        self.sam_backbone_args,
        self.sam_backbone_param_name,
    )

    self.prompt_encoder = builder.get_prompt_encoder(
        self.prompt_encoder_name,
        self.prompt_encoder_args,
        self.prompt_encoder_param_name,
    )
    self.prompt_adapter = builder.get_prompt_adapter(
        self.prompt_adapter_name,
        self.prompt_adapter_args,
        self.prompt_adapter_param_name,
    )

    self.point_predictor = builder.get_point_predictor(
        self.point_predictor_name,
        self.point_predictor_args,
        self.point_predictor_param_name,
    )
    self.textual = builder.get_text_decoder(
        self.text_decoder_name,
        self.vocab_size,
        self.text_decoder_args,
        self.text_decoder_param_name,
    )

    self.visual_project_layers = builder.get_project_layers(
        self.visual_project_layers_name,
        self.visual_project_layers_args,
        self.visual_project_layers_param_name,
    )

    self.box_decoder = builder.get_box_decoder(
        self.box_decoder_name,
        self.box_decoder_args,
        self.box_decoder_param_name,
    )

    self.mask_decoder = builder.get_mask_decoder(
        self.mask_decoder_name,
        self.mask_decoder_args,
        self.mask_decoder_param_name,
    )

    self.mask_adapter = builder.get_mask_adapter(
        self.mask_adapter_name,
        self.mask_adapter_args,
        self.mask_adapter_param_name,
    )

  @nn.compact
  def __call__(
      self,
      images,
      prompt_boxes=None,
      prompt_point_coords=None,
      gt_text_tokens=None,
      context_tokens=None,
      gt_point_coords=None,
      gt_classes=None,
      gt_boxes=None,
      gt_masks=None,
      padding_mask=None,
      cap_loss_valid_mask=None,
      proposal_loss_valid_mask=None,
      objcap_loss_valid_mask=None,
      point_loss_valid_mask=None,
      preprocess=True,
      train=False,
      with_det=True,
      with_point=True,
      with_mask=True,
      debug=False,
      force_init=False,
  ):
    """forward caption model.

    Args:
      images: (batch_size, height, width, 3) for images or (batch_size, t,
        height, width, 3) for videos (when self.num_frames > 0).
      prompt_boxes: (batch_size, num_caps_per_image, 4)
      prompt_point_coords: (batch_size, num_caps_per_image, num_points, 2)
      gt_text_tokens: (batch_size, num_caps_per_image, max_cap_len)
      context_tokens: (batch_size, num_caps_per_image, max_context_len).
        Optional context tokens. E.g., the question in QA,
      gt_point_coords: (batch_size, num_caps_per_image, max_cap_len,
        num_points_per_token)
      gt_classes: (batch_size, num_caps_per_image)
      gt_boxes: (batch_size, num_caps_per_image, 4). Only used in training.
      gt_masks: (batch_size, num_caps_per_image, height, width, 1)
      padding_mask: optional, (batch_size, height, width)
      cap_loss_valid_mask: optional, (batch_size,)
      proposal_loss_valid_mask: optional, (batch_size,)
      objcap_loss_valid_mask: optional, (batch_size,)
      point_loss_valid_mask: optional, (batch_size,)
      preprocess: bool
      train: bool
      with_det: bool, mostly for inference only
      with_point: bool, mostly for inference only
      with_mask: bool, mostly for inference only
      debug: bool
      force_init: bool

    Returns:
      ret: dict of arrays.
    """
    output_dict = {}
    metric_dict = {}
    del debug
    image_shape = utils.get_image_shape(padding_mask, images)
    padded_image_shape = jnp.concatenate([
        jnp.ones((images.shape[0], 1), jnp.float32) * images.shape[1],
        jnp.ones((images.shape[0], 1), jnp.float32) * images.shape[2],
    ], axis=1)  # B x 2, in order (height, width)
    visual_features_dict = self.forward_backbones(
        images, padding_mask, preprocess=preprocess, train=train
    )
    batch = {
        'images': images,
        'prompt_boxes': prompt_boxes,
        'prompt_point_coords': prompt_point_coords,
        'padding_mask': padding_mask,
        'image_shape': image_shape,
        'padded_image_shape': padded_image_shape,
        'gt_text_tokens': gt_text_tokens,
        'context_tokens': context_tokens,
        'gt_point_coords': gt_point_coords,
        'gt_classes': gt_classes,
        'gt_boxes': gt_boxes,
        'gt_masks': gt_masks,
        'cap_loss_valid_mask': cap_loss_valid_mask,
        'proposal_loss_valid_mask': proposal_loss_valid_mask,
        'objcap_loss_valid_mask': objcap_loss_valid_mask,
        'point_loss_valid_mask': point_loss_valid_mask,
    }
    # object detection (classification)
    if with_det and self.box_decoder is not None:
      det_outputs, det_metrics = self.forward_detection(
          visual_features_dict, batch, train=train
      )
      det_outputs.update(
          self.get_prompt_boxes_and_points(det_outputs, batch, train=train)
      )
      # object caption (like GRiT)
      if self.textual is not None:
        # In multi-task training, image captioning is handled here.
        if batch['gt_boxes'] is not None and train:
          matched_outputs = self.get_matched_proposals_train(
              det_outputs, batch, train=train
          )
          det_outputs.update(matched_outputs)
          if self.point_predictor is not None:
            det_outputs.update(
                self.get_gt_points_from_boxes(matched_outputs, batch)
            )
        cap_outputs, cap_metrics = self.forward_caption(
            visual_features_dict,
            det_outputs,
            batch,
            train=train,
        )
        output_dict.update(cap_outputs)
        metric_dict.update(cap_metrics)
      output_dict.update(det_outputs)
      metric_dict.update(det_metrics)

    # image caption
    # NOTE(jiaruixu): we use elif because textual is handled in the detection
    # `if` already
    elif self.textual is not None:
      output_dict.update(
          self.get_prompt_boxes_and_points(output_dict, batch, train=train)
      )
      cap_outputs, cap_metrics = self.forward_caption(
          visual_features_dict,
          output_dict,
          batch,
          train=train,
      )
      output_dict.update(cap_outputs)
      metric_dict.update(cap_metrics)

    if with_point and self.point_predictor is not None:
      if batch['gt_boxes'] is not None and train:
        output_dict.update(self.get_gt_points_from_boxes(output_dict, batch))
      point_outputs, point_metrics = self.forward_point_prediction(
          output_dict,
          batch,
          train=train,
      )
      output_dict.update(point_outputs)
      metric_dict.update(point_metrics)

    if with_mask and self.mask_decoder is not None:
      if (batch['gt_masks'] is not None and train) or force_init:
        if 'detection_boxes' not in output_dict:
          # when force init, not text is decode yet, so just use ones
          if 'point_valid_mask' not in output_dict:
            point_valid_mask = jnp.ones(output_dict['point_coords'].shape[:-2])  # pytype: disable=attribute-error  # jax-ndarray
          else:
            point_valid_mask = output_dict['point_valid_mask']

          output_dict['detection_boxes'] = self.decode_boxes_from_points(
              output_dict['point_coords'],
              point_valid_mask,
          )['point_detection_boxes']
        mask_decode_outputs, mask_decode_metrics = self.forward_mask_decode(
            visual_features_dict, output_dict, batch, train=train
        )
        output_dict.update(mask_decode_outputs)
        metric_dict.update(mask_decode_metrics)

    if train:
      output_dict['metrics'] = metric_dict

    if with_mask and self.mask_decoder is not None and not train:
      sam_image_embeddings = visual_features_dict[self.mask_decoder_feature_key]
      output_dict['sam_image_embeddings'] = sam_image_embeddings
    # save memory
    output_dict.pop('text_feats')
    return output_dict

  def maybe_project_visual_feature(self, visual_features, train=False):
    """Project visual features if self.project_layers_name != 'none'.

    Args:
      visual_features: (batch_size, num_tokens, dim) or (batch_size,
        num_prompts, num_tokens, dim)
      train: bool

    Returns:
      visual_features: (batch_size, new_num_tokens, new_dim)
        or (batch_size, num_prompts, num_tokens, new_dim)
    """
    num_prompts = 0
    if visual_features.ndim == 4:
      num_prompts = visual_features.shape[1]
      visual_features = jnp.reshape(
          visual_features, (-1,) + visual_features.shape[2:]
      )
    if self.visual_project_layers_name == 'linear':
      visual_features = self.visual_project_layers(visual_features, train=train)
    else:
      assert self.visual_project_layers_name == 'none'

    if num_prompts > 0:
      visual_features = jnp.reshape(
          visual_features, (-1, num_prompts) + visual_features.shape[1:]
      )
    return visual_features

  def forward_backbones(self, images, padding_mask, preprocess, train):
    output_dict = {}
    frozen_feature_keys = self.frozen_feature_keys.split(',')
    if self.git_backbone is not None:
      if preprocess:
        processed_images = utils.preprocess(
            images,
            pixel_mean=self.git_preprocess_args.get(
                'pixel_mean', GIT_PIXEL_MEAN
            ),
            pixel_std=self.git_preprocess_args.get('pixel_std', GIT_PIXEL_STD),
            padding_mask=padding_mask,
            image_size=self.git_preprocess_args.get(
                'image_size', GIT_IMAGE_SIZE
            ),
        )
      else:
        processed_images = images
      output_dict['git_image_size'] = processed_images.shape[1:3]
      frozen_git = 'git_visual_features' in frozen_feature_keys
      git_visual_features = self.git_backbone(
          processed_images, train=train and not frozen_git
      )
      if frozen_git:
        git_visual_features = jax.lax.stop_gradient(git_visual_features)
      if (
          self.git_backbone_name == 'git_vit'
          and self.git_backbone.use_class_embedding
      ) or self.git_backbone_name == 'eva02_vit':
        git_visual_features = git_visual_features[:, 1:]
      git_visual_features = git_visual_features.reshape(
          git_visual_features.shape[0],
          processed_images.shape[1] // self.git_backbone.patch_size,
          processed_images.shape[2] // self.git_backbone.patch_size,
          git_visual_features.shape[-1],
      )
      output_dict['git_visual_features'] = git_visual_features

    if self.sam_backbone is not None:
      if preprocess:
        processed_images = utils.preprocess(
            images,
            pixel_mean=self.sam_preprocess_args.get(
                'pixel_mean', SAM_PIXEL_MEAN
            ),
            pixel_std=self.sam_preprocess_args.get('pixel_std', SAM_PIXEL_STD),
            padding_mask=padding_mask,
            image_size=self.sam_preprocess_args.get(
                'image_size', SAM_IMAGE_SIZE
            ),
        )
      else:
        processed_images = images
      output_dict['sam_image_size'] = processed_images.shape[1:3]
      frozen_sam = 'sam_visual_features' in frozen_feature_keys
      sam_visual_features = self.sam_backbone(
          processed_images, train=train and not frozen_sam
      )
      if frozen_sam:
        sam_visual_features = jax.lax.stop_gradient(sam_visual_features)
      output_dict['sam_visual_features'] = sam_visual_features

    if self.det_backbone is not None:
      if preprocess:
        processed_images = utils.preprocess(
            images,
            pixel_mean=self.det_preprocess_args.get(
                'pixel_mean', DET_PIXEL_MEAN
            ),
            pixel_std=self.det_preprocess_args.get(
                'pixel_std', DET_PIXEL_STD
            ),
            padding_mask=padding_mask,
            image_size=self.det_preprocess_args.get(
                'image_size', DET_IMAGE_SIZE
            ),
        )
      else:
        processed_images = images
      output_dict['det_image_size'] = processed_images.shape[1:3]
      frozen_det = 'det_visual_features' in frozen_feature_keys
      det_visual_features = self.det_backbone(
          processed_images, train=train and not frozen_det
      )
      if frozen_det:
        det_visual_features = jax.lax.stop_gradient(det_visual_features)
      output_dict['det_visual_features'] = det_visual_features

    return output_dict

  def forward_detection(self, visual_features_dict, batch, train=False):
    assert self.box_decoder is not None
    # NOTE(jiaruixu): zhouxy use padded_image_shape instead of true shape
    image_shape = batch['padded_image_shape']
    gt_boxes = batch['gt_boxes']
    gt_classes = batch['gt_classes']
    proposal_loss_valid_mask = batch['proposal_loss_valid_mask']
    if proposal_loss_valid_mask is not None:
      gt_classes = gt_classes * proposal_loss_valid_mask[:, None]
      # NOTE: in CenterNet, valid boxes are boxes that areas>0. Here we
      # replace the boxes based on gt_classes
      gt_boxes = jnp.where(
          proposal_loss_valid_mask[:, None, None] > 0,
          gt_boxes, jnp.zeros_like(gt_boxes)
      )
    output_dict = {}
    metric_dict = {}

    visual_features = utils.concat_visual_features(
        visual_features_dict, self.box_decoder_feature_key
    )
    detections, det_metrics = self.box_decoder(
        # visual_features_dict[self.box_decoder_feature_key],
        visual_features,
        image_shape,
        gt_boxes,
        gt_classes,
        train=train,
    )
    if gt_boxes is not None and train:
      _, det_loss_dict = self.box_decoder.loss_function(
          detections, det_metrics, gt_boxes, gt_classes
      )
      metric_dict.update(det_loss_dict)
    detections.pop('box_regs', None)
    detections.pop('heatmaps', None)
    output_dict.update(detections)
    metric_dict.update(det_metrics)
    return output_dict, metric_dict

  def forward_prompt_encoder_adapter(
      self,
      image_embeddings,
      image_size,
      outputs,
      batch,
      *,
      train=False,
  ):
    assert self.prompt_encoder is not None
    assert self.prompt_adapter is not None
    output_dict = {}
    assert self.prompt_encoder_name == 'sam_prompt_encoder'
    point_coords = utils.get_first_possible_value(
        'prompt_point_coords', [outputs, batch]
    )
    assert point_coords is not None
    # [batch_size, num_prompts, box_points_per_side**2]
    point_labels = utils.generate_point_label(
        self.make_rng('dropout') if batch['gt_boxes'] is not None else None,
        point_coords,
        prompt_drop_rate=self.prompt_drop_rate,
        train=train,
    )

    assert point_coords.shape[:2] == point_labels.shape[:2]
    batch_size, num_prompts = point_coords.shape[:2]

    sparse_embeddings = jax.vmap(
        self.prompt_encoder._embed_points, in_axes=(0, 0, None, None)  # pylint:disable=protected-access
    )(point_coords, point_labels, True, image_size)

    # NOTE(jiaruixu) when from `outputs`` the task is (jointly) object
    # detection + caption; when from `batch``, the task is location caption
    # or global caption
    prompt_boxes = utils.get_first_possible_value(
        'prompt_boxes', [outputs, batch]
    )

    # NOTE(jiaruixu): joint training model should always have 'prompt_boxes'
    if prompt_boxes is not None and self.prompt_use_box_rate > 0.0:
      # [batch_size, num_promps, 2, embed_dim]
      box_sparse_embeddings = jax.vmap(
          self.prompt_encoder._embed_boxes, in_axes=(0, None)  # pylint:disable=protected-access
      )(prompt_boxes, image_size)
      # [batch_size, num_promps, num_points - 2, embed_dim]
      box_pad_embeddings = jnp.tile(
          self.prompt_encoder.no_mask_embed[None, None],
          (batch_size, num_prompts, sparse_embeddings.shape[2] - 2, 1),
      )
      # [batch_size, num_promps, num_points, embed_dim]
      box_sparse_embeddings = jnp.concatenate(
          [box_sparse_embeddings, box_pad_embeddings], axis=-2
      )
      # [batch_size, num_prompts, 1, 1]
      box_valid_mask = jnp.max(prompt_boxes, axis=-1)[..., None, None] > 0
      if train:
        box_valid_mask *= jax.random.uniform(
            key=self.make_rng('dropout'),
            shape=box_valid_mask.shape
        ) < self.prompt_use_box_rate
      output_dict['box_valid_mask'] = box_valid_mask
      sparse_embeddings = jnp.where(
          box_valid_mask, box_sparse_embeddings, sparse_embeddings
      )

    dense_embeddings = self.prompt_encoder.no_mask_embed
    # [batch_size, num_prompts, num_outputs, transformer_dim]
    # [batch_size, num_prompts, H, W, transformer_dim]
    sparse_prompt_features, dense_prompt_features = jax.vmap(
        self.prompt_adapter, in_axes=(0, None, 0, None), out_axes=(0, 0)
    )(
        image_embeddings,
        self.prompt_encoder.get_dense_pe(image_embeddings.shape[1:3]),
        sparse_embeddings,
        dense_embeddings,
    )
    # [batch_size, num_prompts, H*W, transformer_dim]
    dense_prompt_features = dense_prompt_features.reshape(
        dense_prompt_features.shape[:2]
        + (-1, dense_prompt_features.shape[-1])
    )
    if self.prompt_fuse_fn == 'dense':
      visual_features = dense_prompt_features
    elif self.prompt_fuse_fn == 'sparse':
      visual_features = sparse_prompt_features
    else:
      raise ValueError(f'Unknown prompt fuse function {self.prompt_fuse_fn}.')

    output_dict['visual_features'] = visual_features

    return output_dict

  def forward_caption(
      self, visual_features_dict, outputs, batch, *, train=False
  ):
    output_dict = {}
    metric_dict = {}

    image_size = utils.get_image_size(
        visual_features_dict, self.text_decoder_feature_key
    )
    visual_features = utils.concat_visual_features(
        visual_features_dict, self.text_decoder_feature_key
    )

    if self.prompt_encoder is not None:
      prompt_outputs = self.forward_prompt_encoder_adapter(
          visual_features, image_size, outputs, batch, train=train
      )
      output_dict.update(prompt_outputs)
      visual_features = prompt_outputs['visual_features']
    else:
      # [batch_size, 1, H*W, embed_dim]
      visual_features = visual_features.reshape(
          (visual_features.shape[0], 1, -1, visual_features.shape[-1])
      )

    visual_features = self.maybe_project_visual_feature(
        visual_features, train=train
    )
    output_dict['visual_features'] = visual_features

    batch_size = visual_features.shape[0]
    num_caps_per_image = visual_features.shape[1]
    total_batch_size = batch_size * num_caps_per_image

    # unravel batch and num_prompts dim
    # [total_batch_size, visual_seq_len, embed_dim]
    visual_features = visual_features.reshape(
        (total_batch_size,) + visual_features.shape[2:]
    )

    # NOTE(jiaruixu): we get from outputs first because
    # get_matched_proposals_train may update them according to proposal matching
    gt_text_tokens = utils.get_first_possible_value(
        'gt_text_tokens', [outputs, batch]
    )
    context_tokens = utils.get_first_possible_value(
        'context_tokens', [outputs, batch]
    )
    # gt_classes indicates whehter it's a padding/background proposal or not
    gt_text_valid_mask = utils.get_first_possible_value(
        'gt_classes', [outputs, batch]
    )
    if gt_text_valid_mask is not None:
      gt_text_valid_mask = gt_text_valid_mask > 0

    # inference mode
    if gt_text_tokens is None:
      # (batch_size * num_caps_per_image, max_cap_len)
      text_tokens = jnp.full(
          (total_batch_size, self.max_caption_length),
          self.end_token_id,
          dtype=jnp.int32,
      )
      # [batch_size * num_caps_per_image, max_cap_len]
      text_tokens = text_tokens.at[:, 0].set(self.begin_token_id)
      if context_tokens is not None:
        context_tokens = context_tokens[:, :num_caps_per_image]
        # [batch_size * num_caps_per_image, max_cap_len]
        context_tokens = context_tokens.reshape(
            total_batch_size, context_tokens.shape[-1]
        )
    else:
      text_tokens = gt_text_tokens.reshape(
          total_batch_size,
          gt_text_tokens.shape[-1],
      )  # (total_batch_size, num_caps_per_image, max_cap_len)
      if context_tokens is not None:
        context_tokens = context_tokens.reshape(
            total_batch_size, context_tokens.shape[-1]
        )
    # [total_batch_size, max_cap_len, vocab_size]
    text_outputs, text_feats = self.textual(
        text_tokens,
        visual_features,
        context_tokens=context_tokens,
        train=train,
        return_logit_and_feat=True,
    )

    # get num_caps_per_image dim back
    text_outputs = text_outputs.reshape(
        (batch_size, num_caps_per_image) + text_outputs.shape[1:]
    )
    text_feats = text_feats.reshape(
        (batch_size, num_caps_per_image) + text_feats.shape[1:]
    )
    output_dict['text_feats'] = text_feats
    if not train:
      # reshape back
      output_dict['begin_tokens'] = text_tokens.reshape(
          batch_size, num_caps_per_image, text_tokens.shape[-1]
      )
      if context_tokens is not None:
        output_dict['context_tokens'] = context_tokens.reshape(
            batch_size, num_caps_per_image, context_tokens.shape[-1]
        )
      else:
        output_dict['context_tokens'] = None

    if gt_text_tokens is not None:
      metric_dict.update(
          losses_lib.text_loss(
              text_outputs,
              gt_text_tokens,
              gt_text_valid_mask,
              label_smooth=self.label_smooth,
              end_token_id=self.end_token_id,
              vocab_size=self.vocab_size,
          )
      )

    return output_dict, metric_dict

  def forward_point_prediction(self, outputs, batch, *, train=False):
    """Forward point prediction."""
    output_dict = {}
    metric_dict = {}

    text_feats = outputs['text_feats']
    gt_text_tokens = utils.get_first_possible_value(
        'gt_text_tokens', [outputs, batch]
    )
    image_shape = batch['image_shape']

    visual_features = outputs['visual_features']
    point_coords, point_logits = self.point_predictor(
        visual_features, text_feats
    )

    point_coords = utils.points_to_absolute(point_coords, image_shape)
    if train and gt_text_tokens is not None:
      point_valid_mask = utils.get_token_valid_mask(
          gt_text_tokens,
          self.point_output_ignore,
          self.begin_token_id,
          self.end_token_id,
      )
      trace_point_valid_mask = utils.get_token_valid_mask(
          gt_text_tokens,
          self.trace_point_output_ignore,
          self.begin_token_id,
          self.end_token_id,
      )
      if batch['gt_point_coords'] is not None:
        input_gt_point_coords = batch['gt_point_coords'][
            :, : gt_text_tokens.shape[1]
        ]
        input_gt_points_batch_mask = (
            jnp.max(input_gt_point_coords, axis=(1, 2, 3, 4)) > 0
        )
        input_gt_points_batch_mask = input_gt_points_batch_mask[:, None, None]

        point_valid_mask = jnp.where(
            input_gt_points_batch_mask, trace_point_valid_mask, point_valid_mask
        )
      output_dict['point_valid_mask'] = point_valid_mask
      gt_point_coords = utils.get_first_possible_value(
          'gt_point_coords', [outputs, batch]
      )
      # pytype: disable=unsupported-operands
      gt_point_valid_mask = point_valid_mask
      gt_classes = utils.get_first_possible_value(
          'gt_classes', [outputs, batch]
      )
      gt_point_valid_mask *= gt_classes[..., None] > 0
      # pytype: enable=unsupported-operands

      metric_dict.update(
          losses_lib.point_loss(
              point_coords,
              point_valid_mask,
              gt_point_coords,
              gt_point_valid_mask,
              loss_type=self.point_loss_type
          )
      )
    output_dict['point_coords'] = point_coords
    output_dict['point_logits'] = point_logits

    return output_dict, metric_dict

  def forward_mask_decode(
      self, visual_features_dict, outputs, batch, train=False):
    assert self.prompt_encoder is not None
    assert self.mask_decoder is not None
    output_dict = {}
    metric_dict = {}
    assert self.mask_decoder_name == 'sam_mask_decoder'

    image_size = visual_features_dict[
        self.mask_decoder_feature_key.replace('visual_features', 'image_size')
    ]
    image_embeddings = visual_features_dict[self.mask_decoder_feature_key]
    image_embedding_size = image_embeddings.shape[1:3]

    # TODO(jiaruixu): support points besides boxes
    sparse_embeddings = jax.vmap(
        self.prompt_encoder._embed_boxes, in_axes=(0, None)  # pylint:disable=protected-access
    )(outputs['detection_boxes'], image_size)
    if self.mask_adapter is not None:
      sparse_embeddings = self.mask_adapter(
          sparse_embeddings,
          outputs['visual_features'],
          outputs['text_feats'],
      )
    dense_embeddings = self.prompt_encoder.no_mask_embed
    # [batch_size, num_prompts, num_masks, h, w]
    # [batch_size, num_prompts, num_masks]
    low_res_masks, iou_predictions = jax.vmap(
        self.mask_decoder, in_axes=(0, None, 0, None, None), out_axes=(0, 0)
    )(
        image_embeddings,
        self.prompt_encoder.get_dense_pe(image_embedding_size),
        sparse_embeddings,
        dense_embeddings,
        False,
    )
    gt_masks = utils.get_first_possible_value('gt_masks', [outputs, batch])
    if train and gt_masks is not None:
      gt_classes = utils.get_first_possible_value(
          'gt_classes', [outputs, batch]
      )
      metric_dict.update(
          losses_lib.sam_mask_loss(
              low_res_masks,
              iou_predictions,
              gt_masks,
              gt_classes > 0,
              batch['padding_mask'],
          )
      )
    output_dict['detection_masks'] = low_res_masks
    output_dict['iou_predictions'] = iou_predictions

    return output_dict, metric_dict

  def get_matched_proposals_train(
      self,
      detections,
      batch,
      train=False,
  ):
    assert self.box_decoder is not None
    gt_boxes = batch['gt_boxes']
    gt_classes = batch['gt_classes']
    gt_text_tokens = batch['gt_text_tokens']
    context_tokens = batch['context_tokens']

    output_dict = {}

    if self.use_roi_box_in_training or not train:
      # Apply the text loss to the second stage outputs
      # (vs. to the proposal).
      # This needs the second stage to be pretrained.
      # Otherwise the training easily goes NaN.
      # (batch, num_text_proposals, 4)
      last_proposals = detections['detection_boxes'][
          :, : self.num_text_proposals
      ]  # pytype: disable=attribute-error  # jax-ndarray
    else:
      # (batch, num_text_proposals, 4)
      last_proposals = detections['last_proposals'][
          :, : self.num_text_proposals
      ]  # pytype: disable=attribute-error  # jax-ndarray
    objcap_loss_valid_mask = batch['objcap_loss_valid_mask']
    # TODO(zhouxy): try other options: e.g., use proposals or concate prompt
    # boxes and proposals. Now it's using prompt boxes for non-detection data.
    if objcap_loss_valid_mask is not None and batch['prompt_boxes'] is not None:
      prompt_boxes = batch['prompt_boxes']
      # NOTE: if it's not det task, don't use proposal
      last_proposals = jnp.where(
          objcap_loss_valid_mask[:, None, None],
          last_proposals,
          prompt_boxes[:, : self.num_text_proposals],
      )
    output_dict['prompt_boxes'] = last_proposals
    point_coords = self.get_prompt_boxes_and_points(
        output_dict, {}, train=True
    )['prompt_point_coords']
    if objcap_loss_valid_mask is not None and (
        batch['prompt_point_coords'] is not None):
      prompt_points = batch['prompt_point_coords']
      # NOTE: if it's not det task, don't use input prompt points
      point_coords = jnp.where(
          objcap_loss_valid_mask[:, None, None, None],
          point_coords,
          prompt_points[:, : self.num_text_proposals],
      )
    output_dict['prompt_point_coords'] = point_coords

    matched_idxs, matched_gt_classes = self.box_decoder.match_gt(
        last_proposals, gt_boxes, gt_classes
    )
    if objcap_loss_valid_mask is not None:
      # NOTE(jiaruixu): if it's not det task, use all proposals
      matched_idxs = jnp.where(
          objcap_loss_valid_mask[:, None],
          matched_idxs,
          jnp.arange(last_proposals.shape[1])[None],
      )
      matched_gt_classes = jnp.where(
          objcap_loss_valid_mask[:, None],
          matched_gt_classes,
          gt_classes[:, :self.num_text_proposals],
      )
    output_dict['gt_classes'] = matched_gt_classes

    # [batch_size, num_text_proposals, max_cap_len]
    matched_gt_text_tokens = jnp.take_along_axis(
        gt_text_tokens,
        matched_idxs[..., None],
        axis=1,
        mode='promise_in_bounds',
    )
    output_dict['gt_text_tokens'] = matched_gt_text_tokens

    # [batch_size, num_text_proposals, 4]
    matched_gt_boxes = jnp.take_along_axis(
        gt_boxes,
        matched_idxs[..., None],
        axis=1,
        mode='promise_in_bounds',
    )
    output_dict['gt_boxes'] = matched_gt_boxes

    if context_tokens is not None:
      matched_context_tokens = jnp.take_along_axis(
          context_tokens,
          matched_idxs[..., None],
          axis=1,
          mode='promise_in_bounds',
      )
      output_dict['context_tokens'] = matched_context_tokens

    return output_dict

  def get_prompt_boxes_and_points(self, outputs, batch, train=False):
    output_dict = {}
    if 'detection_boxes' in outputs or 'last_proposals' in outputs:
      if self.use_roi_box_in_training or not train:
        detection_boxes = outputs['detection_boxes']
      else:
        detection_boxes = outputs['last_proposals']
      output_dict['prompt_boxes'] = detection_boxes

    prompt_boxes = utils.get_first_possible_value(
        'prompt_boxes', [outputs, output_dict, batch]
    )
    # sample points inside boxes
    # [batch_size, num_prompts, box_points_per_side**2, 2]
    point_coords = utils.boxes_to_points(
        prompt_boxes,
        utils.build_solid_grid(
            self.box_points_per_side, self.box_point_with_offset
        ),
    )
    output_dict['prompt_point_coords'] = point_coords
    return output_dict

  def get_gt_points_from_boxes(self, outputs, batch):
    output_dict = {}
    gt_boxes = utils.get_first_possible_value(
        'gt_boxes', [outputs, batch]
    )
    assert gt_boxes is not None
    # [batch_size, num_text_proposals, num_points, 2]
    gt_point_coords = utils.boxes_to_points(
        gt_boxes,
        utils.build_donut_grid(
            self.gt_box_points_per_side,
            with_offset=self.gt_box_point_with_offset,
        ),
    )
    # [batch_size, num_text_proposals, max_cap_len, num_points, 2]
    gt_point_coords = jnp.tile(
        jnp.expand_dims(gt_point_coords, axis=2),
        [1, 1, self.max_caption_length, 1, 1],
    )
    # NOTE(jiaruixu): for localized narrative trace prediction branch, we
    # shouldn't set gt_point_coords with input gt_boxes, because we are using
    # gt_point_coords for a sequence of points
    if batch['gt_point_coords'] is not None:
      input_gt_point_coords = batch['gt_point_coords'][:, : gt_boxes.shape[1]]
      input_gt_points_batch_mask = (
          jnp.max(input_gt_point_coords, axis=(1, 2, 3, 4)) > 0
      )
      input_gt_points_batch_mask = jnp.reshape(
          input_gt_points_batch_mask, (-1, 1, 1, 1, 1))
      gt_point_coords = jnp.where(
          input_gt_points_batch_mask, input_gt_point_coords, gt_point_coords
      )
    output_dict['gt_point_coords'] = gt_point_coords
    return output_dict

  def decode_text(
      self, text_tokens, visual_features, context_tokens=None, return_feat=False
  ):
    """Generate logits of a single word.

    Args:
      text_tokens: (batch_size, caption_length).
      visual_features: (batch_size, feature_length, feat_size).
      context_tokens: (batch_size, context_length) or None
      return_feat: bool; if True, return shape will be ( batch_size,
        caption_length, hidden_size).

    Returns:
      output_logits: (batch_size, caption_length, vocab_size).
    """
    return self.textual(
        text_tokens, visual_features, context_tokens=context_tokens,
        return_feat=return_feat, train=False)

  def decode_point(self, visual_features, text_feats, image_shape):
    """Generate point coords of a single word.

    Args:
      visual_features: (batch_size, num_caps_per_image, seq_len, emebd_dim)
      text_feats: (batch_size, num_caps_per_image, caption_length, embed_dim)
      image_shape: (batch_size, 2)

    Returns:
      point_coords: (batch_size, num_caps_per_image, caption_length, 2).
    """
    point_outputs, _ = self.forward_point_prediction(
        outputs={'visual_features': visual_features, 'text_feats': text_feats},
        batch={'image_shape': image_shape},
    )

    return point_outputs

  def decode_mask(
      self, image_embeddings, boxes, image_size, visual_features, text_feats):
    assert self.prompt_encoder is not None
    assert self.mask_decoder is not None
    output_dict = {}
    assert self.mask_decoder_name == 'sam_mask_decoder'
    visual_features_dict = {}
    visual_features_dict[self.mask_decoder_feature_key] = image_embeddings
    image_size_key = self.mask_decoder_feature_key.replace(
        'visual_features', 'image_size')
    visual_features_dict[image_size_key] = image_size

    mask_output, _ = self.forward_mask_decode(
        visual_features_dict,
        outputs={
            'visual_features': visual_features,
            'text_feats': text_feats,
            'detection_boxes': boxes,
        },
        batch={},
        train=False,
    )

    low_res_masks = mask_output['detection_masks']
    iou_predictions = mask_output['iou_predictions']
    # low_res_masks = low_res_masks.max(axis=2)
    # iou_predictions = iou_predictions.mean(axis=2)
    top_iou = iou_predictions.max(axis=2)
    top_iou_ind = iou_predictions.argmax(axis=2, keepdims=True)

    top_masks = jnp.take_along_axis(
        low_res_masks, top_iou_ind[..., None, None], axis=2
    )[:, :, 0]
    output_dict['detection_masks'] = top_masks
    output_dict['iou_predictions'] = top_iou

    return output_dict

  def decode_boxes_from_points(self, point_coords, valid_mask):
    """Convert points to detection boxes."""
    output_dict = {}

    # [batch_size, num_boxes, max_caption_length, points_per_token]
    valid_mask = jnp.broadcast_to(
        valid_mask[..., None], point_coords.shape[:-1]  # pytype: disable=attribute-error  # jax-ndarray
    )

    # [batch_size, num_boxes, num_points, 2]
    point_coords = jnp.reshape(point_coords, point_coords.shape[:2] + (-1, 2))  # pytype: disable=attribute-error  # jax-ndarray
    # [batch_size, num_boxes, num_points]
    valid_mask = jnp.reshape(valid_mask, point_coords.shape[:2] + (-1,))

    # [batch_size, num_boxes, 4]
    point_boxes = utils.points_to_boxes(
        point_coords,
        self.gt_box_points_per_side
        if self.gt_box_point_with_offset
        else 0,
        valid_mask=valid_mask,
    )
    point_boxes = jax.lax.stop_gradient(point_boxes)

    output_dict['point_detection_boxes'] = point_boxes

    return output_dict

  def loss_function(
      self,
      outputs: Any,
      batch: Any,
  ):
    """Loss function of PixelLLM.

    Args:
      outputs: dict
      batch: dict
    Returns:
      total_loss: Total loss weighted appropriately.
      metrics: auxiliary metrics for debugging and visualization.
    """
    metrics = outputs['metrics']
    total_loss = 0
    loss_weights = {
        'text_loss': self.text_loss_weight,
        'det_loss': self.det_loss_weight,
        'point_loss': self.point_loss_weight,
        'mask_loss': self.mask_loss_weight,
    }
    cap_loss_valid_mask = batch['label'].get('cap_loss_valid_mask', None)
    objcap_loss_valid_mask = batch['label'].get('objcap_loss_valid_mask', None)
    if cap_loss_valid_mask is not None and objcap_loss_valid_mask is not None:
      text_loss_mask = jnp.minimum(
          cap_loss_valid_mask + objcap_loss_valid_mask, 1.)
    else:
      text_loss_mask = cap_loss_valid_mask if (
          cap_loss_valid_mask is not None) else objcap_loss_valid_mask
    loss_masks = {
        'text_loss': text_loss_mask,
        'det_loss': batch['label'].get('proposal_loss_valid_mask', None),
        'point_loss': batch['label'].get('point_loss_valid_mask', None),
        'mask_loss': batch['label'].get('mask_loss_valid_mask', None),
    }
    for loss_name, loss_weight in loss_weights.items():
      if loss_name in metrics:
        loss = metrics[loss_name]
        if loss_masks[loss_name] is not None:
          # we assume loss_mask should be the same in one batch
          loss_weight *= loss_masks[loss_name].mean()
        metrics[loss_name + '_scaled'] = loss_weight * loss
        total_loss = metrics[loss_name + '_scaled'] + total_loss

    metrics['total_loss'] = total_loss

    return total_loss, metrics


class PixelLlmModel(base_model.BaseModel):
  """Scenic Model Wrapper."""

  def build_flax_model(self):
    fields = set(x.name for x in dataclasses.fields(PixelLlmFlaxModel))
    config_dict = {
        k: v for k, v in self.config.model.items() if k in fields}
    return PixelLlmFlaxModel(**config_dict)

  def prepare_input_spec(self, meta_data):
    """Prepare input spec for model."""
    input_spec = [(
        meta_data['input_shape'],
        meta_data.get('input_dtype', jnp.float32),
    )]
    if (
        self.flax_model.box_decoder_name == 'none'
        and self.flax_model.text_decoder_name != 'none'
    ):
      input_spec.append((
          meta_data['prompt_box_shape'],
          meta_data.get('prompt_box_dtype', jnp.float32),
      ))

    return input_spec

  def train_forward_step(self, model_rng, variables, batch, debug=False):
    flax_model = self.flax_model
    kwargs = {}
    if 'prompt_boxes' in batch['label']:
      kwargs['prompt_boxes'] = batch['label']['prompt_boxes']
    if 'prompt_points' in batch['label']:
      kwargs['prompt_point_coords'] = batch['label']['prompt_points']
    if 'boxes' in batch['label']:
      kwargs['gt_boxes'] = batch['label']['boxes']
    if 'labels' in batch['label']:
      kwargs['gt_classes'] = batch['label']['labels']
    if 'masks' in batch['label']:
      kwargs['gt_masks'] = batch['label']['masks']
    if 'text_tokens' in batch['label']:
      kwargs['gt_text_tokens'] = batch['label']['text_tokens']
    if 'context_tokens' in batch['label']:
      kwargs['context_tokens'] = batch['label']['context_tokens']
    if 'points' in batch['label']:
      kwargs['gt_point_coords'] = batch['label']['points']

    if 'cap_loss_valid_mask' in batch['label']:
      kwargs['cap_loss_valid_mask'] = batch['label']['cap_loss_valid_mask']
    if 'proposal_loss_valid_mask' in batch['label']:
      kwargs['proposal_loss_valid_mask'] = batch[
          'label']['proposal_loss_valid_mask']
    if 'objcap_loss_valid_mask' in batch['label']:
      kwargs['objcap_loss_valid_mask'] = batch[
          'label']['objcap_loss_valid_mask']
    if 'point_loss_valid_mask' in batch['label']:
      kwargs['point_loss_valid_mask'] = batch['label']['point_loss_valid_mask']

    predictions, new_model_state = flax_model.apply(
        variables,
        batch['inputs'],
        padding_mask=batch['padding_mask'],
        preprocess=True,
        mutable=['batch_stats'],
        train=True,
        rngs={'dropout': model_rng},
        debug=debug,
        **kwargs,
    )

    return predictions, new_model_state

  def inference(
      self,
      variables,
      batch,
      with_cap=True,
      with_det=True,
      with_point=True,
      with_mask=True,
      with_gt_prompt=False,
  ):
    """Inference on batch.

    The inference pipeline is in following order:
    1. predict visual features for captions, prepare begin tokens.
    2. autoregressively predict text tokens.
    3. (optional) rescore and add more the boxes with beam search results.
    4. (optional) predict point location based on visual and text feature.
    5. (optional) add point box results.
    6. (optional) forward mask decoder

    NOTE: rescore (step 3) will change the order of detected visual features
    and text tokens. And 3->4->5 yields similar accuracy as 4->5->3.

    Args:
      variables (dict): with params
      batch (dict): with input and label
      with_cap (bool): with caption
      with_det (bool): with detection
      with_point (bool): with point prediction
      with_mask (bool): with mask prediction
      with_gt_prompt (bool): with gt box prompt

    Returns:
      dict
    """
    params = variables['params']
    predictions = self.prepare_caption_prediction(
        params,
        batch,
        with_det=with_det,
        with_point=with_point,
        with_mask=with_mask,
        with_gt_prompt=with_gt_prompt,
    )
    if self.flax_model.text_decoder_name != 'none':
      if with_cap:
        predictions = self.autoregressive_predict(
            params, predictions, feature_key='visual_features'
        )

      if (
          with_det
          and self.flax_model.box_decoder_name != 'none'
          and self.config.model.get('mult_caption_score', False)
      ):
        predictions = self.rescore_detections(predictions)

      if with_point and self.flax_model.point_predictor_name != 'none':
        if 'text_feats' not in predictions:
          predictions = self.prepare_text_features(
              params, batch, predictions, with_cap=with_cap
          )

        predictions = self.predict_points(
            params, batch, predictions, with_cap=with_cap
        )
      if with_point and self.config.model.get('use_points_as_det', False):
        predictions = self.add_point_detection(params, predictions)
    if (
        self.flax_model.mask_decoder_name != 'none'
        and 'detection_boxes' in predictions
    ):
      if (
          'text_feats' not in predictions
          and self.flax_model.mask_adapter_name != 'none'
      ):
        predictions = self.prepare_text_features(
            params, batch, predictions, with_cap=with_cap
        )

      predictions = self.pred_mask(params, predictions, with_cap=with_cap)

    # sav memory
    predictions.pop('text_feats', None)

    return predictions

  def prepare_caption_prediction(
      self,
      params,
      batch,
      with_det=True,
      with_point=True,
      with_mask=True,
      with_gt_prompt=False,
  ):
    """Prepare visual feature and begin token for captioning."""
    kwargs = {}
    if 'context_tokens' in batch['label']:
      # Prompts or questions in QA.
      kwargs['context_tokens'] = batch['label']['context_tokens']
    if 'prompt_boxes' in batch['label']:
      kwargs['prompt_boxes'] = batch['label']['prompt_boxes']
    if with_gt_prompt:
      kwargs['prompt_boxes'] = batch['label']['boxes']
    # get starting tokens for captioning
    predictions = self.flax_model.apply(
        variables={'params': params},
        images=batch['inputs'],
        padding_mask=batch['padding_mask'],
        preprocess=True,
        train=False,
        with_det=with_det,
        with_point=with_point,
        with_mask=with_mask,
        mutable=False,
        **kwargs,
    )
    if with_gt_prompt:
      predictions['detection_boxes'] = batch['label']['boxes']
      predictions['detection_scores'] = jnp.ones(
          batch['label']['boxes'].shape[:-1])

    return predictions

  def autoregressive_predict(
      self, params, predictions, feature_key='visual_features'
  ):
    """Autoregressive decoding text tokens."""
    predictions = auto_regressive_decode.autoregressive_predict(
        self.flax_model,
        params,
        predictions,
        feature_key=feature_key,
        method=self.config.model.get('decode_method', 'greedy'),
        beam_size=self.config.model.get('decode_beam_size', 1),
        per_node_beam_size=self.config.model.get(
            'decode_per_node_beam_size', 2
        ),
    )
    return predictions

  def rescore_detections(self, predictions):
    """Reorder detection boxes given text scores."""
    detection_scores = predictions.pop('detection_scores')
    detection_scores = jnp.maximum(detection_scores, 0.0)
    decode_beam_size = self.config.model.get('decode_beam_size', 1)

    if decode_beam_size == 1:
      predictions['detection_scores'] = (
          detection_scores * jnp.exp(predictions['log_probs'])
      ) ** 0.5
    else:
      predictions.pop('log_probs')

      # [batch_size, roi_post_nms_num_detections, decode_beam_size]
      beam_log_probs = predictions.pop('beam_log_probs')
      # [batch_size, roi_post_nms_num_detections, decode_beam_size]
      beam_scores = (
          detection_scores[..., None] * jnp.exp(beam_log_probs)
      ) ** 0.5
      # [batch_size, roi_post_nms_num_detections * decode_beam_size]
      beam_scores = jnp.reshape(beam_scores, (beam_scores.shape[0], -1))

      # [batch_size, roi_post_nms_num_detections, decode_beam_size,
      # max_caption_length]
      beam_text_tokens = predictions.pop('beam_text_tokens')
      # [batch_size, roi_post_nms_num_detections * decode_beam_size,
      # max_caption_length]
      beam_text_tokens = jnp.reshape(
          beam_text_tokens,
          (beam_text_tokens.shape[0], -1, beam_text_tokens.shape[-1]),
      )
      assert beam_scores.shape[1] == beam_text_tokens.shape[1]

      topk_scores, indices = lax.top_k(
          beam_scores, k=self.flax_model.num_detections
      )
      # [batch_size, roi_post_nms_num_detections, max_caption_length]
      text_tokens = jnp.take_along_axis(
          beam_text_tokens, indices[..., None], axis=1
      )

      predictions['detection_scores'] = topk_scores
      # predictions['detection_boxes'] = detection_boxes
      # predictions['detection_classes'] = detection_classes
      det_indices = indices // decode_beam_size
      for det_filed in [
          'detection_boxes',
          'detection_classes',
          'point_coords',
          'point_logits',
          'point_detection_boxes',
          'inference_point_coords',
          'visual_features',
      ]:
        # in case that the rescore happens before point replace detection
        if det_filed not in predictions:
          continue
        det_pred = predictions.pop(det_filed)
        predictions[det_filed] = jnp.take_along_axis(
            det_pred,
            jnp.reshape(
                det_indices, det_indices.shape + (1,) * (det_pred.ndim - 2)
            ),
            axis=1,
        )

      predictions['text_tokens'] = text_tokens
      predictions['num_detections'] = jnp.sum(
          (topk_scores > 0.0).astype(jnp.int32), axis=-1
      )
    return predictions

  def prepare_text_features(self, params, batch, predictions, with_cap=True):
    """Extract text features of text token."""
    context_tokens = utils.get_first_possible_value(
        'context_tokens', [predictions, batch['label']]
    )
    if with_cap:
      text_tokens = predictions['text_tokens']
    else:
      # in RefCOCO, use ground truth text as input text tokens
      text_tokens = batch['label']['text_tokens']
    visual_features = predictions['visual_features']

    # replace redundent eos with padding token to align with training input
    eos_mask = text_tokens == self.flax_model.end_token_id
    cumsum_eos_mask = jnp.cumsum(eos_mask, axis=-1)
    text_tokens = jnp.where(cumsum_eos_mask <= 1, text_tokens, 0)
    predictions['text_tokens'] = text_tokens

    batch_size = text_tokens.shape[0]
    num_caps_per_image = text_tokens.shape[1]
    total_batch_size = batch_size * num_caps_per_image
    # [total_batch_size, visual_seq_len, embed_dim]
    visual_features = visual_features.reshape(
        (total_batch_size,) + visual_features.shape[2:]
    )
    text_tokens = text_tokens.reshape(
        total_batch_size,
        text_tokens.shape[2],
    )  # (total_batch_size, num_caps_per_image, max_cap_len)
    if context_tokens is not None:
      context_tokens = context_tokens.reshape(
          total_batch_size, context_tokens.shape[2]
      )
    text_feats = self.flax_model.apply(
        variables={'params': params},
        text_tokens=text_tokens,
        visual_features=visual_features,
        context_tokens=context_tokens,
        return_feat=True,
        method=self.flax_model.decode_text,
    )
    text_feats = text_feats.reshape(
        (
            batch_size,
            num_caps_per_image,
        )
        + text_feats.shape[1:]
    )
    predictions['text_feats'] = text_feats

    return predictions

  def predict_points(self, params, batch, predictions, with_cap=True):
    """Predict coordinates from text, for LN trace and RefCOCO."""
    del with_cap
    image_shape = utils.get_image_shape(
        batch['padding_mask'], batch['inputs']
    )

    point_predictions = self.flax_model.apply(
        variables={'params': params},
        visual_features=predictions['visual_features'],
        text_feats=predictions['text_feats'],
        image_shape=image_shape,
        method=self.flax_model.decode_point,
    )
    point_predictions['point_valid_mask'] = (
        utils.get_token_valid_mask(
            predictions['text_tokens'],
            self.flax_model.point_output_ignore,
            self.flax_model.begin_token_id,
            self.flax_model.end_token_id,
        )
    )
    predictions.update(point_predictions)
    return predictions

  def add_point_detection(self, params, predictions):
    """Predict segmentation mask from bounding boxes."""
    box_outputs = self.flax_model.apply(
        variables={'params': params},
        point_coords=predictions['point_coords'],
        valid_mask=predictions['point_valid_mask'],
        method=self.flax_model.decode_boxes_from_points,
    )

    point_boxes = box_outputs['point_detection_boxes']

    if 'detection_boxes' in predictions:
      predictions['point_detection_boxes'] = point_boxes
    else:
      predictions['detection_boxes'] = point_boxes

    return predictions

  def pred_mask(self, params, predictions, with_cap=True):
    """Convert point to bounding boxes (for RefCOCO)."""
    del with_cap
    image_embeddings = predictions['sam_image_embeddings']
    image_size = self.flax_model.sam_preprocess_args.get(
        'image_size', SAM_IMAGE_SIZE
    )
    boxes = predictions['detection_boxes']
    visual_features = predictions['visual_features']
    text_feats = predictions.get('text_feats', None)
    mask_outputs = self.flax_model.apply(
        variables={'params': params},
        image_embeddings=image_embeddings,
        visual_features=visual_features,
        text_feats=text_feats,
        boxes=boxes,
        image_size=image_size,
        method=self.flax_model.decode_mask,
    )

    predictions.update(mask_outputs)

    return predictions

  def loss_function(self, outputs, batch):
    return self.flax_model.loss_function(outputs, batch)
