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

"""Image or video captioning model."""

import dataclasses
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from scenic.model_lib.base_models import base_model
from scenic.projects.streaming_dvc.modeling import text_decoder as bert_text_decoder
from scenic.projects.streaming_dvc.modeling import vit as git_vit

GIT_PIXEL_MEAN = (0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255)
GIT_PIXEL_STD = (0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255)


def get_image_encoder(encoder_type: str,
                      encoder_args: ml_collections.ConfigDict,
                      param_name: str = 'image_encoder') -> nn.Module:
  """Returns an image encoder."""
  if encoder_type == 'git_vit':
    return git_vit.ViT(**encoder_args, name=param_name)
  else:
    raise ValueError(f'Unknown encoder type {encoder_type}.')


class LinearProjectLayers(nn.Module):
  """Linear projection layer."""
  emb_dim: int = 1024
  use_projection_ln: bool = True

  @nn.compact
  def __call__(self, x, train=False):
    # The name `visual_projection.x` is for a historical reason to load
    # weights for other decoders. This is not meaningful here now.
    x = nn.Dense(
        self.emb_dim, name='visual_projection.0',
        kernel_init=nn.initializers.normal(stddev=0.02))(
            x)  # (batch_size, feature_length, hidden_size)
    if self.use_projection_ln:
      x = nn.LayerNorm(
          epsilon=1e-5, name='visual_projection.1')(x)
    return x


class CaptioningFlaxModel(nn.Module):
  """GIT captioning model."""
  max_caption_length: int = 40
  begin_token_id: int = 101  # tokenizer.cls_token_id == 101
  end_token_id: int = 102  # tokenizer.sep_token_id == 102
  vocab_size: int = 30522  # size of BertTokenizer
  label_smooth: float = 0.1
  num_frames: int = 0
  with_temp_emb: bool = True
  frame_fuse_fn: str = 'concat'
  pixel_mean: Any = GIT_PIXEL_MEAN
  pixel_std: Any = GIT_PIXEL_STD
  backbone_name: str = 'git_vit'
  backbone_args: ml_collections.ConfigDict = dataclasses.field(
      default_factory=ml_collections.ConfigDict)
  text_decoder_name: str = 'git'
  text_decoder_args: ml_collections.ConfigDict = dataclasses.field(
      default_factory=ml_collections.ConfigDict)
  decode_method: str = 'greedy'
  decode_beam_size: int = 1
  decode_brevity_penalty_alpha: float = 0.6
  freeze_image_encoder: bool = False
  num_pooled_tokens: int = -1
  backbone_param_name: str = 'image_encoder'
  decode_feature_key: str = 'visual_features'
  project_layers_name: str = 'none'
  project_layers_args: ml_collections.ConfigDict = dataclasses.field(
      default_factory=ml_collections.ConfigDict)
  project_param_name: str = 'project_layers'
  per_frame_qformer: bool = False
  num_bins: int = 100
  show_densecap_loss: bool = False
  # loc_loss_weight is only used in densecap. Negative means do not apply the
  # weight and normalize localization and captioning loss together. If positive,
  # normalize the two losses separately and apply loss weighting.
  loc_loss_weight: float = -1.0
  ignore_empty_data: bool = False

  def setup(self):
    self.image_encoder = get_image_encoder(
        self.backbone_name, self.backbone_args, self.backbone_param_name)
    # pylint: disable=not-a-mapping
    if self.text_decoder_name == 'git':
      self.textual = bert_text_decoder.TransformerDecoderTextualHead(
          vocab_size=self.vocab_size,
          **self.text_decoder_args, name='textual')
    else:
      raise NotImplementedError(self.text_decoder_name)

    if self.project_layers_name == 'linear':
      self.project_layers = LinearProjectLayers(
          **self.project_layers_args,
          name=self.project_param_name)
    elif self.project_layers_name == 'bert':
      self.bert_project_layers = (
          bert_text_decoder.TransformerDecoderTextualHead(
              **self.project_layers_args,
              name=self.project_param_name))
    elif self.project_layers_name != 'none':
      raise NotImplementedError(self.project_layers_name)
    # pylint: enable=not-a-mapping

  @nn.compact
  def __call__(
      self, images,
      context_tokens=None,
      gt_text_tokens=None,
      preprocess=True, train=False, debug=False):
    """forward caption model.

    Args:
      images: (batch_size, height, width, 3) for images or
        (batch_size, t, height, width, 3) for videos (when self.num_frames > 0).
      context_tokens: (batch_size, num_caps_per_image, max_context_len).
        Optional context tokens. E.g., the question in QA,
      gt_text_tokens: (batch_size, num_caps_per_image, max_cap_len)
      preprocess: bool
      train: bool
      debug: bool
    Returns:
      ret: dict of arrays.
        if train == True, return
          'text_outputs': (text_batch_size, max_cap_len, vocab_size)
        if train == False, return
          'visual_features': (text_batch_size, feature_len, feature_dim)
          'begin_tokens': (batch_size, num_caps_per_image, max_cap_len)

    """
    del debug
    if self.num_frames > 0:  # video
      # flattern time to batch
      assert images.ndim == 5
      images = images.reshape(
          (images.shape[0] * images.shape[1],) + images.shape[2:])

    if preprocess:
      images = self.preprocess(images)

    visual_features = self.get_visual_features(
        images, train=train)  # (batch_size, num_tokens, dim)

    visual_features = self.maybe_project_visual_feature(
        visual_features, train=train)  # (batch_size, num_vis_tokens, proj_dim)

    text_tokens, visual_features, context_tokens = (
        self.get_text_tokens_and_pad_visual_features(
            visual_features, gt_text_tokens, context_tokens))
    # text_tokens: (text_batch_size, max_cap_len)
    # visual_features: (text_batch_size, num_vis_tokens, proj_dim)
    # context_tokens: (text_batch_size, num_context_tokens)

    text_outputs = self.textual(
        text_tokens,
        visual_features,
        context_tokens=context_tokens,
        train=train,
    )  # (text_batch_size, max_cap_len, vocab_size)

    if train:
      ret = {'text_outputs': text_outputs}
    else:
      # del text_outputs
      ret = {
          'visual_features': visual_features,
          'begin_tokens': text_tokens,
          'context_tokens': context_tokens,
          'text_outputs': text_outputs,
      }
    return ret

  def maybe_project_visual_feature(self, visual_features, train=False):
    """Project visual features if self.project_layers_name != 'none'.

    Args:
      visual_features: (batch_size, num_tokens, dim)
      train: bool
    Returns:
      visual_features: (batch_size, new_num_tokens, new_dim)
    """
    if self.project_layers_name == 'qformer':
      batch_size = visual_features.shape[0]
      if self.per_frame_qformer:
        assert self.frame_fuse_fn == 'concat'
        visual_features = visual_features.reshape(
            batch_size * self.num_frames, -1, visual_features.shape[-1])
      query_tokens = jnp.broadcast_to(
          self.query_tokens,
          (visual_features.shape[0],
           self.project_layers_args.num_query_tokens,
           self.project_layers_args.qformer_dim))
      query_output = self.qformer(
          query_tokens, visual_features, train=train)
      visual_features = self.t5_proj(
          query_output)  # (batch_size, num_query_tokens, t5_dim)
      if self.per_frame_qformer:
        visual_features = visual_features.reshape(
            batch_size, -1, visual_features.shape[-1])
    elif self.project_layers_name == 'bert':
      visual_features = self.bert_project_layers(
          jnp.zeros(
              (visual_features.shape[0], 0),
              dtype=jnp.int32),
          visual_features,
          train=train, return_feat=True, return_visual_feature=True)
    elif self.project_layers_name == 'linear':
      visual_features = self.project_layers(visual_features, train=train)
    else:
      assert self.project_layers_name == 'none'
    return visual_features

  def get_visual_features(self, images, train=False):
    """Forward image backbone and aggregate video features.

    Args:
      images: (total_batch_size, height, width, 3). Note for videos, the
        total_batch_size is batch_size * num_frames.
      train: bool
    Returns:
      visual_features: (batch_size, num_tokens, dim). Here the batch_size is
        the actual batch_size.
    """
    visual_features = self.image_encoder(images, train=train)  # (B, hw, D)
    if self.freeze_image_encoder:
      visual_features = jax.lax.stop_gradient(visual_features)
    if self.num_frames > 0:  # video model
      num_tokens = visual_features.shape[1]
      visual_features = visual_features.reshape(
          (-1, self.num_frames) + visual_features.shape[1:]
      )  # (B // t, t, hw, D)
      if self.with_temp_emb:
        visual_feat_dim = visual_features.shape[-1]
        temp_emb = self.param(
            'temperal_embedding',
            nn.initializers.zeros,
            (self.num_frames, 1, 1, visual_feat_dim),
        )
        visual_features = visual_features + temp_emb[
            None, :, 0]  # (B // t, t, hw, D)
      if self.frame_fuse_fn == 'concat':
        visual_features = visual_features.reshape(
            visual_features.shape[0], self.num_frames * num_tokens,
            visual_features.shape[-1],
        )  # (B // t, t * hw, D)
      else:
        visual_features = self.pool_video_feature(visual_features, train=train)
    else:  # image model
      visual_features = visual_features.reshape(
          visual_features.shape[0], -1, visual_features.shape[-1],
      )  # (B, hw, D)
    return visual_features

  def get_text_tokens_and_pad_visual_features(
      self, visual_features, gt_text_tokens, context_tokens=None):
    """Get inputs to the text decoder.

    In evaluation, we create the zero-padded text-token with the first token
      being BOS. In training, we handle multiple caption annotations for a
      video and repeat the visual_feature to align with that.

    Args:
      visual_features: (batch_size, num_tokens, dim)
      gt_text_tokens: (batch_size, num_caps_per_image, max_cap_len) or None.
      context_tokens: (batch_size, num_caps_per_image, max_cap_len) or None.
    Returns:
      text_tokens: (text_batch_size, max_cap_len). text_batch_size = (batch_size
        * num_caps_per_image)
      visual_features: (text_batch_size, num_tokens, dim)
      context_tokens: (text_batch_size, num_tokens) or None.
    """
    if gt_text_tokens is None:  # Evaluation, create BOS tokens.
      text_tokens = jnp.full(
          (visual_features.shape[0], self.max_caption_length),
          self.end_token_id, dtype=jnp.int32)  # (B, max_cap_len)
      text_tokens = text_tokens.at[:, 0].set(
          self.begin_token_id)  # (batch_size, max_cap_len)
      if context_tokens is not None:
        context_tokens = context_tokens[:, 0]  # (batch_size, max_cap_len)
    else:  # Training
      batch_size, num_caps_per_image = gt_text_tokens.shape[:2]
      text_tokens = gt_text_tokens.reshape(
          batch_size * num_caps_per_image,
          gt_text_tokens.shape[2],
      )  # (batch_size, num_caps_per_image, max_cap_len)
      visual_features = jnp.broadcast_to(
          visual_features[:, None],
          (batch_size, num_caps_per_image,) + visual_features.shape[1:],
      ).reshape(
          (batch_size * num_caps_per_image,) + visual_features.shape[1:])
      if context_tokens is not None:
        context_tokens = context_tokens.reshape(
            batch_size * num_caps_per_image, context_tokens.shape[2])
    return text_tokens, visual_features, context_tokens

  def pool_video_feature(self, visual_features, train=False):
    """Pool video features before feeding them to the language decoder.

    Args:
      visual_features: (video_batch_size, t, hw, D)
      train: bool
    Returns:
      visual_features: (video_batch_size, num_new_tokens, D)
    """
    video_batch_size, t, hw, dim = visual_features.shape
    if self.frame_fuse_fn == 'temporal_mean_pool':
      visual_features = visual_features.mean(
          axis=1)  # (video_batch_size, hw, D)
    elif self.frame_fuse_fn == 'spatial_mean_pool':
      visual_features = visual_features.mean(axis=2)  # (video_batch_size, t, D)
    elif self.frame_fuse_fn == 'uniform_token_sample':
      assert self.num_pooled_tokens > 0
      visual_features = visual_features.reshape(
          video_batch_size, t * hw, dim)
      if train:
        inds = jax.random.permutation(
            self.make_rng('dropout'),
            jnp.arange(t * hw, dtype=jnp.int32))[:self.num_pooled_tokens]
      else:
        inds = jnp.linspace(
            0, t * hw, self.num_pooled_tokens, endpoint=False, dtype=jnp.int32)
      visual_features = jnp.take_along_axis(
          visual_features, inds[None, :, None], axis=1)
    else:
      raise NotImplementedError(self.frame_fuse_fn)
    return visual_features

  def decode_text(
      self, text_tokens, visual_features,
      context_tokens=None, return_feat=False):
    """Generate logits of a single word.

    Args:
      text_tokens: (batch_size, caption_length).
      visual_features: (batch_size, feature_length, feat_size).
      context_tokens: (batch_size, context_length) or None
      return_feat: bool; if True, return shape will be (
          batch_size, caption_length, hidden_size).
    Returns:
      output_logits: (batch_size, caption_length, vocab_size).
    """
    return self.textual(
        text_tokens, visual_features, context_tokens=context_tokens,
        return_feat=return_feat, train=False)

  def preprocess(self, inputs):
    """Proprocess images. Normalize pixels for non-padded pixels."""
    mean = jnp.asarray(self.pixel_mean, dtype=jnp.float32).reshape(1, 1, 1, 3)
    std = jnp.asarray(self.pixel_std, dtype=jnp.float32).reshape(1, 1, 1, 3)
    inputs = (inputs - mean) / std
    # inputs = inputs * padding_mask[..., None]  # Padded pixels remain 0
    return inputs

  def loss_function(self, outputs, batch):
    """Text loss with label smoothing.

    Args:
      outputs: dict
        'text_outputs':
          (batch_size * num_caps_per_image, max_cap_len, vocab_size)
      batch: dict
        'text_tokens': (batch_size, num_caps_per_image, max_cap_len)
    Returns:
      loss: float
    """
    text_outputs = outputs['text_outputs']
    gt_text = batch['label']['text_tokens']
    gt_text = gt_text.reshape(
        gt_text.shape[0] * gt_text.shape[1], gt_text.shape[2],
    )  # (batch_size * num_caps_per_image, max_cap_len)
    text_outputs = text_outputs[:, :-1]  # Move gt 1 word to the right.
    gt_text = gt_text[:, 1:]  # No need to predict BOS
    # valid: (text_batch_size, max_cap_len - 1)
    valid = (gt_text > 0).astype(jnp.float32)
    if self.ignore_empty_data:
      # Ignore samples with empty ground truth.
      valid = (valid.astype(bool) & (
          gt_text[:, 0] != self.end_token_id)[:, None]).astype(jnp.float32)
    # gt: (text_batch_size, max_cap_len - 1, vocab_size)
    gt = jax.nn.one_hot(gt_text, self.vocab_size)
    # customized label smoothing following GRiT
    #   https://github.com/JialianW/GRiT/blob/master/grit/modeling/text/
    #   text_decoder.py#L668
    gt = gt * (1. - self.label_smooth) + (
        1. - gt) * self.label_smooth / (self.vocab_size - 1)
    # loss:  (text_batch_size, max_cap_len - 1)
    gt = jax.lax.stop_gradient(gt)
    loss = optax.softmax_cross_entropy(text_outputs, gt)
    loss_dict = {}
    # TODO(zhouxy): Create a new DensecapModel class and move this code there.
    if self.show_densecap_loss or self.loc_loss_weight >= 0.0:
      thresh = self.vocab_size - self.num_bins
      cap_idx = ((gt_text < thresh) & (valid > 0)).astype(jnp.float32)
      loc_idx = ((gt_text >= thresh) & (valid > 0)).astype(jnp.float32)
      loss_dict['cap_loss'] = (loss * cap_idx).sum() / (cap_idx.sum() + 1e-8)
      loss_dict['loc_loss'] = (loss * loc_idx).sum() / (loc_idx.sum() + 1e-8)
      loss_dict['num_cap_tokens'] = cap_idx.sum() / cap_idx.shape[0]
      loss_dict['num_loc_tokens'] = loc_idx.sum() / loc_idx.shape[0]
    loss = (loss * valid).sum() / (valid.sum() + 1e-8)
    if self.loc_loss_weight >= 0.0:
      loss = loss_dict['cap_loss'] + (
          loss_dict['loc_loss'] * self.loc_loss_weight)
    loss_dict['total_loss'] = loss
    return loss, loss_dict


class CaptioningModel(base_model.BaseModel):
  """Scenic Model Wrapper."""

  def get_dict_from_config(self):
    return dict(
        max_caption_length=self.config.model.get('max_caption_length', 40),
        begin_token_id=self.config.model.get('begin_token_id', 101),
        end_token_id=self.config.model.get('end_token_id', 102),
        vocab_size=self.config.model.get('vocab_size', 30522),
        label_smooth=self.config.model.get('label_smooth', 0.1),
        num_frames=self.config.model.get('num_frames', 0),
        with_temp_emb=self.config.model.get('with_temp_emb', True),
        frame_fuse_fn=self.config.model.get('frame_fuse_fn', 'concat'),
        pixel_mean=self.config.model.get('pixel_mean', GIT_PIXEL_MEAN),
        pixel_std=self.config.model.get('pixel_std', GIT_PIXEL_STD),
        backbone_name=self.config.model.get('backbone_name', 'git_vit'),
        backbone_args=self.config.model.get(
            'backbone_args', ml_collections.ConfigDict()),
        text_decoder_name=self.config.model.get('text_decoder_name', 'git'),
        text_decoder_args=self.config.model.get(
            'text_decoder_args', ml_collections.ConfigDict()),
        decode_method=self.config.model.get('decode_method', 'greedy'),
        decode_beam_size=self.config.model.get('decode_beam_size', 1),
        decode_brevity_penalty_alpha=self.config.model.get(
            'decode_brevity_penalty_alpha', 0.6),
        freeze_image_encoder=self.config.model.get(
            'freeze_image_encoder', False),
        num_pooled_tokens=self.config.model.get('num_pooled_tokens', -1),
        backbone_param_name=self.config.model.get(
            'backbone_param_name', 'image_encoder'),
        decode_feature_key=self.config.model.get(
            'decode_feature_key', 'visual_features'),
        project_layers_name=self.config.model.get(
            'project_layers_name', 'none'),
        project_layers_args=self.config.model.get(
            'project_layers_args', ml_collections.ConfigDict()),
        project_param_name=self.config.model.get(
            'project_param_name', 'project_layers'),
        per_frame_qformer=self.config.model.get(
            'per_frame_qformer', False),
        num_bins=self.config.model.get('num_bins', 100),
        show_densecap_loss=self.config.model.get(
            'show_densecap_loss', False),
        loc_loss_weight=self.config.model.get('loc_loss_weight', -1.0),
        ignore_empty_data=self.config.model.get('ignore_empty_data', False),
    )

  def build_flax_model(self):
    return CaptioningFlaxModel(**self.get_dict_from_config())

  def loss_function(self, outputs, batch):
    return self.flax_model.loss_function(outputs, batch)
