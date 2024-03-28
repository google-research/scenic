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

"""Clip4clip model for video and text contrastive learning."""
from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models import base_model
from scenic.projects.baselines.clip import layers as clip_layers
from scenic.projects.baselines.clip import model as clip
from scenic.projects.verbs_in_action import losses
from scenic.projects.verbs_in_action import utils


class TextEncoder(nn.Module):
  """Encoder model for text.

  Attributes:
    config: The CLIP text tower config.
  """
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, tokens, *, train: bool, debug: bool = False):
    if tokens is None:
      return tokens
    tokens = tokens.reshape((-1,) + tokens.shape[2:])
    text_code = clip_layers.TextEncoder(name='TextTower',
                                        **self.config)(tokens)
    text_code = jnp.expand_dims(text_code, 1)
    return text_code


class VideoEncoder(nn.Module):
  """Encoder model for video.

  Attributes:
    config: The config to create the image model from.
    temporal_agg: How to agregate embeddings of the different frames.
  """
  config: ml_collections.ConfigDict
  temporal_agg: str

  @nn.compact
  def __call__(self, x_rgb, *, train: bool, debug: bool = False):
    if x_rgb is None:
      return x_rgb
    video_embedding = Image2VideoEncoder(self.config, self.temporal_agg)(
        x=x_rgb, train=train, debug=debug)
    return video_embedding


class Image2VideoEncoder(nn.Module):
  """Use a clip image encoder to encode video frames as in CLIP4CLIP.

  Attributes:
    config: The config to create the image model from.
    temporal_agg: How to agregate embeddings of the different frames.
  """

  config: ml_collections.ConfigDict
  temporal_agg: str
  transformer_aggregation_num_layer: int = 4

  @nn.compact
  def __call__(self, x, *, train: bool, debug: bool = False):
     # Reshape and normalise video frames
    all_frames = x.reshape((-1,) + x.shape[-3:])
    all_frames = clip.normalize_image(all_frames)
    image_encoder = clip_layers.VisionTransformer(name='ImageTower',
                                                  **self.config)
    video_embeddings, _ = image_encoder(x=all_frames)
    video_embeddings = video_embeddings.reshape(x.shape[:-3] + (-1,))

    # Mean pooling across frame features.
    if self.temporal_agg == 'meanpool':
      video_embeddings = jnp.mean(video_embeddings, axis=1)

    # Temporal aggregation with a transformer across frame features.
    elif self.temporal_agg == 'transformer':
      feature_dim = video_embeddings.shape[-1]
      positional_embedding = self.param(
          'seqTrans_positional_embedding', jax.nn.initializers.zeros,
          (video_embeddings.shape[1], feature_dim))
      frame_embs = video_embeddings + positional_embedding[None]
      frame_embs = clip.layers.Transformer(
          feature_dim, self.transformer_aggregation_num_layer,
          feature_dim // 64, name='seqTrans_transformer')(frame_embs)
      video_embeddings += frame_embs  # residual
      # seqTrans Clip4clip l2-normalizes *before* and after mean pooling.
      video_embeddings /= jnp.linalg.norm(
          video_embeddings, axis=-1, keepdims=True) + 1e-8
      video_embeddings = jnp.mean(video_embeddings, axis=1)
    return video_embeddings


class VideoAndTextModule(nn.Module):
  """Dual encoder model for text and video.

  Attributes:
    clip_vision_config: The config to create the video tower from.
    clip_text_config: The config to create the text tower from.
    temporal_agg: How to agregate embeddings of the different frames.
  """
  clip_vision_config: ml_collections.ConfigDict
  clip_text_config: ml_collections.ConfigDict
  temporal_agg: str

  @nn.compact
  def __call__(self, x_rgb, text_tokens, *, train: bool = True,
               debug: bool = False):
    # Video encoding.
    video_encoder = VideoEncoder(
        self.clip_vision_config, self.temporal_agg, name='video_encoder')
    video_emb = None
    if x_rgb is not None:
      video_emb = video_encoder(x_rgb=x_rgb, train=train, debug=debug)
      video_emb /= jnp.linalg.norm(video_emb, axis=-1, keepdims=True) + 1e-8

    # Text encoding.
    text_encoder = TextEncoder(self.clip_text_config, name='text_encoder')
    text_emb = None
    if text_tokens is not None:
      text_emb = text_encoder(text_tokens, train=train, debug=debug)
      text_emb /= jnp.linalg.norm(text_emb, axis=-1, keepdims=True) + 1e-8
    return video_emb, text_emb

  def loss_function(self, encoded_video: jnp.ndarray,
                    encoded_text: jnp.ndarray,
                    batch: Any,
                    config: ml_collections.ConfigDict,
                    encoded_verbs: Optional[jnp.ndarray] = None) -> float:
    """Returns the loss for VFC training."""

    verb_hard_negatives = config.get('verb_hard_negatives', False)
    # Training with hard negatives ...
    if verb_hard_negatives:
      batch_mask_text = jax.lax.all_gather(batch['text_mask'], 'batch')
      batch_mask_text = batch_mask_text.reshape(
          (-1,) + batch_mask_text.shape[2:])
      loss = losses.verb_hard_neg_nce(
          encoded_video, encoded_text, batch_mask_text, config.temperature,
          config.get('v2t_weight', 1.0), config.get('t2v_weight', 1.0),
          config.get('beta_hnnce', 0.))
    # ... or not. In this case, this is the baseline.
    else:
      loss = losses.baseline_nce(
          encoded_video, encoded_text, config.temperature,
          config.get('v2t_weight', 1.0), config.get('t2v_weight', 1.0),
          config.get('beta_hnnce', 0.))

    # Second, we (optionally) compute the verb-phrase loss.
    if encoded_verbs is not None:
      batch_mask_verb = jax.lax.all_gather(batch['verb_mask'], 'batch')
      batch_mask_verb = batch_mask_verb.reshape(
          (-1,) + batch_mask_verb.shape[2:])
      verb_phrase_loss = losses.verb_phrase_nce(
          encoded_video, encoded_verbs, batch_mask_verb, config.temperature)
      loss += config.get('verb_phrase_loss_weight') * verb_phrase_loss
    return jnp.mean(loss)


class VideoAndTextModel(base_model.BaseModel):
  """Video Text Dual Transformer model as defined in CLIP4CLIP."""

  def build_flax_model(self) -> nn.Module:
    clip_vision_config = utils.get_vit_clip_config(
        self.config.model.clip_version)
    clip_text_config = utils.get_text_clip_config(
        self.config.model.clip_version)
    return VideoAndTextModule(
        clip_vision_config=clip_vision_config,
        clip_text_config=clip_text_config,
        temporal_agg=self.config.model.temporal_agg)

  def get_metrics_fn(self, split: Optional[str] = None):
    pass
