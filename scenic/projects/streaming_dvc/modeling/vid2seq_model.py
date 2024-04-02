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

"""Vid2Seq model.

Forked from
https://github.com/google-research/scenic/blob/main/scenic/projects/
vid2seq/models.py

Add decoding-point mechanism.
"""
import dataclasses
from typing import Any, Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from scenic.model_lib.base_models import base_model
from scenic.projects.baselines import vit
from scenic.projects.t5 import layers as t5_model
from scenic.projects.t5 import model as t5_pretrained
from t5x import decoding

beam_search = decoding.beam_search
temperature_sample = decoding.temperature_sample

Batch = Dict[str, jnp.ndarray]
PyTree = Any
SP_VOCAB_SIZE = 32128


class CatEncoder(nn.Module):
  """Concat ViT temporal encodings with T5 text encodings."""
  enc_type: str
  enc_config: ml_collections.ConfigDict
  embedder: nn.Module
  num_bins: int
  num_dense_outputs: int = -1
  num_dense_outputs_test: int = -1

  def setup(self):
    self.visual_encoder = vit.Encoder(
        mlp_dim=self.enc_config.get('dim', 2048),
        num_layers=self.enc_config.get('layers', 12),
        num_heads=self.enc_config.get('heads', 12),
        positional_embedding=self.enc_config.get('pos_embed', 'learned_1d'),
        dropout_rate=self.enc_config.get('dropout_rate', 0.),
        attention_dropout_rate=self.enc_config.get('dropout_rate', 0.),
        stochastic_depth=self.enc_config.get('stochastic_depth', 0.))
    enc_cfg = self.enc_config.get('pretrained_config', 't5_1_1_base')
    assert enc_cfg == 't5_1_1_base', enc_cfg
    t5_config = t5_pretrained.CONFIGS[enc_cfg]
    t5_config['dropout_rate'] = self.enc_config.get('t5_dropout_rate', 0.)
    t5_config['vocab_size'] = SP_VOCAB_SIZE + self.num_bins
    self.t5_encoder = t5_model.T5Encoder(
        **t5_config,
        shared_embedding=self.embedder,
        name='video_encoder')  # Actually ASR encoder.
    self.proj_dim = self.enc_config.get('proj_dim', 768)

  def __call__(
      self, features, encoder_input_tokens=None, checkpoint_inds=None,
      train=False):
    """Forward model.

    Args:
      features: (batch_size, num_tokens, dim)
      encoder_input_tokens: (batch_size, max_cap_len) for ASR or
          (batch_size, num_caps_per_image, max_cap_len) for prefix.
      checkpoint_inds: (batch_size, num_caps_per_image) or None
      train: bool
    Returns:
      if num_dense_outputs == -1:
        encoded: (batch_size, num_tot_tokens, dim)
        cat_mask: (batch_size, num_tot_tokens)
      else:
        encoded: (batch_size, num_dense_outputs, num_tot_tokens, dim)
        cat_mask: (batch_size, num_dense_outputs, num_tot_tokens)
    """
    visual_embeddings = self.encode_visual(features, checkpoint_inds, train)
    visual_mask = jnp.ones(visual_embeddings.shape[:-1]) > 0
    if encoder_input_tokens is not None:  # ASR or prefix
      text_feature = self.encode_text(encoder_input_tokens, train)
      cat_feature = jnp.concatenate([visual_embeddings, text_feature], axis=-2)
      cat_mask = jnp.concatenate(
          [visual_mask, encoder_input_tokens > 0], axis=-1)
    else:
      cat_feature = visual_embeddings
      cat_mask = visual_mask
    return {'encoded': cat_feature, 'mask': cat_mask}

  def encode_visual(self, features, checkpoint_inds, train=False):
    """Encode visual features.

    Args:
      features: (batch_size, num_tokens, dim)
      checkpoint_inds: (batch_size, num_caps_per_image) or None
      train: bool
    Returns:
      (batch_size, num_tot_tokens, dim) or
        (batch_size, num_dense_outputs, num_tot_tokens, dim)
    """
    if self.num_dense_outputs > 0:
      batch_size, num_frames, dim = features.shape
      num_tokens = num_frames
      # NOTE: currently we duplicate features to fill the full num_tokens at
      # each time stamp. For example, at time 2, we will fill the 100-tokens
      # feature as [0] * 50 + [1] * 50.
      # This is done as there is a subsequent transformer which expects a fixed
      # number (num_frames) of tokens.
      # TODO(zhouxy): a better option might be bilinear interpolation.
      inds = (0.5 + jnp.linspace(
          0, jnp.arange(num_frames),
          num_frames, endpoint=True,
          dtype=jnp.float32)).astype(jnp.int32)  # (num_frames, num_tokens)
      features_expanded = jnp.broadcast_to(
          features[:, None],
          (batch_size, num_frames, num_tokens, dim))
      # features_expanded[:, t] is now the resized features until time t.
      streaming_features_per_frame = jnp.take_along_axis(
          features_expanded,
          inds[None, :, :, None],
          axis=2)  # (batch_size, num_frames, num_tokens, dim)
      if train:
        num_dense_outputs = self.num_dense_outputs
        features = jnp.take_along_axis(
            streaming_features_per_frame,
            checkpoint_inds[:, :, None, None],
            axis=1)  # (batch_size, num_caps_per_image, num_tokens, dim)
      else:
        num_dense_outputs = self.num_dense_outputs_test
        checkpoint_stride = num_frames // num_dense_outputs
        features = streaming_features_per_frame[
            :, (jnp.arange(num_dense_outputs) + 1) * checkpoint_stride - 1]
      features = features.reshape(
          batch_size * num_dense_outputs, num_tokens, dim)

      visual_embeddings = self.visual_encoder(
          features, train=train,
      )  # (batch_size * num_dense_outputs, num_tokens, dim)

      visual_embeddings = visual_embeddings.reshape(
          batch_size, num_dense_outputs, num_tokens, dim)
    else:
      visual_embeddings = self.visual_encoder(
          features, train=train,
      )  # (batch_size, num_tokens, dim) or
    return visual_embeddings

  def encode_text(self, encoder_input_tokens, train=False):
    """Encode text.

    Args:
      encoder_input_tokens: (batch_size, max_cap_len) for ASR or
          (batch_size, num_caps_per_image, max_cap_len) for prefix.
      train: bool
    Returns:
      x: (batch_size, max_cap_len, dim) for ASR or
          (batch_size, num_caps_per_image, max_cap_len, dim) for prefix.
    """
    is_prefix = len(encoder_input_tokens.shape) == 3
    if is_prefix:  # prefix
      batch_size, num_caps_per_image, max_cap_len = encoder_input_tokens.shape
      # Reshape to match dimention for the text encoder
      encoder_input_tokens_flatten = encoder_input_tokens.reshape(
          batch_size * num_caps_per_image, max_cap_len)
      x = self.t5_encoder(
          encoder_input_tokens=encoder_input_tokens_flatten,
          enable_dropout=train)
      # Reshape back
      x = x.reshape(
          batch_size, num_caps_per_image, max_cap_len, self.proj_dim)
    else:  # ASR
      x = self.t5_encoder(
          encoder_input_tokens=encoder_input_tokens,
          enable_dropout=train)
    return x


class Vid2SeqDenseVideoCaptioningModule(nn.Module):
  """Dense video captioning module that encodes a video and generate tokens."""
  max_caption_length: int = 40
  begin_token_id: int = 0
  end_token_id: int = 1
  vocab_size: int = SP_VOCAB_SIZE
  label_smooth: float = 0.1
  label_smooth_bias: int = -1
  ignore_empty_data: bool = True
  num_bins: int = 100
  decode_method: str = 'beam'
  decode_beam_size: int = 4
  decode_brevity_penalty_alpha: float = 0.6
  decode_feature_key: str = 'visual_features'
  num_dense_outputs: int = -1
  num_dense_outputs_test: int = -1
  no_timestamp_in_context: bool = True
  normalize_early_timestamps: bool = True
  early_segments_as_context: bool = False
  remove_segments_from_wrong_checkpoint: bool = False
  copy_context: bool = False
  config: ml_collections.ConfigDict = dataclasses.field(
      default_factory=ml_collections.ConfigDict)

  def _get_encoder(self,
                   enc_type: str,
                   enc_config: ml_collections.ConfigDict,
                   embedder: Optional[nn.Module] = None,
                   num_bins: int = 0):
    assert enc_type == 'cat_encoder', enc_type
    encoder = CatEncoder(
        enc_type=enc_type,
        enc_config=enc_config,
        embedder=embedder,
        num_bins=num_bins,
        num_dense_outputs=self.num_dense_outputs,
        num_dense_outputs_test=self.num_dense_outputs_test,
    )
    return encoder

  def _get_decoder(self,
                   dec_type: str,
                   dec_config: ml_collections.ConfigDict,
                   num_bins: int):
    assert dec_type == 't5_decoder', dec_type
    t5_config = t5_pretrained.CONFIGS[dec_config.pretrained_config]
    t5_config['dropout_rate'] = dec_config.dropout_rate
    t5_config['logits_via_embedding'] = dec_config.logits_via_embedding
    t5_config['vocab_size'] = SP_VOCAB_SIZE + num_bins
    decoder_embedder = t5_model.t5_layers.Embed(
        num_embeddings=t5_config['vocab_size'],
        features=t5_config['emb_dim'],
        dtype=t5_config['dtype'],
        attend_dtype=jnp.float32,  # For logit training stability.
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=True,
        name='shared_decoder_token_embedder')
    decoder = t5_model.T5Decoder(
        **t5_config,
        shared_embedding=decoder_embedder,
        name='text_decoder')
    return (decoder_embedder, decoder)

  def setup(self):
    decoder_type = 't5_decoder'
    decoder_config = self.config.decoder.get(decoder_type)
    num_bins = self.num_bins  # self.config.decoder.get('num_bins')
    self.encoder_type = self.config.encoder.get('encoder_type')
    encoder_type = 'cat_encoder'
    encoder_config = self.config.encoder.get(encoder_type)
    self.embedder, self.decoder = self._get_decoder(
        decoder_type,
        decoder_config,
        num_bins)

    self.encoder = self._get_encoder(
        self.encoder_type,
        encoder_config,
        self.embedder,
        num_bins)

  def decode(
      self, text_tokens, encoded, encoder_mask, train=False):
    """Forward decoder.

    Args:
      text_tokens: (batch_size, max_caption_length)
      encoded: (batch_size, num_tot_tokens, dim)
      encoder_mask: (batch_size, num_tot_tokens)
      train: bool
    Returns:
      logits: (batch_size, num_tot_tokens, vocab_size).
    """
    decoder_target = jnp.concatenate(
        [text_tokens[:, 1:],
         jnp.zeros((text_tokens.shape[0], 1), dtype=jnp.int32)],
        axis=1)
    logits = self.decoder(
        encoded,
        encoder_mask,
        text_tokens,
        decoder_target,
        enable_dropout=train,
        decode=False)
    return logits

  def __call__(
      self,
      images=None,
      image_features=None,
      context_tokens=None,
      gt_text_tokens=None,
      checkpoint_inds=None,
      preprocess=True, train=False, debug=False):
    """Forward model.

    Args:
      images: must be None.
      image_features: (batch_size, num_frames, dim)
      context_tokens: (batch_size, num_caps_per_image, max_context_len) or None.
        The output text from previous decoding point. Note this can NOT be ASR.
      gt_text_tokens: (batch_size, num_caps_per_image, max_cap_len)
      checkpoint_inds: (batch_size, num_caps_per_image) or None
      preprocess: bool; unused.
      train: bool
      debug: bool
    Returns:
      if train == True, return
        'text_outputs': (text_batch_size, max_cap_len, vocab_size)
      if train == False, return
        'visual_features': (text_batch_size, feature_len, feature_dim)
        'begin_tokens': (batch_size, num_caps_per_image, max_cap_len)
        'context_tokens': (text_batch_size, max_cap_len)
        'text_outputs': (text_batch_size, max_cap_len, vocab_size)
    """
    del images
    assert image_features is not None
    del preprocess
    visual_features = self.encoder.encode_visual(
        image_features,
        checkpoint_inds=checkpoint_inds,
        train=train)  # (batch_size, num_tot_tokens, dim)
    # Here num_tot_tokens = num_frames + max_context_len,
    # when self.num_dense_outputs > 0, visual_features will be of shape
    # (batch_size, num_dense_outputs, num_tot_tokens, dim)

    text_tokens, encoded, context_tokens = (
        self.get_text_tokens_and_pad_visual_features(
            visual_features, gt_text_tokens, context_tokens))
    # text_tokens: (text_batch_size, max_cap_len)
    # encoded: (text_batch_size, num_tot_tokens, proj_dim)
    # context_tokens: (text_batch_size, max_context_len)
    # text_batch_size = batch_size * num_caps_per_image
    # when self.num_dense_outputs > 0, training shape is the same. For
    # evaluation, encoded will be in shape
    # (batch_size, num_dense_outputs, num_tot_tokens, dim)
    encoder_mask = jnp.ones(encoded.shape[:-1]) > 0

    if context_tokens is not None:
      context_features = self.encoder.encode_text(context_tokens, train=train)
      encoded = jnp.concatenate([encoded, context_features], axis=-2)
      encoder_mask = jnp.concatenate(
          [encoder_mask, context_tokens > 0], axis=-1)

    if train:
      text_outputs = self.decode(
          text_tokens,
          encoded,
          encoder_mask,
          train=train,
      )  # (text_batch_size, max_cap_len, vocab_size)
      ret = {'text_outputs': text_outputs}
    else:
      text_outputs = self.decode(
          text_tokens,
          encoded if self.num_dense_outputs < 0 else encoded[:, 0],
          encoder_mask if self.num_dense_outputs < 0 else encoder_mask[:, 0],
          train=train,
      )  # (text_batch_size, max_cap_len, vocab_size)
      ret = {
          'visual_features': encoded,
          'begin_tokens': text_tokens,
          'context_tokens': context_tokens,
          'text_outputs': text_outputs,
      }
    return ret

  def decode_text(
      self, text_tokens, visual_features,
      context_tokens=None, return_feat=False):
    """Forward one step in the auto-regressive decoding.

    Args:
      text_tokens: (batch_size, caption_length).
      visual_features: (batch_size, feature_length, feat_size).
      context_tokens: (batch_size, context_length) or None
      return_feat: bool; Unused, but kept to keep a consistent API with other
        models.
    Returns:
      output_logits: (batch_size, caption_length, vocab_size).
    """
    del return_feat
    encoder_mask = jnp.ones(visual_features.shape[:-1]) > 0
    # TODO(zhouxy): the behaviour of context_tokens can be optimized in
    # encoder-decoder architecture in all our codebase. Currently we rerun
    # encoder at every decoding step (if no xla optimization).
    # Also, we do not cache the decoder outputs either, which will slow down
    # generation of long sequences.
    if context_tokens is not None:
      context_features = self.encoder.encode_text(context_tokens, train=False)
      visual_features = jnp.concatenate(
          [visual_features, context_features], axis=-2)
      encoder_mask = jnp.concatenate(
          [encoder_mask, context_tokens > 0], axis=-1)
    return self.decode(text_tokens, visual_features, encoder_mask, train=False)

  def get_text_tokens_and_pad_visual_features(
      self, visual_features, gt_text_tokens, context_tokens=None):
    """Get inputs to the text decoder.

    In evaluation, we create the zero-padded text-token with the first token
    being BOS. In training, we handle multiple caption annotations for a
    video and repeat the visual_feature to align with that.

    This is mostly the same as the GIT model, except for the initialization
    behaviour noted below.

    Args:
      visual_features: (batch_size, num_tokens, dim) or
        (batch_size, num_dense_outputs, num_tokens, dim)
      gt_text_tokens: (batch_size, num_caps_per_image, max_cap_len) or None.
      context_tokens: (batch_size, num_caps_per_image, max_cap_len) or None.
    Returns:
      text_tokens: (text_batch_size, max_cap_len). text_batch_size = (batch_size
        * num_caps_per_image)
      visual_features: (text_batch_size, num_tokens, dim) or
        (batch_size, num_dense_outputs, num_tokens, dim)
      context_tokens: (text_batch_size, num_tokens) or None.
    """
    if gt_text_tokens is None:  # Evaluation, create BOS tokens.
      text_tokens = jnp.full(
          (visual_features.shape[0], self.max_caption_length),
          self.end_token_id, dtype=jnp.int32)  # (B, max_cap_len)
      text_tokens = text_tokens.at[:, 0].set(
          self.begin_token_id)  # (batch_size, max_cap_len)
      # NOTE: We don't delete context_tokens here, as we need it to initialize
      # the text encoder weights during initialization.
    else:  # Training
      batch_size, num_caps_per_image = gt_text_tokens.shape[:2]
      text_tokens = gt_text_tokens.reshape(
          batch_size * num_caps_per_image,
          gt_text_tokens.shape[2],
      )  # (batch_size, num_caps_per_image, max_cap_len)
      num_tokens, dim = visual_features.shape[-2:]
      if len(visual_features.shape) == 3:  # no dense_outputs
        visual_features = jnp.broadcast_to(
            visual_features[:, None],
            (batch_size, num_caps_per_image, num_tokens, dim))
      visual_features = visual_features.reshape(
          batch_size * num_caps_per_image, num_tokens, dim)
      if context_tokens is not None:
        context_tokens = context_tokens.reshape(
            batch_size * num_caps_per_image, context_tokens.shape[2])
    return text_tokens, visual_features, context_tokens

  def loss_function(self, outputs, batch):
    """Text loss with label smoothing.

    This is exactly the same as traditional captioning.

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
    gt = gt * (1. - self.label_smooth) + (
        1. - gt) * self.label_smooth / (
            self.vocab_size + self.label_smooth_bias)
    # loss:  (text_batch_size, max_cap_len - 1)
    gt = jax.lax.stop_gradient(gt)
    loss = optax.softmax_cross_entropy(text_outputs, gt)
    loss_dict = {}
    loss = (loss * valid).sum() / (valid.sum() + 1e-8)
    loss_dict['total_loss'] = loss
    return loss, loss_dict


class Vid2SeqModel(base_model.BaseModel):
  """Scenic Model Wrapper."""

  def get_dict_from_config(self):
    return dict(
        max_caption_length=self.config.model.get('max_caption_length', 256),
        begin_token_id=self.config.model.get('begin_token_id', 0),
        end_token_id=self.config.model.get('end_token_id', 1),
        vocab_size=self.config.model.get('vocab_size', SP_VOCAB_SIZE),
        label_smooth=self.config.model.get('label_smooth', 0.1),
        label_smooth_bias=self.config.model.get('label_smooth_bias', -1),
        ignore_empty_data=self.config.model.get('ignore_empty_data', True),
        num_bins=self.config.model.get('num_bins', 100),
        decode_method=self.config.model.get('decode_method', 'beam'),
        decode_beam_size=self.config.model.get('decode_beam_size', 4),
        decode_brevity_penalty_alpha=self.config.model.get(
            'decode_brevity_penalty_alpha', 0.6),
        decode_feature_key=self.config.model.get(
            'decode_feature_key', 'visual_features'),
        num_dense_outputs=self.config.model.get('num_dense_outputs', -1),
        num_dense_outputs_test=self.config.model.get(
            'num_dense_outputs_test', -1),
        no_timestamp_in_context=self.config.model.get(
            'no_timestamp_in_context', True),
        normalize_early_timestamps=self.config.model.get(
            'normalize_early_timestamps', True),
        early_segments_as_context=self.config.model.get(
            'early_segments_as_context', False),
        config=self.config.model,
    )

  def build_flax_model(self):
    return Vid2SeqDenseVideoCaptioningModule(**self.get_dict_from_config())

  def loss_function(self, outputs, batch):
    return self.flax_model.loss_function(outputs, batch)
