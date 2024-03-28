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

"""GIT caption model."""

import dataclasses
from typing import Any

# from absl import logging
from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from scenic.model_lib.base_models import base_model
from scenic.projects.gerald.models import git_vit
from scenic.projects.gerald.models import text_decoder

GIT_PIXEL_MEAN = (0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255)
GIT_PIXEL_STD = (0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255)
NEG_INF = float('-inf')


class WordAndPositionalEmbedding(nn.Module):
  """GRiT embedding layer."""
  vocab_size: int = 30522
  hidden_size: int = 768
  max_caption_length: int = 1024
  dropout_prob: float = 0.1

  def setup(self):
    self.words = nn.Embed(
        self.vocab_size, self.hidden_size,
        embedding_init=nn.initializers.normal(stddev=0.02),
        name='words')

  @nn.compact
  def __call__(self, x, train=False):
    """forward embedding.

    Args:
      x: (batch_size, caption_length).
      train: bool.
    Returns:
      embeddings: (batch_size, max_caption_length, hidden_size).
    """
    bs = x.shape[0]
    position_indices = jnp.tile(jnp.arange(self.max_caption_length)[None],
                                [bs, 1])
    word_embeddings = self.words(x)
    position_embeddings = nn.Embed(
        self.max_caption_length, self.hidden_size,
        embedding_init=nn.initializers.normal(stddev=0.02),
        name='positions')(position_indices)
    embeddings = nn.LayerNorm(epsilon=1e-8, name='layer_norm')(
        word_embeddings + position_embeddings[:, :x.shape[1]]
    )  # eps checked.
    embeddings = nn.Dropout(self.dropout_prob, name='dropout')(
        embeddings, deterministic=not train)
    return embeddings


class TransformerDecoder(nn.Module):
  """Transformer Decoder Textual Head of GIT."""
  ger_vocab_size: int = 30522
  ger_max_code_length: int = 5
  text_vocab_size: int = 30522
  max_context_length: int = 1024
  dropout_prob: float = 0.1
  hidden_size: int = 768
  num_heads: int = 12
  num_hidden_layers: int = 6
  stochastic_depth: float = 0.0
  attention_dropout: float = 0.1

  def setup(self):
    self.embedding = WordAndPositionalEmbedding(
        vocab_size=self.text_vocab_size,
        hidden_size=self.hidden_size,
        max_caption_length=self.max_context_length,
        dropout_prob=self.dropout_prob,
        name='embedding')
    if self.ger_vocab_size != self.text_vocab_size:
      # If text and GER code vocabulary sizes do not match, we use separate
      # embedding layers for them.
      self.separate_ger_embedding = WordAndPositionalEmbedding(
          vocab_size=self.ger_vocab_size,
          hidden_size=self.hidden_size,
          max_caption_length=self.ger_max_caption_length,
          dropout_prob=self.dropout_prob,
          name='separate_ger_embedding')

  def concate_context_tokens_to_visual(
      self, visual_features, context_tokens, train=False):
    """Concatenate context tokens (e.g., input question) to visual tokens.

    Args:
      visual_features: (batch_size, feature_length, object_feat_size).
      context_tokens: (batch_size, context_length)
      train: bool
    Returns:
      visual_features: (batch_size, feature_length+context_length, hidden_size)
      feat_valid_mask: (batch_size, feature_length+context_length): bool array.
        if the visual_features is padded (to handle different context_lengths).
    """
    feat_valid_mask = jnp.ones(
        (visual_features.shape[:2]),
        dtype=bool)  # (text_bs, num_tokens)
    context_tokens = context_tokens.reshape(
        -1, context_tokens.shape[-1])  # (text_bs, num_context_tokens)
    context_features = self.embedding(context_tokens, train=train)

    # Note context_tokens do not have BOS or EOS. All padded tokens are 0.
    context_valid_mask = context_tokens > 0  # (text_bs, num_context_tokens)
    feat_valid_mask = jnp.concatenate(
        [feat_valid_mask, context_valid_mask],
        axis=1)  # (text_bs, num_tot_tokens)
    visual_features = jnp.concatenate(
        [visual_features, context_features],
        axis=1)  # (text_bs, num_tot_tokens, dim)
    return visual_features, feat_valid_mask

  @nn.compact
  def __call__(
      self, ger_tokens, visual_features,
      context_tokens=None, train=False,):
    """Generate logits of a single word.

    Args:
      ger_tokens: (batch_size, code_length).
      visual_features: (batch_size, feature_length, feat_size).
      context_tokens:  (batch_size, context_length).
      train: bool.
    Returns:
      #output_logits: (batch_size, caption_length, vocab_size).
      #trans_out: (batch_size, caption_length, hidden_size) or
      #  (batch_size, feature_length + caption_length, hidden_size) when
      #  return_visual_feature is True.
    """
    x = nn.Dense(
        self.hidden_size, name='visual_projection.0',
        kernel_init=nn.initializers.normal(stddev=0.02))(
            visual_features)  # (batch_size, feature_length, hidden_size)
    x = nn.LayerNorm(epsilon=1e-5, name='visual_projection.1')(x)

    memory_key_padding_mask = None
    if context_tokens is not None:
      x, hidden_valid_mask = self.concate_context_tokens_to_visual(
          x, context_tokens, train=train)
      memory_key_padding_mask = ~hidden_valid_mask
    embedding_fn = self.embedding
    if self.ger_vocab_size != self.text_vocab_size:
      embedding_fn = self.separate_ger_embedding
    code_embeddings = embedding_fn(ger_tokens, train=train)
    uni_mask_zero_neg = text_decoder.generate_future_mask(ger_tokens.shape[1])
    trans_out = text_decoder.BertEncoderAsDecoder(
        num_hidden_layers=self.num_hidden_layers,
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        name='transformer')(
            code_embeddings, x,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_mask=uni_mask_zero_neg, train=train,
        )

    # Decoded Logits
    output_logits = nn.Dense(
        self.ger_vocab_size,
        kernel_init=nn.initializers.normal(stddev=0.02),
        name='output')(
            trans_out)  # (batch_size, code_length, vocab_size)
    return output_logits


class GERFlaxModel(nn.Module):
  """Inspired from GIT captioning model."""
  ger_vocab_size: int = 30522  # size of BertTokenizer
  ger_max_code_length: int = 5
  ger_begin_token_id: int = 101  # tokenizer.cls_token_id == 101
  ger_end_token_id: int = 102  # tokenizer.sep_token_id == 102
  max_context_length: int = 40  # the context is the input question
  text_vocab_size: int = 30522  # size of BertTokenizer
  text_begin_token_id: int = 101  # tokenizer.cls_token_id == 101
  text_end_token_id: int = 102  # tokenizer.sep_token_id == 102
  label_smooth: float = 0.1
  backbone_args: ml_collections.ConfigDict = dataclasses.field(
      default_factory=ml_collections.ConfigDict)
  pixel_mean: Any = GIT_PIXEL_MEAN
  pixel_std: Any = GIT_PIXEL_STD
  dropout_prob: float = 0.1

  def setup(self):
    self.image_encoder = git_vit.ViT(**self.backbone_args, name='image_encoder')
    self.decoder = TransformerDecoder(
        ger_vocab_size=self.ger_vocab_size,
        ger_max_code_length=self.ger_max_code_length,
        text_vocab_size=self.text_vocab_size,
        dropout_prob=self.dropout_prob,
        name='textual')

  @nn.compact
  def __call__(
      self, images, context_text_tokens=None, code_tokens=None,
      preprocess=True, train=False, debug=False):
    """Forward GIT model used for GER."""
    del debug
    if preprocess:
      images = self.preprocess(images)
    visual_features = self.image_encoder(images, train=train)  # (B, hw, D)
    visual_features = visual_features.reshape(
        visual_features.shape[0], -1, visual_features.shape[-1],
    )  # (B, hw, D)
    if code_tokens is None:
      code_tokens = jnp.full(
          (visual_features.shape[0],
           self.ger_max_code_length), self.ger_end_token_id, dtype=jnp.int32)
      code_tokens = code_tokens.at[:, 0].set(self.ger_begin_token_id)
      if context_text_tokens is None and self.max_context_length:
        context_text_tokens = jnp.full(
            (visual_features.shape[0], self.max_context_length),
            self.text_end_token_id, dtype=jnp.int32)  # (B, max_cap_len)
    else:
      batch_size = code_tokens.shape[0]
      visual_features = jnp.broadcast_to(
          visual_features[:, None],
          (batch_size, 1,) + visual_features.shape[1:],
      ).reshape((batch_size,) + visual_features.shape[1:])
      if context_text_tokens is not None:
        context_text_tokens = jnp.broadcast_to(
            context_text_tokens[:, None],
            (batch_size, 1,) + context_text_tokens.shape[1:],
        ).reshape((batch_size,) + context_text_tokens.shape[1:])
    outputs = self.decoder(
        code_tokens,
        visual_features,
        context_tokens=context_text_tokens,
        train=train,
    )  # (text_batch_size, max_code_len, vocab_size)
    if train:
      res = {'outputs': outputs}
    else:
      res = {'visual_features': visual_features, 'outputs': outputs,
             'begin_tokens': code_tokens}
      if context_text_tokens is not None:
        res['context_tokens'] = context_text_tokens
    return res

  def decode_text(self, code_tokens, visual_features, context_tokens=None):
    """Generate logits of a single token.

    Args:
      code_tokens: (batch_size, caption_length).
      visual_features: (batch_size, feature_length, feat_size).
      context_tokens: (batch_size, context_length).
    Returns:
      output_logits: (batch_size, caption_length, vocab_size).
    """
    return self.decoder(
        code_tokens, visual_features,
        context_tokens=context_tokens, train=False)

  def preprocess(self, inputs):
    """Proprocess images. Normalize pixels for non-padded pixels."""
    mean = jnp.asarray(self.pixel_mean, dtype=jnp.float32).reshape(1, 1, 1, 3)
    std = jnp.asarray(self.pixel_std, dtype=jnp.float32).reshape(1, 1, 1, 3)
    inputs = (inputs - mean) / std
    return inputs

  def loss_function(self, outputs, batch):
    """Next code token prediction loss with label smoothing."""
    outputs = outputs['outputs']
    vocab_size = outputs.shape[-1]
    gt_code = batch['code_tokens']
    outputs = outputs[:, :-1]  # Move GT one token to the right.
    # We don't want to predict a EOS from a EOS.
    valid = (gt_code != self.ger_end_token_id).astype(
        jnp.float32)[:, :-1]
    gt_code = gt_code[:, 1:]  # No need to predict BOS
    gt = jax.nn.one_hot(gt_code, vocab_size)
    # customized label smoothing following GRiT
    #   https://github.com/JialianW/GRiT/blob/master/grit/modeling/text/
    #   text_decoder.py#L668
    gt = gt * (1. - self.label_smooth) + (
        1. - gt) * self.label_smooth / (vocab_size - 1)
    gt = jax.lax.stop_gradient(gt)
    loss = optax.softmax_cross_entropy(outputs, gt)
    loss = (loss * valid[:, :]).sum() / (valid.sum() + 1e-8)

    preds = jnp.argmax(outputs, axis=-1)
    targets = jnp.argmax(gt, axis=-1)
    correct = jnp.equal(preds, targets)
    correct = (correct * valid[:, :]).sum() / (valid.sum() + 1e-8)
    return loss, {'total_loss': loss, 'accuracy': correct}


class GERModel(base_model.BaseModel):
  """Scenic Model Wrapper."""

  def get_dict_from_config(self):
    return dict(
        ger_vocab_size=self.config.get('vocab_size', 30520) + 2,
        ger_max_code_length=self.config.get('code_length', 4) + 1,
        ger_end_token_id=self.config.get('ger_eos', 102),
        ger_begin_token_id=self.config.get('ger_bos', 101),
        max_context_length=self.config.dataset_configs.get(
            'max_context_tokens', 40),
        text_begin_token_id={
            'bert': 101, 't5': 0
            }[self.config.dataset_configs.get('tokenizer_type', 'bert')],
        text_end_token_id={
            'bert': 102, 't5': 1
            }[self.config.dataset_configs.get('tokenizer_type', 'bert')],
        text_vocab_size={
            'bert': 30522, 't5': 32100
            }[self.config.dataset_configs.get('tokenizer_type', 'bert')],
        backbone_args=self.config.model.get(
            'backbone_args', ml_collections.ConfigDict()),
        label_smooth=self.config.model.get('label_smooth', 0.1),
        pixel_mean=self.config.model.get('pixel_mean', GIT_PIXEL_MEAN),
        pixel_std=self.config.model.get('pixel_std', GIT_PIXEL_STD),
        dropout_prob=self.config.model.get('dropout_prob', 0.1),
    )

  def build_flax_model(self):
    return GERFlaxModel(**self.get_dict_from_config())

  def loss_function(self, outputs, batch):
    return self.flax_model.loss_function(outputs, batch)
