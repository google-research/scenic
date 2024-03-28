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

"""Auto-regressive text decoder in GIT paper.

GIT: A Generative Image-to-text Transformer for Vision and Language. Wang et al.

arXiv: https://arxiv.org/abs/2205.14100

reference torch implementation:
https://github.com/microsoft/GenerativeImage2Text/blob/main/
generativeimage2text/layers/decoder.py

"""

from flax import linen as nn
import jax
import jax.numpy as jnp
from scenic.projects.pixel_llm.modeling import utils as pllm_utils

NEG_INF = float('-inf')


class BertSelfAttention(nn.Module):
  """Bert layer self attention."""

  num_heads: int = 12
  hidden_size: int = 768
  attention_dropout: float = 0.1

  @nn.compact
  def __call__(
      self, input_tensor, attention_mask, train=False):
    # input_tensor: (batch_size, tot_len, hidden_size)
    # attention_mask: (1, 1, tot_len, tot_len): NEG_INF to mask entry out.
    q = nn.Dense(
        self.hidden_size,
        kernel_init=nn.initializers.normal(stddev=0.02),
        name='query')(input_tensor)
    k = nn.Dense(
        self.hidden_size,
        kernel_init=nn.initializers.normal(stddev=0.02),
        name='key')(input_tensor)
    v = nn.Dense(
        self.hidden_size,
        kernel_init=nn.initializers.normal(stddev=0.02),
        name='value')(input_tensor)
    # TODO(zhouxy): implement decoding cache here.

    head_dim = self.hidden_size // self.num_heads
    transpose = lambda x: x.reshape(  # pylint: disable=g-long-lambda
        x.shape[0], x.shape[1], self.num_heads, head_dim).transpose(0, 2, 1, 3)
    q = transpose(q)
    k = transpose(k)
    v = transpose(v)  # (batch_size, num_heads, tot_len, head_dim)
    attention_scores = (q * (head_dim ** -0.5)) @ k.transpose(
        0, 1, 3, 2)  # (batch_size, num_heads, tot_len, tot_len)
    attention_scores = attention_scores + attention_mask
    attention_scores = jax.nn.softmax(attention_scores, axis=-1)
    attention_scores = nn.Dropout(self.attention_dropout)(
        attention_scores, deterministic=not train)
    out = (attention_scores @ v).transpose(0, 2, 1, 3).reshape(
        v.shape[0], v.shape[2], self.hidden_size)
    return out


class BertSelfOutput(nn.Module):
  """Bert layer self output."""

  hidden_size: int = 768
  hidden_dropout: float = 0.1

  @nn.compact
  def __call__(self, hidden_states, input_tensor, train=False):
    hidden_states = nn.Dense(
        self.hidden_size,
        kernel_init=nn.initializers.normal(stddev=0.02),
        name='dense')(hidden_states)
    hidden_states = nn.Dropout(self.hidden_dropout)(
        hidden_states, deterministic=not train)
    hidden_states = hidden_states + input_tensor
    hidden_states = nn.LayerNorm(
        epsilon=1e-5, name='LayerNorm')(hidden_states)
    return hidden_states


class BertAttention(nn.Module):
  """Bert layer attention."""
  hidden_size: int = 768
  num_heads: int = 12

  @nn.compact
  def __call__(
      self, input_tensor, attention_mask, train=False):
    self_outputs = BertSelfAttention(
        num_heads=self.num_heads, hidden_size=self.hidden_size, name='self')(
            input_tensor, attention_mask, train=train,
        )  # (batch_size, tot_len, hidden_size)
    attention_output = BertSelfOutput(
        hidden_size=self.hidden_size, name='output')(
            self_outputs, input_tensor, train=train,
        )  # (batch_size, tot_len, hidden_size)
    return attention_output


class BertIntermediate(nn.Module):
  """Bert layer intermediate."""

  intermediate_size: int = 768 * 4

  @nn.compact
  def __call__(
      self, hidden_states, train=False):
    hidden_states = nn.Dense(
        self.intermediate_size,
        kernel_init=nn.initializers.normal(stddev=0.02),
        name='dense')(hidden_states)
    hidden_states = nn.gelu(hidden_states, approximate=False)
    return hidden_states


class BertOutput(nn.Module):
  """Bert layer output."""

  hidden_size: int = 768
  hidden_dropout: float = 0.1

  @nn.compact
  def __call__(
      self, hidden_states, input_tensor, train=False):
    hidden_states = nn.Dense(
        self.hidden_size,
        kernel_init=nn.initializers.normal(stddev=0.02),
        name='dense')(hidden_states)
    hidden_states = nn.Dropout(self.hidden_dropout)(
        hidden_states, deterministic=not train)
    hidden_states = hidden_states + input_tensor
    hidden_states = nn.LayerNorm(
        epsilon=1e-12, name='LayerNorm')(
            hidden_states)  # eps following official implementation.
    return hidden_states


class BertLayer(nn.Module):
  """GIT encoder Layer."""
  hidden_size: int = 768
  num_heads: int = 12

  @nn.compact
  def __call__(
      self, hidden_states, attention_mask, train=False):
    """Forward layer.

    Args:
      hidden_states: (batch_size, tot_len, hidden_size).
      attention_mask: (1, 1, tot_len, tot_len).
      train: bool.
    Returns:
      hidden_states: (batch_size, tot_len, hidden_size).
    """
    attention_outputs = BertAttention(
        num_heads=self.num_heads, hidden_size=self.hidden_size,
        name='attention')(
            hidden_states, attention_mask, train=train,
        )  # (batch_size, tot_len, hidden_size)
    intermediate_output = BertIntermediate(
        intermediate_size=self.hidden_size * 4, name='intermediate')(
            attention_outputs, train=train,
        )  # (batch_size, tot_len, intermediate_size)
    layer_output = BertOutput(hidden_size=self.hidden_size, name='output')(
        intermediate_output, attention_outputs, train=train,
    )  # (batch_size, tot_len, hidden_size)
    return layer_output


class BertEncoder(nn.Module):
  """GIT Encoder."""
  num_hidden_layers: int = 6
  hidden_size: int = 768
  num_heads: int = 12

  @nn.compact
  def __call__(
      self, hidden_states, attention_mask, train=False):
    """forward encoder.

    Args:
      hidden_states: (batch_size, tot_len, hidden_size).
      attention_mask: (1, 1, tot_len, tot_len).
      train: bool.
    Returns:
      hidden_states: (batch_size, tot_len, hidden_size).
    """
    all_hidden_states = [hidden_states]
    for i in range(self.num_hidden_layers):
      hidden_states = BertLayer(
          hidden_size=self.hidden_size, num_heads=self.num_heads,
          name=f'layer.{i}')(
              hidden_states, attention_mask, train=train)
      all_hidden_states.append(hidden_states)
    return hidden_states, all_hidden_states


class BertEncoderAsDecoder(nn.Module):
  """GIT Decoder."""
  num_hidden_layers: int = 6
  hidden_size: int = 768
  num_heads: int = 12

  @nn.compact
  def __call__(
      self, tgt, memory, tgt_mask=None,
      memory_key_padding_mask=None, train=False, return_visual_feature=False):
    """forward transformer.

    Args:
      tgt: (batch_size, cap_len, hidden_size)
      memory: (batch_size, feat_len, hidden_size)
      tgt_mask: (cap_len, cap_len)
      memory_key_padding_mask: (batch_size, feat_len). Padded is 1, valid is 0.
      train: bool
      return_visual_feature: bool
    Returns:
      result:  (batch_size, cap_len, hidden_size)
    """
    cap_len = tgt.shape[1]
    feat_len = memory.shape[1]
    hidden_states = jnp.concatenate(
        [memory, tgt], axis=1
    )  # (batch_size, feat_len + cap_len, hidden_size)
    top_left = jnp.zeros((feat_len, feat_len), dtype=jnp.float32)
    top_right = jnp.full((feat_len, cap_len), NEG_INF, dtype=jnp.float32)
    bottom_left = jnp.zeros((cap_len, feat_len), dtype=jnp.float32)
    left = jnp.concatenate([top_left, bottom_left], axis=0)
    right = jnp.concatenate([top_right, tgt_mask], axis=0)

    full_attention_mask = jnp.concatenate(
        [left, right],
        axis=1)[None]  # (1, feat_len + cap_len, feat_len + cap_len)
    if memory_key_padding_mask is None:
      memory_key_padding_mask = jnp.full(
          (1, memory.shape[1]), False, dtype=bool,
      )  # (1, feat_len)
    else:
      full_attention_mask = jnp.broadcast_to(
          full_attention_mask,
          (memory_key_padding_mask.shape[0],
           full_attention_mask.shape[1], full_attention_mask.shape[2]))
    zero_negative_infinity = jnp.zeros_like(
        memory_key_padding_mask, dtype=tgt.dtype)  # (1, feat_len)
    zero_negative_infinity = jnp.where(
        memory_key_padding_mask, NEG_INF, zero_negative_infinity)
    origin_left = full_attention_mask[:, :, :feat_len]
    update = zero_negative_infinity[:, None, :]  # (1, 1, feat_len)
    full_attention_mask = jnp.concatenate(
        [origin_left + update, full_attention_mask[:, :, feat_len:]],
        axis=2)
    full_attention_mask = full_attention_mask[
        :, None, :, :]  # (1, 1, feat_len + cap_len, feat_len + cap_len)

    result, all_hidden_states = BertEncoder(
        num_hidden_layers=self.num_hidden_layers,
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        name='encoder')(
            hidden_states=hidden_states,
            attention_mask=full_attention_mask,
            train=train,
        )  # (batch_size, feat_len + cap_len, hidden_size)
    if not return_visual_feature:
      result = result[:, feat_len:]  #  (batch_size, cap_len, hidden_size)
      all_hidden_states = [
          hidden_states[:, feat_len:] for hidden_states in all_hidden_states]
    return result, all_hidden_states


class WordAndPositionalEmbedding(nn.Module):
  """GRiT embedding layer."""
  vocab_size: int = 30522
  hidden_size: int = 768
  max_caption_length: int = 1024
  dropout_prob: float = 0.1

  @nn.compact
  def __call__(self, x, train=False):
    """forward embedding.

    Args:
      x: (batch_size, caption_length).
      train: bool.
    Returns:
      embeddings: (batch_size, max_caption_length, hidden_size).
    """
    position_indices = jnp.arange(
        self.max_caption_length)[None]  # 1 x  max_caption_length
    word_embeddings = nn.Embed(
        self.vocab_size, self.hidden_size,
        embedding_init=nn.initializers.normal(stddev=0.02),
        name='words')(x)
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


class TransformerDecoderTextualHead(nn.Module):
  """TransformerDecoderTextualHead of GIT."""
  vocab_size: int = 30522
  hidden_size: int = 768
  num_heads: int = 12
  max_caption_length: int = 1024
  num_hidden_layers: int = 6
  out_feat_fuse_method: str = ''

  def setup(self):
    self.embedding = WordAndPositionalEmbedding(
        vocab_size=self.vocab_size,
        hidden_size=self.hidden_size,
        max_caption_length=self.max_caption_length,
        name='embedding')

  def concate_context_tokens_to_visual(
      self, visual_features, context_tokens, train=False):
    """Concatenate context tokens (e.g., questions in QA) to visual tokens.

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
      self, text_tokens, visual_features,
      context_tokens=None, train=False,
      return_feat=False, return_visual_feature=False,
      return_logit_and_feat=False):
    """Generate logits of a single word.

    Args:
      text_tokens: (batch_size, caption_length).
      visual_features: (batch_size, feature_length, feat_size).
      context_tokens:  (batch_size, context_length).
      train: bool.
      return_feat: bool. If true, return the feature before vocabulary.
      return_visual_feature: bool. If true, in addition return the outputs from
        visual features.
      return_logit_and_feat: bool
    Returns:
      output_logits: (batch_size, caption_length, vocab_size).
      trans_out: (batch_size, caption_length, hidden_size) or
        (batch_size, feature_length + caption_length, hidden_size) when
        return_visual_feature is True.
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

    text_embeddings = self.embedding(
        text_tokens, train=train,
    )  # (batch_size, max_caption_length, hidden_size)

    caption_length = text_tokens.shape[1]
    uni_mask_zero_neg = self._generate_future_mask(
        caption_length)  # (caption_length, caption_length)
    trans_out, all_hidden_states = BertEncoderAsDecoder(
        num_hidden_layers=self.num_hidden_layers,
        hidden_size=self.hidden_size,
        num_heads=self.num_heads,
        name='transformer')(
            text_embeddings, x,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_mask=uni_mask_zero_neg, train=train,
            return_visual_feature=return_visual_feature,
        )  # (batch_size, caption_length, hidden_size)
    if self.out_feat_fuse_method:
      output_feature = pllm_utils.fuse_out_feat(
          all_hidden_states, self.out_feat_fuse_method)
    else:
      output_feature = trans_out
    if return_feat:
      return output_feature

    output_logits = nn.Dense(
        self.vocab_size,
        kernel_init=nn.initializers.normal(stddev=0.02),
        name='output')(
            trans_out)  # (batch_size, caption_length, vocab_size)
    if return_logit_and_feat:
      return output_logits, output_feature
    # TODO(zhouxy): tie weight output and embedding.words
    return output_logits

  def _generate_future_mask(self, size):
    """Generate attention mask."""
    mask = jnp.triu(jnp.ones((size, size), jnp.float32), k=1)
    mask = jnp.where(mask > 0, NEG_INF, 0)
    return mask
