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

from scenic.model_lib.layers import nn_layers

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
  stochastic_depth: float = 0.0

  @nn.compact
  def __call__(self, hidden_states, input_tensor, train=False):
    hidden_states = nn.Dense(
        self.hidden_size,
        kernel_init=nn.initializers.normal(stddev=0.02),
        name='dense')(hidden_states)
    hidden_states = nn.Dropout(self.hidden_dropout)(
        hidden_states, deterministic=not train)
    hidden_states = nn_layers.StochasticDepth(self.stochastic_depth)(
        hidden_states, deterministic=not train)
    hidden_states = hidden_states + input_tensor
    hidden_states = nn.LayerNorm(
        epsilon=1e-5, name='LayerNorm')(hidden_states)
    return hidden_states


class BertAttention(nn.Module):
  """Bert layer attention."""
  hidden_size: int = 768
  num_heads: int = 12
  dropout: float = 0.1
  attention_dropout: float = 0.1
  stochastic_depth: float = 0.0

  @nn.compact
  def __call__(
      self, input_tensor, attention_mask, train=False):
    self_outputs = BertSelfAttention(
        num_heads=self.num_heads,
        hidden_size=self.hidden_size,
        attention_dropout=self.attention_dropout,
        name='self')(
            input_tensor, attention_mask, train=train,
        )  # (batch_size, tot_len, hidden_size)
    attention_output = BertSelfOutput(
        hidden_size=self.hidden_size,
        hidden_dropout=self.dropout,
        stochastic_depth=self.stochastic_depth,
        name='output')(
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
  stochastic_depth: float = 0.0

  @nn.compact
  def __call__(
      self, hidden_states, input_tensor, train=False):
    hidden_states = nn.Dense(
        self.hidden_size,
        kernel_init=nn.initializers.normal(stddev=0.02),
        name='dense')(hidden_states)
    hidden_states = nn.Dropout(self.hidden_dropout)(
        hidden_states, deterministic=not train)
    hidden_states = nn_layers.StochasticDepth(self.stochastic_depth)(
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
  dropout: float = 0.1
  attention_dropout: float = 0.1
  stochastic_depth: float = 0.0

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
        num_heads=self.num_heads,
        hidden_size=self.hidden_size,
        dropout=self.dropout,
        attention_dropout=self.attention_dropout,
        stochastic_depth=self.stochastic_depth,
        name='attention')(
            hidden_states, attention_mask, train=train,
        )  # (batch_size, tot_len, hidden_size)
    intermediate_output = BertIntermediate(
        intermediate_size=self.hidden_size * 4, name='intermediate')(
            attention_outputs, train=train,
        )  # (batch_size, tot_len, intermediate_size)
    layer_output = BertOutput(
        hidden_size=self.hidden_size,
        hidden_dropout=self.dropout,
        stochastic_depth=self.stochastic_depth,
        name='output')(
            intermediate_output, attention_outputs, train=train,
        )  # (batch_size, tot_len, hidden_size)
    return layer_output


class BertEncoder(nn.Module):
  """GIT Encoder."""
  num_hidden_layers: int = 6
  hidden_size: int = 768
  num_heads: int = 12
  stochastic_depth: float = 0.0
  dropout: float = 0.1
  attention_dropout: float = 0.1

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
    assert self.stochastic_depth >= 0.0 and self.stochastic_depth < 1.0
    assert self.dropout >= 0.0 and self.dropout < 1.0
    assert self.attention_dropout >= 0.0 and self.attention_dropout < 1.0

    for i in range(self.num_hidden_layers):
      stochastic_depth_layer = (
          i / max(self.num_hidden_layers - 1, 1)) * self.stochastic_depth
      hidden_states = BertLayer(
          hidden_size=self.hidden_size,
          num_heads=self.num_heads,
          stochastic_depth=stochastic_depth_layer,
          dropout=self.dropout,
          attention_dropout=self.attention_dropout,
          name=f'layer.{i}',
      )(hidden_states, attention_mask, train=train)
    return hidden_states


class BertEncoderAsDecoder(nn.Module):
  """GIT Decoder."""
  num_hidden_layers: int = 6
  hidden_size: int = 768
  num_heads: int = 12

  @nn.compact
  def __call__(
      self, tgt, memory, tgt_mask=None,
      memory_key_padding_mask=None, train=False, return_visual_feature=False,):
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

    result = BertEncoder(
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
    return result


def generate_future_mask(size):
  """Generate attention mask."""
  mask = jnp.triu(jnp.ones((size, size), jnp.float32), k=1)
  mask = jnp.where(mask > 0, NEG_INF, 0)
  return mask
