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

"""Deformable attention model blocks."""

import functools
from typing import Optional

import flax.linen as nn
from flax.linen import attention
import jax
import jax.numpy as jnp
import numpy as np
from scenic.projects.boundary_attention.models.model_lib import deformable_attention_utils
from scenic.projects.boundary_attention.models.model_lib import misc_blocks
from scenic.projects.boundary_attention.models.model_lib import rope_embedding


class DeformableTransformerBlock(nn.Module):
  """Finds next hidden state given junction features and prev hidden state."""

  deformation_type: str = 'simple'
  offset_fn: str = 'dense'
  max_offset: int = 30
  num_samples: int = 16
  M: int = 3
  C: int = 3
  patchsize: int = 5
  encoding_dim: int = 256
  num_heads: Optional[int] = 4
  num_layers: Optional[int] = 1
  attn_dropout_prob: Optional[float] = 0

  @nn.compact
  def __call__(
      self,
      hidden_state: jnp.ndarray,
      embedded_hidden_state: jnp.ndarray,
      train=True,
  ):

    ref_xy = self.get_ref_xy(hidden_state.shape[1], hidden_state.shape[2])

    # Process through transformer encoder layers
    for ll in range(self.num_layers):
      if self.deformation_type == 'simple':
        hidden_state = DeformableEncoder(
            self.num_samples,
            self.offset_fn,
            self.max_offset,
            self.encoding_dim,
            self.num_heads,
            self.attn_dropout_prob,
            name='EncoderBlock_{:d}'.format(ll),
        )(hidden_state, embedded_hidden_state, ref_xy, train=train)
      elif self.deformation_type == 'simple_with_rotary_embedding':
        hidden_state = DeformableEncoderWithRotaryEmbedding(
            self.num_samples,
            self.offset_fn,
            self.max_offset,
            hidden_dim=hidden_state.shape[-1],
            encoding_dim=self.encoding_dim,
            num_heads=self.num_heads,
            attn_dropout_prob=self.attn_dropout_prob,
            name='EncoderBlock_{:d}'.format(ll),
        )(hidden_state, embedded_hidden_state, ref_xy, train=train)
      else:
        hidden_state = DeformableEncoder(
            self.num_samples,
            self.offset_fn,
            self.max_offset,
            self.encoding_dim,
            self.num_heads,
            self.attn_dropout_prob,
            name='EncoderBlock_{:d}'.format(ll),
        )(hidden_state, embedded_hidden_state, ref_xy, train=train)

    # Return new hidden state
    return hidden_state

  def get_ref_xy(self, hpatches, wpatches):

    # to do: change this...maybe
    ref_xy = jnp.stack(
        jnp.meshgrid(jnp.arange(hpatches), jnp.arange(wpatches), indexing='ij'),
        axis=-1,
    )

    return ref_xy


class PredictOffsets(nn.Module):
  """Predicts deformable attention sampling offsets."""

  num_samples: int = 16
  offset_fn: str = 'dense'
  max_offset: int = 30

  @nn.compact
  def __call__(self, hidden_state):

    if self.offset_fn == 'dense':
      offsets = nn.DenseGeneral(features=(self.num_samples, 2), axis=-1,
                                kernel_init=nn.initializers.zeros,
                                bias_init=self.offset_bias_init,
                                name='SamplingOffsets')(hidden_state)
    else:
      offsets = nn.Conv(
          features=(self.num_samples * 2),
          kernel_size=(3, 3),
          name='SamplingOffsets',
      )(hidden_state)
      offsets = jnp.reshape(
          offsets, (*hidden_state.shape[:-1], self.num_samples, 2)
      )

    # Normalize offsets
    normalized_offsets = (
        2 * self.max_offset * nn.sigmoid(offsets) - self.max_offset
    )

    return normalized_offsets

  def offset_bias_init(self, rng, flat_shape, dtype) -> jax.Array:
    """Initializes deformable attention sampling offsets."""

    del rng, flat_shape, dtype

    sqn = np.ceil(np.sqrt(self.num_samples))
    n = np.square(sqn).astype(int)
    init_bias = np.stack(
        np.meshgrid(
            np.linspace(-sqn // 2, sqn // 2, sqn.astype(int)),
            np.linspace(-sqn // 2, sqn // 2, sqn.astype(int)),
            indexing='ij',
        ),
        axis=-1,
    ).reshape((n, 2))[: self.num_samples, :]
    init_bias = init_bias / 15

    return init_bias


class DeformableEncoder(nn.Module):
  """Finds next hidden state."""

  num_samples: int = 16
  offset_fn: str = 'dense'
  max_offset: int = 30
  encoding_dim: int = 256
  num_heads: int = 4
  attn_dropout_prob: float = 0

  @nn.compact
  def __call__(
      self,
      hidden_state_q: jnp.ndarray,
      hidden_state_kv: jnp.ndarray,
      ref_xy: jnp.ndarray,
      train: bool = True,
  ):

    # 1. Calculate offsets
    normalized_offsets = PredictOffsets(self.num_samples,
                                        self.offset_fn,
                                        self.max_offset)(hidden_state_q)

    # 2. Calculate sampling locations
    sampling_locations = jnp.expand_dims(ref_xy, (0, 3,)) + normalized_offsets

    # 3. Threshold sampling locations so that they remain inside the image
    sampling_locations = jnp.where(sampling_locations >
                                   jnp.array((hidden_state_q.shape[1]-1,
                                              hidden_state_q.shape[2]-1)),
                                   jnp.array((hidden_state_q.shape[1]-1,
                                              hidden_state_q.shape[2]-1)),
                                   sampling_locations)
    sampling_locations = jnp.where(sampling_locations <
                                   jnp.zeros((2)),
                                   jnp.zeros((2)),
                                   sampling_locations)

    # 4. Sample hidden state using the offsets to get inputs_kv
    inputs_kv = deformable_attention_utils.linearly_interpolate(
        hidden_state_kv, sampling_locations
    )

    # 5. Add projected offsets as positional encodings to inputs_kv
    inputs_q = jnp.expand_dims(hidden_state_q, 3)
    inputs_kv = inputs_kv + nn.Dense(hidden_state_kv.shape[-1])(
        normalized_offsets
    )

    # 6. Process through attention block to update the hidden state
    attention_block = AttentionBlock(
        hidden_state_q.shape[-1],
        self.encoding_dim,
        self.num_heads,
        self.attn_dropout_prob,
        name='AttentionBlock',
    )
    hidden_state = attention_block(inputs_q, inputs_kv, train=train)

    return hidden_state


class AttentionBlock(nn.Module):
  """Attention block.

  Attributes:
    num_samples: number of points to sample per query
    inputs_q: [N, H, W, 1, D)] = [batch_sizes...,length, features]
    inputs_kv: [N, H, W, num_samples, D] = [batch_sizes..., length, features]
  """

  hidden_dim: int
  mlp_hidden_dim: int
  num_heads: int = 4
  dropout_prob: float = 0

  @nn.compact
  def __call__(self, inputs_q, inputs_kv, train=True):

    assert self.hidden_dim % self.num_heads == 0, (
        'hidden_dim must be divisible by num_heads.')

    # Estimate the next hidden state
    hidden_state = inputs_q + nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads)(inputs_q, inputs_kv,
                                  deterministic=not train)
    hidden_state = nn.LayerNorm()(hidden_state)

    # Process the hidden state with an additive MLP
    mlp = misc_blocks.MLP(hidden_size=self.mlp_hidden_dim,
                          dropout_rate=self.dropout_prob,
                          name='MLP')
    hidden_state = hidden_state + mlp(hidden_state, train=train)

    # A final layer norm
    hidden_state = nn.LayerNorm()(hidden_state)

    return hidden_state.squeeze(-2)


class DeformableEncoderWithRotaryEmbedding(nn.Module):
  """Finds next hidden state."""

  num_samples: int = 16
  offset_fn: str = 'dense'
  max_offset: int = 30
  hidden_dim: int = 72
  encoding_dim: int = 256
  num_heads: int = 3
  attn_dropout_prob: float = 0

  rot_embed = rope_embedding.RotaryEmbedding2D(hidden_dim // num_heads)

  @nn.compact
  def __call__(self, hidden_state_q, hidden_state_kv, ref_xy, train=True):

    # 0. Predict offsets
    normalized_offsets = PredictOffsets(self.num_samples,
                                        self.offset_fn,
                                        self.max_offset)(hidden_state_q)

    # 1. Calculate sampling locations
    sampling_locations = jnp.expand_dims(ref_xy, (0, 3,)) + normalized_offsets

    # 2. Threshold sampling locations so that they remain inside the image
    sampling_locations = jnp.where(sampling_locations >
                                   jnp.array((hidden_state_q.shape[1]-1,
                                              hidden_state_q.shape[2]-1)),
                                   jnp.array((hidden_state_q.shape[1]-1,
                                              hidden_state_q.shape[2]-1)),
                                   sampling_locations)
    sampling_locations = jnp.where(sampling_locations <
                                   jnp.zeros((2)),
                                   jnp.zeros((2)),
                                   sampling_locations)

    # 3. Project Hidden state to Q and KV
    dense = functools.partial(
        nn.DenseGeneral,
        axis=-1,
        features=(self.num_heads, hidden_state_q.shape[-1] // self.num_heads)
        )

    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (
        dense(name='query')(hidden_state_q),
        dense(name='key')(hidden_state_kv),
        dense(name='value')(hidden_state_kv))

    query = jnp.expand_dims(query, 3)  # (1, 109, 109, 1, 4, 32)
    key_flat = key.reshape((*key.shape[:-2], -1))
    value_flat = value.reshape((*value.shape[:-2], -1))

    # 4. Sample KV at offset locations
    sampled_keys = deformable_attention_utils.linearly_interpolate(
        key_flat, sampling_locations).reshape((*sampling_locations.shape[:-1],
                                               self.num_heads, -1))
    sampled_values = deformable_attention_utils.linearly_interpolate(
        value_flat, sampling_locations).reshape((*sampling_locations.shape[:-1],
                                                 self.num_heads, -1))

    # 4. Add rotary embedding to q as well as sampled keys and values
    sinusoid_inp = self.rot_embed.get_pos(sampling_locations)

    sampled_keys_with_pos = self.rot_embed.apply_2d_rotary_pos_emb(
        sampled_keys, sinusoid_inp)
    sampled_values_with_pos = self.rot_embed.apply_2d_rotary_pos_emb(
        sampled_values, sinusoid_inp)

    xy_grid = jnp.expand_dims(
        jnp.stack(jnp.meshgrid(jnp.arange(hidden_state_q.shape[1]),
                               jnp.arange(hidden_state_q.shape[2]),
                               indexing='ij'), axis=-1), (0, 3,))
    query_with_pos = self.rot_embed.calc_and_apply(query, xy_grid)

    # Define attention block and update hidden state
    attention_block = RotaryAttentionBlock(hidden_state_q.shape[-1],
                                           self.encoding_dim,
                                           self.num_heads,
                                           self.attn_dropout_prob,
                                           name='AttentionBlock')
    hidden_state = attention_block(hidden_state_q,
                                   query_with_pos,
                                   sampled_keys_with_pos,
                                   sampled_values_with_pos,
                                   train=train)

    return hidden_state


class RotaryAttentionBlock(nn.Module):
  """Attention block with rotary positional embedding.

  Attributes:
    num_samples: number of points to sample per query
    query: [N, H, W, 1, D)]
    key: [N, H, W, num_samples, D]
    value: [N, H, W, num_samples, D]
  """

  hidden_dim: int
  mlp_hidden_dim: int
  num_heads: int = 4
  dropout_prob: float = 0

  @nn.compact
  def __call__(self, hidden_state, query, key, value, train=True):

    new_hidden = nn.DenseGeneral(
        features=self.hidden_dim,
        axis=(-3, -2, -1))(attention.dot_product_attention(query, key, value))

    # 5. Calculate attention and find updated hidden state
    hidden_state = hidden_state + new_hidden

    # 6. Continue normally
    hidden_state = nn.LayerNorm()(hidden_state)

    # Process the hidden state with an additive MLP
    mlp = misc_blocks.MLP(hidden_size=self.mlp_hidden_dim,
                          dropout_rate=self.dropout_prob,
                          name='MLP')
    hidden_state = hidden_state + mlp(hidden_state, train=train)

    # A final layer norm
    hidden_state = nn.LayerNorm()(hidden_state)

    return hidden_state
