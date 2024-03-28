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

"""Utility functions for defining models."""

from typing import Callable, Iterable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from scenic.model_lib.layers import attention_layers
from scenic.projects.baselines import vit

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


def shuffle_and_partition(n_batch: int,
                          n_tokens: int,
                          n_masked: int,
                          rng: jax.Array):
  """Implements random shuffling and partitioning necessary for MAE.

  Args:
    n_batch: The batch size of the sequence to generate.
    n_tokens: The number of tokens.
    n_masked: The number of tokens to mask. Must have 0 <= n_masked < n_tokens.
    rng: The random number key.

  Returns:
    Two arrays. The first one contains indices of masked tokens, and has
    shape [n_batch, n_masked]. The second contains indices of unmasked tokens
    and has shape [n_batch, n_tokens - n_masked].
  """
  if n_masked >= n_tokens or n_masked < 0:
    raise ValueError(f'n_masked = {n_masked} should be >=0 and <{n_tokens}.')

  ids = jnp.tile(jnp.arange(n_tokens), n_batch).reshape((n_batch, n_tokens))
  n_remainder = n_tokens - n_masked
  if n_masked > 0:
    rng_keys = jax.random.split(rng, n_batch)
    ids = jax.vmap(
        lambda seq, rng: jax.random.permutation(rng, seq, independent=True))(
            ids, rng_keys)
  masked = jax.lax.dynamic_slice(ids, (0, 0,), (n_batch, n_masked,))
  unmasked = jax.lax.dynamic_slice(ids, (0, n_masked,), (n_batch, n_remainder,))
  return masked, unmasked


def get_mask_indices(n_batch: int,
                     n_tokens: int,
                     n_masked: int,
                     rng: jax.Array):
  """Returns indices to use for masking in MAE.

  Args:
    n_batch: The batch size of the sequence to generate.
    n_tokens: The number of tokens.
    n_masked: The number of tokens to mask. Must have 0 <= n_masked < n_tokens.
    rng: The random number key.

  Returns:
    Three arrays. masked_indices of shape [n_batch, n_masked], unmasked_indices
    of shape [n_batch, n_tokens - n_masked] and binary_mask of shape
    [n_batch, n_tokens] where 1 indicates that the token is masked.
  """
  batch_indices = jnp.arange(n_batch).reshape(n_batch, 1)
  mask_indices, unmasked_indices = shuffle_and_partition(
      n_batch, n_tokens, n_masked, rng)
  binary_mask = jnp.zeros((n_batch, n_tokens)).at[batch_indices,
                                                  mask_indices].set(1.0)

  return mask_indices, unmasked_indices, binary_mask


def get_tube_mask_indices(n_batch: int,
                          n_tokens: int,
                          token_mask_probability: float,
                          temporal_dims: int,
                          rng: jax.Array):
  """Returns indices to use for tube masking in VideoMAE.

  The difference between the random and tube masking is that the tube masking
  takes into account the temporal dimension when masking.

  Args:
    n_batch: The batch size of the sequence to generate.
    n_tokens: The number of tokens.
    token_mask_probability: Probability of dropping out the input tokens
    during training.
    temporal_dims: The temporal dimension.
    rng: The random number key.

  Returns:
    Three arrays. masked_indices of shape [n_batch, n_masked], unmasked_indices
    of shape [n_batch, n_tokens - n_masked] and binary_mask of shape
    [n_batch, n_tokens] where 1 indicates that the token is masked.
  """

  n_tokens_frame = n_tokens // temporal_dims
  n_masked_frame = int(token_mask_probability * n_tokens_frame)
  batch_indices = jnp.arange(n_batch).reshape(n_batch, 1)

  mask_indices_frame, _ = shuffle_and_partition(n_batch, n_tokens_frame,
                                                n_masked_frame, rng)
  binary_mask_frame = jnp.zeros((n_batch, n_tokens_frame)
                                ).at[batch_indices, mask_indices_frame].set(1.0)

  # Add temporal dims
  binary_mask = jnp.tile(binary_mask_frame, [1, temporal_dims])

  # Apply binary_mask
  n_masked_tokens = n_masked_frame * temporal_dims
  n_unmasked_tokens = n_tokens - n_masked_tokens
  masked_indices = jnp.nonzero(binary_mask, size=(n_batch * n_masked_tokens)
                               )[1].reshape(n_batch, -1)
  unmasked_indices = jnp.nonzero(binary_mask - 1,
                                 size=(n_batch * n_unmasked_tokens)
                                 )[1].reshape(n_batch, -1)

  return masked_indices, unmasked_indices, binary_mask


class AddFactorisedSpaceTimePositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    posemb_init_space: Positional embedding initializer. Default value is taken
      from BERT.
    posemb_init_time: Positional embedding initializer. Default value is taken
      from BERT.

  Returns:
    Output with same shape as input.
  """
  posemb_init_space: Initializer = nn.initializers.normal(stddev=0.02)
  posemb_init_time: Initializer = nn.initializers.normal(stddev=0.02)

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    # Inputs.shape is [batch_size, time, space, hidden_dim].
    assert inputs.ndim == 4, ('Number of dimensions should be 4,'
                              ' but it is: %d' % inputs.ndim)
    _, time, space, hidden_dim = inputs.shape
    pos_emb_shape_space = (1, 1, space, hidden_dim)
    pos_emb_shape_time = (1, time, 1, hidden_dim)
    pe_spatial = self.param('pos_embedding_space', self.posemb_init_space,
                            pos_emb_shape_space, inputs.dtype)
    pe_temporal = self.param('pos_embedding_time', self.posemb_init_time,
                             pos_emb_shape_time, inputs.dtype)
    return inputs + pe_spatial + pe_temporal


def add_positional_embeddings(
    inputs: jnp.ndarray,
    posemb_type: str,
    input_shape: Optional[Iterable[int]] = None,
    layer_name: str = 'posembed_input') -> jnp.ndarray:
  """Adds positional embeddings to an input sequence.

  Args:
    inputs: Tokens of shape [batch, num_tokens, hidden_size].
    posemb_type: The type of positional encoding. Must be one of
      {sinusoidal_1d, sinusoidal_2d, sinusoidal_3d, learned_1d}.
    input_shape: Used for "sinusoidal_2d" and "sinusoidal_3d". In this case,
      the input is reshaped to this size ie [batch, height, width, hidden_size],
      before applying the positional encodings and then reshaping back.
    layer_name: The layer name for learned embedddings.

  Returns:
    The input tokens with the positional encodings added. The shape is
      [batch, num_tokens, hidden_size].
  """

  if posemb_type == 'learned_1d':
    x_posemb = vit.AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
        name=layer_name)(inputs)
  elif posemb_type == 'learned_space_time':
    x_reshape = inputs.reshape(input_shape)
    x_posemb = AddFactorisedSpaceTimePositionEmbs(
        posemb_init_space=nn.initializers.normal(stddev=0.02),  # from BERT.
        posemb_init_time=nn.initializers.normal(stddev=0.02),
        name=layer_name)(x_reshape)
    x_posemb = jnp.reshape(x_posemb, inputs.shape)
  elif posemb_type == 'sinusoidal_1d':
    x_posemb = attention_layers.Add1DPositionEmbedding(
        posemb_init=None)(inputs)
  elif posemb_type in {'sinusoidal_2d', 'sinusoidal_3d'}:
    x_reshape = inputs.reshape(input_shape)
    x_posemb = attention_layers.AddFixedSinCosPositionEmbedding()(x_reshape)
    x_posemb = jnp.reshape(x_posemb, inputs.shape)
  elif posemb_type == 'none':
    x_posemb = inputs
  else:
    raise ValueError(f'Unknown positional embedding {posemb_type}')

  return x_posemb


def embed_2d_patch(x, patches, embedding_dim, return_1d=True, name='embedding'):
  """Embedding input patches with 2D conv."""

  assert patches.get('size') is not None, ('patches.size is now the only way'
                                           'to define the patches')
  assert embedding_dim, 'embedding_dim must be specified'
  fh = patches.size[0]
  fw = patches.size[1]

  x = nn.Conv(
      embedding_dim, (fh, fw),
      strides=(fh, fw),
      padding='VALID',
      name=name)(x)

  if return_1d:
    batch_size = x.shape[0]
    x = jnp.reshape(x, [batch_size, -1, embedding_dim])
  return x

