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

"""Util functions for ViViT models."""

import functools
from typing import Any, Callable, Iterable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.layers import attention_layers
from scenic.projects.objectvivit.object_attention import ObjectBlock

Initializer = Callable[[jnp.ndarray, Iterable[int], jnp.dtype], jnp.ndarray]


def get_object_inds(
    scores, num_tokens_per_frame, configs, factorized_encoder=False):
  """Generate inds from scores.

  Args:
    scores: batch x num_objs x num_tokens
    num_tokens_per_frame: int
    configs: config dict
    factorized_encoder: bool; if it is used in the factorized_encoder model.
      Get index for each frame if so.
  Returns:
    inds: batch x num_attach_tokens if factorized_encoder==False else
      batch x num_frames x num_frame_attach_tokens
  """
  num_total_attach_tokens = configs.get('num_total_attach_tokens', -1)
  num_frame_attach_tokens = configs.get('num_frame_attach_tokens', -1)
  assert num_total_attach_tokens == -1 or num_frame_attach_tokens == -1
  assert (not factorized_encoder) or num_frame_attach_tokens > 0
  batch = scores.shape[0]
  # combine heatmaps from different objects
  pooled_scores = scores.max(axis=1)  # batch x num_tokens
  if num_total_attach_tokens > 0:
    pooled_scores = pooled_scores.reshape(batch, -1)
    _, inds = jax.lax.top_k(pooled_scores, k=num_total_attach_tokens)
  else:
    pooled_scores = pooled_scores.reshape(
        batch, -1, num_tokens_per_frame)  # B x T x HW
    num_frames = pooled_scores.shape[1]
    _, inds = jax.lax.top_k(
        pooled_scores, k=num_frame_attach_tokens)  # B x T x k
    if not factorized_encoder:
      inds = inds.reshape(batch, -1)  # [B * num_objs, Tk]
      base = jnp.arange(
          num_frames * num_frame_attach_tokens
      ) // num_frame_attach_tokens * num_tokens_per_frame
      inds = inds + base[None]  # [B * num_objs, Tk]
      inds = inds.reshape(batch, -1)
  return inds


def resize_token_score(scores, patch_size):
  batch, in_t, in_h, in_w, n_obj = scores.shape
  fh, fw, ft = patch_size
  gt, gh, gw = in_t // ft, in_h // fh, in_w // fw
  scores = scores.reshape(batch, gt, ft, gh, fh, gw, fw, n_obj).mean(
      axis=6).mean(axis=4).mean(axis=2).reshape(batch, gt * gh * gw, n_obj)
  return scores


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
  del layer_name
  del input_shape
  if posemb_type == 'sinusoidal_1d':
    x_posemb = attention_layers.Add1DPositionEmbedding(
        posemb_init=None)(inputs)
  elif posemb_type == 'none':
    x_posemb = inputs
  else:
    raise ValueError(f'Unknown positional embedding {posemb_type}')

  return x_posemb


class MLP(nn.Module):
  """Simple MLP."""
  num_layers: int
  hidden_dim: int

  @nn.compact
  def __call__(self, x):
    """Forward module.

    Args:
      x: array in shape batch_size x num_tokens x hidden_dim
    Returns:
      batch_size x num_tokens x out_dim: 2 for softmax
    """
    x = nn.LayerNorm()(x)
    for i in range(self.num_layers):
      x = nn.Dense(self.hidden_dim, name=f'linear.{i}')(x)
      x = nn.gelu(x)
    return x


class CustomEncoderBlock(nn.Module):
  """The same as ViViT Transformer encoder block. Supports masked tokens."""
  mlp_dim: Optional[int]
  num_heads: int
  dtype: jnp.dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  attention_kernel_initializer: Initializer = nn.initializers.xavier_uniform()
  mlp_kernel_initializer: Initializer = nn.initializers.xavier_uniform()
  mlp_bias_initializer: Initializer = nn.initializers.normal(stddev=1e-6)
  attention_fn: Any = nn.dot_product_attention
  droplayer_p: float = 0.0
  use_approximate_gelu: bool = True

  def get_drop_pattern(self, x, deterministic):
    if not deterministic and self.droplayer_p:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng('dropout'), self.droplayer_p, shape).astype('float32')
    else:
      return 0.0

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, deterministic: bool,
      empty_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Applies Encoder1DBlock module."""

    # Expanding object mask to pairwise attention masks. This is used in
    #  factorized-encoder models where removing tokens are hard.
    # empty_mask: B x L --> mask: B x 1 x L x L.
    #   The second dimention 1 will be broadcasted to num_heads
    mask = None if empty_mask is None else empty_mask[
        ..., None, None, :] * empty_mask[..., None, :, None]
    # Attention block.
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=self.attention_kernel_initializer,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        attention_fn=self.attention_fn,
        dtype=self.dtype)(
            x, x, mask=mask, deterministic=deterministic)  # added mask here.
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)

    drop_pattern = self.get_drop_pattern(x, deterministic)
    x = x * (1.0 - drop_pattern) + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = attention_layers.MlpBlock(  # pytype: disable=wrong-arg-types  # jnp-type
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=functools.partial(
            nn.gelu, approximate=self.use_approximate_gelu),
        kernel_init=self.mlp_kernel_initializer,
        bias_init=self.mlp_bias_initializer)(
            y, deterministic=deterministic)

    drop_pattern = self.get_drop_pattern(x, deterministic)
    return y * (1.0 - drop_pattern) + x


class CustomEncoder(nn.Module):
  """Encoder that supports dropped tokens and object-aware attention."""

  temporal_dims: Optional[int]
  hidden_size: int
  mlp_dim: int
  num_layers: int
  num_heads: int
  attention_config: Optional[ml_collections.ConfigDict] = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_droplayer_rate: float = 0.0
  fold_first_token: bool = False
  dtype: jnp.dtype = jnp.float32
  n_posembed: Optional[int] = None
  positional_embedding: str = 'learned_1d'
  normalise_output: bool = True
  use_approximate_gelu: bool = True
  num_tokens_per_frame: int = -1
  run_cross_frame_attention: bool = False
  video_batch: int = -1
  object_config: ml_collections.ConfigDict = ml_collections.ConfigDict()
  learn_token_configs: ml_collections.ConfigDict = ml_collections.ConfigDict()
  attach_configs: ml_collections.ConfigDict = ml_collections.ConfigDict()

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, empty_mask=None, fg_inds=None,
               token_scores=None, boxes=None, *,
               train: bool = False, debug: bool = False):
    """Applies Transformer model on the inputs.

    Args:
      inputs: array in shape batch x len x emb
      empty_mask: bool array in shape batch x len: whether to keep a token
      fg_inds: int array in shape batch x K, K is smaller then
        len, the index of kept tokens. One of empty_mask and fg_inds
        should be None.
      token_scores: float: batch x num_objects x num_tokens
      boxes: float: batch x T x num_objs x 4 in range 0 - 1
      train: in training or evaluation.
      debug: print debug information
    Returns:
      an array: the updated feature.
    """
    assert inputs.ndim == 3  # (batch, len, emb)
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)
    learn_token = self.learn_token_configs.get('enabled', False)
    learn_token_idx = self.learn_token_configs.get('layer_index', 0)
    assert not learn_token
    assert not learn_token_idx
    attach_tokens = self.attach_configs.get('enabled', False)
    drop_pixel_tokens = self.attach_configs.get('drop_pixel_tokens', False)
    assert not (attach_tokens and drop_pixel_tokens)
    add_context_tokens = self.attach_configs.get('add_context_tokens', -1)
    object_block_idx = self.attach_configs.get('object_block_idx', [])
    drop_block_idx = self.attach_configs.get('drop_block_idx', 0)

    n_posembed = self.n_posembed or inputs.shape[1]
    assert n_posembed <= inputs.shape[1], f'{n_posembed} > {inputs.shape[1]}'

    if self.positional_embedding == 'sinusoidal_1d':
      x = attention_layers.Add1DPositionEmbedding(
          posemb_init=None)(inputs)
    elif self.positional_embedding == 'none':
      x = inputs
    else:
      raise ValueError(
          f'Unknown positional embedding {self.positional_embedding}')
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    learned_token_scores = None
    aux = {}

    # Input Encoder
    for lyr in range(self.num_layers):
      droplayer_p = (
          lyr / max(self.num_layers - 1, 1)) * self.stochastic_droplayer_rate

      if lyr in object_block_idx and self.name != 'TemporalTransformer':
        assert lyr > 0, lyr
        x = ObjectBlock(
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            droplayer_p=droplayer_p,
            name=f'encoderblock_{lyr}',
            use_approximate_gelu=self.use_approximate_gelu,
            configs=self.attach_configs,
            dtype=dtype)(
                x, token_scores=token_scores, deterministic=not train)
        continue

      block = CustomEncoderBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          droplayer_p=droplayer_p,
          name=f'encoderblock_{lyr}',
          use_approximate_gelu=self.use_approximate_gelu,
          dtype=dtype)

      if drop_pixel_tokens and lyr == drop_block_idx and fg_inds is not None:
        if add_context_tokens > 0:
          context_tokens = self._add_context_tokens(
              x, add_context_tokens, fg_inds
          )  # batch x num_context_token x hidden_dim
        x, token_scores = self._drop_tokens(
            x, fg_inds, token_scores,
            drop_scores=object_block_idx)
        # x: batch x num_fg_tokens x hidden_dim
        # token_scores: batch x num_objects x num_fg_tokens
        if add_context_tokens > 0:
          x = jnp.concatenate([x, context_tokens], axis=1)
          if object_block_idx and token_scores is not None:
            num_objs = token_scores.shape[1]
            num_bg_tokens = context_tokens.shape[1]
            token_scores = jnp.concatenate(
                [token_scores,
                 jnp.zeros((x.shape[0], num_objs, num_bg_tokens), jnp.float32)],
                axis=2)

      x = block(
          x, empty_mask=empty_mask,
          deterministic=not train)  # empty_mask is used here

    if self.normalise_output:
      encoded = nn.LayerNorm(name='encoder_norm')(x)
    else:
      encoded = x

    return encoded, learned_token_scores, aux

  def _add_context_tokens(
      self, full_x, num_context_tokens, fg_inds=None):
    """Append random/uniformly sampled token from full_x to x.

    Args:
      full_x: batch x num_tokens x hidden_dim
      num_context_tokens: number of context tokens to add
      fg_inds: batch x num_fg_tokens
    Returns:
      context_tokens: batch x num_context_token x hidden_dim
    """
    batch, num_total_tokens, hidden_dim = full_x.shape
    k = fg_inds.shape[1]
    random_score = jax.random.uniform(
        self.make_rng('dropout'), (batch, num_total_tokens))

    obj_inds = jnp.arange(batch * k).reshape(
        batch, k) // k * num_total_tokens + fg_inds
    random_score = random_score.reshape(-1)
    random_score = random_score.at[obj_inds].set(0)
    random_score = random_score.reshape(batch, num_total_tokens)

    _, random_inds = jax.lax.top_k(random_score, k=num_context_tokens)
    base = jnp.arange(batch * num_context_tokens).reshape(
        batch, num_context_tokens) // num_context_tokens * num_total_tokens
    inds = base + random_inds
    full_x = full_x.reshape(batch * num_total_tokens, hidden_dim)
    context_tokens = full_x[inds.reshape(-1)]
    context_tokens = context_tokens.reshape(
        batch, num_context_tokens, hidden_dim)
    return context_tokens

  def _drop_tokens(self, x, fg_inds, token_scores=None, drop_scores=False):
    """Subsample token x according to the keeped index in fg_inds.

    Args:
      x: batch x num_tokens x hidden_dim
      fg_inds: batch x num_fg_tokens
      token_scores: batch x num_objects x num_tokens
      drop_scores: bool
    Returns:
      x: batch x num_fg_tokens x hidden_dim
      sampled_token_scores: batch x num_objects x num_fg_tokens
    """
    batch, num_tokens, hidden_dim = x.shape
    k = fg_inds.shape[1]
    #  This is identical to torch.gather(x, 1, fg_inds)
    inds = jnp.arange(batch * k).reshape(
        batch, k) // k * num_tokens + fg_inds
    x = x.reshape(batch * num_tokens, hidden_dim)
    x = x[inds.reshape(-1)].reshape(batch, k, hidden_dim)
    sampled_token_scores = None
    if drop_scores:
      num_objs = token_scores.shape[1]
      token_scores = token_scores.reshape(-1)
      expand_fg_inds = jnp.broadcast_to(fg_inds[:, None], (batch, num_objs, k))
      inds = jnp.arange(
          batch * num_objs * k) // k * num_tokens + expand_fg_inds.reshape(-1)
      sampled_token_scores = token_scores[inds].reshape(batch, num_objs, k)
    return x, sampled_token_scores
