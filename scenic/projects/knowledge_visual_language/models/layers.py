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

"""Layers and Modules for Knowledge-FID."""

import functools
from typing import Optional, Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp
from scenic.projects.knowledge_visual_language.models import constants
from t5x.examples.t5 import layers as t5_layers
from t5x.examples.t5 import network as t5_network


@jax.vmap
def batch_index_select(data, idx):
  return jnp.take(data, idx, axis=0)


def _mask_select(data, mask):
  return jax.lax.select(
      mask > 0, data, jnp.full(data.shape, 0).astype(data.dtype)
  )


def l2_norm(x):
  """Compute the l2 norm of a vector."""
  return jnp.sqrt((x * x).sum(axis=-1))


def l2_normalize(x, axis=-1, eps=1e-10):
  """Normalizes along dimension `axis` using an L2 norm.

  This specialized function exists for numerical stability reasons.
  Args:
    x: An input ndarray.
    axis: Dimension along which to normalize, e.g. `1` to separately normalize
      vectors in a batch. Passing `None` views `t` as a flattened vector when
      calculating the norm (equivalent to Frobenius norm).
    eps: Epsilon to avoid dividing by zero.

  Returns:
    An array of the same shape as 'x' L2-normalized along 'axis'.
  """
  denorm = (x * x).sum(axis=axis, keepdims=True) + eps
  return (x * jax.lax.rsqrt(denorm)).astype(x.dtype)


class AffineTransform(nn.Module):
  """Do affine Transform for modulating attention score."""

  @nn.compact
  def __call__(self, x):
    scale = self.param('scale', nn.initializers.ones, (1,), jnp.float32)
    bias = self.param('bias', nn.initializers.zeros, (1,), jnp.float32)
    return x * nn.sigmoid(scale) * 5 + bias


class TransformerHead(nn.Module):
  """A stack of encoder layers."""

  num_head_layers: int
  key_dim: int
  vocab_size: int
  emb_dim: int
  num_heads: int
  num_encoder_layers: int
  num_decoder_layers: int
  head_dim: int
  mlp_dim: int
  dropout_rate: float
  out_head: nn.Module
  dtype: str = 'bfloat16'
  mlp_activations: Sequence[str] = ('gelu', 'linear')
  logits_via_embedding: bool = False

  def setup(self):
    self.t5_config = t5_network.T5Config(
        vocab_size=self.vocab_size,
        emb_dim=self.emb_dim,
        num_heads=self.num_heads,
        num_encoder_layers=self.num_encoder_layers,
        num_decoder_layers=self.num_decoder_layers,
        head_dim=self.head_dim,
        mlp_dim=self.mlp_dim,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        mlp_activations=self.mlp_activations,
        logits_via_embedding=self.logits_via_embedding,
    )

  @nn.compact
  def __call__(self, encoded_emb, encoder_mask=None, use_dropout=True):
    """transform the encoded representation."""
    cfg = self.t5_config
    assert encoded_emb.ndim == 3  # [batch, length, emb_dim]
    x = encoded_emb
    if encoder_mask is not None:
      encoder_mask = t5_layers.make_attention_mask(
          encoder_mask, encoder_mask, dtype=cfg.dtype
      )

    rel_emb = t5_layers.RelativePositionBiases(
        num_buckets=32,
        max_distance=128,
        num_heads=self.num_heads,
        dtype=self.dtype,
        embedding_init=nn.initializers.variance_scaling(
            1.0, 'fan_avg', 'uniform'
        ),
    )
    for _ in range(
        cfg.num_encoder_layers - self.num_head_layers, cfg.num_encoder_layers
    ):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      x = t5_network.EncoderLayer(config=cfg, relative_embedding=rel_emb)(
          x, encoder_mask, deterministic=not use_dropout
      )
    x = t5_layers.LayerNorm(dtype=cfg.dtype)(x[:, 0, :])
    return l2_normalize(self.out_head(x), axis=-1)


class LowerT5Encoder(nn.Module):
  """T5 encoder as a separate model which fuse multi-modal input.

  This module contains the encoder part of a pretrained T5. It is useful when
  adopting the pretrained T5 encoder as a part of a larger network. Note that
  the embedding layer should be created outside the module and provided as a
  parameter `shared_embedding` to share it in other parts of the network (e.g.,
  text encoder). If `shared_embedding` is not provided, the embedding layer is
  created within the module.

  Attributes:
    vocab_size: Size of the vocabulary.
    emb_dim: Size of the embeddings.
    num_heads: Number of attention heads.
    num_encoder_layers: Number of encoder layers.
    num_decoder_layers: Number of decoder layers.
    head_dim: Size of the embeddings in each head.
    mlp_dim: Size of the MLP output embeddings.
    dropout_rate: Dropout rate.
    dtype: Data type.
    mlp_activations: Sequence of activations in MLP.
    logits_via_embedding: Use the embedding weights for computing logits.
    shared_embedding: Optional. Embedding layer that is shared outside this
      module. If not given, a non-shared embedding layer will be created within
      the module.
  """

  vocab_size: int
  emb_dim: int
  num_heads: int
  num_encoder_layers: int
  num_decoder_layers: int
  num_fusion_layers: int
  head_dim: int
  mlp_dim: int
  dropout_rate: float
  dtype: str = 'bfloat16'
  mlp_activations: Sequence[str] = ('gelu', 'linear')
  logits_via_embedding: bool = False
  shared_embedding: Optional[nn.Module] = None

  def setup(self):
    self.t5_config = t5_network.T5Config(
        vocab_size=self.vocab_size,
        emb_dim=self.emb_dim,
        num_heads=self.num_heads,
        num_encoder_layers=self.num_encoder_layers,
        num_decoder_layers=self.num_decoder_layers,
        head_dim=self.head_dim,
        mlp_dim=self.mlp_dim,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        mlp_activations=self.mlp_activations,
        logits_via_embedding=self.logits_via_embedding,
    )
    if self.shared_embedding is None:
      self.shared_embedding = t5_layers.Embed(
          num_embeddings=self.vocab_size,
          features=self.emb_dim,
          dtype=self.dtype,
          attend_dtype=jnp.float32,  # For logit training stability.
          embedding_init=nn.initializers.normal(stddev=1.0),
          one_hot=True,
      )

  @nn.compact
  def __call__(
      self,
      encoder_input_tokens,
      encoder_segment_ids=None,
      use_dropout=True,
      frozen_base=True,
  ):
    """encode the text sentence only.

    Args:
      encoder_input_tokens: input text tokens
      encoder_segment_ids: segmend ID in packing mode
      use_dropout: whether to use dropout during Training
      frozen_base: whether froze the text encoder

    Returns:
      Sequence of token embedding with or without fusion
    """
    cfg = self.t5_config
    assert encoder_input_tokens.ndim == 2  # (batch, len)
    # Make padding attention mask.
    encoder_mask = encoder_input_tokens > 0
    mask_matrix = t5_layers.make_attention_mask(
        encoder_input_tokens > 0, encoder_input_tokens > 0, dtype=cfg.dtype
    )
    # Add segmentation block-diagonal attention mask if using segmented data.
    if encoder_segment_ids is not None:
      mask_matrix = t5_layers.combine_masks(
          mask_matrix,
          t5_layers.make_attention_mask(
              encoder_segment_ids,
              encoder_segment_ids,
              jnp.equal,
              dtype=cfg.dtype,
          ),
      )

    rel_emb = t5_layers.RelativePositionBiases(
        num_buckets=32,
        max_distance=128,
        num_heads=self.t5_config.num_heads,
        dtype=self.t5_config.dtype,
        embedding_init=nn.initializers.variance_scaling(
            1.0, 'fan_avg', 'uniform'
        ),
    )

    # [batch, length] -> [batch, length, emb_dim]
    x = self.shared_embedding(encoder_input_tokens.astype('int32'))
    x = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
        x, deterministic=not use_dropout
    )
    x = x.astype(cfg.dtype)
    n_layer = cfg.num_encoder_layers - self.num_fusion_layers
    frozen_layer_id = int(n_layer * 0.8) - 1
    for lyr in range(n_layer):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      x = t5_network.EncoderLayer(config=cfg, relative_embedding=rel_emb)(
          x, mask_matrix, deterministic=not use_dropout
      )
      if frozen_base and lyr == frozen_layer_id:
        x = jax.lax.stop_gradient(x)
    return x, encoder_mask


class FusedT5Encoder(nn.Module):
  """T5 encoder as a separate model which fuse multi-modal input.

  This module contains the encoder part of a pretrained T5. It is useful when
  adopting the pretrained T5 encoder as a part of a larger network. Note that
  the embedding layer should be created outside the module and provided as a
  parameter `shared_embedding` to share it in other parts of the network (e.g.,
  text encoder). If `shared_embedding` is not provided, the embedding layer is
  created within the module.

  Attributes:
    vocab_size: Size of the vocabulary.
    emb_dim: Size of the embeddings.
    num_heads: Number of attention heads.
    num_encoder_layers: Number of encoder layers.
    num_decoder_layers: Number of decoder layers.
    head_dim: Size of the embeddings in each head.
    mlp_dim: Size of the MLP output embeddings.
    dropout_rate: Dropout rate.
    dtype: Data type.
    mlp_activations: Sequence of activations in MLP.
    logits_via_embedding: Use the embedding weights for computing logits.
  """

  vocab_size: int
  emb_dim: int
  num_heads: int
  num_encoder_layers: int
  num_decoder_layers: int
  num_fusion_layers: int
  head_dim: int
  mlp_dim: int
  dropout_rate: float
  dtype: str = 'bfloat16'
  mlp_activations: Sequence[str] = ('gelu', 'linear')
  logits_via_embedding: bool = False

  def setup(self):
    self.t5_config = t5_network.T5Config(
        vocab_size=self.vocab_size,
        emb_dim=self.emb_dim,
        num_heads=self.num_heads,
        num_encoder_layers=self.num_encoder_layers,
        num_decoder_layers=self.num_decoder_layers,
        head_dim=self.head_dim,
        mlp_dim=self.mlp_dim,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        mlp_activations=self.mlp_activations,
        logits_via_embedding=self.logits_via_embedding,
    )

  @nn.compact
  def __call__(
      self,
      fused_input_embs,
      encoder_input_embs=None,
      encoder_mask=None,
      fused_mask=None,
      att_mask=None,
      use_dropout=True,
      output=False,
  ):
    """Function to fuse text and imaget embedding.

    encode both the encoded text embedding (encoder_input_embs) and
    encoded image embedding (fused_input_embs) together using
    self-attentive Transformer.

    Args:
      fused_input_embs: pre-encoded embeddings of other modalities
      encoder_input_embs: encoded text embedding sequence
      encoder_mask: mask for encoding part
      fused_mask: mask for fusion part
      att_mask: pre-computed attention product to each layer's output
      use_dropout: whether to use dropout.
      output: whether it's output layer.

    Returns:
      Sequence of token embedding after fusion
    """
    cfg = self.t5_config
    if encoder_input_embs is not None:
      x = jnp.concatenate([encoder_input_embs, fused_input_embs], axis=1)
    else:
      x = fused_input_embs
    rel_emb = t5_layers.RelativePositionBiases(
        num_buckets=32,
        max_distance=128,
        num_heads=self.t5_config.num_heads,
        dtype=self.t5_config.dtype,
        embedding_init=nn.initializers.variance_scaling(
            1.0, 'fan_avg', 'uniform'
        ),
    )

    if encoder_mask is not None:
      if fused_mask is None:
        pad_width = fused_input_embs.shape[1]
        fused_mask = jnp.pad(
            array=encoder_mask,
            pad_width=((0, 0), (0, pad_width)),
            mode='constant',
            constant_values=1.0,
        )
      else:
        fused_mask = jnp.concatenate([encoder_mask, fused_mask], axis=1)

    mask_matrix = t5_layers.make_attention_mask(
        fused_mask, fused_mask, dtype=cfg.dtype
    )
    attn_weights_all_layers = []
    for _ in range(
        cfg.num_encoder_layers - self.num_fusion_layers, cfg.num_encoder_layers
    ):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      x, attn_weights = FusionEncoderLayer(
          config=cfg, relative_embedding=rel_emb
      )(
          x,
          encoder_mask=mask_matrix,
          att_mask=att_mask,
          deterministic=not use_dropout,
      )
      attn_weights_all_layers += [attn_weights]
    if output:
      x = t5_layers.LayerNorm(dtype=cfg.dtype)(x)
    if att_mask is not None:
      x = x * att_mask
    x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=not use_dropout)
    return x, fused_mask, attn_weights_all_layers


class FusionEncoderLayer(nn.Module):
  """Transformer encoder layer."""

  config: t5_network.T5Config
  relative_embedding: nn.Module

  @nn.compact
  def __call__(
      self, inputs, att_mask=None, encoder_mask=None, deterministic=False
  ):
    cfg = self.config

    # Relative position embedding as attention biases.
    encoder_bias = self.relative_embedding(
        inputs.shape[-2], inputs.shape[-2], True
    )

    # Attention block.
    assert inputs.ndim == 3
    x = t5_layers.LayerNorm(dtype=cfg.dtype)(inputs)
    if att_mask is not None:
      x = x * att_mask
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    x, attn_weights = MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        float32_logits=cfg.float32_attention_logits,
    )(x, x, encoder_mask, encoder_bias, deterministic=deterministic)
    x = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
        x, deterministic=deterministic
    )
    x = x + inputs

    # MLP block.
    y = t5_layers.LayerNorm(dtype=cfg.dtype)(x)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    y = t5_layers.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
    )(y, deterministic=deterministic)
    y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
        y, deterministic=deterministic
    )
    y = y + x

    return y, attn_weights


class PerceiverEncoder(nn.Module):
  """Reimplementation of Perceiver.

  Perceiver: General Perception with Iterative Attention
  (https://arxiv.org/abs/2103.03206)
  """

  perceiver_output_dim: int
  vocab_size: int
  emb_dim: int
  num_heads: int
  num_encoder_layers: int
  num_decoder_layers: int
  num_fusion_layers: int
  head_dim: int
  mlp_dim: int
  dropout_rate: float
  dtype: str = 'bfloat16'
  mlp_activations: Sequence[str] = ('gelu', 'linear')
  logits_via_embedding: bool = False

  def setup(self):
    self.t5_config = t5_network.T5Config(
        vocab_size=self.vocab_size,
        emb_dim=self.emb_dim,
        num_heads=self.num_heads,
        num_encoder_layers=self.num_encoder_layers,
        num_decoder_layers=self.num_decoder_layers,
        head_dim=self.head_dim,
        mlp_dim=self.mlp_dim,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        mlp_activations=self.mlp_activations,
        logits_via_embedding=self.logits_via_embedding,
    )

    self.perceive_embedding = self.param(
        'perceive_embedding',
        nn.initializers.normal(stddev=1.0),
        (1, self.perceiver_output_dim, self.emb_dim),
        jnp.float32,
    )
    v = jnp.arange(self.perceiver_output_dim)
    self.batch_triangle_select = jax.vmap(
        functools.partial(_mask_select, mask=v < v.reshape([-1, 1]))
    )

  def linear_disentangle(self, y):
    mean = y.mean(axis=-2, keepdims=True)
    norm_y = l2_normalize(y - mean)
    pairwise_mat = jnp.square(jnp.einsum('bqd,btd->bqt', norm_y, norm_y))
    masked_mat = self.batch_triangle_select(pairwise_mat)
    return jnp.mean(masked_mat)

  @nn.compact
  def __call__(self, encoded, encoded_mask, use_dropout=False):
    cfg = self.t5_config
    rel_emb = t5_layers.RelativePositionBiases(
        num_buckets=32,
        max_distance=128,
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.variance_scaling(
            1.0, 'fan_avg', 'uniform'
        ),
    )

    # [batch, length] -> [batch, length, emb_dim]
    encoded = t5_layers.LayerNorm(dtype=cfg.dtype)(encoded)
    bsz = encoded.shape[0]
    y = jnp.asarray(self.perceive_embedding, dtype=cfg.dtype)
    y = jnp.repeat(y, bsz, axis=0)
    y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
        y, deterministic=not use_dropout
    )
    y = y.astype(cfg.dtype)

    mask = jnp.ones([bsz, self.perceiver_output_dim]).astype(bool)
    encoder_decoder_mask = t5_layers.make_attention_mask(
        mask, encoded_mask, dtype=self.dtype
    )

    for _ in range(self.num_fusion_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      y = t5_network.DecoderLayer(config=cfg, relative_embedding=rel_emb)(
          y,
          encoded,
          deterministic=not use_dropout,
          encoder_decoder_mask=encoder_decoder_mask,
          decode=False,
      )

    return y * 4, mask, self.linear_disentangle(y)


def dot_product_attention(
    query: constants.JTensor,
    key: constants.JTensor,
    value: constants.JTensor,
    bias: Optional[constants.JTensor] = None,
    dropout_rng: Optional[constants.JTensor] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: constants.DType = jnp.float32,
    float32_logits: bool = False,
):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Args:
    query: queries for calculating attention with shape of `[batch, q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch, kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch, kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch, num_heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.

  Returns:
    Output of shape `[batch, length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert (
      query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
  ), 'q, k, v batch dims must match.'
  assert (
      query.shape[-2] == key.shape[-2] == value.shape[-2]
  ), 'q, k, v num_heads must match.'
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # Casting logits and softmax computation for float32 for model stability.
  if float32_logits:
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)

  # `attn_weights`: [batch, num_heads, q_length, kv_length]
  attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)

  # Apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias.astype(attn_weights.dtype)

  # Normalize the attention weights across `kv_length` dimension.
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # Apply attention dropout.
  if not deterministic and dropout_rate > 0.0:
    keep_prob = 1.0 - dropout_rate
    # T5 broadcasts along the "length" dim, but unclear which one that
    # corresponds to in positional dimensions here, assuming query dim.
    dropout_shape = list(attn_weights.shape)
    dropout_shape[-2] = 1
    keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    keep = jnp.broadcast_to(keep, attn_weights.shape)
    multiplier = keep.astype(attn_weights.dtype) / jnp.asarray(
        keep_prob, dtype=dtype
    )
    attn_weights = attn_weights * multiplier

  # Take the linear combination of `value`.
  return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value), attn_weights


class MultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention.

  Attributes:
    num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    head_dim: dimension of each head.
    dtype: the dtype of the computation.
    dropout_rate: dropout rate
    kernel_init: initializer for the kernel of the Dense layers.
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.
  """

  num_heads: int
  head_dim: int
  dtype: constants.DType = jnp.float32
  dropout_rate: float = 0.0
  kernel_init: constants.Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'normal'
  )
  float32_logits: bool = False  # computes logits in float32 for stability.

  @nn.compact
  def __call__(
      self,
      inputs_q: constants.JTensor,
      inputs_kv: constants.JTensor,
      mask: Optional[constants.JTensor] = None,
      bias: Optional[constants.JTensor] = None,
      *,
      decode: bool = False,
      deterministic: bool = False,
  ) -> constants.JTensor:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    There are two modes: decoding and non-decoding (e.g., training). The mode is
    determined by `decode` argument. For decoding, this method is called twice,
    first to initialize the cache and then for an actual decoding process. The
    two calls are differentiated by the presence of 'cached_key' in the variable
    dict. In the cache initialization stage, the cache variables are initialized
    as zeros and will be filled in the subsequent decoding process.

    In the cache initialization call, `inputs_q` has a shape [batch, length,
    q_features] and `inputs_kv`: [batch, length, kv_features]. During the
    incremental decoding stage, query, key and value all have the shape [batch,
    1, qkv_features] corresponding to a single step.

    Args:
      inputs_q: input queries of shape `[batch, q_length, q_features]`.
      inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
      mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
      bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
      decode: Whether to prepare and use an autoregressive cache.
      deterministic: Disables dropout if set to True.

    Returns:
      output of shape `[batch, length, q_features]`.
    """
    projection = functools.partial(
        t5_layers.DenseGeneral,
        axis=-1,
        features=(self.num_heads, self.head_dim),
        kernel_axes=('embed', 'joined_kv'),
        dtype=self.dtype,
    )

    # NOTE: T5 does not explicitly rescale the attention logits by
    #       1/sqrt(depth_kq)!  This is folded into the initializers of the
    #       linear transformations, which is equivalent under Adafactor.
    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    query_init = lambda *args: self.kernel_init(*args) / depth_scaling

    # Project inputs_q to multi-headed q/k/v
    # dimensions are then [batch, length, num_heads, head_dim]
    query = projection(kernel_init=query_init)(inputs_q)
    key = projection(kernel_init=self.kernel_init)(inputs_kv)
    value = projection(kernel_init=self.kernel_init)(inputs_kv)

    query = t5_layers.with_sharding_constraint(
        query, ('batch', 'length', 'heads', 'kv')
    )
    key = t5_layers.with_sharding_constraint(
        key, ('batch', 'length', 'heads', 'kv')
    )
    value = t5_layers.with_sharding_constraint(
        value, ('batch', 'length', 'heads', 'kv')
    )

    if decode:
      # Detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      # The key and value have dimension [batch, length, num_heads, head_dim],
      # but we cache them as [batch, num_heads, head_dim, length] as a TPU
      # fusion optimization. This also enables the "scatter via one-hot
      # broadcast" trick, which means we do a one-hot broadcast instead of a
      # scatter/gather operations, resulting in a 3-4x speedup in practice.
      swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])
      cached_key = self.variable(
          'cache', 'cached_key', jnp.zeros, swap_dims(key.shape), key.dtype
      )
      cached_value = self.variable(
          'cache',
          'cached_value',
          jnp.zeros,
          swap_dims(value.shape),
          value.dtype,
      )
      cache_index = self.variable(
          'cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.int32)
      )
      if is_initialized:
        batch, num_heads, head_dim, length = cached_key.value.shape
        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        # Sanity shape check of cached key against input query.
        expected_shape = (batch, 1, num_heads, head_dim)
        if expected_shape != query.shape:
          raise ValueError(
              'Autoregressive cache shape error, '
              'expected query shape %s instead got %s.'
              % (expected_shape, query.shape)
          )

        # Create a OHE of the current index. NOTE: the index is increased below.
        cur_index = cache_index.value
        one_hot_indices = jax.nn.one_hot(cur_index, length, dtype=key.dtype)
        # In order to update the key, value caches with the current key and
        # value, we move the length axis to the back, similar to what we did for
        # the cached ones above.
        # Note these are currently the key and value of a single position, since
        # we feed one position at a time.
        one_token_key = jnp.moveaxis(key, -3, -1)
        one_token_value = jnp.moveaxis(value, -3, -1)
        # Update key, value caches with our new 1d spatial slices.
        # We implement an efficient scatter into the cache via one-hot
        # broadcast and addition.
        key = cached_key.value + one_token_key * one_hot_indices
        value = cached_value.value + one_token_value * one_hot_indices
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # Move the keys and values back to their original shapes.
        key = jnp.moveaxis(key, -1, -3)
        value = jnp.moveaxis(value, -1, -3)

        # Causal mask for cached decoder self-attention: our single query
        # position should only attend to those key positions that have already
        # been generated and cached, not the remaining zero elements.
        mask = t5_layers.combine_masks(
            mask,
            jnp.broadcast_to(
                jnp.arange(length) <= cur_index,
                # (1, 1, length) represent (head dim, query length, key length)
                # query length is 1 because during decoding we deal with one
                # index.
                # The same mask is applied to all batch elements and heads.
                (batch, 1, 1, length),
            ),
        )

        # Grab the correct relative attention bias during decoding. This is
        # only required during single step decoding.
        if bias is not None:
          # The bias is a full attention matrix, but during decoding we only
          # have to take a slice of it.
          # This is equivalent to bias[..., cur_index:cur_index+1, :].
          bias = t5_layers.dynamic_vector_slice_in_dim(
              jnp.squeeze(bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2
          )

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = jax.lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.0).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype),
      )
    else:
      attention_bias = None

    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = t5_layers.combine_biases(attention_bias, bias)

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.0:
      dropout_rng = self.make_rng('dropout')

    # Apply attention.
    x, attn_weights = dot_product_attention(
        query,
        key,
        value,
        bias=attention_bias,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        deterministic=deterministic,
        dtype=self.dtype,
        float32_logits=self.float32_logits,
    )

    # Back to the original inputs dimensions.
    out = t5_layers.DenseGeneral(
        features=inputs_q.shape[-1],  # output dim is set to the input dim.
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        kernel_axes=('joined_kv', 'embed'),
        dtype=self.dtype,
    )(x)
    return out, attn_weights
