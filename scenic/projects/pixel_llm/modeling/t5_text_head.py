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

"""Wrapper of T5 text decoder."""

import functools
from typing import Optional

from flax import struct
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from scenic.projects.t5 import model as t5_pretrained
from scenic.projects.t5.layers import t5
from scenic.projects.t5.layers import t5_layers

param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint

# Type annotations
Array = jnp.ndarray


@struct.dataclass
class CustomT5Config(t5.T5Config):
  encoder_lora_rank: int = 0
  decoder_lora_rank: int = 0
  encoder_lora_scale: float = 1.0
  decoder_lora_scale: float = 1.0
  encoder_lora_modules: str = 'q,v'
  decoder_lora_modules: str = 'q,v'


class LoRADenseGeneral(t5_layers.DenseGeneral):
  """A linear transformation (without bias) with flexible axes.


    Attributes:
      features: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
  """
  rank: int = 4
  scale: float = 1.0

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    features = t5_layers._canonicalize_tuple(self.features)
    axis = t5_layers._canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = t5_layers._normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
    kernel_param_shape = (
        np.prod([inputs.shape[ax] for ax in axis]),
        np.prod(features),
    )
    kernel = param_with_axes(
        'kernel',
        self.kernel_init,
        kernel_param_shape,
        jnp.float32,
        axes=self.kernel_axes,
    )
    kernel = jnp.asarray(kernel, self.dtype)
    kernel = jnp.reshape(kernel, kernel_shape)

    # begin LoRA code
    kernel_left_shape = tuple([inputs.shape[ax] for ax in axis] + [self.rank])
    kernel_left_param_shape = (
        np.prod([inputs.shape[ax] for ax in axis]),
        self.rank,
    )
    kernel_right_shape = tuple([self.rank] + list(features))
    kernel_right_param_shape = (self.rank, np.prod(features))
    kernel_left_axis_names = list(
        self.kernel_axes[: len(kernel_left_param_shape) - 1]
    ) + ['stack']
    kernel_right_axis_names = ['stack'] + list(
        self.kernel_axes[-(len(kernel_right_param_shape) - 1) :]
    )
    kernel_left = param_with_axes(
        'kernel_left_lora',
        self.kernel_init,
        kernel_left_param_shape,
        jnp.float32,
        axes=tuple(kernel_left_axis_names),
    )
    kernel_right = param_with_axes(
        'kernel_right_lora',
        nn.initializers.zeros_init(),
        kernel_right_param_shape,
        jnp.float32,
        axes=tuple(kernel_right_axis_names),
    )

    kernel_left = jnp.asarray(kernel_left, self.dtype)
    kernel_left = jnp.reshape(kernel_left, kernel_left_shape)

    kernel_right = jnp.asarray(kernel_right, self.dtype)
    kernel_right = jnp.reshape(kernel_right, kernel_right_shape)
    einsum_str = 'abcdefghijklmnopqrstuvwxy'
    assert len(features) <= len(einsum_str)
    feat_einsum = einsum_str[: len(features)]
    kernel_delta = jnp.einsum(f'...z, z{feat_einsum}->...{feat_einsum}',
                              kernel_left, kernel_right)
    kernel = kernel + self.scale * kernel_delta
    # end LoRA code

    contract_ind = tuple(range(0, len(axis)))
    return lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))


class CustomMultiHeadDotProductAttention(
    t5_layers.MultiHeadDotProductAttention
):
  """MultiHeadDotProductAttention that supports LoRA.
  """

  lora_rank: int = 0
  lora_scale: float = 1.0
  lora_modules: str = 'q,v'

  @nn.compact
  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      mask: Optional[Array] = None,
      bias: Optional[Array] = None,
      *,
      decode: bool = False,
      deterministic: bool = False,
  ) -> Array:
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
    if self.lora_rank > 0:
      lora_projection = functools.partial(
          LoRADenseGeneral,
          rank=self.lora_rank,
          scale=self.lora_scale,
          axis=-1,
          features=(self.num_heads, self.head_dim),
          kernel_axes=('embed', 'joined_kv'),
          dtype=self.dtype)
    else:
      lora_projection = None
    projection = functools.partial(
        t5_layers.DenseGeneral,
        axis=-1,
        features=(self.num_heads, self.head_dim),
        kernel_axes=('embed', 'joined_kv'),
        dtype=self.dtype)
    projection_q = projection
    projection_k = projection
    projection_v = projection
    if lora_projection is not None:
      if 'q' in self.lora_modules:
        projection_q = lora_projection
      if 'k' in self.lora_modules:
        projection_k = lora_projection
      if 'v' in self.lora_modules:
        projection_v = lora_projection

    # NOTE: T5 does not explicitly rescale the attention logits by
    #       1/sqrt(depth_kq)!  This is folded into the initializers of the
    #       linear transformations, which is equivalent under Adafactor.
    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    query_init = lambda *args: self.kernel_init(*args) / depth_scaling

    # Project inputs_q to multi-headed q/k/v
    # dimensions are then [batch, length, num_heads, head_dim]
    query = projection_q(kernel_init=query_init, name='query')(inputs_q)
    key = projection_k(kernel_init=self.kernel_init, name='key')(inputs_kv)
    value = projection_v(kernel_init=self.kernel_init, name='value')(inputs_kv)

    query = with_sharding_constraint(query, ('batch', 'length', 'heads', 'kv'))
    key = with_sharding_constraint(key, ('batch', 'length', 'heads', 'kv'))
    value = with_sharding_constraint(value, ('batch', 'length', 'heads', 'kv'))

    if decode:
      # Detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      # The key and value have dimension [batch, length, num_heads, head_dim],
      # but we cache them as [batch, num_heads, head_dim, length] as a TPU
      # fusion optimization. This also enables the "scatter via one-hot
      # broadcast" trick, which means we do a one-hot broadcast instead of a
      # scatter/gather operations, resulting in a 3-4x speedup in practice.
      swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])
      cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                                 swap_dims(key.shape), key.dtype)
      cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                   swap_dims(value.shape), value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        batch, num_heads, head_dim, length = (cached_key.value.shape)
        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        # Sanity shape check of cached key against input query.
        expected_shape = (batch, 1, num_heads, head_dim)
        if expected_shape != query.shape:
          raise ValueError('Autoregressive cache shape error, '
                           'expected query shape %s instead got %s.' %
                           (expected_shape, query.shape))

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
                (batch, 1, 1, length)))

        # Grab the correct relative attention bias during decoding. This is
        # only required during single step decoding.
        if bias is not None:
          # The bias is a full attention matrix, but during decoding we only
          # have to take a slice of it.
          # This is equivalent to bias[..., cur_index:cur_index+1, :].
          bias = t5_layers.dynamic_vector_slice_in_dim(
              jnp.squeeze(bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2)

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype))
    else:
      attention_bias = None

    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = t5_layers.combine_biases(attention_bias, bias)

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # Apply attention.
    x = t5_layers.dot_product_attention(
        query,
        key,
        value,
        bias=attention_bias,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        deterministic=deterministic,
        dtype=self.dtype,
        float32_logits=self.float32_logits)

    # Back to the original inputs dimensions.
    if self.lora_rank > 0 and  'o' in self.lora_modules:
      out = LoRADenseGeneral(
          features=inputs_q.shape[-1],  # output dim is set to the input dim.
          axis=(-2, -1),
          kernel_init=self.kernel_init,
          kernel_axes=('joined_kv', 'embed'),
          dtype=self.dtype,
          rank=self.lora_rank,
          scale=self.lora_scale,
          name='out')(
              x)
    else:
      out = t5_layers.DenseGeneral(
          features=inputs_q.shape[-1],  # output dim is set to the input dim.
          axis=(-2, -1),
          kernel_init=self.kernel_init,
          kernel_axes=('joined_kv', 'embed'),
          dtype=self.dtype,
          name='out')(
              x)
    return out


class CustomEncoderLayer(t5.EncoderLayer):
  """Encoder layer that support LoRA."""

  config: CustomT5Config

  @nn.compact
  def __call__(self, inputs, encoder_mask=None, deterministic=False):
    cfg = self.config

    # Relative position embedding as attention biases.
    encoder_bias = self.relative_embedding(inputs.shape[-2], inputs.shape[-2],
                                           True)

    # Attention block.
    assert inputs.ndim == 3
    x = t5_layers.LayerNorm(
        dtype=cfg.dtype, name='pre_attention_layer_norm')(
            inputs)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    x = CustomMultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        float32_logits=cfg.float32_attention_logits,
        lora_rank=cfg.encoder_lora_rank,
        lora_scale=cfg.encoder_lora_scale,
        lora_modules=cfg.encoder_lora_modules,
        name='attention')(
            x, x, encoder_mask, encoder_bias, deterministic=deterministic)
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = t5_layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(x)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    y = t5_layers.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
    )(y, deterministic=deterministic)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y + x

    return y


class CustomDecoderLayer(t5.DecoderLayer):
  """Decoder layer that supports LoRA."""
  config: CustomT5Config

  @nn.compact
  def __call__(self,
               inputs,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decode=False,
               max_decode_length=None):
    cfg = self.config

    # Relative position embedding as attention biases.
    l = max_decode_length if decode and max_decode_length else inputs.shape[-2]
    decoder_bias = self.relative_embedding(l, l, False)

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    x = t5_layers.LayerNorm(
        dtype=cfg.dtype, name='pre_self_attention_layer_norm')(
            inputs)

    # Self-attention block
    x = CustomMultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        float32_logits=cfg.float32_attention_logits,
        lora_rank=cfg.decoder_lora_rank,
        lora_scale=cfg.decoder_lora_scale,
        lora_modules=cfg.decoder_lora_modules,
        name='self_attention')(
            x,
            x,
            decoder_mask,
            decoder_bias,
            deterministic=deterministic,
            decode=decode)
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x + inputs

    # Encoder-Decoder block.
    y = t5_layers.LayerNorm(
        dtype=cfg.dtype, name='pre_cross_attention_layer_norm')(
            x)
    y = CustomMultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        float32_logits=cfg.float32_attention_logits,
        lora_rank=cfg.decoder_lora_rank,
        lora_scale=cfg.decoder_lora_scale,
        lora_modules=cfg.decoder_lora_modules,
        name='encoder_decoder_attention')(
            y, encoded, encoder_decoder_mask, deterministic=deterministic)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y + x

    # MLP block.
    z = t5_layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(y)
    z = t5_layers.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
    )(z, deterministic=deterministic)
    z = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            z, deterministic=deterministic)
    z = z + y

    return z


class CustomEncoder(t5.Encoder):
  """Encoder that accepts visual embeddings as input."""

  config: CustomT5Config

  @nn.compact
  def __call__(self,
               encoder_input_embeddings,
               encoder_input_tokens=None,
               encoder_mask=None,
               deterministic=False):  # pytype: disable=signature-mismatch
    """Run encoder.

    Args:
      encoder_input_embeddings: (batch_size, num_vision_tokens, dim). The token
        feature after the embedding layer or the features from other moduels
        (e.g., from vision encoder).
      encoder_input_tokens: (batch_size, num_context_tokens), int. The text
        token IDs. We will concatenate the `encoder_input_tokens` at the end of
        the existing `encoder_input_embeddings`.
      encoder_mask: (batch_size, num_total_tokens), padding mask of
        `encoder_input_tokens`.
      deterministic: bool
    Returns:
      output: (batch_size, num_total_tokens, dim)
    """
    cfg = self.config
    rel_emb = t5_layers.RelativePositionBiases(
        num_buckets=32,
        max_distance=128,
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.variance_scaling(1.0, 'fan_avg',
                                                        'uniform'),
        name='relpos_bias')

    if encoder_input_tokens is not None:
      assert encoder_input_tokens.ndim == 2  # [batch, length]
      # [batch, length] -> [batch, length, emb_dim]
      x = self.shared_embedding(encoder_input_tokens.astype('int32'))
      x = nn.Dropout(
          rate=cfg.dropout_rate, broadcast_dims=(-2,))(
              x, deterministic=deterministic)
      x = x.astype(cfg.dtype)
      x = jnp.concatenate([encoder_input_embeddings, x], axis=1)
    else:
      x = encoder_input_embeddings

    for lyr in range(cfg.num_encoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      x = CustomEncoderLayer(
          config=cfg, relative_embedding=rel_emb,
          name=f'layers_{lyr}')(x, encoder_mask, deterministic)

    x = t5_layers.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(x)
    return nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)


class CustomDecoder(t5.Decoder):
  """Decoder that returns features before word logits."""

  config: CustomT5Config

  @nn.compact
  def __call__(self,
               encoded,
               decoder_input_tokens,
               decoder_positions=None,
               decoder_mask=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decode=False,
               max_decode_length=None,
               return_logit_and_feat=True):
    cfg = self.config
    assert decoder_input_tokens.ndim == 2  # [batch, len]
    rel_emb = t5_layers.RelativePositionBiases(
        num_buckets=32,
        max_distance=128,
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.variance_scaling(1.0, 'fan_avg',
                                                        'uniform'),
        name='relpos_bias')

    # [batch, length] -> [batch, length, emb_dim]
    y = self.shared_embedding(decoder_input_tokens.astype('int32'))
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    for lyr in range(cfg.num_decoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      y = CustomDecoderLayer(
          config=cfg,
          relative_embedding=rel_emb,
          name=f'layers_{lyr}')(
              y,
              encoded,
              decoder_mask=decoder_mask,
              encoder_decoder_mask=encoder_decoder_mask,
              deterministic=deterministic,
              decode=decode,
              max_decode_length=max_decode_length)

    y = t5_layers.LayerNorm(dtype=cfg.dtype, name='decoder_norm')(y)
    decode_feat = y
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.shared_embedding.attend(y)
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = t5_layers.DenseGeneral(
          cfg.vocab_size,
          dtype=jnp.float32,  # Use float32 for stabiliity.
          kernel_axes=('embed', 'vocab'),
          name='logits_dense')(
              y)
    if return_logit_and_feat:
      return logits, decode_feat
    return logits


class CustomTransformer(t5.Transformer):
  """T5 Transformer that accepts visual embeddings as input."""

  config: CustomT5Config

  def setup(self):
    cfg = self.config
    self.shared_embedding = t5_layers.Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=True,
        name='token_embedder')

    self.encoder = CustomEncoder(
        config=cfg,
        shared_embedding=self.shared_embedding,
    )
    self.decoder = CustomDecoder(
        config=cfg,
        shared_embedding=self.shared_embedding,
    )

  def encode(self,
             encoder_input_embeddings,
             encoder_input_tokens=None,
             enable_dropout=True):  # pytype: disable=signature-mismatch
    """Applies Transformer encoder-branch on the inputs."""
    cfg = self.config
    visual_mask = jnp.ones(encoder_input_embeddings.shape[:2], dtype=bool)
    if encoder_input_tokens is not None:
      assert encoder_input_tokens.ndim == 2, (
          f'Expected `encoder_input_tokens` to be of shape (batch, len). '
          f'Got {encoder_input_tokens.shape}')
      # Make padding attention mask.
      context_mask = encoder_input_tokens > 0
      valid_mask = jnp.concatenate([visual_mask, context_mask], axis=1)
      encoder_mask = t5_layers.make_attention_mask(
          valid_mask, valid_mask, dtype=cfg.dtype)
    else:
      encoder_mask = None
      valid_mask = visual_mask

    return self.encoder(
        encoder_input_embeddings,
        encoder_input_tokens,
        encoder_mask,
        deterministic=not enable_dropout), valid_mask

  def decode(
      self,
      encoded,
      encoder_input_tokens,  # only needed for masks
      decoder_input_tokens,
      decoder_target_tokens,
      encoder_segment_ids=None,
      decoder_segment_ids=None,
      decoder_positions=None,
      enable_dropout=True,
      decode=False,
      max_decode_length=None,
      return_logit_and_feat=False):
    """Applies Transformer decoder-branch on encoded-input and target."""
    cfg = self.config

    # Make padding attention masks.
    if decode:
      # Do not mask decoder attention based on targets padding at
      # decoding/inference time.
      decoder_mask = None
      encoder_decoder_mask = t5_layers.make_attention_mask(
          jnp.ones_like(decoder_target_tokens),
          encoder_input_tokens > 0,
          dtype=cfg.dtype)
    else:
      decoder_mask = t5_layers.make_decoder_mask(
          decoder_target_tokens=decoder_target_tokens,
          dtype=cfg.dtype,
          decoder_segment_ids=decoder_segment_ids)
      encoder_decoder_mask = t5_layers.make_attention_mask(
          decoder_target_tokens > 0, encoder_input_tokens > 0, dtype=cfg.dtype)

    # Add segmentation block-diagonal attention masks if using segmented data.
    if encoder_segment_ids is not None:
      if decode:
        raise ValueError(
            'During decoding, packing should not be used but '
            '`encoder_segment_ids` was passed to `Transformer.decode`.')

      encoder_decoder_mask = t5_layers.combine_masks(
          encoder_decoder_mask,
          t5_layers.make_attention_mask(
              decoder_segment_ids,
              encoder_segment_ids,
              jnp.equal,
              dtype=cfg.dtype))

    ret = self.decoder(
        encoded,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        deterministic=not enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
        return_logit_and_feat=return_logit_and_feat)
    return ret

  @nn.compact
  def __call__(
      self, text_tokens, visual_features,
      context_tokens=None, train=False,
      return_feat=False, return_logit_and_feat=False):  # pytype: disable=signature-mismatch
    """Generate logits of a single word.

    Args:
      text_tokens: (batch_size, caption_length). text_tokens[0] = BOS.
      visual_features: (batch_size, feature_length, feat_size).
      context_tokens:  (batch_size, context_length).
      train: bool.
      return_feat: bool. If true, return the feature before vocabulary.
      return_logit_and_feat: bool

    Returns:
      output_logits: (batch_size, caption_length, vocab_size).
    """
    encoded, encoder_valid_mask = self.encode(
        encoder_input_embeddings=visual_features,
        encoder_input_tokens=context_tokens,
        enable_dropout=train,
    )

    # T5 decoder needs a "decoder_target" input to compute the attention mask.
    # This is the target sentence without the BOS token. We pad it with 0 to
    # retain the same length as the input tokens.
    decoder_target = jnp.concatenate(
        [text_tokens[:, 1:],
         jnp.zeros((text_tokens.shape[0], 1), dtype=jnp.int32)],
        axis=1)
    output_logits, output_feat = self.decode(
        encoded,
        encoder_valid_mask,  # only needed for masks
        text_tokens,
        decoder_target,
        enable_dropout=train,
        decode=False,
        return_logit_and_feat=True,
    )
    if return_feat:
      return output_feat
    if return_logit_and_feat:
      return output_logits, output_feat
    return output_logits


class T5TextualHead(nn.Module):
  """Wrapper of T5 text decoder."""
  t5_model: str = 'flan_t5_small'
  dtype: str = 'bfloat16'
  dropout_rate: float = 0.0
  vocab_size: int = 32128
  encoder_lora_rank: int = 0
  decoder_lora_rank: int = 0
  encoder_lora_scale: float = 1.0
  decoder_lora_scale: float = 1.0
  encoder_lora_modules: str = 'q,v'
  decoder_lora_modules: str = 'q,v'

  @nn.compact
  def __call__(
      self, text_tokens, visual_features,
      context_tokens=None, train=False, return_feat=False,
      return_logit_and_feat=False):
    """Generate logits of a single word.

    Args:
      text_tokens: (batch_size, caption_length).
      visual_features: (batch_size, feature_length, feat_size).
      context_tokens:  (batch_size, context_length).
      train: bool.
      return_feat: bool. If true, return the feature before vocabulary.
      return_logit_and_feat: bool
    Returns:
      output_logits: (batch_size, caption_length, vocab_size).
    """
    config_dict = t5_pretrained.CONFIGS[self.t5_model]
    config_dict['dtype'] = self.dtype
    config_dict['dropout_rate'] = self.dropout_rate
    config_dict['vocab_size'] = self.vocab_size
    config_dict['encoder_lora_rank'] = self.encoder_lora_rank
    config_dict['decoder_lora_rank'] = self.decoder_lora_rank
    config_dict['encoder_lora_scale'] = self.encoder_lora_scale
    config_dict['decoder_lora_scale'] = self.decoder_lora_scale
    config_dict['encoder_lora_modules'] = self.encoder_lora_modules
    config_dict['decoder_lora_modules'] = self.decoder_lora_modules
    t5_config = CustomT5Config(**config_dict)

    return CustomTransformer(
        t5_config,
        name='t5_module',
    )(
        text_tokens,
        visual_features,
        context_tokens,
        train,
        return_feat,
        return_logit_and_feat=return_logit_and_feat,
    )
