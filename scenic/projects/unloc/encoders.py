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

"""Wraps video and text encoders to have the same API.

All video encoders have the same __call__ function as follows:

@nn.compact
def __call__(self, x: jnp.ndarray, *, train: bool, debug: bool = False):
  # implementation goes here

All text encoders have the same __call__ function as follows:

@nn.compact
def __call__(self, x: Dict[str, jnp.ndarray],, *, train: bool, debug: bool =
False):
  # implementation goes here
"""

import functools
from typing import Callable, Dict, Optional, Sequence

from absl import logging
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import vit
from scenic.projects.baselines.clip import layers as clip_layers


Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


class TransformerEncoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of self-attention heads.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value.

  Returns:
    output after transformer encoder block.
  """

  mlp_dim: int
  num_heads: int
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  dtype: jnp.dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      input_mask: Optional[jnp.ndarray] = None,
      deterministic: bool = False,
  ) -> jnp.ndarray:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data of shape (batch, sequence_length, channels).
      input_mask: Input mask of shape (batch, sequence_length). Only applicable
        for text encoder.
      deterministic: Deterministic or not (to apply dropout).

    Returns:
      Output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=self.dtype)(inputs)

    if input_mask is not None:
      attention_mask = input_mask[:, None, None, :] * jnp.ones(
          [1, 1, x.shape[1], 1]
      )
    else:
      attention_mask = None

    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        name='MultiHeadAttention_0',
    )(x, x, mask=attention_mask, deterministic=deterministic)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = nn_layers.StochasticDepth(rate=self.stochastic_depth)(x, deterministic)
    x = x + inputs
    # We don't want to overwrite x for residual connection.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = attention_layers.MlpBlock(  # pytype: disable=wrong-arg-types  # jnp-type
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )(y, deterministic=deterministic)
    y = nn_layers.StochasticDepth(rate=self.stochastic_depth)(y, deterministic)
    y = y + x
    return y


class TransformerEncoder(nn.Module):
  """Transformer encoder.

  Attributes:
    num_layers: Number of layers.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of heads.
    positional_embedding: 'learned', 'sinusoid', or 'none'.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Attention dropout rate.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value. Our implementation of stochastic depth follows the
      timm library, which does per-example layer dropping and uses independent
      dropping patterns for each skip-connection.
    dtype: Dtype of activations.
  """

  num_layers: int
  mlp_dim: int
  num_heads: int
  positional_embedding: str = 'learned'
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  dtype: jnp.dtype = jnp.float32

  def _add_positional_embedding(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Adds positional embedding."""
    posemb = jnp.zeros_like(inputs)
    if self.positional_embedding == 'learned':
      posemb = vit.AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='posembed_input',
      )(posemb)
    elif self.positional_embedding == 'sinusoid':
      posemb = attention_layers.Add1DPositionEmbedding(posemb_init=None)(posemb)
    elif self.positional_embedding == 'none':
      logging.info('No positional embedding is used.')
    else:
      raise ValueError(
          f'Invalid positional_embedding: {self.positional_embedding}.'
      )

    inputs += posemb
    return inputs

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      input_mask: Optional[jnp.ndarray] = None,
      train: bool = False,
  ):
    """Applies Transformer model on the inputs."""

    assert inputs.ndim == 3  # Shape is `[batch, len, emb]`.
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)
    x = self._add_positional_embedding(inputs)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
    for lyr in range(self.num_layers):
      x = TransformerEncoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_depth=(lyr / max(self.num_layers - 1, 1))
          * self.stochastic_depth,
          name=f'encoderblock_{lyr}',
          dtype=dtype,
      )(x, input_mask=input_mask, deterministic=not train)
    x = nn.LayerNorm(name='encoder_norm')(x)
    return x


class ResidualAttentionBlock(nn.Module):
  """Self-attention block of Transformer.

  Branched from clip_layers with additional attribute `stochastic_depth` added.

  Attributes:
    num_heads: Number of heads.
    stochastic_depth: Probability of dropping a layer.
  """
  num_heads: int
  stochastic_depth: float = 0.0

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      attn_mask: Optional[jnp.ndarray],
      deterministic: bool,
  ) -> jnp.ndarray:
    xn = clip_layers.LayerNorm(name='ln_1')(x)
    xn = nn.SelfAttention(
        self.num_heads, name='attn', deterministic=True)(xn, attn_mask)
    xn = nn_layers.StochasticDepth(rate=self.stochastic_depth)(xn,
                                                               deterministic)
    x = x + xn

    y = clip_layers.LayerNorm(name='ln_2')(x)
    y = clip_layers.MLP(name='mlp')(y)
    y = nn_layers.StochasticDepth(rate=self.stochastic_depth)(y, deterministic)
    return x + y

  @functools.partial(nn.remat, static_argnums=(3,))
  def remat_call(
      self,
      x: jnp.ndarray,
      attn_mask: Optional[jnp.ndarray],
      deterministic: bool,
  ) -> jnp.ndarray:
    return self(x, attn_mask, deterministic)


class ClipTransformer(nn.Module):
  """Clip Transformer module.

  Attributes:
    features: Number of features.
    num_layers: Number of layers for each block.
    num_heads: Number of heads.
    use_underscore_module_name: Optionally replace '.' with '_' in parameter
      naming for PAX checkpoint loading. This follows `Transformer` defined in
      third_party/py/scenic/projects/baselines/clip/layers.py.
    stochastic_depth: Probability of dropping a layer linearly grows from 0 to
      the provided value.
  """

  features: int
  num_layers: int
  num_heads: int
  use_underscore_module_name: bool = False
  stochastic_depth: float = 0.0
  remat_block: bool = False

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               attn_mask: Optional[jnp.ndarray] = None,
               train: bool = False) -> jnp.ndarray:

    def _n(name):
      """A helper function that optionally replace '.' with '_'."""
      if self.use_underscore_module_name:
        return name.replace('.', '_')
      else:
        return name

    for i in range(self.num_layers):
      sd = (i / max(self.num_layers - 1, 1)) * self.stochastic_depth
      block = ResidualAttentionBlock(
          num_heads=self.num_heads,
          stochastic_depth=sd,
          name=_n(f'resblocks.{i}'),
      )
      if self.remat_block:
        x = block.remat_call(x, attn_mask, not train)
      else:
        x = block(x, attn_mask, not train)
    return x


class ClipVisionTransformer(nn.Module):
  r"""Clip Vision Transformer.

  This class is branched from third_party/py/scenic/projects/baselines/clip/\
  layers.py. The difference is that in the __call__ function we pass in the
  class_embedding because we want each frame to have a different class
  embedding.

  Attributes:
    patches: patches.size is a three element tuple representing the tubelet
      sizes as (height, width, time).
    features: Number of features.
    num_layers: Number of transformer blocks (self-attn + MLP).
    num_heads: Number of attention heads.
    out_features: Number of output features. If None, return transformer output.
    classifier: 'token' or 'gap'.
    stochastic_depth: Probability of dropping a layer linearly grows from 0 to
      the provided value.
  """
  patches: ml_collections.ConfigDict
  features: int
  num_layers: int
  num_heads: int
  out_features: Optional[int] = None
  classifier: str = 'token'
  stochastic_depth: float = 0.0
  remat_block: bool = False
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               class_embedding: Optional[jnp.ndarray] = None,
               attn_mask: Optional[jnp.ndarray] = None,
               is_train: bool = False) -> jnp.ndarray:
    """Executes the transformer encoder.

    Args:
      x: A 3D float tensor of shape (batch_size, sequence_length, features)
        representing the tubelet tokens.
      class_embedding: 1D float tensor of shape (features,) representing the
        class embedding. This is only used when classifier = `token`.
      attn_mask: Optional. Attention mask.
      is_train: Whether or not the model is in training.

    Returns:
      Encoded tokens. They have a shape of (batch_size, sequence_length,
      features) if out_features is None and (batch_size, out_features)
      otherwise.
    """
    if self.classifier == 'token':
      x = jnp.concatenate((jnp.tile(class_embedding[None, None, :],
                                    (x.shape[0], 1, 1)), x),
                          axis=1)
    scale = 1.0 / jnp.sqrt(self.features)
    positional_embedding = self.param('positional_embedding',
                                      jax.nn.initializers.normal(stddev=scale),
                                      (x.shape[1], self.features), x.dtype)
    x = x + positional_embedding[None]

    x = clip_layers.LayerNorm(dtype=self.dtype, name='ln_pre')(x)
    x = ClipTransformer(
        features=self.features,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        stochastic_depth=self.stochastic_depth,
        remat_block=self.remat_block,
        name='transformer',
    )(x, attn_mask=attn_mask, train=is_train)

    if self.out_features is not None:
      x = clip_layers.LayerNorm(dtype=self.dtype, name='ln_post')(x[:, 0])
      x = nn.Dense(
          self.out_features, use_bias=False, dtype=self.dtype, name='proj')(
              x)
    else:
      x = clip_layers.LayerNorm(dtype=self.dtype, name='ln_post')(x)

    return x


class ClipVideoTower(nn.Module):
  """Implements CLIP video tower.

  Attributes:
    num_classes: Number of output classes.
    image_encoder_config: Configuration of the frame encoder.
    temporal_encoder_config: Configuration of the temporal encoder.
    classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
      'token'.
    representation_size: Size of the representation layer in the model's head.
      if None, we skip the extra projection + tanh activation at the end.
    final_endpoint: The name of the output endpoint, 'logits',
      'temporal_tokens', or 'pre_logits'. When final_endpoint is 'logits', the
      output has a shape of (batch_size, num_classes). When final_endpoint is
      'temporal_tokens', the output shape is (batch_size, time, channels). When
      final_endpoint is 'pre_logits', the output shape is (batch_size, time,
      height, width, channels) when keep_spatiotemporal_features is True and
      (batch_size, channels) when it is False.
    dtype: JAX data type for activations.
  """

  num_classes: int
  image_encoder_config: ml_collections.ConfigDict
  temporal_encoding_config: ml_collections.ConfigDict
  temporal_encoder_config: Optional[ml_collections.ConfigDict] = None
  representation_size: Optional[int] = None
  classifier: str = 'token'
  final_endpoint: str = 'logits'
  dtype: jnp.dtype = jnp.float32

  def _add_cls_token(self, x: jnp.ndarray) -> jnp.ndarray:
    """Prepends CLS token.

    Args:
      x: A 3D float tensor of shape (batch, sequence_len, channels) representing
        the tokens.

    Returns:
      A 3D float tensor with prepended CLS token. Its new shape is (batch,
      sequence_len+1, channels).
    """
    if self.classifier == 'token':
      bs, _, c = x.shape
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      cls = jnp.tile(cls, [bs, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
    return x

  def _extract_encoder_output(self,
                              x: jnp.ndarray,
                              axis: int = 1) -> jnp.ndarray:
    """Extracts encoder output."""
    if self.classifier in ['token', '0']:
      x = x.take(indices=0, axis=axis)
    elif self.classifier in ('gap', 'gmp', 'gsp'):
      fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
      x = fn(x, axis=list(range(axis, x.ndim - 1)))
    else:
      raise ValueError(f'Unknown classifier `{self.classifier}`.')
    return x

  def _temporal_encode(self, x: jnp.ndarray, is_train: bool) -> jnp.ndarray:
    """Encodes the tokens in the temporal dimension."""
    # CLIP uses CLS token as embeddings.
    if self.image_encoder_config.get('classifier', 'token') == 'token':
      x = x[:, :, 0]
    else:
      x = jnp.mean(x, axis=2)
    if self.temporal_encoder_config is None:
      return jnp.mean(x, axis=1)
    temporal_encoder = TransformerEncoder(
        dtype=self.dtype,
        name='TemporalTransformer',
        **self.temporal_encoder_config,  # pylint: disable=not-a-mapping
    )  # pylint:disable=not-a-mapping
    self._add_cls_token(x)
    x = temporal_encoder(x, train=is_train)
    x = self._extract_encoder_output(x, axis=1)
    return x

  def _image_to_patch(self, vid: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    """Converts an image to patches.

    Args:
      vid: A 5D tensor of shape [B, T, H, W, C].
      patch_size: integer, dimension of a square patch.

    Returns:
      Flattened patches of shape [B, T, (H * W / P^2), P^2 * C].
    """

    _, height, width, channels = vid.shape[1:]

    if height % patch_size != 0 or width % patch_size != 0:
      raise ValueError(
          f'Image height ({height}) and width ({width}) should be multiples '
          f'of patch_size ({patch_size}).'
      )

    row_blocks = height // patch_size
    column_blocks = width // patch_size

    return einops.rearrange(
        vid,
        '... (m p)(n q) c->...(m n)(p q c)',
        m=row_blocks,
        n=column_blocks,
        p=patch_size,
        q=patch_size,
        c=channels,
    )

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, debug: bool = False):
    """Executes the CLIP video tower.

    Args:
      x: A 5D float tensor of shape (batch_size, num_frames, height, width, 3)
        representing the input images.
      train: Whether or not the model is under training.
      debug: whether or not it is in debug mode.

    Returns:
      tokens after the image encoder if final_endpoint = 'temporal_tokens',
      pre_logits before the final projection layer if final_endpoint =
      'pre_logits', logits if final_endpoint = 'logits'.
    """
    assert (
        self.image_encoder_config.patches.size[0]
        == self.image_encoder_config.patches.size[1]
    )
    x = self._image_to_patch(x, self.image_encoder_config.patches.size[0])
    x = nn.Dense(
        self.image_encoder_config.features, use_bias=False, name='conv1'
    )(x)
    features = self.image_encoder_config.features
    scale = 1.0 / jnp.sqrt(features)
    per_frame_encoder = functools.partial(
        ClipVisionTransformer(
            name='VisionTransformer', **self.image_encoder_config
        ),
        is_train=train,
    )
    image_encoder_classifier = self.image_encoder_config.get(
        'classifier', 'token')
    if image_encoder_classifier == 'token':
      num_frames = x.shape[1]
      class_embedding = self.param('class_embedding',
                                   jax.nn.initializers.normal(stddev=scale),
                                   (num_frames, features), x.dtype)
      x = jax.vmap(
          per_frame_encoder, in_axes=[1, 0], out_axes=1)(x, class_embedding)
    else:
      x = jax.vmap(per_frame_encoder, in_axes=1, out_axes=1)(x)
    if self.final_endpoint == 'temporal_tokens':
      if image_encoder_classifier == 'token':
        return x[:, :, 0]
      else:
        return jnp.mean(x, axis=2)
    x = self._temporal_encode(x, train)
    if self.representation_size is not None:
      x = nn.Dense(self.representation_size, name='proj')(x)
      x = nn.tanh(x)
    if self.final_endpoint == 'pre_logits':
      return x
    x = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        dtype=self.dtype,
        name='output_projection')(
            x)
    return x


class ClipTextEncoder(nn.Module):
  """CLIP text encoder."""

  vocab_size: int
  num_layers: int
  hidden_size: int
  num_heads: int
  dtype: jnp.dtype = jnp.float32
  classifier: str = 'eos'
  remat_block: bool = False

  @nn.compact
  def __call__(self,
               inputs: Dict[str, jnp.ndarray],
               *,
               train: bool,
               debug: bool = False):
    assert self.classifier == 'eos'
    text = inputs['input_word_ids']
    positional_embedding = self.param('positional_embedding',
                                      jax.nn.initializers.zeros,
                                      (text.shape[1], self.hidden_size),
                                      self.dtype)
    mask = nn.combine_masks(
        nn.make_attention_mask(text > 0, text > 0), nn.make_causal_mask(text))
    x = nn.Embed(
        self.vocab_size,
        self.hidden_size,
        dtype=self.dtype,
        name='token_embedding')(
            text)
    x = x + positional_embedding[None]
    x = ClipTransformer(
        self.hidden_size,
        self.num_layers,
        self.num_heads,
        remat_block=self.remat_block,
        name='transformer',
    )(x, attn_mask=mask, train=False)
    return clip_layers.LayerNorm(dtype=self.dtype, name='ln_final')(x)


class PassThroughEncoder(nn.Module):
  """An encoder that simply copies the input to the output."""

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, train: bool, debug: bool = False):
    del train, debug
    return inputs

ENCODERS = {
    'clip_text_encoder': ClipTextEncoder,
    'clip_video_encoder': ClipVideoTower,
    # This one is mainly for precomputed embeddings.
    'pass_through_encoder': PassThroughEncoder,
}
