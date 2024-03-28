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

"""Matryoshka Vision Transformer."""

from typing import Any, Callable, List, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import vit
from scenic.projects.matvit import layers

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of self-attention heads.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows
      from 0 to the provided value.

  Returns:
    output after transformer encoder block.
  """
  mlp_dim: int
  num_heads: int
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      deterministic: bool,
      matvit_mask: Optional[Any] = None,
  ) -> jnp.ndarray:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data.
      deterministic: Deterministic or not (to apply dropout).
      matvit_mask: matvit masks for this block.

    Returns:
      Output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate)(x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = nn_layers.StochasticDepth(rate=self.stochastic_depth)(x, deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )(y, deterministic=deterministic, matvit_mask=matvit_mask)
    y = nn_layers.StochasticDepth(rate=self.stochastic_depth)(y, deterministic)
    return y + x


class Encoder(nn.Module):
  """Transformer Encoder.

  Attributes:
    num_layers: Number of layers.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: The number of heads for multi-head self-attention.
    positional_embedding: The type of positional embeddings to add to the
      input tokens. Options are {learned_1d, sinusoidal_2d, none}.
    dropout_rate: Dropout rate.
    stochastic_depth: probability of dropping a layer linearly grows
      from 0 to the provided value. Our implementation of stochastic depth
      follows timm library, which does per-example layer dropping and uses
      independent dropping patterns for each skip-connection.
    dtype: Dtype of activations.
  """
  num_layers: int
  mlp_dim: int
  num_heads: int
  positional_embedding: str = 'learned_1d'
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      *,
      matvit_mask_dims: List[float],
      train: bool = False,
  ):
    """Applies Transformer model on the inputs.

    Args:
      inputs: Input tokens of shape [batch, num_tokens, channels].
      matvit_mask_dims: matvit nesting dimensions, a list of size num_layers.
      train: If in training mode, dropout and stochastic depth is applied.

    Returns:
      Encoded tokens.
    """

    assert inputs.ndim == 3  # Shape is `[batch, len, emb]`.
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)

    # Add positional embeddings to tokens.
    if self.positional_embedding == 'learned_1d':
      x = vit.AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='posembed_input')(
              inputs)
    elif self.positional_embedding == 'sinusoidal_1d':
      x = attention_layers.Add1DPositionEmbedding(posemb_init=None)(inputs)
    elif self.positional_embedding == 'sinusoidal_2d':
      batch, num_tokens, hidden_dim = inputs.shape
      height = width = int(np.sqrt(num_tokens))
      if height * width != num_tokens:
        raise ValueError('Input is assumed to be square for sinusoidal init.')
      inputs_reshape = inputs.reshape([batch, height, width, hidden_dim])
      x = attention_layers.AddFixedSinCosPositionEmbedding()(inputs_reshape)
      x = x.reshape([batch, num_tokens, hidden_dim])
    elif self.positional_embedding == 'none':
      x = inputs
    else:
      raise ValueError('Unknown positional embedding: '
                       f'{self.positional_embedding}')
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder.
    for lyr in range(self.num_layers):
      nesting_mask = jnp.ones(matvit_mask_dims[lyr], dtype=jnp.int32)
      ffn_mask = jnp.zeros(self.mlp_dim, dtype=jnp.int32)
      ffn_mask = jax.lax.dynamic_update_slice(ffn_mask, nesting_mask, (0,))
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_depth=(lyr / max(self.num_layers - 1, 1))
          * self.stochastic_depth,
          name=f'encoderblock_{lyr}',
          dtype=dtype,
      )(x, deterministic=not train, matvit_mask=ffn_mask)
    encoded = nn.LayerNorm(name='encoder_norm')(x)
    return encoded


class MatViT(nn.Module):
  """Vision Transformer model.

    Attributes:
    num_classes: Number of output classes.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    positional_embedding: The type of positional embeddings to add to the
      tokens at the beginning of the transformer encoder. Options are
      {learned_1d, sinusoidal_2d, none}.
    representation_size: Size of the representation layer in the model's head.
      if None, we skip the extra projection + tanh activation at the end.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
      'token', 'none'.
    dtype: JAX data type for activations.
  """

  num_classes: int
  mlp_dim: int
  num_layers: int
  num_heads: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  positional_embedding: str = 'learned_1d'
  representation_size: Optional[int] = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  classifier: str = 'gap'
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      *,
      train: bool,
      matvit_mask_dims: Optional[List[int]] = None,
      debug: bool = False,
      return_feat: bool = False,
  ):
    if matvit_mask_dims is None:
      matvit_mask_dims = [self.mlp_dim] * self.num_layers
    fh, fw = self.patches.size
    # Extracting patches and then embedding is in fact a single convolution.
    x = nn.Conv(
        self.hidden_size,
        (fh, fw),
        strides=(fh, fw),
        padding='VALID',
        name='embedding',
    )(x)
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    x = Encoder(
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        positional_embedding=self.positional_embedding,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
        dtype=self.dtype,
        name='Transformer')(
            x, train=train, matvit_mask_dims=matvit_mask_dims)

    if self.classifier in ('token', '0'):
      x = x[:, 0]
    elif self.classifier in ('gap', 'gmp', 'gsp'):
      fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
      x = fn(x, axis=1)
    elif self.classifier == 'map':
      x = vit.MAPHead(
          num_heads=self.num_heads, mlp_dim=self.mlp_dim, dtype=self.dtype)(x)
    elif self.classifier == 'none':
      pass
    else:
      raise ValueError(f'Unknown classifier {self.classifier}')

    if self.representation_size is not None:
      x = nn.Dense(self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = nn_layers.IdentityLayer(name='pre_logits')(x)

    if return_feat:
      return x

    if self.num_classes > 0:
      # If self.num_classes <= 0, we just return the backbone features.
      x = nn.Dense(
          self.num_classes,
          kernel_init=nn.initializers.zeros,
          name='output_projection')(
              x)
    return x


class MatViTMultiLabelClassificationModel(vit.ViTMultiLabelClassificationModel):
  """Matryoshka Vision Transformer model for multi-label classification task."""

  def build_flax_model(self)-> nn.Module:
    dtype_str = self.config.get('model_dtype_str', 'float32')
    if dtype_str != 'float32':
      raise ValueError('`dtype` argument is not propagated properly '
                       'in the current implmentation, so only '
                       '`float32` is supported for now.')
    return MatViT(
        num_classes=self.dataset_meta_data['num_classes'],
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        positional_embedding=self.config.model.get('positional_embedding',
                                                   'learned_1d'),
        representation_size=self.config.model.representation_size,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate'),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate'),
        stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
        dtype=getattr(jnp, dtype_str),
    )
