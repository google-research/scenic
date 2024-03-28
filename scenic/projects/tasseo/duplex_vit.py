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

"""Duplex-input Vision Transformer."""

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models.classification_model import ClassificationModel
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import vit


class Encoder(nn.Module):
  """Transformer Encoder.

  **This is same as vit.Encoder(), but without adding positional embedding.**

  Attributes:
    num_layers: Number of layers.
    mlp_dim: Dimension of the mlp on top of attention block.
    inputs_positions: Input subsequence positions for packed examples.
    dropout_rate: Dropout rate.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value. Our implementation of stochastic depth follows timm
      library, which does per-example layer dropping and uses independent
      dropping patterns for each skip-connection.
    dtype: Dtype of activations.
  """
  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool = False):
    """Applies Transformer model on the inputs."""
    assert x.ndim == 3  # Shape is `[batch, len, emb]`.
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)

    # Input Encoder.
    for lyr in range(self.num_layers):
      x = vit.Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_depth=(lyr / max(self.num_layers - 1, 1)) *
          self.stochastic_depth,
          name=f'encoderblock_{lyr}',
          dtype=dtype)(
              x, deterministic=not train)
    encoded = nn.LayerNorm(name='encoder_norm')(x)
    return encoded


class DuplexViT(nn.Module):
  """Duplex input Vision Transformer model.

    Attributes:
    num_classes: Number of output classes.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    encoder: Configuration of the encoders used in the model.
    dropout_rate: Dropout rate.
    classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
      'token'.
    dtype: JAX data type for activations.
  """

  num_classes: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  encoder: ml_collections.ConfigDict
  dropout_rate: float = 0.0
  classifier: str = 'gap'
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, context_x: jnp.ndarray,
               *, train: bool, debug: bool = False):

    # Extracting patches and then embedding is in fact a single convolution.
    fh, fw = self.patches.input_size
    x = nn.Conv(
        self.hidden_size, (fh, fw),
        strides=(fh, fw),
        padding='VALID',
        name='embedding')(
            x)
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # Repeat the extraction of patches for the context image.
    context_fh, context_fw = self.patches.context_size
    context_x = nn.Conv(
        self.hidden_size, (context_fh, context_fw),
        strides=(context_fh, context_fw),
        padding='VALID',
        name='context_embedding')(
            context_x)
    context_n, context_h, context_w, context_c = context_x.shape
    context_x = jnp.reshape(context_x,
                            [context_n, context_h * context_w, context_c])

    # If we want to add a class token, add it to only input (not context).
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    # Add potitional embedding for both input and context
    x = vit.AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02), name='posembed_input')(
            x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    context_x = vit.AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),
        name='posembed_context')(
            context_x)
    context_x = nn.Dropout(rate=self.dropout_rate)(
        context_x, deterministic=not train)

    # Optional encoding for input.
    if self.encoder.get('input'):
      x = Encoder(
          mlp_dim=self.encoder.input.mlp_dim,
          num_layers=self.encoder.input.num_layers,
          num_heads=self.encoder.input.num_heads,
          dropout_rate=self.encoder.input.dropout_rate,
          attention_dropout_rate=self.encoder.input.attention_dropout_rate,
          stochastic_depth=self.encoder.input.stochastic_depth,
          dtype=self.dtype,
          name='Input_Transformer')(
              x, train=train)

    # Optional encoding for context.
    if self.encoder.get('context'):
      context_x = Encoder(
          mlp_dim=self.encoder.context.mlp_dim,
          num_layers=self.encoder.context.num_layers,
          num_heads=self.encoder.context.num_heads,
          dropout_rate=self.encoder.context.dropout_rate,
          attention_dropout_rate=self.encoder.context.attention_dropout_rate,
          stochastic_depth=self.encoder.context.stochastic_depth,
          dtype=self.dtype,
          name='Context_Transformer')(
              context_x, train=train)

    # Concat input and context for optionally extra processing on both.
    x = jnp.concatenate([x, context_x], axis=1)

    # Optional encoding for input+context.
    if self.encoder.get('fused'):
      x = Encoder(
          mlp_dim=self.encoder.fused.mlp_dim,
          num_layers=self.encoder.fused.num_layers,
          num_heads=self.encoder.fused.num_heads,
          dropout_rate=self.encoder.fused.dropout_rate,
          attention_dropout_rate=self.encoder.fused.attention_dropout_rate,
          stochastic_depth=self.encoder.fused.stochastic_depth,
          dtype=self.dtype,
          name='Transformer')(
              x, train=train)

    if self.classifier in ('token', '0'):
      if (self.encoder.get('fused') is None) and (self.encoder.get('context')
                                                  is not None):
        raise ValueError('You are encoding the context but'
                         'not using it since the CLS token come from the input'
                         'encoder. Either use gap/gmp/gsp or add a few layers'
                         'of fused encoder.')
      x = x[:, 0]
    elif self.classifier in ('gap', 'gmp', 'gsp'):
      fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
      x = fn(x, axis=1)

    x = nn_layers.IdentityLayer(name='pre_logits')(x)
    x = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(
            x)
    return x


class DuplexViTClassificationModel(ClassificationModel):
  """Duplex (chromosome + metaphase) ViT model for classification task."""

  def build_flax_model(self):
    return DuplexViT(
        num_classes=self.dataset_meta_data['num_classes'],
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        encoder=self.config.model.encoder,
        dropout_rate=self.config.model.get('dropout_rate', 0.0),
        classifier=self.config.model.classifier,
        dtype='float32',
    )
