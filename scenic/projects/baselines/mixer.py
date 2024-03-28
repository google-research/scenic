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

"""Implementation of MLP-Mixer model."""

from typing import Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models.multilabel_classification_model import MultiLabelClassificationModel
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers


class MixerBlock(nn.Module):
  """Mixer block consisting of a token- and a channel-mixing phase.

  Attributes:
    channels_mlp_dim: Hidden dimension of the channel mixing MLP.
    sequence_mlp_dim: Hidden dimension of the token (sequence) mixing MLP.
    dropout_rate: Dropout rate.
    stochastic_depth: The layer dropout rate (= stochastic depth).
    layer_scale: The scalar value used to initialise layer_scale. If None,
      layer_scale is not used.

  Returns:
    Output after mixer block.
  """
  channels_mlp_dim: int
  sequence_mlp_dim: int
  dropout_rate: float = 0.0
  stochastic_depth: float = 0.0
  layer_scale: Optional[float] = None

  # Having this as a separate function makes it possible to capture the
  # intermediate representation via capture_intermediandarrates.
  def combine_branches(self, long_branch: jnp.ndarray,
                       short_branch: jnp.ndarray) -> jnp.ndarray:
    """Merges residual connections."""
    return long_branch + short_branch

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
    """Applies the Mixer block to inputs."""
    if inputs.ndim != 3:
      raise ValueError('Input should be of shape `[batch, tokens, channels]`.')

    if self.layer_scale is not None:
      layerscale_init = nn_layers.get_constant_initializer(
          self.layer_scale)

    # Token mixing part, provides between-patches communication.
    x = nn.LayerNorm()(inputs)
    x = jnp.swapaxes(x, 1, 2)

    x = attention_layers.MlpBlock(
        mlp_dim=self.sequence_mlp_dim,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        name='token_mixing')(
            x, deterministic=deterministic)
    x = jnp.swapaxes(x, 1, 2)
    if self.layer_scale is not None:
      x = nn_layers.Affine(scale_init=layerscale_init, use_bias=False)(x)

    x = nn_layers.StochasticDepth(rate=self.stochastic_depth)(x, deterministic)
    x = self.combine_branches(x, inputs)

    # Channel-mixing part, which provides within-patch communication.
    y = nn.LayerNorm()(x)
    y = attention_layers.MlpBlock(
        mlp_dim=self.channels_mlp_dim,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        name='channel_mixing')(
            y, deterministic=deterministic)
    if self.layer_scale is not None:
      x = nn_layers.Affine(scale_init=layerscale_init, use_bias=False)(x)

    y = nn_layers.StochasticDepth(rate=self.stochastic_depth)(y, deterministic)
    return self.combine_branches(y, x)


class Mixer(nn.Module):
  """Mixer model.

  Attributes:
    num_classes: Number of output classes.
    patch_size: Patch size of the stem.
    hidden_size: Size of the hidden state of the output of model's stem.
    num_layers: Number of layers.
    channels_mlp_dim: hidden dimension of the channel mixing MLP.
    sequence_mlp_dim: hidden dimension of the token (sequence) mixing MLP.
    dropout_rate: Dropout rate.
    stochastic_depth: overall stochastic depth rate.
  """

  num_classes: int
  patch_size: Sequence[int]
  hidden_size: int
  num_layers: int
  channels_mlp_dim: int
  sequence_mlp_dim: int
  dropout_rate: float = 0.0
  stochastic_depth: float = 0.0
  layer_scale: Optional[float] = None

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               *,
               train: bool,
               debug: bool = False) -> jnp.ndarray:

    x = nn.Conv(
        self.hidden_size,
        self.patch_size,
        strides=self.patch_size,
        padding='VALID',
        name='embedding')(
            x)
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])
    for i in range(self.num_layers):
      p = (i / max(self.num_layers - 1, 1)) * self.stochastic_depth
      x = MixerBlock(
          channels_mlp_dim=self.channels_mlp_dim,
          sequence_mlp_dim=self.sequence_mlp_dim,
          dropout_rate=self.dropout_rate,
          stochastic_depth=p,
          layer_scale=self.layer_scale,
          name=f'mixerblock_{i}')(
              x, deterministic=not train)
    x = nn.LayerNorm(name='pre_logits_norm')(x)
    # Use global average pooling for classifier:
    x = jnp.mean(x, axis=1)
    x = nn_layers.IdentityLayer(name='pre_logits')(x)
    return nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(
            x)


class MixerMultiLabelClassificationModel(MultiLabelClassificationModel):
  """Mixer model for multi-label classification task."""

  def build_flax_model(self) -> nn.Module:
    return Mixer(
        num_classes=self.dataset_meta_data['num_classes'],
        patch_size=self.config.model.patch_size,
        hidden_size=self.config.model.hidden_size,
        num_layers=self.config.model.num_layers,
        channels_mlp_dim=self.config.model.channels_mlp_dim,
        sequence_mlp_dim=self.config.model.sequence_mlp_dim,
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
        layer_scale=self.config.model.get('layer_scale', None)
    )

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict({
        'model':
            dict(
                patch_size=(4, 4),
                hidden_size=16,
                num_layers=1,
                channels_mlp_dim=32,
                sequence_mlp_dim=32,
                dropout_rate=0.,
                stochastic_depth=0,
            )
    })
