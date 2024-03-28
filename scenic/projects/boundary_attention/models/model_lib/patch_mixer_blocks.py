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

"""Code for Patch-Mixer."""

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


class PatchMixer(nn.Module):
  """Patch-Mixer architecture.

  Attributes:
    tokens_rf: receptive field of token mixer
    num_blocks: number of mixer blocks
    hidden_dim: dimension to project input to
    tokens_conv_dim: number of filters for token mixer
    channels_conv_dim: number of filters for channel mixer
  """

  tokens_rf: int
  num_blocks: int
  hidden_dim: int
  tokens_conv_dim: int
  channels_conv_dim: int
  padding: str = 'VALID'
  stride: Optional[int] = 1

  @nn.compact
  def __call__(self, inputs):

    if self.stride == 1:
      # Each pixel of the input image gets mapped to self.hidden_dim
      x = nn.Dense(self.hidden_dim, bias_init=nn.initializers.uniform(),
                   name='stem')(inputs)  # [N, H, W, self.hidden_dim]

    else:
      # Extract pixels with stride:
      x = nn.Conv(self.hidden_dim, kernel_size=(3, 3),
                  bias_init=nn.initializers.uniform(),
                  strides=self.stride, padding='SAME')(inputs)
      x = nn.Dense(self.hidden_dim, bias_init=nn.initializers.uniform(),
                   name='stem')(x)

    # Conv mixer blocks
    for _ in range(self.num_blocks):
      x = MixerBlock(self.tokens_conv_dim, self.tokens_rf,
                     self.channels_conv_dim, padding=self.padding)(x)

    # Output of mixer blocks [N, H, W, self.hidden_dim]
    x = nn.LayerNorm(name='pre_head_layer_norm')(x)

    return x


class MixerBlock(nn.Module):
  """Mixer block layer.

  Attributes:
    tokens_conv_dim: number of filters for token mixer
    tokens_rf: filter size for token mixer (receptive field of tokens)
    channels_conv_dim: number of filters for channel mixer
  """
  tokens_conv_dim: int
  tokens_rf: int
  channels_conv_dim: int
  padding: str = 'SAME'

  @nn.compact
  def __call__(self, x):

    # Future to do: Reshape array prior to layer normalization
    y = nn.LayerNorm()(x)

    # First, token mixing
    y = TiedConvBlock(self.tokens_conv_dim, self.tokens_rf,
                      padding=self.padding, name='token_mixing')(y)

    x = x + y
    y = nn.LayerNorm()(x)

    # Next, channel mixing
    y = ConvBlock(self.channels_conv_dim, 1, name='channel_mixing')(y)

    return x + y


class TiedConvBlock(nn.Module):
  """Two convolutions with gelu activation.

  Attributes:
    conv_dim: number of filters
    conv_rf: filter size
    padding: 'SAME' or 'VALID'
  """
  conv_dim: int
  conv_rf: int
  padding: str = 'SAME'

  @nn.compact
  def __call__(self, x):
    y = TiedWeightsConv(self.conv_dim, self.conv_rf, padding=self.padding)(x)
    y = nn.gelu(y)
    y = TiedWeightsConv(x.shape[-1], self.conv_rf, padding=self.padding)(y)
    return y


class ConvBlock(nn.Module):
  """Two convolutions with gelu activation.

  Attributes:
    conv_dim: number of filters
    conv_rf: filter size
  """
  conv_dim: int
  conv_rf: int

  @nn.compact
  def __call__(self, x):
    y = nn.Dense(self.conv_dim, bias_init=nn.initializers.uniform())(x)
    y = nn.gelu(y)
    y = nn.Dense(x.shape[-1], bias_init=nn.initializers.uniform())(y)
    return y


class TiedWeightsConv(nn.Module):
  """Convolution with tied weights.

  Attributes:
    filters: number of filters
    kernel_size: filter size
    strides: strides
    padding: 'SAME' or 'VALID'
    dilation: dilation
  """

  filters: int
  kernel_size: int
  strides: tuple[int, int] = (1, 1)
  padding: str = 'SAME'
  dilation: tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, inputs):
    """Applies convolution with tied weights.

    Args:
      inputs: input data

    Returns:
      output of convolution
    """

    def tied_weights_conv(inputs, kernel, bias):
      num_channels = inputs.shape[-1]
      kernel = jnp.repeat(kernel, num_channels, axis=2)
      dimension_numbers = ('NHWC', 'HWIO', 'NHWC')

      return jax.lax.conv_general_dilated(inputs, kernel,
                                          self.strides, self.padding,
                                          self.dilation, self.dilation,
                                          dimension_numbers) + bias

    kernel_init = nn.initializers.lecun_normal()
    kernel_shape = (self.kernel_size, self.kernel_size, 1, self.filters)
    kernel = self.param('kernel', kernel_init, kernel_shape)

    bias_init = nn.initializers.uniform()
    bias_shape = (1, self.filters)
    bias = jnp.expand_dims(self.param('bias', bias_init, bias_shape), (0, 1))

    return tied_weights_conv(inputs, kernel, bias)

