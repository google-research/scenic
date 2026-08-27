"""Encoder and Decoder stuctures modified from VQGAN paper.

https://arxiv.org/abs/2012.09841
Here we remove the non-local layer for faster speed.

"""

from typing import Any

import flax.linen as nn
import jax.numpy as jnp
import ml_collections
from scenic.projects.robust_vit.gvt import common


class Downsample(nn.Module):
  """Downsample Blocks."""

  use_conv: bool

  @nn.compact
  def __call__(self, x):
    out_dim = x.shape[-1]
    if self.use_conv:
      x = nn.Conv(out_dim, kernel_size=(4, 4), strides=(2, 2))(x)
    else:
      x = common.dsample(x)
    return x


class ResBlock(nn.Module):
  """Basic Residual Block."""
  filters: int
  norm_fn: Any
  conv_fn: Any
  dtype: int = jnp.float32
  activation_fn: Any = nn.relu
  use_conv_shortcut: bool = False

  @nn.compact
  def __call__(self, x):
    input_dim = x.shape[-1]
    residual = x
    x = self.norm_fn()(x)
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
    x = self.norm_fn()(x)
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)

    if input_dim != self.filters:
      if self.use_conv_shortcut:
        residual = self.conv_fn(
            self.filters, kernel_size=(3, 3), use_bias=False)(
                x)
      else:
        residual = self.conv_fn(
            self.filters, kernel_size=(1, 1), use_bias=False)(
                x)
    return x + residual


class Encoder(nn.Module):
  """Encoder Blocks."""

  config: ml_collections.ConfigDict
  train: bool
  dtype: int = jnp.float32

  def setup(self):
    self.filters = self.config.vqvae.filters
    self.num_res_blocks = self.config.vqvae.num_res_blocks
    self.channel_multipliers = self.config.vqvae.channel_multipliers
    self.embedding_dim = self.config.vqvae.embedding_dim
    self.conv_downsample = self.config.vqvae.conv_downsample
    self.norm_type = self.config.vqvae.norm_type
    if self.config.vqvae.activation_fn == "relu":
      self.activation_fn = nn.relu
    elif self.config.vqvae.activation_fn == "swish":
      self.activation_fn = nn.swish
    else:
      raise NotImplementedError

  @nn.compact
  def __call__(self, x):
    conv_fn = nn.Conv
    norm_fn = common.get_norm_layer(
        train=self.train, dtype=self.dtype, norm_type=self.norm_type)
    block_args = dict(
        norm_fn=norm_fn,
        conv_fn=conv_fn,
        dtype=self.dtype,
        activation_fn=self.activation_fn,
        use_conv_shortcut=False,
    )
    x = conv_fn(self.filters, kernel_size=(3, 3), use_bias=False)(x)
    num_blocks = len(self.channel_multipliers)
    for i in range(num_blocks):
      filters = self.filters * self.channel_multipliers[i]
      for _ in range(self.num_res_blocks):
        x = ResBlock(filters, **block_args)(x)
      if i < num_blocks - 1:
        if self.conv_downsample:
          x = conv_fn(filters, kernel_size=(4, 4), strides=(2, 2))(x)
        else:
          x = common.dsample(x)
    for _ in range(self.num_res_blocks):
      x = ResBlock(filters, **block_args)(x)
    x = norm_fn()(x)
    x = self.activation_fn(x)
    x = conv_fn(self.embedding_dim, kernel_size=(1, 1))(x)
    return x


class Decoder(nn.Module):
  """Decoder Blocks."""

  config: ml_collections.ConfigDict
  train: bool
  output_dim: int = 3
  dtype: Any = jnp.float32

  def setup(self):
    self.filters = self.config.vqvae.filters
    self.num_res_blocks = self.config.vqvae.num_res_blocks
    self.channel_multipliers = self.config.vqvae.channel_multipliers
    self.norm_type = self.config.vqvae.norm_type
    if self.config.vqvae.activation_fn == "relu":
      self.activation_fn = nn.relu
    elif self.config.vqvae.activation_fn == "swish":
      self.activation_fn = nn.swish
    else:
      raise NotImplementedError

  @nn.compact
  def __call__(self, x):
    conv_fn = nn.Conv
    norm_fn = common.get_norm_layer(
        train=self.train, dtype=self.dtype, norm_type=self.norm_type)
    block_args = dict(
        norm_fn=norm_fn,
        conv_fn=conv_fn,
        dtype=self.dtype,
        activation_fn=self.activation_fn,
        use_conv_shortcut=False,
    )
    num_blocks = len(self.channel_multipliers)
    filters = self.filters * self.channel_multipliers[-1]
    x = conv_fn(filters, kernel_size=(3, 3), use_bias=True)(x)
    for _ in range(self.num_res_blocks):
      x = ResBlock(filters, **block_args)(x)
    for i in reversed(range(num_blocks)):
      filters = self.filters * self.channel_multipliers[i]
      for _ in range(self.num_res_blocks):
        x = ResBlock(filters, **block_args)(x)
      if i > 0:
        x = common.upsample(x, 2)
        x = conv_fn(filters, kernel_size=(3, 3))(x)
    x = norm_fn()(x)
    x = self.activation_fn(x)
    x = conv_fn(self.output_dim, kernel_size=(3, 3))(x)
    return x
