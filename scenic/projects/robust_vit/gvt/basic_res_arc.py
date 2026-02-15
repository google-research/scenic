"""Network Blocks for basic VQVAE/VQGAN model."""
import functools
from typing import Any, Callable, Tuple

from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp
import ml_collections
from scenic.projects.robust_vit.gvt import config_lib

Array = jnp.ndarray
DType = Any
ModuleDef = Any


class ResnetBlock(nn.Module):
  """ResNet block."""
  hidden_size: int
  stride_size: Tuple[int, int] = (1, 1)
  conv_fn: ModuleDef = nn.Conv
  act_fn: Callable[[Array], Array] = nn.relu

  @nn.compact
  def __call__(self, x: Array) -> Array:
    residual = x
    x = self.conv_fn(self.hidden_size, (3, 3), self.stride_size)(x)
    x = self.act_fn(x)
    x = self.conv_fn(self.hidden_size, (3, 3))(x)
    if residual.shape != x.shape:
      residual = self.conv_fn(self.hidden_size, (1, 1), self.stride_size)(
          residual)
    return self.act_fn(x + residual)


class ResnetStack(nn.Module):
  """ResNet stack."""
  hidden_sizes: Tuple[int, ...]
  downsample_indices: Tuple[int, ...] = ()
  upsample_indices: Tuple[int, ...] = ()
  conv_fn: ModuleDef = nn.Conv
  act_fn: Callable[[Array], Array] = nn.relu
  train: bool = True

  @nn.compact
  def __call__(self, x: Array) -> Array:
    """ResNet stack."""

    num_layers = len(self.hidden_sizes)
    downsample_indices = {i % num_layers for i in self.downsample_indices}
    upsample_indices = {i % num_layers for i in self.upsample_indices}
    for i, hidden_size in enumerate(self.hidden_sizes):
      stride_size = (2, 2) if i in downsample_indices else (1, 1)
      x = ResnetBlock(hidden_size, stride_size, self.conv_fn, self.act_fn)(x)
      # x = norm_fn()(x)  # Delete zhanghan
      if i in upsample_indices:
        b, h, w, c = x.shape
        x = jax.image.resize(
            x, (b, h * 2, w * 2, c), method=jax.image.ResizeMethod.NEAREST)
    return x


@struct.dataclass
class Config:
  """VQ-VAE config."""
  codebook_size: int
  encoder_hidden_sizes: Tuple[int, ...]
  encoder_downsample_indices: Tuple[int, ...]
  decoder_hidden_sizes: Tuple[int, ...]
  decoder_upsample_indices: Tuple[int, ...]
  conv_fn: ModuleDef = nn.Conv
  act_fn: Callable[[Array], Array] = nn.relu
  dtype: DType = jnp.float32


def get_model_config(train_config: ml_collections.ConfigDict) -> Config:
  """Initializes the model config."""
  return Config(
      codebook_size=train_config.vqvae.codebook_size,
      encoder_hidden_sizes=tuple(
          map(int, train_config.vqvae.encoder_hidden_sizes.split(','))),
      encoder_downsample_indices=tuple(
          map(int, train_config.vqvae.encoder_downsample_indices.split(','))),
      decoder_hidden_sizes=tuple(
          map(int, train_config.vqvae.decoder_hidden_sizes.split(','))),
      decoder_upsample_indices=tuple(
          map(int, train_config.vqvae.decoder_upsample_indices.split(','))),
      conv_fn=functools.partial(
          nn.Conv,
          dtype=train_config.dtype,
          kernel_init=jax.nn.initializers.glorot_normal()),
      act_fn=nn.relu,
      dtype=config_lib.get_tf_dtype(train_config))


class EncoderResnetStack(nn.Module):
  """Encoder Net."""

  config: ml_collections.ConfigDict
  train: bool
  dtype: int = jnp.float32

  def setup(self):
    cfg = get_model_config(self.config)
    self.res_block = ResnetStack(
        hidden_sizes=cfg.encoder_hidden_sizes,
        downsample_indices=cfg.encoder_downsample_indices,
        conv_fn=cfg.conv_fn,
        act_fn=cfg.act_fn,
        train=self.train)

  def __call__(self, x):
    return self.res_block(x)


class DisResnetStack(nn.Module):
  """Discriminator Net."""

  config: ml_collections.ConfigDict
  train: bool
  dtype: int = jnp.float32

  def setup(self):
    cfg = get_model_config(self.config)
    self.res_block = ResnetStack(
        hidden_sizes=cfg.encoder_hidden_sizes,
        downsample_indices=cfg.encoder_downsample_indices,
        conv_fn=cfg.conv_fn,
        act_fn=cfg.act_fn,
        train=self.train)
    self.dense = nn.Dense(1)

  def __call__(self, x):
    x = self.res_block(x)
    x = jnp.mean(x, (1, 2))
    x = self.dense(x)
    return x


class DecoderResnetStack(nn.Module):
  """Decoder Net."""

  config: ml_collections.ConfigDict
  train: bool
  dtype: int = jnp.float32

  def setup(self):
    cfg = get_model_config(self.config)
    self.res_block = ResnetStack(
        hidden_sizes=cfg.decoder_hidden_sizes,
        upsample_indices=cfg.decoder_upsample_indices,
        conv_fn=cfg.conv_fn,
        act_fn=cfg.act_fn,
        train=self.train)
    self.rgb = cfg.conv_fn(3, (1, 1))

  def __call__(self, x):
    x = self.res_block(x)
    x = self.rgb(x)
    x = jax.nn.sigmoid(x)
    return x
