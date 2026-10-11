"""Common Blocks."""
import functools
from typing import Any
import flax.linen as nn

import jax
import jax.numpy as jnp


def get_norm_layer(train, dtype, norm_type="BN"):
  """Normalization layer."""
  if norm_type == "BN":
    norm_fn = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        axis_name=None,
        axis_index_groups=None,
        dtype=jnp.float32)
  elif norm_type == "LN":
    norm_fn = functools.partial(nn.LayerNorm, dtype=dtype)
  elif norm_type == "GN":
    norm_fn = functools.partial(nn.GroupNorm, dtype=dtype)
  else:
    raise NotImplementedError
  return norm_fn


def tensorflow_style_avg_pooling(x, window_shape, strides, padding: str):
  """Avg pooling as done by TF (Flax layer gives different results).

  To be specific, Flax includes padding cells when taking the average,
  while TF does not.

  Args:
    x: Input tensor
    window_shape: Shape of pooling window; if 1-dim tuple is just 1d pooling, if
      2-dim tuple one gets 2d pooling.
    strides: Must have the same dimension as the window_shape.
    padding: Either 'SAME' or 'VALID' to indicate pooling method.

  Returns:
    pooled: Tensor after applying pooling.
  """
  pool_sum = jax.lax.reduce_window(x, 0.0, jax.lax.add,
                                   (1,) + window_shape + (1,),
                                   (1,) + strides + (1,), padding)
  pool_denom = jax.lax.reduce_window(
      jnp.ones_like(x), 0.0, jax.lax.add, (1,) + window_shape + (1,),
      (1,) + strides + (1,), padding)
  return pool_sum / pool_denom


def upsample(x, factor=2):
  n, h, w, c = x.shape
  x = jax.image.resize(x, (n, h * factor, w * factor, c), method="nearest")
  return x


def dsample(x):
  return tensorflow_style_avg_pooling(x, (2, 2), strides=(2, 2), padding="same")


class DiscBlock(nn.Module):
  """Discriminator Basic Block."""
  filters: int
  downsample: bool
  conv_fn: Any
  activation_fn: Any = nn.relu
  dtype: int = jnp.float32

  @nn.compact
  def __call__(self, x):
    needs_projection = self.downsample or x.shape[-1] != self.filters
    x0 = x
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3))(x)
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3))(x)
    if needs_projection:
      x0 = self.conv_fn(self.filters, kernel_size=(1, 1))(x0)
    if self.downsample:
      x = dsample(x)
      x0 = dsample(x0)
    return x0 + x


class DiscOptimizedBlock(nn.Module):
  """Discriminator Optimized Block."""
  filters: int
  conv_fn: Any
  activation_fn: Any = nn.relu
  dtype: int = jnp.float32

  @nn.compact
  def __call__(self, x):
    x0 = x
    x = self.conv_fn(self.filters, kernel_size=(3, 3))(x)
    x = self.activation_fn(x)
    x = self.conv_fn(self.filters, kernel_size=(3, 3))(x)
    x = dsample(x)
    x0 = dsample(x0)
    x0 = self.conv_fn(self.filters, kernel_size=(1, 1))(x0)
    return x + x0


