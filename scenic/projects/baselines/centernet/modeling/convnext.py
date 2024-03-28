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

"""Implementation of ConvNeXt."""

import functools
from typing import Callable, Any, Optional, Union, Dict

import flax.linen as nn
import jax
from jax.nn import initializers
import jax.numpy as jnp
from scenic.model_lib.layers import nn_layers


class ConvNeXtBlock(nn.Module):
  """ConvNeXt block: DwConv -> LayerNorm -> Linear -> GELU -> Linear."""

  dim: int
  droplayer_p: float = 0
  layer_scale_init_value: float = 1e-6
  gelu_approximate: bool = False
  scale_drop_path: bool = False
  dtype: jnp.dtype = jnp.float32

  def get_drop_pattern(self,
                       x: jnp.ndarray,
                       deterministic: bool) -> jnp.ndarray:
    """DropPath Layer."""
    if not deterministic and self.droplayer_p:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng('dropout'), self.droplayer_p, shape).astype(self.dtype)
    else:
      return 0.0  # pytype: disable=bad-return-type  # jax-ndarray

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
    residual = x
    x = nn.Conv(
        self.dim, (7, 7),
        1,
        padding=3,
        feature_group_count=self.dim,
        use_bias=True,
        dtype=self.dtype,
        name='dwconv')(
            x)
    x = nn.LayerNorm(epsilon=1e-6, name='norm', dtype=self.dtype)(x)
    x = nn.Dense(
        4 * self.dim, name='pwconv1', dtype=self.dtype)(x)  # B x H x W x 4C
    x = nn.gelu(x, approximate=self.gelu_approximate)
    x = nn.Dense(self.dim, name='pwconv2', dtype=self.dtype)(x)  # B x H x W x C
    if self.layer_scale_init_value > 0:
      gamma = self.param(
          'gamma',
          initializers.constant(self.layer_scale_init_value),
          (self.dim))
      x = x * gamma[..., :]
    drop_pattern = self.get_drop_pattern(x, deterministic=not train)

    keep_prob = (1 - self.droplayer_p)
    if self.scale_drop_path and train and keep_prob > 1e-3:
      divisor = keep_prob
    else:
      divisor = 1.0

    x = residual + (1.0 - drop_pattern) * x / divisor
    return x

SIZE_OPTIONS = {
    'T': ([3, 3, 9, 3], [96, 192, 384, 768], 0.1),
    'S': ([3, 3, 27, 3], [96, 192, 384, 768], 0.4),
    'B': ([3, 3, 27, 3], [128, 256, 512, 1024], 0.5),
    'L': ([3, 3, 27, 3], [192, 384, 768, 1536], 0.5),
    'XL': ([3, 3, 27, 3], [256, 512, 1024, 2048], 0.5),
}


class ConvNeXt(nn.Module):
  """ConvNeXt architecture.

  Attributes:
    num_outputs: Num output classes. If None, a dict of intermediate feature
      maps is returned.
    size: size as pre-defined in the paper. Options: T, S, B, L, XL
    kernel_init: Kernel initialization.
    bias_init: Bias initialization.
    dtype: Data type, e.g. jnp.float32.
  """
  num_outputs: Optional[int]
  size: str = 'T'
  layer_scale_init_value: float = 1e-6
  gelu_approximate: bool = False
  drop_path_rate: Optional[float] = None
  scale_drop_path: bool = False
  kernel_init: Callable[..., Any] = initializers.lecun_normal()
  bias_init: Callable[..., Any] = initializers.zeros
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      train: bool = False,
      debug: bool = False) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Applies ResNet model to the inputs.

    Args:
      x: Inputs to the model.
      train: Whether it is training or not.
      debug: Whether the debug mode is enabled. debug=True enables model
        specific logging/storing some values using jax.host_callback.

    Returns:
       Un-normalized logits.
    """
    if self.size not in SIZE_OPTIONS:
      raise ValueError('Please provide a valid size')
    depths, dims, default_drop_path_rate = SIZE_OPTIONS[self.size]
    drop_path_rate = self.drop_path_rate or default_drop_path_rate
    sum_depth = sum(depths)
    dp_rates = [drop_path_rate * i / (sum_depth - 1) for i in range(sum_depth)]
    layernorm = functools.partial(nn.LayerNorm, epsilon=1e-6, dtype=self.dtype)
    block = functools.partial(
        ConvNeXtBlock,
        layer_scale_init_value=self.layer_scale_init_value,
        scale_drop_path=self.scale_drop_path,
        dtype=self.dtype)
    x = nn.Conv(
        dims[0],
        kernel_size=(4, 4),
        strides=(4, 4),
        dtype=self.dtype,
        name='downsample_layers.0.0')(
            x)
    x = layernorm(name='downsample_layers.0.1')(x)
    representations = {'stem': x}
    cur = 0
    for i, (depth, dim) in enumerate(zip(depths, dims)):
      if i > 0:
        x = layernorm(name='downsample_layers.{}.0'.format(i))(x)
        x = nn.Conv(
            dims[i],
            kernel_size=(2, 2),
            strides=(2, 2),
            dtype=self.dtype,
            name='downsample_layers.{}.1'.format(i))(
                x)
      for j in range(depth):
        x = block(
            dim=dim, droplayer_p=dp_rates[cur + j],
            name='stages.{}.{}'.format(i, j))(x, train)
      cur += depth
      representations[f'stage_{i + 1}'] = x

    # Head.
    if self.num_outputs:
      x = jnp.mean(x, axis=(1, 2))
      x = layernorm(name='norm')(x)
      x = nn_layers.IdentityLayer(name='pre_logits')(x)
      x = nn.Dense(
          self.num_outputs,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          dtype=self.dtype,
          name='output_projection')(
              x)
      return x
    else:
      return representations
