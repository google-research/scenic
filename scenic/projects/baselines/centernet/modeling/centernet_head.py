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

"""CenterNet head."""

import functools
from typing import List, Dict

import flax.linen as nn
from jax.nn import initializers
import jax.numpy as jnp


ArrayList = List[jnp.ndarray]
ArrayListDict = Dict[str, ArrayList]


class Tower(nn.Module):
  """Layers between backbone and outputs.

  Attributes:
    num_layers: number of layers.
    out_channels: number of channels of all layers.
    norm: normalization layer type after each layer. Currently only support GN.
    dtype: Data type of the computation (default: float32).
  """
  num_layers: int = 4
  out_channels: int = 256
  norm: str = 'GN'
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
    """Apply module to a feature.

    Args:
      x: array in shape B x H x W x C: features from the backbone or a
        single level of FPN.
      train: whether it is training.
    Returns:
      array in shape B x H x W x C.
    """
    conv = functools.partial(
        nn.Conv, features=self.out_channels,
        kernel_size=(3, 3), padding=1, dtype=self.dtype,
        kernel_init=initializers.normal(stddev=0.01),
        bias_init=initializers.constant(0.0),
    )
    if self.norm == 'GN':
      norm = functools.partial(nn.GroupNorm, dtype=self.dtype)
    elif self.norm == 'LN':
      norm = functools.partial(nn.LayerNorm, dtype=self.dtype)
    else:
      raise ValueError(f'Unsupported norm: {self.norm}')
    for i in range(self.num_layers):
      x = conv(name=f'{i * 3}')(x)
      x = norm(name=f'{i * 3 + 1}')(x)
      x = nn.relu(x)
    return x


class Scale(nn.Module):
  """Multiplying a feature by a single learnable scale.

  Attributes:
    init_value: initialization value. Default 1: no effect.
    dtype: data type of the computation (default: float32).
  """
  init_value: float = 1.0
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
    """Apply module to a feature.

    Args:
      x: array in any shape.
      train: whether it is training.
    Returns:
      array with the same shape as the input.
    """
    return x * self.param('scale', initializers.constant(self.init_value), (1,))


class CenterNetHead(nn.Module):
  """CenterNet output layers.

  Attributes:
    num_classes: Number of object classes. num_classes = 0 will merge the
      classification feature layers and regression layers. This is
      class-agnostic detection. This is used for proposal-only mode.
    num_layers: number of layers between backbone and outputs.
    out_channels: channels of the layers.
    num_levels: number of FPN levels.
    dtype: data type of the computation (default: float32).
  """
  num_classes: int = 80
  num_layers: int = 4
  out_channels: int = 256
  num_levels: int = 5
  dtype: jnp.dtype = jnp.float32
  norm: str = 'GN'

  @nn.compact
  def __call__(
      self,
      features: ArrayList,
      train: bool = False) -> ArrayListDict:
    """Apply model to a list of features (from FPN).

    Args:
      features: list of arrays in FPN levels. Each array has a shape of
        B x H_l x W_l x C, where l is the FPN level index. Different levels
        have different spatial size, but the same channels.
      train: whether it is training.
    Returns:
      A dict with 'heatmaps' and 'box_regs', both are list of arrays in FPN
      levels. Each level of 'heatmaps' is in shape B x H_l x W_l x num_classes,
      each level of `box_regs` is in shape B x H_l x W_l x 4.
    """
    heatmaps = []
    box_regs = []
    tower = functools.partial(
        Tower, num_layers=self.num_layers,
        out_channels=self.out_channels, norm=self.norm, dtype=self.dtype)
    bbox_tower = tower(name='bbox_tower')
    bbox_pred = nn.Conv(
        4, kernel_size=(3, 3), padding=1,
        # make the initial prediction close to the regression range.
        bias_init=initializers.constant(8.0),
        kernel_init=initializers.normal(stddev=0.01),
        dtype=self.dtype, name='bbox_pred')
    if self.num_classes > 0:
      cls_tower = tower(name='cls_tower')
      cls_logits = nn.Conv(
          self.num_classes, kernel_size=(3, 3), padding=1,
          bias_init=initializers.constant(-4.6),  # sigmoid(-4.6) = 0.01
          kernel_init=initializers.normal(stddev=0.01),
          dtype=self.dtype, name='cls_logits')
    else:
      agn_hm = nn.Conv(
          1, kernel_size=(3, 3), padding=1,
          bias_init=initializers.constant(-4.6),  # sigmoid(-4.6) = 0.01
          kernel_init=initializers.normal(stddev=0.01),
          dtype=self.dtype, name='agn_hm')
    for l in range(self.num_levels):
      feature = features[l]
      bbox_feat = bbox_tower(feature, train=train)
      reg = bbox_pred(bbox_feat)
      reg = Scale(name=f'scales.{l}')(reg)
      reg = nn.relu(reg)
      box_regs.append(reg)
      if self.num_classes > 0:
        cls_feat = cls_tower(feature, train=train)
        heatmaps.append(cls_logits(cls_feat))
      else:
        heatmaps.append(agn_hm(bbox_feat))
    return {'heatmaps': heatmaps, 'box_regs': box_regs}

