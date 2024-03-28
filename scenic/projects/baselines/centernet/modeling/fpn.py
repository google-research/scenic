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

"""FPN Backbones for object detection."""

import functools
from typing import List

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

from scenic.projects.baselines.centernet.modeling import convnext

ArrayList = List[jnp.ndarray]

BOTTOM_UP_CLASS = {
    'convnext': convnext.ConvNeXt,
}


class TwiceDownsampleBlock(nn.Module):
  """Generate two more downsampled feature maps from a input feature."""
  out_channels: int = 256
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
    """Apply model on a single output feature.

    Args:
      x: Array in shape B x H x W x C.
      train: Whether it is training.

    Returns:
      List of array [p6, p7]. p6 is in shape B x H//2 x W/2 x out_channels.
        p7 is in shape B x H//4 x W// 4 x out_channels.
    """
    conv = functools.partial(
        nn.Conv, features=self.out_channels, kernel_size=(3, 3),
        strides=2, padding=1, dtype=self.dtype)
    p6 = conv(name='p6')(x)
    p7 = conv(name='p7')(nn.relu(p6))
    return [p6, p7]  # pytype: disable=bad-return-type  # jax-ndarray


class FPN(nn.Module):
  """FPN implementation following detectron2.

  Attributes:
    backbone_name: string of the backbone name.
    in_features: names of the input feature from the backbone. For example,
      ['stage_2', 'stage_3', 'stage_4']. The name should be from the output dict
      of the backbone.
    out_channels: number of channels of the FPN output. All levels should have
      the same channel.
    num_out_levels: number of output FPN levels.
    start_idx: the stage index of the first output block. Following the
      convention in detectron2, this implies the output stride of the first
      level. I.e., the stride of the first level is 2 ** start_idx.
    norm: normalization layer type. Only no normalization is supported yet.
      TODO(zhouxy): add other normalization types.
    dtype: data type of the computation (default: float32).
  """
  backbone_name: str
  in_features: List[str]
  out_channels: int = 256
  num_out_levels: int = 5
  start_idx: int = 3
  norm: str = ''
  backbone_args: ml_collections.ConfigDict = flax.struct.field(
      default_factory=ml_collections.ConfigDict
  )
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = False) -> ArrayList:
    """Apply backbone + FPN to the input image.

    Args:
      x: array of the preprocessed input images, in shape B x H x W x 3.
      train: Whether it is training.

    Returns:
      outputs: A list of array in different FPN layers. Array in level l is in
        shape B x H_l x W_l x out_channels, where H_l and W_l are the spatial
        size of each level. H_l = H // (2 ** l).
    """
    assert not self.norm, 'Normalization layers in FPN is not supported yet!'
    use_bias = not self.norm

    lateral_conv = functools.partial(
        nn.Conv, features=self.out_channels, kernel_size=(1, 1),
        use_bias=use_bias, dtype=self.dtype)
    output_conv = functools.partial(
        nn.Conv, features=self.out_channels, kernel_size=(3, 3), padding=1,
        use_bias=use_bias, dtype=self.dtype)

    bottom_up_class = BOTTOM_UP_CLASS[self.backbone_name]
    bottom_up_features = bottom_up_class(
        num_outputs=None, **self.backbone_args, name='bottom_up')(
            x, train=train)
    results = {}
    mid_idx = self.start_idx + len(self.in_features) - 1
    prev_features = lateral_conv(name=f'fpn_lateral{mid_idx}')(
        bottom_up_features[self.in_features[-1]])
    out = output_conv(name=f'fpn_output{mid_idx}')(
        prev_features)
    results[f'p{mid_idx}'] = out

    for idx in range(1, len(self.in_features)):
      features = bottom_up_features[self.in_features[-1 - idx]]
      top_down_features = jax.image.resize(
          prev_features,
          (prev_features.shape[0], prev_features.shape[1] * 2,
           prev_features.shape[2] * 2, prev_features.shape[3]),
          method='nearest'
      )
      lateral_features = lateral_conv(
          name=f'fpn_lateral{mid_idx-idx}')(features)
      prev_features = lateral_features + top_down_features
      out = output_conv(
          name=f'fpn_output{mid_idx-idx}')(prev_features)
      results[f'p{mid_idx-idx}'] = out

    # TODO(zhouxy): Support other top blocks
    top_block = TwiceDownsampleBlock(
        out_channels=self.out_channels, dtype=self.dtype, name='top_block')
    results['p6'], results['p7'] = top_block(results[f'p{mid_idx}'])
    outputs = [results[f'p{i}'] for i in range(3, 8)]

    return outputs
