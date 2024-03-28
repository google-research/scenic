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

"""UNet model (http://arxiv.org/abs/1505.04597)."""

import functools
from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models.segmentation_model import SegmentationModel
from scenic.model_lib.layers import nn_layers
from scenic.model_lib.layers import nn_ops

Conv3x3 = functools.partial(nn.Conv, kernel_size=(3, 3))


class DeConv3x3(nn.Module):
  """Deconvolution layer for upscaling.

  Attributes:
    features: Num convolutional features.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

  features: int
  padding: str = 'SAME'
  use_batch_norm: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
    """Applies deconvolution with 3x3 kernel."""
    if self.padding == 'SAME':
      padding = ((1, 2), (1, 2))
    elif self.padding == 'VALID':
      padding = ((0, 0), (0, 0))
    else:
      raise ValueError(f'Unkonwn padding: {self.padding}')
    x = nn.Conv(
        features=self.features,
        kernel_size=(3, 3),
        input_dilation=(2, 2),
        padding=padding)(
            x)
    if self.use_batch_norm:
      x = nn.BatchNorm(use_running_average=not train)(x)
    return x


class ConvRelu2(nn.Module):
  """Two unpadded convolutions & relus.

  Attributes:
    features: Num convolutional features.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

  features: int
  padding: str = 'SAME'
  use_batch_norm: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
    x = Conv3x3(features=self.features, name='conv1', padding=self.padding)(x)
    if self.use_batch_norm:
      x = nn.BatchNorm(use_running_average=not train)(x)
    x = nn.relu(x)
    x = Conv3x3(features=self.features, name='conv2', padding=self.padding)(x)
    if self.use_batch_norm:
      x = nn.BatchNorm(use_running_average=not train)(x)
    x = nn.relu(x)
    return x


class DownsampleBlock(nn.Module):
  """Two unpadded convolutions & downsample 2x.

  Attributes:
    features: Num convolutional features.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

  features: int
  padding: str = 'SAME'
  use_batch_norm: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
    residual = x = ConvRelu2(
        features=self.features,
        padding=self.padding,
        use_batch_norm=self.use_batch_norm)(
            x, train=train)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    return x, residual  # pytype: disable=bad-return-type  # jax-ndarray


class BottleneckBlock(nn.Module):
  """Two unpadded convolutions, dropout & deconvolution.

  Attributes:
    features: Num convolutional features.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

  features: int
  padding: str = 'SAME'
  use_batch_norm: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
    x = ConvRelu2(
        self.features, padding=self.padding,
        use_batch_norm=self.use_batch_norm)(
            x, train=train)
    x = DeConv3x3(
        features=self.features // 2,
        name='deconv',
        padding=self.padding,
        use_batch_norm=self.use_batch_norm)(
            x, train=train)
    return x


class UpsampleBlock(nn.Module):
  """Two unpadded convolutions and upsample.

  Attributes:
    features: Num convolutional features.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

  features: int
  padding: str = 'SAME'
  use_batch_norm: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, residual, *, train: bool) -> jnp.ndarray:
    if residual is not None:
      x = jnp.concatenate([x, nn_ops.central_crop(residual, x.shape)], axis=-1)
    x = ConvRelu2(
        self.features, padding=self.padding,
        use_batch_norm=self.use_batch_norm)(
            x, train=train)
    x = DeConv3x3(
        features=self.features // 2,
        name='deconv',
        padding=self.padding,
        use_batch_norm=self.use_batch_norm)(
            x, train=train)
    return x


class OutputBlock(nn.Module):
  """Two unpadded convolutions followed by linear FC.


  Attributes:
    features: Num convolutional features.
    num_classes: Number of classes.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

  features: int
  num_classes: int
  padding: str = 'SAME'
  use_batch_norm: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
    x = ConvRelu2(
        self.features, padding=self.padding,
        use_batch_norm=self.use_batch_norm)(
            x, train=train)
    x = nn.Conv(
        features=self.num_classes, kernel_size=(1, 1), name='conv1x1')(
            x)
    if self.use_batch_norm:
      x = nn.BatchNorm(use_running_average=not train)(x)
    return x


class UNet(nn.Module):
  """U-Net from http://arxiv.org/abs/1505.04597.

  Based on:
  https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow2/Segmentation/UNet_Medical/model/unet.py
  Note that the default configuration `config.padding="VALID"` does only work
  with images that have a certain minimum size (e.g. 128x128 is too small).

  Attributes:
    num_classes: Number of classes.
    block_size: Sequence of feature sizes used in UNet blocks.
    padding: Type of padding.
    use_batch_norm: Whether to use batchnorm or not.
  """

  num_classes: int
  block_size: Tuple[int, ...] = (64, 128, 256, 512)
  padding: str = 'SAME'
  use_batch_norm: bool = True

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               *,
               train: bool,
               debug: bool = False) -> jnp.ndarray:
    """Applies the UNet model."""
    del debug
    skip_connections = []
    for i, features in enumerate(self.block_size):
      x, residual = DownsampleBlock(
          features=features,
          padding=self.padding,
          use_batch_norm=self.use_batch_norm,
          name=f'0_down_{i}')(
              x, train=train)
      skip_connections.append(residual)
    x = BottleneckBlock(
        features=2 * self.block_size[-1],
        padding=self.padding,
        use_batch_norm=self.use_batch_norm,
        name='1_bottleneck')(
            x, train=train)

    *upscaling_features, final_features = self.block_size[::-1]
    for i, features in enumerate(upscaling_features):
      x = UpsampleBlock(
          features=features,
          padding=self.padding,
          use_batch_norm=self.use_batch_norm,
          name=f'2_up_{i}')(
              x, residual=skip_connections.pop(), train=train)

    x = nn_layers.IdentityLayer(name='pre_logits')(x)
    x = OutputBlock(
        features=final_features,
        num_classes=self.num_classes,
        padding=self.padding,
        use_batch_norm=self.use_batch_norm,
        name='output_projection')(
            x, train=train)
    return x


class UNetSegmentationModel(SegmentationModel):
  """UNet model for segmentation task."""

  def build_flax_model(self):
    return UNet(
        num_classes=self.dataset_meta_data['num_classes'],
        padding=self.config.model.get('padding', 'SAME'),
        use_batch_norm=self.config.model.get('use_batch_norm', True),
        block_size=self.config.model.get('block_size', (64, 128, 256, 512)))

  def default_flax_model_config(self):
    return ml_collections.ConfigDict({
        'model':
            dict(
                padding='SAME',
                use_batch_norm=False,
                block_size=(64, 128, 256, 512),
                data_dtype_str='float32')
    })
