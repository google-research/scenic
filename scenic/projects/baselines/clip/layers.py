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

"""OpenAI's CLIP models in Flax.

The implementation is based on an initial port of code in
https://github.com/openai/CLIP to JAX, by pooleb@google.com.
"""

import functools
from typing import Sequence, Optional, Union, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

# TODO(scenic): Make initialization of all layers identical to official one.
# Note: this doesn't matter for loading pretrained models.

# Match PyTorch default LayerNorm epsilon of 1e-5 (FLAX defaults to 1e-6).
LayerNorm = functools.partial(nn.LayerNorm, epsilon=1e-5)


def quick_gelu(x: jnp.ndarray) -> jnp.ndarray:
  return x * jax.nn.sigmoid(1.702 * x)


class Shortcut(nn.Module):
  """Shortcut in ResNet.

  Attributes:
    features: Number of features.
    stride: Stride of the down-sampled output.
  """
  features: int
  stride: int

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x = nn.avg_pool(x, (self.stride, self.stride), (self.stride, self.stride))
    x = nn.Conv(
        self.features, (1, 1), strides=(1, 1), use_bias=False, name='0')(x)
    x = nn.BatchNorm(use_running_average=True, name='1')(x)
    return x


class Bottleneck(nn.Module):
  """Bottleneck layer of ResNet.

  Attributes:
    features: Number of features.
    stride: Stride of the down-sampled output.
    expansion: Expansion of feature dimension.
  """
  features: int
  stride: int = 1
  expansion: int = 4

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    conv1 = nn.Conv(self.features, (1, 1), use_bias=False, name='conv1')
    bn1 = nn.BatchNorm(use_running_average=True, name='bn1')

    conv2 = nn.Conv(self.features, (3, 3), padding=[(1, 1), (1, 1)],
                    use_bias=False, name='conv2')
    bn2 = nn.BatchNorm(use_running_average=True, name='bn2')

    conv3 = nn.Conv(
        self.features * self.expansion, (1, 1), use_bias=False, name='conv3')
    bn3 = nn.BatchNorm(use_running_average=True, name='bn3')

    out = nn.relu(bn1(conv1(x)))
    out = nn.relu(bn2(conv2(out)))
    out = nn.avg_pool(out, (self.stride, self.stride),
                      (self.stride, self.stride))
    out = bn3(conv3(out))

    downsample = (
        self.stride > 1 or x.shape[-1] != self.features * self.expansion
    )
    if downsample:
      x = Shortcut(features=self.features * self.expansion,
                   stride=self.stride, name='downsample')(x)

    out += x
    out = nn.relu(out)
    return out


class AttentionPool(nn.Module):
  """Attention pooling layer.

  Attributes:
    num_heads: Number of heads.
    features: Number of features.
  """
  num_heads: int
  features: Optional[int] = None

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x = x.reshape(x.shape[0], -1, x.shape[3])

    x = jnp.concatenate([x.mean(axis=1, keepdims=True), x], axis=1)

    positional_embedding = self.param(
        'positional_embedding',
        jax.nn.initializers.normal(1. / x.shape[-1]**0.5),
        (x.shape[1], x.shape[2]))
    attn = nn.MultiHeadDotProductAttention(
        self.num_heads,
        qkv_features=x.shape[-1],
        use_bias=True,
        out_features=self.features,
        name='attn')

    x = x + positional_embedding[jnp.newaxis].astype(x.dtype)
    x = attn(x[:, :1], x)
    return x[:, 0]


class ResNetStage(nn.Module):
  """Attention pooling layer.

  Attributes:
    features: Number of features.
    num_layers: Number of bottleneck blocks.
    stride: Stride in the Bottleneck module.
  """
  features: int
  num_layers: int
  stride: int = 1

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x = Bottleneck(self.features, self.stride, name='0')(x)
    for i in range(1, self.num_layers):
      x = Bottleneck(self.features, name=str(i))(x)
    return x


class ModifiedResNet(nn.Module):
  """A ResNet class that is similar to torchvision's with changes.

  - There are now 3 "stem" convolutions as opposed to 1, with an average pool
  instead of a max pool.
  - Performs anti-aliasing strided convolutions, where an avgpool is
  prepended to convolutions with stride > 1 - The final pooling layer is a
  QKV attention instead of an average pool.

  Attributes:
    features: Number of features.
    out_features: Number of output features. If None, return resnet feature-map.
    num_layers: Number of layers for each block.
    num_heads: Number of heads.
  """
  features: int
  out_features: Optional[int]
  num_layers: Sequence[int]
  num_heads: Optional[int]

  def setup(self):
    # The 3-layer stem.
    self.conv1 = nn.Conv(
        self.features // 2,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=[(1, 1), (1, 1)],
        use_bias=False,
        name='conv1')
    self.bn1 = nn.BatchNorm(use_running_average=True, name='bn1')
    self.conv2 = nn.Conv(
        self.features // 2,
        kernel_size=(3, 3),
        padding=[(1, 1), (1, 1)],
        use_bias=False,
        name='conv2')
    self.bn2 = nn.BatchNorm(use_running_average=True, name='bn2')
    self.conv3 = nn.Conv(
        self.features,
        kernel_size=(3, 3),
        padding=[(1, 1), (1, 1)],
        use_bias=False,
        name='conv3')
    self.bn3 = nn.BatchNorm(use_running_average=True, name='bn3')

    # Residual layers.
    self.layer1 = ResNetStage(self.features, self.num_layers[0], name='layer1')
    self.layer2 = ResNetStage(
        self.features * 2, self.num_layers[1], stride=2, name='layer2')
    self.layer3 = ResNetStage(
        self.features * 4, self.num_layers[2], stride=2, name='layer3')
    self.layer4 = ResNetStage(
        self.features * 8, self.num_layers[3], stride=2, name='layer4')
    if self.out_features is not None:
      self.attnpool = AttentionPool(
          self.num_heads, self.out_features, name='attnpool')

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

    def stem(x):
      for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2),
                       (self.conv3, self.bn3)]:
        x = nn.relu(bn(conv(x)))
      x = nn.avg_pool(x, (2, 2), (2, 2))
      return x

    x = stem(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = feature_map = self.layer4(x)

    if self.out_features is not None:
      x = self.attnpool(x)

    return x, feature_map  # pytype: disable=bad-return-type  # jax-ndarray


class MLP(nn.Module):
  """Simple MLP for Transformer."""

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    ch = x.shape[-1]
    x = nn.Dense(4 * ch, name='c_fc')(x)
    x = quick_gelu(x)
    x = nn.Dense(ch, name='c_proj')(x)
    return x


class ResidualAttentionBlock(nn.Module):
  """Self-attention block of Transformer.

  Attributes:
    num_heads: Number of heads.
  """
  num_heads: int

  @nn.compact
  def __call__(self, x: jnp.ndarray, attn_mask=None) -> jnp.ndarray:
    xn = LayerNorm(name='ln_1')(x)
    x = x + nn.SelfAttention(
        self.num_heads, name='attn', deterministic=True)(xn, attn_mask)
    xn = LayerNorm(name='ln_2')(x)
    x = x + MLP(name='mlp')(xn)
    return x


class Transformer(nn.Module):
  """Transformer module.

  Attributes:
    features: Number of features.
    num_layers: Number of layers for each block.
    num_heads: Number of heads.
    use_underscore_module_name: Optionally replace '.' with '_' in parameter
      naming for PAX checkpoint loading.
  """
  features: int
  num_layers: int
  num_heads: int
  use_underscore_module_name: bool = False

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               attn_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    def _n(name):
      """A helper function that optionally replace '.' with '_'."""
      if self.use_underscore_module_name:
        return name.replace('.', '_')
      else:
        return name

    for i in range(self.num_layers):
      x = ResidualAttentionBlock(
          num_heads=self.num_heads, name=_n(f'resblocks.{i}'))(x, attn_mask)
    return x


class VisionTransformer(nn.Module):
  """Vision Transformer.

  Attributes:
    patch_size: The size of the patches to embed.
    features: Number of features.
    num_layers: Number of transformer blocks (self-attn + MLP).
    num_heads: Number of attention heads.
    out_features: Number of output features. If None, return transformer output.
    use_underscore_module_name: Optionally replace '.' with '_' in parameter
      naming for PAX checkpoint loading.
  """
  patch_size: int
  features: int
  num_layers: int
  num_heads: int
  out_features: Optional[int]
  use_underscore_module_name: bool = False

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               attn_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    x = nn.Conv(self.features,
                kernel_size=(self.patch_size, self.patch_size),
                strides=(self.patch_size, self.patch_size),
                use_bias=False, name='conv1')(x)
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    scale = 1.0 / jnp.sqrt(self.features)
    class_embedding = self.param('class_embedding',
                                 jax.nn.initializers.normal(stddev=scale),
                                 (self.features,))
    x = jnp.concatenate((jnp.tile(class_embedding[None, None, :],
                                  (x.shape[0], 1, 1)), x),
                        axis=1)
    positional_embedding = self.param('positional_embedding',
                                      jax.nn.initializers.normal(stddev=scale),
                                      (x.shape[1], self.features))
    x = x + positional_embedding[None]

    x = LayerNorm(name='ln_pre')(x)
    x = feature_map = Transformer(
        features=self.features,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        use_underscore_module_name=self.use_underscore_module_name,
        name='transformer')(
            x)

    if self.out_features is not None:
      x = LayerNorm(name='ln_post')(x[:, 0])
      x = nn.Dense(self.out_features, use_bias=False, name='proj')(x)
    else:
      x = LayerNorm(name='ln_post')(x)

    return x, feature_map  # pytype: disable=bad-return-type  # jax-ndarray


class TextEncoder(nn.Module):
  """Text Transformer.

  Attributes:
    vocab_size: Size of the vocabulary.
    features: Number of features.
    num_layers: Number of transformer blocks (self-attn + MLP).
    num_heads: Number of attention heads.
    out_features: Size of the final text embedding.
    use_underscore_module_name: Optionally replace '.' with '_' in parameter
      naming for PAX checkpoint loading.
  """
  vocab_size: int
  features: int
  num_layers: int
  num_heads: int
  out_features: int
  use_underscore_module_name: bool = False

  @nn.compact
  def __call__(self, text: jnp.ndarray) -> jnp.ndarray:
    positional_embedding = self.param('positional_embedding',
                                      jax.nn.initializers.zeros,
                                      (text.shape[1], self.features))
    mask = nn.combine_masks(
        nn.make_attention_mask(text > 0, text > 0), nn.make_causal_mask(text))
    x = nn.Embed(self.vocab_size, self.features, name='token_embedding')(text)
    x = x + positional_embedding[None]
    x = Transformer(
        self.features,
        self.num_layers,
        self.num_heads,
        use_underscore_module_name=self.use_underscore_module_name,
        name='transformer')(
            x, attn_mask=mask)
    x = LayerNorm(name='ln_final')(x)
    x = x[jnp.arange(x.shape[0]), text.argmax(-1)]
    x = nn.Dense(self.out_features, use_bias=False, name='text_projection')(x)
    return x


class CLIP(nn.Module):
  """Clip model consisting of a vision and text transformer.

  Attributes:
    vocab_size: Size of the vocabulary.
    embed_dim: Size of the text and vision embeddings.
    text_features: Number of features in text transformer.
    text_num_layers: Number of text transformer blocks (self-attn + MLP).
    text_num_heads: Number of heads in text transformer.
    vision_features: Number of features in vision transformer.
    vision_num_layers: Number of vision transformer blocks (self-attn + MLP).
    vision_head_dim: Number of features per vision transformer attention head.
    vision_patch_size: Size of patches to embed in vision transformer.
    use_underscore_module_name: Optionally replace '.' with '_' in parameter
      naming for PAX checkpoint loading.
  """
  vocab_size: int
  embed_dim: int
  # Text.
  text_features: int
  text_num_layers: int
  text_num_heads: int
  # Vision.
  vision_features: int
  vision_num_layers: Union[int, Sequence[int]]
  vision_head_dim: int = 64
  vision_patch_size: Optional[int] = None
  vision_return_map: bool = False
  use_underscore_module_name: bool = False

  def setup(self):
    if isinstance(self.vision_num_layers, (tuple, list)):
      self.vision_num_heads = self.vision_features * 32 // self.vision_head_dim
      self.visual = ModifiedResNet(
          num_layers=self.vision_num_layers,
          features=self.vision_features,
          num_heads=self.vision_num_heads,
          out_features=None if self.vision_return_map else self.embed_dim)
    else:
      self.vision_num_heads = self.vision_features // self.vision_head_dim
      self.visual = VisionTransformer(
          patch_size=self.vision_patch_size,
          features=self.vision_features,
          num_layers=self.vision_num_layers,
          num_heads=self.vision_num_heads,
          out_features=None if self.vision_return_map else self.embed_dim,
          use_underscore_module_name=self.use_underscore_module_name)
    self.text = TextEncoder(
        out_features=self.embed_dim,
        vocab_size=self.vocab_size,
        features=self.text_features,
        num_layers=self.text_num_layers,
        num_heads=self.text_num_heads,
        use_underscore_module_name=self.use_underscore_module_name)
    self.logit_scale = self.param('logit_scale', jax.nn.initializers.zeros, ())

  def encode_image(self,
                   image: jnp.ndarray,
                   normalize: bool = True) -> jnp.ndarray:
    x = self.visual(image)[0]
    if normalize:
      x /= jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x

  def encode_text(self,
                  text: jnp.ndarray,
                  normalize: bool = True) -> jnp.ndarray:
    x = self.text(text)
    if normalize:
      x /= jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x

  def __call__(self,
               image: jnp.ndarray,
               text: jnp.ndarray,
               normalize: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = y = None
    if image is not None:
      x = self.encode_image(image, normalize)
    if text is not None:
      y = self.encode_text(text, normalize)
    return x, y
