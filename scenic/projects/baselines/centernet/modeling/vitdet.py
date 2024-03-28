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

"""ViTDet with simple FPN."""

import copy
import functools
from typing import Any, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.projects.baselines.centernet.modeling import fpn

__all__ = ['ViT', 'SimpleFeaturePyramid']

KERNEL_INIT = {
    'normal': nn.initializers.normal(stddev=0.02),
}


class Attention(nn.Module):
  """Multi-head Attention block with relative position embeddings.

  Attributes:
  dim (int): Number of input channels.
  num_heads (int): Number of attention heads.
  qkv_bias (bool:  If True, add a learnable bias to query, key, value.
  beit_like_qkv_bias (bool): no bias for k.
  use_rel_pos (bool): If True, add relative positional embeddings to the
    attention map.
  rel_pos_zero_init (bool): If True, zero initialize relative positional
    parameters.
  input_size (int or None): Input resolution for calculating the relative
    positional parameter size.
  """
  dim: int
  num_heads: int = 8
  qkv_bias: bool = True
  beit_like_qkv_bias: bool = False
  use_rel_pos: bool = False
  rel_pos_zero_init: bool = True
  input_size: Optional[Any] = None
  kernel_init: str = 'normal'
  dtype: jnp.dtype = jnp.float32

  def get_rel_pos(self, q_size, k_size, rel_pos):
    """Get relative positional embeddings.

    Args:
      q_size (int): size of query q.
      k_size (int): size of key k.
      rel_pos (Tensor): relative position embeddings (L, C).
    Returns:
      Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
      # Interpolate rel pos.
      rel_pos_resized = jax.image.resize(
          rel_pos,
          shape=(max_rel_dist, rel_pos.shape[1]),
          method='linear',
      )
    else:
      rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = jnp.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = jnp.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(
        q_size / k_size, 1.0)
    relative_coords = relative_coords.astype(jnp.int32).reshape(-1)
    return rel_pos_resized[relative_coords].reshape(q_size, k_size, -1)

  def add_decomposed_rel_pos(
      self, attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """Calculate decomposed Relative Positional Embeddings from paper:`mvitv2`.

    Args:
      attn (Tensor): attention map.
      q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
      rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
      rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
      q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
      k_size (Tuple): spatial sequence size of key k with (k_h, k_w).
    Returns:
      attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    rh = self.get_rel_pos(q_h, k_h, rel_pos_h)
    rw = self.get_rel_pos(q_w, k_w, rel_pos_w)

    batch, _, dim = q.shape
    r_q = q.reshape(batch, q_h, q_w, dim)
    rel_h = jnp.einsum('bhwc,hkc->bhwk', r_q, rh)
    rel_w = jnp.einsum('bhwc,wkc->bhwk', r_q, rw)

    attn = (
        attn.reshape(batch, q_h, q_w, k_h, k_w) + rel_h[
            :, :, :, :, None] + rel_w[:, :, :, None, :]
    ).reshape(batch, q_h * q_w, k_h * k_w)

    return attn

  @nn.compact
  def __call__(self, x):
    batch, height, width, _ = x.shape
    head_dim = self.dim // self.num_heads
    if self.beit_like_qkv_bias:
      q_bias = self.param(
          'q_bias', nn.initializers.zeros, (self.dim,))
      v_bias = self.param(
          'v_bias', nn.initializers.zeros, (self.dim,))
      k_bias = jnp.zeros((self.dim,), dtype=jnp.float32)
      qkv_bias = jnp.concatenate([q_bias, k_bias, v_bias], axis=0)
      qkv = nn.Dense(
          self.dim * 3, use_bias=False, dtype=self.dtype,
          kernel_init=KERNEL_INIT[self.kernel_init], name='qkv')(
              x)  # batch x height x width x 3dim
      qkv = qkv + qkv_bias[None, None, None, :]
    else:
      qkv = nn.Dense(
          self.dim * 3, use_bias=self.qkv_bias, dtype=self.dtype,
          kernel_init=KERNEL_INIT[self.kernel_init], name='qkv')(
              x)  # batch x height x width x 3dim
    qkv = qkv.reshape(batch, height * width, 3, self.num_heads, -1).transpose(
        2, 0, 3, 1, 4)  # 3 x batch x num_heads x num_tokens x D
    qkv = qkv.reshape(3, batch * self.num_heads, height * width, -1)
    q, k, v = qkv[0], qkv[1], qkv[2]  # [batch * num_heads, num_tokens, D]
    attn = (q * (head_dim ** -0.5)) @ k.transpose(
        0, 2, 1)  # [batch * num_heads, num_tokens, num_tokens]
    if self.use_rel_pos:
      rel_pos_h = self.param(
          'rel_pos_h', nn.initializers.zeros,
          (2 * self.input_size[0] - 1, head_dim))
      rel_pos_w = self.param(
          'rel_pos_w', nn.initializers.zeros,
          (2 * self.input_size[0] - 1, head_dim))
      attn = self.add_decomposed_rel_pos(
          attn, q, rel_pos_h, rel_pos_w,
          (height, width), (height, width))
    attn = jax.nn.softmax(attn)
    x = (attn @ v).reshape(batch, self.num_heads, height, width, -1).transpose(
        0, 2, 3, 1, 4).reshape(batch, height, width, -1)
    x = nn.Dense(
        self.dim, dtype=self.dtype, kernel_init=KERNEL_INIT[self.kernel_init],
        name='proj')(x)
    return x


class Mlp(nn.Module):
  """Multilayer perceptron."""

  hidden_features: int
  out_features: int
  kernel_init: str = 'normal'
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(
        self.hidden_features, dtype=self.dtype,
        kernel_init=KERNEL_INIT[self.kernel_init], name='fc1')(x)
    x = nn.gelu(x, approximate=False)
    x = nn.Dense(
        self.out_features, dtype=self.dtype,
        kernel_init=KERNEL_INIT[self.kernel_init], name='fc2')(x)
    return x


class Block(nn.Module):
  """Transformer blocks with support of window attention and residual blocks.

  Attributes:
    dim (int): Number of input channels.
    num_heads (int): Number of attention heads in each ViT block.
    mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    qkv_bias (bool): If True, add a learnable bias to query, key, value.
    beit_like_qkv_bias (bool): no bias for k.
    drop_path (float): Stochastic depth rate.
    use_rel_pos (bool): If True, add relative positional embeddings to the
      attention map.
    rel_pos_zero_init (bool): If True, zero initialize relative positional
      parameters.
    window_size (int): Window size for window attention blocks. If it equals 0,
      then not use window attention.
    input_size (int or None): Input resolution for calculating the relative
      positional parameter size.
  """
  dim: int
  num_heads: int
  mlp_ratio: float = 4.0
  qkv_bias: bool = True
  beit_like_qkv_bias: bool = False
  drop_path: float = 0.0
  use_rel_pos: bool = False
  rel_pos_zero_init: bool = True
  window_size: int = 0
  input_size: Optional[Any] = None
  kernel_init: str = 'normal'
  layer_scale_init_value: float = -1.0
  dtype: jnp.dtype = jnp.float32

  def window_partition(self, x):
    """Partition into non-overlapping windows with padding if needed.

    Args:
      x (array): input tokens with [B, H, W, C].
    Returns:
      windows: windows after partition with [B * num_windows, window_size,
        window_size, C].
      (Hp, Wp): padded height and width before partition
    """
    batch, h, w, c = x.shape

    pad_h = (self.window_size - h % self.window_size) % self.window_size
    pad_w = (self.window_size - w % self.window_size) % self.window_size
    if pad_h > 0 or pad_w > 0:
      x = jnp.pad(
          x, ((0, 0), (0, pad_w), (0, pad_h), (0, 0)),
          'constant', constant_values=0)
    hp, wp = h + pad_h, w + pad_w

    x = x.reshape(
        batch, hp // self.window_size, self.window_size,
        wp // self.window_size, self.window_size, c)
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(
        -1, self.window_size, self.window_size, c)
    return windows, (hp, wp)

  def window_unpartition(self, windows, pad_hw, hw):
    """Window unpartition into original sequences and removing padding.

    Args:
      windows (array): inputs: [B * num_windows, window_size, window_size, C].
      pad_hw (Tuple): padded height and width (Hp, Wp).
      hw (Tuple): original height and width (H, W) before padding.

    Returns:
      x: unpartitioned sequences with [B, H, W, C].
    """
    hp, wp = pad_hw
    h, w = hw
    batch = windows.shape[0] // (
        hp * wp // self.window_size // self.window_size)
    x = windows.reshape(
        batch,
        hp // self.window_size, wp // self.window_size,
        self.window_size, self.window_size, -1)
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(batch, hp, wp, -1)
    if hp > h or wp > w:
      x = x[:, :h, :w, :]
    return x

  def get_keep_pattern(self,
                       x: jnp.ndarray,
                       deterministic: bool) -> jnp.ndarray:
    """DropPath Layer."""
    if not deterministic and self.drop_path:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      drop_pattern = jax.random.bernoulli(
          self.make_rng('dropout'), self.drop_path, shape).astype(self.dtype)
      keep_pattern = (1. - drop_pattern)
      if self.drop_path < 1.:
        keep_pattern = keep_pattern / (1. - self.drop_path)
      return keep_pattern
    else:
      return 1.0  # pytype: disable=bad-return-type  # jax-ndarray

  @nn.compact
  def __call__(self, x, train=False):
    shortcut = x
    ln = functools.partial(nn.LayerNorm, epsilon=1e-6, dtype=self.dtype)
    x = ln(name='norm1')(x)
    # Window partition
    if self.window_size > 0:
      h, w = x.shape[1], x.shape[2]
      x, pad_hw = self.window_partition(x)

    x = Attention(
        self.dim,
        num_heads=self.num_heads,
        qkv_bias=self.qkv_bias,
        beit_like_qkv_bias=self.beit_like_qkv_bias,
        use_rel_pos=self.use_rel_pos,
        rel_pos_zero_init=self.rel_pos_zero_init,
        input_size=self.input_size if self.window_size == 0 else (
            self.window_size, self.window_size),
        kernel_init=self.kernel_init,
        dtype=self.dtype,
        name='attn')(x)

    # Reverse window partition
    if self.window_size > 0:
      x = self.window_unpartition(x, pad_hw, (h, w))

    if self.layer_scale_init_value > 0:
      gamma_1 = self.param(
          'gamma_1',
          nn.initializers.constant(self.layer_scale_init_value),
          (self.dim))
      x = x * gamma_1[..., :]
    x = shortcut + self.get_keep_pattern(x, not train) * x
    y = ln(name='norm2')(x)
    y = Mlp(
        int(self.dim * self.mlp_ratio),
        self.dim,
        kernel_init=self.kernel_init,
        dtype=self.dtype,
        name='mlp')(y)
    if self.layer_scale_init_value > 0:
      gamma_2 = self.param(
          'gamma_2',
          nn.initializers.constant(self.layer_scale_init_value),
          (self.dim))
      y = y * gamma_2[..., :]
    x = x + self.get_keep_pattern(y, not train) * y
    return x


class ViT(nn.Module):
  """This module implements Vision Transformer (ViT) backbone in paper:`vitdet`.

  "Exploring Plain Vision Transformer Backbones for Object Detection",
  https://arxiv.org/abs/2203.16527

  Attributes:
    img_size (int): Input image size.
    patch_size (int): Patch size.
    in_chans (int): Number of input image channels.
    embed_dim (int): Patch embedding dimension.
    depth (int): Depth of ViT.
    num_heads (int): Number of attention heads in each ViT block.
    mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    qkv_bias (bool): If True, add a learnable bias to query, key, value.
    beit_like_qkv_bias (bool): no bias for k.
    drop_path_rate (float): Stochastic depth rate.
    use_abs_pos (bool): If True, use absolute positional embeddings.
    use_rel_pos (bool): If True, add relative positional embeddings to the
      attention map.
    rel_pos_zero_init (bool): If True, zero initialize relative positional
      parameters.
    window_size (int): Window size for window attention blocks.
    window_block_indexes (list): Indexes for blocks using window attention.
    pretrain_img_size (int): input image size for pretraining models.
    pretrain_use_cls_token (bool): If True, pretrainig models use class token.
  """
  img_size: int = 1024
  patch_size: int = 16
  in_chans: int = 3
  embed_dim: int = 768
  depth: int = 12
  num_heads: int = 12
  mlp_ratio: float = 4.0
  qkv_bias: bool = True
  beit_like_qkv_bias: bool = False
  drop_path_rate: float = 0.1
  use_abs_pos: bool = True
  use_rel_pos: bool = True
  rel_pos_zero_init: bool = True
  window_size: int = 14
  window_block_indexes: Any = (0, 1, 3, 4, 6, 7, 9, 10)
  pretrain_img_size: int = 224
  pretrain_use_cls_token: int = 1
  kernel_init: str = 'normal'
  layer_scale_init_value: float = -1.0
  freeze_vit_layer: int = -1
  use_ln_pre: bool = False
  dtype: jnp.dtype = jnp.float32
  return_intermediate: Optional[int] = None

  def _get_abs_pos(self, abs_pos, hw):
    """Calculate absolute positional embeddings.

    If needed, resize embeddings and remove cls_token dimension for the original
      embeddings.
    Args:
      abs_pos (array): absolute positional embeddings with (1, num_position, C).
      hw (Tuple): size of input image tokens.
    Returns:
      Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if self.pretrain_use_cls_token:
      abs_pos = abs_pos[:, self.pretrain_use_cls_token:]
    xy_num = abs_pos.shape[1]
    size = int(xy_num ** 0.5)
    assert size * size == xy_num
    abs_pos = abs_pos.reshape(abs_pos.shape[0], size, size, -1)
    if size != h or size != w:
      new_abs_pos = jax.image.resize(
          abs_pos,
          (abs_pos.shape[0], h, w, abs_pos.shape[3]),
          method='bicubic',
      )
    else:
      new_abs_pos = abs_pos
    return new_abs_pos

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               train: bool = False,):
    x = nn.Conv(
        self.embed_dim, (self.patch_size, self.patch_size),
        strides=(self.patch_size, self.patch_size),
        padding='VALID',
        dtype=self.dtype,
        name='patch_embed.proj')(x)
    if self.use_abs_pos:
      num_patches = (self.pretrain_img_size // self.patch_size) ** 2
      num_positions = num_patches + self.pretrain_use_cls_token
      pos_embed = self.param(
          'pos_embed', nn.initializers.zeros,
          (1, num_positions, self.embed_dim))
      x = x + self._get_abs_pos(pos_embed, (x.shape[1], x.shape[2]))
    dp_rates = [
        self.drop_path_rate * i / (self.depth - 1) for i in range(self.depth)]
    if self.use_ln_pre:
      x = nn.LayerNorm(name='ln_pre')(x)

    intermediate_layer = None
    for i in range(self.depth):
      x = Block(
          dim=self.embed_dim,
          num_heads=self.num_heads,
          mlp_ratio=self.mlp_ratio,
          qkv_bias=self.qkv_bias,
          beit_like_qkv_bias=self.beit_like_qkv_bias,
          drop_path=dp_rates[i],
          use_rel_pos=self.use_rel_pos,
          rel_pos_zero_init=self.rel_pos_zero_init,
          window_size=self.window_size if i in self.window_block_indexes else 0,
          input_size=(
              self.img_size // self.patch_size,
              self.img_size // self.patch_size),
          kernel_init=self.kernel_init,
          dtype=self.dtype,
          layer_scale_init_value=self.layer_scale_init_value,
          name=f'blocks.{i}',
          )(x, train=train)
      if i + 1 < self.freeze_vit_layer:
        x = jax.lax.stop_gradient(x)
      if self.return_intermediate is not None and (
          i == self.return_intermediate):
        intermediate_layer = x

    if intermediate_layer is not None:
      return intermediate_layer
    return x

SIZE_CONFIGS = {
    'B': (768, 12, 12, 0.1, (0, 1, 3, 4, 6, 7, 9, 10)),
    'L': (1024, 24, 16, 0.4, (
        0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22)),
    'H': (1280, 32, 16, 0.5, (
        0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21,
        22, 24, 25, 26, 27, 28, 29, 30)),
}


class SimpleFeaturePyramid(nn.Module):
  """This module implements SimpleFeaturePyramid in paper:`vitdet`.

  It creates pyramid features built on top of the input feature map.

  Attributes:
    in_dim (int): input dim
    out_channels (int): number of channels in the output feature maps.
    scale_factors (list[float]): list of scaling factors to upsample or
      downsample the input features for creating pyramid features.
    num_top_blocks (int): top level downsample block
    norm (str): the normalization to use.
  """
  in_dim: int = 768
  out_channels: int = 256
  scale_factors: Any = (2.0, 1.0, 0.5)
  num_top_blocks: int = 2
  num_additional_convs: int = 0
  backbone_args: ml_collections.ConfigDict = flax.struct.field(
      default_factory=ml_collections.ConfigDict
  )
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = False):

    backbone_args = copy.deepcopy(self.backbone_args)
    backbone_args.unlock()
    sz = backbone_args.get('size', 'B')
    freeze_backbone = backbone_args.get('freeze_backbone', False)
    for delete_key in [
        'size', 'freeze_backbone', 'bifpn_norm', 'num_repeats']:
      if delete_key in backbone_args:
        del backbone_args[delete_key]
    dim, depth, num_heads, dp, window_block_indexes = SIZE_CONFIGS[sz]
    backbone_args['embed_dim'] = backbone_args.get('embed_dim', dim)
    backbone_args['depth'] = backbone_args.get('depth', depth)
    backbone_args['num_heads'] = backbone_args.get('num_heads', num_heads)
    backbone_args['drop_path_rate'] = backbone_args.get('drop_path_rate', dp)
    backbone_args['window_block_indexes'] = backbone_args.get(
        'window_block_indexes', window_block_indexes)
    backbone_args.lock()
    backbone_net = ViT(**backbone_args, dtype=self.dtype, name='net')

    if freeze_backbone:
      features = jax.lax.stop_gradient(backbone_net(x, train=False))
    else:
      features = backbone_net(x, train=train)
    results = []
    dim = self.in_dim
    conv_transpose = functools.partial(
        nn.ConvTranspose, kernel_size=(2, 2), strides=(2, 2), dtype=self.dtype)
    ln = functools.partial(nn.LayerNorm, epsilon=1e-6)
    conv = functools.partial(nn.Conv, use_bias=False, dtype=self.dtype)
    for scale in self.scale_factors:
      x = features
      if scale == 4.0:
        stage, idx_base = 2, 4
        x = conv_transpose(dim // 2, name='simfp_2.0')(x)
        x = ln(name='simfp_2.1')(x)
        x = nn.gelu(x, approximate=False)
        x = conv_transpose(dim // 4, name='simfp_2.3')(x)
      elif scale == 2.0:
        stage, idx_base = 3, 1
        x = conv_transpose(dim // 2, name='simfp_3.0')(x)
      elif scale == 1.0:
        stage, idx_base = 4, 0
      elif scale == 0.5:
        stage, idx_base = 5, 1
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
      else:
        raise NotImplementedError(f'scale_factor={scale} is not supported yet.')
      x = conv(
          self.out_channels, kernel_size=(1, 1),
          name=f'simfp_{stage}.{idx_base}')(x)
      x = ln(name=f'simfp_{stage}.{idx_base}.norm')(x)
      x = conv(
          self.out_channels, kernel_size=(3, 3), padding=[(1, 1), (1, 1)],
          name=f'simfp_{stage}.{idx_base + 1}')(x)
      x = ln(name=f'simfp_{stage}.{idx_base + 1}.norm')(x)
      if self.num_additional_convs > 0:
        for i in range(self.num_additional_convs):
          x = conv(
              self.out_channels,
              kernel_size=(3, 3),
              padding=[(1, 1), (1, 1)],
              name=f'simfp_{stage}.{idx_base + 2 + i}')(
                  x)
          x = ln(name=f'simfp_{stage}.{idx_base + 2 + i}.norm')(x)
      results.append(x)

    if self.num_top_blocks == 1:
      x = nn.max_pool(
          results[-1], (1, 1), strides=(2, 2), padding=[(0, 0), (0, 0)])
      results.append(x)
    elif self.num_top_blocks == 2:
      top_block = fpn.TwiceDownsampleBlock(
          out_channels=self.out_channels, dtype=self.dtype, name='top_block')
      p6, p7 = top_block(results[-1])
      results.extend([p6, p7])
    else:
      if self.num_top_blocks != 0:
        raise NotImplementedError(
            f'num_top_blocks={self.num_top_blocks} is not supported yet.')
    return results

