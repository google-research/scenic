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

r"""ViT implementation.

Pytorch reference: https://github.com/microsoft/GenerativeImage2Text/blob/\
main/generativeimage2text/layers/CLIP/model.py

Compare to a plain ViT, this implementation uses quick_gelu, supports
configurable normalizations before/ after the transformer blocks.

Currently the code also supports windows attention and relative positional
embedding. These are not used in the original GIT, but can be used for larger
input size in future developed.

"""

import functools
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

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
  """
  dim: int
  num_heads: int = 8
  qkv_bias: bool = True
  beit_like_qkv_bias: bool = False
  kernel_init: str = 'normal'
  with_grid_tokens: bool = False
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x):
    """Forward a block.

    Args:
      x: if self.with_grid_tokens == False (default), x should be in shape
        (batch_size, num_tokens, dim);
        if self.with_grid_tokens == True, x should be in shape
        (batch_size, height, width, dim);
    Returns:
      x: the same shape as the input.
    """

    batch, num_tokens, _ = x.shape
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
      qkv = qkv + qkv_bias[None, None, :]
    else:
      qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias, name='qkv')(
          x)  # batch x num_tokens x 3dim
    qkv = qkv.reshape(batch, num_tokens, 3, self.num_heads, -1).transpose(
        2, 0, 3, 1, 4)  # 3 x batch x num_heads x num_tokens x D

    qkv = qkv.reshape(3, batch * self.num_heads, num_tokens, -1)
    q, k, v = qkv[0], qkv[1], qkv[2]  # [batch * num_heads, num_tokens, D]
    attn = (q * (head_dim ** -0.5)) @ k.transpose(
        0, 2, 1)  # [batch * num_heads, num_tokens, num_tokens]

    attn = jax.nn.softmax(attn)
    x = (attn @ v).reshape(
        batch, self.num_heads, num_tokens, -1).transpose(
            0, 2, 1, 3).reshape(batch, num_tokens, -1)

    x = nn.Dense(self.dim, name='proj')(x)
    return x


def quick_gelu(x: jnp.ndarray) -> jnp.ndarray:
  return x * jax.nn.sigmoid(1.702 * x)


class Mlp(nn.Module):
  """Multilayer perceptron."""

  hidden_features: int
  out_features: int
  kernel_init: str = 'normal'
  dtype: jnp.dtype = jnp.float32
  activation: str = 'quick_gelu'

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(
        self.hidden_features, dtype=self.dtype,
        kernel_init=KERNEL_INIT[self.kernel_init], name='fc1')(x)
    if self.activation == 'quick_gelu':
      x = quick_gelu(x)
    elif self.activation == 'gelu':
      x = nn.gelu(x, approximate=False)
    else:
      raise NotImplementedError(self.activation)
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
  """
  dim: int
  num_heads: int
  mlp_ratio: float = 4.0
  qkv_bias: bool = True
  beit_like_qkv_bias: bool = False
  mlp_activation: str = 'quick_gelu'
  drop_path: float = 0.0
  layer_scale_init_value: float = -1.0
  kernel_init: str = 'normal'
  with_grid_tokens: bool = False
  dtype: jnp.dtype = jnp.float32

  def get_keep_pattern(self,
                       x: jnp.ndarray,
                       deterministic: bool):
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
      return 1.0

  @nn.compact
  def __call__(self, x, train: bool = False):
    shortcut = x
    ln = functools.partial(nn.LayerNorm, epsilon=1e-6)
    x = ln(name='norm1')(x)

    x = Attention(
        self.dim,
        num_heads=self.num_heads,
        qkv_bias=self.qkv_bias,
        beit_like_qkv_bias=self.beit_like_qkv_bias,
        with_grid_tokens=self.with_grid_tokens,
        name='attn')(x)

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
        activation=self.mlp_activation,
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
  """This module implements Vision Transformer (ViT) backbone.

  Attributes:
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
    pretrain_img_size (int): input image size for pretraining models.
    pretrain_use_cls_token (bool): If True, pretrainig models use class token.
    layer_scale_init_value (float): if add a scaling layer with the initialized
      value. Negative means not add such layers.
    kernel_init (str): functions to initialize layers. Currently only supports
      'normal'.
    freeze_vit_layer: (int). Freeze early layers.
    use_ln_pre (bool): if use a layer norm before transformer blocks. Used in
      CLIP/ GIT. Not used in MAE/ ViTDet.
    use_ln_post (bool): if use a layer norm after transformer blocks. Used in
      CLIP/ GIT. Not used in MAE/ ViTDet.
    pe_bias (bool): if the patch-embedding layer has bias. Not used in
      CLIP/ GIT. Used in MAE/ ViTDet.
    use_class_embedding (bool): if use the cls_token in the attention. If True,
      the attention block takes flattened tokens as input. If False, the
      attention block takes grid feature as input.
    dtype: jnp.dtype.
    window_block_indexes: Never used. Keep to make legacy configs runable.
    use_rel_pos: Never used. Keep to make lagacy configs runable.
  """
  patch_size: int = 16
  in_chans: int = 3
  embed_dim: int = 768
  depth: int = 12
  num_heads: int = 12
  mlp_ratio: float = 4.0
  qkv_bias: bool = True
  beit_like_qkv_bias: bool = False
  mlp_activation: str = 'quick_gelu'
  drop_path_rate: float = 0.1
  use_abs_pos: bool = True
  pretrain_img_size: int = 224
  pretrain_use_cls_token: bool = True
  layer_scale_init_value: float = -1.0
  kernel_init: str = 'normal'
  freeze_vit_layer: int = -1
  use_ln_pre: bool = False
  use_ln_post: bool = False
  pe_bias: bool = True
  use_class_embedding: bool = True
  dtype: jnp.dtype = jnp.float32
  token_mask_probability: float = -1.0
  token_mask_test: bool = False
  window_block_indexes: Any = None
  use_rel_pos: Any = None

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
      abs_pos_no_cls = abs_pos[:, 1:]
    else:
      abs_pos_no_cls = abs_pos
    xy_num = abs_pos_no_cls.shape[1]
    size = int(xy_num ** 0.5)
    assert size * size == xy_num
    abs_pos_no_cls = abs_pos_no_cls.reshape(
        abs_pos_no_cls.shape[0], size, size, -1)
    if size != h or size != w:
      abs_pos_no_cls = jax.image.resize(
          abs_pos_no_cls,
          (abs_pos_no_cls.shape[0], h, w, abs_pos_no_cls.shape[3]),
          method='bicubic',
      )
      if self.use_class_embedding:
        abs_pos_no_cls = abs_pos_no_cls.reshape(
            abs_pos_no_cls.shape[0], h * w, -1)
        new_abs_pos = jnp.concatenate([abs_pos[:, :1], abs_pos_no_cls], axis=1)
      else:
        new_abs_pos = abs_pos_no_cls
    else:
      if self.use_class_embedding:
        new_abs_pos = abs_pos
      else:
        new_abs_pos = abs_pos_no_cls
    return new_abs_pos

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = False):
    """Forward ViT backbone.

    Args:
      x: (batch_size, height, width, 3) the input image
      train: bool;
    Returns:
      x: the features after the backbone. (batch_size, seq_length, embed_dim).
    """
    x = nn.Conv(
        self.embed_dim, (self.patch_size, self.patch_size),
        strides=(self.patch_size, self.patch_size),
        padding='VALID',
        use_bias=self.pe_bias,
        name='patch_embed.proj')(x)

    if self.use_class_embedding:
      class_embedding = self.param(
          'class_embedding', nn.initializers.zeros, (1, 1, self.embed_dim))
      class_embedding = jnp.broadcast_to(
          class_embedding, (x.shape[0], 1, self.embed_dim))
      x = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, hw, C)
      x = jnp.concatenate([class_embedding, x], axis=1)

    if self.use_abs_pos:
      num_patches = (self.pretrain_img_size // self.patch_size) ** 2
      num_positions = (
          num_patches + 1) if self.pretrain_use_cls_token else num_patches
      pos_embed = self.param(
          'pos_embed', nn.initializers.zeros,
          (1, num_positions, self.embed_dim))
      if self.use_class_embedding:
        input_size = int((x.shape[1] - 1) ** 0.5)
        x = x + self._get_abs_pos(pos_embed, (input_size, input_size))
      else:
        x = x + self._get_abs_pos(pos_embed, (x.shape[1], x.shape[2]))

    # TODO(zhouxy): The current MAE is not optimal. We sample a single index
    # for all images in the batch. We should use different indexes each image.
    if self.token_mask_probability > 0:
      assert self.use_class_embedding
      num_pixel_tokens = x.shape[1] - 1
      num_remaining_tokens = int(
          (1.0 - self.token_mask_probability) * num_pixel_tokens)
      if train:
        inds = jax.random.permutation(
            self.make_rng('dropout'),
            jnp.arange(num_pixel_tokens, dtype=jnp.int32),
            independent=True,
        )[:num_remaining_tokens]
      else:
        if self.token_mask_test:
          inds = jnp.linspace(
              0, num_pixel_tokens, num_remaining_tokens,
              endpoint=False, dtype=jnp.int32)
        else:
          inds = jnp.arange(num_pixel_tokens, dtype=jnp.int32)
      unmasked_pixel_tokens = jnp.take_along_axis(
          x[:, 1:], inds[None, :, None], axis=1)
      x = jnp.concatenate([x[:, :1], unmasked_pixel_tokens], axis=1)

    dp_rates = [
        self.drop_path_rate * i / (self.depth - 1) for i in range(self.depth)]
    if self.use_ln_pre:
      x = nn.LayerNorm(name='ln_pre')(x)
    for i in range(self.depth):
      x = Block(
          dim=self.embed_dim,
          num_heads=self.num_heads,
          mlp_ratio=self.mlp_ratio,
          qkv_bias=self.qkv_bias,
          beit_like_qkv_bias=self.beit_like_qkv_bias,
          mlp_activation=self.mlp_activation,
          drop_path=dp_rates[i],
          with_grid_tokens=not self.use_class_embedding,
          layer_scale_init_value=self.layer_scale_init_value,
          name=f'blocks.{i}',
          )(x, train=train)
      if i + 1 == self.freeze_vit_layer:
        x = jax.lax.stop_gradient(x)
    if self.use_ln_post:
      x = nn.LayerNorm(name='ln_post')(x)
    return x
