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

r"""ViT backbone in EVA02.

arXiv: https://arxiv.org/pdf/2303.11331.pdf

Pytorch reference: https://github.com/baaivision/EVA/blob/HEAD/EVA-02/asuka/
modeling_pretrain.py

Weight converted in
https://colab.research.google.com/drive/1xlyHCUavh0OVaTUMmN-jOSRljvy5K4HI

Verified logits error in ~1e-1 with random inputs.
"""

import functools
from typing import Any, Callable, Optional

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp


class Attention(nn.Module):
  """Multi-head Attention block with relative position embeddings.

  Attributes:
  dim (int): Number of input channels.
  num_heads (int): Number of attention heads.
  """
  dim: int
  rope: Callable[[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]
  num_heads: int = 8
  kernel_init: str = 'normal'
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x, inds=None):
    """Forward a block.

    Args:
      x: (batch_size, num_tokens, dim);
      inds: (num_mae_tokens,)

    Returns:
      x: the same shape as the input.
    """

    batch, num_tokens, _ = x.shape
    head_dim = self.dim // self.num_heads

    q = nn.Dense(self.dim, use_bias=False, name='q_proj')(x)
    k = nn.Dense(self.dim, use_bias=False, name='k_proj')(x)
    v = nn.Dense(self.dim, use_bias=False, name='v_proj')(x)

    q = q + self.param('q_bias', nn.initializers.zeros, (self.dim,))[None, None]
    v = v + self.param('v_bias', nn.initializers.zeros, (self.dim,))[None, None]

    q = q.reshape(batch, num_tokens, self.num_heads, -1).transpose(
        0, 2, 1, 3).reshape(batch * self.num_heads, num_tokens, -1)
    k = k.reshape(batch, num_tokens, self.num_heads, -1).transpose(
        0, 2, 1, 3).reshape(batch * self.num_heads, num_tokens, -1)
    v = v.reshape(batch, num_tokens, self.num_heads, -1).transpose(
        0, 2, 1, 3).reshape(batch * self.num_heads, num_tokens, -1)

    q_t = q[:, 1:, :]
    ro_q_t = self.rope(q_t, inds)
    q = jnp.concatenate([q[:, :1, :], ro_q_t], axis=-2)

    k_t = k[:, 1:, :]
    ro_k_t = self.rope(k_t, inds)
    k = jnp.concatenate([k[:, :1, :], ro_k_t], axis=-2)

    attn = (q * (head_dim ** -0.5)) @ k.transpose(
        0, 2, 1)  # [batch * num_heads, num_tokens, num_tokens]

    attn = jax.nn.softmax(attn)
    x = (attn @ v).reshape(
        batch, self.num_heads, num_tokens, -1).transpose(
            0, 2, 1, 3).reshape(batch, num_tokens, -1)

    x = nn.Dense(self.dim, name='proj')(x)
    return x


class SwiGLU(nn.Module):
  """SwiGLU layer."""

  hidden_features: int

  @nn.compact
  def __call__(self, x):
    in_features = x.shape[-1]
    x1 = nn.Dense(self.hidden_features, name='w1')(x)
    x2 = nn.Dense(self.hidden_features, name='w2')(x)
    hidden = jax.nn.silu(x1) * x2
    x = nn.LayerNorm(epsilon=1e-5, name='ffn_ln')(hidden)
    x = nn.Dense(in_features, name='w3')(x)
    return x


class Block(nn.Module):
  """Transformer blocks with support of window attention and residual blocks.

  Attributes:
    dim (int): Number of input channels.
    num_heads (int): Number of attention heads in each ViT block.
    mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    drop_path (float): Stochastic depth rate.
  """
  dim: int
  num_heads: int
  rope: Callable[[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]
  mlp_ratio: float = 4.0
  drop_path: float = 0.0
  kernel_init: str = 'normal'
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
  def __call__(self, x, inds=None, train: bool = False):
    shortcut = x
    ln = functools.partial(nn.LayerNorm, epsilon=1e-6)
    x = ln(name='norm1')(x)

    x = Attention(
        self.dim,
        num_heads=self.num_heads,
        rope=self.rope,
        name='attn')(x, inds=inds)

    x = shortcut + self.get_keep_pattern(x, not train) * x

    y = ln(name='norm2')(x)

    y = SwiGLU(
        hidden_features=int(self.dim * self.mlp_ratio),
        name='mlp')(y)
    x = x + self.get_keep_pattern(y, not train) * y
    return x


def _broadcast_and_concatenate(x, y):
  """Broadcast and concatenate two tensors (_broadcat in original).

  Args:
    x: (b1, 1, d)
    y: (1, b2, d)
  Returns:
    ret: (b1, b2, d * 2)
  """
  b1, b2, d = x.shape[0], y.shape[1], x.shape[2]
  x = jnp.broadcast_to(x, (b1, b2, d))
  y = jnp.broadcast_to(y, (b1, b2, d))
  return jnp.concatenate([x, y], axis=2)


def vision_rotary_embedding_fast(
    x, inds, dim, pt_seq_len=16, ft_seq_len=None, theta=10000):
  """Rotary positional embedding used in EVA02."""
  freqs = 1. / (theta ** (jnp.arange(0, dim, 2)[
      :(dim // 2)].astype(jnp.float32) / dim))
  ft_seq_len = ft_seq_len if (
      ft_seq_len is not None) else pt_seq_len
  t = jnp.arange(
      ft_seq_len).astype(jnp.float32) / ft_seq_len * pt_seq_len

  freqs = jnp.einsum('..., f -> ... f', t, freqs)
  freqs = einops.repeat(freqs, '... n -> ... (n r)', r=2)
  freqs = _broadcast_and_concatenate(freqs[:, None, :], freqs[None, :, :])
  freqs = freqs.reshape(-1, freqs.shape[-1])
  freqs_cos = jnp.cos(freqs)
  freqs_sin = jnp.sin(freqs)
  def _rotate_half(x):
    x = einops.rearrange(x, '... (d r) -> ... d r', r=2)
    x = jnp.stack([-x[..., 1], x[..., 0]], axis=-1)
    return einops.rearrange(x, '... d r -> ... (d r)')
  if inds is not None:
    freqs_cos = jnp.take_along_axis(freqs_cos, inds[:, None], axis=0)
    freqs_sin = jnp.take_along_axis(freqs_sin, inds[:, None], axis=0)
  return x * freqs_cos[None] + _rotate_half(x) * freqs_sin[None]


class ViT(nn.Module):
  """ViT backbone used in EVA02 paper.

  Main differences with respect to the original ViT include:

  - Use sub-LN instead of pre-LN
  - Use SwiGLU as FFN instead of MLP
  - Use 2D Rotary positional embedding instead of abs PE.
  - Use a different initialization
  """
  patch_size: int = 16
  in_chans: int = 3
  embed_dim: int = 768
  depth: int = 12
  num_heads: int = 12
  mlp_ratio: float = 4.0
  drop_path_rate: float = 0.1
  use_abs_pos: bool = True
  pretrain_img_size: int = 224
  pretrain_use_cls_token: bool = True
  kernel_init: str = 'normal'
  freeze_vit_layer: int = -1
  use_ln_pre: bool = False
  use_ln_post: bool = False
  pe_bias: bool = True
  dtype: jnp.dtype = jnp.float32
  token_mask_probability: float = -1.0
  window_block_indexes: Any = None
  use_rel_pos: Any = None
  stop_grad_conv1: bool = False

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
      abs_pos_no_cls = abs_pos_no_cls.reshape(
          abs_pos_no_cls.shape[0], h * w, -1)
      new_abs_pos = jnp.concatenate([abs_pos[:, :1], abs_pos_no_cls], axis=1)
    else:
      new_abs_pos = abs_pos
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
    image_size = x.shape[1]
    x = nn.Conv(
        self.embed_dim, (self.patch_size, self.patch_size),
        strides=(self.patch_size, self.patch_size),
        padding='VALID',
        use_bias=self.pe_bias,
        name='patch_embed.proj')(x)
    x = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, hw, C)

    if self.stop_grad_conv1:
      x = jax.lax.stop_gradient(x)

    cls_token = self.param(
        'cls_token', nn.initializers.zeros, (1, 1, self.embed_dim))
    cls_token = jnp.broadcast_to(
        cls_token, (x.shape[0], 1, self.embed_dim))
    x = jnp.concatenate([cls_token, x], axis=1)

    if self.use_abs_pos:
      num_patches = (self.pretrain_img_size // self.patch_size) ** 2
      num_positions = (
          num_patches + 1) if self.pretrain_use_cls_token else num_patches
      pos_embed = self.param(
          'pos_embed', nn.initializers.zeros,
          (1, num_positions, self.embed_dim))
      input_size = int((x.shape[1] - 1) ** 0.5)
      x = x + self._get_abs_pos(pos_embed, (input_size, input_size))

    rope = functools.partial(
        vision_rotary_embedding_fast,
        dim=self.embed_dim // self.num_heads // 2,
        pt_seq_len=image_size // self.patch_size)

    inds = None
    # TODO(zhouxy): The current MAE is not optimal. We sample a single index
    # for all images in the batch. We should use different indexes each image.
    # TODO(zhouxy): move this to a model_utils.py file and reuse in other files.
    if self.token_mask_probability > 0:
      num_pixel_tokens = x.shape[1] - 1
      if train:
        num_remaining_tokens = int(
            (1.0 - self.token_mask_probability) * num_pixel_tokens)
        inds = jax.random.permutation(
            self.make_rng('dropout'),
            jnp.arange(num_pixel_tokens, dtype=jnp.int32),
            independent=True,
        )[:num_remaining_tokens]
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
          drop_path=dp_rates[i],
          rope=rope,
          name=f'blocks.{i}',
          )(x, inds, train=train)
      if i + 1 == self.freeze_vit_layer:
        x = jax.lax.stop_gradient(x)
    if self.use_ln_post:
      x = nn.LayerNorm(name='ln_post')(x)
    return x
