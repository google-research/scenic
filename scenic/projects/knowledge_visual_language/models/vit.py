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

"""A refactored and simplified ViT.

However, the names of modules are made to match the old ones for easy loading.
"""

from typing import Optional, Sequence, Union

from big_vision.models import vit as bv_vit
import flax.linen as nn
import jax
import jax.numpy as jnp


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  dropout: float = 0.0
  dtype: str = jnp.bfloat16

  @nn.compact
  def __call__(self, x, deterministic=True):
    """Applies Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )

    n, l, d = x.shape  # pylint: disable=unused-variable
    x = jnp.array(x, dtype=self.dtype)
    x = nn.Dense(self.mlp_dim or 4 * d, **inits, dtype=self.dtype)(x)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic)
    x = nn.Dense(d, **inits, dtype=self.dtype)(x)
    return x


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  dtype: str = jnp.bfloat16

  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}
    y = nn.LayerNorm()(x)
    y = out["sa"] = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=deterministic,
        dtype=self.dtype)(y, y)
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+sa"] = x + y

    y = nn.LayerNorm()(x)
    y = out["mlp"] = MlpBlock(
        mlp_dim=self.mlp_dim, dropout=self.dropout,
        dtype=self.dtype)(y, deterministic)
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+mlp"] = x + y
    return x, out


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  depth: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  dtype: str = jnp.bfloat16
  num_frozen_layers: int = -1

  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}

    # Input Encoder
    for lyr in range(self.depth):
      block = Encoder1DBlock(
          name=f"encoderblock_{lyr}",
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout=self.dropout,
          dtype=self.dtype)
      x, out[f"block{lyr:02d}"] = block(x, deterministic)
      if self.num_frozen_layers > 0 and lyr == self.num_frozen_layers - 1:
        x = jax.lax.stop_gradient(x)
    out["pre_ln"] = x  # Alias for last block, but without the number in it.

    return nn.LayerNorm(name="encoder_norm")(x), out


class _Model(nn.Module):
  """ViT model."""

  num_classes: int
  patch_size: Sequence[int] = (16, 16)
  width: int = 768
  depth: int = 12
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  posemb: str = "learn"  # Can also be "sincos2d"
  rep_size: Union[int, bool] = False
  dropout: float = 0.0
  pool_type: str = "gap"  # Can also be "map" or "tok"
  head_zeroinit: bool = True
  dtype: str = jnp.bfloat16
  num_frozen_layers: int = -1

  @nn.compact
  def __call__(self, image, *, train=False):
    out = {}

    # Patch extraction
    x = out["stem"] = nn.Conv(
        self.width,
        self.patch_size,
        strides=self.patch_size,
        padding="VALID",
        name="embedding")(
            image)
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # Add posemb before adding extra token.
    x = out["with_posemb"] = x + bv_vit.get_posemb(self, self.posemb, (h, w), c,
                                                   "pos_embedding", x.dtype)

    if self.pool_type == "tok":
      cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
      x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

    n, l, c = x.shape  # pylint: disable=unused-variable
    x = nn.Dropout(rate=self.dropout)(x, not train)
    x, out["encoder"] = Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout,
        dtype=self.dtype,
        num_frozen_layers=self.num_frozen_layers,
        name="Transformer")(
            x, deterministic=not train)

    encoded = out["encoded"] = x

    if self.pool_type == "map":
      x = out["head_input"] = bv_vit.MAPHead(
          num_heads=self.num_heads, mlp_dim=self.mlp_dim)(
              x)
    elif self.pool_type == "gap":
      x = out["head_input"] = jnp.mean(x, axis=1)
    elif self.pool_type == "0":
      x = out["head_input"] = x[:, 0]
    elif self.pool_type == "tok":
      x = out["head_input"] = x[:, 0]
      encoded = encoded[:, 1:]
    else:
      raise ValueError(f"Unknown pool type: '{self.pool_type}'")

    x_2d = jnp.reshape(encoded, [n, h, w, -1])

    if self.rep_size:
      rep_size = self.width if self.rep_size is True else self.rep_size  # pylint: disable=g-bool-id-comparison
      hid = nn.Dense(rep_size, name="pre_logits", dtype=self.dtype)
      # NOTE: In the past we did not include tanh in pre_logits.
      # For few-shot, it should not matter much, as it whitens anyways.
      x_2d = nn.tanh(hid(x_2d))
      x = nn.tanh(hid(x))

    out["pre_logits_2d"] = x_2d
    out["pre_logits"] = x

    if self.num_classes:
      kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
      head = nn.Dense(self.num_classes, name="head", dtype=self.dtype, **kw)
      x_2d = out["logits_2d"] = head(x_2d)
      x = out["logits"] = head(x)
    x = jnp.array(x, dtype=self.dtype)
    return x, out


def Model(num_classes,
          num_frozen_layers=-1,
          dtype=jnp.bfloat16,
          variant=None,
          **kw):  # pylint: disable=invalid-name
  """Factory function, because linen really don't like what I'm doing!"""
  vit_config = bv_vit.decode_variant(variant)
  if isinstance(num_frozen_layers, float):
    num_frozen_layers = int(num_frozen_layers * vit_config["depth"])
  return _Model(
      num_classes=num_classes,
      num_frozen_layers=num_frozen_layers,
      dtype=dtype,
      **{
          **vit_config,
          **kw
      })
