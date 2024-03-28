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

"""Transformer."""

from collections.abc import Callable
from collections.abc import Sequence
from typing import Optional
from typing import Union

import flax.linen as nn
import jax.numpy as jnp
from scenic.projects.lang4video.model.base_encoders import ImageEncoder
from scenic.projects.lang4video.model.base_encoders import TextEncoder

from flaxformer.architectures.t5 import t5_1_1
from flaxformer.components import layer_norm
from flaxformer.components.attention import dense_attention


class Transformer(nn.Module):
  """Transformer."""

  num_heads: int = 12
  head_dim: int = 64
  mlp_dim: int = 2048
  num_layers: int = 1
  dropout_rate: float = 0.0
  activations: Sequence[Union[str, Callable[[jnp.ndarray],
                                            jnp.ndarray]]] = ('gelu', 'linear')
  dtype: jnp.dtype = jnp.bfloat16
  return_all_tokens: bool = False

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,  # Shape: (N, ..., X)
      mask: Optional[jnp.ndarray] = None,
      *,
      train: bool = False,
      debug: bool = False,
  ) -> jnp.ndarray:  # Shape: (N, ..., E)
    if mask is None:
      encoder_mask = logit_mask = None
    else:
      encoder_mask = dense_attention.make_attention_mask(
          mask, mask, dtype=self.dtype)
      logit_mask = mask[..., jnp.newaxis]

    for _ in range(self.num_layers):
      x = t5_1_1.t5_common_layers.encoder_layer(
          num_heads=self.num_heads,
          head_dim=self.head_dim,
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          activations=self.activations,
          dtype=self.dtype)(
              x,
              encoder_mask=encoder_mask,
              logit_mask=logit_mask,
              enable_dropout=train)

    x = layer_norm.T5LayerNorm(dtype=self.dtype)(x)
    x = nn.Dropout(
        rate=self.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=not train)

    if logit_mask is not None:
      x = logit_mask * x

    return x if self.return_all_tokens else x[:, 0]


class TransformerImageEncoder(ImageEncoder):
  """Transformer image encoder."""

  num_heads: int = 12
  head_dim: int = 64
  mlp_dim: int = 2048
  num_layers: int = 1
  dropout_rate: float = 0.0
  activations: Sequence[Union[str, Callable[[jnp.ndarray],
                                            jnp.ndarray]]] = ('gelu', 'linear')
  dtype: jnp.dtype = jnp.bfloat16
  return_all_tokens: bool = False

  @nn.compact
  def __call__(
      self,
      image: jnp.ndarray,
      *,
      train: bool = False,
      debug: bool = False,
  ) -> jnp.ndarray:
    return Transformer(
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        dropout_rate=self.dropout_rate,
        activations=self.activations,
        dtype=self.dtype,
        return_all_tokens=self.return_all_tokens)(
            image, train=train, debug=debug)


class TransformerTextEncoder(TextEncoder):
  """Transformer text encoder."""

  num_heads: int = 12
  head_dim: int = 64
  mlp_dim: int = 2048
  num_layers: int = 1
  dropout_rate: float = 0.0
  activations: Sequence[Union[str, Callable[[jnp.ndarray],
                                            jnp.ndarray]]] = ('gelu', 'linear')
  dtype: jnp.dtype = jnp.bfloat16
  return_all_tokens: bool = False

  @nn.compact
  def __call__(
      self,
      text: jnp.ndarray,
      mask: Optional[jnp.ndarray] = None,
      *,
      train: bool = False,
      debug: bool = False,
  ) -> jnp.ndarray:
    return Transformer(
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        dropout_rate=self.dropout_rate,
        activations=self.activations,
        dtype=self.dtype,
        return_all_tokens=self.return_all_tokens)(
            text, mask, train=train, debug=debug)
