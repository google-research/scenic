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

"""MLP."""

from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from scenic.projects.lang4video.model import activations
from scenic.projects.lang4video.model import normalizations
from scenic.projects.lang4video.model.base_encoders import ImageEncoder
from scenic.projects.lang4video.model.base_encoders import TextEncoder


class Mlp(nn.Module):
  """MLP."""

  num_layers: int = 1
  hidden_size: int = 1024
  normalization: Optional[str] = None
  activation: str = 'relu'
  embedding_size: int = 512
  skip_connection: bool = True
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,  # Shape: (N, ..., X)
      *,
      train: bool = False,
      debug: bool = False,
  ) -> jnp.ndarray:  # Shape: (N, ..., E)
    is_batch_norm = self.normalization == 'batch'

    y = x

    for _ in range(self.num_layers):
      y = nn.Dense(
          self.hidden_size, use_bias=not is_batch_norm, dtype=self.dtype)(
              y)

      if self.normalization and self.normalization != 'none':
        kwargs = {}
        if is_batch_norm:
          kwargs['use_running_average'] = not train

        # We leave the default eps because the BatchNorm and LayerNorm are
        # always computed in FP32.
        y = normalizations.NORMALIZATIONS_BY_NAME[self.normalization](
            dtype=self.dtype)(y, **kwargs)  # pytype: disable=wrong-keyword-args

      y = activations.ACTIVATIONS_BY_NAME[self.activation](y)

    y = nn.Dense(self.embedding_size, dtype=self.dtype)(y)

    # TODO(sacastro): dropout?
    # TODO(sacastro): norm at the end?

    if self.skip_connection:
      if x.shape[-1] != y.shape[-1]:
        x = nn.Dense(self.embedding_size, use_bias=False, dtype=self.dtype)(x)

      return y + x
    else:
      return y


class MlpImageEncoder(ImageEncoder):
  """MLP image encoder."""

  num_layers: int = 1
  hidden_size: int = 1024
  normalization: Optional[str] = None
  activation: str = 'relu'
  embedding_size: int = 512
  skip_connection: bool = True

  @nn.compact
  def __call__(
      self,
      image: jnp.ndarray,
      *,
      train: bool = False,
      debug: bool = False,
  ) -> jnp.ndarray:
    return Mlp(
        num_layers=self.num_layers,
        hidden_size=self.hidden_size,
        normalization=self.normalization,
        activation=self.activation,
        embedding_size=self.embedding_size,
        skip_connection=self.skip_connection,
        dtype=self.dtype)(
            image, train=train, debug=debug)


class MlpTextEncoder(TextEncoder):
  """MLP text encoder."""

  num_layers: int = 1
  hidden_size: int = 1024
  normalization: Optional[str] = None
  activation: str = 'relu'
  embedding_size: int = 512
  skip_connection: bool = True

  @nn.compact
  def __call__(
      self,
      text: jnp.ndarray,
      mask: Optional[jnp.ndarray] = None,
      *,
      train: bool = False,
      debug: bool = False,
  ) -> jnp.ndarray:
    del mask  # We ignore the mask because we don't use MLP with sequences.
    return Mlp(
        num_layers=self.num_layers,
        hidden_size=self.hidden_size,
        normalization=self.normalization,
        activation=self.activation,
        embedding_size=self.embedding_size,
        skip_connection=self.skip_connection,
        dtype=self.dtype)(
            text, train=train, debug=debug)
