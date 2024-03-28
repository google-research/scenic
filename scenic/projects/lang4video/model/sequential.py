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

"""Sequential encoder."""

from collections.abc import Sequence
from typing import Optional

import flax
from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
import jax.numpy as jnp
from scenic.projects.lang4video.model.base_encoders import ImageEncoder
from scenic.projects.lang4video.model.base_encoders import TextEncoder


class SequentialImageEncoder(ImageEncoder):
  """Like `nn.Sequential` but for `ImageEncoder`.

  One difference with `nn.Sequential` is that `train` and `debug` args are
  passed to each encoder.
  """

  encoders: Sequence[ImageEncoder] = ()

  @nn.compact
  def __call__(
      self,
      image: jnp.ndarray,
      *,
      train: bool = False,
      debug: bool = False,
  ) -> jnp.ndarray:
    if not self.encoders:
      raise ValueError(f'Empty SequentialEncoder module {self.name}.')

    outputs = image
    for encoder in self.encoders:
      outputs = encoder(outputs, train=train, debug=debug)
    return outputs

  def get_pretrained_vars(self) -> tuple[FrozenDict, FrozenDict]:
    params, model_state = {}, {}
    for i, encoder in enumerate(self.encoders):
      params[f'encoders_{i}'], model_state[
          f'encoders_{i}'] = encoder.get_pretrained_vars()
    return flax.core.freeze(params), flax.core.freeze(model_state)


class SequentialTextEncoder(TextEncoder):
  """Like `nn.Sequential` but for `TextEncoder`.

  One difference with `nn.Sequential` is that `train` and `debug` args are
  passed to each encoder.
  """

  encoders: Sequence[TextEncoder] = ()

  @nn.compact
  def __call__(
      self,
      text: jnp.ndarray,  # Shape: (N, L)
      mask: Optional[jnp.ndarray] = None,  # Shape: (N, L)
      *,
      train: bool = False,
      debug: bool = False,
  ) -> jnp.ndarray:
    if not self.encoders:
      raise ValueError(f'Empty SequentialEncoder module {self.name}.')

    outputs = text
    for encoder in self.encoders:
      outputs = encoder(outputs, mask=mask, train=train, debug=debug)
    return outputs

  def get_pretrained_vars(self) -> tuple[FrozenDict, FrozenDict]:
    params, model_state = {}, {}
    for i, encoder in enumerate(self.encoders):
      params[f'encoders_{i}'], model_state[
          f'encoders_{i}'] = encoder.get_pretrained_vars()
    return flax.core.freeze(params), flax.core.freeze(model_state)
