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

"""Contains modules for junction images."""

from typing import Any, Optional, Callable

import flax.linen as nn
import jax.numpy as jnp
import ml_collections


class Hidden2OutputsBlock(nn.Module):
  """Finds junction images given hidden state."""

  num_wedges: int
  parameterization: str
  params2maps: Callable[..., Any]
  return_jparams: Optional[bool] = False

  @nn.compact
  def __call__(self,
               hidden_state: jnp.ndarray,
               max_patchsize: Any,
               input_image: jnp.ndarray,
               global_features: jnp.ndarray,
               train_opts: ml_collections.ConfigDict,
               train: bool = True):

    if self.parameterization == 'standard':
      num_classes = self.num_wedges + 4
    else:
      raise NotImplementedError('No valid parameterization found')

    jparams_un = nn.Dense(num_classes,
                          bias_init=nn.initializers.uniform(),
                          name='head')(hidden_state)  # [N, H, W, C_out]

    jparams = NormalizeOutputs(self.num_wedges,
                               self.parameterization)(jparams_un)

    if self.return_jparams:
      return jparams
    else:
      all_maps = self.params2maps(
          jparams, max_patchsize, input_image, global_features, train_opts
      )
      all_maps['hidden_state'] = hidden_state
      return all_maps


class NormalizeOutputs(nn.Module):
  """Normalize junction parameters.

  Normalized so that sqrt(cos(omega)**2 + sin(omega)**2) = 1
  """

  num_wedges: int
  parameterization: str

  @nn.compact
  def __call__(self, x) -> jnp.ndarray:

    if self.parameterization == 'standard':
      # First map the orientation sin/cosine to be between -1 and 1
      xtanh = jnp.tanh(x[..., :2])

      # Then normalize  so that cos(theta)**2 + sin(theta)**2 = 1
      alpha = xtanh/jnp.linalg.norm(xtanh, axis=-1, keepdims=True)

      # Next, map the three angles to be between 0 and 1,
      # and then scale to sum to 2*pi
      omega = nn.sigmoid(x[..., 2:self.num_wedges+2])
      omega = omega/jnp.sum(omega, axis=-1, keepdims=True) * 2*jnp.pi

      out = jnp.concatenate((alpha, omega, x[..., self.num_wedges+2:]), axis=-1)

    else:
      raise NotImplementedError('No valid parameterization found')

    return out


class ResidualBlock(nn.Module):
  """Adds a residual and returns next hidden state."""

  hidden_dim: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, hidden_state, prev_state, train=True):

    hidden_state = hidden_state + nn.Dense(self.hidden_dim,
                                           name='Residual')(prev_state)
    hidden_state = nn.LayerNorm()(hidden_state)

    return hidden_state


class GRU(nn.Module):
  """GRU cell as nn.Module."""

  @nn.compact
  def __call__(self, carry: jnp.ndarray, inputs: jnp.ndarray,
               train: bool = False) -> jnp.ndarray:
    del train  # Unused.
    carry, _ = nn.GRUCell(features=carry.shape[-1])(carry, inputs)
    # carry, _ = nn.GRUCell()(carry, inputs)
    return carry


class Identity(nn.Module):
  """Module that applies the identity function, ignoring any additional args."""

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, **args) -> jnp.ndarray:
    return inputs


class MLP(nn.Module):
  """Simple MLP with one hidden layer and optional pre-/post-layernorm."""

  hidden_size: int
  output_size: Optional[int] = None
  num_hidden_layers: int = 1
  activation_fn: str = 'relu'
  output_activation_fn: str = 'relu'
  layernorm: Optional[str] = None
  activate_output: bool = False
  residual: bool = False
  use_bias: bool = True
  kernel_init: Optional[str] = None
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool = False) -> jnp.ndarray:
    output_size = self.output_size or inputs.shape[-1]

    x = inputs

    if self.layernorm == 'pre':
      x = nn.LayerNorm()(x)

    activation_fn = getattr(nn, self.activation_fn)
    output_activation_fn = getattr(nn, self.output_activation_fn)
    kernel_init = (
        getattr(nn.initializers, self.kernel_init)()
        if self.kernel_init
        else nn.linear.default_kernel_init
    )
    for _ in range(self.num_hidden_layers):
      x = nn.Dense(
          self.hidden_size,
          use_bias=self.use_bias,
          kernel_init=kernel_init,
      )(x)
      x = activation_fn(x)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
    x = nn.Dense(
        output_size,
        use_bias=self.use_bias,
        kernel_init=kernel_init,
    )(x)

    if self.activate_output:
      x = output_activation_fn(x)

    if self.residual:
      x = x + inputs

    if self.layernorm == 'post':
      x = nn.LayerNorm()(x)

    return x
