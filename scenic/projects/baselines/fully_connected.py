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

"""Simple fully connected feedforward neural network classifier."""

from typing import Callable, Iterable, Union, Sequence

import flax.linen as nn
from jax.nn import initializers
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models.classification_model import ClassificationModel
from scenic.model_lib.layers import nn_layers


# TODO(mrit): Upstream this to jax.nn.initializers
#   Inputs are PRNGKey, input shape and dtype.
Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


class FullyConnected(nn.Module):
  """Defines a fully connected neural network.

  The model assumes the input data has shape
  [batch_size_per_device, *input_shape] where input_shape may be of arbitrary
  rank. The model flatten the input before applying a dense layer.

  Attributes:
    num_outputs: Number of output classes.
    hid_sizes: Size of hidden units in each layer.
    kernel_init: Kernel initialization.
    bias_init: Bias initialization.
    dtype: Model dtype.
  """
  num_outputs: int
  hid_sizes: Union[Iterable[int], int]
  kernel_init: Initializer = initializers.lecun_normal()
  bias_init: Initializer = initializers.zeros
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, debug: bool = False
               ) -> jnp.ndarray:
    """Applies fully connected model on the input.

    Args:
      x: Input tensor.
      train: bool; Whether the model is running at train time.
      debug: bool; Whether the debug mode is enabled. debug=True enables model
        specific logging/storing some values using jax.host_callback.

    Returns:
      Unnormalized logits.
    """
    del train, debug
    hid_sizes = self.hid_sizes
    if isinstance(hid_sizes, int):
      hid_sizes = [hid_sizes]
    x = jnp.reshape(x, (x.shape[0], -1))
    for num_hid in hid_sizes:
      x = nn.Dense(
          num_hid, kernel_init=self.kernel_init, bias_init=self.bias_init)(
              x)
      x = nn.relu(x)

    # head
    x = nn_layers.IdentityLayer(name='pre_logits')(x)
    x = nn.Dense(
        self.num_outputs,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name='output_projection')(
            x)
    return x


class FullyConnectedClassificationModel(ClassificationModel):
  """Implemets a fully connected model for classification."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return FullyConnected(
        num_outputs=self.dataset_meta_data['num_classes'],
        hid_sizes=self.config.hid_sizes,
        dtype=model_dtype)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(
        dict(hid_sizes=[20, 10], data_dtype_str='float32'))
