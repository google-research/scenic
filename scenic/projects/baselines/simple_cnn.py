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

"""Simple convolutional neural network classifier."""

from typing import Iterable, Callable, Sequence, Optional, Union

import flax.linen as nn
from jax.nn import initializers
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models.classification_model import ClassificationModel
from scenic.model_lib.base_models.segmentation_model import SegmentationModel
from scenic.model_lib.layers import nn_layers


Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


class SimpleCNN(nn.Module):
  """Defines a simple convolutional neural network.

  The model assumes the input shape is [batch, H, W, C].

  Attributes:
    num_outputs: Number of output classes.
    output_projection_type: Type of the output projection layer.
    num_filters: Number of filters in each layer.
    kernel_sizes: Size of kernel in each layer.
    use_bias: If add bias in each layer.
    kernel_init: Kernel initialization.
    bias_init: Bias initialization.
    dtype: Model JAX dtype.
  """

  num_outputs: int
  output_projection_type: str
  num_filters: Sequence[int]
  kernel_sizes: Sequence[int]
  use_bias: Optional[Union[bool, Sequence[bool]]]
  kernel_init: Initializer = initializers.lecun_normal()
  bias_init: Initializer = initializers.zeros
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               *,
               train: bool,
               debug: bool = False) -> jnp.ndarray:
    """Applies SimpleCNN on the input data.

    Args:
      x: Input tensor.
      train: Unused.
      debug: Unused.

    Returns:
      Unnormalized logits.
    """
    del train, debug
    use_bias = True if self.use_bias is None else self.use_bias
    if not isinstance(use_bias, Iterable):
      use_bias = [use_bias] * len(self.num_filters)
    for n_filters, kernel_size, use_bias in zip(self.num_filters,
                                                self.kernel_sizes,
                                                use_bias):
      x = nn.Conv(
          features=n_filters,
          kernel_size=(kernel_size, kernel_size),
          strides=(1, 1),
          use_bias=use_bias,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          dtype=self.dtype)(
              x)
      x = nn.relu(x)

    # Head
    if self.output_projection_type == 'reduce_mean':
      x = jnp.mean(x, axis=(1, 2))

    x = nn_layers.IdentityLayer(name='pre_logits')(x)
    x = nn.Dense(
        self.num_outputs,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dtype=self.dtype,
        name='output_projection')(
            x)
    return x


class SimpleCNNClassificationModel(ClassificationModel):
  """Simple CNN model for classifcation task."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return SimpleCNN(
        num_outputs=self.dataset_meta_data['num_classes'],
        num_filters=self.config.num_filters,
        kernel_sizes=self.config.kernel_sizes,
        use_bias=self.config.get('use_bias', None),
        output_projection_type='reduce_mean',
        dtype=model_dtype)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(
        dict(
            num_filters=[20, 10],
            kernel_sizes=[3, 3],
            data_dtype_str='float32',
        ))


class SimpleCNNSegmentationModel(SegmentationModel):
  """Simple CNN model for segmentation task."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return SimpleCNN(
        num_outputs=self.dataset_meta_data['num_classes'],
        num_filters=self.config.num_filters,
        kernel_sizes=self.config.kernel_sizes,
        use_bias=self.config.get('use_bias', None),
        output_projection_type='keep_dims',
        dtype=model_dtype)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(
        dict(
            num_filters=[20, 10],
            kernel_sizes=[3, 3],
            data_dtype_str='float32',
        ))
