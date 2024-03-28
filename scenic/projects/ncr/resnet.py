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

"""Implementation of ResNet."""

import functools
from typing import Callable, Any, Union, Dict

from absl import logging
import flax
import flax.linen as nn
from jax.nn import initializers
import jax.numpy as jnp
import ml_collections
from scenic.common_lib import debug_utils
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import resnet
from scenic.projects.ncr.base_model import NCRModel


class ResNet(nn.Module):
  """ResNet architecture.

  Attributes:
    num_outputs: Num output classes.
    num_filters: Num filters.
    num_layers: Num layers.
    kernel_init: Kernel initialization.
    bias_init: Bias initialization.
    dtype: Data type, e.g. jnp.float32.
  """
  num_outputs: int
  num_filters: int = 64
  num_layers: int = 50
  kernel_init: Callable[..., Any] = initializers.lecun_normal()
  bias_init: Callable[..., Any] = initializers.zeros
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      train: bool = False,
      debug: bool = False) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Applies ResNet model to the inputs.

    Args:
      x: Inputs to the model.
      train: Whether it is training or not.
      debug: Whether the debug mode is enabled. debug=True enables model
        specific logging/storing some values using jax.host_callback.

    Returns:
       Un-normalized logits.
    """
    if self.num_layers not in BLOCK_SIZE_OPTIONS:
      raise ValueError('Please provide a valid number of layers')
    block_sizes, bottleneck = BLOCK_SIZE_OPTIONS[self.num_layers]
    x = nn.Conv(
        self.num_filters,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding=[(3, 3), (3, 3)],
        use_bias=False,
        dtype=self.dtype,
        name='stem_conv')(
            x)
    x = nn.BatchNorm(
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
        name='init_bn')(
            x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)])

    residual_block = functools.partial(
        resnet.ResidualBlock, dtype=self.dtype, bottleneck=bottleneck)
    representations = {'stem': x}
    for i, block_size in enumerate(block_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        filters = self.num_filters * 2**i
        x = residual_block(filters=filters, strides=strides)(x, train)
      representations[f'stage_{i + 1}'] = x

    # Head.
    x = jnp.mean(x, axis=(1, 2))
    x = nn_layers.IdentityLayer(name='pre_logits')(x)
    representations['pre_logits'] = x
    x = nn.Dense(
        self.num_outputs,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dtype=self.dtype,
        name='output_projection')(
            x)
    representations['logits'] = x

    return representations


# A dictionary mapping the number of layers in a resnet to the number of
# blocks in each stage of the model. The second argument indicates whether we
# use bottleneck layers or not.
BLOCK_SIZE_OPTIONS = {
    5: ([1], True),  # Only strided blocks. Total stride 4.
    8: ([1, 1], True),  # Only strided blocks. Total stride 8.
    11: ([1, 1, 1], True),  # Only strided blocks. Total stride 16.
    14: ([1, 1, 1, 1], True),  # Only strided blocks. Total stride 32.
    9: ([1, 1, 1, 1], False),  # Only strided blocks. Total stride 32.
    18: ([2, 2, 2, 2], False),
    26: ([2, 2, 2, 2], True),
    34: ([3, 4, 6, 3], False),
    50: ([3, 4, 6, 3], True),
    101: ([3, 4, 23, 3], True),
    152: ([3, 8, 36, 3], True),
    200: ([3, 24, 36, 3], True)
}


class ResNetNCRModel(NCRModel):
  """Implements the NCR ResNet model for classification."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return ResNet(
        num_outputs=self.dataset_meta_data['num_classes'],
        num_filters=self.config.num_filters,
        num_layers=self.config.num_layers,
        dtype=model_dtype)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return _get_default_configs_for_testing()

  def init_from_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
    """Updates the train_state with data from `restored_train_state`.

    This function is writen to be used for 'finetuning' experiments.
    Args:
      train_state: A raw TrainState for the model.
      restored_train_state: A TrainState that is loaded with parameters/state of
        a pretrained model.
      restored_model_cfg: Configuration of the model from which the
        `restored_train_state` come from. Usually used for some asserts.

    Returns:
      Updated train_state.
    """
    del restored_model_cfg
    params = flax.core.unfreeze(train_state.optimizer.target)
    restored_params = flax.core.unfreeze(restored_train_state.optimizer.target)
    for pname, pvalue in restored_params.items():
      if pname == 'output_projection':
        # The `output_projection` is used as the name of the linear layer at the
        # head of the model that maps the representation to the label space.
        # By default, for finetuning to another dataset, we drop this layer as
        # the label space is different.
        continue
      else:
        params[pname] = pvalue
    logging.info('Parameter summary after initialising from train state:')
    debug_utils.log_param_shapes(params)
    return train_state.replace(
        optimizer=train_state.optimizer.replace(
            target=flax.core.freeze(params)),
        model_state=restored_train_state.model_state)


def _get_default_configs_for_testing() -> ml_collections.ConfigDict:
  return ml_collections.ConfigDict(
      dict(
          num_filters=16,
          num_layers=5,
          data_dtype_str='float32',
      ))
