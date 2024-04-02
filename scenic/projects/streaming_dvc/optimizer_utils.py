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

"""Utilities for optimizers."""

import copy
from typing import Any, Callable, Optional, Union

from absl import logging
import flax
import ml_collections
import optax
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers as optimizer_lib

ScalarOrSchedule = Union[float, optax.Schedule]
MaskOrFn = Optional[Union[Any, Callable[[optax.Params], Any]]]
PyTree = Any  # JAX team is working on type annotation for pytree:


def optimizer_with_layerwise_decay(
    config: ml_collections.ConfigDict,
    params: PyTree,
    use_frozen_params: bool = True):
  """Returns an optimizer with layerwise decay.

  Implementation of layerwise decay follows BEIT and MAE.
  Reference: https://github.com/facebookresearch/mae/blob/main/util/lr_decay.py

  This function can apply layerwise decay to any optimizer, although this is
  typically done with Adam.

  Args:
    config: The training config.
    params: The parameters of the model being trained.
    use_frozen_params: If True, the optimizer will always expect to receive
      a FrozenDict of parameters and gradients.

  Returns:
    An Optax optimizer.
  """
  optimizer_config = config.optimizer
  # Avoid modifying original config and allow alteration.
  optimizer_config = copy.deepcopy(optimizer_config).unlock()
  base_learning_rate = config.lr_configs.base_learning_rate

  decay_layer_prefix = optimizer_config.decay_layer_prefix
  decay_stem_layers = optimizer_config.decay_stem_layers
  num_transformer_layers = config.optimizer.num_layers
  del optimizer_config.decay_layer_prefix
  del optimizer_config.decay_stem_layers
  del optimizer_config.num_layers

  if optimizer_config.get('layerwise_decay', 0) <= 0:
    logging.info('Not performing any layerwise decay.')
    if 'layerwise_decay' in optimizer_config:
      del optimizer_config.layerwise_decay
    lr_fn = lr_schedules.get_learning_rate_fn(config)
    return optimizer_lib.get_optimizer(optimizer_config, lr_fn, params)

  num_layers = num_transformer_layers + 1
  layer_decay = optimizer_config.layerwise_decay
  learning_rate_scales = [
      layer_decay**(num_layers - i) for i in range(num_layers + 1)
  ]
  logging.info('Learning rate scales: %s', learning_rate_scales)

  layer_configs = [copy.deepcopy(config) for _ in range(num_layers + 1)]
  for index in range(len(layer_configs)):
    learning_rate = base_learning_rate * learning_rate_scales[index]
    layer_configs[index].lr_configs.base_learning_rate = learning_rate

  learning_rate_fns = [
      lr_schedules.get_learning_rate_fn(layer_config)
      for layer_config in layer_configs
  ]

  # Weight decay mask is applied within optimizer_lib.get_optimizer.
  # Note that we need to delete the layerwise_decay attribute, as Optax
  # optimizers do not accept this argument.
  del optimizer_config.layerwise_decay
  optimizers = {
      i: optimizer_lib.get_optimizer(
          optimizer_config, learning_rate_fns[i], params)
      for i in range(num_layers + 1)
  }

  def _get_layer_id(name: str, num_layers: int) -> int:
    for k in decay_stem_layers:
      if k in name:
        return 0
    if name.startswith(decay_layer_prefix):
      l = len(decay_layer_prefix)
      substring = name[l: name[l:].find('/') + l]
      layer_id = int(substring)
      return layer_id + 1
    else:
      return num_layers

  flat_params = flax.traverse_util.flatten_dict(
      flax.core.unfreeze(params), keep_empty_nodes=True, sep='/')
  flat_layer_map = {k: _get_layer_id(k, num_layers) for k in flat_params}
  layer_map = flax.traverse_util.unflatten_dict(flat_layer_map, sep='/')
  if use_frozen_params:
    layer_map = flax.core.freeze(layer_map)

  logging.info(
      'Layer assignments:\n%s',
      flax.traverse_util.flatten_dict(layer_map, sep='/'))
  tx = optax.multi_transform(optimizers, layer_map)
  return tx


def optimizer_with_backbone_multiplier(
    config: ml_collections.ConfigDict,
    params: PyTree,
    use_frozen_params: bool = True):
  """Returns an optimizer with backbone learning rate multiplier.


  Args:
    config: The training config.
    params: The parameters of the model being trained.
    use_frozen_params: If True, the optimizer will always expect to receive
      a FrozenDict of parameters and gradients.

  Returns:
    An Optax optimizer.
  """
  optimizer_config = config.optimizer
  # Avoid modifying original config and allow alteration.
  optimizer_config = copy.deepcopy(optimizer_config).unlock()
  base_learning_rate = config.lr_configs.base_learning_rate

  backbone_layer_prefix = optimizer_config.backbone_layer_prefix
  backbone_multiplier = optimizer_config.backbone_multiplier
  backbone_learning_rate = base_learning_rate * backbone_multiplier
  del optimizer_config.backbone_layer_prefix
  del optimizer_config.backbone_multiplier
  logging.info('Learning rate scales: %s', backbone_learning_rate)

  backbone_config = copy.deepcopy(config)
  backbone_config.lr_configs.base_learning_rate = backbone_learning_rate

  learning_rate_fns = lr_schedules.get_learning_rate_fn(config)
  backbone_learning_rate_fns = lr_schedules.get_learning_rate_fn(
      backbone_config)

  optimizers = {
      False: optimizer_lib.get_optimizer(  # not backbone
          optimizer_config, learning_rate_fns, params),
      True: optimizer_lib.get_optimizer(  # is backbone
          optimizer_config, backbone_learning_rate_fns, params),
  }

  def is_backbone(name: str) -> bool:
    return name.startswith(backbone_layer_prefix)

  flat_params = flax.traverse_util.flatten_dict(
      flax.core.unfreeze(params), keep_empty_nodes=True, sep='/')
  flat_layer_map = {k: is_backbone(k) for k in flat_params}
  layer_map = flax.traverse_util.unflatten_dict(flat_layer_map, sep='/')
  if use_frozen_params:
    layer_map = flax.core.freeze(layer_map)

  logging.info(
      'Layer assignments:\n%s',
      flax.traverse_util.flatten_dict(layer_map, sep='/'))
  tx = optax.multi_transform(optimizers, layer_map)
  return tx
