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

"""Contains optimizer utils."""

import copy
from typing import Any

from absl import logging
import flax
import ml_collections
import optax
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers as optimizer_lib

PyTree = Any


def optimizer_with_multi_lrs(config: ml_collections.ConfigDict,
                             params: PyTree) -> optax.GradientTransformation:
  """Builds a optimizer with different learning rates on different layers.

  Users can specify different base learning rates for different params with
  prefixes defined in the config. For example, users can specify different LRs
  for temporal encoder and spatial encoder using the following config:

  config.layer_prefix_to_base_lrs = {
      'video_encoder/VisionTransformer': 0.005,
      'video_encoder/TemporalTransformer': 0.05,
  }

  Args:
    config: Model Configuration.
    params: A nested dict containing model parameters.

  Returns:
    An optax.GradientTransformation object.
  """

  optimizer_config = optimizer_lib.get_optax_optimizer_config(config)
  # Avoid modifying original config and allow alteration.
  optimizer_config = copy.deepcopy(optimizer_config).unlock()

  layer_prefix_to_base_lrs = copy.deepcopy(
      config.layer_prefix_to_base_lrs).unlock()
  # If parameters do not start with defined prefixes, they use
  # `base_learning_rate` defined in config.
  layer_prefix_to_base_lrs.update(
      {'none_of_above': config.lr_configs.base_learning_rate})
  optimizers = {}
  for prefix, base_lr in layer_prefix_to_base_lrs.items():
    layer_config = copy.deepcopy(config)
    layer_config.lr_configs.base_learning_rate = base_lr
    lr_fn = lr_schedules.get_learning_rate_fn(layer_config)
    optimizers[prefix] = optimizer_lib.get_optimizer(optimizer_config, lr_fn,
                                                     params)

  flat_params = flax.traverse_util.flatten_dict(
      flax.core.unfreeze(params), keep_empty_nodes=True, sep='/')
  flat_layer_map = {}
  for key in flat_params:
    assigned = False
    for prefix in layer_prefix_to_base_lrs:
      if key.startswith(prefix):
        flat_layer_map[key] = prefix
        assigned = True
        break
    if not assigned:
      flat_layer_map[key] = 'none_of_above'

  layer_map = flax.traverse_util.unflatten_dict(flat_layer_map, sep='/')
  layer_map = flax.core.freeze(layer_map)

  logging.info('Layer assignments:\n%s',
               flax.traverse_util.flatten_dict(layer_map, sep='/'))
  tx = optax.multi_transform(optimizers, layer_map)
  return tx
