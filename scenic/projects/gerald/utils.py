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

"""Utilities for GER training."""

import copy
from typing import Any

from absl import logging
import flax
from flax import jax_utils
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from scenic.dataset_lib import dataset_utils
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers as optimizer_lib

from tensorflow.io import gfile

PyTree = Any  # JAX team is working on type annotation for pytree:


class EntityIds2Code():
  """Quantization with given token ids at initialization."""

  def __init__(self, config: ml_collections.ConfigDict):
    """Entity id to code."""
    self.config = config
    self.bos = config.get('ger_bos', 101)
    if self.config.get('load_codes_from'):
      logging.info('Loading all codes from: %s', config.load_codes_from)
      with gfile.Open(config.load_codes_from, 'rb') as f:
        codes = np.load(f)
    else:
      # If we don't find a code file to start from we simply use random codes.
      logging.info('Codes not found --> we use from randomly atomic ids.')
      np.random.seed(config.get('seed', 0))
      ne = config.get('n_entities', 6084491)
      nq = config.code_length
      codes = np.random.choice(config.vocab_size, ne * nq,).reshape((ne, nq))
    self.codes = jnp.array(codes.astype(np.int32))

  def __call__(
      self, inputs: jax.Array, train: bool = False,
      debug: bool = False) -> jax.Array:
    del debug, train
    tokens = self.encode_to_indices(inputs)
    # We add two to the vocabulary: sos and eos
    tokens = tokens + 2
    # We shift right. <SOS> is 0.
    b = tokens.shape[0]
    tokens = jnp.concatenate(
        [self.bos * jnp.ones((b, 1)), tokens], axis=-1).astype('int32')
    return jax.lax.stop_gradient(tokens)

  def encode_to_indices(self, inputs: jax.Array) -> jax.Array:
    return self.codes[inputs]


def get_code2id(entity_codes):
  """Gets a code to entity id mapping."""
  code2id = {}
  entity_codes += 2
  for i, code in enumerate(entity_codes):
    code_str = '-'.join([str(int(c))for c in code])
    code2id[code_str] = i
  return code2id


def load_weights(train_state, config):
  """Load pretrained weights or checkpoint.

  Args:
    train_state: the parameters that need to be restored.
    config: config dict that should contain "weights": the path of the
      checkpoint.
  Returns:
    train_state: restored train_state.
    start_step: step number of the checkpoint.
  """
  start_step = 0
  weight_path = config.get('weights', '')
  skip_wrong_shape = config.get('skip_wrong_shape', False)
  load_prefix = config.get('load_prefix', '')
  ignored_keys = config.get('ignored_keys', '')
  if weight_path:
    logging.info('Loading weights from %s', weight_path)
    weight_data = checkpoints.restore_checkpoint(weight_path, None)
    if 'params' in weight_data:
      restored_params = weight_data['params']
    else:
      # Old Scenic train state format.
      restored_params = weight_data['optimizer']['target']
      if 'params' in restored_params:  # Backward compatibility.
        restored_params = restored_params['params']

    expected_params = train_state.params.unfreeze()
    flattened_restored_params = flax.traverse_util.flatten_dict(
        restored_params, sep='/')
    if load_prefix:
      flattened_restored_params = {
          load_prefix + k: v for k, v in flattened_restored_params.items()}
    flattened_expected_params = flax.traverse_util.flatten_dict(
        expected_params, sep='/')
    extra_keys = flattened_restored_params.keys(
        ) - flattened_expected_params.keys()
    missing_keys = flattened_expected_params.keys(
        ) - flattened_restored_params.keys()
    logging.info('Inspect extra keys:%s', extra_keys)
    logging.info('Inspect missing keys:%s', missing_keys)
    for k, v in flattened_restored_params.items():
      if ignored_keys and k.startswith(ignored_keys):
        logging.info('Skipping parameter %s because it starts with %s.', k,
                     ignored_keys)
        continue
      if k not in flattened_expected_params:
        logging.info(
            'Skipping parameter %s in restored model, but not in target.', k)
        continue
      if flattened_expected_params[k].shape != v.shape:
        logging.info(
            'Key: %s. Expected shape: %s. Restored shape: %s', k,
            flattened_expected_params[k].shape, v.shape)
        if not skip_wrong_shape:
          assert ValueError(
              'Shape mismatch between restored and target model'
              'Set config.skip_wrong_shape = True if this is expected.')
      else:
        flattened_expected_params[k] = v
    new_params = flax.traverse_util.unflatten_dict(
        flattened_expected_params, sep='/')
    train_state = train_state.replace(params=flax.core.FrozenDict(new_params))
  return train_state, start_step


def optimizer_with_decoder_multiplier(
    config: ml_collections.ConfigDict,
    params: PyTree,
    use_frozen_params: bool = True):
  """Returns an optimizer with decoder learning rate multiplier.


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

  decoder_layer_prefix = optimizer_config.decoder_layer_prefix
  decoder_multiplier = optimizer_config.decoder_multiplier
  decoder_learning_rate = base_learning_rate * decoder_multiplier
  del optimizer_config.decoder_layer_prefix
  del optimizer_config.decoder_multiplier
  logging.info('Learning rate scales: %s', decoder_learning_rate)

  decoder_config = copy.deepcopy(config)
  decoder_config.lr_configs.base_learning_rate = decoder_learning_rate

  learning_rate_fns = lr_schedules.get_learning_rate_fn(config)
  decoder_learning_rate_fns = lr_schedules.get_learning_rate_fn(
      decoder_config)

  optimizers = {
      False: optimizer_lib.get_optimizer(  # not decoder
          optimizer_config, learning_rate_fns, params),
      True: optimizer_lib.get_optimizer(  # is decoder
          optimizer_config, decoder_learning_rate_fns, params),
  }

  def is_decoder(name: str) -> bool:
    return name.startswith(decoder_layer_prefix)

  flat_params = flax.traverse_util.flatten_dict(
      flax.core.unfreeze(params), keep_empty_nodes=True, sep='/')
  flat_layer_map = {k: is_decoder(k) for k in flat_params}
  layer_map = flax.traverse_util.unflatten_dict(flat_layer_map, sep='/')
  if use_frozen_params:
    layer_map = flax.core.freeze(layer_map)

  logging.info(
      'Layer assignments:\n%s',
      flax.traverse_util.flatten_dict(layer_map, sep='/'))
  tx = optax.multi_transform(optimizers, layer_map)
  return tx


def to_cpu(array: jnp.ndarray):
  """Transfers array (replicated on multiple hosts) to a single host.

  Args:
    array: Replicated array of shape
      [num_hosts, num_devices, local_batch_size, ...].

  Returns:
    array of shape [global_batch_size, ...] where
      global_batch_size = num_devices * local_batch_size
  """
  return jax.device_get(dataset_utils.unshard(jax_utils.unreplicate(array)))
