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

"""Utils for training.

'pop_axes_names' and 're_add_axis_names' are forked from
scenic/projects/vid2seq/train_utils.py

"""

from typing import Any, Optional, Tuple

from absl import logging
import flax
from flax.core import frozen_dict
from flax.training import checkpoints
from scenic.common_lib import debug_utils
from scenic.train_lib import train_utils

TrainState = train_utils.TrainState
FrozenDict = flax.core.FrozenDict


def pop_axes_names(
    train_state: TrainState,
    axes_name: str = 'param_axes') -> Tuple[TrainState, Optional[Any]]:
  """Removes axes_names from model_state for a train state.

  Args:
    train_state: Training state.
    axes_name: the string specifying the name in the model_state

  Returns:
    New train state without axes_names in model_state, axes_names metadata if it
    was removed (so it can be re-added).
  """
  model_state = train_state.model_state
  if axes_name in train_state.model_state:
    model_state, param_axes = frozen_dict.freeze(model_state).pop(axes_name)
    return train_state.replace(model_state=model_state), param_axes
  else:
    return train_state, None


def re_add_axis_names(train_state: TrainState,
                      param_axes: Any,
                      axes_name: str = 'param_axes') -> TrainState:
  """Adds axes_names to model_state for a train state.

  Args:
    train_state: Training state.
    param_axes: Model axes metadata to re-add.
    axes_name: the string specifying the name in the model_state

  Returns:
    New train state without axes_names in model_state, axes_names metadata if it
    was removed (so it can be re-added).
  """
  if param_axes:
    model_state = frozen_dict.unfreeze(train_state.model_state)
    model_state[axes_name] = param_axes
    return train_state.replace(model_state=frozen_dict.freeze(model_state))
  else:
    return train_state


def copy_matched_params(
    expected_params, restored_params, load_prefix='', load_replace=(),
    skip_wrong_shape=False, load_available_shape=()):
  """Copy matched parameters from a restored one."""
  flattened_restored_params = flax.traverse_util.flatten_dict(
      restored_params, sep='/')
  if load_prefix:
    flattened_restored_params = {
        load_prefix + k: v for k, v in flattened_restored_params.items()}
  if load_replace:
    for x in load_replace:
      flattened_restored_params = {
          k.replace(
              x[0], x[1]): v for k, v in flattened_restored_params.items()}
  flattened_expected_params = flax.traverse_util.flatten_dict(
      expected_params, sep='/')
  extra_keys = flattened_restored_params.keys(
      ) - flattened_expected_params.keys()
  missing_keys = flattened_expected_params.keys(
      ) - flattened_restored_params.keys()
  logging.info('Inspect extra keys:%s', extra_keys)
  logging.info('Inspect missing keys:%s', missing_keys)
  for k, v in flattened_restored_params.items():
    if k not in flattened_expected_params:
      logging.info(
          'Skipping parameter %s in restored model, but not in target.', k)
      continue
    if flattened_expected_params[k].shape != v.shape:
      logging.info(
          'Key: %s. Expected shape: %s. Restored shape: %s', k,
          flattened_expected_params[k].shape, v.shape)
      if k in load_available_shape:
        logging.info('Loading available shape for Key: %s.', k)
        if len(v.shape) == 1:
          flattened_expected_params[k] = flattened_expected_params[k].at[
              :v.shape[0]].set(v)
        else:
          flattened_expected_params[k] = flattened_expected_params[k].at[
              :v.shape[0], :v.shape[1]].set(v)
      elif not skip_wrong_shape:
        raise ValueError(
            'Shape mismatch between restored and target model'
            'Set config.skip_wrong_shape = True if this is expected.')
    else:
      flattened_expected_params[k] = v
  new_params = flax.traverse_util.unflatten_dict(
      flattened_expected_params, sep='/')
  return new_params


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
  load_available_shape = config.get('load_available_shape', ())
  load_prefix = config.get('load_prefix', '')
  load_replace = config.get('load_replace', ())
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
    new_params = copy_matched_params(
        expected_params, restored_params,
        load_prefix=load_prefix, load_replace=load_replace,
        skip_wrong_shape=skip_wrong_shape,
        load_available_shape=load_available_shape)
    train_state = train_state.replace(params=FrozenDict(new_params))
  debug_utils.log_param_shapes(train_state.params)
  logging.info('Finish loading weights from %s', weight_path)

  return train_state, start_step

