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

"""Utility functions for using pretrained models."""

from collections import abc
import os
import re
from typing import Any, Dict, Mapping, List, Optional, Union

from absl import logging
from big_vision import utils
import flax
from flax.training import checkpoints
import numpy as np

from scenic.train_lib import train_utils
from tensorflow.io import gfile

# JAX team is working on type annotation for pytree:
# https://github.com/google/jax/issues/1555
PyTree = Union[Mapping[str, Mapping], Any]


def _replace_dict(model: PyTree,
                  restored: PyTree,
                  ckpt_prefix_path: Optional[List[str]] = None,
                  model_prefix_path: Optional[List[str]] = None,
                  name_mapping: Optional[Mapping[str, str]] = None,
                  skip_regex: Optional[str] = None) -> PyTree:
  """Replaces values in model dictionary with restored ones from checkpoint."""
  name_mapping = name_mapping or {}

  model = flax.core.unfreeze(model)  # pytype: disable=wrong-arg-types
  restored = flax.core.unfreeze(restored)  # pytype: disable=wrong-arg-types

  if ckpt_prefix_path:
    for p in ckpt_prefix_path:
      restored = restored[p]

  if model_prefix_path:
    for p in reversed(model_prefix_path):
      restored = {p: restored}

  # Flatten nested parameters to a dict of str -> tensor. Keys are tuples
  # from the path in the nested dictionary to the specific tensor. E.g.,
  # {'a1': {'b1': t1, 'b2': t2}, 'a2': t3}
  # -> {('a1', 'b1'): t1, ('a1', 'b2'): t2, ('a2',): t3}.
  restored_flat = flax.traverse_util.flatten_dict(
      dict(restored), keep_empty_nodes=True)
  model_flat = flax.traverse_util.flatten_dict(
      dict(model), keep_empty_nodes=True)

  for m_key, m_params in restored_flat.items():
    # pytype: disable=attribute-error
    for name, to_replace in name_mapping.items():
      m_key = tuple(to_replace if k == name else k for k in m_key)
    # pytype: enable=attribute-error
    m_key_str = '/'.join(m_key)
    if m_key not in model_flat:
      logging.warning('%s in checkpoint doesn\'t exist in model. Skip.',
                      m_key_str)
      continue
    if skip_regex and re.findall(skip_regex, m_key_str):
      logging.info('Skip loading parameter %s.', m_key_str)
      continue
    logging.info('Loading %s from checkpoint into model', m_key_str)
    model_flat[m_key] = m_params

  return flax.core.freeze(flax.traverse_util.unflatten_dict(model_flat))


def init_from_pretrain_state(
    train_state: train_utils.TrainState,
    pretrain_state: Union[PyTree, train_utils.TrainState],
    ckpt_prefix_path: Optional[List[str]] = None,
    model_prefix_path: Optional[List[str]] = None,
    name_mapping: Optional[Mapping[str, str]] = None,
    skip_regex: Optional[str] = None) -> train_utils.TrainState:
  """Updates the train_state with data from pretrain_state.

  Args:
    train_state: A raw TrainState for the model.
    pretrain_state: A TrainState that is loaded with parameters/state of
      a  pretrained model.
    ckpt_prefix_path: Prefix to restored model parameters.
    model_prefix_path: Prefix to the parameters to replace in the subtree model.
    name_mapping: Mapping from parameter names of checkpoint to this model.
    skip_regex: If there is a parameter whose parent keys match the regex,
      the parameter will not be replaced from pretrain_state.

  Returns:
    Updated train_state.
  """
  name_mapping = name_mapping or {}
  restored_params = pretrain_state['params']
  restored_model_state = pretrain_state['model_state']
  model_params = _replace_dict(train_state.params, restored_params,
                               ckpt_prefix_path, model_prefix_path,
                               name_mapping, skip_regex)
  train_state = train_state.replace(params=model_params)
  # TODO(scenic): Add support for optionally restoring optimizer state.
  if (restored_model_state is not None and
      train_state.model_state is not None and train_state.model_state):
    if model_prefix_path:
      # Insert model prefix after 'batch_stats'.
      model_prefix_path = ['batch_stats'] + model_prefix_path
      if 'batch_stats' in restored_model_state:
        ckpt_prefix_path = ckpt_prefix_path or []
        ckpt_prefix_path = ['batch_stats'] + ckpt_prefix_path
    elif 'batch_stats' not in restored_model_state:  # Backward compatibility.
      model_prefix_path = ['batch_stats']
    if ckpt_prefix_path and ckpt_prefix_path[0] != 'batch_stats':
      ckpt_prefix_path = ['batch_stats'] + ckpt_prefix_path
    model_state = _replace_dict(train_state.model_state,
                                restored_model_state,
                                ckpt_prefix_path,
                                model_prefix_path,
                                name_mapping,
                                skip_regex)
    train_state = train_state.replace(  # pytype: disable=attribute-error
        model_state=model_state)
  return train_state


def restore_pretrained_checkpoint(
    checkpoint_path: str,
    train_state: Optional[train_utils.TrainState] = None,
    assert_exist: bool = False,
    step: Optional[int] = None) -> train_utils.TrainState:
  """Restores the last checkpoint.

  First restores the checkpoint, which is an instance of TrainState that holds
  the state of training. This function also take care converting pre-Linen
  checkpoints.

  Args:
    checkpoint_path: Directory for saving the checkpoint.
    train_state: An instance of TrainState that holds the state of training.
    assert_exist: Assert that there is at least one checkpoint exists in the
      given path.
    step: Step number to load or None to load latest. If specified,
      checkpoint_path must be a directory.

  Returns:
    Training state and an int which is the current step.
  """
  if assert_exist:
    glob_path = os.path.join(checkpoint_path, 'checkpoint_*')
    if not gfile.glob(glob_path):
      raise ValueError('No checkpoint for the pretrained model is found in: '
                       f'{checkpoint_path}')
  restored_train_state = checkpoints.restore_checkpoint(checkpoint_path, None,
                                                        step)
  if restored_train_state is None:
    raise ValueError('No checkpoint for the pretrained model is found in: '
                     f'{checkpoint_path}')
  if 'params' in restored_train_state:
    # restored_train_state was trained using optax
    restored_params = flax.core.freeze(restored_train_state['params'])
  else:
    # restored_train_state was trained using flax.optim. Note that this does
    # not convert the naming of pre-Linen checkpoints.
    restored_params = restored_train_state['optimizer']['target']
    if 'params' in restored_params:  # Backward compatibility.
      restored_params = restored_params['params']
      restored_params = dict(checkpoints.convert_pre_linen(restored_params))
    restored_params = flax.core.freeze(restored_params)

  restored_model_state = (
      None if restored_train_state['model_state'] is None else
      flax.core.freeze(restored_train_state['model_state'])
  )

  if not train_state:
    train_state = train_utils.TrainState()
    params = restored_params
  else:
    # Inspect and compare the parameters of the model with the init-model.
    params = inspect_params(
        expected_params=train_state.params,
        restored_params=restored_params,
        fail_if_extra=False,
        fail_if_missing=False,
        fail_if_shapes_mismatch=False)
  train_state = train_state.replace(
      # Inspect and compare the parameters of the model with the init-model.
      params=params,
      model_state=restored_model_state,
      global_step=int(restored_train_state['global_step']),
      rng=restored_train_state['rng'],
      metadata=restored_train_state.get('metadata', None))
  return train_state


# pylint: disable=g-doc-args,g-doc-return-or-yield
def inspect_params(*,
                   expected_params: PyTree,
                   restored_params: PyTree,
                   fail_if_extra: bool = True,
                   fail_if_missing: bool = True,
                   fail_if_shapes_mismatch: bool = False) -> PyTree:
  """Inspects whether the params are consistent with the expected keys.

  Based on
  https://github.com/google-research/big_vision/blob/main/big_vision/model/common.py.
  """

  def _flatten_params(d, parent_key='', sep='/'):
    """Flattens a dictionary, keeping empty leaves."""
    items = []
    for k, v in d.items():
      path = parent_key + sep + k if parent_key else k
      if isinstance(v, abc.MutableMapping):
        items.extend(_flatten_params(v, path, sep=sep).items())
      else:
        items.append((path, v))
    # Keeps the empty dict if it was set explicitly.
    if parent_key and not d:
      items.append((parent_key, {}))
    return dict(items)

  expected_flat = _flatten_params(flax.core.unfreeze(expected_params))
  restored_flat = _flatten_params(flax.core.unfreeze(restored_params))
  missing_keys = expected_flat.keys() - restored_flat.keys()
  extra_keys = restored_flat.keys() - expected_flat.keys()

  is_shape_mismatch = False
  for key in restored_flat:
    if key in expected_flat:
      restored_shape = None
      expected_shape = None
      # Handle empty nodes (without trainable params)
      if not isinstance(restored_flat[key], dict):
        restored_shape = restored_flat[key].shape
      if not isinstance(expected_flat[key], dict):
        expected_shape = expected_flat[key].shape

      if restored_shape != expected_shape:
        is_shape_mismatch = True
        logging.warning('Key: %s. Expected shape: %s. Restored shape: %s', key,
                        expected_flat[key].shape, restored_flat[key].shape)

  # Adds back empty dict explicitly, to support layers without weights.
  # Context: FLAX ignores empty dict during serialization.
  empty_keys = set()
  for k in missing_keys:
    if isinstance(expected_flat[k], dict) and not expected_flat[k]:
      restored_params[k] = {}  # pytype: disable=unsupported-operands
      empty_keys.add(k)
  missing_keys -= empty_keys

  if empty_keys:
    logging.warning('Inspect recovered empty keys:\n%s', empty_keys)

  logging.info('Inspect missing keys:\n%s', missing_keys)
  logging.info('Inspect extra keys:\n%s', extra_keys)

  if fail_if_shapes_mismatch and is_shape_mismatch:
    raise ValueError('Shape mismatch between restored and target model')

  if (missing_keys and fail_if_missing) or (extra_keys and fail_if_extra):
    raise ValueError(
        f'Missing params from checkpoint: {missing_keys}.\n'
        f'Extra params in checkpoint: {extra_keys}.\n'
        f'Restored params from checkpoint: {restored_flat.keys()}.\n'
        f'Expected params from code: {expected_flat.keys()}.')
  return restored_params
# pylint: enable=g-doc-args,g-doc-return-or-yield


def convert_big_vision_to_scenic_checkpoint(
    checkpoint_path: str,
    train_state: Optional[train_utils.TrainState] = None,
    convert_to_linen: bool = True) -> train_utils.TrainState:
  """Converts a big_vision checkpoint to a scenic train state.

  The model weights, global step and accumulated train time are extracted.
  Optimizer state, such as the momentum, is not extracted.

  Args:
    checkpoint_path: Path to big_vision checkpoint.
    train_state: A Scenic TrainState object.
    convert_to_linen: Whether to convert to Linen format.

  Returns:
    restored_train_state: Scenic train state with model weights, global step
      and accumulated training time.
  """

  def unflatten_dict(flattened: Dict[str, Any],
                     separator: str = '/',
                     leaf_idx: int = -1) -> Dict[str, Any]:
    unflattened = {}
    for k, v in flattened.items():
      subtree = unflattened
      if leaf_idx != 0:
        path = k.split(separator)[:leaf_idx]
      else:
        path = k.split(separator)
      for k2 in path[:-1]:
        if k2 not in subtree:
          subtree[k2] = {}
        subtree = subtree[k2]
      subtree[path[-1]] = v
    return unflattened

  logging.info('Loading big_vision checkpoint from %s', checkpoint_path)
  if '.bv' in checkpoint_path:
    checkpoint_data = utils.load_checkpoint_ts(checkpoint_path)
  else:
    checkpoint_data = np.load(gfile.GFile(checkpoint_path, 'rb'))
  tree = unflatten_dict(checkpoint_data, separator='/', leaf_idx=0)

  restored_params = (
      tree['opt']['target'] if 'target' in tree['opt'] else tree['params']
  )
  if convert_to_linen:
    restored_params = checkpoints.convert_pre_linen(restored_params)
  restored_params = dict(restored_params)
  if train_state:
    restored_params = inspect_params(
        expected_params=train_state.params,
        restored_params=restored_params,
        fail_if_extra=False,
        fail_if_missing=False,
        fail_if_shapes_mismatch=False)
  else:
    train_state = train_utils.TrainState()

  # pytype: disable=wrong-arg-types
  restored_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=int(
          tree['opt']['state']['step'] if 'state' in tree['opt'] else 0
      ),
      params=restored_params,
  )
  # pytype: enable=wrong-arg-types

  return restored_train_state
