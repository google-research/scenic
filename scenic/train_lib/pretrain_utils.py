# Copyright 2022 The Scenic Authors.
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

import collections
import os
import re
from typing import Any, Mapping, List, Optional, Union

from absl import logging
import flax
from flax.training import checkpoints

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
  restored_params = flax.core.freeze(restored_train_state['params'])
  restored_model_state = flax.core.freeze(restored_train_state['model_state'])
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
      accum_train_time=restored_train_state.get('accum_train_time', 0))
  return train_state


def inspect_params(*,
                   expected_params: PyTree,
                   restored_params: PyTree,
                   fail_if_extra: bool = True,
                   fail_if_missing: bool = True,
                   fail_if_shapes_mismatch: bool = False) -> PyTree:
  """Inspects whether the params are consistent with the expected keys."""

  def _flatten_params(d, parent_key='', sep='/'):
    """Flattens a dictionary, keeping empty leaves."""
    items = []
    for k, v in d.items():
      path = parent_key + sep + k if parent_key else k
      if isinstance(v, collections.MutableMapping):
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
