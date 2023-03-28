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

from collections import abc
import os
import re
from typing import Any, Dict, Mapping, List, Optional, Union, Tuple

from absl import logging
import flax
from flax.training import checkpoints
import jax
import numpy as np

from scenic.train_lib_deprecated import train_utils
from tensorflow.io import gfile

# JAX team is working on type annotation for pytree:
# https://github.com/google/jax/issues/1555
PyTree = Union[Mapping[str, Mapping], Any]


def get_params_and_model_state_dict(
    restored_train_state: Union[PyTree, train_utils.TrainState],
) -> Tuple[PyTree, Optional[PyTree]]:
  """Restores the params and model state.

  This function also applies the conversion needed for pre-Linen checkpoints.

  Args:
    restored_train_state: A dictionary that contains a check-pointed TrainState.

  Returns:
    A tuple of restored params and restored models state. Note that these are
    not frozen, and need to be frozen before passing them to the optimizer.
  """
  if 'optimizer' in restored_train_state:
    restored_params = restored_train_state['optimizer']['target']
  else:
    restored_params = restored_train_state['params']
  restored_model_state = restored_train_state.get('model_state')
  if 'params' in restored_params:  # Backward compatibility.
    restored_params = restored_params['params']
    restored_params = dict(checkpoints.convert_pre_linen(restored_params))
    if restored_model_state:
      restored_model_state = checkpoints.convert_pre_linen(
          flax.traverse_util.unflatten_dict({
              tuple(k.split('/')[1:]): v
              for k, v in restored_model_state.items()
          }))
  return restored_params, restored_model_state


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
      logging.warning(
          '%s in checkpoint doesn\'t exist in model. Skip.', m_key_str)
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
  (restored_params,
   restored_model_state) = get_params_and_model_state_dict(pretrain_state)
  model_params = train_state.optimizer.target
  model_params = _replace_dict(model_params,
                               restored_params,
                               ckpt_prefix_path,
                               model_prefix_path,
                               name_mapping,
                               skip_regex)
  new_optimizer = train_state.optimizer.replace(
      target=model_params)
  train_state = train_state.replace(  # pytype: disable=attribute-error
      optimizer=new_optimizer)
  if (restored_model_state is not None and
      train_state.model_state is not None and
      train_state.model_state):
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
  (restored_params,
   restored_model_state) = get_params_and_model_state_dict(restored_train_state)
  restored_params = flax.core.freeze(restored_params)
  restored_model_state = flax.core.freeze(restored_model_state)
  if train_state:
    new_train_state = train_state
    new_optimizer = train_state.optimizer.replace(
        # Inspect and compare the parameters of the model with the init-model.
        target=inspect_params(
            expected_params=train_state.optimizer.target,
            restored_params=restored_params,
            fail_if_extra=False,
            fail_if_missing=False,
            fail_if_shapes_mismatch=False))
  else:
    new_train_state = train_utils.TrainState()
    new_optimizer = {'target': restored_params}

  new_train_state = new_train_state.replace(  # pytype: disable=attribute-error
      optimizer=new_optimizer,
      model_state=restored_model_state,
      global_step=int(restored_train_state['global_step']),
      rng=restored_train_state['rng'],
      accum_train_time=restored_train_state.get('accum_train_time', 0))

  return new_train_state


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
  checkpoint_data = np.load(gfile.GFile(checkpoint_path, 'rb'))
  tree = unflatten_dict(checkpoint_data, separator='/', leaf_idx=0)

  restored_params = tree['opt']['target']
  if convert_to_linen:
    restored_params = checkpoints.convert_pre_linen(restored_params)
  restored_params = dict(restored_params)
  if train_state:
    restored_params = inspect_params(
        expected_params=train_state.optimizer.target,
        restored_params=restored_params,
        fail_if_extra=False,
        fail_if_missing=False,
        fail_if_shapes_mismatch=False)
  else:
    train_state = train_utils.TrainState()
  # pytype: disable=wrong-arg-types
  restored_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=int(tree['opt']['state']['step']),
      optimizer={'target': restored_params},
      accum_train_time=int(tree['extra']['accum_train_time']))
  # pytype: enable=wrong-arg-types

  return restored_train_state


def convert_strict_big_vision_to_scenic_checkpoint(
    checkpoint_path: str,
    train_state: train_utils.TrainState) -> train_utils.TrainState:
  """Converts a checkpoint saved by big_vision into a Scenic TrainState.

  This assumes that all the variables in the checkpoint are in the scenic
  train state optimizer.

  Args:
    checkpoint_path: Full path to a big_vision checkpoint file
    train_state: A Scenic TrainState object.

  Returns:
    restored_train_state: Scenic TrainState object with the 'global step',
      'optimizer' and 'accum_train_time' fields.
  """

  def load_big_vision_checkpoint(tree, path):
    assert gfile.exists(path), 'Checkpoint {} does not exist'.format(path)
    with gfile.GFile(path, 'rb') as f:
      data = np.load(f, allow_pickle=False)
      return tree.unflatten(tuple(data.values()))

  checkpoint_tree = {
      'opt': train_state.optimizer,
      'extra': dict(accum_train_time=0.0)
  }
  _, checkpoint_tree = jax.tree_util.tree_flatten(checkpoint_tree)
  checkpoint = load_big_vision_checkpoint(checkpoint_tree, checkpoint_path)
  logging.info('Loaded big_vision checkpoint from %s', checkpoint_path)

  restored_params = checkpoint['opt'].target
  restored_params = dict(checkpoints.convert_pre_linen(restored_params))

  restored_params = inspect_params(
      expected_params=train_state.optimizer.target,
      restored_params=restored_params,
      fail_if_extra=False,
      fail_if_missing=False,
      fail_if_shapes_mismatch=False)

  # pytype: disable=wrong-arg-types
  restored_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=int(checkpoint['opt'].state.step),
      optimizer={'target': restored_params},
      accum_train_time=int(checkpoint['extra']['accum_train_time']))
  # pytype: enable=wrong-arg-types

  return restored_train_state


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
      else:
        logging.info('Key found with matching shape: %s', key)

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
