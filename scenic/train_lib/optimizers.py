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

"""Defines different optimizers with optax.

Based on
https://github.com/google-research/big_vision/blob/main/big_vision/optax.py
and
https://github.com/google-research/big_vision/blob/main/big_vision/utils.py
"""
import copy
import dataclasses
import operator
import re
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

from absl import logging
import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax


# JAX team is working type checking for pytrees:
# https://github.com/google/jax/issues/3340
PyTree = Any
ScalarOrSchedule = Union[float, optax.Schedule]


def get_optimizer(
    optimizer_config: ml_collections.ConfigDict,
    learning_rate_fn: ScalarOrSchedule,
    params: Optional[PyTree] = None,
) -> optax.GradientTransformation:
  """Constructs the optimizer from the given configuration.

  The function is constructed in such a way that it will throw errors if
  fields in the optimizer_config are misspelled.

  Args:
    optimizer_config: Configuration specific to the optimizer. The config
        can contain the following fields:
        - optimizer: name of the optax optimizer.
        - **kwargs: fields specific to the optax optimizer.
        - weight_decay: value of the weight decay.
        - skip_scale_and_bias_regularization: if True, do not apply weight
          decay to scale and biases.
        - grad_clip: configdict with settings of gradient clipping.
        - freeze_params_reg_exp: regular expression to define which weights
          will be frozen during training. This uses re.search, so 'conv' would
          match any parameter which has 'conv' somewhere in its name such as
          'cnn/first_conv_layer/bias'. Note that only parameters will be frozen,
          which means batch_norm remains unaffected.
    learning_rate_fn: Learning rate schedule.
    params: Parameters pytree, used when we want to skip weight decay on bias
      and scale parameters. Also used for freezing weights.

  Returns:
    An optax GradientTransformation, this consists of a pair of pure functions
    implementing a gradient transformation.
  """
  # Avoid modifying original config and allow alteration.
  config = copy.deepcopy(optimizer_config).unlock()

  # Skip weight decay for BatchNorm scale or for the bias parameters.
  weight_decay_mask = None
  if config.get('skip_scale_and_bias_regularization') is not None:
    if (config.skip_scale_and_bias_regularization and
        config.get('weight_decay', 0)):
      if params is None:
        raise ValueError('params must be given to obtain weight_decay_mask.')
      weight_decay_mask = jax.tree_util.tree_map(lambda x: x.ndim != 1, params)
  if 'skip_scale_and_bias_regularization' in config:
    del config.skip_scale_and_bias_regularization

  optim_ops = []
  # Add weight decay for sgd (possibly with momentum and nesterov).
  if config.optimizer == 'sgd' and 'weight_decay' in config:
    if config.weight_decay:
      optim_ops.append(
          optax.add_decayed_weights(config.weight_decay, weight_decay_mask))
    del config.weight_decay

  if weight_decay_mask and config.optimizer in {'adamw', 'lamb', 'adamaxw'}:
    config.mask = weight_decay_mask
  elif weight_decay_mask and config.optimizer in {'adafactor', 'lars'}:
    config.weight_decay_mask = weight_decay_mask

  # Add gradient clipping before optimizer operations.
  if config.get('grad_clip') is not None:
    grad_clip_config = config.grad_clip
    clip_method = grad_clip_config.get('clip_method', None)
    clip_value = grad_clip_config.get('clip_value', None)
    if clip_method is not None and clip_value is not None:
      if clip_method == 'clip_by_global_norm':
        optim_ops.append(optax.clip_by_global_norm(clip_value))
      elif clip_method == 'adaptive_grad_clip':
        optim_ops.append(optax.adaptive_grad_clip(clip_value))
      elif clip_method == 'clip':
        optim_ops.append(optax.clip(clip_value))
      elif clip_method == 'clip_by_block_rms':
        optim_ops.append(optax.clip_by_block_rms(clip_value))
      else:
        logging.info('%s is not supported', clip_method)
  if 'grad_clip' in config:
    del config.grad_clip

  # Remove freeze_params_reg_exp here. This should be the last operation to
  # ensure parameters are truly frozen. But this field needs to be removed
  # because all remaining fields in the config are given to the optimizer.
  freeze_mask = None
  unfreeze_mask = None
  if config.get('freeze_params_reg_exp') is not None:
    if params is None:
      raise ValueError('params must be given to obtain frozen parameters.')
    freeze_mask = tree_mask(params, config.freeze_params_reg_exp)
    unfreeze_mask = jax.tree_util.tree_map(lambda x: not x, freeze_mask)
    del config.freeze_params_reg_exp

    num_params_unfrozen = jax.tree_util.tree_reduce(operator.add, unfreeze_mask)
    if not num_params_unfrozen:
      raise ValueError('freeze_params_reg_exp matched all parameters in '
                       'the model, which prevents any training from happening.')
  if 'freeze_params_reg_exp' in config:
    del config.freeze_params_reg_exp

  # Call the optax optimizer with exact arguments as in the config.
  # This throws an error when the config has (spelling) mistakes.
  optimizer_fn = getattr(optax, config.optimizer)
  del config.optimizer
  optax_optimizer = optimizer_fn(learning_rate=learning_rate_fn, **config)
  # Apply to unfrozen weights to prevent change in optimizer state.
  # In turn, this prevents unnecessary gradient calculations.
  if unfreeze_mask:
    optax_optimizer = optax.masked(optax_optimizer, unfreeze_mask)
  optim_ops.append(optax_optimizer)

  # Freezing params should be the final operation in the optax chain to ensure
  # that freezing overrides everything including weight decay.
  if freeze_mask:
    optim_ops.append(optax.masked(optax.set_to_zero(), freeze_mask))

    # Log variables which will change during training.
    freeze_mask_flat = flax.traverse_util.flatten_dict(freeze_mask, sep='/')
    logging.info('Freeze mask set. Training only on the following params:')
    for param_name, value in freeze_mask_flat.items():
      if not value:
        logging.info('--> %s', param_name)

  return optax.chain(*optim_ops)


def tree_mask(params: PyTree, reg_exp: str):
  """Returns a tree mask based on regular expression for use with optax.masked.

  Args:
    params: PyTree with parameters.
    reg_exp: Regular expression. Will be compiled and used together with
        re.search.
  """
  pattern = re.compile(reg_exp)

  def match_var_name(_, name):
    if pattern.search(name):
      return True
    return False

  return tree_map_with_names_values(match_var_name, params)


def get_optax_optimizer_config(
    config: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Obtain optimizer from main config."""
  optimizer_config = config.get('optimizer_configs',
                                ml_collections.ConfigDict())

  # New-style config: all optimizer-related fields are in optimizer_configs.
  if 'optimizer' in optimizer_config:
    if 'optimizer' in config:
      raise ValueError(
          'Both config.optimizer and config.optimizer_configs.optimizer are '
          'defined. Define it only once to avoid possible contradictions. '
          'The preferred location is in config.optimizer_configs.optimizer')
    return optimizer_config

  # Backwards compatibility: copy optimizer field into the optimizer config.
  optimizer_config = copy.deepcopy(optimizer_config).unlock()
  if 'optimizer' in config:
    optimizer_config.optimizer = config.optimizer

    # The old optimizers have adam with weight decay. However, in optax this is
    # done using the adamw optimizer.
    if config.optimizer == 'adam' and 'weight_decay' in optimizer_config:
      optimizer_config.optimizer = 'adamw'

    if config.optimizer == 'momentum':
      optimizer_config.optimizer = 'sgd'
      if 'momentum' not in optimizer_config:
        # flax.optim had a default momentum value of 0.9.
        # optax.sgd has a default momentum of 0.
        logging.warning(
            'flax.optim had a default momentum value of 0.9. optax has a '
            'default value of 0. As a momentum value was not specified, '
            'adding momentum=0.9 to optimizer config.')
        optimizer_config.momentum = 0.9

    if config.optimizer == 'nesterov':
      optimizer_config.optimizer = 'sgd'
      optimizer_config.nesterov = True

  if 'skip_scale_and_bias_regularization' in config:
    optimizer_config.skip_scale_and_bias_regularization = (
        config.skip_scale_and_bias_regularization)

  optimizer_config = _scenic_optimizer_args_to_optax_args(optimizer_config)

  if 'grad_clip_configs' in config:
    optimizer_config.grad_clip = config.grad_clip_configs

  optimizer_config.lock()
  logging.info('Optimizer config after backwards compatibility operations:\n%s',
               optimizer_config)
  return optimizer_config


def _scenic_optimizer_args_to_optax_args(
    config: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Transform original scenic arguments to optax arguments."""
  if 'beta1' in config:
    config.b1 = config.beta1
    del config.beta1
  if 'beta2' in config:
    config.b2 = config.beta2
    del config.beta2
  if 'epsilon' in config:
    config.eps = config.epsilon
    del config.epsilon
  return config


def _traverse_with_names(
    tree: PyTree) -> Generator[Tuple[str, PyTree], None, None]:
  """Traverses nested dicts/dataclasses and emits (leaf_name, leaf_val)."""
  if dataclasses.is_dataclass(tree):
    tree = flax.serialization.to_state_dict(tree)
  if isinstance(tree, (dict, flax.core.frozen_dict.FrozenDict)):
    keys = sorted(tree.keys())
    for key in keys:
      for path, v in _traverse_with_names(tree[key]):
        yield (key + '/' + path).rstrip('/'), v
  else:
    yield '', tree


def tree_flatten_with_names(
    tree: PyTree) -> Tuple[List[Tuple[str, jnp.ndarray]], PyTree]:
  """Populates tree_flatten with leaf names.

  This function populates output of tree_flatten with leaf names, using a
  custom traversal that produces names is provided. The custom traversal does
  NOT have to traverse tree in the same order as jax, as we take care of
  automatically aligning jax' and custom traversals.

  Args:
    tree: python tree.

  Returns:
    A list of values with names: [(name, value), ...]
  """
  vals, tree_def = jax.tree_util.tree_flatten(tree)

  # "Fake" token tree that is use to track jax internal tree traversal and
  # adjust our custom tree traversal to be compatible with it.
  tokens = range(len(vals))
  token_tree = tree_def.unflatten(tokens)
  val_names, perm = zip(*_traverse_with_names(token_tree))
  inv_perm = np.argsort(perm)

  # Custom traverasal should visit the same number of leaves.
  assert len(val_names) == len(vals)

  return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def tree_map_with_names(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    param_tree: PyTree,
    match_name_fn: Callable[[str], bool] = lambda name: True) -> PyTree:
  """Like jax.tree_util.tree_map but with a filter on the leaf path name.

  Args:
    f: The function to be applied to each parameter in `param_tree`. Takes value
      as argument.
    param_tree: The tree of parameters `f` should be applied to.
    match_name_fn: This function is called with each tree leaf's path name,
      which has a path-like format ("a/b/c"), and decides whether `f` should be
      applied to that leaf or the leaf should be kept as-is.

  Returns:
    A tree identical in structure to `param_tree` but with the leaves the
    result of calling `f` on them in the cases where `match_name_fn` returns
    True for that leaf's path name.
  """
  names_and_vals, tree_def = tree_flatten_with_names(param_tree)
  vals = [f(v) if match_name_fn(name) else v for name, v in names_and_vals]
  return tree_def.unflatten(vals)


def tree_map_with_names_values(
    f: Callable[[jnp.ndarray, str], jnp.ndarray],
    param_tree: PyTree,
    match_name_fn: Callable[[str], bool] = lambda name: True) -> PyTree:
  """Like tree_map_with_names but with `f` having access to values *and* names.

  Args:
    f: The function to be applied to each parameter in `param_tree`. Takes value
      and name as arguments.
    param_tree: The tree of parameters `f` should be applied to.
    match_name_fn: This function is called with each tree leaf's path name,
      which has a path-like format ("a/b/c"), and decides whether `f` should be
      applied to that leaf or the leaf should be kept as-is.

  Returns:
    A tree identical in structure to `param_tree` but with the leaves the
    result of calling `f` on them in the cases where `match_name_fn` returns
    True for that leaf's path name.
  """
  names_and_vals, tree_def = tree_flatten_with_names(param_tree)
  vals = [
      f(v, name) if match_name_fn(name) else v for name, v in names_and_vals
  ]
  return tree_def.unflatten(vals)
