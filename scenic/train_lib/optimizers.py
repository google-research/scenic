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

"""Defines different optimizers with optax."""
import copy
import dataclasses
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

  Args:
    optimizer_config: Configuration specific to the optimizer.
    learning_rate_fn: Learning rate schedule.
    params: Parameters pytree, used when we want to skip weight decay on bias
    and scale parameters.

  Returns:
    An optax GradientTransformation, this consists of a pair of pure functions
    implementing a gradient transformation.
  """
  # Avoid modifying original config and allow alteration.
  config = copy.deepcopy(optimizer_config).unlock()

  # Skip weight decay for BatchNorm scale or for the bias parameters.
  weight_decay_mask = None
  if 'skip_scale_and_bias_regularization' in config:
    if (config.skip_scale_and_bias_regularization and
        config.get('weight_decay', 0)):
      if params is None:
        raise ValueError('params must be given to obtain weight_decay_mask.')
      weight_decay_mask = jax.tree_map(lambda x: x.ndim != 1, params)
    del config.skip_scale_and_bias_regularization

  optim_ops = []
  # Add weight decay for sgd (possibly with momentum and nesterov).
  if config.optimizer == 'sgd' and 'weight_decay' in config:
    if config.weight_decay:
      optim_ops.append(
          optax.add_decayed_weights(config.weight_decay, weight_decay_mask))
    del config.weight_decay

  # Call the optax optimizer with exact arguments as in the config.
  optimizer_fn = getattr(optax, config.optimizer)
  del config.optimizer
  optim_ops.append(optimizer_fn(learning_rate=learning_rate_fn, **config))

  return optax.chain(*optim_ops)


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

    if config.optimizer == 'nesterov':
      optimizer_config.optimizer = 'sgd'
      optimizer_config.nesterov = True

  if 'skip_scale_and_bias_regularization' in config:
    optimizer_config.skip_scale_and_bias_regularization = (
        config.skip_scale_and_bias_regularization)

  optimizer_config = _scenic_optimizer_args_to_optax_args(optimizer_config)
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
  vals, tree_def = jax.tree_flatten(tree)

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
  """Like jax.tree_map but with a filter on the leaf path name.

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
  vals = [f(v, name) if match_name_fn(name) else v
          for name, v in names_and_vals]
  return tree_def.unflatten(vals)
