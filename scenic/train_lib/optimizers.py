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

import dataclasses
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

from absl import logging
import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from scenic.train_lib import lr_schedules

# JAX team is working type checking for pytrees:
# https://github.com/google/jax/issues/3340
PyTree = Any


def get_optimizer(
    config: ml_collections.ConfigDict,
    params: Optional[PyTree] = None,
    ) -> Tuple[optax.GradientTransformation, Union[float, Callable[[int], float]]]:  # pylint: disable=line-too-long
  """Constructs  the optimizer from the given configuration.

  Args:
    config: Configuration.
    params: Parameters pytree, used when we want to skip weight decay on bias
    and scale parameters.

  Returns:
    An optax GradientTransformation, this consists of a pair of pure functions
    implementing a gradient transformation.
  """

  optimizer_configs = config.get('optimizer_configs',
                                 ml_collections.ConfigDict())
  # Learning rate.
  lr_configs = config.get('lr_configs', ml_collections.ConfigDict())
  learning_rate_fn = 0.
  if 'learning_rate_schedule' in lr_configs:
    learning_rate_fn = lr_schedules.get_learning_rate_fn(config)
  elif 'base_learning_rate' in lr_configs:
    learning_rate_fn = lr_configs.base_learning_rate
  # Weight decay.
  wd = optimizer_configs.get('weight_decay', 0.)
  weight_decay_mask = None
  # Skip weight decay for BatchNorm scale or for the bias parameters.
  if wd and params and config.get('skip_scale_and_bias_regularization', False):
    weight_decay_mask = jax.tree_map(lambda x: x.ndim != 1, params)
  optim_ops = []
  if config.optimizer in ['sgd', 'momentum', 'nesterov']:
    if wd:
      optim_ops.append(optax.add_decayed_weights(wd, weight_decay_mask))
    if config.optimizer == 'sgd':
      optim_ops.append(optax.sgd(learning_rate=learning_rate_fn, momentum=0))
    elif config.optimizer == 'momentum':
      optim_ops.append(optax.sgd(
          learning_rate=learning_rate_fn,
          momentum=optimizer_configs.get('momentum', 0.9)))
    elif config.optimizer == 'nesterov':
      optim_ops.append(optax.sgd(
          learning_rate=learning_rate_fn, nesterov=True,
          momentum=optimizer_configs.get('momentum', 0.9)))
  elif config.optimizer == 'adamw':
    optim_ops.append(optax.adamw(
        learning_rate=learning_rate_fn,
        b1=optimizer_configs.get('beta1', 0.9),
        b2=optimizer_configs.get('beta2', 0.999),
        eps=optimizer_configs.get('epsilon', 1e-8),
        weight_decay=wd, mask=weight_decay_mask))
  elif config.optimizer == 'adam':
    optim_ops.append(optax.adam(
        learning_rate=learning_rate_fn,
        b1=optimizer_configs.get('beta1', 0.9),
        b2=optimizer_configs.get('beta2', 0.999),
        eps=optimizer_configs.get('epsilon', 1e-8)))
  elif config.optimizer == 'adafactor':
    optim_ops.append(optax.adafactor(
        learning_rate=learning_rate_fn,
        multiply_by_parameter_scale=False,
        momentum=optimizer_configs.get('momentum', 0.9),
        decay_rate=0.999,
        weight_decay_rate=wd, weight_decay_mask=weight_decay_mask))
  elif config.optimizer == 'lars':
    optim_ops.append(optax.lars(
        learning_rate=learning_rate_fn,
        weight_decay=wd, weight_decay_mask=weight_decay_mask,
        momentum=optimizer_configs.get('momentum', 0.9)))
  elif config.optimizer == 'lamb':
    optim_ops.append(optax.lamb(
        learning_rate=learning_rate_fn,
        weight_decay=wd, mask=weight_decay_mask,
        b1=optimizer_configs.get('beta1', 0.9),
        b2=optimizer_configs.get('beta2', 0.999),
        eps=optimizer_configs.get('epsilon', 1e-8)))

  else:
    logging.info('Unknown optimizer "%s"', config.optimizer)

  return optax.chain(*optim_ops), learning_rate_fn


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
