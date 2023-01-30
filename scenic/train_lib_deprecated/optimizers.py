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

"""Defines different optimizers.

Many of these were originally implemented by: Lucas Beyer, Alex Kolesnikov,
Xiaohua Zhai and other collaborators from Brain ZRH.
"""

import dataclasses
from typing import Any, Callable, Generator, List, Tuple

import flax
from flax import optim as optimizers
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

# JAX team is working type checking for pytrees:
# https://github.com/google/jax/issues/3340
PyTree = Any


def get_optimizer(config: ml_collections.ConfigDict) -> optimizers.OptimizerDef:
  """Constructs  the optimizer from the given HParams.

  Args:
    config: Configurations of the optimizer.

  Returns:
    A flax optimizer.
  """
  if config.optimizer == 'sgd':
    return optimizers.GradientDescent(
        learning_rate=config.lr_configs['base_learning_rate'])
  if config.optimizer == 'nesterov':
    return optimizers.Momentum(
        learning_rate=config.lr_configs['base_learning_rate'],
        beta=config.optimizer_configs.get('momentum', 0.9),
        weight_decay=config.optimizer_configs.get('weight_decay', 0.0),
        nesterov=True)
  if config.optimizer == 'momentum':
    return optimizers.Momentum(
        learning_rate=config.lr_configs['base_learning_rate'],
        beta=config.optimizer_configs.get('momentum', 0.9),
        weight_decay=config.optimizer_configs.get('weight_decay', 0.0),
        nesterov=False)
  if config.optimizer == 'momentum_hp':
    return MomentumHP(
        learning_rate=config.lr_configs['base_learning_rate'],
        beta=config.optimizer_configs.get('momentum', 0.9))
  if config.optimizer == 'adam':
    return optimizers.Adam(
        learning_rate=config.lr_configs['base_learning_rate'],
        beta1=config.optimizer_configs.get('beta1', 0.9),
        beta2=config.optimizer_configs.get('beta2', 0.999),
        eps=config.optimizer_configs.get('epsilon', 1e-8),
        weight_decay=config.optimizer_configs.get('weight_decay', 0.0),
    )
  if config.optimizer == 'lars':
    return optimizers.LARS(
        learning_rate=config.lr_configs['base_learning_rate'],
        beta=config.optimizer_configs.get('momentum', 0.9),
        weight_decay=config.optimizer_configs.get('weight_decay', 0.0))
  if config.optimizer == 'lamb':
    return optimizers.LAMB(
        learning_rate=config.lr_configs['base_learning_rate'],
        beta1=config.optimizer_configs.get('beta1', 0.9),
        beta2=config.optimizer_configs.get('beta2', 0.999),
        eps=config.optimizer_configs.get('epsilon', 1e-8),
        weight_decay=config.optimizer_configs.get('weight_decay', 0.0))
  if config.optimizer == 'adabelief':
    return optimizers.AdaBelief(
        learning_rate=config.lr_configs['base_learning_rate'],
        beta1=config.optimizer_configs.get('beta1', 0.9),
        beta2=config.optimizer_configs.get('beta2', 0.999),
    )
  else:
    raise NotImplementedError('Optimizer {} not implemented'.format(
        config.optimizer))


class MomentumHP(flax.optim.OptimizerDef):
  """Momentum optimizer that stores state using half-precision."""

  @flax.struct.dataclass
  class HyperParams:
    learning_rate: np.ndarray
    beta: np.ndarray

  @flax.struct.dataclass
  class State:
    momentum: np.ndarray

  def __init__(self, learning_rate: float, beta: float = 0.9):
    hyper_params = MomentumHP.HyperParams(
        np.array(learning_rate), np.array(beta))
    super().__init__(hyper_params)

  def init_param_state(self, param: PyTree) -> optimizers.OptimizerState:
    return MomentumHP.State(jnp.zeros_like(param, dtype=jnp.bfloat16))

  def apply_gradient(self, hyper_params: HyperParams, params: PyTree,
                     state: optimizers.OptimizerState,
                     grads: PyTree) -> Tuple[PyTree, optimizers.OptimizerState]:
    step = state.step
    names_and_params_flat, treedef = tree_flatten_with_names(params)
    names_flat, params_flat = zip(*names_and_params_flat)
    states_flat = treedef.flatten_up_to(state.param_states)
    grads_flat = treedef.flatten_up_to(grads)

    out = [
        self.apply_param_gradient(step, hyper_params, name, param, state, grad)
        for name, param, state, grad in zip(names_flat, params_flat,
                                            states_flat, grads_flat)
    ]

    new_params_flat, new_states_flat = list(zip(*out)) if out else ((), ())
    new_params = jax.tree_unflatten(treedef, new_params_flat)
    new_param_states = jax.tree_unflatten(treedef, new_states_flat)
    new_state = optimizers.OptimizerState(step + 1, new_param_states)
    return new_params, new_state

  def apply_param_gradient(
      self, step: int, hyper_params: HyperParams, name: str, param: PyTree,
      state: optimizers.OptimizerState,
      grad: PyTree) -> Tuple[PyTree, optimizers.OptimizerState]:
    del step
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    momentum = state.momentum
    new_momentum = hyper_params.beta * momentum + grad
    new_param = param - hyper_params.learning_rate * new_momentum
    new_state = MomentumHP.State(new_momentum.astype(jnp.bfloat16))
    return new_param, new_state


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
  vals = [f(v, name) if match_name_fn(name) else v
          for name, v in names_and_vals]
  return tree_def.unflatten(vals)


def decay_weight_fn(w: jnp.ndarray,
                    lr: float,
                    decay: float = 1e-3) -> jnp.ndarray:
  return (1.0 - lr * decay) * w if decay else w
