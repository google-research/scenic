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

"""Utilities for partitioned train states.

This is useful when only training some variables in the model, and avoids
calculating gradients, and optimiser states, of frozen variables.
"""

from collections import abc
import copy
import operator
import re
from typing import Any, Dict, Optional, Sequence, Tuple

from absl import logging
import flax
from flax import struct
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from scenic.projects.baselines.centernet import optimizer_utils
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils


PyTree = train_utils.PyTree


@struct.dataclass
class PartitionedTrainState:
  """Dataclass to keep track of state of training.

  Parameters are separated into frozen and learned parameters.

  The state of training is structured as a struct.dataclass, which enables
  instances of this class to be passed into jax transformations like tree_map
  and pmap.
  """

  tx: Optional[optax.GradientTransformation] = struct.field(
      default=None, pytree_node=False
  )
  opt_state: Optional[optax.OptState] = None
  params_frozen: Optional[Any] = struct.field(default_factory=dict)
  params_learned: Optional[Any] = struct.field(default_factory=dict)
  global_step: Optional[int] = 0
  model_state: Optional[Any] = struct.field(default_factory=dict)
  rng: Optional[jnp.ndarray] = None
  metadata: Optional[Dict[str, Any]] = None

  def __getitem__(self, item):
    """Make TrainState a subscriptable object."""
    return getattr(self, item)

  def get(self, keyname: str, default: Optional[Any] = None) -> Any:
    """Return the value for key if it exists otherwise the default."""
    try:
      return self[keyname]
    except KeyError:
      return default


def _tree_merge(tree1, tree2):
  for k, v in tree2.items():
    if isinstance(v, dict) and k in tree1 and isinstance(tree1[k], dict):
      tree1[k] = _tree_merge(tree1[k], v)
    else:
      tree1[k] = v
  return tree1


def train_step_partitioned(
    train_state: PartitionedTrainState,
    batch: Any,
    *,
    model: Any,
    loss_and_metrics_fn: Any,
    learning_rate_fn: Any,
    debug: bool = False) -> Tuple[PartitionedTrainState, float, Any, Any]:
  """Training step which only computes gradients wrt. unfrozen parameters.

  Args:
    train_state: Learnable parameters and optimizer states.
    batch: A batch of data containing images ("inputs") and annotations.
    model: The model definition.
    loss_and_metrics_fn: Loss function.
    learning_rate_fn: Learning rate scheduler which given the global_step
      generates the learning rate.
    debug: Enable debug mode or not.

  Returns:
    new_train_state: Updated network parameters and optimizer states.
    lr: The learning rate of the current step (for visualization).
    predictions: The output of the network.
    metrics: Losses and other metrics for visualization.
  """
  def loss_fn(params_to_learn, params_to_freeze):
    new_rng, rng = jax.random.split(train_state.rng, 2)

    model_rng = train_utils.bind_rng_to_host_device(
        rng, axis_name='batch', bind_to='device')

    params = flax.core.unfreeze(params_to_learn)
    _tree_merge(params, flax.core.unfreeze(params_to_freeze))
    # Gradients do not get computed with the following:
    # params = {**train_state.params_learned, **train_state.params_frozen}
    variables = {'params': params, **train_state.model_state}

    predictions, new_model_state = model.train_forward_step(
        model_rng, variables, batch, debug=debug
    )
    loss, metrics = loss_and_metrics_fn(predictions, batch)

    # Adapt to normalization API in log_train_summary
    metrics = {k: (v, 1.) for k, v in metrics.items()}
    return loss, (new_model_state, new_rng, metrics, predictions)

  compute_gradient_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=0)
  (_, aux), grad = compute_gradient_fn(
      train_state.params_learned, train_state.params_frozen)

  new_model_state, new_rng, metrics, predictions = aux
  step = train_state.global_step
  lr = learning_rate_fn(step)

  grad = jax.lax.pmean(grad, axis_name='batch')
  updates, new_opt_state = train_state.tx.update(  # pytype: disable=attribute-error
      grad, train_state.opt_state, train_state.params_learned)
  new_params_learned = optax.apply_updates(train_state.params_learned, updates)
  new_train_state = train_state.replace(
      global_step=step + 1,
      opt_state=new_opt_state,
      params_learned=new_params_learned,
      model_state=new_model_state,
      rng=new_rng)

  # Let's log some gradient norms as well
  def global_l2_norm(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(x)]))

  metrics['l2_grads'] = (global_l2_norm(grad), 1)
  metrics['l2_params'] = (global_l2_norm(new_params_learned), 1)
  metrics['l2_updates'] = (global_l2_norm(updates), 1)

  return new_train_state, lr, predictions, metrics


def train_step_partitioned_jit(
    train_state: PartitionedTrainState,
    batch: Any,
    *,
    tx: optax.GradientTransformation,
    model: Any,
    loss_and_metrics_fn: Any,
    learning_rate_fn: Any,
    debug: bool = False,
) -> Tuple[PartitionedTrainState, float, Any, Any]:
  """Training step which only computes gradients wrt. unfrozen parameters.

  Args:
    train_state: Learnable parameters and optimizer states.
    batch: A batch of data containing images ("inputs") and annotations.
    tx: The optax optimizer transform to use.
    model: The model definition.
    loss_and_metrics_fn: Loss function.
    learning_rate_fn: Learning rate scheduler which given the global_step
      generates the learning rate.
    debug: Enable debug mode or not.

  Returns:
    new_train_state: Updated network parameters and optimizer states.
    lr: The learning rate of the current step (for visualization).
    predictions: The output of the network.
    metrics: Losses and other metrics for visualization.
  """
  new_rng, rng = jax.random.split(train_state.rng, 2)

  def loss_fn(params_to_learn, params_to_freeze):
    params = flax.core.unfreeze(params_to_learn)
    _tree_merge(params, flax.core.unfreeze(params_to_freeze))
    # Gradients do not get computed with the following:
    # params = {**train_state.params_learned, **train_state.params_frozen}
    variables = {'params': params, **train_state.model_state}

    predictions, new_model_state = model.train_forward_step(
        rng, variables, batch, debug=debug
    )
    loss, metrics = loss_and_metrics_fn(predictions, batch)

    # Adapt to normalization API in log_train_summary
    metrics = {k: (v, 1.0) for k, v in metrics.items()}
    return loss, (new_model_state, metrics, predictions)

  compute_gradient_fn = jax.value_and_grad(loss_fn, has_aux=True, argnums=0)
  (_, aux), grad = compute_gradient_fn(
      train_state.params_learned, train_state.params_frozen
  )

  new_model_state, metrics, predictions = aux
  step = train_state.global_step
  lr = learning_rate_fn(step)

  updates, new_opt_state = tx.update(  # pytype: disable=attribute-error
      grad, train_state.opt_state, train_state.params_learned
  )
  new_params_learned = optax.apply_updates(train_state.params_learned, updates)
  new_train_state = train_state.replace(
      global_step=step + 1,
      opt_state=new_opt_state,
      params_learned=new_params_learned,
      model_state=new_model_state,
      rng=new_rng,
  )

  # Let's log some gradient norms as well
  def global_l2_norm(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(x)]))

  metrics['l2_grads'] = (global_l2_norm(grad), 1)
  metrics['l2_params'] = (global_l2_norm(new_params_learned), 1)
  metrics['l2_updates'] = (global_l2_norm(updates), 1)

  return new_train_state, lr, predictions, metrics


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


def _partition_parameters(params: PyTree, frozen_mapping: Dict[str, bool]):
  """Partitions parameters into frozen and learned parameter trees.

  Args:
    params: Pytree of model parameters.
    frozen_mapping: Dictionary mapping parameter name to a boolean indicating
      whether the parameter is to be frozen or not. Assumed that all the keys
      in `params` are within this dictionary.

  Returns:
    (parames_to_learn, params_to_freeze). Both are FrozenDicts of PyTrees in the
    original structure as `params`.
  """

  params_flat = _flatten_params(flax.core.unfreeze(params))
  params_to_learn = {}
  params_to_freeze = {}

  for k, v in params_flat.items():
    if k not in frozen_mapping:
      raise ValueError(f'{k} not in mapping.')
    if frozen_mapping[k]:
      params_to_freeze[k] = v
    else:
      params_to_learn[k] = v

  params_to_learn = flax.traverse_util.unflatten_dict(
      params_to_learn, sep='/')
  params_to_freeze = flax.traverse_util.unflatten_dict(
      params_to_freeze, sep='/')

  return flax.core.freeze(params_to_learn), flax.core.freeze(params_to_freeze)


def create_partitioned_train_state(
    params: PyTree,
    frozen_mapping: Dict[str, bool],
    config: ml_collections.ConfigDict,
    global_step: int,
    model_state: PyTree,
    rng: jnp.ndarray,
    lr_fn: Any) -> Tuple[PartitionedTrainState, int, int]:
  """Creates a partitioned train state given a parameter tree."""

  params_to_learn, params_to_freeze = _partition_parameters(
      params, frozen_mapping)
  if config.optimizer.get('layerwise_decay', -1.) >= 0.0:
    tx = optimizer_utils.optimizer_with_layerwise_decay(
        config, params=params_to_learn)
  elif config.optimizer.get('backbone_multiplier', -1.) >= 0.0:
    tx = optimizer_utils.optimizer_with_backbone_multiplier(
        config, params=params_to_learn)
  else:
    # Avoid modifying original config and allow alteration.
    optimizer_config = copy.deepcopy(config.optimizer).unlock()
    # remove extra keys
    for key in [
        'layerwise_decay',
        'num_layers',
        'decay_layer_prefix',
        'decay_stem_layers',
    ]:
      if key in optimizer_config:
        del optimizer_config[key]
    tx = optimizers.get_optimizer(
        optimizer_config, lr_fn, params=params_to_learn)
  opt_state = tx.init(params_to_learn)

  def num_parameters_from_tree(tree):
    if len(tree):  # pylint: disable=g-explicit-length-test
      return jax.tree_util.tree_reduce(
          operator.add, jax.tree_util.tree_map(lambda x: x.size, tree))
    else:
      return 0

  num_learnable_params = num_parameters_from_tree(params_to_learn)
  num_frozen_params = num_parameters_from_tree(params_to_freeze)
  logging.info('Number of params to learn: %s', num_learnable_params)
  logging.info('Number of params to freeze: %s', num_frozen_params)
  logging.info('Number of params in optimiser state: %s',
               num_parameters_from_tree(opt_state))

  train_state = PartitionedTrainState(
      tx=tx,
      opt_state=opt_state,
      params_frozen=params_to_freeze,
      params_learned=params_to_learn,
      global_step=global_step,
      model_state=model_state,
      rng=rng,
  )

  return train_state, num_learnable_params, num_frozen_params


def convert_to_train_state(
    p_train_state: PartitionedTrainState) -> train_utils.TrainState:
  """Converts a PartitionedTrainState to a normal TrainState.

  The optimizer state is not changed at all. The parameters are simply merged
  together into a single dictionary.

  Args:
    p_train_state: A partitioned train state.

  Returns:
    Regular Scenic train_state object.
  """

  params_learned_flat = _flatten_params(
      flax.core.unfreeze(p_train_state.params_learned)
  )
  params_frozen = _flatten_params(
      flax.core.unfreeze(p_train_state.params_frozen)
  )

  params = params_learned_flat
  params.update(params_frozen)
  params = flax.core.freeze(
      flax.traverse_util.unflatten_dict(params, sep='/')
  )

  return train_utils.TrainState(
      params=params,
      tx=p_train_state.tx,
      opt_state=p_train_state.opt_state,
      global_step=p_train_state.global_step,
      model_state=p_train_state.model_state,
      rng=p_train_state.rng,
      metadata=p_train_state.metadata,
  )


def convert_to_partitioned_train_state(
    train_state: train_utils.TrainState,
    frozen_mapping: Dict[str, bool]) -> PartitionedTrainState:
  """Converts a normal TrainState to a PartitionedTrainState.

  The optimizer state is not changed at all. The parameters are simply split
  into the learned and frozen components.

  Args:
    train_state: Regular Scenic train-state.
    frozen_mapping: Dictionary mapping parameter name to a boolean indicating
      whether the parameter is to be frozen or not. Assumed that all the keys in
      `params` are within this dictionary.

  Returns:
    Partitioned train-state.
  """

  params_to_learn, params_to_freeze = _partition_parameters(
      train_state.params, frozen_mapping)

  return PartitionedTrainState(
      tx=train_state.tx,
      opt_state=train_state.opt_state,
      params_frozen=params_to_freeze,
      params_learned=params_to_learn,
      global_step=train_state.global_step,
      model_state=train_state.model_state,
      rng=train_state.rng,
      metadata=train_state.metadata,
  )


def create_frozen_mask_from_regex(
    param_tree: PyTree,
    patterns_names: Optional[Sequence[Tuple[str, Optional[str]]]],
    *,
    allow_unmatched: bool = True,
    log: bool = True,
) -> Dict[str, bool]:
  """Returns a mapping of parameter names to if they are frozen or not.

  Adapted from: scenic.train_lib.optax._make_mask_trees

  Args:
    param_tree: PyTree of parameters.
    patterns_names: A sequence of tuples. The tuple consists of (regex, name),
      where regex is the pattern used to match if the parameters are frozen or
      not. And name is a description used for debugging purposes.
    allow_unmatched: If true, allows some variables to be unmatched. This should
      be the case when freezing only some variables in the model.
    log: If True, log each match made.

  Returns:
    A list of flattenned parameter names, and a boolean indicating if the
      variable is to be frozen or not.
  """

  patterns, _ = zip(*patterns_names) if patterns_names is not None else ([], [])
  compiled_patterns = list(map(re.compile, patterns))

  def matchfirst(_, name):
    matches = [bool(pattern.fullmatch(name)) for pattern in compiled_patterns]

    matched = sum(map(int, matches))
    matched_patterns = [patterns_names[i] for i, m in enumerate(matches) if m]
    if matched > 1:
      raise ValueError(
          f'{name} matched by multiple patterns: {matched_patterns}')

    if matched == 0 and not allow_unmatched:
      raise ValueError(f'{name} was *not* matched by a single pattern!')

    if log:
      if any(matches):
        logging.info('%s - matched by %s', name,
                     patterns_names[matches.index(True)])
      else:
        logging.info('%s - not matched by any patterns', name)
    return np.array(matches)

  multimask = optimizers.tree_map_with_names_values(matchfirst, param_tree)
  frozen_mask_tree = jax.tree_util.tree_map(any, multimask)
  return _flatten_params(flax.core.unfreeze(frozen_mask_tree))
