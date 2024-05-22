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

"""Optax utils for Scenic.

This is a fork of
https://github.com/google-research/big_vision/blob/main/big_vision/optax.py.
"""

import itertools
import numbers
import operator
import re
from typing import Any, Optional, Sequence, Tuple, Union, Callable, List

from absl import logging
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers


def find_states(opt_state, cls):
  leaves = jax.tree_util.tree_leaves(
      opt_state, is_leaf=lambda node: isinstance(node, cls))
  return [leaf for leaf in leaves if isinstance(leaf, cls)]


def get_step(opt_state):
  """Returns `ScaleByScheduleState.count` from `opt_state` as an integer."""
  counts = {
      int(state.count)
      for state in find_states(opt_state, optax.ScaleByScheduleState)
  }
  assert len(counts) == 1, f'Expected exactly 1 ScaleByScheduleState: {counts}'
  return next(iter(counts))


def _make_mask_trees(
    params,
    patterns_names_values: Union[Sequence[Tuple[str, str, Any]],
                                 Sequence[Tuple[str, Any]]],
    *,
    allow_unmatched: bool = False,
    log: Optional[str] = None):
  """Wrapper around `make_mask_trees` that supports different input types."""
  if patterns_names_values:
    if len(patterns_names_values[0]) == 3:
      patterns, names, values = zip(*patterns_names_values)
    else:
      patterns, values = zip(*patterns_names_values)
      names = [None] * len(values)
  else:
    patterns, names, values = [], [], []

  masks = make_mask_trees(
      params,
      list(zip(patterns, names, values)),
      allow_unmatched=allow_unmatched,
      log=log,
  )
  return masks, list(zip(names, values))


def _split_frozen(masks, scheds):
  """Computes `frozen_mask` and updates `masks` and `scheds`."""

  def _is_none(sched):
    """Helper to check if sched itself or (fn, base_lr) of sched are None."""
    if isinstance(sched, (tuple, list)):
      _, fn_base_lr = sched  # Check only the tuple of fn and base_lr.
      return not any(fn_base_lr)  # Only false if fn_base_lr = (None, None)
    else:
      return sched is None

  # Specifying `None` as a scheduler freezes params.
  all_false = jax.tree_util.tree_map(lambda *bools: not any(bools), *masks)
  frozen_masks = [
      mask for mask, sched in zip(masks, scheds) if _is_none(sched)]
  frozen_mask = jax.tree_util.tree_map(
      lambda *bools: any(bools), *frozen_masks,
      all_false)  # `all_false` is required when `frozen_masks==[]`.
  masks, scheds = zip(*(
      (mask, sched) for mask, sched in zip(masks,
                                           scheds) if not _is_none(sched)))
  return frozen_mask, masks, scheds


def make_mask_trees(
    tree,
    patterns_names: Sequence[Tuple[str, Optional[str], float]],
    *,
    allow_unmatched: bool = False,
    log: Optional[str] = None,
):
  """Returns a boolean mask tree for every pattern (only first match)."""

  patterns, _, _ = zip(*patterns_names)
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

    if log is not None:
      if any(matches):
        logging.info('%s: %s - matched by %s', log, name,
                     patterns_names[matches.index(True)])
      else:
        logging.info('%s: %s - not matched by any patterns', log, name)
    return np.array(matches)

  multimask = optimizers.tree_map_with_names_values(matchfirst, tree)
  return [
      jax.tree_util.tree_map(lambda matches, i=idx: matches[i], multimask)
      for idx in range(len(patterns))
  ]


def replace_frozen(schedule, pytree, replacement, log: Optional[str] = None):
  """Replaces values matching frozen params in `pytree` with `replacement`."""
  if schedule is None:
    return pytree
  schedule = [(cfg.re, cfg.lr_configs) for name, cfg in schedule.items()]

  masks, scheds = _make_mask_trees(pytree, schedule, log=log)
  frozen_mask, _, _ = _split_frozen(masks, [value for _, value in scheds])
  return jax.tree_util.tree_map(
      lambda v, f: replacement if f else v, pytree, frozen_mask)


def make_schedule(
    schedule: Optional[ml_collections.ConfigDict] = None,
    get_learning_rate_fn: Callable[
        [ml_collections.ConfigDict],
        optax.ScalarOrSchedule] = lr_schedules.get_learning_rate_fn,
) -> List[Tuple[str, str, Tuple[optax.ScalarOrSchedule, float]]]:
  """Creates a schedule dictionary compatible with the `make` function."""
  # Global schedule. No schedule means frozen.
  if schedule is None:
    schedule = ml_collections.ConfigDict(
        {'all': ml_collections.ConfigDict({'re': '(.*)', 'lr_configs': None})})
  schedule = [(cfg.re, name, cfg.lr_configs) for name, cfg in schedule.items()]

  # Create actual schedules funtions.
  def create_schedule(lr_configs):
    if lr_configs is None:
      return None, None  # Parameters are frozen
    fn = get_learning_rate_fn(
        ml_collections.ConfigDict({'lr_configs': lr_configs}))
    # Base LR is used for decoupling WD from LR schedules.
    base_lr = lr_configs.get('base_learning_rate', 1.0)
    return fn, base_lr

  schedule = [(re, name, create_schedule(lr_configs))
              for re, name, lr_configs in schedule]
  return schedule


def make(config: ml_collections.ConfigDict,
         schedule: Sequence[
             Tuple[str, str, Tuple[optax.ScalarOrSchedule, float]]],
         params):
  """Returns gradient transform and learning rate functions.

  Args:
    config: Optimizer config.
    schedule: Learning rate schedules as tuple of regexp, name, learning rate
      schedule function and base learning rate (for WD decoupling).
    params: Model parameters.
  """
  if not config.get('per_example_clipping'):
    # Collect all base_lrs and transform to bool. Each element of schedule fol-
    # lows the structure (re, name, (fn, base_lr)) [see above].
    base_lrs = [fn_base_lr[1] for _, _, fn_base_lr in schedule]
    if any([base_lr == 0 for base_lr in base_lrs]):
      raise ValueError(  # raised if base_lr = 0
          f'`base_learning_rate` contains unsupported values {base_lrs}. If '
          'your intention was to freeze parameters, use Scenic optax and '
          '`config.lr_configs = None` instead.')
  masks, scheds = _make_mask_trees(params, schedule, log='schedule')
  frozen_mask, masks, scheds = _split_frozen(masks, scheds)
  not_frozen_mask = jax.tree_util.tree_map(operator.not_, frozen_mask)
  schedule_fns, schedule_base_lr = zip(
      *[fn_base for _, fn_base in (scheds or [])])
  schedule_txs = [
      optax.masked(optax.scale_by_schedule(schedule_fn), mask)
      for schedule_fn, mask in zip(schedule_fns, masks)
  ] + [
      # Removes weight decay updates. Note that weight decay already has an
      # independent mask (which cannot be combined easily with a second mask),
      # so instead we multiply updates for frozen params with zero.
      optax.masked(optax.set_to_zero(), frozen_mask)
  ]

  # Gradient clipping.
  grad_clip_norm_tx = []
  if config.get('max_grad_norm'):
    if not config.get('per_example_clipping'):
      grad_clip_norm_tx = [
          optax.masked(
              optax.clip_by_global_norm(config.max_grad_norm),
              not_frozen_mask)]
    elif 'optax_grad_pmean' in config:
      if not config.optax_grad_pmean:
        raise ValueError('Per-example gradient aggregateion outside of Optax '
                         'is not supported.')

      # Assume default pmean axis.
      axis_name = 'batch'
      if isinstance(config.optax_grad_pmean, str):
        axis_name = config.optax_grad_pmean

      # Per-example clipping is implemented as differentially private gradients
      # with *zero* noise.
      grad_clip_norm_tx = [
          optax.masked(
              optax.contrib.differentially_private_aggregate(
                  config.max_grad_norm, 0.0, 0),
              not_frozen_mask),
          aggregate_gradients_pmean(axis_name=axis_name)]
    elif 'optax_grad_mean' in config:
      if not config.optax_grad_mean:
        raise ValueError('Per-example gradient aggregation outside of Optax '
                         'is not supported.')
      grad_clip_norm_tx = [
          optax.masked(
              optax.differentially_private_aggregate(
                  config.max_grad_norm, 0.0, 0),
              not_frozen_mask),]
    else:
      raise ValueError(
          'When using per-example clipping, '
          'optimizer.optax_grad_pmean or optimizer.optax_grad_mean must be set.'
      )
  else:
    grad_clip_norm_tx = []

  # Optimizer updates.
  tx_func = operator.attrgetter(config.optax_name)(optax)
  opt_txs = [optax.masked(
      tx_func(**config.get('optax_configs', {})), not_frozen_mask)]

  # Weight decay. Defaults to 0.0.
  # Weight decay is not gradient-based but instead uses "params side-input".
  # Hence, weight decay is additive and independent of previous gradient-based
  # updates.
  assert config.get('weight_decay_decouple', True), (
      'Coupled weight decay not supported anymore.')
  decay_rules = config.get('weight_decay', []) or []
  if isinstance(decay_rules, numbers.Number):
    decay_rules = [('.*kernel.*', decay_rules)]

  if decay_rules:
    decay_masks, mults = _make_mask_trees(
        params, decay_rules,
        allow_unmatched=True, log='config.optimizer.weight_decay')
    mults = [mult for _, mult in mults]  # Remove dummy "name" from the tuples.

    weight_decay_txs = []
    # Create decoupled WD masks by enumerating all schedule x decay mask
    # combinations.
    for (mult, decay_mask), (mask, base_lr) in itertools.product(
        zip(mults, decay_masks), zip(masks, schedule_base_lr)):
      weight_decay_txs.append(
          optax.add_decayed_weights(
              mult / base_lr if base_lr else 0.0,  # Decouple WD from LR.
              jax.tree_util.tree_map(lambda a, b: a and b, decay_mask, mask)))
  else:
    weight_decay_txs = []

  # Combine gradient updates and learning rate schedules.
  opt = optax.chain(
      *grad_clip_norm_tx,
      *opt_txs,
      *weight_decay_txs,
      *schedule_txs,
      optax.scale(-1.0))
  return opt, schedule_fns


def aggregate_gradients_pmean(
    axis_name: str = 'batch',
) -> optax.GradientTransformation:
  """Aggregates gradients using JAX's pmean.

  Args:
    axis_name: Name of the axis for pmean aggregation.

  Returns:
    A `GradientTransformation`.
  """

  def init_fn(params):
    del params
    return None

  def update_fn(updates, state, params=None):
    del params, state
    return jax.lax.pmean(updates, axis_name=axis_name), None

  return optax.GradientTransformation(init_fn, update_fn)

################# Scenic optimizers ##############################
# This is following the BV codebase pattern for defining a custom optimizer.
# A dummy object to allow for foo.bar access syntax, see
# https://stackoverflow.com/a/19476841/2366315
optax.scenic = type('', (), {})()


def scale_by_adafactor(min_dim_size_to_factor=32,
                       decay_rate=0.8, decay_offset=0,
                       beta2_cap=0.999,
                       clipping_threshold=None,
                       momentum=0.9, dtype_momentum=jnp.bfloat16,
                       eps=1e-30):
  """The BigVision variant of Adafactor optimizer."""

  def _decay_rate_pow(i, exponent):
    """Second-order moment decay schedule."""
    t = jnp.array(i, jnp.float32) + 1.0
    return jnp.minimum(beta2_cap, 1.0 - t**(-exponent))

  scale_by_rms = optax.scale_by_factored_rms(
      factored=True,
      decay_rate=decay_rate,
      step_offset=decay_offset,
      min_dim_size_to_factor=min_dim_size_to_factor,
      epsilon=eps,
      decay_rate_fn=_decay_rate_pow)

  clip = (optax.clip_by_block_rms(clipping_threshold) if clipping_threshold
          else optax.identity())

  mom = (optax.ema(momentum, debias=False, accumulator_dtype=dtype_momentum)
         if momentum else optax.identity())

  return optax.chain(scale_by_rms, clip, mom)

optax.scenic.scale_by_adafactor = scale_by_adafactor  # pytype: disable=module-attr


def momentum_hp(momentum=0.9, dtype=jnp.bfloat16, nesterov=False):
  """SGD-Momentum with half-precision accumulator."""
  return optax.trace(decay=momentum, accumulator_dtype=dtype, nesterov=nesterov)


optax.scenic.momentum_hp = momentum_hp  # pytype: disable=module-attr
