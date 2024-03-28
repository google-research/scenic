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

"""Utility functions for training."""

import functools

from flax.training import checkpoints
from flax.training import train_state
import jax
from scenic.projects.boundary_attention.models import boundary_attention


def make_apply(config, ckpt_dir):
  """Make jitted apply function and load trained params."""

  model = boundary_attention.BoundaryAttention(config=config,
                                               dataset_metadata={},
                                               ).build_flax_model()

  apply_jitted = jax.jit(functools.partial(model.apply,
                                           train=False))

  # Load Parameters
  trained_params = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir,
                                                  target=None)['params']

  return apply_jitted, {'params': trained_params}


class TrainState(train_state.TrainState):
  key: jax.Array


def initialize_model_params(model, batch, params_key):
  """Initialize model parameters."""
  params = jax.jit(functools.partial(model.init, train=False))(
      params_key, **batch
  )

  param_count = sum(p.size for p in jax.tree_util.tree_flatten(params)[0])
  print('Total Number of Learnable Parameters=', param_count)

  return params


def count_model_params(params):
  """Count number of parameters."""
  param_count = sum(p.size for p in jax.tree_util.tree_flatten(params)[0])
  print('Total Number of Learnable Parameters=', param_count)

  return param_count


def load_saved_state(ckpt_dir, state, what_use, step_use=-1):
  """Load model parameters."""

  if what_use == 'xm':
    if step_use == -1:
      temp = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)
    else:
      temp = checkpoints.restore_checkpoint(
          ckpt_dir=ckpt_dir, target=None, step=step_use
      )

    state = state.replace(params={'params': temp['params']})

    return state, temp

  elif what_use == 'flax':
    if step_use == -1:
      state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)
    else:
      state = checkpoints.restore_checkpoint(
          ckpt_dir=ckpt_dir, target=state, step=step_use
      )

    return state

  else:
    if step_use == -1:
      state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=state)
    else:
      state = checkpoints.restore_checkpoint(
          ckpt_dir=ckpt_dir, target=state, step=step_use
      )

    return state
