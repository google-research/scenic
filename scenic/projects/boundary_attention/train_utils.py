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

"""Utils for training."""

import functools
import os
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import flax
from flax import linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.common_lib import debug_utils
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils
from tensorflow.io import gfile


# Single-group reg-exps for int or float numerical substrings.
# captures sign:
_SIGNED_FLOAT_RE = re.compile(
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')

SATURATION_MAGNITUDE = 12


def restore_pretrained_params(
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
    glob_path = os.path.join(checkpoint_path, 'checkpoint*')
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
    restored_params = restored_train_state['params']
    if 'params' in restored_params:
      restored_params = restored_params['params']
    restored_params = flax.core.freeze(restored_params)
  else:
    # restored_train_state was trained using flax.optim. Note that this does
    # not convert the naming of pre-Linen checkpoints.
    restored_params = restored_train_state['optimizer']['target']
    if 'params' in restored_params:  # Backward compatibility.
      restored_params = restored_params['params']
      restored_params = dict(checkpoints.convert_pre_linen(restored_params))
    restored_params = flax.core.freeze(restored_params)

  # restored_model_state = flax.core.freeze(restored_train_state['model_state'])

  if not train_state:
    train_state = train_utils.TrainState()
    params = restored_params
  else:
    # Inspect and compare the parameters of the model with the init-model.
    params = pretrain_utils.inspect_params(
        expected_params=train_state.params,
        restored_params=restored_params,
        fail_if_extra=False,
        fail_if_missing=False,
        fail_if_shapes_mismatch=False)

  params = pretrain_utils._replace_dict(train_state.params, restored_params)  # pylint: disable=protected-access
  train_state = train_state.replace(
      # Inspect and compare the parameters of the model with the init-model.
      params=params,
      # model_state=restored_model_state,
      # global_step=int(restored_train_state['global_step']),
      # rng=restored_train_state['rng'],
      # metadata=restored_train_state.get('metadata', None))
      )
  return train_state


def checkpoint_path_step(path: str) -> Optional[float]:
  """Returns the step number of a checkpoint path.

  Copied from a private method in Flax.

  Args:
    path: Checkpoint file path.

  Returns:
    The step number derived from the filename as a float, or None if it can't be
    determined.
  """
  for s in _SIGNED_FLOAT_RE.split(path)[::-1]:
    if _SIGNED_FLOAT_RE.match(s):
      return float(s)
  return None


def get_num_training_steps(
    config: ml_collections.ConfigDict,
    dataset_metadata: Dict[str, Any]) -> Tuple[int, Optional[int]]:
  """Calculates the total number of training step and possibly steps_per_epoch.

  The main training loop is based on number of training steps. Thus, for
  datasets
  that we want to train based on number of epochs, we need to calculate the
  total number of training steps. This function looks for `num_training_steps`
  in config, if it exists it returns that as the total step and `None` as
  `steps_per_epoch`. If num_training_steps doesn't exist, then it looks for
  `num_training_epochs` and given the size of training data calculates the total
  steps and steps_per_epoch. In this computation, we assume that
  drop_remainder=True.

  Args:
    config: Configuration of the experiment.
    dataset_metadata: Meta-data that is generated by the dataset_builder.

  Returns:
    total_steps: Total number of training steps.
    steps_per_epoch: Number of steps in every epoch.
  """
  num_total_train_examples = dataset_metadata.get('num_train_examples', 0)

  # We either use num_training_epochs or num_training_steps.
  steps_per_epoch = num_total_train_examples // config.batch_size

  if config.get('num_training_steps'):
    assert not config.get('num_training_epochs')
    return config.num_training_steps, steps_per_epoch or None
  else:
    assert config.num_training_epochs and not config.get('num_training_steps')
    return (steps_per_epoch * config.num_training_epochs), steps_per_epoch


def get_grad_weight_schedule_fn(config):
  """Retrieves dataset gradient weight schedules."""

  if config.get('grad_weight_schedules') is None:
    grad_weights = config.get('grad_weights', 1.0)
    return lambda _: grad_weights

  return lr_schedules.lr_fn_dict['compound'](config.grad_weight_schedules)


def initialize_model(
    *,
    model_def: nn.Module,
    input_spec: Sequence[Union[Tuple[Tuple[int, ...], jnp.dtype],
                               Tuple[int, ...], None]],
    config: ml_collections.ConfigDict,
    rngs: Union[jnp.ndarray, Mapping[str, jnp.ndarray]],
):
  """Initializes parameters and model state.

  Args:
    model_def: Definition of a model.
    input_spec: An iterable of (shape, dtype) pairs specifying the shape and
      dtype of the inputs. If unspecified the dtype is float32.
    config: Configurations of the initialization.
    rngs: Jax rng keys.

  Returns:
    Initial params, Init model_state, and number of trainable_params.
  """
  batch_size = (config.batch_size //
                jax.device_count()) if config.get('batch_size') else None
  dummy_input = []
  for spec in input_spec:
    if spec is not None:
      in_st = debug_utils.input_spec_to_jax_shape_dtype_struct(
          spec, batch_size=batch_size)
      dummy_input.append(jnp.zeros(in_st.shape, in_st.dtype))
    else:
      dummy_input.append(None)
  print(type(model_def))
  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def _initialize_model(rngs):
    """Initialization function to be jitted."""
    init_model_state, init_params = model_def.init(
        rngs, *dummy_input, train=False).pop('params')
    # Set bias in the head to low value, such that loss is small initially.
    if config.get('init_head_bias', None) is not None:
      init_params = flax.core.unfreeze(init_params)
      init_params['output_projection'] = optimizers.tree_map_with_names(
          lambda p: jnp.full_like(p, config.init_head_bias),
          init_params['output_projection'],
          match_name_fn=lambda name: 'bias' in name)
      init_params = flax.core.freeze(init_params)
    return init_params, init_model_state

  if not isinstance(rngs, dict):
    rngs = {'params': rngs}
  init_params, init_model_state = _initialize_model(rngs)
  # Pop out params rng:
  rngs.pop('params')

  # Count number of trainable parameters:
  num_trainable_params = debug_utils.log_param_shapes(init_params)

  # Count gflops:
  count_flops = config.get('count_flops',
                           ml_collections.ConfigDict({'count_flops': True}))
  if count_flops:
    variables = {'params': init_params, **init_model_state}
    flops = debug_utils.compute_flops(
        flax_model_apply_fn=functools.partial(
            model_def.apply, variables, train=False, debug=False, rngs=rngs),
        input_spec=count_flops.get('input_spec', input_spec),
        fuse_multiply_add=count_flops.get('fuse_multiply_add', True))
    gflops = flops / (10**9)
  else:
    gflops = None

  return init_params, init_model_state, num_trainable_params, gflops


def convert_from_utf8(val):
  return ''.join([chr(b) for b in np.array(val)])


def partial_mkdir(create_dir):
  try:
    gfile.MakeDirs(create_dir)
  except:  # pylint: disable=bare-except
    pass
