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

"""Utility functions for Training."""

import collections.abc as collections
import functools
import os
import time
from typing import Any, Callable, Dict, Tuple, Sequence, Optional, Mapping, Union

from absl import logging
from clu import metric_writers
import flax
from flax import jax_utils
from flax import optim
import flax.linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.common_lib import debug_utils
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.train_lib_deprecated import optimizers
from tensorflow.io import gfile

# JAX team is working on type annotation for pytree:
# https://github.com/google/jax/issues/1555
PyTree = Any
PRNGKey = jnp.ndarray


@flax.struct.dataclass
class TrainState:
  """Dataclass to keep track of state of training.

  The state of training is structured as a flax.struct.dataclass, which enables
  instances of this class to be passed into jax transformations like tree_map
  and pmap.
  """
  global_step: Optional[int] = 0
  optimizer: Optional[optim.Optimizer] = None
  model_state: Optional[Any] = None
  rng: Optional[jnp.ndarray] = None
  accum_train_time: Optional[int] = 0

  def __getitem__(self, item):
    """Make TrainState a subscriptable object."""
    return getattr(self, item)

  def get(self, keyname: str, default: Optional[Any] = None) -> Any:
    """Return the value for key if it exists otherwise the default."""
    try:
      return self[keyname]
    except KeyError:
      return default


def initialize_model(
    *,
    model_def: nn.Module,
    input_spec: Sequence[Union[Tuple[Tuple[int, ...], jnp.dtype],
                               Tuple[int, ...], None]],
    config: ml_collections.ConfigDict,
    rngs: Union[jnp.ndarray, Mapping[str, jnp.ndarray]],
    train: Optional[bool] = False
) -> Tuple[PyTree, PyTree, int, Optional[float]]:
  """Initializes parameters and model state.

  Args:
    model_def: Definition of a model.
    input_spec: An iterable of (shape, dtype) pairs specifying the shape and
      dtype of the inputs. If unspecified the dtype is float32.
    config: Configurations of the initialization.
    rngs: Jax rng keys.
    train: If the scenic model should be initialized in the train mode.

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

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def _initialize_model(rngs):
    """Initialization function to be jitted."""
    init_model_state, init_params = flax.core.pop(model_def.init(
        rngs, *dummy_input, train=train, debug=False), 'params')
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


def initialize_model_with_pytree(
    *,
    model_def: nn.Module,
    input_spec: PyTree,
    config: ml_collections.ConfigDict,
    rngs: Union[jnp.ndarray, Mapping[str, jnp.ndarray]],
) -> Tuple[PyTree, PyTree, int, Optional[float]]:
  """Initializes parameters and model state with a pytree input_spec.

  This is an extension of the above initialize_model function where we can put
  pytree `input_spec`. We keep the original function for backward compatibility.
  If the root type of `input_spec` is `Sequence`, each element is fed to the
  model as position arguments whereas they are fed as keyword arguments if the
  root type is `dict`.

  Args:
    model_def: Definition of a model.
    input_spec: A PyTree whose leaves are (shape, dtype) pairs specifying the
      shape and dtype of the inputs. If unspecified the dtype is float32.
    config: Configurations of the initialization.
    rngs: Jax rng keys.

  Returns:
    Initial params, Init model_state, and number of trainable_params.
  """
  batch_size = (config.batch_size //
                jax.device_count()) if config.get('batch_size') else None

  def check_leaf_spec(spec: Sequence[PyTree]) -> bool:
    return ((len(spec) == 2 and isinstance(spec[0], collections.Sequence) and
             all(isinstance(i, int) for i in spec[0]) and
             isinstance(spec[1], jnp.dtype)) or
            (all(isinstance(i, int) for i in spec[0])))

  def create_dummy_input(spec: PyTree) -> PyTree:
    if isinstance(spec, dict):
      return {k: create_dummy_input(v) for k, v in spec.items()}
    elif isinstance(spec, collections.Sequence):
      if check_leaf_spec(spec):
        in_st = debug_utils.input_spec_to_jax_shape_dtype_struct(
            spec, batch_size=batch_size)
        return jnp.zeros(in_st.shape, in_st.dtype)
      else:
        return tuple(create_dummy_input(child) for child in spec)
    elif spec is None:
      return None
    else:
      raise NotImplementedError('Unsupported spec type.', type(spec))
  dummy_input = create_dummy_input(input_spec)

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def _initialize_model(rngs):
    """Initialization function to be jitted."""
    # If dummy_input is a dict, we feed inputs as keyword arguments, otherwise
    # feed as position arguments.
    if isinstance(dummy_input, dict):
      init_model_state, init_params = flax.core.pop(model_def.init(
          rngs, **dummy_input, train=False, debug=False), 'params')
    else:
      init_model_state, init_params = flax.core.pop(model_def.init(
          rngs, *dummy_input, train=False, debug=False), 'params')
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
    flops = debug_utils.compute_flops_with_pytree(
        flax_model_apply_fn=functools.partial(
            model_def.apply, variables, train=False, debug=False, rngs=rngs),
        input_spec=count_flops.get('input_spec', input_spec),
        fuse_multiply_add=count_flops.get('fuse_multiply_add', True))
    gflops = flops / (10**9)
  else:
    gflops = None

  return init_params, init_model_state, num_trainable_params, gflops


def get_dataset(
    config: ml_collections.ConfigDict,
    data_rng: PRNGKey,
    *,
    dataset_service_address: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_configs: Optional[ml_collections.ConfigDict] = None
) -> dataset_utils.Dataset:
  """Creates dataset.

  By default, the values in the config file are used.
  However, if the optional `dataset_name` and `dataset_configs` are passed,
    those are used instead.

  Args:
    config: The configuration of the experiment.
    data_rng: Random number generator key to use for the dataset.
    dataset_service_address: Used when using the tf.data.experimental.service
    dataset_name: Name of dataset to load, if not reading from the config.
    dataset_configs: Configuration of the dataset, if not reading directly
      from the config.

  Returns:
    A dataset_utils.Dataset object.
  """
  device_count = jax.device_count()
  logging.info('device_count: %d', device_count)
  logging.info('num_hosts : %d', jax.process_count())
  logging.info('host_id : %d', jax.process_index())

  dataset_name = dataset_name or config.dataset_name
  dataset_builder = datasets.get_dataset(dataset_name)

  batch_size = config.batch_size
  if batch_size % device_count > 0:
    raise ValueError(f'Batch size ({batch_size}) must be divisible by the '
                     f'number of devices ({device_count})')

  eval_batch_size = config.get('eval_batch_size', batch_size)
  if eval_batch_size % device_count > 0:
    raise ValueError(f'Eval batch size ({eval_batch_size}) must be divisible '
                     f'by the number of devices ({device_count})')

  local_batch_size = batch_size // jax.process_count()
  eval_local_batch_size = eval_batch_size // jax.process_count()
  device_batch_size = batch_size // device_count
  logging.info('local_batch_size : %d', local_batch_size)
  logging.info('device_batch_size : %d', device_batch_size)

  shuffle_seed = config.get('shuffle_seed', None)
  if dataset_service_address and shuffle_seed is not None:
    raise ValueError('Using dataset service with a random seed causes each '
                     'worker to produce exactly the same data. Add '
                     'config.shuffle_seed = None to your config if you want '
                     'to run with dataset service.')

  dataset_configs = dataset_configs or config.get('dataset_configs')
  dataset = dataset_builder(
      batch_size=local_batch_size,
      eval_batch_size=eval_local_batch_size,
      num_shards=jax.local_device_count(),
      dtype_str=config.data_dtype_str,
      rng=data_rng,
      shuffle_seed=shuffle_seed,
      dataset_configs=dataset_configs,
      dataset_service_address=dataset_service_address)

  return dataset


def initialize_multitask_model(
    *,
    model_def: nn.Module,
    input_spec: Dict[Tuple[Tuple[str, Any], ...],
                     Sequence[Union[Tuple[Tuple[int, ...], jnp.dtype],
                                    Tuple[int, ...]]]],
    config: ml_collections.ConfigDict,
    rngs: Union[jnp.ndarray, Mapping[str, jnp.ndarray]],
) -> Tuple[PyTree, PyTree, int, Optional[Dict[str, float]]]:
  """Initializes parameters and model state.

  Args:
    model_def: Definition of a model.
    input_spec: A dictionary from a dict of keyword arguments to an iterable of
      (shape, dtype) pairs specifying the shape and dtype of the inputs. If
      unspecified the dtype is float32.
    config: Configurations of the initialization.
    rngs: Jax rng keys.

  Returns:
    Initial params, Init model_state, and number of trainable_params.
  """

  def init_fn(model_def):
    for kwargs, in_spec in input_spec.items():

      if config.get('batch_sizes') is not None:
        batch_size = config.batch_sizes.get(dict(kwargs)['dataset'])
      else:
        batch_size = config.batch_size

      batch_size = (batch_size // jax.device_count()) if batch_size else None

      input_shapetype = [
          debug_utils.input_spec_to_jax_shape_dtype_struct(
              spec, batch_size=batch_size) for spec in in_spec
      ]
      dummy_input = []
      for in_st in input_shapetype:
        dummy_input.append(jnp.zeros(in_st.shape, in_st.dtype))
      model_def(
          *dummy_input, train=False, debug=False, **dict(kwargs))

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def _initialize_model(rngs):
    """Initialization function to be jitted."""
    init_model_state, init_params = flax.core.pop(nn.init(
        fn=init_fn, module=model_def)(rngs), 'params')
    # Set bias in the head to low value, such that loss is small initially.
    if (config.get('init_head_bias', None) is not None and
        'output_projection' in init_params):
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
    gflops_dict = {}
    gflops_all = 0
    for kwargs, in_spec in input_spec.items():
      flops = debug_utils.compute_flops(
          flax_model_apply_fn=functools.partial(
              model_def.apply,
              variables,
              train=False,
              debug=False,
              rngs=rngs,
              **dict(kwargs)),
          input_spec=count_flops.get('input_spec', in_spec),
          fuse_multiply_add=count_flops.get('fuse_multiply_add', True))
      gflops = flops / (10**9)
      gflops_key = 'gflops/' + '/'.join(f'{x}={y}' for x, y in kwargs)
      gflops_dict[gflops_key] = gflops
      gflops_all += gflops
    gflops_dict['gflops'] = gflops_all
  else:
    gflops_dict = None

  return init_params, init_model_state, num_trainable_params, gflops_dict


def get_num_training_steps(
    config: ml_collections.ConfigDict,
    dataset_metadata: Dict[str, Any]) -> Tuple[int, Optional[int]]:
  """Calculates the total number of training step and possibly steps_per_epoch.

  The main raining loop is based on number of training steps. Thus, for datasets
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
  # We either use num_training_epochs or num_training_steps.
  steps_per_epoch = dataset_metadata.get('num_train_examples',
                                         0) // config.batch_size

  if config.get('num_training_steps'):
    assert not config.get('num_training_epochs')
    return config.num_training_steps, steps_per_epoch or None
  else:
    assert config.num_training_epochs and not config.get('num_training_steps')
    assert steps_per_epoch > 0, 'num_train_examples should be defined.'
    return int(steps_per_epoch * config.num_training_epochs), steps_per_epoch


@functools.partial(jax.pmap, axis_name='x')
def pmap_mean(x: PyTree) -> PyTree:
  # An axis_name is passed to pmap which can then be used by pmean.
  # In this case each device has its own version of the batch statistics and
  # we average them.
  return jax.lax.pmean(x, 'x')


def sync_model_state_across_replicas(train_state: TrainState) -> TrainState:
  """Sync the model_state (like batch statistics) across replicas.

  Args:
    train_state: TrainState; Current state of training.

  Returns:
    Updated state of training in which model_state is synced across replicas.
  """
  # TODO(dehghani): We simply do "mean" here and this doesn't work with
  #   statistics like variance. (check the discussion in Flax for fixing this).
  if jax.tree_util.tree_leaves(train_state.model_state):
    # If the model_state is not empty.
    new_model_state = train_state.model_state.copy(
        {'batch_stats': pmap_mean(train_state.model_state['batch_stats'])})
    return train_state.replace(  # pytype: disable=attribute-error
        model_state=new_model_state)
  else:
    return train_state


def accumulate_grads_microbatched(
    compute_gradient_fn: Callable[
        [TrainState, Dict[str, jnp.ndarray], jnp.ndarray], Tuple[Any,
                                                                 jnp.ndarray]],
    metrics_fn: Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                         Dict[str, Tuple[float, int]]], train_state: TrainState,
    batch: Dict[str, jnp.ndarray], dropout_rng: jnp.ndarray,
    accum_steps: Optional[int]
) -> Tuple[Optional[jnp.ndarray], Dict[str, Tuple[float, int]], jnp.ndarray]:
  """Accumulate gradients over multiple steps.

  This enables training with larger effective batch sizes.
  Note that currently, gradient accumulation is not supported when the
  `model_state` is in used, e.g., for models that have batch normalization and
  store batch statistics in the `model_state`.

  Args:
    compute_gradient_fn: Gradient function (e.g., `jax.value_and_grad(
      training_loss_fn, ...)
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    train_state: An instance of TrainState that has parameters of the model,
      state of the model, etc.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    dropout_rng: JAX rng key used for dropout.
    accum_steps: Number of accumulating steps (number of micro batches). When
      set to None or =<1, no accumulation is done.

  Returns:
    A tuple of model_state (e.g., batch statistics), calculated metrics, and
      computed gradients.
  """
  params = train_state.optimizer.target
  if accum_steps and accum_steps > 1:
    batch_size = next(iter(batch.values())).shape[0]
    microbatch_size = batch_size // accum_steps
    if batch_size % accum_steps != 0:
      logging.error('Bad accum_steps %d for batch size %d', accum_steps,
                    batch_size)
    logging.info('Using microbatches: %d microbatches, %d size', accum_steps,
                 microbatch_size)

    def get_microbatch(batch: Dict[str, jnp.ndarray],
                       idx: int) -> Dict[str, jnp.ndarray]:
      """Fetch microbatch slice from the given batch."""
      offset = idx * microbatch_size
      length = microbatch_size
      starts = {k: [offset] + [0] * (b.ndim - 1) for k, b in batch.items()}
      limits = {k: [length] + list(b.shape[1:]) for k, b in batch.items()}
      return {
          k: jax.lax.dynamic_slice(b, starts[k], limits[k])
          for k, b in batch.items()
      }

    def per_microbatch_compute_gradient_fn(
        loop_cnt: int, loop_state: Tuple[jnp.ndarray, jnp.ndarray,
                                         Dict[str, Tuple[float, int]]]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Tuple[float, int]]]:
      dropout_rng, grad_accum, metrics_acc = loop_state
      dropout_rng, sub_dropout_rng = jax.random.split(dropout_rng)
      mbatch = get_microbatch(batch, loop_cnt)
      (train_cost,
       (_, mlogits)), grad = compute_gradient_fn(params, mbatch,
                                                 sub_dropout_rng)
      del train_cost
      metrics = metrics_fn(mlogits, mbatch)
      # Accumulate gradients and metrics.
      grad = jax.tree_util.tree_map(jnp.add, grad_accum, grad)
      metrics = jax.tree_util.tree_map(jnp.add, metrics, metrics_acc)
      return dropout_rng, grad, metrics

    # Initialize gradient accumulation loop state.
    dropout_rng, sub_dropout_rng = jax.random.split(dropout_rng)
    init_mbatch = get_microbatch(batch, 0)
    (init_train_cost,
     (model_state,
      init_logits)), grad_init = compute_gradient_fn(params, init_mbatch,
                                                     sub_dropout_rng)
    if jax.tree_util.tree_leaves(model_state):
      # If the model_state is not empty.
      raise ValueError('Gradient accumulation is not supported when the '
                       'model_state is in used (e.g. models w/ batch norm).')

    metrics_init = metrics_fn(init_logits, init_mbatch)
    del init_train_cost, init_logits, init_mbatch

    # Run gradient accumulation loop.
    loop_init = (dropout_rng, grad_init, metrics_init)
    _, grad_acc, metrics_acc = jax.lax.fori_loop(
        1, accum_steps, per_microbatch_compute_gradient_fn, loop_init)
    return model_state, metrics_acc, grad_acc
  else:
    (train_cost, (model_state,
                  logits)), grad = compute_gradient_fn(params, batch,
                                                       dropout_rng)
    del train_cost
    metrics = metrics_fn(logits, batch)
    return model_state, metrics, grad


def save_checkpoint(workdir: str,
                    train_state: TrainState,
                    max_to_keep: int = 3,
                    overwrite: bool = False):
  """Saves a checkpoint.

  First syncs the model state across replicas, then it unreplicates it by taking
  the train state of the first replica and saves it as a checkpoint.

  Args:
    workdir: Experiment directory for saving the checkpoint.
    train_state: An instance of TrainState that holds the state of training.
    max_to_keep: The number of checkpoints to keep.
    overwrite: Overwrite existing checkpoint  if a checkpoint
      at the current or a later step already exits (default: False).
  """
  if jax.process_index() == 0:
    # Get train state from the first replica.
    checkpoint_state = jax.device_get(jax_utils.unreplicate(train_state))
    checkpoints.save_checkpoint(
        workdir,
        checkpoint_state,
        int(checkpoint_state.global_step),
        overwrite=overwrite,
        keep=max_to_keep)


def restore_checkpoint(checkpoint_path: str,
                       train_state: Optional[TrainState] = None,
                       assert_exist: bool = False,
                       step: Optional[int] = None) -> Tuple[TrainState, int]:
  """Restores the last checkpoint.

  First restores the checkpoint, which is an instance of TrainState that holds
  the state of training.

  Args:
    checkpoint_path: Directory to restore the checkpoint.
    train_state: An instance of TrainState that holds the state of
      training.
    assert_exist: Assert that there is at least one checkpoint exists in
      the given path.
    step: Step number to load or None to load latest. If specified,
      checkpoint_path must be a directory.

  Returns:
    training state and an int which is the current step.
  """
  if assert_exist:
    glob_path = os.path.join(checkpoint_path, 'checkpoint_*')
    if not gfile.glob(glob_path):
      raise ValueError('No checkpoint for the pretrained model is found in: '
                       f'{checkpoint_path}')
  if train_state is None:
    raise ValueError('Please use `restore_pretrained_checkpoint` for loading'
                     'a checkpoint without providing a Scenic TrainState.')
  train_state = checkpoints.restore_checkpoint(checkpoint_path, train_state,
                                               step)
  return train_state, int(train_state.global_step)


def bind_rng_to_host_device(rng: jnp.ndarray,
                            axis_name: str,
                            bind_to: Optional[str] = None) -> jnp.ndarray:
  """Binds a rng to the host/device we are on.

  Must be called from within a pmapped function. Note that when binding to
  "device", we also bind the rng to hosts, as we fold_in the rng with axis_index
  which is unique for devices across all hosts.

  Args:
    rng: A jax.random.PRNGKey.
    axis_name: The axis of the devices we are binding rng across.
    bind_to: Must be one of the 'host' or 'device'. None means no binding.

  Returns:
    jax.random.PRNGKey specialized to host/device.
  """
  if bind_to is None:
    return rng
  if bind_to == 'host':
    return jax.random.fold_in(rng, jax.process_index())
  elif bind_to == 'device':
    return jax.random.fold_in(rng, jax.lax.axis_index(axis_name))
  else:
    raise ValueError(
        "`bind_to` should be one of the `[None, 'host', 'device']`")


class TrainingDivergedError(Exception):
  pass


def normalize_metrics_summary(metrics_summary: Dict[str, Tuple[float, int]],
                              split: str) -> Dict[str, float]:
  """Normalize the metrics in summary by its normalizer.

  Args:
    metrics_summary: A dictionary mapping metric name to (value, normalizer).
    split: Split for which we normalize the metrics. Used for logging.

  Returns:
    Normalized metrics summary.

  Raises:
    TrainingDivergedError: Due to observing a NaN in the metrics.
  """
  # TODO(dehghani): Currently we only support metrics of the form 1/N sum
  #   f(x_i). We may need a more general framework for metrics like
  #   precision and recall. Note in particular that while we're normalizing by
  #   the "metric normalization value" that is val[1], this value is previously
  #   summed up and is defined to be an integer.
  normalized_metrics_summary = {}
  for key, val in metrics_summary.items():
    normalized_metrics_summary[key] = val[0] / (val[1] + 1e-9)
    if np.isnan(normalized_metrics_summary[key]):
      raise TrainingDivergedError(
          f'NaN detected in {split}_{key} (Unnormalized values: {val})')

  return normalized_metrics_summary


def stack_forest(forest: PyTree) -> PyTree:
  """Transposes a list of dicts to dict of lists.

  For example,
  given
  [{'a':1,'b':2}, {'a':3,'b':4}],
  the output is:
  {'a': ([1, 3]), 'b': ([2, 4])}

  Args:
    forest: a list of dicts

  Returns:
    a dict of lists.
  """
  if not forest:
    return {}

  stack_args = lambda *args: np.stack(args)
  return jax.tree_util.tree_map(stack_args, *forest)


def unreplicate_and_get(x: Sequence[PyTree]) -> PyTree:
  return jax.device_get(jax_utils.unreplicate(x))


def process_and_fetch_to_host(
    pred_or_tgt: Union[jnp.ndarray, Dict[str,
                                         jnp.ndarray]], batch_mask: jnp.ndarray
) -> Union[Sequence[jnp.ndarray], Dict[str, jnp.ndarray]]:
  """Used to collect predictions and targets of the whole valid/test set.

  Args:
    pred_or_tgt: A jnp-array or dict of arrays, each of shape `[n_dev, bs,
      X,...,Y].
    batch_mask: A nd-array of shape `[nun_devices, bs]`, where zero values
      indicate padded examples.

  Returns:
    A list of length n_dev*bs of items, where each item is a dictionary with
    same keys as `pred_or_tgt` & values are normal np-arrays of shape [X,...,Y].
  """
  def _split_mini_batchs(x):
    # Fetch to host and filter out padded examples.
    x = jax.device_get(x)[np.array(batch_mask).astype(bool)]
    # Split minibatch of examples into a list of examples.
    x_list = jnp.split(x, x.shape[0], axis=0)
    # Squeeze out the dummy dimention.
    return jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), x_list)

  pred_or_tgt = jax.tree_util.tree_map(_split_mini_batchs, pred_or_tgt)

  if isinstance(pred_or_tgt, list):
    # Pred_or_tgt was a single array, so just return the list:
    return pred_or_tgt
  else:
    # Pred_or_tgt was dict of arrays, so convert dict of lists to list of dicts:
    keys, values = zip(*pred_or_tgt.items())
    return [dict(zip(keys, v)) for v in zip(*values)]  # pytype: disable=bad-return-type  # jax-ndarray


def get_backbone_multioptimizer(
    config: ml_collections.ConfigDict) -> optim.MultiOptimizer:
  """Makes a Flax MultiOptimizer with a separate backbone optimizer."""
  other_optim = optimizers.get_optimizer(config)
  if config.get('backbone_training'):
    backbone_optim = optimizers.get_optimizer(config.model.backbone.training)
  else:
    backbone_optim = other_optim
  backbone_traversal = optim.ModelParamTraversal(
      lambda path, param: 'backbone' in path)
  other_traversal = optim.ModelParamTraversal(
      lambda path, param: 'backbone' not in path)
  return optim.MultiOptimizer((backbone_traversal, backbone_optim),
                              (other_traversal, other_optim))


@functools.partial(jax.pmap, axis_name='i')
def _barrier(x):
  return jax.lax.psum(x, axis_name='i')


def barrier():
  """MPI-like barrier."""
  jax.device_get(_barrier(jnp.ones((jax.local_device_count(),))))


def log_eval_summary(step: int,
                     *,
                     writer: metric_writers.MetricWriter,
                     eval_metrics: Sequence[Dict[str, Tuple[float, int]]],
                     extra_eval_summary: Optional[Dict[str, float]] = None,
                     metrics_normalizer_fn: Optional[
                         Callable[[Dict[str, Tuple[float, int]], str],
                                  Dict[str, float]]] = None,
                     prefix: str = 'valid',
                     key_separator: str = '_',
                     flush_writer: bool = True) -> Dict[str, float]:
  """Computes and logs eval metrics.

  Args:
    step: Current step.
    writer: Metric writer object.
    eval_metrics: List of dictionaries of calculated metrics. Usually the
      sequence is the concatenation of the per-eval-step metrics, and every
      dictionary maps a metric name to an array of (value, normalizer) - where
      the array index is usually the batch index.
    extra_eval_summary: A dict containing summaries that are already ready to be
      logged, e.g. global metrics from eval set, like precision/recall.
    metrics_normalizer_fn: Used for normalizing metrics. The api for
      this function is: `new_metrics_dict = metrics_normalizer_fn( metrics_dict,
        split)`. If set to None, we use the `normalize_metrics_summary` which
        uses the normalizer paired with each metric to normalize it (after
        summing both metric and normalizer values).
    prefix: str; Prefix added to the name of the summaries writen by this
      function.
    key_separator: Separator added between the prefix and key.
    flush_writer: If True, flush the writer after logging.

  Returns:
    A dictionary of metrics, mapping both `eval_metrics` and
    `extra_eval_summary` from metric name (incl. `prefix`) to float value.
  """
  eval_metrics = stack_forest(eval_metrics)

  # Compute the sum over all examples in all batches.
  eval_metrics_summary = jax.tree_util.tree_map(lambda x: x.sum(), eval_metrics)
  # Normalize metrics by the total number of exampels.
  metrics_normalizer_fn = metrics_normalizer_fn or normalize_metrics_summary
  eval_metrics_summary = metrics_normalizer_fn(eval_metrics_summary, 'eval')
  # If None, set to an empty dictionary.
  extra_eval_summary = extra_eval_summary or {}

  # Adds extra_eval_summary to the returned eval_summary.
  eval_metrics_summary.update(extra_eval_summary)

  writer.write_scalars(
      step, {
          key_separator.join((prefix, key)): val
          for key, val in eval_metrics_summary.items()
      })

  if flush_writer:
    writer.flush()
  return eval_metrics_summary


def log_train_summary(step: int,
                      *,
                      writer: metric_writers.MetricWriter,
                      train_metrics: Sequence[Dict[str, Tuple[float, int]]],
                      extra_training_logs: Optional[Sequence[Dict[str,
                                                                  Any]]] = None,
                      metrics_normalizer_fn: Optional[
                          Callable[[Dict[str, Tuple[float, int]], str],
                                   Dict[str, float]]] = None,
                      prefix: str = 'train',
                      key_separator: str = '_',
                      flush_writer: bool = True) -> Dict[str, float]:
  """Computes and logs train metrics.

  Args:
    step: Current step.
    writer: Summary writer.
    train_metrics: List of dictionaries of calculated metrics. Usually the
      sequence is the concatenation of the per-eval-step metrics, and every
      dictionary maps a metric name to an array of (value, normalizer) - where
      the array index is usually the batch index.
    extra_training_logs: List of dictionaries, containing additional training
      logs, from every train step, e.g. learning rate, Time, num parameters,
      etc. Their mean will be logged.
    metrics_normalizer_fn: Used for normalizing metrics. The api for
      this function is: `new_metrics_dict = metrics_normalizer_fn( metrics_dict,
        split)`. If set to None, we use the normalize_metrics_summary which uses
        the normzlizer paired with each metric to normalize it.
    prefix: str; Prefix added to the name of the summaries writen by this
      function.
    key_separator: Separator added between the prefix and key.
    flush_writer: If True, flush the writer after logging.

  Returns:
    A dictionary of metrics, mapping `train_metrics from metric name (incl.
    `prefix`) to float value.
  """
  ##### Prepare metrics:
  # Get metrics from devices:
  train_metrics = stack_forest(train_metrics)
  # Compute the sum over all examples in all batches:
  train_metrics_summary = jax.tree_util.tree_map(lambda x: x.sum(),
                                                 train_metrics)
  # Normalize metrics by the total number of exampels:
  metrics_normalizer_fn = metrics_normalizer_fn or normalize_metrics_summary
  train_metrics_summary = metrics_normalizer_fn(train_metrics_summary, 'train')

  ##### Prepare additional training logs:
  # If None, set to an empty dictionary.
  extra_training_logs = extra_training_logs or [{}]
  train_logs = stack_forest(extra_training_logs)

  # Metrics:
  writer.write_scalars(
      step, {
          key_separator.join((prefix, key)): val
          for key, val in train_metrics_summary.items()
      })
  # Additional logs:
  writer.write_scalars(step,
                       {key: val.mean() for key, val in train_logs.items()})

  if flush_writer:
    writer.flush()
  return train_metrics_summary


class Chrono:
  """Measures time and reports progress.

  This class is originally implemented by: Lucas Beyer, Alex Kolesnikov,
  Xiaohua Zhai and other collaborators from Brain ZRH.
  """

  def __init__(self,
               first_step: int,
               total_steps: int,
               steps_per_epoch: int,
               global_bs: int,
               accum_train_time: int = 0,
               example_type='img'):
    self.first_step = first_step
    self.total_steps = total_steps
    self.steps_per_epoch = steps_per_epoch
    self.global_bs = global_bs
    self.accum_train_time = accum_train_time
    self.start_time = None
    self.prev_time = None
    self.prev_step = first_step
    self.pause_start = None
    self.paused_time = 0
    self.warmup = 1  # How many calls to `tick` to skip.
    self.example_type = example_type

  def tick(self, step: int, writer: metric_writers.MetricWriter):
    """A chronometer tick."""
    now = self.pause_start or time.time()

    # Take the start as the first time `tick` is called to avoid measuring the
    # overhead of compilation and don't include it in time estimates.
    if self.warmup:
      self.warmup -= 1
      return
    if None in (self.start_time, self.prev_time):
      self.start_time = self.prev_time = now
      self.prev_step = step
      return

    def hms(s):
      """Format time in hours/minutes/seconds."""
      if s < 60:
        return f'{s:.0f}s'
      m, s = divmod(s, 60)
      if m < 60:
        return f'{m:.0f}m{s:.0f}s'
      h, m = divmod(m, 60)
      return f'{h:.0f}h{m:.0f}m'  # Seconds intentionally omitted.

    # Progress note with 'global' full-program average timings.
    dt = now - self.start_time  # Time since process start.
    steps_done = step - self.first_step
    steps_todo = self.total_steps - step
    note = f'Steps:{step}/{self.total_steps} [{step/self.total_steps:.1%}]'
    note += f'\nETA:{hms(dt / steps_done * steps_todo)}'
    note += f'\nTotal time:{hms(dt / steps_done * self.total_steps)}'

    # Measurement with micro-timings of current training steps speed.
    dt = now - self.prev_time - self.paused_time  # Time between ticks.
    ds = step - self.prev_step  # Steps between ticks.
    ncores = jax.device_count()  # Global device count.

    # Accumulate (integrate) training time.
    self.accum_train_time += dt
    core_hours = self.accum_train_time * ncores / 60 / 60
    devtype = jax.devices()[0].device_kind
    writer.write_scalars(
        step, {
            f'{self.example_type}/sec/core': self.global_bs * ds / dt / ncores,
            f'{self.example_type}/sec': self.global_bs * ds / dt,
            f'core_hours_{devtype}': core_hours,
        })

    self.prev_time = now
    self.prev_step = step
    self.paused_time = 0

  def pause(self):
    assert self.pause_start is None, 'Do not pause twice.'
    if self.start_time:  # Only pause if started.
      self.pause_start = time.time()

  def resume(self):
    if self.pause_start:
      self.paused_time += time.time() - self.pause_start
      self.pause_start = None


def barrier_across_hosts():
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
