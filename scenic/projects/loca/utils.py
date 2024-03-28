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

"""Training utilities."""
import os
from typing import Any, Dict, Tuple, Optional

import flax
from flax import jax_utils
from flax import struct
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import optax
from scenic.dataset_lib import dataset_utils
from tensorflow.io import gfile


@flax.struct.dataclass
class TrainState:
  """Dataclass to keep track of state of training.

  The state of training is structured as a flax.struct.dataclass, which enables
  instances of this class to be passed into jax transformations like tree_map
  and pmap.
  """

  tx: Optional[optax.GradientTransformation] = struct.field(pytree_node=False)
  opt_state: Optional[optax.OptState] = None
  ema_params: Optional[Any] = None
  params: Optional[Any] = None
  state: Optional[Any] = None
  ema_state: Optional[Any] = None
  global_step: Optional[int] = 0
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


def save_checkpoint(workdir: str,
                    train_state: TrainState,
                    max_to_keep: int = 3,
                    overwrite: bool = False,
                    keep_every_n_steps: int = 50000):
  """Saves a checkpoint.

  First syncs the model state across replicas, then it unreplicates it by taking
  the train state of the first replica and saves it as a checkpoint.

  Args:
    workdir: Experiment directory for saving the checkpoint.
    train_state: An instance of TrainState that holds the state of training.
    max_to_keep: The number of checkpoints to keep.
    overwrite: Overwrite existing checkpoint  if a checkpoint
      at the current or a later step already exits (default: False).
    keep_every_n_steps: Keep every checkpoints every n steps.
  """
  if jax.process_index() == 0:
    checkpoint_state = jax.device_get(train_state)
    checkpoints.save_checkpoint(
        workdir,
        checkpoint_state,
        int(checkpoint_state.global_step),
        overwrite=overwrite,
        keep=max_to_keep,
        keep_every_n_steps=keep_every_n_steps)


def restore_checkpoint(checkpoint_path: str,
                       train_state: Optional[TrainState] = None,
                       assert_exist: bool = False,
                       step: Optional[int] = None) -> Tuple[
                           TrainState, int]:
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


def to_cpu(array: jnp.ndarray):
  """Transfers array (replicated on multiple hosts) to a single host.

  Args:
    array: Replicated array of shape
      [num_hosts, num_devices, local_batch_size, ...].

  Returns:
    array of shape [global_batch_size, ...] where
      global_batch_size = num_devices * local_batch_size
  """
  return jax.device_get(dataset_utils.unshard(jax_utils.unreplicate(array)))


def prepare_input(inputs: Dict[str, jnp.ndarray],
                  config: ml_collections.ConfigDict) -> Dict[str, jnp.ndarray]:
  """Prepare the different views for LOCA training."""
  # Reference view.
  batch = dict(reference=inputs['reference'])

  # A bunch of queries.
  n_focal_queries = config.dataset_configs.number_of_focal_queries
  # This one will have "random" dropping.
  batch['query0'] = inputs['query0']
  batch['query0_target_position'] = inputs['query0_mask']
  # Those ones have had "focal" dropping during data processing (i.e. cropping).
  batch['queries'] = jnp.concatenate(
      [inputs['query' + str(i)] for i in range(1, 1 + n_focal_queries)])
  target_pos = jnp.concatenate([inputs[
      'query' + str(i) + '_mask'] for i in range(1, 1 + n_focal_queries)])
  batch['target_positions'] = target_pos.reshape(target_pos.shape[0], -1)
  return batch


def sinkhorn(x, num_itr=3, distributed=True):
  """Sinkhorn-Knopp algorithm."""
  for _ in range(num_itr):
    # Total weight per prototype per device.
    weight_per_proto = jnp.sum(x, axis=0, keepdims=True)
    if distributed:
      # Globally.
      weight_per_proto = jax.lax.psum(weight_per_proto, axis_name='batch')
    x /= weight_per_proto

    # Total weight per sample.
    weight_per_sample = jnp.sum(x, axis=-1, keepdims=True)
    # x sums to 1 for each sample (it is an assignment).
    x /= weight_per_sample
  return x
