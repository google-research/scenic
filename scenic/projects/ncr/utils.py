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

"""Utility functions."""

import os

from typing import Any, Dict, Optional, Tuple

from flax import jax_utils
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
from scenic.train_lib import train_utils
from tensorflow.io import gfile


def save_checkpoint(workdir: str,
                    train_state: train_utils.TrainState,
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


def sync_model_state_across_replicas(
    train_state: train_utils.TrainState) -> train_utils.TrainState:
  """Sync the model_state (like batch statistics) across replicas.

  Args:
    train_state: TrainState; Current state of training.

  Returns:
    Updated state of training in which model_state is synced across replicas.
  """
  #  We simply do "mean" here and this doesn't work with
  #  statistics like variance. (check the discussion in Flax for fixing this).
  if jax.tree_util.tree_leaves(train_state.model_state):
    # If the model_state is not empty.
    new_model_state = train_state.model_state.copy(
        {'batch_stats': train_utils.pmap_mean(
            train_state.model_state['batch_stats'])})
    return train_state.replace(  # pytype: disable=attribute-error
        model_state=new_model_state)
  else:
    return train_state


def restore_checkpoint(checkpoint_path: str,
                       train_state: Optional[train_utils.TrainState] = None,
                       assert_exist: bool = False,
                       step: Optional[int] = None) -> Tuple[
                           train_utils.TrainState, int]:
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


def all_gather(tree: Any):
  """Gather across different hosts and flatten the first two dimensions."""
  gather_flat = lambda x: jnp.concatenate(jax.lax.all_gather(x, 'batch'), 0)
  return jax.tree_util.tree_map(gather_flat, tree)


def mixup(batch: Dict['str', jnp.ndarray],
          alpha: float = 1.0,
          image_format: str = 'NHWC',
          rng: Optional[Any] = None) -> Dict['str', jnp.ndarray]:
  """Mixes images and labels within a single batch.

  Unlike dataset_utils.mixup, this implementation samples a different weight
  for each instance in the mini-batch.

  Args:
    batch: dict; A batch of data with 'inputs' and 'label'.
    alpha: float; Used to control the beta distribution that weight is sampled
      from.
    image_format: string; The format of the input images.
    rng: JAX rng key. If given, JAX numpy will be used as the backend, and if
      None (default value), normal numpy will be used.

  Returns:
    Tuple (mixed_images, mixed_labels).
  """
  images, labels = batch['inputs'], batch['label']
  if labels.shape[-1] == 1:
    raise ValueError('Mixup requires one-hot targets.')
  if 'N' not in image_format:
    raise ValueError('Mixup requires "N" to be in "image_format".')

  batch_size = labels.shape[0]

  # Setup the the numpy backend and prepare mixup weights.
  if rng is None:
    np_backend = np  # Ordinary numpy
    weight = np_backend.random.beta(alpha, alpha, size=(batch_size, 1))
  else:
    np_backend = jnp  # JAX numpy
    weight = jax.random.beta(rng, alpha, alpha, shape=(batch_size, 1))

  # Make sure that the original image has the higher weight during mixup
  weight = np_backend.maximum(weight, 1.0 - weight)

  # Mixup labels.
  batch['label'] = weight * labels + (1.0 - weight) * labels[::-1]

  # Mixup inputs.
  # Shape calculations use np to avoid device memory fragmentation:
  image_weight_shape = np.ones((images.ndim))
  image_weight_shape[image_format.index('N')] = batch_size
  weight = np_backend.reshape(weight,
                              image_weight_shape.astype(np_backend.int32))
  reverse = tuple(
      slice(images.shape[i]) if d != 'N' else slice(-1, None, -1)
      for i, d in enumerate(image_format))
  batch['inputs'] = weight * images + (1.0 - weight) * images[reverse]

  return batch
