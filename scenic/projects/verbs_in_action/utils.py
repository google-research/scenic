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

"""Utilities."""
import os
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from absl import logging
import flax
from flax import struct
from flax.core import frozen_dict
import flax.linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
import optax
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import base_model
from scenic.projects.baselines.clip import model as clip
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils
from tensorflow.io import gfile

PyTree = Union[Mapping[str, Mapping], Any]


def get_vit_clip_config(model_name: str) -> ml_collections.ConfigDict:
  configs = clip.CONFIGS[model_name]
  return ml_collections.ConfigDict({
      'patch_size': configs['vision_patch_size'],
      'features': configs['vision_features'],
      'num_layers': configs['vision_num_layers'],
      'num_heads': configs['vision_features'] // 64,
      'out_features': configs['embed_dim'],
  })


def get_text_clip_config(model_name: str) -> ml_collections.ConfigDict:
  configs = clip.CONFIGS[model_name]
  return ml_collections.ConfigDict({
      'out_features': configs['embed_dim'],
      'vocab_size': configs['vocab_size'],
      'features': configs['text_features'],
      'num_layers': configs['text_num_layers'],
      'num_heads': configs['text_num_heads'],
  })


def init_state(model: base_model.BaseModel, dataset: dataset_utils.Dataset,
               config: ml_collections.ConfigDict, workdir: str,
               rng: jnp.ndarray):
  """Initialize the model state."""
  input_shapes = dataset.meta_data['input_shape']
  input_dtype = dataset.meta_data.get('input_dtype', jnp.float32)
  final_spec_list = [
      (input_shapes['rgb'], input_dtype),
      (dataset.meta_data['text_shape'], dataset.meta_data['text_dtype'])]
  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  params, model_state, num_params, gflops = train_utils.initialize_model(
      model_def=model.flax_model,
      input_spec=final_spec_list,
      config=config,
      rngs=init_rng)
  logging.info('The model has %d params', num_params)
  if gflops is not None:
    logging.info('The model uses %d gflops', gflops)

  chrono = train_utils.Chrono()
  # Create optimizer.
  tx, opt_state = build_optimizer(config, params)
  # Create train state.
  train_state = OptaxTrainState(
      global_step=0,
      opt_state=opt_state,
      tx=tx,
      model_state=model_state,
      weights=params,
      rng=rng,
      metadata={'chrono': chrono.save()},)

  start_step = train_state.global_step
  if config.checkpoint:
    train_state, param_axes = pop_axes_names(
        train_state, axes_name='params_axes')
    train_state, start_step = restore_checkpoint(workdir, train_state)
    train_state = re_add_axis_names(
        train_state, param_axes=param_axes, axes_name='params_axes')
  new_params = None
  if start_step == 0 and not config.get('train_from_scratch'):
    if config.init_from.get('checkpoint_path', None):
      checkpoint_path = config.init_from.checkpoint_path
      logging.info('Loading weights from %s', checkpoint_path)
      new_params = checkpoints.restore_checkpoint(checkpoint_path, None)
      if 'weights' in new_params: new_params = new_params['weights']
    else:
      new_params = load_clip_params(
          train_state.weights, config.model.clip_version,
          config.model.temporal_agg)
  if new_params is not None:
    tx, opt_state = build_optimizer(config, new_params)
    train_state = train_state.replace(tx=tx, opt_state=opt_state,
                                      weights=new_params)
    logging.info('Weights succesfully loaded.')
  elif start_step == 0:
    logging.info('Training completely from scratch. '
                 'Not restoring from any checkpoint.')
  return train_state, start_step, chrono


def load_clip_params(random_params, model_name, temporal_agg):
  """Load CLIP parameters."""
  logging.info('Loading CLIP weights...')
  clip_vars = clip.load_model_vars(model_name)
  clip_params = clip_vars['params']
  params = {
      'text_encoder': dict(TextTower=clip_params['text']),
      'video_encoder':
          dict(Image2VideoEncoder_0=dict(ImageTower=clip_params['visual'])),
  }
  # seqTrans Clip4Clip initializes the transformer temporal aggregation
  # module with weights from CLIP Text tower.
  if temporal_agg == 'transformer':
    clip_text = clip_params['text']['transformer']
    num_layer_transformer = len(random_params['video_encoder']['Image2VideoEncoder_0']['seqTrans_transformer'])  # pylint: disable=line-too-long
    params['video_encoder']['Image2VideoEncoder_0']['seqTrans_transformer'] = {
        'resblocks.' + str(n): clip_text['resblocks.' + str(n)] for n in range(num_layer_transformer) if n < len(clip_text)  # pylint: disable=line-too-long
    }
    number_of_frame = random_params['video_encoder']['Image2VideoEncoder_0']['seqTrans_positional_embedding'].shape[0]  # pylint: disable=line-too-long
    params['video_encoder']['Image2VideoEncoder_0']['seqTrans_positional_embedding'] = clip_params['text']['positional_embedding'][:number_of_frame]  # pylint: disable=line-too-long
  params = jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=jnp.float32),
                                  params)
  return replace_dict(random_params, params, name_mapping={})


def replace_dict(model: PyTree,
                 restored: PyTree,
                 ckpt_prefix_path: Optional[List[str]] = None,
                 model_prefix_path: Optional[List[str]] = None,
                 name_mapping: Optional[Mapping[str, str]] = None,
                 skip_regex: Optional[str] = None) -> PyTree:
  """Replaces values in model dictionary with restored ones from checkpoint."""
  model = flax.core.unfreeze(model)  # pytype: disable=wrong-arg-types
  restored = flax.core.unfreeze(restored)  # pytype: disable=wrong-arg-types

  if ckpt_prefix_path:
    for p in ckpt_prefix_path:
      restored = restored[p]

  if model_prefix_path:
    for p in reversed(model_prefix_path):
      restored = {p: restored}

  # Flatten nested parameters to a dict of str -> tensor. Keys are tuples
  # from the path in the nested dictionary to the specific tensor. E.g.,
  # {'a1': {'b1': t1, 'b2': t2}, 'a2': t3}
  # -> {('a1', 'b1'): t1, ('a1', 'b2'): t2, ('a2',): t3}.
  restored_flat = flax.traverse_util.flatten_dict(
      dict(restored), keep_empty_nodes=True)
  model_flat = flax.traverse_util.flatten_dict(
      dict(model), keep_empty_nodes=True)

  for m_key, m_params in restored_flat.items():
    # pytype: disable=attribute-error
    for name, to_replace in name_mapping.items():
      m_key = tuple(to_replace if k == name else k for k in m_key)
    # pytype: enable=attribute-error
    m_key_str = '/'.join(m_key)
    if m_key not in model_flat:
      logging.warning('%s in checkpoint doesn\'t exist in model. Skip.',
                      m_key_str)
      continue
    if skip_regex and re.findall(skip_regex, m_key_str):
      logging.info('Skip loading parameter %s.', m_key_str)
      continue
    logging.info('Loading %s from checkpoint into model', m_key_str)
    model_flat[m_key] = m_params

  return flax.core.freeze(flax.traverse_util.unflatten_dict(model_flat))


@struct.dataclass
class OptaxTrainState:
  """Dataclass to keep track of state of training.

  The state of training is structured as a flax.struct.dataclass, which enables
  instances of this class to be passed into jax transformations like tree_map
  and pmap.
  """

  tx: Optional[optax.GradientTransformation] = struct.field(pytree_node=False)
  opt_state: Optional[optax.OptState] = None
  weights: Optional[Any] = None
  model_state: Optional[Any] = None
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


def get_optim(optimizer_name: str, learning_rate_fn: Callable[[int], float],
              wd: float):
  """Returns list of operation for optax optimizer."""
  optim_ops = []
  if optimizer_name in ['sgd', 'momentum']:
    if wd:
      optim_ops.append(optax.add_decayed_weights(wd))
    if optimizer_name == 'sgd':
      optim_ops.append(optax.sgd(learning_rate=learning_rate_fn, momentum=0))  # pytype: disable=wrong-arg-types  # numpy-scalars
    else:
      optim_ops.append(optax.sgd(learning_rate=learning_rate_fn, momentum=0.9))  # pytype: disable=wrong-arg-types  # numpy-scalars
  elif optimizer_name == 'adamw':
    optim_ops.append(optax.adamw(learning_rate=learning_rate_fn,  # pytype: disable=wrong-arg-types  # numpy-scalars
                                 weight_decay=wd))
  elif optimizer_name == 'adafactor':
    optim_ops.append(optax.adafactor(learning_rate=learning_rate_fn,  # pytype: disable=wrong-arg-types  # numpy-scalars
                                     multiply_by_parameter_scale=False,
                                     momentum=0.9,
                                     decay_rate=0.999,
                                     weight_decay_rate=wd))
  else:
    logging.info('Unknown optimizer "%s"', optimizer_name)
  return optim_ops


def build_optimizer(config: ml_collections.ConfigDict, params: PyTree):
  """Builds optimizer."""
  # Defaults
  wd = config.get('weight_decay', 0)
  optimizer_name = config.get('optimizer', 'sgd')
  lr_config = config.get('lr_configs',
                         ml_collections.ConfigDict({'factors': ''}))
  optim_ops = get_optim(
      optimizer_name, lr_schedules.compound_lr_scheduler(lr_config), wd)

  if config.get('multi_optim', False):
    # Define the multiple optimizers.
    transforms = {'default': optax.chain(*optim_ops)}
    for sub_optim_name in config.multi_optim_configs.strings:
      sub_optim_ops = get_optim(
          config.get('optimizer_' + sub_optim_name, optimizer_name),
          lr_schedules.compound_lr_scheduler(config.get('lr_' + sub_optim_name,
                                                        lr_config)),
          config.get('weight_decay_' + sub_optim_name, wd),
          )
      transforms[sub_optim_name] = optax.chain(*sub_optim_ops)

    # Combine optimizers.
    def label_param(name):
      for sub_optim_name in config.multi_optim_configs.strings:
        if sub_optim_name in name:
          return sub_optim_name
      return 'default'
    optim_ops = [optax.multi_transform(
        transforms, optimizers.tree_map_with_names_values(
            lambda _, name: label_param(name), params))]

  # Explicit freezing of some parts of the model. Note that freezing can also
  # be done by setting learning rate to 0 with config.multi_optim set to True.
  if config.get('freeze_video_encoder', False):
    # Zero out updates for video encoder parameters.
    mask = optimizers.tree_map_with_names_values(
        lambda _, n: True if 'video_encoder' in n else False, params)
    optim_ops.append(optax.masked(optax.set_to_zero(), mask))

  if config.get('freeze_text_encoder', False):
    # Zero out updates for text encoder parameters.
    mask = optimizers.tree_map_with_names_values(
        lambda _, n: True if 'text_encoder' in n else False, params)
    optim_ops.append(optax.masked(optax.set_to_zero(), mask))

  tx = optax.chain(*optim_ops)
  opt_state = jax.jit(tx.init, backend='cpu')(params)
  return tx, opt_state


def save_checkpoint(workdir: str,
                    train_state: OptaxTrainState,
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
    # Get train state from the first replica.
    checkpoint_state = jax.device_get(train_state)
    checkpoints.save_checkpoint(
        workdir,
        checkpoint_state,
        int(checkpoint_state.global_step),
        overwrite=overwrite,
        keep=max_to_keep,
        keep_every_n_steps=keep_every_n_steps)


def restore_checkpoint(checkpoint_path: str,
                       train_state: Optional[OptaxTrainState] = None,
                       assert_exist: bool = False,
                       step: Optional[int] = None) -> Tuple[
                           OptaxTrainState, int]:
  """Restores the last checkpoint.

  First restores the checkpoint, which is an instance of TrainState that holds
  the state of training.

  Args:
    checkpoint_path: Directory to restore the checkpoint.
    train_state: An instance of OptaxTrainState that holds the state of
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


def pop_axes_names(
    train_state: OptaxTrainState,
    axes_name: str = 'param_axes') -> Tuple[OptaxTrainState, Optional[Any]]:
  """Removes axes_names from model_state for a train state.

  Args:
    train_state: Training state.
    axes_name: the string specifying the name in the model_state

  Returns:
    New train state without axes_names in model_state, axes_names metadata if it
    was removed (so it can be re-added).
  """
  model_state = train_state.model_state
  if axes_name in train_state.model_state:
    model_state, param_axes = frozen_dict.freeze(model_state).pop(axes_name)
    return train_state.replace(model_state=model_state), param_axes
  else:
    return train_state, None


def re_add_axis_names(train_state: OptaxTrainState,
                      param_axes: Any,
                      axes_name: str = 'param_axes') -> OptaxTrainState:
  """Adds axes_names to model_state for a train state.

  Args:
    train_state: Training state.
    param_axes: Model axes metadata to re-add.
    axes_name: the string specifying the name in the model_state

  Returns:
    New train state without axes_names in model_state, axes_names metadata if it
    was removed (so it can be re-added).
  """
  if param_axes:
    model_state = frozen_dict.unfreeze(train_state.model_state)
    model_state[axes_name] = param_axes
    return train_state.replace(model_state=frozen_dict.freeze(model_state))
  else:
    return train_state


def convert_strings_to_uint8_arrays(str_tensor, max_str_len=None):
  """Convert string numpy array into uint8 arrays to transfer to TPUs.

  Given the input string array, outputs a uint8 tensor with an additional
  dimension at the end with the size of max_str_len.

  Args:
    str_tensor: The input string array.
    max_str_len: The maximum number of characters to keep in the converted uint8
      array. If None, it is set to the longest string length in the input array.

  Returns:
    Converted uint8 numpy array with an additional dim of size max_str_len.
  """
  # Make sure that the input str_tensor is an np.ndarray of bytes not of object.
  # An object array stores pointers only whereas a bytes array stores actual
  # string bytes
  str_tensor = np.array(str_tensor, dtype=bytes)
  uint8_tensor = np.frombuffer(str_tensor,
                               np.uint8).reshape(str_tensor.shape + (-1,))
  if max_str_len:
    to_pad = max(0, max_str_len - uint8_tensor.shape[-1])
    uint8_tensor = np.pad(uint8_tensor[..., :max_str_len],
                          [[0, 0]] * str_tensor.ndim + [[0, to_pad]])

  return uint8_tensor


def compute_recall_at_k(video_embeddings,
                        text_embeddings,
                        k_values,
                        suffix='',
                        suffix_separator='_',
                        text_to_video_retrieval=True,):
  """Compute text -> video retrieval recall at K.

  Args:
    video_embeddings: shape [batch_size, d], or list of such shapes
    text_embeddings: shape [batch_size, d]
    k_values: Recall@K computed for different K ranks.
    suffix: Suffix to add to the summary
    suffix_separator: Separator before adding the suffix
    text_to_video_retrieval: Text to video retrieval vs video to text retrieval

  Returns:
    summary: Dictionary containing the recall@K for different K values.
  """
  logits = compute_inner_product(video_embeddings, text_embeddings)
  # Transpose logits from text - video retrieval
  if text_to_video_retrieval:
    logits = np.transpose(logits)
  summary = {}
  if suffix:
    suffix = suffix_separator + suffix
  # K x Batch_size
  matches = np.zeros((len(k_values), logits.shape[0]))

  for i in range(logits.shape[0]):
    logits_text_i = logits[i, :]
    logits_argsorted = np.argsort(logits_text_i, axis=-1)
    inners_indicator = (logits_argsorted == i)
    for j, k in enumerate(k_values):  # Over all captions for video i.
      matches[j, i] = np.mean(np.any(inners_indicator[-k:], axis=-1))
  for matches_for_k, k in zip(matches, k_values):
    summary[f'recall@{k}{suffix}'] = np.mean(matches_for_k)

  return summary


def compute_inners(encoded_video: jnp.ndarray,
                   encoded_text: jnp.ndarray,
                   axis_name: str = 'batch',
                   return_embeddings: bool = False):
  """Computes the inner products between the videos and the text prompts.

  The videos and texts are first gathered across all devices over the given
  axis name (should agree with the one given to the enclosing pmap).
  We then normalize the tensors along the channel dimension.

  Args:
    encoded_video: The encoded videos, shape [batch_size, t, d].
    encoded_text: The encoded text, shape [batch_size, t, n, d]. batch_size is
      the num of samples in the batch, t is the number of test clips, n is the
      number of captions per vid, can include context sentences.
    axis_name: The axis over which the (all-)gather should be done.
    return_embeddings: Whether to return the embeddings or not

  Returns:
    A matrix with shape [global_batch_size, global_batch_size * n], s.t.
    position [i, j] holds the inner product of the i'th video with the j%n-th
    prompt of the j//n-th video.
  """
  encoded_video = jax.lax.all_gather(encoded_video, axis_name)
  encoded_text = jax.lax.all_gather(encoded_text, axis_name)

  # Merge batch_size and t into a single dimension
  encoded_video = encoded_video.reshape((-1,) + encoded_video.shape[2:])
  encoded_text = encoded_text.reshape((-1,) + encoded_text.shape[2:])
  if return_embeddings:
    return compute_inner_product(
        encoded_video, encoded_text), encoded_video, encoded_text
  else:
    return compute_inner_product(encoded_video, encoded_text)


def compute_inner_product(encoded_video, encoded_text):
  """Compute inner product between videos and text embeddings."""
  assert len(encoded_video.shape) == 2  # [batch_size, d]
  logging.info('Shape of encoded text is %s', encoded_text.shape)
  assert len(encoded_text.shape) == 3 or len(encoded_text.shape) == 2

  # Shape is [batch_size, batch_size, n]
  if len(encoded_text.shape) == 3:
    inners = jnp.einsum('nd,mfd -> nmf', encoded_video, encoded_text)
    # Reshaping to [batch_size, batch_size x n]
    return inners.reshape((inners.shape[0], -1))
  return jnp.einsum('nd,md -> nm', encoded_video, encoded_text)


def get_representation(
    train_state: OptaxTrainState,
    video_inputs, text_indices,
    flax_model: nn.Module) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Feeds the inputs to the model and returns their representations."""
  variables = {'params': train_state['weights'], **train_state['model_state']}
  (encoded_video, encoded_text) = flax_model.apply(
      variables,
      video_inputs[0],
      text_indices,
      mutable=False,
      train=False,
      debug=False)
  return encoded_video, encoded_text


def all_gather_and_unreplicate(tensor):
  return flax.jax_utils.unreplicate(
      jax.pmap(lambda x: jax.lax.all_gather(x, 'batch'), 'batch')(tensor))


def convert_uint8_array_to_string(uint8_array):
  return uint8_array.tobytes().rstrip(b'\x00').decode('utf-8')
