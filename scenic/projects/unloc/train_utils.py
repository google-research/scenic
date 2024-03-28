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

"""Contains UnLoc training utils."""

import collections.abc as collections
import functools
import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
from absl import logging
import flax
import flax.linen as nn
import jax
from jax.example_libraries.optimizers import clip_grads
import jax.numpy as jnp
import ml_collections
import optax
from scenic.common_lib import debug_utils
from scenic.dataset_lib import dataset_utils
from scenic.projects.baselines.clip import model as clip_model
from scenic.projects.unloc import eval_utils as unloc_eval_utils
from scenic.projects.unloc import model_utils as unloc_model_utils
from scenic.projects.vivit import model_utils as vivit_model_utils
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils
import scipy.ndimage

# JAX team is working on type annotation for pytree:
# https://github.com/google/jax/issues/1555
PyTree = Any
Batch = Dict[str, Any]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]
LrFn = Callable[[int], jnp.ndarray]


def create_input_spec(input_shapes: Union[Mapping[str, Any], Tuple[int, ...]],
                      input_dtypes: Union[Mapping[str, Any], jnp.dtype]):
  if isinstance(input_shapes, tuple):
    return (input_shapes, input_dtypes)
  return {
      name: create_input_spec(input_shapes[name], input_dtypes[name])
      for name in input_shapes.keys()
  }


def interpolate_class_embedding(
    class_embedding: jnp.ndarray,
    restored_class_embedding: jnp.ndarray) -> jnp.ndarray:
  """Interpolates class embeddings.

  This function is used to initialize class embeddings for UnLoc models when
  the current model has a different number of class embeddings than the
  pretrained ones.

  Args:
    class_embedding: A 2D float tensor of shape (new_time, channels)
      representing the class embeddings to be updated.
    restored_class_embedding: A 2D float tensor of shape (old_time, channels)
      representing the class embeddings from which we load the weights.

  Returns:
    A 2D float tensor of shape (new_time, channels) representing the resized
    class embeddings.
  """
  logging.info('Resizing class embeddings from %s to %s.',
               restored_class_embedding.shape, class_embedding.shape)
  zoom = (class_embedding.shape[0] / restored_class_embedding.shape[0], 1)
  return scipy.ndimage.zoom(restored_class_embedding, zoom, order=1)


def initialize_from_unloc_parameters(
    params: Dict[str, Any],
    restored_params: Dict[str, Any],
    skip_regex: Optional[str] = None,
) -> Dict[str, Any]:
  """Initialize model parameters from an UnLoc model.

  Args:
    params: The parameters of the model.
    restored_params: Restored parameters from the given pretrained checkpoint.
    skip_regex: Regular expression of parameters to skip loading.

  Returns:
    Initialized parameters of the current model.
  """

  restored_flat = flax.traverse_util.flatten_dict(
      dict(restored_params), keep_empty_nodes=True, sep='/')
  model_flat = flax.traverse_util.flatten_dict(
      dict(params), keep_empty_nodes=True, sep='/')
  logging.info('model_flat keys: %s', list(model_flat.keys()))

  for m_key, m_params in restored_flat.items():
    if m_key not in model_flat:
      logging.warning('%s in checkpoint doesn\'t exist in model. Skip.', m_key)
      continue
    if skip_regex and re.findall(skip_regex, m_key):
      logging.info('Skip loading parameter %s.', m_key)
      continue
    if 'encoder/class_embedding' in m_key:
      if m_params.shape != model_flat[m_key].shape:
        model_flat[m_key] = interpolate_class_embedding(model_flat[m_key],
                                                        m_params)
      else:
        model_flat[m_key] = m_params
    elif 'encoder/VisionTransformer/positional_embedding' in m_key:
      if m_params.shape != model_flat[m_key].shape:
        model_flat[m_key] = vivit_model_utils.interpolate_positional_embeddings(
            m_params, model_flat[m_key].shape[0])[0]
      else:
        model_flat[m_key] = m_params
    elif m_key == 'text_encoder/positional_embedding':
      if m_params.shape != model_flat[m_key].shape:
        cur_len = model_flat[m_key].shape[0]
        pretrain_len = m_params.shape[0]
        if pretrain_len > cur_len:
          model_flat[m_key] = m_params[:cur_len]
        else:
          model_flat[m_key] = jnp.concatenate(
              [m_params, model_flat[m_key][pretrain_len:]], axis=0
          )
        logging.info(
            'Changing shape of %s from %s to %s.',
            m_key,
            m_params.shape,
            model_flat[m_key].shape,
        )
      else:
        model_flat[m_key] = m_params
    elif 'conv1' in m_key:
      # backward compatible to 3D conv implementation.
      if m_params.shape != model_flat[m_key].shape:
        assert len(m_params.shape) == 5 and len(model_flat[m_key].shape) == 2
        model_flat[m_key] = jnp.reshape(m_params, model_flat[m_key].shape)
        logging.info(
            'Changing shape of %s from %s to %s.',
            m_key,
            m_params.shape,
            model_flat[m_key].shape,
        )
      else:
        model_flat[m_key] = m_params
    else:
      logging.info('Loading %s from checkpoint into model', m_key)
      if m_params.shape != model_flat[m_key].shape:
        raise ValueError(
            'Inconsistent shapes between the current model (%s) and the '
            'pretrained model\'s (%s).' %
            (model_flat[m_key].shape, m_params.shape))
      model_flat[m_key] = m_params
  return flax.traverse_util.unflatten_dict(model_flat, sep='/')


def initialize_from_unloc_train_state(
    train_state: train_utils.TrainState,
    restored_train_state: train_utils.TrainState,
    skip_regex: Optional[str] = None,
) -> train_utils.TrainState:
  """Updates UnLoc's train_state with a pretrained UnLoc model weights.

  Args:
    train_state: A raw TrainState for the current model.
    restored_train_state: TrainState of the pretrained model.
    skip_regex: Regular expression of parameters to skip loading.

  Returns:
    Updated train_state.
  """

  params = flax.core.unfreeze(train_state.params)
  restored_params = flax.core.unfreeze(restored_train_state.params)
  params = initialize_from_unloc_parameters(params, restored_params, skip_regex)
  return train_state.replace(params=flax.core.freeze(params))


def init_from_unloc_checkpoint(
    config: ml_collections.ConfigDict,
    train_state: train_utils.TrainState,
) -> train_utils.TrainState:
  """Initialize train state from an UnLoc checkpoint.

  The checkpoint can be specified either by an xid:
  config.init_from.xm = (55208837, 1)
  or a file path:
  config.init_from.checkpoint_path = '/is-d/home/foo/55208837/1'

  Args:
    config: Config points to checkpoint location.
    train_state: TrainState of currement model.

  Returns:
    Updated train_state.
  Raises:
    RuntimeError: if checkpoint is not provided by the config.
  """
  init_checkpoint_path = None
  checkpoint_path = config.init_from.get('checkpoint_path')
  if checkpoint_path is not None:
    init_checkpoint_path = checkpoint_path
  if init_checkpoint_path is None:
    raise RuntimeError('Set either "xm" or a file path to UnLoc checkpoint.')

  restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
      init_checkpoint_path, train_state, assert_exist=True)
  return initialize_from_unloc_train_state(
      train_state,
      restored_train_state,
      skip_regex=config.init_from.get('skip_regex'),
  )


def init_video_encoder_from_clip_checkpoint(
    config: ml_collections.ConfigDict,
    train_state: train_utils.TrainState,
    modality_name: str = 'video',
) -> train_utils.TrainState:
  """Initializes the video encoder with a CLIP model."""
  if config.init_from.get('video_encoder'):
    checkpoint_path = config.init_from.video_encoder.checkpoint_path
  else:
    checkpoint_path = config.init_from.video_encoders.get(
        modality_name
    ).checkpoint_path
  clip_params = clip_model.load_model_vars('', checkpoint_path)
  clip_params = jax.tree_util.tree_map(jnp.float32, clip_params)
  return unloc_model_utils.initialize_from_clip_model(
      config,
      train_state,
      clip_params,
      load_image_tower=True,
      load_text_tower=False,
      video_modality_name=modality_name,
  )


def init_text_encoder_from_clip_checkpoint(
    config: ml_collections.ConfigDict,
    train_state: train_utils.TrainState) -> train_utils.TrainState:
  """Initializes the text encoder with a CLIP model."""
  checkpoint_path = config.init_from.text_encoder.checkpoint_path
  clip_params = clip_model.load_model_vars('', checkpoint_path)
  clip_params = jax.tree_util.tree_map(jnp.float32, clip_params)
  return unloc_model_utils.initialize_from_clip_model(
      config,
      train_state,
      clip_params,
      load_image_tower=False,
      load_text_tower=True,
  )


def init_video_text_encoders_from_clip_checkpoint(
    config: ml_collections.ConfigDict,
    train_state: train_utils.TrainState,
    load_image_tower: bool = True,
    load_text_tower: bool = True) -> train_utils.TrainState:
  """Initializes video+text encoders with a CLIP model."""
  checkpoint_path = config.init_from.checkpoint_path
  clip_params = clip_model.load_model_vars('', checkpoint_path)
  return unloc_model_utils.initialize_from_clip_model(
      config,
      train_state,
      clip_params,
      load_image_tower=load_image_tower,
      load_text_tower=load_text_tower,
  )


VIDEO_ENCODER_INIT_FN = {
    'clip': init_video_encoder_from_clip_checkpoint,
}
TEXT_ENCODER_INIT_FN = {
    'clip': init_text_encoder_from_clip_checkpoint,
}


def initialize_model_with_pytree(
    *,
    model_def: nn.Module,
    input_spec: PyTree,
    config: ml_collections.ConfigDict,
    rngs: Union[jnp.ndarray, Mapping[str, jnp.ndarray]],
) -> Tuple[PyTree, PyTree, int, Optional[float]]:
  """Initializes parameters and model state with a pytree input_spec.

  This function is branched from scenic/train_lib/train_utils.py. Here, the
  model function takes two additional args, `task` and `dataset`.

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
      init_model_state, init_params = flax.core.pop(
          flax.core.freeze(
              model_def.init(
                  rngs,
                  **dummy_input,
                  task=config.dataset_configs.task,
                  dataset=config.dataset_configs.get('name', ''),
                  train=False,
                  debug=False,
              )
          ),
          'params',
      )
    else:
      init_model_state, init_params = flax.core.pop(
          flax.core.freeze(
              model_def.init(
                  rngs,
                  *dummy_input,
                  task=config.dataset_configs.task,
                  dataset=config.dataset_configs.get('name', ''),
                  train=False,
                  debug=False,
              )
          ),
          'params',
      )
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
            model_def.apply,
            variables,
            task=config.dataset_configs.task,
            dataset=config.dataset_configs.get('name', ''),
            train=False,
            debug=False,
            rngs=rngs),
        input_spec=count_flops.get('input_spec', input_spec),
        fuse_multiply_add=count_flops.get('fuse_multiply_add', True))
    gflops = flops / (10**9)
  else:
    gflops = None

  return init_params, init_model_state, num_trainable_params, gflops


def train_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    task: str,
    dataset: str,
    flax_model: nn.Module,
    loss_fn: LossFn,
    lr_fn: LrFn,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False
) -> Tuple[train_utils.TrainState, Dict[str, Tuple[float, int]], Dict[str,
                                                                      Any]]:
  """Runs a single step of training.

  This function is branched from scenic/train_lib/classification_trainer.py.
  Here, the model function takes two additional args, `task` and `dataset`.

  Args:
    train_state: The state of training including the current global_step,
      model_state, rng, params, and optimizer. The buffer of this argument can
      be donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    task: The task name, 'temporal_localization', 'moment_retrieval',
      'highlight_detection' or 'action_segmentation'.
    dataset: The dataset name.
    flax_model: A Flax model.
    loss_fn: A loss function that given logits, a batch, and parameters of the
      model calculates the loss.
    lr_fn: The learning rate fn used for the logging the learning rate.
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    config: Configurations of the experiment.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training and computed metrics and some training logs.
  """

  training_logs = {}
  new_rng, rng = jax.random.split(train_state.rng)

  if config.get('mixup') and config.mixup.alpha:
    mixup_rng, rng = jax.random.split(rng, 2)
    mixup_rng = train_utils.bind_rng_to_host_device(
        mixup_rng,
        axis_name='batch',
        bind_to=config.mixup.get('bind_to', 'device'))
    batch['rgb'] = batch['inputs']['rgb']
    batch = dataset_utils.mixup(
        batch,
        config.mixup.alpha,
        config.mixup.get('image_format', 'NHWC'),
        input_key='rgb',
        rng=mixup_rng)
    batch['inputs']['rgb'] = batch.pop('rgb')

  # Bind the rng to the host/device we are on.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device')

  all_gather_loss = config.get('all_gather_loss', False)
  gathered_batch = (
      unloc_eval_utils.all_gather_metrics_inputs(batch)
      if all_gather_loss
      else None
  )

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    logits = flax_model.apply(
        variables,
        batch['inputs'],
        task=task,
        dataset=dataset,
        train=True,
        rngs={'dropout': dropout_rng},
        debug=debug)
    loss = loss_fn(logits, batch, variables['params'])
    return loss, logits

  def training_loss_fn_all_gather(params):
    variables = {'params': params, **train_state.model_state}
    logits = unloc_eval_utils.run_model_all_gather_results(
        variables,
        batch,
        task,
        flax_model,
        train=True,
        dropout_rng=dropout_rng,
        debug=debug,
    )
    loss = loss_fn(logits, gathered_batch, params)
    return loss, logits

  compute_gradient_fn = jax.value_and_grad(
      training_loss_fn_all_gather if all_gather_loss else training_loss_fn,
      has_aux=True)
  (train_cost, logits), grad = compute_gradient_fn(train_state.params)

  del train_cost
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')

  if config.get('max_grad_norm') is not None:
    grad = clip_grads(grad, config.max_grad_norm)

  assert train_state.tx is not None
  updates, new_opt_state = train_state.tx.update(grad, train_state.opt_state,
                                                 train_state.params)
  new_params = optax.apply_updates(train_state.params, updates)

  training_logs['l2_grads'] = jnp.sqrt(
      sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grad)]))
  ps = jax.tree_util.tree_leaves(new_params)
  training_logs['l2_params'] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
  us = jax.tree_util.tree_leaves(updates)
  training_logs['l2_updates'] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))
  # TODO(dehghani): Can we get this from the optimizer instead?
  training_logs['learning_rate'] = lr_fn(train_state.global_step)

  metrics = metrics_fn(logits, gathered_batch if all_gather_loss else batch)

  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      params=new_params,
      rng=new_rng)

  return new_train_state, metrics, training_logs
