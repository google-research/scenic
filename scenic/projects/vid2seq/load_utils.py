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

"""Utils for loading weights for Vid2Seq."""

import copy
import os
import re
from typing import List, Mapping, Optional, Union, Any

from absl import logging
import flax

from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
from scenic.common_lib import debug_utils
from scenic.train_lib_deprecated.pretrain_utils import get_params_and_model_state_dict
from scenic.train_lib_deprecated.pretrain_utils import inspect_params
from scenic.train_lib_deprecated.train_utils import TrainState
from tensorflow.io import gfile

PyTree = Union[Mapping[str, Mapping], Any]


def init_from_pretrain_weights(
    train_state: TrainState,
    restored_params: PyTree,
    ckpt_prefix_path: Optional[List[str]] = None,
    model_prefix_path: Optional[List[str]] = None,
    name_mapping: Optional[Mapping[str, str]] = None,
    skip_regex: Optional[str] = None) -> TrainState:
  """Updates the train_state with pretrained weights.

  Args:
    train_state: A raw TrainState for the model.
    restored_params: Loaded parameters.
    ckpt_prefix_path: Prefix to restored model parameters.
    model_prefix_path: Prefix to the parameters to replace in the subtree model.
    name_mapping: Mapping from parameter names of checkpoint to this model.
    skip_regex: If there is a parameter whose parent keys match the regex, the
      parameter will not be replaced from pretrain_state.

  Returns:
    Updated train_state.
  """
  name_mapping = name_mapping or {}
  model_params = train_state.optimizer.target
  logging.info(
      'model_params: %s',
      jax.tree_util.tree_map(
          lambda x: x.shape, flax.core.unfreeze(model_params)
      ),
  )
  logging.info(
      'restored_params: %s',
      jax.tree_util.tree_map(lambda x: x.shape, restored_params),
  )
  model_params = replace_dict(model_params, restored_params, ckpt_prefix_path,
                              model_prefix_path, name_mapping, skip_regex, True)
  new_optimizer = train_state.optimizer.replace(target=model_params)
  train_state = train_state.replace(  # pytype: disable=attribute-error
      optimizer=new_optimizer)
  return train_state


def replace_dict(model: PyTree,
                 restored: PyTree,
                 ckpt_prefix_path: Optional[List[str]] = None,
                 model_prefix_path: Optional[List[str]] = None,
                 name_mapping: Optional[Mapping[str, str]] = None,
                 skip_regex: Optional[str] = None,
                 compat: bool = False) -> PyTree:
  """Restores checkpoint in model dictionary, compatible with additional timestamp tokens."""

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
      if ('tmp_' + m_key[0], *m_key[1:]) in model_flat:
        m_key = ('tmp_' + m_key[0], *m_key[1:])
      else:
        logging.warning('%s in checkpoint doesn\'t exist in model. Skip.',
                        m_key_str)
        continue
    if skip_regex and re.findall(skip_regex, m_key_str):
      logging.info('Skip loading parameter %s.', m_key_str)
      continue
    logging.info('Loading %s from checkpoint into model', m_key_str)
    # shared_decoder_token_embedder/embedding 32128 768
    if compat and m_params.shape[0] < model_flat[m_key].shape[0]:
      logging.info(
          'Loading everything but the last axis 0 %d values',
          (model_flat[m_key].shape[0] - m_params.shape[0])
      )
      model_flat[m_key] = jnp.concatenate([
          m_params,
          model_flat[m_key][-(model_flat[m_key].shape[0] - m_params.shape[0]):]
      ],
                                          axis=0)
    # text_decoder/decoder_module/logits_dense/kernel 768 32128
    elif compat and m_params.shape[0] > model_flat[m_key].shape[0]:  # tmp only
      continue
    elif compat and len(
        m_params.shape) == 2 and m_params.shape[1] < model_flat[m_key].shape[1]:
      logging.info(
          'Loading everything but the last axis 1 %d values',
          (model_flat[m_key].shape[1] - m_params.shape[1])
      )
      model_flat[m_key] = jnp.concatenate([
          m_params,
          model_flat[m_key][:,
                            -(model_flat[m_key].shape[1] - m_params.shape[1]):]
      ],
                                          axis=1)
    elif compat and len(
        m_params.shape) == 2 and m_params.shape[1] > model_flat[m_key].shape[1]:
      continue  # tmp only
    else:
      model_flat[m_key] = m_params

  return flax.core.freeze(flax.traverse_util.unflatten_dict(model_flat))


def restore_pretrained_checkpoint(
    checkpoint_path: str,
    train_state: Optional[TrainState] = None,
    assert_exist: bool = False,
    step: Optional[int] = None) -> TrainState:
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
    glob_path = os.path.join(checkpoint_path, 'checkpoint_*')
    if not gfile.glob(glob_path):
      raise ValueError('No checkpoint for the pretrained model is found in: '
                       f'{checkpoint_path}')
  restored_train_state = checkpoints.restore_checkpoint(checkpoint_path, None,
                                                        step)

  # no bins at PT case
  if train_state.optimizer.target['text_decoder']['decoder_module'][  # pytype: disable=attribute-error
      'logits_dense']['kernel'].shape > restored_train_state['optimizer'][
          'target']['text_decoder']['decoder_module']['logits_dense'][
              'kernel'].shape:
    x = restored_train_state['optimizer']['target']['text_decoder'][
        'decoder_module']['logits_dense']['kernel']
    x2 = train_state.optimizer.target['text_decoder']['decoder_module'][  # pytype: disable=attribute-error
        'logits_dense']['kernel']
    restored_train_state['optimizer']['target']['text_decoder'][
        'decoder_module']['logits_dense']['kernel'] = jnp.concatenate(
            [x, x2[:, x.shape[1]:]], 1)
    y = restored_train_state['optimizer']['state']['param_states'][
        'text_decoder']['decoder_module']['logits_dense']['kernel']
    y2 = train_state.optimizer.state.param_states['text_decoder'][  # pytype: disable=attribute-error
        'decoder_module']['logits_dense']['kernel']
    restored_train_state['optimizer']['state']['param_states']['text_decoder'][
        'decoder_module']['logits_dense']['kernel'][
            'grad_ema'] = jnp.concatenate(
                [y['grad_ema'], y2.grad_ema[:, y['grad_ema'].shape[1]:]], 1)
    restored_train_state['optimizer']['state']['param_states']['text_decoder'][
        'decoder_module']['logits_dense']['kernel'][
            'grad_sq_ema'] = jnp.concatenate([
                y['grad_sq_ema'], y2.grad_sq_ema[:, y['grad_sq_ema'].shape[1]:]
            ], 1)
  # no bins at FT case
  elif train_state.optimizer.target['text_decoder']['decoder_module'][  # pytype: disable=attribute-error
      'logits_dense']['kernel'].shape < restored_train_state['optimizer'][
          'target']['text_decoder']['decoder_module']['logits_dense'][
              'kernel'].shape:
    x = restored_train_state['optimizer']['target']['text_decoder'][
        'decoder_module']['logits_dense']['kernel']
    x2 = train_state.optimizer.target['text_decoder']['decoder_module'][  # pytype: disable=attribute-error
        'logits_dense']['kernel']
    restored_train_state['optimizer']['target']['text_decoder'][
        'decoder_module']['logits_dense']['kernel'] = x[:, :x2.shape[1]]
    y = restored_train_state['optimizer']['state']['param_states'][
        'text_decoder']['decoder_module']['logits_dense']['kernel']
    y2 = train_state.optimizer.state.param_states['text_decoder'][  # pytype: disable=attribute-error
        'decoder_module']['logits_dense']['kernel']
    restored_train_state['optimizer']['state']['param_states']['text_decoder'][
        'decoder_module']['logits_dense']['kernel'][
            'grad_ema'] = y['grad_ema'][:, :y2.grad_ema.shape[1]]
    restored_train_state['optimizer']['state']['param_states']['text_decoder'][
        'decoder_module']['logits_dense']['kernel'][
            'grad_sq_ema'] = y['grad_sq_ema'][:, :y2.grad_sq_ema.shape[1]]

  # no bins at PT case
  if restored_train_state['optimizer']['target'][
      'shared_decoder_token_embedder'][
          'embedding'].shape < train_state.optimizer.target[  # pytype: disable=attribute-error
              'shared_decoder_token_embedder']['embedding'].shape:
    x = copy.deepcopy(
        restored_train_state['optimizer']['state']['param_states']
        ['shared_decoder_token_embedder']['embedding'])
    x2 = copy.deepcopy(
        train_state.optimizer.state  # pytype: disable=attribute-error
        .param_states['shared_decoder_token_embedder']['embedding'])
    restored_train_state['optimizer']['state']['param_states'][
        'shared_decoder_token_embedder'] = {
            'embedding': {
                'grad_ema':
                    jnp.concatenate([
                        x['grad_ema'], x2.grad_ema[x['grad_ema'].shape[0]:]
                    ], 0),
                'grad_sq_ema':
                    jnp.concatenate([
                        x['grad_sq_ema'],
                        x2.grad_sq_ema[x['grad_sq_ema'].shape[0]:]
                    ], 0)
            }
        }
    y = copy.deepcopy(restored_train_state['optimizer']['target']
                      ['shared_decoder_token_embedder']['embedding'])
    y2 = copy.deepcopy(train_state.optimizer.target  # pytype: disable=attribute-error
                       ['shared_decoder_token_embedder']['embedding'])
    restored_train_state['optimizer']['target'][
        'shared_decoder_token_embedder'] = {
            'embedding': jnp.concatenate([y, y2[y.shape[0]:]], 0)
        }
  # no bins at FT case
  elif restored_train_state['optimizer']['target'][
      'shared_decoder_token_embedder'][
          'embedding'].shape > train_state.optimizer.target[  # pytype: disable=attribute-error
              'shared_decoder_token_embedder']['embedding'].shape:
    x = copy.deepcopy(
        restored_train_state['optimizer']['state']['param_states']
        ['shared_decoder_token_embedder']['embedding'])
    x2 = copy.deepcopy(
        train_state.optimizer.state  # pytype: disable=attribute-error
        .param_states['shared_decoder_token_embedder']['embedding'])
    restored_train_state['optimizer']['state']['param_states'][
        'shared_decoder_token_embedder'] = {
            'embedding': {
                'grad_ema': x['grad_ema'][:x2.grad_ema.shape[0]],
                'grad_sq_ema': x['grad_sq_ema'][:x2.grad_sq_ema.shape[0]],
            }
        }
    y = copy.deepcopy(restored_train_state['optimizer']['target']
                      ['shared_decoder_token_embedder']['embedding'])
    y2 = copy.deepcopy(train_state.optimizer.target  # pytype: disable=attribute-error
                       ['shared_decoder_token_embedder']['embedding'])
    restored_train_state['optimizer']['target'][
        'shared_decoder_token_embedder'] = {
            'embedding': y[:y2.shape[0]]
        }

  if restored_train_state is None:
    raise ValueError('No checkpoint for the pretrained model is found in: '
                     f'{checkpoint_path}')
  (restored_params,
   restored_model_state) = get_params_and_model_state_dict(restored_train_state)
  restored_params = flax.core.freeze(restored_params)
  restored_model_state = flax.core.freeze(restored_model_state)
  if train_state:
    new_train_state = train_state
    new_optimizer = train_state.optimizer.replace(
        # Inspect and compare the parameters of the model with the init-model.
        target=inspect_params(
            expected_params=train_state.optimizer.target,
            restored_params=restored_params,
            fail_if_extra=False,
            fail_if_missing=False,
            fail_if_shapes_mismatch=False))
  else:
    new_train_state = TrainState()
    new_optimizer = {'target': restored_params}

  new_train_state = new_train_state.replace(  # pytype: disable=attribute-error
      optimizer=new_optimizer,
      model_state=restored_model_state,
      global_step=int(restored_train_state['global_step']),
      rng=restored_train_state['rng'],
      accum_train_time=restored_train_state.get('accum_train_time', 0))

  return new_train_state


def initialise_from_train_state(
    config,
    train_state: Any,
    restored_train_state: Any,
    restored_model_cfg: ml_collections.ConfigDict,
    restore_output_proj: bool,
    log_initialised_param_shapes: bool = True,
    one_config: bool = True) -> Any:
  """Updates the train_state with data from restored_train_state.

  This function is written to be used for 'fine-tuning' experiments. Here, we
  do some surgery to support larger resolutions (longer sequence length) in
  the transformer block, with respect to the learned pos-embeddings.

  Args:
    config: Configurations for the model being updated, or tuple of configs.
    train_state: A raw TrainState for the model.
    restored_train_state: A TrainState that is loaded with parameters/state of a
      pretrained model.
    restored_model_cfg: Configuration of the model from which the
      restored_train_state come from. Usually used for some asserts.
    restore_output_proj: If true, load the final output projection. Set
      to False if finetuning to a new dataset.
    log_initialised_param_shapes: If true, print tabular summary of all the
      variables in the model once they have been initialised.
    one_config: If true, we have only a single config. If false, we get a tuple
      of configs in the order [init_config, model_config, dataset_config]. This
      is useful for works that build upon MBT and have different models in their
      config.

  Returns:
    Updated train_state.
  """
  # Split up configs
  if one_config:
    init_config = config.init_from
    model_config = config.model
  else:
    init_config, model_config = config

  # Inspect and compare the parameters of the model with the init-model
  params = flax.core.unfreeze(train_state.optimizer.target)

  if init_config.get('checkpoint_format', 'scenic') == 'bigvision':
    restored_params = restored_train_state.optimizer['target']
  else:
    restored_params = restored_train_state.optimizer.target
  restored_params = flax.core.unfreeze(restored_params)
  for m_key, m_params in restored_params.items():
    if m_key == 'output_projection':
      if restore_output_proj:
        params[m_key] = m_params
      else:
        pass
    elif m_key == 'pre_logits':
      if model_config.representation_size is None:
        # We don't have representation_size in the new model, so let's ignore
        #   if from the pretained model, in case it has it.
        # Note, removing the key from the dictionary is necessary to prevent
        #   obscure errors from the Flax optimizer.
        params.pop(m_key, None)
      else:
        assert restored_model_cfg.representation_size
        params[m_key] = m_params
    else:
      if m_key in train_state.optimizer.target:
        params[m_key] = m_params
      else:
        logging.info('Skipping %s. In restored model but not in target',
                     m_key)

  if log_initialised_param_shapes:
    logging.info('Parameter summary after initialising from train state')
    debug_utils.log_param_shapes(params)
  return train_state.replace(
      optimizer=train_state.optimizer.replace(target=flax.core.freeze(params)))
