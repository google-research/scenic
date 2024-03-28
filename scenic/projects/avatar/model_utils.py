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

"""Model utils for MBT."""

from collections import abc
from typing import Any

from absl import logging
import flax
import ml_collections
from scenic.common_lib import debug_utils
from scenic.projects.mbt.model_utils import init_embedding
from scenic.projects.mbt.model_utils import init_encoderblock
from scenic.projects.mbt.model_utils import init_posemb


def flatten_params(d, parent_key='', sep='/'):
  """Flattens a dictionary, keeping empty leaves."""
  items = []
  for k, v in d.items():
    path = parent_key + sep + k if parent_key else k
    if isinstance(v, abc.MutableMapping):
      items.extend(flatten_params(v, path, sep=sep).items())
    else:
      items.append((path, v))
  # Keeps the empty dict if it was set explicitly.
  if parent_key and not d:
    items.append((parent_key, {}))
  return dict(items)


def nest_params(flat_dic, sep='/'):
  """Nest (un-flatten) a dictionary."""
  res = dict()
  for key, value in flat_dic.items():
    parts = key.split(sep)
    d = res
    for part in parts[:-1]:
      if part not in d:
        d[part] = dict()
      d = d[part]
    d[parts[-1]] = value
  return res


def initialise_from_train_state(config,
                                train_state: Any,
                                restored_train_state: Any,
                                restored_model_cfg: ml_collections.ConfigDict,
                                restore_output_proj: bool,
                                mbt_transformer_key: str = 'Transformer',
                                log_initialised_param_shapes: bool = True,
                                one_config: bool = True,
                                prefix_path: Any = None,
                                restore_from_old_format: bool = True) -> Any:
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
    restore_output_proj: If true, load the final output projection. Set to False
      if finetuning to a new dataset.
    mbt_transformer_key: The key used for storing the subtree in the parameters
      that keeps Transformer weights, that are supposed to be initialized from
      the given pre-trained model.
    log_initialised_param_shapes: If true, print tabular summary of all the
      variables in the model once they have been initialised.
    one_config: If true, we have only a single config. If false, we get a tuple
      of configs in the order [init_config, model_config, dataset_config]. This
      is useful for works that build upon MBT and have different models in their
      config.
    prefix_path: If parameters are in a subtree.
    restore_from_old_format: Whether to restore from old MBT network names.

  Returns:
    Updated train_state.
  """

  # Split up configs
  if one_config:
    init_config = config.init_from
    model_config = config.model
    dataset_config = config.dataset_configs
  else:
    init_config, model_config, dataset_config = config

  # Inspect and compare the parameters of the model with the init-model
  params = flax.core.unfreeze(train_state.optimizer.target)

  if init_config.get('checkpoint_format',
                     'scenic') in ('big_vision', 'bigvision'):
    restored_params = restored_train_state.optimizer['target']
  else:
    restored_params = restored_train_state.optimizer.target
  restored_params = flax.core.unfreeze(restored_params)
  if init_config.get('init_from_vit', True):
    if prefix_path:
      video_params = params[prefix_path]
    else:
      video_params = params
    # Start moving parameters, one-by-one and apply changes if needed
    for m_key, m_params in restored_params.items():
      if m_key == 'output_projection':
        if restore_output_proj:
          video_params[m_key] = m_params
        else:
          pass
      elif m_key == 'pre_logits':
        if model_config.get('representation_size', None) is None:
          # We don't have representation_size in the new model, so let's ignore
          #   if from the pretained model, in case it has it.
          # Note, removing the key from the dictionary is necessary to prevent
          #   obscure errors from the Flax optimizer.
          video_params.pop(m_key, None)
        else:
          assert restored_model_cfg.model.representation_size
          video_params[m_key] = m_params

      elif m_key in ['Transformer']:
        for tm_key, tm_params in m_params.items():
          if tm_key == 'posembed_input':  # Might need resolution change
            init_posemb(
                video_params[mbt_transformer_key],
                m_params,
                init_config,
                model_config,
                dataset_config,
                restored_model_cfg,
                'posembed_input',
                prefix_path=prefix_path)
            init_posemb(
                video_params[mbt_transformer_key],
                m_params,
                init_config,
                model_config,
                dataset_config,
                restored_model_cfg,
                'posembed_input_spectrogram',
                prefix_path=prefix_path)
            init_posemb(
                video_params[mbt_transformer_key],
                m_params,
                init_config,
                model_config,
                dataset_config,
                restored_model_cfg,
                'posembed_input_flow',
                prefix_path=prefix_path)
            init_posemb(
                video_params[mbt_transformer_key],
                m_params,
                init_config,
                model_config,
                dataset_config,
                restored_model_cfg,
                'posembed_input_wave',
                prefix_path=prefix_path)
            init_posemb(
                video_params,
                m_params,
                init_config,
                model_config,
                dataset_config,
                restored_model_cfg,
                'bottleneck',
                prefix_path=prefix_path)
          elif 'encoderblock' in tm_key:
            logging.info('Loading encoder parameters.')
            init_encoderblock(video_params[mbt_transformer_key], m_params,
                              tm_key)
          else:  # Other parameters of the Transformer encoder
            video_params[mbt_transformer_key][tm_key] = tm_params
      elif m_key == 'embedding':
        init_embedding(video_params, m_params, init_config, model_config,
                       'embedding')
        init_embedding(video_params, m_params, init_config, model_config,
                       'embedding_flow')
        init_embedding(video_params, m_params, init_config, model_config,
                       'embedding_spectrogram')
        init_embedding(video_params, m_params, init_config, model_config,
                       'embedding_wave')
      else:
        if m_key in train_state.optimizer.target:
          video_params[m_key] = m_params
        if '%s_spectrogram' % m_key in train_state.optimizer.target:
          video_params['%s_spectrogram' % m_key] = m_params
        if '%s_flow' % m_key in train_state.optimizer.target:
          video_params['%s_flow' % m_key] = m_params
        if '%s_wave' % m_key in train_state.optimizer.target:
          video_params['%s_wave' % m_key] = m_params
        else:
          logging.info('Skipping %s. In restored model but not in target',
                       m_key)
  else:
    from_flat = flatten_params(restored_params)
    to_flat = flatten_params(params)

    for key in from_flat:
      if key in to_flat:
        if from_flat[key].shape == to_flat[key].shape:
          to_flat[key] = from_flat[key]
        elif key == 'video_encoder/Transformer/posembed_input/pos_embedding':
          nb_pos = to_flat[key].shape[1]
          to_flat[key] = from_flat[key][:, :nb_pos, :]
      else:
        logging.info('Skipping %s. In restored model but not in target', key)

    if restore_from_old_format:
      assert one_config
      model_modalities = config.mbt.model.modality_fusion

      for key in to_flat:
        if '_spectrogram/' not in key:
          continue
        if len(model_modalities) == 1 and 'embedding' not in key:
          from_key = key.replace('_spectrogram/', '/')
        else:
          from_key = key.replace('_spectrogram/', '_spec/')

        if from_key in from_flat:
          if from_flat[from_key].shape == to_flat[key].shape:
            to_flat[key] = from_flat[from_key]
            logging.info(
                'Restoring with converted key from %s to %s.', key, from_key,
            )
          else:
            logging.info(
                'Shape missmatch with converted key %s from %s.', key, from_key
            )
        else:
          logging.info('Not found: converted key %s from %s.', key, from_key)

    if init_config.get('dual_stream_init', False):
      for to_key in to_flat:
        if '_spec' in to_key:
          from_key = to_key.replace('_spec', '')
          if from_key in from_flat:
            if from_flat[from_key].shape == to_flat[to_key].shape:
              to_flat[to_key] = from_flat[from_key]
            elif (
                from_key
                == 'video_encoder/Transformer/posembed_input/pos_embedding'
            ):
              nb_pos = to_flat[to_key].shape[1]
              to_flat[to_key] = from_flat[from_key][:, :nb_pos, :]
            logging.info('Dual stream loading %s from %s', to_key, from_key)
          else:
            logging.info(
                'Skipping %s. In restored model but not in target', to_key
            )

    params = nest_params(to_flat)

  if log_initialised_param_shapes:
    logging.info('Parameter summary after initialising from train state')
    debug_utils.log_param_shapes(params)
  return train_state.replace(
      optimizer=train_state.optimizer.replace(target=flax.core.freeze(params)))
