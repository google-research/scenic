"""Utils for ViViT-based regression models."""

from typing import Any

from absl import logging
import flax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.common_lib import debug_utils
from scenic.projects.vivit import model_utils as vivit_model_utils


average_frame_initializer = vivit_model_utils.average_frame_initializer
central_frame_initializer = vivit_model_utils.central_frame_initializer


def initialise_from_train_state(
    config,
    train_state: Any,
    restored_train_state: Any,
    restored_model_cfg: ml_collections.ConfigDict,
    restore_output_proj: bool,
    vivit_transformer_key: str = 'Transformer',
    log_initialised_param_shapes: bool = True) -> Any:
  """Updates the train_state with data from restored_train_state.

  We do not reuse this from vivit/model_utils in order to handle position
  embeddings and input embeddings differently in init_posemb and
  init_embedding, respectively.

  This function is written to be used for 'fine-tuning' experiments. Here, we
  do some surgery to support larger resolutions (longer sequence length) in
  the transformer block, with respect to the learned pos-embeddings.

  Args:
    config: Configurations for the model being updated.
    train_state: A raw TrainState for the model.
    restored_train_state: A TrainState that is loaded with parameters/state of a
      pretrained model.
    restored_model_cfg: Configuration of the model from which the
      restored_train_state come from. Usually used for some asserts.
    restore_output_proj: If true, load the final output projection. Set
      to False if finetuning to a new dataset.
    vivit_transformer_key: The key used for storing the subtree in the
      parameters that keeps Transformer weights, that are supposed to be
      initialized from the given pre-trained model.
    log_initialised_param_shapes: If true, print tabular summary of all the
      variables in the model once they have been initialised.

  Returns:
    Updated train_state.
  """
  # Inspect and compare the parameters of the model with the init-model.
  params = flax.core.unfreeze(train_state.optimizer.target)
  if config.init_from.get('checkpoint_format', 'scenic') == 'bigvision':
    restored_params = restored_train_state.optimizer['target']
  else:
    restored_params = restored_train_state.optimizer.target
  restored_params = flax.core.unfreeze(restored_params)

  # Start moving parameters, one-by-one and apply changes if needed.
  for m_key, m_params in restored_params.items():
    if m_key == 'output_projection':
      if restore_output_proj:
        params[m_key] = m_params
      else:
        logging.info('Not restoring output projection.')
        pass

    elif m_key == 'pre_logits':
      if config.model.representation_size is None:
        # We don't have representation_size in the new model, so let's ignore
        #   if from the pretained model, in case it has it.
        # Note, removing the key from the dictionary is necessary to prevent
        #   obscure errors from the Flax optimizer.
        params.pop(m_key, None)
      else:
        assert restored_model_cfg.model.representation_size
        params[m_key] = m_params

    elif m_key in {'Transformer', 'SpatialTransformer', 'TemporalTransformer'}:
      key_to_load = vivit_transformer_key
      is_temporal = False
      if m_key == 'TemporalTransformer':
        key_to_load = m_key
        is_temporal = True
      for tm_key, tm_params in m_params.items():
        if tm_key == 'posembed_input':  # Might need resolution change.
          init_posemb(params[key_to_load], m_params, config, restored_model_cfg,
                      is_temporal=is_temporal)
        elif 'encoderblock' in tm_key:
          vivit_model_utils.init_encoderblock(
              params[key_to_load], m_params, tm_key, config)
        else:  # Other parameters of the Transformer encoder.
          params[key_to_load][tm_key] = tm_params

    elif m_key == 'embedding':
      init_embedding(params, m_params, config)
    else:
      if m_key in train_state.optimizer.target:
        params[m_key] = m_params
      else:
        logging.info('Skipping %s. In restored model but not in target', m_key)

  if log_initialised_param_shapes:
    logging.info('Parameter summary after initialising from train state')
    debug_utils.log_param_shapes(params)
  return train_state.replace(
      optimizer=train_state.optimizer.replace(target=flax.core.freeze(params)))


def init_posemb(to_params, from_params, config, restored_model_cfg,
                is_temporal):
  """Initialize the positional embeddings."""
  if config.init_from.restore_positional_embedding:
    posemb = to_params['posembed_input']['pos_embedding']
    restored_posemb = from_params['posembed_input']['pos_embedding']
    if restored_posemb.shape != posemb.shape:
      # Rescale the grid of pos, embeddings.
      # Default parameter shape is (1, N, 768)
      logging.info('Adapting positional embeddings from %s to %s',
                   restored_posemb.shape, posemb.shape)
      ntok = posemb.shape[1]
      if restored_model_cfg.model.classifier == 'token':
        # The first token is the CLS token.
        cls_tok = restored_posemb[:, :1]
        restored_posemb_grid = restored_posemb[0, 1:]
      else:
        cls_tok = restored_posemb[:, :0]
        restored_posemb_grid = restored_posemb[0]
      if config.model.classifier == 'token':
        ntok -= 1

      if ((config.model.classifier == 'token') !=
          (restored_model_cfg.model.classifier == 'token')):
        logging.warning('Only one of target and restored model uses'
                        'classification token')
        if restored_posemb_grid == ntok:
          # In case the following `if` is not going to run, lets add batch dim:
          restored_posemb = restored_posemb_grid[None, ...]

      if len(restored_posemb_grid) != ntok:  # We need a resolution change.
        if is_temporal:
          if config.init_from.restore_temporal_embedding_for_goal:
            restored_posemb_grid = (
                vivit_model_utils.interpolate_1d_positional_embeddings(
                    restored_posemb_grid, ntok))
          else:
            restored_posemb_grid = (
                vivit_model_utils.interpolate_1d_positional_embeddings(
                    restored_posemb_grid, ntok - 1))

        elif config.init_from.positional_embed_size_change == 'resize':
          restored_posemb_grid = (
              vivit_model_utils.interpolate_positional_embeddings(
                  restored_posemb_grid, ntok))

        elif config.init_from.positional_embed_size_change == 'tile':
          restored_posemb_grid = (
              vivit_model_utils.tile_positional_embeddings(
                  restored_posemb_grid, ntok))

        elif config.init_from.positional_embed_size_change == 'resize_tile':
          temp_encoding = config.model.temporal_encoding_config
          if temp_encoding.method == 'temporal_sampling':
            tokens_per_frame = int(ntok / temp_encoding.n_sampled_frames)
          elif temp_encoding.method == '3d_conv':
            n_frames = (
                config.dataset_configs.num_frames //
                config.model.patches.size[2])
            tokens_per_frame = ntok // n_frames
          else:
            raise AssertionError(
                f'Unknown temporal encoding {temp_encoding.method}')
          restored_posemb_grid = (
              vivit_model_utils.interpolate_positional_embeddings(
                  restored_posemb_grid, tokens_per_frame))
          restored_posemb_grid = restored_posemb_grid[0]
          restored_posemb_grid = vivit_model_utils.tile_positional_embeddings(
              restored_posemb_grid, ntok)

        else:
          raise AssertionError(
              'Unknown positional embedding size changing method')
        # Attach the CLS token again.
        if config.model.classifier == 'token':
          restored_posemb = jnp.array(
              np.concatenate([cls_tok, restored_posemb_grid], axis=1))
        else:
          restored_posemb = restored_posemb_grid
    if is_temporal and not config.init_from.restore_temporal_embedding_for_goal:
      logging.info('Not restoring temporal embedding for goal')
      restored_posemb = jnp.array(
          np.concatenate(
              [restored_posemb,
               to_params['posembed_input']['pos_embedding'][:, -1:]], axis=1))

    to_params['posembed_input']['pos_embedding'] = restored_posemb
  else:
    logging.info('Not restoring positional encodings from pretrained model')


def init_embedding(to_params, from_params, config):
  """Initialize input embedding."""
  if config.init_from.get('restore_input_embedding', True):
    input_kernel = to_params['embedding']['kernel']
    restored_kernel = from_params['kernel']
    restored_bias = from_params['bias']

    if input_kernel.shape != restored_kernel.shape:
      # Kernel dimensions are [t, h, w, c_in, c_out].
      # assert config.model.temporal_encoding_config.method == '3d_conv', (
      #     'Input kernel dimensions should only differ if 3d_conv is the'
      #     'temporal encoding method')
      assert (input_kernel.shape[1:] == restored_kernel.shape
              or input_kernel.shape[1:] == restored_kernel.shape[1:]), (
                  'All filter dimensions besides the temporal dimension should '
                  'be equal. {} vs {}'.format(
                      input_kernel.shape, restored_kernel.shape))

      kernel_init_method = config.model.temporal_encoding_config.kernel_init_method
      if kernel_init_method == 'reduce_mean_initializer':
        logging.info('Initializing 2D input kernel with mean temporal frame.')
        restored_kernel = np.mean(restored_kernel, axis=0)
        restored_kernel = np.expand_dims(restored_kernel, axis=0)
      elif kernel_init_method == 'reduce_sum_initializer':
        logging.info(
            'Initializing 2D input kernel with sum of temporal frames.')
        restored_kernel = np.sum(restored_kernel, axis=0)
        restored_kernel = np.expand_dims(restored_kernel, axis=0)
      elif kernel_init_method == 'last_frame_initializer':
        logging.info('Initializing 2D input kernel with last temporal frame.')
        restored_kernel = restored_kernel[:1]
      else:
        raise AssertionError(
            'Unknown input kernel initialization {}'.format(kernel_init_method))

    to_params['embedding']['kernel'] = restored_kernel
    to_params['embedding']['bias'] = restored_bias
  else:
    logging.info('Not restoring input embedding parameters')

