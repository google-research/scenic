"""Utility functions for Training."""
import functools
import os
import re
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
from clu import checkpoint
import flax
import flax.linen as nn
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
from scenic.common_lib import debug_utils
from scenic.train_lib import optimizers
from scenic.train_lib.pretrain_utils import inspect_params
from tensorflow.io import gfile


PyTree = Union[Mapping[str, Mapping], Any]
PRNGKey = jnp.ndarray


def all_steps(checkpoints_dir: str) -> Sequence[int]:
  """Returns list of available step numbers in ascending order."""
  # Assumes the checkpoint has the format "checkpoint_\d+"
  glob_pattern = os.path.join(checkpoints_dir, 'checkpoint_*')
  checkpoint_paths = gfile.glob(glob_pattern)
  steps = []
  for each in checkpoint_paths:
    steps.append(int(each.split('_')[-1]))
  sorted(steps)
  return steps


def average_list_of_dicts(list_of_dicts):
  """Takes the average of a list of dicts with identical keys."""
  ret_dict = {}
  for cur_dict in list_of_dicts:
    for key, value in cur_dict.items():
      aggregated_value = ret_dict.get(key, 0)
      aggregated_value += value * 1.0 / len(list_of_dicts)
      ret_dict[key] = aggregated_value
  return ret_dict


def initialize_model(
    *,
    model_def: nn.Module,
    input_spec: Sequence[Union[Tuple[Tuple[int, ...], jnp.dtype],
                               Tuple[int, ...], None]],
    config: ml_collections.ConfigDict,
    rngs: Union[jnp.ndarray, Mapping[str, jnp.ndarray]],
) -> Tuple[PyTree, PyTree, int, Optional[float]]:
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

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def _initialize_model(rngs):
    """Initialization function to be jitted."""
    init_model_state, init_params = model_def.init(
        rngs, *dummy_input, train=False, debug=False).pop('params')
    #  Set bias in the head to low value, such that loss is small initially.
    if config.get('init_head_bias', None) is not None:
      init_params = flax.core.unfreeze(init_params)
      if 'vit' in init_params:
        init_params['vit'][
            'output_projection'] = optimizers.tree_map_with_names(
                lambda p: jnp.full_like(p, config.init_head_bias),
                init_params['vit']['output_projection'],
                match_name_fn=lambda name: 'bias' in name)
      else:
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


def get_params_and_model_state_dict(
    restored_train_state: PyTree,
    config: Optional[ml_collections.ConfigDict] = None
) -> Tuple[PyTree, Optional[PyTree]]:
  """Restores the params and model state.

  This function also applies the conversion needed for pre-Linen checkpoints.

  Args:
    restored_train_state: A dictionary that contains a check-pointed TrainState.
    config: Configurations of the initialization.

  Returns:
    A tuple of restored params and restored models state. Note that these are
    not frozen, and need to be frozen before passing them to the optimizer.
  """
  if config is not None:
    class_model_list = ['robust_vit_multilabel_classification',
                        'bert_vit_multilabel_classification',
                        'block_bert_vit_multilabel_classification',
                        'robust_mixer_multilabel_classification']
    if config.model_name in class_model_list:
      if config.get('dvae', False):
        restored_params = restored_train_state
      else:
        restored_params = restored_train_state['g_optimizer']['target']
        # Code for handling ResNet CNN input.
        # try:
        #   restored_params = restored_train_state['g_optimizer']['target']
        # except:
        #   # For VQVAE model
        #   restored_params = restored_train_state['optimizer']['target']
    else:
      restored_params = restored_train_state['optimizer']['target']
  else:
    restored_params = restored_train_state['optimizer']['target']

  restored_model_state = restored_train_state.get('model_state')

  if 'params' in restored_params:  # Backward compatibility.
    restored_params = restored_params['params']
    if config is not None and not config.get('dvae', False):
      # this line will convert name from 1 to start from 0.
      # note that dvae start from 1, so need to remove this line.
      restored_params = dict(checkpoints.convert_pre_linen(restored_params))
    if restored_model_state:
      restored_model_state = checkpoints.convert_pre_linen(
          flax.traverse_util.unflatten_dict({
              tuple(k.split('/')[1:]): v
              for k, v in restored_model_state.items()
          }))
  return restored_params, restored_model_state


def _replace_dict(model: PyTree,
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
    tmp_m_key = ('vqgan',) + m_key
    if m_key not in model_flat and tmp_m_key not in model_flat:
      logging.warning(
          '%s in checkpoint doesn\'t exist in model. Skip.', m_key_str)
      continue
    if skip_regex and re.findall(skip_regex, m_key_str):
      logging.info('Skip loading parameter %s.', m_key_str)
      continue
    logging.info('Loading %s from checkpoint into model', m_key_str)

    if tmp_m_key in model_flat:
      model_flat[tmp_m_key] = m_params
    elif m_key in model_flat:
      model_flat[m_key] = m_params

  return flax.core.freeze(flax.traverse_util.unflatten_dict(model_flat))


def init_from_vq_pretrain_state(
    train_state: Any,
    config: ml_collections.ConfigDict,
    ckpt_prefix_path: Optional[List[str]] = None,
    model_prefix_path: Optional[List[str]] = None,
    name_mapping: Optional[Mapping[str, str]] = None,
    skip_regex: Optional[str] = None
) -> Any:
  """Updates the train_state with data from pretrain_state.

  Args:
    train_state: A raw TrainState for the model.
    config: Configurations of the initialization.
    ckpt_prefix_path: Prefix to restored model parameters.
    model_prefix_path: Prefix to the parameters to replace in the subtree model.
    name_mapping: Mapping from parameter names of checkpoint to this model.
    skip_regex: If there is a parameter whose parent keys match the regex,
      the parameter will not be replaced from pretrain_state.

  Returns:
    Updated train_state.
  """
  ckpt_dir = config.vqgan_dir
  if config.get('new_vq_version', False):
    pretrain_state = checkpoints.restore_checkpoint(ckpt_dir, None)
  elif config.get('vq_model_step_specified', False):
    pretrain_state = checkpoints.restore_checkpoint(
        ckpt_dir, config.get('vq_model_load_step'))
  else:
    pretrain_state = checkpoint.load_state_dict(f'/readahead/1G/{ckpt_dir}')
  # code for loading resnet CNN encoder instead of VQEncoder.
  dict_dfs_update_bias_scale(pretrain_state['g_optimizer']['target'], '')
  dict_dfs_update_bias_scale(pretrain_state['optimizer']['target'], '')
  name_mapping = name_mapping or {}
  (restored_params, restored_model_state) = get_params_and_model_state_dict(
      pretrain_state, config=config)

  model_params = train_state.optimizer.target
  model_params = _replace_dict(
      model_params,
      restored_params,
      ckpt_prefix_path,
      model_prefix_path,
      name_mapping,
      skip_regex)

  # debug
  restored_params = inspect_params(
      expected_params=train_state.optimizer.target,
      restored_params=model_params,
      fail_if_extra=False,
      fail_if_missing=False,
      fail_if_shapes_mismatch=False)

  new_optimizer = train_state.optimizer.replace(
      target=model_params)
  train_state = train_state.replace(  # pytype: disable=attribute-error
      optimizer=new_optimizer)
  if (restored_model_state is not None and
      train_state.model_state is not None and
      train_state.model_state):
    if model_prefix_path:
      # Insert model prefix after 'batch_stats'.
      model_prefix_path = ['batch_stats'] + model_prefix_path
      if 'batch_stats' in restored_model_state:
        ckpt_prefix_path = ckpt_prefix_path or []
        ckpt_prefix_path = ['batch_stats'] + ckpt_prefix_path
    elif 'batch_stats' not in restored_model_state:  # Backward compatibility.
      model_prefix_path = ['batch_stats']
    if ckpt_prefix_path and ckpt_prefix_path[0] != 'batch_stats':
      ckpt_prefix_path = ['batch_stats'] + ckpt_prefix_path
    model_state = _replace_dict(train_state.model_state,
                                restored_model_state,
                                ckpt_prefix_path,
                                model_prefix_path,
                                name_mapping,
                                skip_regex)
    train_state = train_state.replace(  # pytype: disable=attribute-error
        model_state=model_state)
  return train_state


def dict_dfs_update_bias_scale(dict_obj, parent_name):
  """Temporal fix to load the model by flatting bias and scale."""
  for key, value in dict_obj.items():
    if isinstance(value, dict):
      dict_dfs_update_bias_scale(value, key)
    else:
      if parent_name.startswith('GroupNorm') and (key in ['scale', 'bias']) or (
          key in ['scale', 'bias']):
        dict_obj[key] = jnp.squeeze(value)
