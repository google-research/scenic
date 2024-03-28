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

"""Train/eval/model utility functions."""

import dataclasses
import operator
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from absl import logging
from big_vision import utils as bv_utils
from big_vision.models import vit as vit_model
from clu import metric_writers
import flax
from flax import jax_utils
from flax.core import frozen_dict
import jax
import jax.numpy as jnp
import ml_collections
import optax
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.projects.knowledge_visual_language.models import constants
from scenic.projects.t5 import model as t5_model
from scenic.train_lib import train_utils
from scenic.train_lib.train_utils import TrainState

# Note this list must be in the exact order of the inputs required by the model.
PyTree = Union[Mapping[str, Mapping], Any]


def froze_param_optax(
    params: optax.Params,
    tx: optax.GradientTransformation,
    frozen_patterns: Optional[List[str]] = None,
    not_frozen_patterns: Optional[List[str]] = None,
) -> optax.GradientTransformation:
  r"""change optax optimizer to not optimize frozen parameter.

  Args:
    params: model parameter in FrozenDict (Tree)
    tx: optax GradientTransform function
    frozen_patterns: a list of re patterns for frozen parameter
    not_frozen_patterns: a tupe of re patterns for trainable parameter with
      scale

  Returns:
    optax GradientTransform function that mask out frozen parameters.

  Example:
    class Encoder(nn.Module):
      @nn.compact
      def __call__(self, x, train=True):
        h = nn.Dense(8, name='dense_init')(x)
        return nn.Dense(1, name='dense_output')(h)

    ......(after model initialization)
    tx = optax.adamw(learning_rate=1e-3, weight_decay=1e-2)
    tx = froze_param_optax(params=params, tx = tx,\
        frozen_patterns=["params/dense_init/.*"])
    opt_state = tx.init(params)
    ......
  """
  if frozen_patterns and not_frozen_patterns is None:
    frozen_masks = bv_utils.make_mask_trees(params, frozen_patterns)
    frozen_mask = jax.tree_util.tree_map(
        lambda *bools: any(bools), *frozen_masks
    )
    not_frozen_mask = jax.tree_util.tree_map(operator.not_, frozen_mask)
    tx = optax.chain(
        optax.masked(tx, not_frozen_mask),
        optax.masked(optax.set_to_zero(), frozen_mask))
  elif not_frozen_patterns:
    not_frozen_keys = []
    scale_vals = []
    for pattern, val in not_frozen_patterns:
      if frozen_patterns is not None and pattern in frozen_patterns:
        continue
      not_frozen_keys += [pattern]
      scale_vals += [val]
    not_frozen_masks = bv_utils.make_mask_trees(params, not_frozen_keys)
    not_frozen_mask = jax.tree_util.tree_map(
        lambda *bools: any(bools), *not_frozen_masks
    )
    frozen_mask = jax.tree_util.tree_map(operator.not_, not_frozen_mask)
    scale_txs = [
        optax.masked(optax.scale(scale_val), mask)
        for scale_val, mask in zip(scale_vals, not_frozen_masks)
    ]
    tx = optax.chain(
        optax.masked(tx, not_frozen_mask),
        optax.masked(optax.set_to_zero(), frozen_mask), *scale_txs)
  return tx


def update_config(config: ml_collections.ConfigDict, meta_data: Dict[str, Any]):
  config.num_train_examples = meta_data['num_train_examples']
  steps_per_epoch = config.num_train_examples // config.batch_size
  config.lr_configs.total_steps = int(config.num_training_epochs *
                                      steps_per_epoch)


def get_dataset(
    config: ml_collections.ConfigDict,
    data_rng: jnp.ndarray,
    *,
    dataset_service_address: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_configs: Optional[ml_collections.ConfigDict] = None
) -> dataset_utils.Dataset:
  """Creates dataset.

  Copy from scenic.train_lib.train_utils.get_dataset. Only use`
  this alternative function when your dataset is not registered in
  secnic library, and you need to add dependency into BUILD.

  By default, the values in the config file are used.
  However, if the optional `dataset_name` and `dataset_configs` are passed,
    those are used instead.

  Args:
    config: The configuration of the experiment.
    data_rng: Random number generator key to use for the dataset.
    dataset_service_address: Used when using the tf.data.experimental.service
    dataset_name: Name of dataset to load, if not reading from the config.
    dataset_configs: Configuration of the dataset, if not reading directly from
      the config.

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


@dataclasses.dataclass
class SummaryBuilder:
  """A helper class to build the summary over the training iterations."""
  metrics: List[Dict[str, Tuple[float, int]]]
  extra_logs: List[Dict[str, Any]]

  def update(self, metrics_update, extra_logs_update):
    """Update with the given per-step metrics."""
    self.metrics.append(metrics_update)
    self.extra_logs.append(extra_logs_update)

  def write(self, writer: metric_writers.MetricWriter, step: int):
    """Write to the given writer and training step.

    After writing, the state gets reset.

    Args:
      writer: The summary will be written with this writer.
      step: The current training step.

    Returns:
      The summary since the last write.
    """
    summary = train_utils.log_train_summary(
        step=step,
        train_metrics=jax.tree_util.tree_map(train_utils.unreplicate_and_get,
                                             self.metrics),
        extra_training_logs=jax.tree_util.tree_map(
            train_utils.unreplicate_and_get, self.extra_logs),
        writer=writer,
        key_separator='/')
    self.metrics = []
    self.extra_logs = []
    return summary


def all_gather_and_unreplicate(tensor):
  return jax_utils.unreplicate(
      jax.pmap(lambda x: jax.lax.all_gather(x, 'batch'), 'batch')(tensor))


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


def pop_axes_names(
    train_state: TrainState,
    axes_name: str = 'params_axes') -> Tuple[TrainState, Optional[Any]]:
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
    model_state, params_axes = frozen_dict.freeze(model_state).pop(axes_name)
    return train_state.replace(model_state=model_state), params_axes
  else:
    return train_state, None


def re_add_axis_names(train_state: TrainState,
                      params_axes: Any,
                      axes_name: str = 'params_axes') -> TrainState:
  """Adds axes_names to model_state for a train state.

  Args:
    train_state: Training state.
    params_axes: Model axes metadata to re-add.
    axes_name: the string specifying the name in the model_state

  Returns:
    New train state without axes_names in model_state, axes_names metadata if it
    was removed (so it can be re-added).
  """
  if params_axes:
    model_state = frozen_dict.unfreeze(train_state.model_state)
    model_state[axes_name] = params_axes
    return train_state.replace(model_state=frozen_dict.freeze(model_state))
  else:
    return train_state


def load_key_params(params: constants.PyTree,
                    config: ml_collections.ConfigDict):
  """Load T5 & ViT for Key Encoders."""
  t5_params = t5_model.load_pretrained_weights(config.model.t5_name)
  if 'key_text_encoder' in params:
    params = replace_dict(
        params,
        t5_params,
        ckpt_prefix_path=['params', 't5_module', 'encoder'],
        model_prefix_path=['key_text_encoder'],
        name_mapping={})
  return params


def load_text_params(params: constants.PyTree,
                     config: ml_collections.ConfigDict):
  """Load T5 params from a checkpoint."""
  t5_params = t5_model.load_pretrained_weights(config.model.t5_name)
  logging.info('T5 params are:')
  logging.info(jax.tree_util.tree_map(lambda x: x.shape, t5_params))
  # first load the shared token embeddings

  params = replace_dict(
      params,
      t5_params,
      ckpt_prefix_path=['params', 't5_module', 'token_embedder'],
      model_prefix_path=['shared_token_embedder'],
      name_mapping={})

  # then load the encoder and decoder weights
  params = replace_dict(
      params,
      t5_params,
      ckpt_prefix_path=['params', 't5_module', 'encoder'],
      model_prefix_path=['text_encoder'],
      name_mapping={})

  params = replace_dict(
      params,
      t5_params,
      ckpt_prefix_path=['params', 't5_module', 'encoder'],
      model_prefix_path=['fusion_encoder'],
      name_mapping={})

  params = replace_dict(
      params,
      t5_params,
      ckpt_prefix_path=['params', 't5_module', 'encoder'],
      model_prefix_path=['query_head'],
      name_mapping={})

  params = replace_dict(
      params,
      t5_params,
      ckpt_prefix_path=['params', 't5_module', 'encoder'],
      model_prefix_path=['key_head'],
      name_mapping={})

  params = replace_dict(
      params,
      t5_params,
      ckpt_prefix_path=['params', 't5_module', 'decoder'],
      model_prefix_path=['out_decoder', 'decoder_module'],
      name_mapping={})

  return params


def load_visual_params(params: constants.PyTree,
                       config: ml_collections.ConfigDict):
  """Load encoder parameters."""
  load_params = vit_model.load(
      init_params=params.get('img_encoder'),
      init_file=config.model.vit_model_path,
      model_cfg='',
      dont_load='MAPHead(.+)|head(.+)')

  params = replace_dict(
      params, load_params, model_prefix_path=['img_encoder'], name_mapping={})
  return params
