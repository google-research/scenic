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

"""Main file for running classification eval."""

import functools
from typing import Callable, Dict, List, Optional, Tuple

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.projects.matvit import matvit
from scenic.train_lib import train_utils

Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[
    [jnp.ndarray, Dict[str, jnp.ndarray]], Dict[str, Tuple[float, int]]
]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]
LrFn = Callable[[jnp.ndarray], jnp.ndarray]

_MODEL_PATH = flags.DEFINE_string(
    'model_path', None, 'The path to the model.', required=True
)
_MATVIT_DIMS = flags.DEFINE_string(
    'matvit_dims', None, 'The MatViT dims.', required=True
)

_IMAGENET_TRAIN_SIZE = 1281167
VARIANT = 'B/16'
NUM_CLASSES = 1000


def get_config():
  """Returns the ViT experiment configuration for ImageNet.

  This file is a copy of config/imagenet_augreg_matvit_config.py.
  """
  version, patch = VARIANT.split('/')
  config = ml_collections.ConfigDict()
  config.experiment_name = 'imagenet-regularized_vit'
  # Dataset.
  config.dataset_name = 'bit'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = 'imagenet2012'
  config.dataset_configs.num_classes = NUM_CLASSES
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.val_split = 'validation'
  config.dataset_configs.pp_train = {
      'L': (
          'decode|resize(384)'
          '|value_range(-1, 1)'
          f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
          '|keep("image", "labels")'
      ),
      'B': (
          'decode|resize_small(256)|central_crop(224)'
          '|value_range(-1, 1)'
          f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
          '|keep("image", "labels")'
      ),
  }[version]
  config.dataset_configs.pp_eval = {
      'L': (
          'decode|resize(384)'
          '|value_range(-1, 1)'
          f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
          '|keep("image", "labels")'
      ),
      'B': (
          'decode|resize_small(256)|central_crop(224)'
          '|value_range(-1, 1)'
          f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
          '|keep("image", "labels")'
      ),
  }[version]

  config.dataset_configs.prefetch_to_device = 2
  # Shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = 250_000
  # Model.
  config.model_name = 'vit_multilabel_classification'
  config.model = ml_collections.ConfigDict()
  config.model.hidden_size = {
      'Ti': 192,
      'S': 384,
      'B': 768,
      'L': 1024,
      'H': 1280,
  }[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [int(patch), int(patch)]
  config.model.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16, 'H': 16}[version]
  config.model.mlp_dim = {
      'Ti': 768,
      'S': 1536,
      'B': 3072,
      'L': 4096,
      'H': 5120,
  }[version]
  config.model.num_layers = {'Ti': 12, 'S': 12, 'B': 12, 'L': 24, 'H': 32}[
      version
  ]
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.0
  config.model.dropout_rate = 0.1
  config.model.stochastic_depth = 0.1
  config.model_dtype_str = 'float32'
  # Training.
  config.trainer_name = 'classification_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 0.1
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  config.num_training_epochs = 300
  config.log_eval_steps = 1000
  config.batch_size = 2048
  config.rng_seed = 42
  config.init_head_bias = -6.9  # -log(1000)

  # Learning rate.
  steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 0.001
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 10_000
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = base_lr
  # Mixup.
  config.mixup = ml_collections.ConfigDict()
  config.mixup.bind_to = None
  config.mixup.alpha = 0.5
  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 5000
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  config.m = None  # Placeholder for randaug strength.
  config.l = None  # Placeholder for randaug layers.
  return config


def get_logits_fn(
    train_state: train_utils.TrainState,
    batch: Batch,
    dims: Optional[List[int]],
    *,
    flax_model: nn.Module,
    gather_to_host: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Gets the logits from an input abtch.

  Args:
    train_state: Loaded train state from the ckpt.
    batch: Input batch.
    dims: Matvit nesting dimensions, a list of size num_layers.
    flax_model: The trained mode.
    gather_to_host: Whether to gather all outputs to host device.

  Returns:
    Logits for this batch.
  """
  variables = {'params': train_state.params, **train_state.model_state}
  logits = flax_model.apply(
      variables,
      batch['inputs'],
      train=False,
      debug=False,
      matvit_mask_dims=dims,
  )
  if gather_to_host:
    logits = jax.lax.all_gather(logits, 'batch')
    batch = jax.lax.all_gather(batch, 'batch')
  return logits, batch['label'], batch['batch_mask']


def create_train_state(ckpt_path, config):
  """Gets the train state from the input ckpt.

  Args:
    ckpt_path: Path to the ckpt to be loaded..
    config: The configuration used to train the model.

  Returns:
    Loaded train state and a function to return the logits.
  """
  ckpt_info = ckpt_path.split('/')
  ckpt_dir = '/'.join(ckpt_info[:-1])
  ckpt_num = ckpt_info[-1].split('_')[-1]

  rng = jax.random.PRNGKey(config.rng_seed)
  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=None
  )
  model = matvit.MatViTMultiLabelClassificationModel(config, dataset.meta_data)

  init_rng, _ = jax.random.split(rng)
  (params, model_state, _, _) = train_utils.initialize_model(
      model_def=model.flax_model,
      input_spec=[(
          dataset.meta_data['input_shape'],
          dataset.meta_data.get('input_dtype', jnp.float32),
      )],
      config=config,
      rngs=init_rng,
  )
  train_state = train_utils.TrainState(params=params, model_state=model_state)
  train_state, _ = train_utils.restore_checkpoint(
      ckpt_dir,
      train_state,
      assert_exist=True,
      step=int(ckpt_num),
  )
  if not _MATVIT_DIMS.value:
    matvit_dims = None
  else:
    matvit_dims = [int(val) for val in _MATVIT_DIMS.value.split(',')]
    assert (
        len(matvit_dims) == config.model.num_layers
    ), 'Number of matvit dimensions needs to match the number of layers.'

  logits_fn = functools.partial(
      get_logits_fn, flax_model=model.flax_model, dims=matvit_dims
  )
  return train_state, logits_fn


def main(_):
  config = get_config()
  train_state, logits_fn = create_train_state(_MODEL_PATH.value, config)
  train_state = jax_utils.replicate(train_state)
  p_logits_fn = jax.pmap(logits_fn, donate_argnums=(1,), axis_name='batch')
  rng = jax.random.PRNGKey(config.rng_seed)
  data_rng, _ = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=None
  )
  total_eval_steps = int(
      np.ceil(dataset.meta_data['num_eval_examples'] / config.batch_size)
  )
  top1_correct_list = []
  for step in range(total_eval_steps):
    logging.info('Eval inference step: %d', step)
    batch = next(dataset.valid_iter)
    logits, labels, mask = p_logits_fn(train_state, batch)
    mask = np.array(jax_utils.unreplicate(mask)).astype(bool)
    logits = np.array(jax_utils.unreplicate(logits))[mask]
    labels = np.array(jax_utils.unreplicate(labels))[mask]
    top1_idx = jnp.argmax(logits, axis=-1)[..., None]
    top1_correct = jnp.take_along_axis(labels, top1_idx, axis=-1)
    top1_correct_list.append(top1_correct)

  top1_correct = np.concatenate(top1_correct_list, axis=0)
  acc = jnp.mean(top1_correct)
  logging.info('Classification acc: %f', acc)


if __name__ == '__main__':
  app.run(main=main)
