# Copyright 2023 The Scenic Authors.
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

# pylint: disable=line-too-long
r"""Default configs for ViT on chrmID with input examples cropped to 199x99.

"""
# pylint: disable=line-too-long

import ml_collections

_TRAIN_SIZE = 368_106
VARIANT = 'Ti/4'


def get_config(runlocal=''):
  """Returns the ViT experiment configuration for chrmID."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'chrmID-vit'
  # Dataset.
  config.dataset_name = 'chrmID_baseline'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  # For available cropped shapes, see chrmID_baseline_dataset:DATASET_BASE_DIRS.
  config.dataset_configs.chrm_image_shape = (199, 99)

  # Model.
  version, patch = VARIANT.split('/')
  config.model_name = 'topological_vit_classification'
  config.model = ml_collections.ConfigDict()
  config.model.hidden_size = {
      'Ti': 192,
      'S': 384,
      'B': 768,
      'L': 1024,
      'H': 1280
  }[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [int(patch), int(patch)]
  config.model.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16, 'H': 16}[version]
  config.model.mlp_dim = {
      'Ti': 768,
      'S': 1536,
      'B': 3072,
      'L': 4096,
      'H': 5120
  }[version]
  config.model.num_layers = {
      'Ti': 12,
      'S': 12,
      'B': 12,
      'L': 24,
      'H': 32
  }[version]
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.
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
  config.num_training_epochs = 500
  config.log_eval_steps = 1000
  config.batch_size = 8 if runlocal else 512
  config.rng_seed = 42
  config.init_head_bias = -10.0

  # Learning rate.
  steps_per_epoch = _TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 3e-3
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*linear_warmup*linear_decay'
  config.lr_configs.total_steps = total_steps
  config.lr_configs.end_learning_rate = 1e-6
  config.lr_configs.warmup_steps = 10_000
  config.lr_configs.base_learning_rate = base_lr

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 5000
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  if runlocal:
    config.count_flops = False

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
