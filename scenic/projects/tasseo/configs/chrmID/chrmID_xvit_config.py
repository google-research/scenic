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

# pylint: disable=line-too-long
r"""Default configs for X-ViT on chromosome identification task.

"""
# pylint: enable=line-too-long

import ml_collections

_TRAIN_SIZE = 409_007
VARIANT = 'B/4'


def get_config():
  """Returns the X-ViT experiment configuration for metaphase sexID."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'chrmID-xvit-jf'

  # Dataset.
  config.dataset_name = 'chrmID'
  config.data_dtype_str = 'float32'

  # Model.
  version, patch = VARIANT.split('/')
  config.model_name = 'xvit_classification'
  config.model_dtype_str = 'float32'
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.hidden_size = {
      'Ti': 192,
      'S': 384,
      'B': 768,
      'L': 1024,
      'H': 1280,
  }[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [int(patch), int(patch)]
  config.model.mlp_dim = {
      'Ti': 768,
      'S': 1536,
      'B': 3072,
      'L': 4096,
      'H': 5120,
  }[version]
  config.model.num_layers = {
      'Ti': 12,
      'S': 12,
      'B': 12,
      'L': 24,
      'H': 32,
  }[version]
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.
  config.model.transformer_encoder_configs = ml_collections.ConfigDict()
  config.model.transformer_encoder_configs.type = 'global'
  config.model.attention_fn = 'standard'
  config.model.attention_configs = ml_collections.ConfigDict()
  config.model.attention_configs.num_heads = {
      'Ti': 3,
      'S': 6,
      'B': 12,
      'L': 16,
      'H': 16,
  }[version]

  # Training.
  config.trainer_name = 'classification_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 0.1
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = None
  config.label_smoothing = None
  config.num_training_epochs = 500
  config.log_eval_steps = 1000
  config.batch_size = 512  # >=1024 causes RESOURCE EXHAUSTED errors.
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

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
