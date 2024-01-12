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
r"""Default configs for X-ViT on metaphase sex ID.

"""
# pylint: enable=line-too-long

import ml_collections

_TRAIN_SIZE = 80_000


def get_config():
  """Returns the X-ViT experiment configuration for metaphase sexID."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'meta-sexid-xvit'

  # Dataset.
  config.dataset_name = 'metaphase_sexid'
  config.data_dtype_str = 'float32'

  # Model.
  config.model_name = 'xvit_multilabel_classification'
  config.model_dtype_str = 'float32'
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.hidden_size = 768
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [4, 4]
  config.model.mlp_dim = 2048
  config.model.num_layers = 12
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.
  config.model.transformer_encoder_configs = ml_collections.ConfigDict()
  config.model.transformer_encoder_configs.type = 'global'
  config.model.attention_fn = 'standard'
  config.model.attention_configs = ml_collections.ConfigDict()
  config.model.attention_configs.num_heads = 12

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
  config.num_training_epochs = 100
  config.log_eval_steps = 1000
  config.batch_size = 1024
  config.rng_seed = 42
  config.init_head_bias = -10.0

  # Learning rate.
  steps_per_epoch = _TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 8e-4
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*linear_warmup*linear_decay'
  config.lr_configs.total_steps = total_steps
  config.lr_configs.end_learning_rate = 1e-5
  config.lr_configs.warmup_steps = 1000
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
