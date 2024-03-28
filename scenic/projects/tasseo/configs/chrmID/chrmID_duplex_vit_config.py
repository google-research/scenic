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
r"""Default configs for DuplexViT on chrmID.

"""
# pylint: disable=line-too-long

import ml_collections

_TRAIN_SIZE = 409_007

# NOTE: Currently, VARIANT is used to configure  input, context, and fused
# encoders, so if you want different configs, you should manually change
# them bellow.
VARIANT = 'Ti/16'


HIDDEN_SIZE = {'Ti': 192, 'S': 384, 'B': 768, 'L': 1024}
NUM_HEADS = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16}
MLP_DIM = {'Ti': 768, 'S': 1536, 'B': 3072, 'L': 4096}
NUM_LAYERS = {'Ti': 12, 'S': 12, 'B': 12, 'L': 24}


def get_config(runlocal=''):
  """Gets ViT config for chrmID task with (chromosome+metaphase) input."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'chrmID-vit'
  # Dataset.
  config.dataset_name = 'chrmID_metaphase_context'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()

  # Model.
  config.model_name = 'duplex_vit_classification'
  config.model = ml_collections.ConfigDict()
  version, patch = VARIANT.split('/')
  config.model.hidden_size = HIDDEN_SIZE[version]
  config.model.classifier = 'token'
  config.model.dropout_rate = 0.0
  config.model.patches = ml_collections.ConfigDict()
  config.model.encoder = ml_collections.ConfigDict()
  config.model_dtype_str = 'float32'
  # Input encoder
  config.model.patches.input_size = [int(patch), int(patch)]
  config.model.encoder.input = ml_collections.ConfigDict()
  config.model.encoder.input.num_heads = NUM_HEADS[version]
  config.model.encoder.input.mlp_dim = MLP_DIM[version]
  config.model.encoder.input.num_layers = NUM_LAYERS[version] // 3
  config.model.encoder.input.attention_dropout_rate = 0.
  config.model.encoder.input.dropout_rate = 0.
  config.model.encoder.input.stochastic_depth = 0.
  # Context encoder
  config.model.patches.context_size = [int(patch), int(patch)]
  config.model.encoder.context = ml_collections.ConfigDict()
  config.model.encoder.context.num_heads = NUM_HEADS[version]
  config.model.encoder.context.mlp_dim = MLP_DIM[version]
  config.model.encoder.context.num_layers = NUM_LAYERS[version] // 3
  config.model.encoder.context.attention_dropout_rate = 0.
  config.model.encoder.context.dropout_rate = 0.
  config.model.encoder.context.stochastic_depth = 0.
  # # Fused encoder
  config.model.encoder.fused = ml_collections.ConfigDict()
  config.model.encoder.fused.num_heads = NUM_HEADS[version]
  config.model.encoder.fused.mlp_dim = MLP_DIM[version]
  config.model.encoder.fused.num_layers = NUM_LAYERS[version] // 3
  config.model.encoder.fused.attention_dropout_rate = 0.
  config.model.encoder.fused.dropout_rate = 0.
  config.model.encoder.fused.stochastic_depth = 0.

  # Training.
  config.trainer_name = 'duplex_vit_classification_trainer'
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
  # Log eval summary (heavy due to global metrics.)
  config.log_eval_steps = 1000
  # Log training summary (rather light).
  config.log_summary_steps = 100
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
