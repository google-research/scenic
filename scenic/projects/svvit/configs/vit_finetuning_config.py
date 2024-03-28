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
r"""Default configs for ViT on structural variant classification using pileups.

"""

import ml_collections

_TRAIN_SIZE = 30_000 * 19
VERSION = 'S'  # Version has to match with the pretraining job.



def get_config(runlocal=''):
  """Returns the ViT experiment configuration for SV classification."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'sv-vit'

  # Dataset.
  config.dataset_name = 'pileup_window'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()

  # Model.
  config.model_name = 'vit_classification'
  config.model = ml_collections.ConfigDict()
  config.model.hidden_size = {
      'Ti': 192,
      'S': 384,
      'B': 768,
      'L': 1024,
      'H': 1280
  }[VERSION]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [1, 256]
  config.model.num_heads = {
      'Ti': 3,
      'S': 6,
      'B': 12,
      'L': 16,
      'H': 16,
  }[VERSION]
  config.model.mlp_dim = {
      'Ti': 768,
      'S': 1536,
      'B': 3072,
      'L': 4096,
      'H': 5120
  }[VERSION]
  config.model.num_layers = {
      'Ti': 12,
      'S': 12,
      'B': 12,
      'L': 24,
      'H': 32,
  }[VERSION]
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.
  config.model_dtype_str = 'float32'

  # Pretrained model info.
  config.init_from = ml_collections.ConfigDict()
  config.init_from.xm = None
  config.init_from.checkpoint_path = None

  # Training.
  config.trainer_name = 'transfer_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 0.1
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  config.num_training_epochs = 200
  config.log_eval_steps = 1000
  config.batch_size = 8 if runlocal else 512
  config.rng_seed = 42
  config.init_head_bias = -10.0

  # Learning rate.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant'
  config.lr_configs.base_learning_rate = None

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 5000
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  # Evaluation:
  config.global_metrics = [
      'truvari_recall_events',
      'truvari_precision_events',
      'truvari_recall',
      'truvari_precision',
      'gt_concordance',
      'nonref_concordance',
  ]

  if runlocal:
    config.count_flops = False
  return config


