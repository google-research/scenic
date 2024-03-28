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
r"""Default configs for CityScapes Semantic Segmentation.

"""
# pylint: enable=line-too-long

import ml_collections

_CITYSCAPES_TRAIN_SIZE = 2975


def get_config():
  """Returns the base experiment configuration for CityScapes."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'cityscapes'

  # Dataset.
  config.dataset_name = 'cityscapes'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.target_size = (512, 1024)  # Image size.

  # Model.
  config.model_name = 'unet_segmentation'
  config.model = ml_collections.ConfigDict()
  # config.model.block_size = (64, 128, 256, 512, 1024, 1024, 1024)
  config.model.block_size = (64, 128, 256, 512, 1024, 1024)
  config.model.use_batch_norm = True
  config.model.padding = 'SAME'

  # Trainer.
  config.trainer_name = 'segmentation_trainer'

  # Optimizer.
  config.batch_size = 32
  config.num_training_epochs = 400
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 1e-4
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = None
  config.label_smoothing = None
  config.class_rebalancing_factor = 0.0
  config.rng_seed = 0

  # Learning rate.
  config.steps_per_epoch = (_CITYSCAPES_TRAIN_SIZE //
                            config.get_ref('batch_size'))
  config.total_steps = (config.get_ref('num_training_epochs') *
                        config.get_ref('steps_per_epoch'))
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.base_learning_rate = 1e-3
  config.lr_configs.warmup_steps = 1 * config.get_ref('steps_per_epoch')
  # Setting 'steps_per_cycle' to total_steps basically means non-cycling cosine.
  config.lr_configs.steps_per_cycle = config.get_ref('total_steps')

  # Data type.
  config.model_dtype_str = 'float32'
  config.data_dtype_str = 'float32'

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  return config


