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
r"""NCR config on mini-ImageNet-Blue with 40% noise ratio.

Based on: https://arxiv.org/abs/2202.02200

"""
# pylint: disable=line-too-long

import ml_collections

MINI_IMAGENET_BLUE_TRAIN_SIZE = 60000
NUM_CLASSES = 100


def get_config(runlocal=''):
  """Returns the experiment configuration."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'ncr'
  # Dataset.
  config.dataset_name = 'bit'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = 'controlled_noisy_web_labels/mini_imagenet_blue'
  config.dataset_configs.num_classes = NUM_CLASSES
  config.dataset_configs.train_split = 'train_40'   # Choose between train_00, train_20, train_40, train_80
  config.dataset_configs.val_split = 'validation'
  config.dataset_configs.pp_train = (
      'decode_jpeg_and_inception_crop(224)'
      '|value_range(-1, 1)'
      '|flip_lr'
      f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
      '|keep("image", "labels")')
  config.dataset_configs.pp_eval = (
      'decode'
      '|value_range(-1, 1)'
      '|resize_small(256)|central_crop(224)'
      f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
      '|keep("image", "labels")')
  config.dataset_configs.prefetch_to_device = 2
  # Shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = 1000

  # Model.
  config.model_name = 'resnet'
  config.num_filters = 64
  config.num_layers = 18
  config.model_dtype_str = 'float32'

  # Training.
  config.trainer_name = 'classification_trainer'
  config.optimizer = 'momentum'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.momentum = 0.9
  config.optimizer_configs.weight_decay = 0.0005
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = None
  config.label_smoothing = None
  config.num_training_epochs = 130
  config.log_eval_steps = 5000
  config.batch_size = 8 if runlocal else 128
  config.rng_seed = 1

  # Learning rate.
  steps_per_epoch = MINI_IMAGENET_BLUE_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.warmup_steps = steps_per_epoch * 5
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = 0.1

  # NCR specific flags
  config.loss_type = 'ncr'  # Options: cross_entropy or ncr
  config.ncr = ml_collections.ConfigDict()
  config.ncr.ncr_feature = 'pre_logits'
  config.ncr.number_neighbours = 100
  config.ncr.smoothing_gamma = 1
  config.ncr.temperature = 2.0
  config.ncr.loss_weight = 0.3
  config.ncr.starting_epoch = 10

  # Mixup.
  config.mixup = ml_collections.ConfigDict()
  config.mixup.bind_to = None
  config.mixup.alpha = 0.0  # Set to 0.5 to enable mixup

  # Logging.
  config.write_summary = True  # write TB and/or XM summary
  config.write_xm_measurements = True  # write XM measurements
  config.xprof = True  # Profile using xprof
  config.checkpoint = True  # do checkpointing
  config.checkpoint_steps = 1000
  config.log_summary_steps = 500
  config.debug_train = False  # debug mode during training
  config.debug_eval = False  # debug mode during eval


  return config


