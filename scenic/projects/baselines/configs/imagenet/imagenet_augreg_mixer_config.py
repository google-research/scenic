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
r"""Default configs for Mixer on ImageNet2012.


Based on: https://arxiv.org/abs/2105.01601

"""
# pylint: disable=line-too-long

import ml_collections

_IMAGENET_TRAIN_SIZE = 1281167
NUM_CLASSES = 1000

VARIANT = 'B/16'


def get_config(runlocal=''):
  """Returns the Mixer experiment configuration for ImageNet."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'imagenet-regularized_mixer'
  # Dataset.
  config.dataset_name = 'bit'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = 'imagenet2012'
  config.dataset_configs.num_classes = NUM_CLASSES
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.val_split = 'validation'
  config.dataset_configs.pp_train = (
      'decode_jpeg_and_inception_crop(224)|flip_lr'
      '|randaug(2, 15)'
      '|value_range(-1, 1)'
      f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
      '|keep("image", "labels")')
  config.dataset_configs.pp_eval = (
      'decode'
      '|resize_small(256)|central_crop(224)'
      '|value_range(-1, 1)'
      f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
      '|keep("image", "labels")')
  config.dataset_configs.prefetch_to_device = 2
  # Shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = 250_000

  # Model.
  version, patch = VARIANT.split('/')
  config.model_name = 'mixer_multilabel_classification'
  config.model = ml_collections.ConfigDict()
  config.model.hidden_size = {
      'Ti': 192,
      'S': 384,
      'B': 768,
      'L': 1024,
      'H': 1280}[version]
  config.model.patch_size = [int(patch), int(patch)]
  config.model.channels_mlp_dim = {
      'Ti': 768,
      'S': 1536,
      'B': 3072,
      'L': 4096,
      'H': 5120
  }[version]
  config.model.sequence_mlp_dim = {
      'Ti': 96,
      'S': 192,
      'B': 384,
      'L': 512,
      'H': 640
  }[version]
  config.model.num_layers = {
      'Ti': 12,
      'S': 12,
      'B': 12,
      'L': 24,
      'H': 32}[version]
  config.model.dropout_rate = 0.
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
  config.batch_size = 8 if runlocal else 4096
  config.rng_seed = 42
  config.init_head_bias = -6.9  # -log(1000)

  # Learning rate.
  steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 0.001 if version in {'L', 'H'} else 0.003
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*linear_warmup*linear_decay'
  config.lr_configs.total_steps = total_steps
  config.lr_configs.end_learning_rate = 1e-5
  config.lr_configs.warmup_steps = 10_000
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


