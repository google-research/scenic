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
r"""Default configs for Regularized ViT on ImageNet2012.

Based on: https://arxiv.org/pdf/2106.10270.pdf

"""
# pylint: disable=line-too-long

import ml_collections

_IMAGENET_TRAIN_SIZE = 1281167
NUM_CLASSES = 1000

VARIANT = 'B_16'


def get_default_adversarial_config():
  """Get default adversarial config."""
  # Adversarial Training
  config = ml_collections.ConfigDict()
  config.adversarial_augmentation_mode = ''
  config.advprop = ml_collections.ConfigDict()
  config.advprop.init_mode = 'zero'
  config.advprop.optimizer = 'GradientDescent'
  config.advprop.use_sign = True
  config.advprop.epsilon = 6.0 / 255.0
  config.advprop.num_steps = 5
  config.advprop.pyramid_sizes = '()'
  config.advprop.pyramid_scalars = '()'
  config.advprop.attack_fn_str = 'random_target'
  config.advprop.attack_in_train_mode = True
  config.advprop.aux_update_in_train_mode = True
  config.advprop.adv_loss_weight = 1.0
  config.advprop.aux_dropout_rate = 0.1  # equal to clean param
  config.advprop.aux_stochastic_depth = 0.1  # equal to clean param
  config.advprop.sd_direction = 'drop_late'
  config.advprop.aux_sd_direction = 'drop_late'

  advprop_lr_configs = ml_collections.ConfigDict()
  advprop_lr_configs.learning_rate_schedule = 'compound'
  advprop_lr_configs.factors = 'constant'
  advprop_lr_configs.warmup_steps = 0
  advprop_lr_configs.steps_per_cycle = 5
  advprop_lr_configs.base_learning_rate = 1.0 / 255.0
  config.advprop.lr_configs = advprop_lr_configs

  # Advprop has to know about the number of classes because it's not
  # reported in a consistent style in other types of configs.
  config.advprop.num_classes = 1000
  config.advprop.no_metrics = False  # for colab usage
  return config


def get_config(variant=VARIANT, runlocal=''):
  """Returns the ViT experiment configuration for ImageNet."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.update(get_default_adversarial_config())

  config.experiment_name = 'imagenet-regularized_vit'
  # Dataset.
  config.dataset_name = 'bit'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.rvt_aug_strength = 0
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
  version, patch = variant.split('_')
  config.model_name = 'vit_advtrain_multilabel_classification'
  config.model = ml_collections.ConfigDict()

  config.model.hidden_size = {'Ti': 192,
                              'S': 384,
                              'B': 768,
                              'L': 1024,
                              'H': 1280}[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [int(patch), int(patch)]
  config.model.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16, 'H': 16}[version]
  config.model.mlp_dim = {'Ti': 768,
                          'S': 1536,
                          'B': 3072,
                          'L': 4096,
                          'H': 5120}[version]
  config.model.num_layers = {'Ti': 12,
                             'S': 12,
                             'B': 12,
                             'L': 24,
                             'H': 32}[version]
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.0
  config.model.dropout_rate = 0.1
  config.model.stochastic_depth = 0.1
  config.model_dtype_str = 'float32'

  config.activation = ml_collections.ConfigDict()
  config.activation.activation_str = 'gelu'
  config.activation.activation_params = (0.0, 0.0)
  config.activation.application_str = '0-12'

  # Training.
  config.trainer_name = 'classification_adversarialtraining_trainer'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.optimizer = 'adamw'
  config.optimizer_configs.b1 = 0.9
  config.optimizer_configs.b2 = 0.999
  config.optimizer_configs.weight_decay = 0.1
  config.l2_decay_factor = None
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  config.num_training_epochs = 300
  config.log_eval_steps = 1000
  config.batch_size = 8 if runlocal else 4096
  config.rng_seed = 0
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
  config.write_summary = True  # write TB and/or XM summary
  config.write_xm_measurements = True  # write XM measurements
  config.xprof = True  # Profile using xprof
  config.checkpoint = True  # do checkpointing
  config.checkpoint_steps = 5000
  config.debug_train = False  # debug mode during training
  config.debug_eval = False  # debug mode during eval

  return config


def get_hyper(hyper):
  return hyper.product([
      hyper.sweep('config.advprop.attack_fn_str', ['random_target']),
      hyper.sweep('config.advprop.epsilon', [6 / 255]),
      hyper.sweep('config.advprop.lr_configs.base_learning_rate', [1 / 255]),
      hyper.sweep('config.advprop.num_steps', [5]),
      hyper.chainit([
          # Baseline, reg-vit
          hyper.product([
              hyper.sweep('config.adversarial_augmentation_mode', ['']),
          ]),
          # Pixel
          hyper.product([
              hyper.sweep('config.adversarial_augmentation_mode',
                          ['advprop_pyramid']),
              hyper.sweep('config.advprop.pyramid_sizes', ['(224)']),
              hyper.sweep('config.advprop.pyramid_scalars', ['(1)']),
          ]),
          # Pyramid
          hyper.product([
              hyper.sweep('config.adversarial_augmentation_mode',
                          ['advprop_pyramid']),
              hyper.sweep('config.advprop.pyramid_sizes', ['(7, 14, 224)']),
              hyper.sweep('config.advprop.pyramid_scalars', ['(20, 10, 1)']),
          ]),
      ]),
  ])
