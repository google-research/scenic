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

r"""MAE finetuning on ImageNet-1K.

"""

import ml_collections

_IMAGENET_TRAIN_SIZE = 1281167
NUM_CLASSES = 1000
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]

VARIANT = 'L/16'


def get_config(runlocal=''):
  """Returns the ViT experiment configuration for ImageNet."""

  runlocal = bool(runlocal)
  random_erase_prob = 0.25

  config = ml_collections.ConfigDict()
  config.experiment_name = 'imagenet-mae-vit'
  # Dataset.
  config.dataset_name = 'bit'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = 'imagenet2012'
  config.dataset_configs.num_classes = NUM_CLASSES
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.pp_train = (
      'decode_jpeg_and_inception_crop(224, 8, 100, resize_method="bicubic")'
      '|flip_lr'
      '|randaug(2, 15)'
      '|value_range(0, 1)'
      f'|standardize({MEAN_RGB}, {STDDEV_RGB})'
      f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
      f'|random_erase({random_erase_prob})'
      '|keep("image", "labels")')
  pp_eval = (
      'decode'
      '|resize_small(256, "bicubic")'
      '|central_crop(224)'
      '|value_range(0, 1)'
      f'|standardize({MEAN_RGB}, {STDDEV_RGB})'
      f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
      '|keep("image", "labels")')
  pp_eval_real = (
      'decode'
      '|resize_small(256, "bicubic")'
      '|central_crop(224)'
      '|value_range(0, 1)'
      f'|standardize({MEAN_RGB}, {STDDEV_RGB})'
      f'|onehot({NUM_CLASSES}, key="real_label", key_result="labels")'
      '|keep("image", "labels")')

  config.dataset_configs.val_split = [
      ('valid', 'imagenet2012', 'validation', pp_eval),
      ('imagenet-v2', 'imagenet_v2', 'test', pp_eval),
      ('imagenet-real', 'imagenet2012_real', 'validation', pp_eval_real),
      ('imagenet_adversarial', 'imagenet_a', 'test', pp_eval),
      ('imagenet_sketch', 'imagenet_sketch', 'test', pp_eval),
      ('imagenet_rendition', 'imagenet_r', 'test', pp_eval)
  ]
  config.dataset_configs.prefetch_to_device = 2
  # Shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = 250_000

  # Model.
  version, patch = VARIANT.split('/')
  config.model_name = 'vit_classification_mae'
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
  config.model.dropout_rate = 0.0
  config.model.stochastic_depth = {'B': 0.1, 'L': 0.2}[version]
  config.model_dtype_str = 'float32'
  config.model.positional_embedding = 'sinusoidal_2d'

  # Training.
  config.trainer_name = 'avmae_transfer_trainer'
  config.optimizer = 'adamw'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.b1 = 0.9
  config.optimizer_configs.b2 = 0.999
  config.optimizer_configs.weight_decay = 0.05
  config.optimizer_configs.skip_scale_and_bias_regularization = True
  config.optimizer_configs.layerwise_decay = 0.75

  config.max_grad_norm = None
  config.label_smoothing = 0  # Do it with Mixup, as done in TIMM.
  config.num_training_epochs = 50
  config.batch_size = 8 if runlocal else 1024
  config.rng_seed = 0

  # Learning rate.
  steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 4e-3
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 5 * steps_per_epoch
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = base_lr
  config.lr_configs.alpha = 1e-6 / base_lr

  # Mixup.
  config.mixup = ml_collections.ConfigDict()
  config.mixup.bind_to = None
  config.mixup.alpha = 0.8
  config.mixup.cutmix_alpha = 1.0
  config.mixup.cutmix_switch_prob = 0.5
  config.mixup.label_smoothing = 0.1

  # Logging.
  config.write_summary = True
  config.log_summary_steps = 100
  config.log_eval_steps = 2 * steps_per_epoch
  config.checkpoint_steps = steps_per_epoch
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  config.m = None  # Placeholder for randaug strength.
  config.l = None  # Placeholder for randaug layers.

  # Initialisation from checkpoint
  config.init_from = ml_collections.ConfigDict()
  NB: Set this path correctly to the pretrained checkpoint
  config.init_from.checkpoint_path = 'path_to_pretrained_checkpoint'
  return config


