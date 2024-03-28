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
r"""Masked Autoencoder on ImageNet-1K.

```

"""
# pylint: disable=line-too-long

import copy
import ml_collections


_IMAGENET_TRAIN_SIZE = 1281167
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]
VARIANT = 'L/16'


def get_config(runlocal=''):
  """Returns the ViT experiment configuration for ImageNet."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'imagenet-mae-vit'

  # Dataset.
  config.dataset_name = 'bit'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = 'imagenet2012'
  config.dataset_configs.val_split = 'validation'
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.num_classes = 1000
  INPUT_RES = 224  # pylint: disable=invalid-name
  RESIZE_RES = int(INPUT_RES * (256 / 224))  # pylint: disable=invalid-name

  config.dataset_configs.pp_train = (
      f'decode_jpeg_and_inception_crop({INPUT_RES}, 20, 100, resize_method="bicubic")'
      '|flip_lr'
      '|value_range(0, 1)'
      f'|standardize({MEAN_RGB}, {STDDEV_RGB})'
      f'|onehot({config.dataset_configs.num_classes}, key="label", key_result="labels")'   # pylint: disable=line-too-long
      f'|keep("image", "labels")')
  config.dataset_configs.pp_eval = (
      f'decode'
      f'|resize_small({RESIZE_RES}, "bicubic")'
      f'|central_crop({INPUT_RES})'
      '|value_range(0, 1)'
      f'|standardize({MEAN_RGB}, {STDDEV_RGB})'
      f'|onehot({config.dataset_configs.num_classes}, key="label", key_result="labels")'   # pylint: disable=line-too-long
      f'|keep("image", "labels")')
  config.dataset_configs.prefetch_to_device = 2

  # shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = 250_000

  # Model.
  version, patch = VARIANT.split('/')
  config.model_name = 'vit_masked_autoencoder'
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
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.
  config.model.positional_embedding = 'sinusoidal_2d'
  config.model.positional_embedding_decoder = 'sinusoidal_2d'

  # Taken from https://github.com/facebookresearch/mae/blob/main/models_mae.py#L223
  config.model.decoder_config = ml_collections.ConfigDict()
  config.model.decoder_config.hidden_size = {
      'B': 512,
      'L': 512,
      'H': 512
  }[version]
  config.model.decoder_config.num_layers = {
      'B': 8,
      'L': 8,
      'H': 8
  }[version]
  config.model.decoder_config.num_heads = {
      'B': 16,
      'L': 16,
      'H': 16
  }[version]
  config.model.decoder_config.mlp_dim = {
      'B': 2048,
      'L': 2048,
      'H': 2048
  }[version]
  config.model.decoder_config.dropout_rate = 0
  config.model.decoder_config.attention_dropout_rate = 0
  config.model.decoder_config.stochastic_depth = 0

  config.model_dtype_str = 'float32'

  # Masked Feature loss
  config.masked_feature_loss = ml_collections.ConfigDict()
  config.masked_feature_loss.target = 'rgb'
  config.masked_feature_loss.token_mask_probability = 0.75
  config.masked_feature_loss.normalise_by_output_dimension = True
  config.model.classifier = 'token'
  config.masked_feature_loss.standardise_per_patch = True

  # Training.
  config.trainer_name = 'feature_regression_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.95
  config.optimizer_configs.weight_decay = 0
  config.explicit_weight_decay = 0.05
  config.l2_decay_factor = None
  config.max_grad_norm = None
  config.label_smoothing = None
  config.num_training_epochs = 800
  config.batch_size = 8 if runlocal else 4096
  config.rng_seed = 0
  config.init_head_bias = 0.0  # -10.0

  # Learning rate.
  steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 1.5e-4 * config.batch_size / 256
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*linear_warmup*cosine_decay'
  config.lr_configs.total_steps = total_steps
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.end_learning_rate = 0
  config.lr_configs.warmup_steps = 40 * steps_per_epoch
  config.lr_configs.base_learning_rate = base_lr

  # Fewshot.
  config.fewshot = common_fewshot.get_config(config.batch_size)
  config.fewshot.representation_layer = 'representation'
  config.fewshot.log_eval_steps = 5 * steps_per_epoch

  # Logging.
  config.write_summary = True
  config.log_summary_steps = 100
  config.checkpoint = True  # Do checkpointing.
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.


  return config


