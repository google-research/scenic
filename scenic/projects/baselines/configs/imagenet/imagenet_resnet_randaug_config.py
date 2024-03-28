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
r"""Default configs for ResNet on ImageNet with randaugment.

"""

import ml_collections

_IMAGENET_TRAIN_SIZE = 1281167


def get_config():
  """Returns the base experiment configuration for ImageNet."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'imagenet_resnet'
  # Dataset.
  config.dataset_name = 'bit'
  config.data_dtype_str = 'float32'

  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = 'imagenet2012'
  # aka tiny_test/test[:5%] in task_adapt
  config.dataset_configs.train_split = 'train[:99%]'

  config.dataset_configs.num_classes = 1000
  INPUT_RES = 224  # pylint: disable=invalid-name
  RESIZE_RES = int(INPUT_RES * (256 / 224))  # pylint: disable=invalid-name
  LS = 1e-4  # pylint: disable=invalid-name
  config.dataset_configs.pp_train = (
      f'decode_jpeg_and_inception_crop_plus({INPUT_RES})|flip_lr'
      f'|randaug(2, 15)'
      f'|value_range(-1, 1)'
      f'|onehot({config.dataset_configs.num_classes},'
      f' key="label", key_result="labels", '
      f'on={1.0-LS}, off={LS})|keep("image", '
      f'"labels", "crop_hr", "crop_wr")')  # pylint: disable=line-too-long

  pp_eval_common = (
      f'decode|resize_small({RESIZE_RES})|'
      f'central_crop_plus({INPUT_RES})|value_range(-1, '
      f'1)|onehot({config.dataset_configs.num_classes},'
      ' key="{lbl}", '
      f'key_result="labels")|keep("image", '
      f'"labels", "crop_hr", "crop_wr")')  # pylint: disable=line-too-long

  pp_real = pp_eval_common.format(lbl='real_label')
  pp_val = pp_eval_common.format(lbl='label')

  config.dataset_configs.val_split = [
      ('valid', 'imagenet2012', 'train[99%:]', pp_val),
      ('test', 'imagenet2012', 'validation', pp_val),
      ('v2', 'imagenet_v2', 'test', pp_val),
      ('real', 'imagenet2012_real', 'validation', pp_real),
  ]
  config.dataset_configs.prefetch_to_device = 2
  # shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = 250_000

  # Model.
  config.model_name = 'resnet_classification'
  config.num_filters = 64
  config.num_layers = 50
  config.model_dtype_str = 'float32'

  # Training.
  config.trainer_name = 'classification_trainer'
  config.optimizer = 'momentum'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.momentum = 0.9
  config.l2_decay_factor = .00005
  config.max_grad_norm = None
  config.label_smoothing = None
  config.num_training_epochs = 300
  config.batch_size = 8192
  config.rng_seed = 0
  config.init_head_bias = -10.0

  # Learning rate.
  steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 0.1 * config.batch_size / 256
  # setting 'steps_per_cycle' to total_steps basically means non-cycling cosine.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 7 * steps_per_epoch
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = base_lr

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 10 * steps_per_epoch
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  return config


