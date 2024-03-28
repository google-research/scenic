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
r"""Default configs for Plain ViT on ImageNet2012.

Based on: https://arxiv.org/pdf/2205.01580.pdf

"""
# pylint: disable=line-too-long

import ml_collections

_IMAGENET_TRAIN_SIZE = 1281167
NUM_CLASSES = 1000

VARIANT = 'S/16'

HIDDEN_SIZES = {
    'Ti': 192,
    'S': 384,
    'M': 512,
    'B': 768,
    'L': 1024,
    'H': 1280,
    'g': 1408,
    'G': 1664,
    'e': 1792
}
MLP_DIMS = {
    'Ti': 768,
    'S': 1536,
    'M': 2048,
    'B': 3072,
    'L': 4096,
    'H': 5120,
    'g': 6144,
    'G': 8192,
    'e': 15360
}
NUM_HEADS = {
    'Ti': 3,
    'S': 6,
    'M': 8,
    'B': 12,
    'L': 16,
    'H': 16,
    'g': 16,
    'G': 16,
    'e': 16
}
NUM_LAYERS = {
    'Ti': 12,
    'S': 12,
    'M': 12,
    'B': 12,
    'L': 24,
    'H': 32,
    'g': 40,
    'G': 48,
    'e': 56
}


def get_config(runlocal=''):
  """Returns the ViT experiment configuration for ImageNet."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'imagenet-regularized_vit'
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
      '|randaug(2, 10)'
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
  config.model_name = 'plainvit'
  config.model = ml_collections.ConfigDict()
  config.model.hidden_size = HIDDEN_SIZES[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [int(patch), int(patch)]
  config.model.num_heads = NUM_HEADS[version]
  config.model.mlp_dim = MLP_DIMS[version]
  config.model.num_layers = NUM_LAYERS[version]
  config.model.dropout_rate = 0.
  config.model.classifier = 'map'
  config.model.representation_size = None
  config.model.positional_embedding = 'learn'
  config.init_head_bias = -6.9  # -log(1000)

  # Training.
  config.trainer_name = 'plainvit_trainer'
  config.loss = 'sigmoid_xent'
  config.l2_decay_factor = None
  config.label_smoothing = None
  config.num_training_epochs = 300
  config.log_eval_steps = 1000
  config.batch_size = 8 if runlocal else 1024
  config.rng_seed = 42
  sched = ml_collections.ConfigDict()
  sched.re = '(.*)'
  sched.lr_configs = ml_collections.ConfigDict()
  sched.lr_configs.learning_rate_schedule = 'compound'
  sched.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  sched.lr_configs.total_steps = config.num_training_epochs * steps_per_epoch
  sched.lr_configs.steps_per_cycle = sched.lr_configs.total_steps
  sched.lr_configs.warmup_steps = 10_000
  sched.lr_configs.base_learning_rate = 0.001
  sched.lr_configs.timescale = 10_000
  config.schedule = ml_collections.ConfigDict({'all': sched})

  # *Single* optimizer.
  optim = ml_collections.ConfigDict()
  optim.optax_name = 'scale_by_adam'
  # optim.optax = dict(mu_dtype='bfloat16')
  optim.optax_configs = ml_collections.ConfigDict(
      {  # Optimizer settings.
          'b1': 0.9,
          'b2': 0.999,
      })
  config.optax = dict(mu_dtype='bfloat16')
  optim.max_grad_norm = 1.0

  optim.weight_decay = 0.0001
  optim.weight_decay_decouple = True
  config.optimizer = optim

  # Mixup.
  config.mixup = ml_collections.ConfigDict()
  config.mixup.bind_to = None
  config.mixup.alpha = 0.2

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


