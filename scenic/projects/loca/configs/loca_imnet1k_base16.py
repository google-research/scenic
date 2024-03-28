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
"""Default config for LOCA training on ImageNet2012 for 100 epochs."""

import ml_collections

VARIANT = 'B/16'
_IMAGENET_TRAIN_SIZE = 1281167
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def get_config():
  """Returns the default config for a 100 epoch LOCA training on ImageNet2012."""

  config = ml_collections.ConfigDict()
  config.experiment_name = '100ep_run'
  # Dataset.
  config.dataset_name = 'loca_dataset'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.shuffle_buffer_size = 250_000
  reference_resolution = 224
  n_queries = 10
  config.dataset_configs.number_of_focal_queries = n_queries - 1
  config.dataset_configs.pp_train = (
      'decode' +
      '|copy("image", "reference")' +
      '|init_patch_matching_tracker(14, "target_mask")' +
      '|init_box_tracker("target_box")' +
      f'|cropflip_generatemask({reference_resolution}, 32, flip=False, inkey=("reference", "target_mask", "target_box"), outkey=("reference", "target_mask", "target_box"))' +
      '|value_range(0, 1, data_key="reference")' +
      '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="reference")' +
      '|random_grayscale(0.2, data_key="reference")' +
      '|random_blur(1.0, data_key="reference")' +
      f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="reference")' +
      ''.join([f'|copy("image", "query{i}")' for i in range(n_queries)]) +
      '|inception_crop_with_mask((224, 224), 32, 100, (14, 14), inkey=("query0", "target_mask", "target_box"), outkey=("query0", "query0_mask", "query0_box"))' +
      ''.join([f'|inception_crop_with_mask((96, 96), 5, 32, (6, 6), inkey=("query{i}", "target_mask", "target_box"), outkey=("query{i}", "query{i}_mask", "query{i}_box"))' for i in range(1, n_queries)]) +
      ''.join([f'|flip_with_mask(inkey=("query{i}", "query{i}_mask"), outkey=("query{i}", "query{i}_mask"))' for i in range(n_queries)]) +
      ''.join([f'|value_range(0, 1, data_key="query{i}")' for i in range(n_queries)]) +
      ''.join([f'|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="query{i}")' for i in range(n_queries)]) +
      ''.join([f'|random_grayscale(0.2, data_key="query{i}")' for i in range(n_queries)]) +
      ''.join([f'|random_blur(0.5, data_key="query{i}")' for i in range(1, n_queries)]) +
      '|random_blur(0.1, data_key="query0")|random_solarize(0.2, data_key="query0")' +
      ''.join([f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="query{i}")' for i in range(n_queries)]) +
      '|keep("reference"' + ''.join([f', "query{i}", "query{i}_box", "query{i}_mask"' for i in range(n_queries)]) + ')')
  # For IMAGENET-1K
  config.dataset_configs.dataset = 'imagenet2012'
  config.dataset_configs.train_split = 'train'

  # Model.
  version, patch = VARIANT.split('/')
  patch = int(patch)
  config.model = ml_collections.ConfigDict()
  config.model.hidden_size = {'Ti': 192,
                              'S': 384,
                              'B': 768,
                              'L': 1024,
                              'H': 1280}[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [patch, patch]
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
  config.model.head_output_dim = 4096
  config.model.attention_dropout_rate = 0.0
  config.model.dropout_rate = 0.0
  config.model.stochastic_depth = 0.1
  config.model_dtype_str = 'float32'
  config.model.temperature = 0.1
  config.sharpening = 0.05

  # LOCA specific parameters.
  config.n_ref_positions = int((reference_resolution // patch)**2)
  config.apply_cluster_loss = True
  config.reference_seqlen = int(0.2 * config.n_ref_positions)  # 20% of 196 is 39
  config.reference_seqlen_selection = 'consecutive'  # or 'unstructured' or 'first'
  config.query_max_seqlen = 70

  # Training.
  config.max_grad_norm = 1
  config.num_training_epochs = 100
  config.batch_size = 1024
  steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  config.rng_seed = 42
  total_steps = config.num_training_epochs * steps_per_epoch

  # Learning rate.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = steps_per_epoch * 15
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = 0.001 * config.batch_size / 1024
  config.lr_configs.alpha = 0.01

  # Weight decay.
  config.weight_decay = 0.1

  # Momentum rate scheduler.
  config.momentum_rate = ml_collections.ConfigDict()
  config.momentum_rate.factors = 'constant*cosine_decay'
  config.momentum_rate.steps_per_cycle = total_steps
  config.momentum_rate.base_learning_rate = 0.996
  config.momentum_rate.alpha = 1. / 0.996

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 5000
  config.log_summary_steps = 1000

  return config


