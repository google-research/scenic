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

r"""Configs for finetuning a MTV B/2 model on Epic-Kitchens.

"""

import ml_collections
from scenic.projects.mtv import config_utils

EPIC_TRAIN_SIZE = 67217
EPIC_VALID_SIZE = 9668
MODEL_VARIANT = 'Ti/8+S/4+B/2'


def get_config():
  """Returns the base experiment configuration."""
  config = ml_collections.ConfigDict()
  config.experiment_name = f'epic_mtv_{MODEL_VARIANT}'

  # Dataset.
  config.dataset_name = 'video_tfrecord_dataset'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.base_dir = (
      '/path/to/dataset')
  config.dataset_configs.tables = {
      'train': 'train.tfrecord@1024',
      'validation': 'validation.tfrecord@1024',
      'test': 'test.tfrecord@1024'
  }
  config.dataset_configs.examples_per_subset = {
      'train': EPIC_TRAIN_SIZE,
      'validation': EPIC_VALID_SIZE,
      'test': EPIC_VALID_SIZE
  }
  config.dataset_configs.num_frames = 64
  config.dataset_configs.stride = 1
  config.dataset_configs.min_resize = 256
  config.dataset_configs.crop_size = 224
  config.dataset_configs.class_splits = [300, 97]
  config.dataset_configs.split_names = ['noun', 'verb']
  config.dataset_configs.one_hot_labels = True
  config.dataset_configs.zero_centering = True
  config.data_dtype_str = 'float32'

  # Multicrop eval settings
  config.dataset_configs.test_on_val = True
  config.dataset_configs.do_multicrop_test = True  # Do during training.
  config.dataset_configs.do_three_spatial_crops = False  # Do during training.
  config.dataset_configs.log_test_epochs = 10
  # The effective batch size per host when testing is
  # num_test_clips * test_batch_size
  config.dataset_configs.num_test_clips = 4
  config.dataset_configs.test_batch_size = 8  # Needs to be num_local_devices
  config.multicrop_clips_per_device = 2

  config.dataset_configs.augmentation_params = ml_collections.ConfigDict()
  config.dataset_configs.augmentation_params.do_jitter_scale = True
  config.dataset_configs.augmentation_params.scale_min_factor = 0.9
  config.dataset_configs.augmentation_params.scale_max_factor = 1.33
  config.dataset_configs.augmentation_params.prob_scale_jitter = 1.0
  config.dataset_configs.augmentation_params.do_color_augment = True
  config.dataset_configs.augmentation_params.prob_color_augment = 0.8
  config.dataset_configs.augmentation_params.prob_color_drop = 0.1
  config.dataset_configs.augmentation_params.do_rand_augment = True
  config.dataset_configs.augmentation_params.rand_augment_num_layers = 3
  config.dataset_configs.augmentation_params.rand_augment_magnitude = 10
  config.dataset_configs.prefetch_to_device = 2

  # Model.
  config.model_name = 'mtv_multihead_classification'
  config.model = ml_collections.ConfigDict()
  config.model.classifier = 'token'
  config.model.dropout_rate = 0.0
  config.model.attention_dropout_rate = 0.0
  config.model.stochastic_depth = 0.1
  config.model.view_configs = config_utils.parse_view_configs(MODEL_VARIANT)
  # config.model.cross_view_fusion = None
  config.model.cross_view_fusion = ml_collections.ConfigDict({
      'type': 'cross_view_attention',
      'fuse_in_descending_order': True,
      'use_query_config': True,
      'fusion_layers': (5, 11),
  })
  config.model.global_encoder_config = ml_collections.ConfigDict({
      'num_heads': 8,
      'mlp_dim': 3072,
      'num_layers': 12,
      'hidden_size': 768,
      'merge_axis': 'channel',
  })
  config.model.temporal_encoding_config = ml_collections.ConfigDict({
      'kernel_init_method': 'central_frame_initializer',
      'method': '3d_conv',
  })

  # Training
  config.trainer_name = 'mtv_trainer'
  config.optimizer = 'momentum'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.l2_decay_factor = 0
  config.max_grad_norm = 1
  config.label_smoothing = 0.2
  config.num_training_epochs = 80
  config.batch_size = 128
  config.rng_seed = 0

  config.init_from = ml_collections.ConfigDict()
  # Use a pretrained MTV-B checkpoint
  config.init_from.checkpoint_path = 'path_to_checkpoint'
  config.init_from.model_type = 'mtv'
  config.init_from.restore_positional_embedding = True
  config.init_from.positional_embed_size_change = 'resize'

  # Learning rate
  steps_per_epoch = EPIC_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 2.5 * steps_per_epoch
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = 2e-1

  # Logging
  config.log_eval_steps = 3000
  config.write_summary = True  # write TB and/or XM summary
  config.write_xm_measurements = True  # write XM measurements
  config.checkpoint = True  # do checkpointing
  config.debug_train = False  # debug mode during training
  config.debug_eval = False  # debug mode during eval
  return config


