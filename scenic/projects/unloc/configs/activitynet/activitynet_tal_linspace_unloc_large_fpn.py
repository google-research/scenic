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

r"""Configs for finetuning the UnLoc-L model on ActivityNet TAL.

"""

import ml_collections
from scenic.projects.unloc import config_utils as unloc_config_utils

# Replace with the actual dataset size.
ACTIVITYNET_TAL_TRAIN_SIZE = 9941
MODEL_VARIANT = 'L/14x1'
FEATURE_PYRAMID_LEVELS = [1, 2, 3, 4, 5]
FEATURE_PYRAMID_DOWNSAMPLE_STRIDE = 2
NUM_FEATURES_LEVEL0 = 160


def get_config():
  """Returns the base experiment configuration."""
  config = ml_collections.ConfigDict()
  config.experiment_name = f'anet_unloc_clip_{MODEL_VARIANT}'

  # Dataset.
  config.dataset_name = 'temporal_localization_dataset'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.task = 'temporal_localization'
  config.dataset_configs.modality_configs = {
      'rgb':
          ml_collections.ConfigDict({
              'type': 'rgb',
              'min_resize': 256,
              'crop_size': 224,
              'zero_centering': False,
              'normalization_mean': [0.48145466, 0.4578275, 0.40821073],
              'normalization_std': [0.26862954, 0.26130258, 0.27577711],
          }),
  }
  config.dataset_configs.modality_configs.rgb.augmentation_params = (
      ml_collections.ConfigDict())
  config.dataset_configs.modality_configs.rgb.augmentation_params.do_jitter_scale = True
  config.dataset_configs.modality_configs.rgb.augmentation_params.scale_min_factor = 0.9
  config.dataset_configs.modality_configs.rgb.augmentation_params.scale_max_factor = 1.33
  config.dataset_configs.modality_configs.rgb.augmentation_params.prob_scale_jitter = 1.0
  config.dataset_configs.modality_configs.rgb.augmentation_params.do_color_augment = True
  config.dataset_configs.modality_configs.rgb.augmentation_params.prob_color_augment = 0.8
  config.dataset_configs.modality_configs.rgb.augmentation_params.prob_color_drop = 0.1
  config.dataset_configs.modality_configs.rgb.augmentation_params.do_rand_augment = True
  config.dataset_configs.modality_configs.rgb.augmentation_params.rand_augment_num_layers = 3
  config.dataset_configs.modality_configs.rgb.augmentation_params.rand_augment_magnitude = 10

  config.dataset_configs.num_frames = NUM_FEATURES_LEVEL0
  config.dataset_configs.stride = 1
  config.dataset_configs.sampling_strategy = 'linspace'
  config.dataset_configs.radius = 3.0
  # `duration`, `sampled_span`, or `none`.
  config.dataset_configs.displacement_normalizer = 'none'
  config.dataset_configs.max_num_segments = 24
  config.dataset_configs.feature_pyramid_config = ml_collections.ConfigDict()
  config.dataset_configs.feature_pyramid_config.num_features_level0 = (
      NUM_FEATURES_LEVEL0
  )
  config.dataset_configs.feature_pyramid_config.feature_pyramid_levels = (
      FEATURE_PYRAMID_LEVELS
  )
  config.dataset_configs.feature_pyramid_config.feature_pyramid_downsample_stride = (
      FEATURE_PYRAMID_DOWNSAMPLE_STRIDE
  )
  config.dataset_configs.feature_pyramid_config.normalize_displacements_by_downsample_stride = (
      True
  )
  config.dataset_configs.feature_pyramid_config.regression_ranges = [
      (0, 4),
      (4, 8),
      (8, 16),
      (16, 32),
      (32, float('inf')),
  ]

  config.dataset_configs.base_dir = '/path/to/base_dir'
  config.dataset_configs.tables = {
      'train': '',
      'validation': '',
      'test': '',
  }
  config.dataset_configs.num_classes = 200
  config.dataset_configs.num_prompts = 1
  config.dataset_configs.class_name_embedding_npy = '/path/to/embeddings'
  config.dataset_configs.include_video_id = True

  config.dataset_configs.do_multicrop_test = True  # Do during training.
  config.dataset_configs.test_batch_size = 4
  config.dataset_configs.secs_per_timestep = 0.1  # 10fps
  config.dataset_configs.log_test_epochs = 10
  config.dataset_configs.total_eval_epochs = 1.1

  # Model.
  config.model_name = 'unloc_temporal_localization'
  config.model = ml_collections.ConfigDict()
  config.model.classifier = 'token'
  config.model.video_tower_config = ml_collections.ConfigDict({
      'encoder_name': 'clip_video_encoder',
      'encoder_config': ml_collections.ConfigDict({
          'num_classes': -1,
          'image_encoder_config': unloc_config_utils.parse_image_encoder_config(
              MODEL_VARIANT
          ),
          'temporal_encoding_config': ml_collections.ConfigDict({
              'method': '3d_conv',
              'kernel_init_method': 'central_frame_initializer',
              'use_bias': False,
          }),
          'temporal_encoder_config': None,
          'final_endpoint': 'temporal_tokens',
          'classifier': 'token',
      }),
      'projection_size': 768,
      'projection_use_bias': False,
      'freeze': False,
  })
  config.model.video_tower_config.encoder_config.image_encoder_config.remat_block = (
      True
  )
  config.model.video_tower_config.encoder_config.image_encoder_config.classifier = 'gap'  # pylint:disable=line-too-long
  config.model.text_tower_config = ml_collections.ConfigDict({
      'encoder_name': 'pass_through_encoder',
      'encoder_config': {},
      'input_type': 'text_emb',
  })
  config.model.video_text_fusion_config = ml_collections.ConfigDict({
      'type': 'video_text_emb_self_attention',
      'config': ml_collections.ConfigDict({
          'self_attention_encoder_config': ml_collections.ConfigDict({
              'num_heads': 12,
              'mlp_dim': 3072,
              'num_layers': 6,
              'dropout_rate': 0.0,
              'attention_dropout_rate': 0.0,
              'stochastic_depth': 0.0,
              'positional_embedding': 'sinusoid',
              'downsample_strategy': 'max_pool',
              'feature_pyramid_config': ml_collections.ConfigDict({
                  'num_features_level0': NUM_FEATURES_LEVEL0,
                  'feature_pyramid_levels': FEATURE_PYRAMID_LEVELS,
                  'feature_pyramid_downsample_stride': (
                      FEATURE_PYRAMID_DOWNSAMPLE_STRIDE
                  ),
              }),
          }),
          'self_attention_encoder_name': 'simple_pyramid',
      }),
  })
  config.model.head_config = ml_collections.ConfigDict({
      'temporal_localization': ml_collections.ConfigDict({
          'type': 'query_dependent_localization_head',
          'config': ml_collections.ConfigDict({
              'num_conv_layers': 4,
              'kernel_size': 3,
              'init_classification_head_bias': -5.0,
              'init_regression_head_bias': 2.0,
              # 'relu', 'sigmoid', or 'relu_clip'
              'distance_normalizer': 'relu',
              'weight_sharing': True,
              'feature_pyramid_config': ml_collections.ConfigDict({
                  'num_features_level0': NUM_FEATURES_LEVEL0,
                  'feature_pyramid_levels': FEATURE_PYRAMID_LEVELS,
                  'feature_pyramid_downsample_stride': (
                      FEATURE_PYRAMID_DOWNSAMPLE_STRIDE
                  ),
              }),
          }),
      }),
  })

  # Training.
  config.trainer_name = 'single_task_trainer'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.optimizer = 'sgd'
  config.optimizer_configs.momentum = 0.9
  config.optimizer_configs.skip_scale_and_bias_regularization = True
  config.optimizer_configs.weight_decay = 0.0
  config.l2_decay_factor = 0
  config.max_grad_norm = 1.0
  config.label_smoothing = 0.0
  config.num_training_epochs = 30
  config.batch_size = 64
  config.rng_seed = 0
  config.count_flops = False

  config.init_from = ml_collections.ConfigDict()
  config.init_from.checkpoint_path = '/path/to/checkpoint'
  config.init_from.load_from_unloc_checkpoint = True
  config.init_from.load_image_tower = False
  config.init_from.load_text_tower = False

  # Learning rate.
  steps_per_epoch = ACTIVITYNET_TAL_TRAIN_SIZE // config.batch_size
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * piecewise_constant * linear_warmup'
  config.lr_configs.decay_events = [35 * steps_per_epoch]
  config.lr_configs.decay_factors = [0.1]
  config.lr_configs.warmup_steps = 5.0 * steps_per_epoch
  config.lr_configs.base_learning_rate = 0.01

  config.score_threshold = 0.001
  config.soft_nms_sigma = 0.3
  config.max_detections = 100
  # Setting it to True gives better results but slower.
  config.multiclass_nms = False

  config.classification_loss_alpha = 10.0
  config.classification_loss_type = 'focal'
  config.focal_loss_alpha = 0.25
  config.focal_loss_gamma = 2.0

  config.box_loss_type = 'iou'

  # Logging.
  config.checkpoint = True  # do checkpointing
  config.debug_train = False  # debug mode during training
  config.debug_eval = False  # debug mode during eval
  config.log_eval_steps = 400
  return config


