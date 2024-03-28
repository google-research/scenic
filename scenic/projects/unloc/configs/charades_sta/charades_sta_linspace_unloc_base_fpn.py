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

r"""Configs for finetuning UnLoc-B model on Charades-STA.

"""

import ml_collections
from scenic.projects.unloc import config_utils as unloc_config_utils

# Replace with the actual dataset size.
CHARADES_STA_TRAIN_SIZE = 12408
MODEL_VARIANT = 'B/16x1'
FEATURE_PYRAMID_LEVELS = [2, 3, 4, 5]
FEATURE_PYRAMID_DOWNSAMPLE_STRIDE = 2
NUM_FEATURES_LEVEL0 = 128


def get_config():
  """Returns the base experiment configuration."""
  config = ml_collections.ConfigDict()
  config.experiment_name = f'charades_sta_unloc_clip_{MODEL_VARIANT}'

  # Dataset.
  config.dataset_name = 'moment_retrieval_dataset'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.task = 'moment_retrieval'
  config.dataset_configs.name = 'charades_sta'
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
      'caption':
          ml_collections.ConfigDict({
              'type': 'text',
              'max_num_tokens': 16,
              'tokenizer_type': 'clip',
              'input_feature_name': 'segment/label/string',
          }),
  }
  config.dataset_configs.train_max_num_captions = 1
  config.dataset_configs.eval_max_num_captions = 1

  config.dataset_configs.num_frames = NUM_FEATURES_LEVEL0
  config.dataset_configs.stride = 1
  config.dataset_configs.sampling_strategy = 'linspace'
  config.dataset_configs.radius = 3.0
  # `duration`, `sampled_span`, or `none`.
  config.dataset_configs.displacement_normalizer = 'none'
  config.dataset_configs.feature_pyramid_config = ml_collections.ConfigDict()
  config.dataset_configs.feature_pyramid_config.num_features_level0 = NUM_FEATURES_LEVEL0  # pylint:disable=line-too-long
  config.dataset_configs.feature_pyramid_config.feature_pyramid_levels = FEATURE_PYRAMID_LEVELS  # pylint:disable=line-too-long
  config.dataset_configs.feature_pyramid_config.feature_pyramid_downsample_stride = FEATURE_PYRAMID_DOWNSAMPLE_STRIDE  # pylint:disable=line-too-long
  config.dataset_configs.feature_pyramid_config.normalize_displacements_by_downsample_stride = True  # pylint:disable=line-too-long
  config.dataset_configs.feature_pyramid_config.regression_ranges = [
      (0, 4),
      (4, 8),
      (8, 16),
      (16, float('inf')),
  ]

  config.dataset_configs.base_dir = '/path/to/base_dir'
  config.dataset_configs.tables = {
      'train': '',
      'validation': '',
      'test': '',
  }
  config.dataset_configs.num_classes = 1
  config.dataset_configs.include_video_id = True
  config.dataset_configs.is_video_id_int = True
  config.dataset_configs.vid_input_feature_name = 'example_id'

  config.dataset_configs.do_multicrop_test = True  # Do during training.
  config.dataset_configs.test_batch_size = 4
  config.dataset_configs.secs_per_timestep = 0.1  # 10fps
  config.dataset_configs.log_test_epochs = 5
  config.dataset_configs.total_eval_epochs = 1.1

  # Model.
  config.model_name = 'unloc_moment_retrieval'
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
      'projection_size': 512,
      'projection_use_bias': False,
      'freeze': False,
  })
  # config.model.video_tower_config.encoder_config.image_encoder_config.classifier = 'gap'  # pylint:disable=line-too-long
  config.model.text_tower_config = ml_collections.ConfigDict({
      'encoder_name': 'clip_text_encoder',
      'encoder_config': unloc_config_utils.parse_text_encoder_config(
          MODEL_VARIANT
      ),
      'projection_size': 512,
      'projection_use_bias': False,
      'freeze': True,
  })
  config.model.video_text_fusion_config = ml_collections.ConfigDict({
      'type':
          'video_text_self_attention',
      'config':
          ml_collections.ConfigDict({
              'self_attention_encoder_config':
                  ml_collections.ConfigDict({
                      'num_heads':
                          8,
                      'mlp_dim':
                          2048,
                      'num_layers':
                          6,
                      'dropout_rate':
                          0.0,
                      'attention_dropout_rate':
                          0.0,
                      'stochastic_depth':
                          0.0,
                      'positional_embedding':
                          'sinusoid',
                      'downsample_strategy':
                          'max_pool',
                      'feature_pyramid_config':
                          ml_collections.ConfigDict({
                              'num_features_level0':
                                  NUM_FEATURES_LEVEL0,
                              'feature_pyramid_levels':
                                  FEATURE_PYRAMID_LEVELS,
                              'feature_pyramid_downsample_stride':
                                  FEATURE_PYRAMID_DOWNSAMPLE_STRIDE,
                          }),
                  }),
              'self_attention_encoder_name':
                  'fpn',
              'text_tower_classifier':
                  'eos',
              'use_all_text_tokens':
                  False
          }),
  })
  config.model.head_config = ml_collections.ConfigDict({
      'moment_retrieval':
          ml_collections.ConfigDict({
              'type':
                  'query_dependent_localization_head',
              'config':
                  ml_collections.ConfigDict({
                      'num_conv_layers':
                          3,
                      'kernel_size':
                          3,
                      'init_classification_head_bias':
                          0.0,
                      'distance_normalizer':
                          'relu',
                      'weight_sharing':
                          False,
                      'feature_pyramid_config':
                          ml_collections.ConfigDict({
                              'num_features_level0':
                                  NUM_FEATURES_LEVEL0,
                              'feature_pyramid_levels':
                                  FEATURE_PYRAMID_LEVELS,
                              'feature_pyramid_downsample_stride':
                                  FEATURE_PYRAMID_DOWNSAMPLE_STRIDE,
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
  config.all_gather_loss = True

  config.init_from = ml_collections.ConfigDict()
  config.init_from.checkpoint_path = '/path/to/checkpoint'
  config.init_from.load_from_unloc_checkpoint = True
  config.init_from.load_image_tower = False
  config.init_from.load_text_tower = False

  # Learning rate.
  steps_per_epoch = CHARADES_STA_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  # config.lr_configs.factors = 'constant * piecewise_constant * linear_warmup'
  config.lr_configs.factors = 'constant * linear_warmup'
  config.lr_configs.warmup_steps = 5.0 * steps_per_epoch
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = 0.01
  config.layer_prefix_to_base_lrs = {
      'moment_retrieval_head': 0.1,
      'video_text_fusion': 0.1
  }

  config.score_threshold = 0.001
  config.iou_threshold = 0.5
  config.soft_nms_sigma = 0.3
  config.max_detections = 100
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
  config.log_eval_steps = 200
  return config


