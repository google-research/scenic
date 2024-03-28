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

r"""Configs for finetuning the UnLoc-L model on COIN.

"""

import ml_collections
from scenic.projects.unloc import config_utils as unloc_config_utils

# Replace with the actual dataset size.
COIN_TRAIN_SIZE = 8461
MODEL_VARIANT = 'L/14x1'


def get_config():
  """Returns the base experiment configuration."""
  config = ml_collections.ConfigDict()
  config.experiment_name = f'coin_action_seg_unloc_clip_{MODEL_VARIANT}'

  # Dataset.
  config.dataset_name = 'action_segmentation_dataset'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.task = 'action_segmentation'
  config.dataset_configs.modality_configs = {
      'rgb': ml_collections.ConfigDict({
          'type': 'rgb',
          'min_resize': 256,
          'crop_size': 224,
          'zero_centering': False,
          'normalization_mean': [0.48145466, 0.4578275, 0.40821073],
          'normalization_std': [0.26862954, 0.26130258, 0.27577711],
      }),
  }
  config.dataset_configs.modality_configs.rgb.augmentation_params = (
      ml_collections.ConfigDict()
  )
  config.dataset_configs.modality_configs.rgb.augmentation_params.do_jitter_scale = (
      True
  )
  config.dataset_configs.modality_configs.rgb.augmentation_params.scale_min_factor = (
      0.9
  )
  config.dataset_configs.modality_configs.rgb.augmentation_params.scale_max_factor = (
      1.33
  )
  config.dataset_configs.modality_configs.rgb.augmentation_params.prob_scale_jitter = (
      1.0
  )
  config.dataset_configs.modality_configs.rgb.augmentation_params.do_color_augment = (
      True
  )
  config.dataset_configs.modality_configs.rgb.augmentation_params.prob_color_augment = (
      0.8
  )
  config.dataset_configs.modality_configs.rgb.augmentation_params.prob_color_drop = (
      0.1
  )
  config.dataset_configs.modality_configs.rgb.augmentation_params.do_rand_augment = (
      True
  )
  config.dataset_configs.modality_configs.rgb.augmentation_params.rand_augment_num_layers = (
      3
  )
  config.dataset_configs.modality_configs.rgb.augmentation_params.rand_augment_magnitude = (
      10
  )

  config.dataset_configs.num_frames = 512
  config.dataset_configs.max_num_segments = 28

  config.dataset_configs.base_dir = '/path/to/base_dir'
  config.dataset_configs.tables = {
      'train': '',
      'validation': '',
      'test': '',
  }
  config.dataset_configs.num_classes = 778
  # This file contains text embeddings from class names augmented by prompts.
  config.dataset_configs.class_name_embedding_npy = '/path/to/embeddings'
  config.dataset_configs.include_video_id = True

  config.dataset_configs.do_multicrop_test = False  # Do during training.
  config.dataset_configs.test_batch_size = 8
  config.dataset_configs.secs_per_timestep = 0.1  # 10fps
  config.dataset_configs.log_test_epochs = 10
  config.dataset_configs.total_eval_epochs = 1.1

  # Model.
  config.model_name = 'unloc_action_segmentation'
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
  config.model.video_tower_config.encoder_config.image_encoder_config.classifier = (
      'gap'
  )
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
              'mlp_dim': 768 * 4,
              'num_layers': 6,
              'dropout_rate': 0.0,
              'attention_dropout_rate': 0.0,
              'stochastic_depth': 0.0,
              'positional_embedding': 'sinusoid',
              'remat_block': True,
          }),
          'self_attention_encoder_name': 'transformer',
      }),
  })
  config.model.head_config = ml_collections.ConfigDict({
      'action_segmentation': ml_collections.ConfigDict({
          'type': 'linear_head',
          'config': ml_collections.ConfigDict({
              'init_head_bias': -6.6,
              'output_dim': 1,
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
  config.num_training_epochs = 55
  config.batch_size = 64
  config.rng_seed = 0
  config.count_flops = False

  config.init_from = ml_collections.ConfigDict()
  config.init_from.checkpoint_path = '/path/to/checkpoint'
  config.init_from.load_from_unloc_checkpoint = True
  config.init_from.load_image_tower = False
  config.init_from.load_text_tower = False

  # Learning rate.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * exponential_decay * linear_warmup'
  config.lr_configs.decay_steps = 6600  # 50th epoch
  config.lr_configs.decay_rate = 0.1
  config.lr_configs.staircase = True
  config.lr_configs.warmup_steps = 500
  config.lr_configs.base_learning_rate = 0.01
  config.layer_prefix_to_base_lrs = {
      'action_segmentation_head': 0.1,
      'video_text_fusion': 0.1,
  }

  # Logging.
  config.checkpoint = True  # do checkpointing
  config.debug_train = False  # debug mode during training
  config.debug_eval = False  # debug mode during eval
  config.log_eval_steps = 600
  return config


