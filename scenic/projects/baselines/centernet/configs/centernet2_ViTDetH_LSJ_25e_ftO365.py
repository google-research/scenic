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
r"""Default configs for COCO detection using CenterNet.

"""

import ml_collections


def get_config():
  """get config."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'centernet2_ViTDetH_LSJ_25e_ftO365'

  # Dataset.
  config.dataset_name = 'coco_centernet_detection'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.shuffle_buffer_size = 10_000
  config.dataset_configs.max_boxes = 100
  config.dataset_configs.scale_range = (0.1, 2.0)
  config.dataset_configs.crop_size = 1024
  config.dataset_configs.size_divisibility = 32
  config.dataset_configs.remove_crowd = True
  config.data_dtype_str = 'float32'

  config.rng_seed = 0

  # Model.
  config.model = ml_collections.ConfigDict()
  config.model.model_dtype_str = 'float32'
  config.model.model_name = 'centernet2'
  config.model.backbone_name = 'vitdet'
  config.model.num_classes = -1
  config.model.strides = (8, 16, 32, 64, 128)
  config.model.pixel_mean = (123.675, 116.28, 103.53)
  config.model.pixel_std = (58.395, 57.12, 57.375)
  config.model.backbone_args = ml_collections.ConfigDict()
  config.model.backbone_args.size = 'H'
  config.model.backbone_args.embed_dim = 1280
  config.model.backbone_args.depth = 32
  config.model.backbone_args.num_heads = 16
  config.model.backbone_args.drop_path_rate = 0.5
  config.model.backbone_args.window_block_indexes = (
      list(range(0, 7)) + list(range(8, 15)) + list(range(16, 23)) + list(range(24, 31))
  )
  config.model.freeze_model_state = False

  # CenterNet2 parameters
  config.model.hm_weight = 0.5
  config.model.reg_weight = 1.0
  config.model.score_thresh = 0.05
  config.model.pre_nms_topk_train = 2000
  config.model.post_nms_topk_train = 1000
  config.model.pre_nms_topk_test = 1000
  config.model.post_nms_topk_test = 256
  config.model.iou_thresh = 0.9
  config.model.roi_matching_threshold = (0.6, 0.7, 0.8)
  config.model.roi_nms_threshold = 0.7

  # optimizer
  config.optimizer = ml_collections.ConfigDict()
  config.optimizer.optimizer = 'adamw'
  config.optimizer.b1 = 0.9
  config.optimizer.b2 = 0.999
  config.optimizer.weight_decay = 0.1
  config.optimizer.skip_scale_and_bias_regularization = True
  config.optimizer.layerwise_decay = 0.9
  config.optimizer.num_layers = 32
  config.optimizer.decay_layer_prefix = 'backbone/net/blocks.'
  config.optimizer.decay_stem_layers = ['patch_embed.proj', 'pos_embed']

  # learning rate and training schedule
  config.num_training_steps = 184375 * 1 // 4
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * piecewise_constant * linear_warmup'
  config.lr_configs.decay_events = [163889 * 1 // 4, 177546 * 1 // 4]
  config.lr_configs.decay_factors = [0.1, 0.01]
  config.lr_configs.warmup_steps = 250
  config.lr_configs.base_learning_rate = 0.00005

  # Pretrained_backbone.
  train config centernet2_O365_ViTDetH_LSJ_75e or download at
  https://storage.googleapis.com/scenic-bucket/centernet/
  centernet2_O365_ViTDetH_LSJ_75e/checkpoint
  config.weights = '/path/to/centernet2_O365_ViTDetH_LSJ_75e/'
  config.skip_wrong_shape = True
  config.checkpoint_steps = 500
  config.log_eval_steps = 5000

  # Training.
  config.batch_size = 64

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.log_summary_steps = 20  # train summary steps
  config.log_large_summary_steps = 1000  # Expensive summary operations freq
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  return config


