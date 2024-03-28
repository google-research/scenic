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
# pylint: enable=line-too-long

import ml_collections


def get_config():
  """get config."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'centernet_ViTDetB_LSJ_4x'

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
  config.model.model_name = 'centernet'
  config.model.backbone_name = 'vitdet'
  config.model.num_classes = 80
  config.model.strides = (8, 16, 32, 64, 128)
  config.model.pixel_mean = (103.530, 116.280, 123.675)
  config.model.pixel_std = (57.375, 57.120, 58.395)  # For most other models
  config.model.backbone_args = ml_collections.ConfigDict()
  config.model.backbone_args.drop_path_rate = 0.1
  config.model.freeze_model_state = False

  # Evaluation parameters
  config.model.score_thresh = 0.05
  config.model.pre_nms_topk = 1000
  config.model.post_nms_topk = 100
  config.model.iou_thresh = 0.6  # Note: CenterNet uses 0.6

  # optimizer
  config.optimizer = ml_collections.ConfigDict()
  config.optimizer.optimizer = 'adamw'
  config.optimizer.weight_decay = 0.05
  config.optimizer.skip_scale_and_bias_regularization = True
  config.optimizer.layerwise_decay = 0.7
  config.optimizer.num_layers = 12
  config.optimizer.decay_layer_prefix = 'backbone/net/blocks.'
  config.optimizer.decay_stem_layers = ['patch_embed.proj', 'pos_embed']

  # learning rate and training schedule
  config.num_training_steps = 90000
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * piecewise_constant * linear_warmup'
  config.lr_configs.decay_events = [80000, 87500]
  config.lr_configs.decay_factors = [0.1, 0.01]
  config.lr_configs.warmup_steps = 250
  config.lr_configs.base_learning_rate = 0.0001

  # Pretrained_backbone.
  config.weights = '/path/to/mae_pretrain_vit_base/'
  config.load_prefix = 'backbone/net/'
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
