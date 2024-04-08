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
r"""DenseVOC config.

"""

import ml_collections
from scenic.projects.densevoc.configs import common


def get_config():
  """Returns the config."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'densevoc_vidstg_ftgrit_soft_aggregation'
  config.model = ml_collections.ConfigDict()

  # Dataset.
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.is_video_dataset_train = True
  config.dataset_configs.is_video_dataset_test = False
  config.model.flatten_video_input = True
  config.dataset_configs.max_frames_train = 16
  config.dataset_configs.max_frames_test = 200
  config.dataset_configs.train_data_path = common.VIDSTG_TRAIN_VIDEO_TFRECORD_PATH
  config.dataset_configs.test_data_path = common.VIDSTG_VAL_IMAGE_TFRECORD_PATH
  config.dataset_configs.test_annotation_path = common.VIDSTG_VAL_IMAGE_ANN_PATH
  config.dataset_configs.tokenizer_weight_path = common.BERT_TOKENIZER_PATH
  config.dataset_configs.num_train_examples = common.NUM_VIDSTG_TRAIN_VIDEOS
  config.dataset_configs.num_eval_examples = common.NUM_VIDSTG_VAL_FPS1_MAX40F_IMAGES
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.shuffle_buffer_size = 1000
  config.dataset_configs.max_boxes = 8
  config.dataset_configs.max_text_tokens = 40
  config.dataset_configs.scale_range = (0.1, 2.0)
  config.dataset_configs.crop_size = 384
  config.dataset_configs.size_divisibility = 32
  config.dataset_configs.ensure_sample_has_objects = False
  config.data_dtype_str = 'float32'

  config.rng_seed = 0

  # Model.
  config.model.model_dtype_str = 'float32'
  config.model.model_name = 'densevoc'
  config.model.backbone_name = 'vitdet'
  config.model.num_classes = -1
  config.model.strides = (8, 16, 32, 64, 128)
  config.model.pixel_mean = (103.530, 116.280, 123.675)
  config.model.pixel_std = (57.375, 57.120, 58.395)
  config.model.backbone_args = ml_collections.ConfigDict()
  config.model.backbone_args.drop_path_rate = 0.1
  config.model.backbone_args.img_size = config.dataset_configs.crop_size
  config.model.freeze_model_state = False
  config.model.backbone_args.use_ln_pre = True

  s = 2
  config.model.fpn_range = (
      (0, 80 / s), (64 / s, 160 / s), (128 / s, 320 / s),
      (256 / s, 640 / s), (512 / s, 100000 / s))

  # CenterNet2 parameters
  config.model.roi_num_classes = 1
  config.model.hm_weight = 0.5
  config.model.reg_weight = 1.0
  config.model.score_thresh = 0.0001
  config.model.pre_nms_topk_train = 2000
  config.model.post_nms_topk_train = 1000
  config.model.pre_nms_topk_test = 1000
  config.model.post_nms_topk_test = 256
  config.model.iou_thresh = 0.9
  config.model.roi_matching_threshold = (0.6,)
  config.model.roi_nms_threshold = 0.5

  config.model.mult_caption_score = False
  config.model.text_iou_thresh = 0.6
  config.model.object_feat_res = 7
  config.model.with_temporal_localization = False
  config.model.temporal_localization_loss_weight = 1.0
  config.model.with_tracking = True
  config.model.tracking_loss_weight = 1.0
  config.model.num_text_proposals = 8
  config.model.roi_post_nms_num_detections = 16
  config.model.roi_samples_per_image = 32
  config.model.caption_with_track = True
  config.model.roi_append_gt_boxes = False
  config.model.trunc_track_score = 0.0
  config.model.asso_windows = -1
  config.model.consistent_soft_track = True

  config.model.use_roi_box_in_training = True
  config.model.use_tracked_object_features = True

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
  config.num_training_steps = 45000 // 4
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * piecewise_constant * linear_warmup'
  config.lr_configs.decay_events = (40000 // 4, 43750 // 4)
  config.lr_configs.decay_factors = [0.1, 0.01]
  config.lr_configs.warmup_steps = 250
  config.lr_configs.base_learning_rate = 1e-5

  config.weights = '/path/to/grit_vg_384'
  config.skip_wrong_shape = True
  config.checkpoint_steps = 500
  config.log_eval_steps = 5000000

  # Training.
  config.batch_size = 16

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.log_summary_steps = 20  # train summary steps
  config.log_large_summary_steps = 1000  # Expensive summary operations freq
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  config.inference_on_video = config.dataset_configs.is_video_dataset_test
  config.video_eval_task = 'densecap'
  config.eval_chota = False

  return config


