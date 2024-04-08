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
  config.experiment_name = 'densevoc_vidstg_videoeval'

  # Dataset.
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.is_video_dataset_train = True
  config.dataset_configs.is_video_dataset_test = True
  config.dataset_configs.max_frames_train = 16
  config.dataset_configs.max_frames_test = 200
  config.dataset_configs.train_data_path = common.VIDSTG_TRAIN_VIDEO_TFRECORD_PATH
  config.dataset_configs.test_data_path = common.VIDSTG_VAL_VIDEO_TFRECORD_PATH
  config.dataset_configs.test_annotation_path = common.VIDSTG_VAL_VIDEO_ANN_PATH
  config.dataset_configs.tokenizer_weight_path = common.BERT_TOKENIZER_PATH
  config.dataset_configs.num_train_examples = common.NUM_VIDSTG_TRAIN_VIDEOS
  config.dataset_configs.num_eval_examples = common.NUM_VIDSTG_VAL_VIDEOS
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.shuffle_buffer_size = 1
  config.dataset_configs.max_boxes = 100
  config.dataset_configs.max_text_tokens = 40
  config.dataset_configs.scale_range = (0.1, 2.0)
  config.dataset_configs.crop_size = 384
  config.dataset_configs.size_divisibility = 32
  config.data_dtype_str = 'float32'
  config.rng_seed = 0

  config.model = ml_collections.ConfigDict()
  config.model.model_dtype_str = 'float32'
  config.model.model_name = 'densevoc'
  config.model.backbone_name = 'vitdet'
  config.model.num_classes = -1
  config.model.strides = (8, 16, 32, 64, 128)
  config.model.pixel_mean = (103.530, 116.280, 123.675)
  config.model.pixel_std = (57.375, 57.120, 58.395)  # For most other models
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
  config.model.roi_score_threshold = 0.5

  config.model.mult_caption_score = False
  config.model.text_iou_thresh = 0.6
  config.model.object_feat_res = 7
  config.model.with_temporal_localization = False
  config.model.temporal_localization_loss_weight = 1.0
  config.model.with_tracking = True
  config.model.tracking_loss_weight = 1.0
  config.model.num_text_proposals = 128  # 8
  config.model.roi_post_nms_num_detections = 16
  config.model.roi_samples_per_image = 32
  config.model.caption_with_track = True
  config.model.roi_append_gt_boxes = False
  config.model.trunc_track_score = 0.0
  config.model.asso_windows = 8
  config.model.use_roi_box_in_training = True
  config.model.use_tracked_object_features = False

  config.model.hard_tracking = True
  config.model.tracking_score_thresh = 0.7
  config.model.max_num_tracks = 32
  config.model.hard_tracking_frames = 6
  config.model.remove_bg_proposal_for_tracking = True

  config.model.hard_tracking_test = True
  config.model.tracking_iou_thresh = 0.4

  config.weights = '/path/to/densevoc_vidstg'

  config.rng_seed = 0
  config.in_model_postprocess = True
  config.batch_size = 8

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.log_summary_steps = 50  # train summary steps
  config.log_large_summary_steps = 1000  # Expensive summary operations freq
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  config.eval_only = True
  config.inference_on_video = True
  config.video_eval_task = 'densecap'
  config.eval_cap_switch = False
  config.eval_chota = True
  config.chota_caption_metric = ('cider', 'meteor', 'spice')

  return config


