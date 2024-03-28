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
"""Default config for baseline training on SMiT dataset."""

import ml_collections


DATA_TRAIN_SIZE = 481094


def get_config(runlocal=''):
  """Returns the base experiment configuration."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'verb_focused_contrastive_training'

  # Dataset.
  config.dataset_name = 'verbs_in_action_tfrecord_dataset'
  config.dataset_configs = ml_collections.ConfigDict()
  config.data_dtype_str = 'float32'
  The `train` split should contain hard negative captions in the
  `caption/string_neg` field and positive caption in the `caption/strin_pos`
  field of the tfrecord.
  config.dataset_configs.base_dir = 'your_base_directory'
  config.dataset_configs.tables = {
      'train': 'training_path',
      'validation': 'validation_path',
      'test': 'test_path',
  }
  config.dataset_configs.examples_per_subset = {
      'train': DATA_TRAIN_SIZE,
      'validation': 8096,
  }
  config.dataset_configs.caption_string_train = 'caption/string_pos;caption/string_neg'
  config.dataset_configs.num_frames = 32
  config.dataset_configs.stride = 14
  config.dataset_configs.test_stride = 14
  config.dataset_configs.min_resize = 256
  config.dataset_configs.crop_size = 224
  config.dataset_configs.zero_centering = False
  config.dataset_configs.augmentation_params = ml_collections.ConfigDict()
  config.dataset_configs.augmentation_params.do_jitter_scale = True
  config.dataset_configs.augmentation_params.scale_min_factor = 0.9
  config.dataset_configs.augmentation_params.scale_max_factor = 1.33
  config.dataset_configs.augmentation_params.prob_scale_jitter = 1.0
  config.dataset_configs.augmentation_params.do_color_augment = False
  config.dataset_configs.prefetch_to_device = 2
  # Text params
  config.dataset_configs.max_num_words = 77
  config.dataset_configs.caption_string = 'caption/string'
  config.dataset_configs.num_train_captions = 6  # 1 positive and 5 hard negatives
  config.dataset_configs.rmv_full_stop = True
  config.dataset_configs.include_verb = True
  config.dataset_configs.keep_test_key = True

  # Model.
  config.model = ml_collections.ConfigDict()
  config.model.clip_version = 'vit_b32'
  config.model.temporal_agg = 'transformer'  # or 'meanpool'

  # Initalisation configs
  config.init_from = ml_collections.ConfigDict()
  config.init_from.checkpoint_path = ''
  config.train_from_scratch = False

  # Training.
  config.max_grad_norm = 1
  config.batch_size = 256 if not runlocal else 8
  config.rng_seed = 0
  config.temperature = 0.005  # Temperature for the NCE loss
  config.optimizer = 'adamw'
  config.multi_optim = False
  config.weight_decay = 0.01
  config.freeze_video_encoder = False
  config.freeze_text_encoder = False
  steps_per_epoch = DATA_TRAIN_SIZE // config.batch_size
  config.num_training_epochs = 100
  total_steps = config.num_training_epochs * steps_per_epoch
  # Learning schedules.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.factors = 'constant * cosine_decay'
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = 1e-7

  config.verb_hard_negatives = True  # We train with hard negative captions.
  # With this weighting of the loss there is the same amount of contribution
  # from "text-to-vision" matching and "vision-to-text" matching.
  config.verb_phrase_loss_weight = 0.5  # We also include a verb-phrase loss.
  config.v2t_weight = 0.5
  config.t2v_weight = 1.
  # When beta is equal to zero this is the normal Info-NCE loss.
  # When beta is not equal to zero this is the HN-NCE loss as proposed in
  # this paper: https://arxiv.org/abs/2301.02280
  config.beta_hnnce = 0.


  # Logging.
  config.write_summary = True
  config.checkpoint = True
  config.debug_train = False
  config.debug_eval = False
  # Checkpoint more frequently than a val epoch for this model.
  config.log_eval_steps = 1000
  config.checkpoint_steps = 5000
  config.log_summary_steps = 100
  if runlocal:
    config.count_flops = False

  return config


