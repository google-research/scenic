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
r"""Baseline experiments on Something-Something v2.


"""
# pylint: disable=line-too-long

import ml_collections

SSV2_TRAIN_SIZE = 168913
SSV2_VAL_SIZE = 24777
VARIANT = 'B/16x2'


def get_config(runlocal=''):
  """Return the config of baseline experiment on Something-Something v2."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'ssv2_B16_baseline'

  # Dataset.
  config.dataset_name = 'objects_video_tfrecord_dataset'
  config.dataset_configs = ml_collections.ConfigDict()
  config.data_dtype_str = 'float32'

  # This is going to sample 16 frames, sampled at a stride of 2 from the video.
  config.dataset_configs.num_frames = 16
  config.dataset_configs.stride = 2
  config.dataset_configs.min_resize_train = 256
  config.dataset_configs.min_resize_test = 256
  config.dataset_configs.crop_size = 224
  config.dataset_configs.one_hot_labels = True
  config.dataset_configs.zero_centering = True

  config.dataset_configs.base_dir = 'path/to/dataset/root'
  config.dataset_configs.tables = {
      'train': 'something-v2-train.rgb.ori.gt_bbox_track.orvit_bbox.tfrecord@1024',
      'validation':
          'something-v2-validation.rgb.ori.gt_bbox_track.orvit_bbox.tfrecord@128',
      'test':
          'something-v2-validation.rgb.ori.gt_bbox_track.orvit_bbox.tfrecord@128',
  }
  config.dataset_configs.examples_per_subset = {
      'train': SSV2_TRAIN_SIZE,
      'validation': SSV2_VAL_SIZE,
      'test': SSV2_VAL_SIZE
  }
  config.dataset_configs.num_classes = 174
  config.dataset_configs.test_split = 'validation'

  config.dataset_configs.random_flip = False

  # Sampling
  config.dataset_configs.train_frame_sampling_mode = 'segment_sampling'

  # This does Mixup in the train loop. This is fast. But make sure that device
  # batch size is more than 1. On a 4x4 TPU, this means that your batch size
  # needs to be at least 64.
  config.mixup = ml_collections.ConfigDict()
  config.mixup.alpha = 0.8
  config.mixup.cutmix_alpha = 1.0
  config.mixup.mixup_to_cutmix_switch_prob = 0.5

  config.dataset_configs.prefetch_to_device = 2

  # Multicrop eval settings
  config.dataset_configs.do_multicrop_test = True  # Do during training.
  config.dataset_configs.do_three_spatial_crops = True  # Do during training.
  config.dataset_configs.log_test_epochs = 5
  # The effective batch size per host when testing is
  # num_test_clips * test_batch_size
  config.dataset_configs.num_test_clips = 2
  config.dataset_configs.test_batch_size = 8  # Needs to be num_local_devices
  config.multicrop_clips_per_device = 2

  # Model.
  version, tubelet = VARIANT.split('/')
  spatial_dim, temporal_dim = tubelet.split('x')
  spatial_dim, temporal_dim = int(spatial_dim), int(temporal_dim)

  config.model_name = 'vivit_classification'
  config.model = ml_collections.ConfigDict()
  config.model.attention_config = ml_collections.ConfigDict()
  config.model.attention_config.type = 'spacetime'

  config.model.hidden_size = {'Ti': 192,
                              'S': 384,
                              'B': 768,
                              'L': 1024,
                              'H': 1280}[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = (spatial_dim, spatial_dim, temporal_dim)
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
  config.model.representation_size = None
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.0
  config.model_dtype_str = 'float32'

  config.model.temporal_encoding_config = ml_collections.ConfigDict()
  config.model.temporal_encoding_config.method = '3d_conv'
  config.model.temporal_encoding_config.kernel_init_method = 'central_frame_initializer'

  config.model.positional_embedding = 'sinusoidal_1d'
  config.model.classifier = 'gap'
  config.model.stochastic_droplayer_rate = 0.1

  # Training.
  # Hyperparameters follow VideoMAE: https://github.com/MCG-NJU/VideoMAE
  config.optimizer = 'adamw'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.weight_decay = 0.05
  config.optimizer_configs.layerwise_decay = 0.65
  config.optimizer_configs.b1 = 0.9
  config.optimizer_configs.b2 = 0.999
  config.l2_decay_factor = None
  config.label_smoothing = 0.1
  config.num_training_epochs = 30
  config.batch_size = 256
  config.rng_seed = 0
  config.init_head_bias = -6.9  # -log(1000)

  # Learning rate.
  steps_per_epoch = SSV2_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 1e-3
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.warmup_steps = 5 * steps_per_epoch
  config.lr_configs.base_learning_rate = base_lr * config.batch_size / 256
  end_lr = 1e-6 * config.batch_size / 256
  config.lr_configs.alpha = end_lr / config.lr_configs.base_learning_rate

  # Logging.
  config.write_summary = True
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 1000
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  config.log_summary_steps = steps_per_epoch
  config.log_eval_steps = steps_per_epoch

  # Initialisation from checkpoint
  config.init_from = ml_collections.ConfigDict()
  config.init_from.xm = ()
  config.dataset_configs.base_dir = 'path/to/mae/checkpoint'
  config.init_from.checkpoint_format = 'scenic'
  config.init_from.restore_from_non_mae_checkpoint = False

  return config
