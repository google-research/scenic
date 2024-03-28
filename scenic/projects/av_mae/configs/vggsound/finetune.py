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

r"""Audiovisual MAE finetuning.

"""

import ml_collections

# The size of the VGGSound dataset changes as videos are removed from YouTube.
# Set this appropriately.
VGGSOUND_TRAIN_SIZE = 172427
VARIANT = 'L/16x2'


def get_config():
  """Returns the base experiment configuration."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'avmae_vggsound_finetuning'

  # Dataset.
  config.dataset_name = 'avmae_audiovisual_tfrecord_dataset'
  config.dataset_configs = ml_collections.ConfigDict()
  config.data_dtype_str = 'float32'

  config.dataset_configs.base_dir = '/path/to/root_directory'
  config.dataset_configs.tables = {
      'train': 'train@1000',
      'validation': 'val@1000',
  }
  config.dataset_configs.num_classes = 309

  # List of modalities to load, supports `rgb`, `spectrogram` and `waveform`.
  # Note that it only specifies which modalities to load, not which to use,
  # which is controlled by config.model.modality_fusion
  config.dataset_configs.modalities = ('rgb', 'spectrogram')
  config.dataset_configs.return_as_dict = True
  # This is going to sample 32 frames, sampled at a stride of 2 from the video.
  config.dataset_configs.num_frames = 32
  config.dataset_configs.stride = 2
  config.dataset_configs.num_spec_frames = 8
  config.dataset_configs.spec_stride = 1
  config.dataset_configs.spec_mean = 0.
  config.dataset_configs.spec_stddev = 1.
  config.dataset_configs.min_resize = 256
  config.dataset_configs.crop_size = 224
  config.dataset_configs.spec_shape = (100, 128)
  config.dataset_configs.num_waveform_samples = 32256
  config.dataset_configs.waveform_stride = 1
  config.dataset_configs.one_hot_labels = True
  config.dataset_configs.zero_centering = True

  # Multicrop eval settings
  config.dataset_configs.do_multicrop_test = True
  config.dataset_configs.log_test_epochs = 5
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

  config.dataset_configs.prefetch_to_device = 2

  # SpecAugment hyperparameters
  config.dataset_configs.spec_augment = True
  config.dataset_configs.spec_augment_params = ml_collections.ConfigDict()
  config.dataset_configs.spec_augment_params.freq_mask_max_bins = 48
  config.dataset_configs.spec_augment_params.freq_mask_count = 1
  config.dataset_configs.spec_augment_params.time_mask_max_frames = 48
  config.dataset_configs.spec_augment_params.time_mask_count = 4
  config.dataset_configs.spec_augment_params.time_warp_max_frames = 1.0
  config.dataset_configs.spec_augment_params.time_warp_max_ratio = 0
  config.dataset_configs.spec_augment_params.time_mask_max_ratio = 0

  # Model
  version, tubelet = VARIANT.split('/')
  spatial_dim, temporal_dim = tubelet.split('x')
  spatial_dim, temporal_dim = int(spatial_dim), int(temporal_dim)

  config.model_name = 'mbt_classification'
  config.model = ml_collections.ConfigDict()
  # The modalities that we will use for finetuning.
  # NB: Adjust this to finetune on different modalities
  config.model.modality_fusion = ('rgb', 'spectrogram')
  config.model.use_bottleneck = True
  config.model.use_cross_bottleneck = False
  config.model.test_with_bottlenecks = True
  config.model.share_encoder = False
  config.model.n_bottlenecks = 4
  config.model.fusion_layer = 16

  config.model.hidden_size = {'Ti': 192,
                              'S': 384,
                              'B': 768,
                              'L': 1024,
                              'H': 1280}[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = (spatial_dim, spatial_dim, temporal_dim)
  config.model.num_heads = {'Ti': 3,
                            'S': 6,
                            'B': 12,
                            'L': 16,
                            'H': 16}[version]
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

  config.model.attention_config = ml_collections.ConfigDict()
  config.model.attention_config.type = 'spacetime'
  config.model.representation_size = None
  # For simplicity, we disable `token` classifier for multimodal inputs.
  config.model.classifier = 'gap'
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.
  config.model_dtype_str = 'float32'

  config.model.temporal_encoding_config = ml_collections.ConfigDict()
  # 3d_conv is only used for RGB inputs.
  config.model.temporal_encoding_config.method = '3d_conv'
  # 32 frames for RGB. Conv filter is 8. So total of 4 frames at input
  config.model.patches.size = [16, 16, 2]

  # Training.
  config.trainer_name = 'transfer_trainer_multimodal'
  config.optimizer = 'momentum'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.layerwise_decay = 0.75
  config.optimizer_configs.momentum = 0.9
  config.l2_decay_factor = 0
  config.max_grad_norm = 1
  config.grad_clip_after_pmean = True
  config.label_smoothing = 0.3
  config.num_training_epochs = 50
  config.batch_size = 64
  config.rng_seed = 0
  # This does Mixup in the train loop. This is fast. But make sure that device
  # batch size is more than 1. On a 4x4 TPU, this means that your batch size
  # needs to be at least 64.
  config.mixup = ml_collections.ConfigDict()
  config.mixup.alpha = 0.5
  config.mixmod = True
  # Additional regularization
  config.model.stochastic_droplayer_rate = 0.3

  # Initialisation
  config.init_from = ml_collections.ConfigDict()
  config.init_from.model_type = 'multimae'
  NB: Set this path correctly to the pretrained checkpoint
  config.init_from.checkpoint_path = 'path_to_pretrained_checkpoint'
  config.init_from.restore_positional_embedding = True
  config.init_from.restore_input_embedding = True
  config.init_from.positional_embed_size_change = 'resize_tile'

  # Learning rate.
  steps_per_epoch = VGGSOUND_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 2.5 * steps_per_epoch
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = 0.8

  # Logging.
  config.log_summary_steps = 100
  config.checkpoint_steps = 500
  config.write_summary = True
  config.checkpoint = True  # Do checkpointing.
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  return config


