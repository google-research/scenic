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
r"""Audiovisual Masked Autoencoder finetuning.

"""
# pylint: disable=line-too-long

import ml_collections

# The Audioset 500K balanced split from https://arxiv.org/abs/2107.00135.
# The size of the Audioset dataset changes as videos are removed from YouTube.
# Set this appropriately.
AUDIOSET_TRAIN_SIZE = 508994
AUDIOSET_VAL_SIZE = 18589


def get_config():
  """Returns the base experiment configuration."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'avmae_audioset_classification'

  # AudioSet dataset.
  config.dataset_name = 'video_sstable_dataset_mfp'
  config.dataset_configs = ml_collections.ConfigDict()
  config.data_dtype_str = 'float32'

  config.dataset_name = 'avmae_audiovisual_tfrecord_dataset'
  config.dataset_configs = ml_collections.ConfigDict()
  config.data_dtype_str = 'float32'

  config.dataset_configs.base_dir = '/path/to/root_directory'
  config.dataset_configs.tables = {
      'train': 'train@1000',
      'validation': 'val@1000',
  }
  config.dataset_configs.examples_per_subset = {
      'train': AUDIOSET_TRAIN_SIZE,
      'validation': AUDIOSET_VAL_SIZE,
      'test': AUDIOSET_VAL_SIZE
  }

  config.dataset_configs.num_classes = 527
  config.dataset_configs.test_split = 'validation'

  # List of modalities to load, supports `rgb`, `spectrogram`.
  # Note that it only specifies which modalities to load, not which to use,
  # which is controlled by config.model.modality_fusion
  config.dataset_configs.modalities = ('rgb', 'spectrogram')
  config.dataset_configs.return_as_dict = True
  # This is going to sample 32 frames, sampled at a stride of 2 from the video.
  config.dataset_configs.num_frames = 32
  config.dataset_configs.stride = 2
  config.dataset_configs.num_spec_frames = 10
  config.dataset_configs.spec_stride = 1

  # These statistics are calculated over the entire unbalanced train set.
  config.dataset_configs.normalization_mean_spec = 1.102
  config.dataset_configs.normalization_std_spec = 2.762

  config.dataset_configs.min_resize_train = 256
  config.dataset_configs.min_resize_test = 256
  config.dataset_configs.crop_size = 224

  config.dataset_configs.spec_shape = (100, 128)
  config.dataset_configs.inflate_spectrograms = False
  config.dataset_configs.num_waveform_samples = 32256
  config.dataset_configs.waveform_stride = 1
  config.dataset_configs.one_hot_labels = True
  config.dataset_configs.zero_centering = True

  config.dataset_configs.circular_time_shift = True

  # Multicrop eval settings
  config.dataset_configs.do_multicrop_test = True  # Do during training.
  config.dataset_configs.log_test_epochs = 1
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
  config.dataset_configs.spec_augment_params.freq_mask_max_bins = 36
  config.dataset_configs.spec_augment_params.freq_mask_count = 2
  config.dataset_configs.spec_augment_params.time_mask_max_frames = 48
  config.dataset_configs.spec_augment_params.time_mask_count = 4
  config.dataset_configs.spec_augment_params.time_warp_max_frames = 1.0
  config.dataset_configs.spec_augment_params.time_warp_max_ratio = 0
  config.dataset_configs.spec_augment_params.time_mask_max_ratio = 0

  config.model_name = 'mbt_multilabel_classification'
  config.model = ml_collections.ConfigDict()
  # Adjust this to finetune on different modalities
  config.model.modality_fusion = ('rgb', 'spectrogram')
  config.model.use_bottleneck = True
  config.model.test_with_bottlenecks = True
  config.model.share_encoder = False
  config.model.n_bottlenecks = 4

  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [16, 16]
  config.model.attention_config = ml_collections.ConfigDict()
  config.model.attention_config.type = 'spacetime'
  config.model.representation_size = None
  config.model.classifier = 'gap'
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.
  config.model_dtype_str = 'float32'

  config.model.hidden_size = 1024
  config.model.num_heads = 16
  config.model.mlp_dim = 4096
  config.model.num_layers = 24
  config.model.fusion_layer = 18

  config.model.temporal_encoding_config = ml_collections.ConfigDict()
  config.model.temporal_encoding_config.method = '3d_conv'
  config.model.patches.size = [16, 16, 2]
  config.model.temporal_encoding_config.kernel_init_method = 'central_frame_initializer'
  config.model.temporal_encoding_config.n_sampled_frames = 4  # Unused here.

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
  config.batch_size = 128
  config.rng_seed = 0
  config.mixup = ml_collections.ConfigDict()
  config.mixup.alpha = 0.5
  config.mixmod = True
  # Additional regularization
  config.model.stochastic_droplayer_rate = 0.3

  config.init_from = ml_collections.ConfigDict()
  config.init_from.model_type = 'multimae'
  config.init_from.init_from_mae = True

  NB: Set this path correctly to the pretrained checkpoint
  config.init_from.checkpoint_path = 'path_to_pretrained_checkpoint'
  config.init_from.restore_positional_embedding = True
  config.init_from.restore_input_embedding = True
  config.init_from.positional_embed_size_change = 'resize_tile'

  # Learning rate.
  steps_per_epoch = AUDIOSET_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 2.5 * steps_per_epoch
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = 1.6

  # Logging.
  config.log_summary_steps = 100
  config.checkpoint_steps = 1000
  config.write_summary = True
  config.checkpoint = True  # Do checkpointing.
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  return config


