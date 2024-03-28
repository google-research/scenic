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
r"""Multimodal sound classification on the balanced (mini) AudioSet.

"""
# pylint: disable=line-too-long

import ml_collections

# The size of the AudioSet dataset changes as videos are removed from YouTube.
# Update this accordingly.
AUDIOSET_TRAIN_SIZE = 20361


def get_config():
  """Returns the base experiment configuration."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'mbt_balanced_audioset_classification'

  # Dataset.
  config.dataset_configs.base_dir = '/path/to/dataset'
  config.dataset_configs.tables = {
      'train': 'balanced_train.se.melspec.tfrecord.sst@1024',
      'validation': 'eval.se.melspec.tfrecord.sst@1024',
      'test': 'eval.se.melspec.tfrecord.sst@1024',
  }
  config.dataset_configs.examples_per_subset = {
      'train': 20361,
      'validation': 18589,
      'test': 18589
  }
  config.dataset_configs.num_classes = 527
  config.data_dtype_str = 'float32'
  # List of modalities to load, supports `rgb` and `spectrogram'.
  # Note that this only specifies which modalities to load, not which to use,
  # which is controlled by config.model.modality_fusion
  config.dataset_configs.modalities = ('spectrogram', 'rgb')
  config.dataset_configs.return_as_dict = True
  # This is going to sample 32 frames, sampled at a stride of 2 from the video.
  # AudioSet videos are extracted at 25fps.
  config.dataset_configs.num_frames = 32
  config.dataset_configs.stride = 2
  config.dataset_configs.num_spec_frames = 8
  config.dataset_configs.spec_stride = 1

  # These statistics were calculated over the entire unbalanced train set.
  config.dataset_configs.spec_mean = 1.102
  config.dataset_configs.spec_stddev = 2.762

  config.dataset_configs.min_resize = 256
  config.dataset_configs.crop_size = 224
  config.dataset_configs.spec_shape = (100, 128)

  config.dataset_configs.one_hot_labels = True
  config.dataset_configs.zero_centering = True

  # Multicrop eval settings
  config.dataset_configs.do_multicrop_test = True
  config.dataset_configs.log_test_epochs = 4
  # The effective batch size per host when testing is
  # num_test_clips * test_batch_size
  config.dataset_configs.num_test_clips = 4
  config.dataset_configs.test_batch_size = 8  # Needs to be num_local_devices
  config.multicrop_clips_per_device = 2
  # Leaving this empty means that a full test is done each time.
  # About 4200 / 4 = 1050 steps on a 4-host setting (ie 4x4 TPU)
  # config.steps_per_test = 1000  # Number of test steps taken by each host.

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

  # Model: MBT-base
  config.model_name = 'mbt_multilabel_classification'
  config.model = ml_collections.ConfigDict()
  # Supports 'rgb' and 'spectrogram'
  config.model.modality_fusion = ('spectrogram', 'rgb')
  config.model.use_bottleneck = True
  config.model.test_with_bottlenecks = True
  config.model.share_encoder = False
  config.model.n_bottlenecks = 4
  # Layer at which to fuse. '0' refers to early fusion, if fusion_layer is equal
  # to model.num_layers, then there is no cross-modal attention in the transformer
  # and CLS tokens for each modality are averaged right at the end.
  config.model.fusion_layer = 8
  config.model.hidden_size = 768
  config.model.patches = ml_collections.ConfigDict()
  config.model.attention_config = ml_collections.ConfigDict()
  config.model.attention_config.type = 'spacetime'
  config.model.num_heads = 12
  config.model.mlp_dim = 3072
  config.model.num_layers = 12
  config.model.representation_size = None
  config.model.classifier = 'gap'
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.
  config.model_dtype_str = 'float32'

  config.model.temporal_encoding_config = ml_collections.ConfigDict()
  # 3d_conv is only used for RGB inputs.
  config.model.temporal_encoding_config.method = '3d_conv'
  # 32 frames for RGB. Conv filter is 8. So total of 4 frames at input
  config.model.patches.size = [16, 16, 2]
  config.model.temporal_encoding_config.kernel_init_method = 'central_frame_initializer'
  config.model.temporal_encoding_config.n_sampled_frames = 4  # Unused here.

  # Training.
  config.trainer_name = 'mbt_trainer'
  config.optimizer = 'momentum'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.l2_decay_factor = 0
  config.max_grad_norm = 1
  config.label_smoothing = 0.3
  config.num_training_epochs = 50
  config.batch_size = 64
  config.rng_seed = 0
  # This does Mixup in the train loop. This is fast. But make sure that device
  # batch size is more than 1. On a 4x4 TPU, this means that your batch size
  # needs to be at least 64.
  config.mixup = ml_collections.ConfigDict()
  config.mixup.alpha = 0.5
  config.mixmod = False
  # Additional regularization
  config.model.stochastic_droplayer_rate = 0.3

  # Use ImageNet-21k-initialised model from big_vision checkpoint
  config.init_from = ml_collections.ConfigDict()
  config.init_from.model_config = None
  # Download pretrained ImageNet checkpoints from here:
  # https://github.com/google-research/scenic/tree/main/scenic/projects/baselines (checkpoint_format = 'scenic')  pylint: disable=line-too-long
  # https://github.com/google-research/vision_transformer (checkpoint_format = 'big_vision')  pylint: disable=line-too-long
  config.init_from.checkpoint_path = 'path_to_checkpoint_of_vit_b_16'
  config.init_from.checkpoint_format = 'scenic'
  config.init_from.model_config = ml_collections.ConfigDict()
  config.init_from.model_config.model = ml_collections.ConfigDict()
  config.init_from.model_config.model.classifier = 'token'  # Specify if this is 'token' or 'gap'.  pylint: disable=line-too-long
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
  config.lr_configs.base_learning_rate = 5e-1

  # Logging.
  config.write_summary = True
  config.checkpoint = True  # Do checkpointing.
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  config.checkpoint_steps = 500  # Checkpoint more frequently than a val epoch.
  return config


