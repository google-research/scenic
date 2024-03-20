# Copyright 2023 The Scenic Authors.
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

r"""ViViT Factorised Encoder model for Epic Kitchens.

"""


from absl import logging
import ml_collections

EPIC_TRAIN_SIZE = 67217
EPIC_VALID_SIZE = 9668


def get_config():
  """Returns the base experiment configuration."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'vivit_large_factorised_encoder_ek'

  # Dataset.
  config.dataset_name = 'video_tfrecord_dataset'
  config.dataset_configs = ml_collections.ConfigDict()
  config.data_dtype_str = 'float32'
  config.datset_configs = ml_collections.ConfigDict()
  config.dataset_configs.base_dir = (
      '/path/to/dataset')
  config.dataset_configs.tables = {
      'train': 'train.tfrecord@1024',
      'validation': 'validation.tfrecord@1024',
      'test': 'test.tfrecord@1024'
  }
  config.dataset_configs.examples_per_subset = {
      'train': EPIC_TRAIN_SIZE,
      'validation': EPIC_VALID_SIZE,
      'test': EPIC_VALID_SIZE
  }

  ## EpicKitchens-specific flags.
  # We want to train a "multi-head" classification model where we are predicting
  # both "nouns" and "verbs"
  # Setting this as None / leaving it empty, means we concatenate the two labels
  config.dataset_configs.class_splits = [300, 97]
  config.dataset_configs.split_names = ['noun', 'verb']

  # This is going to sample 32 frames, sampled at a stride of 1 from the video.
  config.dataset_configs.num_frames = 32
  config.dataset_configs.stride = 1
  config.dataset_configs.min_resize = 256
  config.dataset_configs.crop_size = 224
  config.dataset_configs.one_hot_labels = True
  config.dataset_configs.zero_centering = True

  # Multicrop evaluation settings:
  config.dataset_configs.do_multicrop_test = True  # Do during training.
  config.dataset_configs.log_test_epochs = 5
  # The effective batch size per host when testing is
  # num_test_clips * test_batch_size.
  config.dataset_configs.num_test_clips = 4
  config.dataset_configs.test_batch_size = 8  # Must equal num_local_devices.
  # To take three spatial crops when testing.
  config.dataset_configs.do_three_spatial_crops = True
  config.multicrop_clips_per_device = 2

  # Data augmentation
  config.dataset_configs.augmentation_params = ml_collections.ConfigDict()
  config.dataset_configs.augmentation_params.do_jitter_scale = True
  config.dataset_configs.augmentation_params.scale_min_factor = 0.9
  config.dataset_configs.augmentation_params.scale_max_factor = 1.33
  config.dataset_configs.augmentation_params.prob_scale_jitter = 1.0
  config.dataset_configs.augmentation_params.do_color_augment = False
  config.dataset_configs.augmentation_params.prob_color_augment = 0.8
  config.dataset_configs.augmentation_params.prob_color_drop = 0.1

  # This does Mixup in the train loop. This is fast. But make sure that device
  # batch size is more than 1.
  config.mixup = ml_collections.ConfigDict()
  config.mixup.alpha = 0.1

  config.dataset_configs.augmentation_params.do_rand_augment = True
  config.dataset_configs.augmentation_params.rand_augment_num_layers = 2
  config.dataset_configs.augmentation_params.rand_augment_magnitude = 15

  config.dataset_configs.prefetch_to_device = 2

  # Model.
  config.model_name = 'vivit_multihead_classification'
  config.model = ml_collections.ConfigDict()
  config.model.hidden_size = 1024
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [16, 16]

  config.model.attention_config = ml_collections.ConfigDict()
  config.model.attention_config.type = 'factorized_encoder'

  config.model.spatial_transformer = ml_collections.ConfigDict()
  config.model.spatial_transformer.num_heads = 16
  config.model.spatial_transformer.mlp_dim = 4096
  config.model.spatial_transformer.num_layers = 24

  config.model.temporal_transformer = ml_collections.ConfigDict()
  config.model.temporal_transformer.num_heads = 16
  config.model.temporal_transformer.mlp_dim = 4096
  config.model.temporal_transformer.num_layers = 4

  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.
  config.model.stochastic_droplayer_rate = 0.2
  config.model_dtype_str = 'float32'

  config.model.temporal_encoding_config = ml_collections.ConfigDict()
  config.model.temporal_encoding_config.method = '3d_conv'
  config.model.patches.size = [16, 16, 2]
  config.model.temporal_encoding_config.kernel_init_method = 'central_frame_initializer'

  # Training.
  config.trainer_name = 'vivit_trainer'
  config.optimizer = 'momentum'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.l2_decay_factor = 0
  config.max_grad_norm = 1
  config.label_smoothing = 0.2
  config.num_training_epochs = 50
  config.batch_size = 64
  config.rng_seed = 0

  # Initialisation.
  config.init_from = ml_collections.ConfigDict()
  Use a pretrained Kinetics checkpoint, or ImageNet checkpoint
  config.init_from.checkpoint_path = 'path_to_checkpoint'
  config.init_from.checkpoint_format = 'scenic'
  config.init_from.model_config = ml_collections.ConfigDict()
  config.init_from.model_config.model = ml_collections.ConfigDict()
  config.init_from.model_config.model.classifier = 'token'  # Specify if this is 'token' or 'gap'.  pylint: disable=line-too-long
  config.init_from.checkpoint_path = None
  config.init_from.restore_positional_embedding = True
  config.init_from.restore_input_embedding = True
  config.init_from.positional_embed_size_change = 'tile'

  # Learning rate.
  steps_per_epoch = EPIC_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = int(2.5 * steps_per_epoch)
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = 0.5

  # Logging.
  config.write_summary = True
  config.checkpoint = True  # Do checkpointing.
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  config.checkpoint_steps = 500  # Checkpoint more frequently than a val epoch.
  config.log_summary_steps = 100
  return config


