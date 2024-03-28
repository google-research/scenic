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
r"""Audiovisual Masked Autoencoder pretraining.

"""
# pylint: disable=line-too-long

import ml_collections

# The size of the Audioset dataset changes as videos are removed from YouTube.
# Set this appropriately.
AUDIOSET_TRAIN_SIZE = 1857210
AUDIOSET_VAL_SIZE = 18589
VARIANT = 'L/16x2'


def get_config(runlocal=''):
  """Returns the ViT experiment configuration for ImageNet."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'vivit-mae-audioset'

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

  # This is going to sample 16 frames, sampled at a stride of 4 from the video.
  config.dataset_configs.num_frames = 16
  config.dataset_configs.stride = 4
  config.dataset_configs.min_resize_train = 256
  config.dataset_configs.min_resize_test = 224
  config.dataset_configs.crop_size = 224
  config.dataset_configs.one_hot_labels = True
  config.dataset_configs.zero_centering = True

  config.dataset_configs.num_spec_frames = 10
  config.dataset_configs.spec_stride = 1
  config.dataset_configs.spec_shape = (100, 128)

  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.return_as_dict = True
  config.dataset_configs.modalities = ('spectrogram', 'rgb')
  config.dataset_configs.inflate_spectrograms = False

  # Model.
  version, tubelet = VARIANT.split('/')
  spatial_dim, temporal_dim = tubelet.split('x')
  spatial_dim, temporal_dim = int(spatial_dim), int(temporal_dim)

  config.model_name = 'vivit_multimodal_masked_autoencoder'
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
  config.model.representation_size = None
  config.model.attention_dropout_rate = 0.
  config.model.dropout_rate = 0.0
  config.model_dtype_str = 'float32'

  config.model.temporal_encoding_config = ml_collections.ConfigDict()
  config.model.temporal_encoding_config.method = '3d_conv'
  config.model.positional_embedding = 'sinusoidal_1d'
  config.model.positional_embedding_decoder = 'sinusoidal_1d'

  # Model decoder
  config.model.decoder_config = ml_collections.ConfigDict()
  config.model.decoder_config.hidden_size = {
      'B': 384,
      'L': 512
  }[version]
  config.model.decoder_config.num_layers = {
      'B': 4,
      'L': 4
  }[version]
  config.model.decoder_config.num_heads = {
      'B': 6,
      'L': 8,
  }[version]
  config.model.decoder_config.mlp_dim = {
      'B': 1536,
      'L': 2048
  }[version]

  config.model.decoder_config.dropout_rate = 0
  config.model.decoder_config.attention_dropout_rate = 0
  config.model.decoder_config.stochastic_depth = 0
  config.model.decoder_config.attention_config = ml_collections.ConfigDict()
  config.model.decoder_config.attention_config.type = 'spacetime'
  config.model.decoder_config.stochastic_droplayer_rate = 0
  config.model.classifier = 'none'
  config.model.encoder_strategy = 'separate_encoders_and_concat'
  config.model.decoder_strategy = 'same_decoder'
  config.model.use_inpainting = False
  config.model.use_modality_tokens = False
  config.model.fusion_layers = 2

  assert not (config.model.encoder_strategy == 'separate_encoders' and config.model.decoder_strategy == 'separate_decoders')

  # Masked Feature loss
  config.masked_feature_loss = ml_collections.ConfigDict()
  # NB: Change the following appropriately to train on a single modality.
  config.masked_feature_loss.target = {'rgb', 'spectrogram'}
  config.masked_feature_loss.token_mask_probability_dict = {'spectrogram': 0.7, 'rgb': 0.9}
  config.masked_feature_loss.select_central_frame = False
  config.masked_feature_loss.summary_num_columns = 1
  config.masked_feature_loss.number_of_img_in_column = 8  # must be divisible with temporal_dim
  config.masked_feature_loss.standardise_per_patch = False
  config.masked_feature_loss.standardise_per_patch_channels = False
  config.masked_feature_loss.normalise_by_output_dimension = True
  config.masked_feature_loss.masking_strategy = 'random'
  config.masked_feature_loss.modality_weight = ml_collections.ConfigDict({'spectrogram': 0.5, 'rgb': 0.5})

  assert not config.masked_feature_loss.select_central_frame

  # Training.
  config.trainer_name = 'avmae_multimodal_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.95
  config.optimizer_configs.weight_decay = 0
  config.explicit_weight_decay = 0.05
  config.l2_decay_factor = None
  config.label_smoothing = None
  config.num_training_epochs = 120
  config.batch_size = 8 if runlocal else 512
  config.rng_seed = 42
  config.init_head_bias = 0

  # Learning rate.
  steps_per_epoch = AUDIOSET_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 3e-4
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*linear_warmup*cosine_decay'
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.warmup_steps = int(4 * steps_per_epoch)
  config.lr_configs.base_learning_rate = base_lr * config.batch_size / 256
  end_lr = 0
  # alpha: float; The minimum value as a fraction of the initial value.
  config.lr_configs.alpha = end_lr / config.lr_configs.base_learning_rate

  # Logging.
  config.write_summary = True
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = steps_per_epoch
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  config.log_summary_steps = 100
  config.log_eval_steps = steps_per_epoch


  return config


