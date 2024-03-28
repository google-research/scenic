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

r"""Trains an ASR model on the howto100m dataset.

"""

import ml_collections


def get_config():
  """Returns the base experiment configuration."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'how2'

  # dataset
  config.dataset_name = 'av_asr_tfrecord_dataset'
  config.dataset_configs = ml_collections.ConfigDict()
  config.data_dtype_str = 'float32'
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.readahead = True
  config.dataset_configs.return_as_dict = True

  config.dataset_configs.base_dir = '/path/to/dataset'
  config.dataset_configs.tables = {
      'train': {
          'name': 'train-how2',
          'path': 'how2-5fps_tfrecord/train.rgb.wave.spec.normtext.visscore.tfrecord@512',
          'len': 184_868,
      },
      'val': {
          'name': 'val-how2',
          'path': 'how2-5fps_tfrecord/val.rgb.wave.spec.normtext.visscore.tfrecord@20',
          'len': int(2020 * 2),
      },
      'test': {
          'name': 'test-how2',
          'path': 'how2-5fps_tfrecord/test.rgb.wave.spec.normtext.visscore.tfrecord@20',
          'len': int(2303 * 2),
      },
  }

  # List of modalities to load, supports `rgb`, `spectrogram` and `waveform`.
  # Note that it only specifies which modalities to load, not which to use,
  # which is controlled by config.model.modality_fusion
  config.dataset_configs.modalities = (
      'rgb',
      'spectrogram',
  )

  # RGB parameters
  config.dataset_configs.num_frames = 8
  # config.dataset_configs.num_frames = 2
  config.dataset_configs.stride = 5
  # config.dataset_configs.stride = 1
  # Augmentation
  config.dataset_configs.augmentation_params = ml_collections.ConfigDict()
  config.dataset_configs.augmentation_params.do_jitter_scale = True
  config.dataset_configs.augmentation_params.scale_min_factor = 0.9
  config.dataset_configs.augmentation_params.scale_max_factor = 1.33
  config.dataset_configs.augmentation_params.prob_scale_jitter = 1.0
  config.dataset_configs.augmentation_params.do_color_augment = True
  config.dataset_configs.augmentation_params.prob_color_augment = 0.8
  config.dataset_configs.augmentation_params.prob_color_drop = 0.1

  # Spectrogram parameters
  config.dataset_configs.num_spec_frames = 25
  config.dataset_configs.eval_num_spec_frames = 25
  config.dataset_configs.spec_shape = (100, 80)

  # Online Spectrogram Computation
  config.dataset_configs.spec_compute_online = True
  config.dataset_configs.spec_from_wave_snr = None
  config.dataset_configs.spec_from_wave_add_train_masking_noise = True
  config.dataset_configs.spec_from_wave_visualness_threshold = 0.01
  # content word rate: 0.27548048555
  config.dataset_configs.spec_from_wave_random_mask_noise_rate = 0.363  # ~10%
  config.dataset_configs.spec_from_wave_max_num_masks = 16

  config.dataset_configs.spec_from_wave_environment_noise_path = (
      'path/to/noise/matrix.npy'
  )

  # pylint: disable=g-complex-comprehension
  config.dataset_configs.spec_from_wave_eval_noise_configs = {
      'environment_noise': [{
          'environment_noise_configs': {
              'snr': 1 / i
          }
      } for i in range(1, 4)],
      'packet_loss_noise': [{
          'packet_loss_noise_configs': {
              'max_num_bursts': 2,
              'max_length_rate': 0.1 * i
          }
      } for i in range(1, 4)],
      'environment_noise,packet_loss_noise': [{
          'environment_noise_configs': {
              'snr': 1 / i
          },
          'packet_loss_noise_configs': {
              'max_num_bursts': 2,
              'max_length_rate': 0.1 * i
          }
      } for i in range(1, 4)],
  }
  # pylint: enable=g-complex-comprehension

  # SpecAugment hyperparameters
  config.dataset_configs.spec_augment = True
  config.dataset_configs.spec_augment_params = ml_collections.ConfigDict()
  config.dataset_configs.spec_augment_params.freq_mask_max_bins = 27
  config.dataset_configs.spec_augment_params.freq_mask_count = 2
  config.dataset_configs.spec_augment_params.time_mask_max_frames = 100
  config.dataset_configs.spec_augment_params.time_mask_count = 2
  config.dataset_configs.spec_augment_params.time_warp_max_frames = 80
  config.dataset_configs.spec_augment_params.time_warp_max_ratio = 0
  config.dataset_configs.spec_augment_params.time_mask_max_ratio = 0

  # Text parameters
  config.dataset_configs.max_num_words = 64
  config.dataset_configs.eval_max_num_words = 128
  config.dataset_configs.tokenizer = ml_collections.ConfigDict()
  bert_tokenizer_path = r'path/to/bert/vocabulary.txt'
  config.dataset_configs.tokenizer.tokenizer_vocab = bert_tokenizer_path
  config.dataset_configs.tokenizer.tokenizer_type = 'bert'

  #
  # Model.
  config.model_name = 'seq2seq_model'
  config.model = ml_collections.ConfigDict()
  dim = 768
  config.model.embedding_dimension = dim
  config.model.encoder_model = 'mbt'
  config.model.decoder_model = 'vd'
  config.model_dtype_str = 'float32'

  #
  # MBT configs
  config.mbt = ml_collections.ConfigDict()
  config.mbt.model = ml_collections.ConfigDict()
  # Supports 'rgb' and 'spectrogram'
  config.mbt.model.modality_fusion = (
      'rgb',
      'spectrogram',
  )
  config.mbt.model.use_bottleneck = True
  config.mbt.model.share_encoder = False
  config.mbt.model.n_bottlenecks = 4
  # Layer at which to fuse. '0' refers to early fusion, if fusion_layer is equal
  # to model.num_layers, then there is no cross-modal attention and CLS tokens
  # for each modality are averaged right at the end.
  config.mbt.model.fusion_layer = 8
  config.mbt.model.hidden_size = dim
  config.mbt.model.patches = ml_collections.ConfigDict()
  config.mbt.model.patches.size = [16, 16, 2]
  config.mbt.model.attention_config = ml_collections.ConfigDict()
  config.mbt.model.attention_config.type = 'spacetime'
  config.mbt.model.num_heads = 12
  config.mbt.model.mlp_dim = dim * 4
  config.mbt.model.num_layers = 12
  # For simplicity, we disable `token` classifier for multimodal inputs.
  config.mbt.model.classifier = 'token'
  config.mbt.model.attention_dropout_rate = 0.1
  config.mbt.model.dropout_rate = 0.0
  config.mbt.model.stochastic_droplayer_rate = 0.2
  config.mbt.model.temporal_encoding_config = ml_collections.ConfigDict()
  # 3d_conv is only used for RGB inputs.
  config.mbt.model.temporal_encoding_config.method = '3d_conv'
  config.mbt.model.temporal_encoding_config.kernel_init_method = (
      'central_frame_initializer'
  )
  config.mbt.model.temporal_encoding_config.n_sampled_frames = 4  # Unused here.

  #
  # VD config
  config.vd = ml_collections.ConfigDict()
  config.vd.model = ml_collections.ConfigDict()

  config.vd.model.dtype = config.model_dtype_str

  # Maximum number of position embeddings
  config.vd.model.max_len = 256

  # Number of transformer layers.
  config.vd.model.num_layers = 8
  # Number of attention heads.
  config.vd.model.num_heads = 4

  # Size of query/key/value for attention.
  config.vd.model.qkv_dim = dim
  # Size of embeddings.
  config.vd.model.emb_dim = dim
  # Size of the MLP.
  config.vd.model.mlp_dim = dim * 4

  # Dropout rate.
  config.vd.model.dropout_rate = 0.1
  # Attention dropout rate.
  config.vd.model.attention_dropout_rate = 0.0

  # Vocabulary size.
  config.vd.model.vocab_size = 30_522

  config.vd.logits_via_embedding = True

  # Initalisation configs
  config.init_from = ml_collections.ConfigDict()
  config.init_from.init_from_vit = False
  config.init_from.xm = (31430128, 1)

  # Training.
  config.trainer_name = 'generation_trainer'
  config.optimizer = 'momentum'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.l2_decay_factor = 0.
  config.max_grad_norm = 1.
  config.label_smoothing = 0.1
  # config.num_training_epochs = 150
  config.num_training_steps = 40_000
  config.batch_size = 256  # Minimum is num_devices = 32
  config.eval_batch_size = 128  # Smaller than batch size because beam search
  config.rng_seed = 0

  # Learning schedule.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 0
  config.lr_configs.steps_per_cycle = config.num_training_steps
  config.lr_configs.base_learning_rate = 0.3

  # Eval
  # Beam size for inference.
  config.beam_size = 4
  config.max_decode_len = 128
  config.eos_id = 102  # Depend on tokenizer

  # Logging
  config.write_summary = True
  config.write_xm_measurements = True
  config.checkpoint = True
  config.debug_train = False
  config.debug_eval = False
  config.log_eval_steps = 500  # Perform evaluation and testing
  config.log_summary_steps = 100  # Log training summary

  config.dataset_configs.base_dir = '/path/to/checkpoint/directory'

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""

  return hyper.product([
      hyper.sweep(
          'config.dataset_configs.spec_from_wave_visualness_threshold',
          [0.1],
      ),
      hyper.sweep(
          'config.dataset_configs.spec_from_wave_random_mask_noise_rate',
          [0.31],
      ),
  ])
