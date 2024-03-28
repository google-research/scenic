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


import ml_collections

ACTIVITYNET_TRAIN_SIZE = 8649  # Number of videos


def get_config(runlocal=''):
  """Returns the base experiment configuration."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.token_loss_coef = 1.
  config.runlocal = runlocal
  config.experiment_name = 'activitynet'

  config.count_flops = False  # if runlocal else ml_collections.ConfigDict({'count_flops': True})

  # dataset
  config.dataset_name = 'dense_video_captioning'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.corrupt = 0.
  config.dataset_configs.span_len = 3.
  config.dataset_configs.preserve = True
  config.dataset_configs.corrupt_coef = 0.
  config.dataset_configs.proba_corrupt = 0.
  notime = ml_collections.config_dict.FieldReference(False)
  config.dataset_configs.notime = notime
  config.dataset_configs.abs_time_token = False
  config.dataset_configs.random_temporal_crop_proba = 0.
  config.dataset_configs.time_format = 'se'
  tmp_only = ml_collections.config_dict.FieldReference(False)
  config.dataset_configs.tmp_only = tmp_only
  config.dataset_configs.split = not runlocal
  order = ml_collections.config_dict.FieldReference('ld')
  config.dataset_configs.order = order
  config.dataset_configs.from_xm = None

  config.data_dtype_str = 'float32'

  config.dataset_configs.base_dir = '/path/to/activitynet'
  config.dataset_configs.tables = {
      'train': 'train.tfrecord.sst@64',
      'validation': 'val.tfrecord.sst@64',
  }
  config.dataset_configs.examples_per_subset = {
      'train': 8649,
      'validation': 4267,
  }

  # List of modalities to load, supports `features` only for now.
  # Note that it only specifies which modalities to load, not which to use,
  # which is controlled by config.model.modality_fusion
  config.dataset_configs.modalities = ('features', 'text')
  config.dataset_configs.features_dim = 768
  config.dataset_configs.return_as_dict = True
  num_frames = ml_collections.config_dict.FieldReference(100)
  config.dataset_configs.num_frames = num_frames
  num_bins = ml_collections.config_dict.FieldReference(100)
  config.dataset_configs.num_bins = num_bins
  config.dataset_configs.one_hot_labels = True
  config.dataset_configs.zero_centering = True
  config.dataset_configs.val_on_test = False
  config.dataset_configs.num_eval_clips = 1
  config.dataset_configs.prefetch_to_device = 2

  # Text params
  config.dataset_configs.max_num_output_words = 256
  config.dataset_configs.max_num_input_words = 1000
  config.dataset_configs.tokenizer = ml_collections.ConfigDict()
  config.dataset_configs.tokenizer.tokenizer_type = 'sentence_piece'
  config.dataset_configs.caption_string = 'caption/string'
  config.dataset_configs.train_caption_string = 'caption/string'
  config.dataset_configs.input_timestamp_name = 'video/timestamps'
  config.dataset_configs.input_duration_name = 'video/duration'
  config.dataset_configs.output_raw_timestamp_name = 'timestamp'
  config.dataset_configs.output_raw_duration_name = 'duration'
  config.dataset_configs.input_feature_name = 'image/clip_embeddings'
  config.dataset_configs.output_raw_feature_name = 'features'
  config.dataset_configs.vocabulary_size = 32128
  config.dataset_configs.max_events = 15 if runlocal else 30
  config.dataset_configs.asr_notime = False
  config.datasets = {'activitynet': config.dataset_configs}

  # Decoding
  config.decoding = ml_collections.ConfigDict()
  config.decoding.decoding_method = 'beamsearch'
  config.decoding.num_decodes = 4
  config.decoding.alpha = 0.6
  config.decoding.temperature = 1.

  # Model
  config.model_name = 'vid2seq'
  config.model = ml_collections.ConfigDict()
  config.model.from_xm = None

  # Encoder configs
  config.model.encoder = ml_collections.ConfigDict()
  config.model.encoder.share_encoder = True
  config.model.encoder.encoder_type = 'cat_encoder'
  config.model.encoder.cat_encoder = ml_collections.ConfigDict()
  config.model.encoder.cat_encoder.dim = 2048
  config.model.encoder.cat_encoder.layers = 12
  config.model.encoder.cat_encoder.heads = 12
  config.model.encoder.cat_encoder.pos_embed = 'learned_1d'
  config.model.encoder.cat_encoder.dropout_rate = 0.
  config.model.encoder.cat_encoder.t5_dropout_rate = 0.
  config.model.encoder.cat_encoder.stochastic_depth = 0.
  config.model.encoder.cat_encoder.pretrained_config = 't5_1_1_base'
  config.model.encoder.from_xm = None

  # Decoder configs
  config.model.decoder_type = 't5_decoder'
  config.model.decoder = ml_collections.ConfigDict()
  config.model.decoder.order = order
  config.model.decoder.t5_decoder = ml_collections.ConfigDict()
  config.model.decoder.t5_decoder.logits_via_embedding = False
  config.model.decoder.t5_decoder.dropout_rate = 0.1
  config.model.decoder.t5_decoder.num_frames = num_frames
  config.model.decoder.notime = notime
  config.model.decoder.num_bins = num_bins
  config.model.decoder.tmp_only = tmp_only
  config.model.decoder.t5_decoder.pretrained_config = 't5_1_1_base'

  # Initalisation configs
  config.init_from = ml_collections.ConfigDict()
  # Replace with your checkpoint pretrained on YT-temporal-1bn, assuming it has
  # been trained for 200K iterations
  config.init_from.checkpoint_path = 'path_to_checkpoint_pretrained_on_yt_temporal_1bn'
  config.init_from.model_config = 'path_to_yt_temporal_1bn_config'
  config.init_from.step = 200000

  config.init_from.encoder = ml_collections.ConfigDict()
  config.init_from.encoder.checkpoint_path = None
  config.init_from.encoder.init_from_vit = False
  config.init_from.encoder = ml_collections.ConfigDict()
  config.init_from.encoder.load_pretrained_weights = True

  config.init_from.decoder = ml_collections.ConfigDict()
  config.init_from.decoder.load_pretrained_weights = True

  config.init_from.t5 = ml_collections.ConfigDict()
  config.init_from.t5.load_pretrained_weights = True

  # Training
  config.trainer_name = 'densevidcap_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.weight_decay = 0.
  config.l2_decay_factor = 0.
  config.max_grad_norm = 1.
  config.label_smoothing = 0.1
  epochs = ml_collections.config_dict.FieldReference(20)
  config.num_training_epochs = epochs
  batch_size = ml_collections.config_dict.FieldReference(32)
  config.batch_size = 1 if runlocal else batch_size  # 128  # Minimum is num_devices = 32
  config.eval_batch_size = 1 if runlocal else 32  # Needs to be num_local_devices
  config.rng_seed = 0

  # Learning schedule.
  steps_per_epoch = 3 if runlocal else ACTIVITYNET_TRAIN_SIZE // batch_size
  total_steps = epochs * steps_per_epoch
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = total_steps // 10
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.total_steps = total_steps
  config.lr_configs.base_learning_rate = 3e-4

  config.eval_metrics = ['cider', 'meteor', 'soda']

  # Logging
  config.log_eval_steps = steps_per_epoch  # write TB and/or XM summary
  config.log_summary_steps = steps_per_epoch  # write TB and/or XM summary
  config.write_summary = True  # write TB and/or XM summary
  config.write_xm_measurements = True  # write XM measurements
  config.xprof = True  # Profile using xprof
  config.checkpoint = True  # do checkpointing
  config.debug_train = False  # debug mode during training
  config.debug_eval = False  # debug mode during eval
  return config

