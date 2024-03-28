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


def get_config(runlocal=''):
  """Returns the base experiment configuration."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.token_loss_coef = 1.
  config.runlocal = runlocal
  config.experiment_name = 'ytt'

  config.count_flops = False if runlocal else ml_collections.ConfigDict(
      {'count_flops': True})

  # dataset
  config.dataset_name = 'dense_video_captioning'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.corrupt = 0.25
  config.dataset_configs.span_len = 5.
  config.dataset_configs.proba_corrupt = 1.
  config.dataset_configs.corrupt_coef = 1.
  config.dataset_configs.preserve = False
  notime = ml_collections.config_dict.FieldReference(False)
  config.dataset_configs.notime = notime
  config.dataset_configs.abs_time_token = False
  config.dataset_configs.random_temporal_crop_proba = 1.
  config.dataset_configs.time_format = 'se'
  tmp_only = ml_collections.config_dict.FieldReference(False)
  config.dataset_configs.tmp_only = tmp_only
  config.dataset_configs.split = not runlocal
  order = ml_collections.config_dict.FieldReference('ld')
  config.dataset_configs.order = order
  config.dataset_configs.from_xm = None

  config.data_dtype_str = 'float32'

  config.dataset_configs.base_dir = '/'
  config.dataset_configs.base_dir = '/path/to/yttemporal'
  config.dataset_configs.tables = {
      'train': 'train.tfrecord.sst@1024',
  }
  config.dataset_configs.examples_per_subset = {
      'train': 14780275,
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
  config.dataset_configs.max_num_output_words = 1000
  config.dataset_configs.max_num_input_words = 1000
  config.dataset_configs.tokenizer = ml_collections.ConfigDict()
  config.dataset_configs.tokenizer.tokenizer_type = 'sentence_piece'
  config.dataset_configs.caption_string = 'ASR/segment/label/string'
  config.dataset_configs.train_caption_string = 'ASR/segment/label/string'
  config.dataset_configs.input_timestamp_start_name = 'ASR/segment/start/timestamp'
  config.dataset_configs.input_timestamp_end_name = 'ASR/segment/end/timestamp'
  config.dataset_configs.input_duration_name = 'video/duration'
  config.dataset_configs.output_raw_timestamp_name = 'timestamp'
  config.dataset_configs.output_raw_duration_name = 'duration'
  config.dataset_configs.input_feature_name = 'image/clip_embeddings'
  config.dataset_configs.output_raw_feature_name = 'features'
  config.dataset_configs.vocabulary_size = 32128
  config.dataset_configs.max_events = 1100
  config.dataset_configs.max_segments = 0
  config.datasets = {'ytt': config.dataset_configs}

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
  config.model.encoder.cat_encoder.dropout_rate = 0.1
  config.model.encoder.cat_encoder.t5_dropout_rate = 0.1
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
  # Obtained from scenic/projects/t5/model.py.
  config.model.decoder.t5_decoder.pretrained_config = 't5_1_1_base'

  config.model.tmp_decoder_type = 't5_decoder'
  config.model.tmp_decoder = ml_collections.ConfigDict()
  config.model.tmp_decoder.t5_decoder = ml_collections.ConfigDict()
  config.model.tmp_decoder.t5_decoder.logits_via_embedding = False
  config.model.tmp_decoder.t5_decoder.dropout_rate = 0.
  config.model.tmp_decoder.t5_decoder.pretrained_config = 't5_1_1_base'
  config.model.decoder.t5_decoder.local = 5

  # Initalisation configs
  config.init_from = ml_collections.ConfigDict()
  config.init_from.step = None
  config.init_from.xm = None

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
  config.max_grad_norm = 0.1
  config.label_smoothing = 0.1
  epochs = ml_collections.config_dict.FieldReference(10)
  config.num_training_epochs = epochs
  batch_size = ml_collections.config_dict.FieldReference(512)
  config.batch_size = 1 if runlocal else batch_size  # 128  # Minimum is num_devices = 32
  config.eval_batch_size = 1 if runlocal else 128  # Needs to be num_local_devices
  config.rng_seed = 0

  # Learning schedule.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * linear_warmup'
  config.lr_configs.warmup_steps = 1000
  config.lr_configs.base_learning_rate = 1e-4

  config.eval_metrics = ['cider', 'meteor', 'soda']

  # Logging
  config.log_summary_steps = 500  # write TB and/or XM summary
  config.checkpoint_steps = 5000
  config.log_eval_steps = 5000
  config.write_summary = True  # write TB and/or XM summary
  config.write_xm_measurements = True  # write XM measurements
  config.xprof = True  # Profile using xprof
  config.checkpoint = True  # do checkpointing
  config.debug_train = False  # debug mode during training
  config.debug_eval = False  # debug mode during eval
  return config
