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
r"""Config for ActivityNet Dense Captioning.

"""

import ml_collections
from scenic.projects.streaming_dvc.configs import common
evaluate_lazily = common.evaluate_lazily


def get_config():
  """Returns the configuration for GIT."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'vid2seq_anet_streaming_input_output'

  # Dataset.
  config.dataset_name = 'flexio_tfrecord'
  config.data_dtype_str = 'float32'

  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.pp_libs = [
      'scenic.projects.streaming_dvc.io.ops',
      'scenic.projects.streaming_dvc.io.densecap_ops']
  config.dataset_configs.test_annotation_path = common.ANET_ANN_VID2SEQ_FORMAT_PATH
  tokenizer_path = 't5'
  config.dataset_configs.tokenizer_weight_path = tokenizer_path  # Used in evaluation

  # Pre-processing settings
  context_features = {
      'caption/string': {'feature_type': 'VarLen', 'dtype': 'string'},
      'media_id': {'feature_type': 'VarLen', 'dtype': 'string'},
      'video/timestamps/start': {'feature_type': 'VarLen', 'dtype': 'int64'},
      'video/timestamps/end': {'feature_type': 'VarLen', 'dtype': 'int64'},
      'video/duration': {'feature_type': 'VarLen', 'dtype': 'int64'},
  }
  sequence_features = {
      'image/encoded': {'feature_type': 'FixedLenSequence', 'shape': [], 'dtype': 'string'},
      'image/timestamp': {'feature_type': 'FixedLenSequence', 'shape': [], 'dtype': 'int64'},
      'image/clip_embeddings': {'feature_type': 'FixedLenSequence', 'shape': [768], 'dtype': 'float32'},
  }

  crop_size = 224
  max_text_tokens = 256
  num_bins = 100

  config.num_bins = num_bins
  config.num_frames = 100
  config.num_dense_outputs = 8
  config.no_timestamp_in_context = True
  config.context_mask_ratio = 0.5
  config.dynamic_location = False
  continuous_random_mask = True

  @evaluate_lazily
  def get_preproc_spec_train(num_frames, num_dense_outputs, no_timestamp_in_context, context_mask_ratio, dynamic_location):
    preproc_spec_train = (
        # NOTE: it's important to turn off the random frame downsampling here.
        # Otherwise the time tokens are not aligned.
        f"decode_and_subsample_densecap_video({num_frames}, False, zero_pad_frames=False, with_clip_embeddings=True)"
        f"|decode_activity_net_dense_caption_annotations_dense_outputs_aug_context"
        f"(True, {num_dense_outputs}, '{tokenizer_path}', {max_text_tokens}, num_bins={num_bins}, with_clip_embeddings=True, "
        f"early_segments_as_context=True, normalize_early_timestamps=True, "
        f"context_mask_ratio={context_mask_ratio}, no_timestamp_in_context={no_timestamp_in_context}, dynamic_location={dynamic_location}, only_use_augmented_context=True, "
        f"continuous_random_mask={continuous_random_mask})"
        f"|video_resize_central_crop({crop_size})"
        f"|init_padding_mask"
    )
    return preproc_spec_train

  @evaluate_lazily
  def get_preproc_spec_eval(num_frames, num_dense_outputs):
    preproc_spec_eval = (
        f"decode_and_subsample_densecap_video({num_frames}, False, zero_pad_frames=False, with_clip_embeddings=True)"
        f"|decode_activity_net_dense_caption_annotations_dense_outputs_aug_context"
        f"(False, {num_dense_outputs}, '{tokenizer_path}',{max_text_tokens}, num_bins={num_bins}, with_clip_embeddings=True, "
        f"early_segments_as_context=True, normalize_early_timestamps=True)"
        f"|video_resize_central_crop({crop_size})"
        f"|init_padding_mask"
    )
    return preproc_spec_eval

  # Train dataset(s).
  config.dataset_configs.train = ml_collections.ConfigDict()
  config.dataset_configs.train.sources = [
      ml_collections.ConfigDict({
          'source': 'tfrecord',
          'tfrecords': common.ANET_TRAIN_TFRECORD_PATH,
          'size': common.ANET_TRAIN_SIZE,
          'context_features': context_features,
          'sequence_features': sequence_features,
          'shuffle_buffer_size': 1024,
          'cache': False,
          'preproc_spec': get_preproc_spec_train(
              config.get_ref('num_frames'),
              config.get_ref('num_dense_outputs'),
              config.get_ref('no_timestamp_in_context'),
              config.get_ref('context_mask_ratio'),
              config.get_ref('dynamic_location'),
          ),
      })
  ]
  config.dataset_configs.train.postproc_spec = 'drop(["_seed"])'

  # Evaluation dataset(s).
  config.dataset_configs.eval = ml_collections.ConfigDict()
  config.dataset_configs.eval.sources = [
      ml_collections.ConfigDict({
          'source': 'tfrecord',
          'tfrecords': common.ANET_VAL_TFRECORD_PATH,
          'size': common.ANET_VAL_SIZE,
          'context_features': context_features,
          'sequence_features': sequence_features,
          'shuffle_buffer_size': 1,
          'cache': False,
          'preproc_spec': get_preproc_spec_eval(
              config.get_ref('num_frames'),
              config.get_ref('num_dense_outputs'),
          ),
      }),
  ]
  config.dataset_configs.eval.postproc_spec = 'drop(["_seed"])'

  @evaluate_lazily
  def get_input_shape(num_frames):
    return [-1, num_frames, crop_size, crop_size, 3]
  # Dataset configs needed by the trainer.
  config.dataset_configs.extra_meta_data = {
      'input_shape': get_input_shape(config.get_ref('num_frames')),
  }

  config.rng_seed = 0

  config.additional_input_spec = [
      ((-1, config.num_frames, 768), 'float32'),
      ((-1, 2, 256), 'int32')]

  # Model.
  config.model = ml_collections.ConfigDict()
  config.model.model_dtype_str = 'float32'
  config.model.model_name = 'streaming_vid2seq'
  config.model.pixel_mean = (
      0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255)
  config.model.pixel_std = (
      0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255)
  config.model.decode_method = 'beam'
  config.model.decode_brevity_penalty_alpha = 0.6
  config.model.decode_beam_size = 4
  config.model.max_caption_length = max_text_tokens
  config.model.vocab_size = common.SP_VOCAB_SIZE + num_bins
  config.model.num_bins = num_bins

  config.model.encoder = ml_collections.ConfigDict()
  config.model.encoder.share_encoder = True
  config.model.encoder.encoder_type = 'cat_encoder'
  config.model.encoder.cat_encoder = ml_collections.ConfigDict()
  config.model.encoder.cat_encoder.encoder_type = 'vit'
  config.model.encoder.cat_encoder.dim = 2048
  config.model.encoder.cat_encoder.layers = 12
  config.model.encoder.cat_encoder.heads = 12
  config.model.encoder.cat_encoder.pos_embed = 'learned_1d'
  config.model.encoder.cat_encoder.dropout_rate = 0.
  config.model.encoder.cat_encoder.t5_dropout_rate = 0.1
  config.model.encoder.cat_encoder.stochastic_depth = 0.
  config.model.encoder.cat_encoder.pretrained_config = 't5_1_1_base'
  config.model.encoder.cat_encoder.proj_dim = 768

  config.model.decoder_type = 't5_decoder'
  config.model.decoder = ml_collections.ConfigDict()
  config.model.decoder.t5_decoder = ml_collections.ConfigDict()
  config.model.decoder.t5_decoder.type_vocab_size = 0
  config.model.decoder.t5_decoder.gate = False
  config.model.decoder.t5_decoder.logits_via_embedding = False
  config.model.decoder.t5_decoder.dropout_rate = 0.1
  config.model.decoder.t5_decoder.num_frames = config.get_ref('num_frames')
  config.model.decoder.t5_decoder.pretrained_config = 't5_1_1_base'

  config.model.num_dense_outputs = config.get_ref('num_dense_outputs')
  config.model.early_segments_as_context = True
  config.model.normalize_early_timestamps = True
  config.model.no_timestamp_in_context = config.get_ref('no_timestamp_in_context')
  config.model.num_dense_outputs_test = 2

  config.weights = '/path/to/vid2seq/yt-temporal-1b'

  # Training.
  config.batch_size = 32
  # optimizer
  config.optimizer = ml_collections.ConfigDict()
  config.optimizer.optimizer = 'adamw'
  config.optimizer.weight_decay = 0.0
  config.optimizer.skip_scale_and_bias_regularization = True
  config.optimizer.grad_clip = ml_collections.ConfigDict()
  config.optimizer.grad_clip.clip_method = 'clip_by_global_norm'
  config.optimizer.grad_clip.clip_value = 1.0

  # learning rate and training schedule
  num_epochs = 10
  iters_per_epoch = common.ANET_TRAIN_SIZE // config.batch_size
  config.num_training_steps = num_epochs * iters_per_epoch
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.steps_per_cycle = config.num_training_steps
  config.lr_configs.warmup_steps = config.num_training_steps // 10
  config.lr_configs.base_learning_rate = 1e-4
  config.log_eval_steps = iters_per_epoch
  config.checkpoint_steps = iters_per_epoch
  config.eval_first_step = False
  config.checkpoint_max_to_keep = 10

  # Logging.
  config.eval_meteor_spice = False  ##
  config.eval_only = False  ##
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.log_summary_steps = 50  # train summary steps
  config.log_large_summary_steps = 1000  # Expensive summary operations freq
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  config.evaluator = 'densecap'
  config.eval_step_multiplier = 1.3

  return config


