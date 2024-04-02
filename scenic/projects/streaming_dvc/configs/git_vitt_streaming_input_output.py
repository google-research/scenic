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
r"""Config for ViTT Dense Captioning.

"""

import ml_collections
from scenic.projects.streaming_dvc.configs import common
evaluate_lazily = common.evaluate_lazily


def get_config():
  """Returns the configuration for GIT."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'git_vitt_streaming_input_output'

  # Dataset.
  config.dataset_name = 'flexio_tfrecord'
  config.data_dtype_str = 'float32'

  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.pp_libs = [
      'scenic.projects.streaming_dvc.io.ops', 'scenic.projects.streaming_dvc.io.densecap_ops']
  tokenizer_path = common.BERT_TOKENIZER_PATH
  config.dataset_configs.tokenizer_weight_path = tokenizer_path
  config.dataset_configs.test_annotation_path = common.VITT_ANN_VID2SEQ_FORMAT_PATH

  # Pre-processing settings
  context_features = {
      'caption/string': {'feature_type': 'VarLen', 'dtype': 'string'},
      'key': {'feature_type': 'VarLen', 'dtype': 'string'},
      'video/timestamps/start': {'feature_type': 'VarLen', 'dtype': 'int64'},
      'video/timestamps/end': {'feature_type': 'VarLen', 'dtype': 'int64'},
      'video/duration': {'feature_type': 'VarLen', 'dtype': 'int64'},
  }
  sequence_features = {
      'image/encoded': {'feature_type': 'FixedLenSequence', 'shape': [], 'dtype': 'string'},
      'image/timestamp': {'feature_type': 'FixedLenSequence', 'shape': [], 'dtype': 'int64'},
  }

  crop_size = 224
  max_text_tokens = 256

  config.num_frames = 64
  config.num_bins = 64
  config.num_dense_outputs = 16
  config.early_segments_as_context = True
  config.normalize_early_timestamps = True
  config.context_mask_ratio = 0.5
  config.only_use_augmented_context = True
  config.dynamic_location = False
  continuous_random_mask = True

  @evaluate_lazily
  def get_preproc_spec_train(num_frames, num_dense_outputs, early_segments_as_context, normalize_early_timestamps, num_bins, context_mask_ratio, only_use_augmented_context, dynamic_location):
    preproc_spec_train = (
        f"decode_and_subsample_densecap_video({num_frames}, False, zero_pad_frames=False, media_id_key='key')"
        f"|decode_activity_net_dense_caption_annotations_dense_outputs_aug_context"
        f"(True, {num_dense_outputs}, '{tokenizer_path}', {max_text_tokens}, "
        f"num_bins={num_bins}, early_segments_as_context={early_segments_as_context}, normalize_early_timestamps={normalize_early_timestamps}, "
        f"context_mask_ratio={context_mask_ratio}, no_timestamp_in_context=True, dynamic_location={dynamic_location}, only_use_augmented_context={only_use_augmented_context}, "
        f"continuous_random_mask={continuous_random_mask})"
        f"|video_resize_central_crop({crop_size})"
        f"|init_padding_mask"
    )
    return preproc_spec_train

  @evaluate_lazily
  def get_preproc_spec_eval(num_frames, num_dense_outputs, early_segments_as_context, normalize_early_timestamps, num_bins):
    # We use num_bins to determine the output location in decode_activity_net_dense_caption_annotations_dense_outputs_aug_context.
    # It has to be the same with num_frames.
    assert num_frames == num_bins
    preproc_spec_eval = (
        f"decode_and_subsample_densecap_video({num_frames}, False, zero_pad_frames=False, media_id_key='key')"
        f"|decode_activity_net_dense_caption_annotations_dense_outputs_aug_context"
        f"(False, {num_dense_outputs}, '{tokenizer_path}', {max_text_tokens}, "
        f"num_bins={num_bins}, early_segments_as_context={early_segments_as_context}, normalize_early_timestamps={normalize_early_timestamps})"
        f"|video_resize_central_crop({crop_size})"
        f"|init_padding_mask"
    )
    return preproc_spec_eval

  # Train dataset(s).
  config.dataset_configs.train = ml_collections.ConfigDict()
  config.dataset_configs.train.sources = [
      ml_collections.ConfigDict({
          'source': 'tfrecord',
          'tfrecords': common.VITT_TRAIN_TFRECORD_PATH,
          'size': common.VITT_TRAIN_SIZE,
          'context_features': context_features,
          'sequence_features': sequence_features,
          'shuffle_buffer_size': 1024,
          'cache': False,
          'preproc_spec': get_preproc_spec_train(
              config.get_ref('num_frames'),
              config.get_ref('num_dense_outputs'),
              config.get_ref('early_segments_as_context'),
              config.get_ref('normalize_early_timestamps'),
              config.get_ref('num_bins'),
              config.get_ref('context_mask_ratio'),
              config.get_ref('only_use_augmented_context'),
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
          'tfrecords': common.VITT_TEST_TFRECORD_PATH,
          'size': common.VITT_TEST_SIZE,
          'context_features': context_features,
          'sequence_features': sequence_features,
          'shuffle_buffer_size': 1,
          'cache': False,
          'preproc_spec': get_preproc_spec_eval(
              config.get_ref('num_frames'),
              config.get_ref('num_dense_outputs'),
              config.get_ref('early_segments_as_context'),
              config.get_ref('normalize_early_timestamps'),
              config.get_ref('num_bins'),
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

  # Model.
  config.model = ml_collections.ConfigDict()
  config.model.model_dtype_str = 'float32'
  config.model.model_name = 'streaming_dense_model'
  config.model.pixel_mean = (
      0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255)
  config.model.pixel_std = (
      0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255)
  config.model.backbone_args = ml_collections.ConfigDict()
  config.model.backbone_args.window_block_indexes = ()
  config.model.backbone_args.use_rel_pos = False
  config.model.backbone_args.use_ln_pre = True
  config.model.backbone_args.use_ln_post = True
  config.model.backbone_args.pe_bias = False
  config.model.backbone_args.use_class_embedding = True
  config.model.backbone_args.embed_dim = 1024
  config.model.backbone_args.depth = 24
  config.model.backbone_args.num_heads = 16
  config.model.backbone_args.patch_size = 14
  config.model.decode_method = 'beam'
  config.model.decode_brevity_penalty_alpha = 0.6
  config.model.decode_beam_size = 4
  config.model.num_frames = config.num_frames
  config.model.max_caption_length = max_text_tokens
  config.model.vocab_size = common.BERT_VOCAB_SIZE + config.get_ref('num_bins')
  config.model.num_bins = config.get_ref('num_bins')
  config.model.show_densecap_loss = True
  config.model.loc_loss_weight = 0.5
  config.model.num_dense_outputs = config.get_ref('num_dense_outputs')
  config.model.ignore_empty_data = True
  config.model.early_segments_as_context = config.get_ref(
      'early_segments_as_context')
  config.model.normalize_early_timestamps = config.get_ref(
      'normalize_early_timestamps')
  config.model.with_temp_emb = False
  config.model.num_dense_outputs_test = 2
  config.model.no_timestamp_in_context = True

  config.model.streaming_method = 'kmeans'
  config.model.streaming_buffer_size = (256 + 1) * 2
  config.model.kmeans_num_iters = 2
  config.model.streaming_feature_implementation = 'given_checkpoints'

  config.weights = '/path/to/git_pretrain'
  config.load_available_shape = (
      'textual/output/bias', 'textual/output/kernel',
      'textual/embedding/words/embedding')

  # Training.
  config.batch_size = 32
  # optimizer
  config.optimizer = ml_collections.ConfigDict()
  config.optimizer.optimizer = 'adamw'
  config.optimizer.weight_decay = 0.0
  config.optimizer.skip_scale_and_bias_regularization = True
  config.frozen_params = (
      ('.*image_encoder.*', 'image_encoder'),)

  # learning rate and training schedule
  config.num_training_steps = 2000
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * piecewise_constant * linear_warmup'
  config.lr_configs.decay_events = (2000,)
  config.lr_configs.decay_factors = [0.1]
  config.lr_configs.warmup_steps = 250
  config.lr_configs.base_learning_rate = 1e-5
  config.log_eval_steps = 500
  config.checkpoint_steps = 500
  config.eval_first_step = False
  config.checkpoint_max_to_keep = 100

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


