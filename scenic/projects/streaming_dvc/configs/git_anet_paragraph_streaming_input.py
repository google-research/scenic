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
r"""Config for ActivityNet Paragraph Captioning.

"""

import ml_collections
from scenic.projects.streaming_dvc.configs import common
evaluate_lazily = common.evaluate_lazily


def get_config():
  """Returns the configuration for GIT."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'git_anet_paragraph_streaming_input'

  # Dataset.
  config.dataset_name = 'flexio_tfrecord'
  config.data_dtype_str = 'float32'

  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.pp_libs = ['scenic.projects.streaming_dvc.io.ops']
  tokenizer_path = common.BERT_TOKENIZER_PATH
  config.dataset_configs.tokenizer_weight_path = tokenizer_path

  # Pre-processing settings
  context_features = {
      'caption/string': {'feature_type': 'VarLen', 'dtype': 'string'},
      'video_id': {'feature_type': 'VarLen', 'dtype': 'string'},
      'media_id': {'feature_type': 'VarLen', 'dtype': 'string'},
  }
  sequence_features = {
      'image/encoded': {'feature_type': 'FixedLenSequence', 'shape': [], 'dtype': 'string'},
  }

  context_features_para = context_features.copy()
  context_features_para.update({
      'split': {'feature_type': 'VarLen', 'dtype': 'int64'},
  })
  sequence_features_para = sequence_features.copy()

  crop_size = 224
  num_captions_per_sample_train = 1  # one paragraph annotation for training.
  num_captions_per_sample_eval = 2  # two sets of paragraph annotations for eval.
  max_text_tokens = 128

  config.num_frames = 64
  concat_captions_train = 'concat_all'
  concat_captions_eval = 'concat_twosplit'

  @evaluate_lazily
  def get_preproc_spec_train(num_frames):
    preproc_spec_train = (
        f"decode_and_subsample_video({num_frames}, True)"
        f"|decode_activity_net_paragraph_caption_annotations('{tokenizer_path}', {num_captions_per_sample_train}, {max_text_tokens}, '{concat_captions_train}')"
        f"|video_resize_central_crop({crop_size})"
        f"|init_padding_mask"
    )
    return preproc_spec_train

  @evaluate_lazily
  def get_preproc_spec_eval(num_frames):
    preproc_spec_eval = (
        f"decode_and_subsample_video({num_frames}, False, additional_keys=('split', 'media_id'), additional_keys_decode_bytes=(False, True))"
        f"|decode_activity_net_paragraph_caption_annotations('{tokenizer_path}', {num_captions_per_sample_eval}, {max_text_tokens}, '{concat_captions_eval}', additional_keys=('media_id',))"
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
          'preproc_spec': get_preproc_spec_train(config.get_ref('num_frames')),
      })
  ]
  config.dataset_configs.train.postproc_spec = 'drop(["_seed"])'

  # Evaluation dataset(s).
  config.dataset_configs.eval = ml_collections.ConfigDict()
  config.dataset_configs.eval.sources = [
      ml_collections.ConfigDict({
          'source': 'tfrecord',
          'tfrecords': common.ANET_PARA_VAL_TFRECORD_PATH,
          'size': common.ANET_PARA_VAL_SIZE,
          'context_features': context_features_para,
          'sequence_features': sequence_features_para,
          'shuffle_buffer_size': 1,
          'cache': False,
          'preproc_spec': get_preproc_spec_eval(config.get_ref('num_frames')),
      }),
  ]
  config.dataset_configs.eval.postproc_spec = 'drop(["_seed"])'

  @evaluate_lazily
  def get_input_shape(num_frames):
    return [-1, num_frames, crop_size, crop_size, 3]
  # Dataset configs needed by the trainer.
  config.dataset_configs.test_annotation_path = ''
  config.dataset_configs.extra_meta_data = {
      'input_shape': get_input_shape(config.get_ref('num_frames')),
  }

  config.rng_seed = 0

  # Model.
  config.model = ml_collections.ConfigDict()
  config.model.model_dtype_str = 'float32'
  config.model.model_name = 'streaming_model'
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
  config.model.max_caption_length = 128

  config.model.streaming_method = 'kmeans'
  config.model.streaming_buffer_size = (256 + 1) * 2
  config.model.kmeans_num_iters = 2

  config.weights = '/path/to/git_pretrain'

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
  config.num_training_steps = 5000
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * piecewise_constant * linear_warmup'
  config.lr_configs.decay_events = (4000,)
  config.lr_configs.decay_factors = [0.1]
  config.lr_configs.warmup_steps = 250
  config.lr_configs.base_learning_rate = 2e-5
  config.log_eval_steps = 500
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
  config.eval_first_step = False
  config.eval_step_multiplier = 1.3

  return config


