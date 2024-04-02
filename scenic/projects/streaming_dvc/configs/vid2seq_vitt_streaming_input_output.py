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
  config.experiment_name = 'vid2seq_vitt_streaming_input_output'

  # Dataset.
  config.dataset_name = 'flexio_tfrecord'
  config.data_dtype_str = 'float32'

  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.pp_libs = [
      'scenic.projects.streaming_dvc.io.ops',
      'scenic.projects.streaming_dvc.io.densecap_ops']
  config.dataset_configs.test_annotation_path = common.VITT_ANN_VID2SEQ_FORMAT_PATH
  tokenizer_path = 't5'
  config.dataset_configs.tokenizer_weight_path = tokenizer_path  # Used in evaluation

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
      'image/clip_embeddings': {'feature_type': 'FixedLenSequence', 'shape': [768], 'dtype': 'float32'},
  }

  crop_size = 224
  max_text_tokens = 256
  num_bins = 100

  config.num_bins = num_bins
  config.num_frames = 100
  config.num_dense_outputs = 8
  context_mask_ratio = 0.5
  config.no_timestamp_in_context = True
  continuous_random_mask = True
  dynamic_location = False

  @evaluate_lazily
  def get_preproc_spec_train(num_frames, num_dense_outputs, no_timestamp_in_context):
    preproc_spec_train = (
        # NOTE: it's important to turn off the random frame downsampling here.
        # Otherwise the time tokens are not aligned.
        f"decode_and_subsample_densecap_video({num_frames}, False, zero_pad_frames=False, with_clip_embeddings=True, media_id_key='key')"
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
        f"decode_and_subsample_densecap_video({num_frames}, False, zero_pad_frames=False, with_clip_embeddings=True, media_id_key='key')"
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
  num_epochs = 20
  iters_per_epoch = common.VITT_TRAIN_SIZE // config.batch_size
  config.num_training_steps = num_epochs * iters_per_epoch
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.steps_per_cycle = config.num_training_steps
  config.lr_configs.warmup_steps = config.num_training_steps // 10
  config.lr_configs.base_learning_rate = 2e-4
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


