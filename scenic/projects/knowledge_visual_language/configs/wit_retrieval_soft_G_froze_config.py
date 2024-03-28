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

r"""WIT Retrieval + Captioning Pre-Training."""

import ml_collections

TRAIN_DATA_SIZE = 5000000


def get_config() -> ml_collections.ConfigDict:
  """Returns the base experiment configuration."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'image_caption_debug'

  config.optimizer = 'adafactor'
  batch_size = 5 * 2 * 256
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.momentum = None
  config.optimizer_configs.weight_decay_rate = 1e-2
  config.optimizer_configs.clipping_threshold = 1.0
  config.optimizer_configs.skip_scale_and_bias_regularization = False

  config.frozen_patterns = []
  config.not_frozen_patterns = [('value_perceiver/.*', 1.5),
                                ('text_encoder/.*', 1),
                                ('shared_token_embedder/.*', 0.5),
                                ('query_head/.*', 1.2), ('out_decoder/.*', 1),
                                ('key_head/.*', 1.2), ('img_encoder/.*', 1),
                                ('head_out/.*', 1.2), ('fusion_encoder/.*', 1),
                                ('att_transform/.*', 1)]

  config.grad_clip_configs = ml_collections.ConfigDict()
  config.grad_clip_configs.clip_method = 'clip_by_global_norm'
  config.grad_clip_configs.clip_value = 1.0

  config.batch_size = batch_size
  config.eval_batch_size = batch_size
  config.rng_seed = 0
  config.update_num = True
  config.num_training_epochs = 50
  config.data_dtype_str = 'float32'

  # Model
  config.model_name = 'retrieval_image_captioner_soft'
  config.model = ml_collections.ConfigDict()
  config.model.image_model = 'vit'
  config.model.t5_name = 't5_1_1_large'
  # ['t5_1_1_small', 't5_1_1_base', 't5_1_1_large', 't5_1_1_xl', 't5_1_1_xxl']
  config.model.num_fusion_layers = 8
  config.model.n_compressed_tokens = 64
  config.model.key_dim = 512
  config.model.dropout_rate = 0.0
  config.model.temperature = 0.2
  config.model.fuse_retrieval = True
  config.model.supervised_retrieval = True
  config.model.in_batch_neg = True
  config.model.retrieval_ratio = 1 / 2
  config.model.label_smoothing = 1e-2
  config.model.vit_name = 'G/14'
  config.model.vit_model_path = 'JFT3b-G/14'
  config.model.vit_num_frozen_layers = 0.6
  # [JFT3b-B/32, JFT3b-B/16, JFT3b-L/16, JFT3b-g/14, JFT3b-G/14]

  # Dataset.
  config.dataset_name = 'wiki_image_text_generation'
  config.dataset_configs = ml_collections.ConfigDict()

  # Learning rate.
  config.num_train_examples = TRAIN_DATA_SIZE
  steps_per_epoch = TRAIN_DATA_SIZE // config.batch_size
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.total_steps = int(config.num_training_epochs *
                                      steps_per_epoch)
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * rsqrt_decay * linear_warmup'
  config.lr_configs.warmup_steps = int(0.1 * config.lr_configs.total_steps)
  config.lr_configs.timescale = 5000
  # config.lr_configs.steps_per_cycle = config.lr_configs.total_steps
  config.lr_configs.base_learning_rate = 3e-4
  config.lr_configs.end_learning_rate = 1e-6

  # Logging.
  config.log_summary_steps = 100
  config.log_eval_steps = 1000
  config.checkpoint_steps = 5000
  config.write_summary = True
  config.xprof = True  # Profile using xprof
  config.checkpoint = True  # Do checkpointing.
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  # Initalisation configs
  config.init_from = ml_collections.ConfigDict()
  # Initializing from a vidcap model.
  config.init_from.only_params = False
  config.init_from.load_key_encoder = False
  config.init_from.encoder = ml_collections.ConfigDict()
  config.init_from.encoder.init_from_vit = False
  config.init_from.encoder.checkpoint_path = None
  return config
