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
"""Default config for launching a GER-ALD pretraining."""

import ml_collections


def get_config():
  """Returns the configuration."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'gerald_training'

  # Dataset.
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.shuffle_buffer_size = 1000
  config.dataset_configs.tokenizer_type = 'bert'
  config.dataset_configs.max_context_tokens = 20
  config.data_dtype_str = 'float32'
  config.dataset_configs.data_dir = 'where_your_data_is'
  config.dataset_configs.train_datasets = ('dataset1-split1', 'dataset2-split2')
  config.dataset_configs.eval_datasets = ('name_of_seen_split', 'name_of_unseen_split')
  config.dataset_configs.wikid2id_path = 'file_mapping_each_entity_identifier_to_an_unique_int_id'

  config.rng_seed = 0

  # GER codes.
  config.vocab_size = 30522 - 2
  config.code_length = 4
  config.ger_bos = 101
  config.ger_eos = 102
  config.load_codes_from = path_to_codes

  # Model.
  config.model = ml_collections.ConfigDict()
  config.model.backbone_args = ml_collections.ConfigDict()
  config.model.backbone_args.use_ln_pre = True
  config.model.backbone_args.use_ln_post = True
  config.model.backbone_args.pe_bias = False
  config.model.backbone_args.use_class_embedding = True
  config.model.backbone_args.embed_dim = 1024
  config.model.backbone_args.depth = 24
  config.model.backbone_args.num_heads = 16
  config.model.backbone_args.patch_size = 14
  config.model.label_smooth = 0.3
  config.model.dropout_prob = 0.1
  config.model.decode_beam_size = 4
  config.weights = path_to_pretrained_weights

  # Training.
  config.batch_size = 4096
  # optimizer
  config.optimizer = ml_collections.ConfigDict()
  config.optimizer.optimizer = 'adamw'
  config.optimizer.weight_decay = 0.0
  config.optimizer.skip_scale_and_bias_regularization = True
  config.optimizer.decoder_multiplier = 10.0
  config.optimizer.decoder_layer_prefix = 'textual'

  # learning rate and training schedule
  config.num_training_steps = 600000
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.steps_per_cycle = config.num_training_steps
  config.lr_configs.warmup_steps = 250
  config.lr_configs.base_learning_rate = 1e-5
  config.log_eval_steps = 13000

  # Logging.
  config.xprof = True  # Profile using xprof.
  config.log_summary_steps = 50  # train summary steps
  config.checkpoint = True

  return config


