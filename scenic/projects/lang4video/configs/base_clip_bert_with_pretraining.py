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

"""Base config for CLIP+BERT w/ pretraining."""

import ml_collections
from scenic.projects.lang4video.configs import base_clip_bert


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = base_clip_bert.get_config(run_local)

  config.experiment_name = 'clip_bert_with_pretraining'

  config.model.text_encoder.config_name = 'base'
  config.model.text_encoder.pretraining_mode = True
  config.model.text_encoder.embedding_size = 512
  config.model.text_encoder.compute_mlm = True

  config.text_dataset_name = 'bert_wikibooks'

  config.text_dataset_configs = ml_collections.ConfigDict()

  config.text_dataset_configs.prefetch_to_device = 2

  config.text_dataset_configs.truncate_to_max_num_words = False

  config.text_dataset_configs.train_data_loader = ml_collections.ConfigDict()
  wikibooks_train_loader = config.text_dataset_configs.train_data_loader
  # The values of `seq_length` and `max_predictions_per_seq` cannot be changed
  # because they are used to parse the features. They could be changed once the
  # data is loaded.
  wikibooks_train_loader.seq_length = 512
  wikibooks_train_loader.max_predictions_per_seq = 76
  wikibooks_train_loader.use_next_sentence_label = True
  wikibooks_train_loader.use_position_id = False
  wikibooks_train_loader.use_v2_feature_names = False
  # Add path to training files containing tf.records.
  wikibooks_train_loader.input_path = ''
  wikibooks_train_loader.drop_remainder = True
  wikibooks_train_loader.shuffle_buffer_size = 100
  wikibooks_train_loader.cycle_length = None
  wikibooks_train_loader.block_length = 1
  wikibooks_train_loader.deterministic = None
  wikibooks_train_loader.sharding = True
  wikibooks_train_loader.enable_tf_data_service = False
  wikibooks_train_loader.tf_data_service_address = None
  wikibooks_train_loader.tfds_name = None
  wikibooks_train_loader.tfds_split = None
  wikibooks_train_loader.tfds_as_supervised = False
  wikibooks_train_loader.tfds_skip_decoding_feature = ''
  wikibooks_train_loader.global_batch_size = config.get_ref('batch_size')
  wikibooks_train_loader.prefetch_buffer_size = None  # Autotune.
  wikibooks_train_loader.enable_shared_tf_data_service_between_parallel_trainers = False
  wikibooks_train_loader.apply_tf_data_service_before_batching = False
  wikibooks_train_loader.trainer_id = None

  # The val split is actually unused, but the config is required.

  config.text_dataset_configs.val_data_loader = ml_collections.ConfigDict()
  wikibooks_val_loader = config.text_dataset_configs.val_data_loader
  wikibooks_val_loader.seq_length = 512
  wikibooks_val_loader.max_predictions_per_seq = 76
  wikibooks_val_loader.use_next_sentence_label = True
  wikibooks_val_loader.use_position_id = False
  wikibooks_val_loader.use_v2_feature_names = False
  # Add path to validation files containing tf.records.
  wikibooks_val_loader.input_path = ''
  wikibooks_val_loader.drop_remainder = False
  wikibooks_val_loader.cycle_length = None
  wikibooks_val_loader.block_length = 1
  wikibooks_val_loader.deterministic = None
  wikibooks_val_loader.sharding = True
  wikibooks_val_loader.enable_tf_data_service = False
  wikibooks_val_loader.tf_data_service_address = None
  wikibooks_val_loader.tfds_name = None
  wikibooks_val_loader.tfds_split = None
  wikibooks_val_loader.tfds_as_supervised = False
  wikibooks_val_loader.tfds_skip_decoding_feature = ''
  wikibooks_val_loader.global_batch_size = config.get_ref('eval_batch_size')
  wikibooks_val_loader.prefetch_buffer_size = None  # Autotune.
  wikibooks_val_loader.enable_shared_tf_data_service_between_parallel_trainers = False
  wikibooks_val_loader.apply_tf_data_service_before_batching = False
  wikibooks_val_loader.trainer_id = None

  return config
