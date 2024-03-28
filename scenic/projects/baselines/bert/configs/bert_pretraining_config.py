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
r"""Default configs for BERT pretraining.

"""
# pylint: enable=line-too-long

import ml_collections
from scenic.projects.baselines.bert.configs.glue import glue_fewshot

VARIANT = 'BERT-B'

EMBEDDING_WIDTH = {'Ti': 128, 'S': 128, 'B': 768, 'L': 1024}
HIDDEN_SIZE = {'Ti': 128, 'S': 256, 'B': 768, 'L': 1024}
NUM_HEADS = {'Ti': 2, 'S': 4, 'B': 12, 'L': 16}
MLP_DIM = {'Ti': 512, 'S': 1024, 'B': 3072, 'L': 4096}
NUM_LAYERS = {'Ti': 6, 'S': 12, 'B': 12, 'L': 24}


def get_config():
  """Returns configuration for BERT."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'bert'

  # Dataset.
  config.dataset_name = 'bert_wikibooks'
  config.data_dtype_str = 'float32'
  config.batch_size = 512
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.prefetch_to_device = 2
  # Training data:
  config.dataset_configs.train_data_loader = ml_collections.ConfigDict()
  wikibooks_train_loader = config.dataset_configs.train_data_loader
  wikibooks_train_loader.seq_length = 512
  wikibooks_train_loader.max_predictions_per_seq = 76
  wikibooks_train_loader.use_next_sentence_label = True
  wikibooks_train_loader.use_position_id = False
  wikibooks_train_loader.use_v2_feature_names = False
  wikibooks_train_loader.file_type = 'tfrecord'
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
  wikibooks_train_loader.enable_shared_tf_data_service_between_parallel_trainers = (
      False  # pylint: disable=line-too-long
  )
  wikibooks_train_loader.apply_tf_data_service_before_batching = False
  wikibooks_train_loader.trainer_id = ''
  wikibooks_train_loader.tfds_name = None
  wikibooks_train_loader.tfds_split = None
  wikibooks_train_loader.tfds_data_dir = None
  wikibooks_train_loader.tfds_as_supervised = False
  wikibooks_train_loader.tfds_skip_decoding_feature = ''
  wikibooks_train_loader.global_batch_size = config.batch_size
  wikibooks_train_loader.prefetch_buffer_size = None  # Autotune.
  wikibooks_train_loader.autotune_algorithm = None
  # Validation data:
  config.dataset_configs.val_data_loader = ml_collections.ConfigDict()
  wikibooks_val_loader = config.dataset_configs.val_data_loader
  wikibooks_val_loader.seq_length = 512
  wikibooks_val_loader.max_predictions_per_seq = 76
  wikibooks_val_loader.use_next_sentence_label = True
  wikibooks_val_loader.use_position_id = False
  wikibooks_val_loader.use_v2_feature_names = False
  wikibooks_val_loader.file_type = 'tfrecord'
  # Add path to validation files containing tf.records.
  wikibooks_val_loader.input_path = ''
  wikibooks_val_loader.drop_remainder = False
  wikibooks_val_loader.cycle_length = None
  wikibooks_val_loader.block_length = 1
  wikibooks_val_loader.deterministic = None
  wikibooks_val_loader.sharding = True
  wikibooks_val_loader.enable_tf_data_service = False
  wikibooks_val_loader.tf_data_service_address = None
  wikibooks_val_loader.enable_shared_tf_data_service_between_parallel_trainers = (
      False  # pylint: disable=line-too-long
  )
  wikibooks_val_loader.apply_tf_data_service_before_batching = False
  wikibooks_val_loader.trainer_id = ''
  wikibooks_val_loader.tfds_name = None
  wikibooks_val_loader.tfds_split = None
  wikibooks_val_loader.tfds_data_dir = None
  wikibooks_val_loader.tfds_as_supervised = False
  wikibooks_val_loader.tfds_skip_decoding_feature = ''
  wikibooks_val_loader.global_batch_size = config.batch_size
  wikibooks_val_loader.prefetch_buffer_size = None  # Autotune.
  wikibooks_val_loader.autotune_algorithm = None

  # Model.
  _, model_size = VARIANT.split('-')
  config.model_name = 'bert'
  config.model_dtype_str = 'float32'
  config.model = ml_collections.ConfigDict()
  config.model.stem = ml_collections.ConfigDict()
  config.model.stem.hidden_size = HIDDEN_SIZE[model_size]
  config.model.stem.embedding_width = EMBEDDING_WIDTH[model_size]
  config.model.stem.max_position_embeddings = 512
  config.model.stem.dropout_rate = 0.1
  config.model.encoder = ml_collections.ConfigDict()
  config.model.encoder.num_heads = NUM_HEADS[model_size]
  config.model.encoder.mlp_dim = MLP_DIM[model_size]
  config.model.encoder.num_layers = NUM_LAYERS[model_size]
  config.model.encoder.attention_dropout_rate = 0.1
  config.model.encoder.dropout_rate = 0.1
  config.model.encoder.pre_norm = True
  config.model.head = ml_collections.ConfigDict()
  config.model.head.type = 'pretraining'
  config.model.head.hidden_size = HIDDEN_SIZE[model_size]

  # Training.
  config.trainer_name = 'bert_trainer'
  optim = ml_collections.ConfigDict()
  optim.optax_name = 'scale_by_adam'
  optim.optax_configs = ml_collections.ConfigDict({  # Optimizer settings.
      'b1': 0.9,
      'b2': 0.999,
  })
  config.optimizer = optim
  config.num_training_epochs = None
  config.num_training_steps = 1000_000
  config.log_eval_steps = 1000
  config.steps_per_eval = 64
  config.rng_seed = 42
  config.optimizer = optim

  # Gradient clipping (BERT clips grads before pmean).
  config.max_grad_norm = 1.0
  config.optimizer.max_grad_norm = None

  # Fewshot.
  config.fewshot = glue_fewshot.get_config(config.batch_size)
  config.fewshot.log_eval_steps = 50_000

  sched = ml_collections.ConfigDict()
  sched.re = '(.*)'
  sched.lr_configs = ml_collections.ConfigDict()
  sched.lr_configs.learning_rate_schedule = 'compound'
  sched.lr_configs.factors = 'constant * linear_warmup * linear_decay'
  sched.lr_configs.total_steps = config.num_training_steps
  sched.lr_configs.steps_per_cycle = sched.lr_configs.total_steps
  sched.lr_configs.warmup_steps = 10_000
  sched.lr_configs.base_learning_rate = 1e-4
  config.schedule = ml_collections.ConfigDict({'all': sched})

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 20000
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  return config


