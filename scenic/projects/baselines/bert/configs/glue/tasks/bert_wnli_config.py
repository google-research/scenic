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
r"""Default configs for BERT finetuning on WNLI.

"""
# pylint: enable=line-too-long

import ml_collections
from scenic.projects.baselines.bert.configs.glue import glue_common

VARIANT = 'BERT-B'
INIT_FROM = ml_collections.ConfigDict({
  'checkpoint_path': '',
  'model_config': 'SET-MODEL-CONFIG',
})


def get_config(variant=VARIANT, init_from=INIT_FROM):
  """Returns configuration for BERT."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'bert'

  # Dataset.
  config.dataset_name = 'bert_glue'
  config.data_dtype_str = 'float32'
  config.batch_size = 32
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.task = 'wnli'
  config.dataset_configs.prefetch_to_device = 2
  task_path = glue_common.GLUE_TASK_PATH[config.dataset_configs.task]
  config.dataset_configs.input_meta_data_path = glue_common.INPUT_MEAT_DATA_PATH.format(
      task_path=task_path)
  config.dataset_configs.train_data_path = glue_common.TRAIN_DATA_PATH.format(
      task_path=task_path)
  config.dataset_configs.eval_data_path = glue_common.EVAL_DATA_PATH.format(
      task_path=task_path)

  # Model.
  _, model_size = variant.split('-')
  config.model_name = 'bert_classification'
  config.model_dtype_str = 'float32'
  config.model = ml_collections.ConfigDict()
  config.model.stem = ml_collections.ConfigDict()
  config.model.stem.hidden_size = glue_common.HIDDEN_SIZE[model_size]
  config.model.stem.embedding_width = glue_common.EMBEDDING_WIDTH[model_size]
  config.model.stem.max_position_embeddings = 512
  config.model.stem.dropout_rate = 0.1
  config.model.encoder = ml_collections.ConfigDict()
  config.model.encoder.num_heads = glue_common.NUM_HEADS[model_size]
  config.model.encoder.mlp_dim = glue_common.MLP_DIM[model_size]
  config.model.encoder.num_layers = glue_common.NUM_LAYERS[model_size]
  config.model.encoder.attention_dropout_rate = 0.1
  config.model.encoder.dropout_rate = 0.1
  config.model.encoder.pre_norm = True
  config.model.head = ml_collections.ConfigDict()
  config.model.head.type = 'classification'
  config.model.head.hidden_size = glue_common.HIDDEN_SIZE[model_size]

  # Pre-training.
  config.init_from = init_from
  config.init_from.unlock()
  config.init_from.restore_next_sentence_prediction_head_params = True

  # Training.
  config.trainer_name = 'bert_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  config.num_training_epochs = None
  # Training data size 635 examples, 3 epochs.
  config.num_training_steps = 57
  config.log_summary_steps = 1000
  config.log_eval_steps = 9
  # Eval data size is 71 examples.
  config.steps_per_eval = 3
  config.rng_seed = 42

  # Learning rate.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant*linear_warmup*linear_decay'
  config.lr_configs.total_steps = config.num_training_steps
  config.lr_configs.end_learning_rate = 0.0
  config.lr_configs.warmup_steps = 5
  config.lr_configs.base_learning_rate = 3e-05

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 1000
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
