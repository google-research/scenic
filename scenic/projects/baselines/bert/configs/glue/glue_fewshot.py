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

"""Most common few-shot eval configuration for GLUE."""

import ml_collections
from scenic.projects.baselines.bert.configs.glue import glue_common

FEW_SHOT_TASKS = ['sst2', 'mnli_matched', 'mnli_mismatched']


def get_glue_task_config(task_name, batch_size):
  """Returns GLUE task config."""
  config = ml_collections.ConfigDict()
  config.dataset_name = 'bert_glue'
  config.data_dtype_str = 'float32'
  config.batch_size = batch_size
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.task = task_name
  task_path = glue_common.GLUE_TASK_PATH[task_name]
  config.dataset_configs.input_meta_data_path = glue_common.INPUT_MEAT_DATA_PATH.format(
      task_path=task_path)
  config.dataset_configs.train_data_path = glue_common.TRAIN_DATA_PATH.format(
      task_path=task_path)
  config.dataset_configs.eval_data_path = glue_common.EVAL_DATA_PATH.format(
      task_path=task_path)
  config.dataset_configs.prefetch_to_device = 2
  return config


def get_config(batch_size=None):
  """Returns a standard-ish fewshot eval configuration for BERT."""
  config = ml_collections.ConfigDict()
  config.batch_size = batch_size
  config.rng_seed = 42
  # We use the prelogits of the the next_sentence_prediction head for fewshot
  # eval on classification tasks.
  config.representation_layer = 'next_sentence_prediction_head/pre_logits'
  config.log_steps = 50_000
  config.datasets = [
      get_glue_task_config(task_name, batch_size)
      for task_name in FEW_SHOT_TASKS
  ]
  config.shots = [1, 5, 10, 25, 100, 500, 1000]
  config.l2_regs = [2.0**i for i in range(-10, 20)]
  config.walk_first = ('sst2', 100)

  return config
