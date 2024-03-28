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
r"""Default configs for BERT finetuning on GLUE.

"""
# pylint: enable=line-too-long

import ml_collections

_GLUE_TASKS = [
    'stsb', 'cola', 'sst2', 'mrpc', 'qqp', 'mnli_matched', 'mnli_mismatched',
    'rte', 'wnli', 'qnli'
]

VARIANT = 'BERT-B'

INIT_FROM = ml_collections.ConfigDict({
  'checkpoint_path': '',
  'model_config': 'SET-MODEL-CONFIG',
})


def get_config():
  """Returns configuration for BERT."""
  config = ml_collections.ConfigDict()
  config.rng_seed = 42
  config.glue_task = ''
  config.variant = VARIANT
  config.init_from = INIT_FROM
  return config


