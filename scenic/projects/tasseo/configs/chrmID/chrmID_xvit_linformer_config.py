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
r"""Linformer configs for X-ViT on chromosome identification task.

"""
# pylint: enable=line-too-long

from scenic.projects.tasseo.configs.chrmID import chrmID_xvit_config


def get_config():
  """Returns the X-ViT experiment configuration for metaphase sexID."""
  config = chrmID_xvit_config.get_config()
  config.experiment_name = 'chrmID-xvit-jf'
  config.model_name = 'xvit_classification'

  # Model.
  config.model.attention_fn = 'linformer'
  config.model.attention_configs.low_rank_features = 16

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
