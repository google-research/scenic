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

"""Mixin config for BigTransfer datasets."""

import ml_collections
from scenic.projects.lang4video.configs.datasets import mixin


def _split(name: str, start: str, end: str) -> str:
  return f'{name}[{start}:{end}]'


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = mixin.get_config(run_local)

  run_local = bool(run_local)

  config.dataset_name = 'bit'

  config.dataset_configs = ml_collections.ConfigDict()

  # We need to re-add this here in case we merge a config with an eval config in
  # which the datasets have different values for them.
  config.dataset_configs.visual_key = 'image'
  config.dataset_configs.is_video = False
  config.dataset_configs.text_in_key = 'texts'

  config.dataset_configs.train_cache = None
  config.dataset_configs.val_cache = 'batched'

  config.dataset_configs.train_split_name = 'full'
  config.dataset_configs.train_split_start = '50000'
  config.dataset_configs.train_split_end = ''
  config.dataset_configs.train_split = _split(
      name=config.dataset_configs.get_ref('train_split_name'),
      start=config.dataset_configs.get_ref('train_split_start'),
      end=config.dataset_configs.get_ref('train_split_end'))

  config.dataset_configs.val_split_name = 'full'
  config.dataset_configs.val_split_start = ''
  config.dataset_configs.val_split_end = '4' if run_local else '50000'
  config.dataset_configs.val_split = _split(
      name=config.dataset_configs.get_ref('val_split_name'),
      start=config.dataset_configs.get_ref('val_split_start'),
      end=config.dataset_configs.get_ref('val_split_end'))

  return config
