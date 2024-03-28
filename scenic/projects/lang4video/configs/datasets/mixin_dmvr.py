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

"""Mixin config for DMVR datasets."""

import ml_collections
from scenic.projects.lang4video.configs.datasets import mixin


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = mixin.get_config(run_local)

  config.dataset_configs = ml_collections.ConfigDict()

  config.dataset_configs.load_train = False
  config.dataset_configs.load_test = False

  config.dataset_configs.val_on_test = False
  config.dataset_configs.test_on_val = False  # TODO(sacastro): unused.

  config.dataset_configs.keep_val_key = True
  config.dataset_configs.keep_test_key = True  # TODO(sacastro): unused.

  return config
