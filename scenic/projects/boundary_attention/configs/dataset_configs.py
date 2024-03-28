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

"""Dataset configurations."""

from typing import Optional, Any

import ml_collections
from scenic.projects.boundary_attention.configs import kaleidoshapes_config

DATASET_CONFIG = {
    'kaleidoshapes': kaleidoshapes_config.get_config_kaleidoshapes,
    'testing': kaleidoshapes_config.get_config_testing,
}


def get_dataset_config(
    dataset_name: str,
    dataset_dir: str,
    input_shape: Optional[Any],
) -> ml_collections.ConfigDict:
  """Returns the dataset config."""

  # Set a default input size if not defined
  if input_shape is None:
    input_shape = (125, 125, 3)

  # Fetch the dataset config
  try:
    return DATASET_CONFIG[dataset_name](dataset_dir, input_shape)
  except:
    raise NotImplementedError(
        f'Did not recognize dataset_name {dataset_name}') from None
