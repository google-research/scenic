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

"""Define all model configs."""

import ml_collections
from scenic.projects.boundary_attention.configs import boundary_attention_model_config


MODEL_CONFIGS = {
    'boundary_attention':
        boundary_attention_model_config.get_boundary_attention_model_config(),
}


def get_model_config(model_name: str) -> ml_collections.ConfigDict:
  try:
    return MODEL_CONFIGS[model_name]
  except:
    raise NotImplementedError(
        'Did not recognize model_name %s' % model_name) from None
