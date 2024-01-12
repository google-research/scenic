# Copyright 2023 The Scenic Authors.
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

"""Base training config for CLIP + T5 w Transformers."""

import ml_collections
from scenic.projects.lang4video.configs import base_clip_trf_t5_trf as general_base_clip_trf_t5_trf
from scenic.projects.lang4video.configs.train import mixin


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = general_base_clip_trf_t5_trf.get_config(run_local)
  config.update(mixin.get_config(run_local))

  config.optimizer_configs.params_to_freeze = (r'^\w+_encoder/encoders_0/',)

  return config
