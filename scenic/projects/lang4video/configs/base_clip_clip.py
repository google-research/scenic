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

"""Base config for CLIP visual + CLIP text."""

import ml_collections
from scenic.projects.lang4video.configs import base_clip


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = base_clip.get_config(run_local)

  config.experiment_name = 'clip_clip'

  del config.model.encoder_name
  del config.model.encoder.config_name
  del config.model.encoder

  config.model.image_encoder_name = 'clip'
  config.model.image_encoder = ml_collections.ConfigDict()
  config.model.image_encoder.config_name = 'vit_b32'

  config.model.text_encoder_name = 'clip'
  config.model.text_encoder = ml_collections.ConfigDict()
  config.model.text_encoder.config_name = 'vit_b32'

  # The following configurations are meant to be used by DMVR datasets but also
  # as input to other parts of the config.

  config.dataset_configs.min_resize = base_clip.get_clip_image_size(
      config.model.image_encoder.config_name)
  config.dataset_configs.crop_size = base_clip.get_clip_image_size(
      config.model.image_encoder.config_name)

  return config
