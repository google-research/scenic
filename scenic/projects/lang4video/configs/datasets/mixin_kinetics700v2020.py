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

"""Mixin config for Kinetics700 v2020."""

import ml_collections
from scenic.projects.lang4video.configs import base_clip
from scenic.projects.lang4video.configs.datasets import mixin_dmvr


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = mixin_dmvr.get_config(run_local)
  config.trainer_name = 'zero_shot_classification_trainer'
  config.dataset_name = 'kinetics700v2020'
  config.dataset_canonical_name = 'kinetics700v2020'
  config.dataset_configs.is_classification = True
  config.class_templates = (['{}']
                            if run_local else base_clip.CLIP_KINETICS_TEMPLATES)
  return config
