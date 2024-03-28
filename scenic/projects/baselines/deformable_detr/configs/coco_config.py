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
r"""Default configs for COCO detection using DeformableDETR.

"""
# pylint: enable=line-too-long

from scenic.projects.baselines.deformable_detr.configs.common import get_coco_config


def get_config():
  """Returns the configuration for COCO detection using DeformableDETR."""
  config = get_coco_config()

  # Download pretrained ResNet50 checkpoints from here:
  # https://github.com/google-research/scenic/tree/main/scenic/projects/baselines pylint: disable=line-too-long
  config.pretrained_backbone_configs.checkpoint_path = 'path_to_checkpoint_of_resnet_50'

  return config
