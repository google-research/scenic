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
r"""Default configs for COCO detection using DeformableDETR on Google Cloud.

"""
# pylint: enable=line-too-long

from scenic.projects.baselines.deformable_detr.configs.common import get_coco_config


def get_config():
  """Returns the configuration for COCO detection using DeformableDETR."""
  config = get_coco_config()

  config.dataset_configs.data_dir = 'gs://tensorflow-datasets/datasets'

  # pylint: disable=line-too-long
  config.pretrained_backbone_configs.checkpoint_path = '/workdir/scenic/scenic/projects/baselines/deformable_detr/checkpoints/ResNet50_ImageNet1k'
  # pylint: enable=line-too-long

  return config
