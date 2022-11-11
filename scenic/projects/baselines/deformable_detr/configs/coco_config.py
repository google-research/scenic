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
