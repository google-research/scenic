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
