"""Mixin config for UCF-101."""

import ml_collections
from scenic.projects.lang4video.configs import base_clip
from scenic.projects.lang4video.configs.datasets import mixin_dmvr


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = mixin_dmvr.get_config(run_local)
  config.trainer_name = 'zero_shot_classification_trainer'
  config.dataset_name = 'ucf101_dmvr'
  config.dataset_canonical_name = 'ucf101'
  config.dataset_configs.is_classification = True
  config.dataset_configs.split_number = 1
  config.dataset_configs.val_on_test = True  # It doesn't have a val split.
  config.class_templates = ['{}'] if run_local else base_clip.CLIP_UCF_TEMPLATES
  return config
