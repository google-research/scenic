"""Mixin config for Kinetics600."""

import ml_collections
from scenic.projects.lang4video.configs import base_clip
from scenic.projects.lang4video.configs.datasets import mixin_dmvr


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = mixin_dmvr.get_config(run_local)
  config.trainer_name = 'zero_shot_classification_trainer'
  config.dataset_name = 'kinetics600'
  config.dataset_canonical_name = 'kinetics600'
  config.dataset_configs.is_classification = True
  config.class_templates = (['{}'] if run_local else
                            base_clip.CLIP_KINETICS_TEMPLATES)
  return config
