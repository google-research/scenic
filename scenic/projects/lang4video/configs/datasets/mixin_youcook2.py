"""Mixin config for YouCook2."""

import ml_collections
from scenic.projects.lang4video.configs.datasets import mixin_dmvr


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = mixin_dmvr.get_config(run_local)
  config.trainer_name = 'zero_shot_text_to_visual_retrieval_trainer'
  config.dataset_name = 'youcook2'
  config.dataset_canonical_name = 'youcook2'
  config.dataset_configs.test_on_val = True  # It doesn't have a val split.
  return config
