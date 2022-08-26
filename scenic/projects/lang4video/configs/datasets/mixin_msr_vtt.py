"""Mixin config for MSR-VTT."""

import ml_collections
from scenic.projects.lang4video.configs.datasets import mixin_dmvr


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = mixin_dmvr.get_config(run_local)
  config.trainer_name = 'zero_shot_text_to_visual_retrieval_trainer'
  config.dataset_name = 'msrvtt'
  config.dataset_canonical_name = 'msrvtt'
  return config
