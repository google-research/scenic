"""Base training config for CLIP+T5."""

import ml_collections
from scenic.projects.lang4video.configs import base_clip_t5 as general_base_clip_t5
from scenic.projects.lang4video.configs.train import mixin


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = general_base_clip_t5.get_config(run_local)
  config.update(mixin.get_config(run_local))
  return config
