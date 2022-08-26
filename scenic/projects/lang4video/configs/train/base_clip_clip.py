"""Base training config for CLIP visual + CLIP text."""

import ml_collections
from scenic.projects.lang4video.configs import base_clip_clip as general_base_clip_clip
from scenic.projects.lang4video.configs.train import mixin


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = general_base_clip_clip.get_config(run_local)
  config.update(mixin.get_config(run_local))
  return config
