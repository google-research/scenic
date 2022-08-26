"""Mixin config for CC12M."""

import ml_collections
from scenic.projects.lang4video.configs.datasets import mixin_bit


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = mixin_bit.get_config(run_local)
  config.dataset_name = 'cc12m'
  config.dataset_canonical_name = 'cc12m'
  return config
