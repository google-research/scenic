"""Mixin config for HowTo100M."""

import ml_collections
from scenic.projects.lang4video.configs.datasets import mixin_dmvr


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = mixin_dmvr.get_config(run_local)
  config.dataset_name = 'howto100m'
  config.dataset_canonical_name = 'howto100m'
  config.dataset_configs.test_on_val = True  # It doesn't have a val split.
  return config
