"""Mixin config for DMVR datasets."""

import ml_collections
from scenic.projects.lang4video.configs.datasets import mixin


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = mixin.get_config(run_local)

  config.dataset_configs = ml_collections.ConfigDict()

  config.dataset_configs.load_train = False
  config.dataset_configs.load_test = False

  config.dataset_configs.val_on_test = False
  config.dataset_configs.test_on_val = False  # TODO(sacastro): unused.

  config.dataset_configs.keep_val_key = True
  config.dataset_configs.keep_test_key = True  # TODO(sacastro): unused.

  return config
