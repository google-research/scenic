"""Mixin config for RedCaps."""

import ml_collections
from scenic.projects.lang4video.configs.datasets import mixin_bit


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = mixin_bit.get_config(run_local)
  config.dataset_canonical_name = 'red_caps'
  config.dataset_configs.dataset = 'huggingface:red_caps/all'
  config.dataset_configs.train_split_name = 'train'
  config.dataset_configs.val_split_name = 'train'
  config.dataset_configs.text_in_key = 'raw_caption'
  # TODO(sacastro): fix that this dataset doesn't work, as the images are not
  #  provided but the URLs.
  return config
