"""Mixin config for datasets."""

import ml_collections


def get_config(run_local: str = '') -> ml_collections.ConfigDict:  # pylint: disable=unused-argument
  """Returns the experiment configuration."""
  config = ml_collections.ConfigDict()
  # Used when we need to show the name of the dataset:
  config.dataset_canonical_name = ''
  return config
