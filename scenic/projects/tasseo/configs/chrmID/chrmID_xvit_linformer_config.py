# pylint: disable=line-too-long
r"""Linformer configs for X-ViT on chromosome identification task.

"""
# pylint: enable=line-too-long

from scenic.projects.tasseo.configs.chrmID import chrmID_xvit_config


def get_config():
  """Returns the X-ViT experiment configuration for metaphase sexID."""
  config = chrmID_xvit_config.get_config()
  config.experiment_name = 'chrmID-xvit-jf'
  config.model_name = 'xvit_classification'

  # Model.
  config.model.attention_fn = 'linformer'
  config.model.attention_configs.low_rank_features = 16

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
