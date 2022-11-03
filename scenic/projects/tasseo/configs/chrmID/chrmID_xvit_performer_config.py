# pylint: disable=line-too-long
r"""Performer configs for X-ViT on chromosome identification task.

"""
# pylint: enable=line-too-long

from scenic.projects.tasseo.configs.chrmID import chrmID_xvit_config


def get_config():
  """Returns the X-ViT experiment configuration for metaphase sexID."""
  config = chrmID_xvit_config.get_config()
  config.experiment_name = 'chrmID-performer-xvit'
  config.model_name = 'xvit_classification'

  # Model.
  config.model.attention_fn = 'performer'
  config.model.attention_configs.attention_fn_cls = 'generalized'
  config.model.attention_configs.attention_fn_configs = None

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
