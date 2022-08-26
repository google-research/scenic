r"""Training config that supports specifying the model and dataset.

"""  # pylint: disable=line-too-long

import importlib

import ml_collections


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""

  args = run_local.split('+')
  if not args:
    args = ('clip', 'cc12m')
  elif len(args) == 1:
    args = (args[0] or 'clip', 'cc12m')
  assert len(args) in {2, 3}
  model_config_suffix, dataset_config_suffix = args[:2]
  dataset_config_suffix = dataset_config_suffix or 'cc12m'
  run_local = args[2] if len(args) == 3 else ''

  model_config_module = importlib.import_module(
      f'scenic.projects.lang4video.configs.train.base_{model_config_suffix}')
  config = model_config_module.get_config(run_local)

  dataset_config_module = importlib.import_module(
      f'scenic.projects.lang4video.configs.datasets.mixin_{dataset_config_suffix}'
  )
  config.update(dataset_config_module.get_config(run_local))

  config.experiment_name = f'{model_config_suffix}_on_{dataset_config_suffix}'

  return config
