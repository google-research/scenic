# pylint: disable=line-too-long
r"""Default configs for BERT finetuning on GLUE.

"""
# pylint: enable=line-too-long

import ml_collections

_GLUE_TASKS = [
    'stsb', 'cola', 'sst2', 'mrpc', 'qqp', 'mnli_matched', 'mnli_mismatched',
    'rte', 'wnli', 'qnli'
]

VARIANT = 'BERT-B'

INIT_FROM = ml_collections.ConfigDict({
  'checkpoint_path': '',
  'model_config': 'SET-MODEL-CONFIG',
})


def get_config():
  """Returns configuration for BERT."""
  config = ml_collections.ConfigDict()
  config.rng_seed = 42
  config.glue_task = ''
  config.variant = VARIANT
  config.init_from = INIT_FROM
  return config


