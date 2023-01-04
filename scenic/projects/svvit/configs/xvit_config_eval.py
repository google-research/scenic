# pylint: disable=line-too-long
r"""Default configs for ViT on structural variant classification using pileups.

"""

import ml_collections


def get_config(runlocal=''):
  """Returns the ViT experiment configuration for SV classification."""
  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'sv-vit_offline_eval'
  config.trainer_name = 'inference'

  # Dataset.
  config.dataset_name = 'pileup_window'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.train_path = ''
  config.dataset_configs.test_path = ''
  config.dataset_configs.eval_path = ''

  # Model to be evaluated.
  config.model_name = 'xvit_classification'
  config.init_from = ml_collections.ConfigDict()
  config.init_from.xm = (None, None)
  config.batch_size = 8 if runlocal else 512
  config.rng_seed = 42
  config.save_predictions_on_cns = True

  return config


