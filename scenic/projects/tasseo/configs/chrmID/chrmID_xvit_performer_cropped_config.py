# pylint: disable=line-too-long
r"""Performer configs for X-ViT on chromosome identification task.

"""
# pylint: enable=line-too-long

import ml_collections
from scenic.projects.tasseo.configs.chrmID import chrmID_xvit_config


def get_config():
  """Returns the X-ViT experiment configuration for metaphase sexID."""
  config = chrmID_xvit_config.get_config()
  config.experiment_name = 'chrmID-performer-xvit'
  config.model_name = 'xvit_classification'

  # Dataset.
  config.dataset_name = 'chrmID_baseline'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  # For available cropped shapes, see chrmID_baseline_dataset:DATASET_BASE_DIRS.
  config.dataset_configs.chrm_image_shape = (199, 99)

  # Model.
  config.model.attention_fn = 'performer'
  config.model.attention_configs.attention_fn_cls = 'generalized'
  config.model.attention_configs.attention_fn_configs = None

  return config


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([])
