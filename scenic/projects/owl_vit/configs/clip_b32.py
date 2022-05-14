# pylint: disable=line-too-long
"""CLIP OWL-ViT B/32.

See owl_vit/notebooks/OWL_ViT_minimal_example.ipynb for how to use this model.

Performance:
LVIS AP:  19.3 %
LVIS APr: 16.9 %
COCO AP:  28.1 %
"""

import ml_collections


def get_config():
  """Returns the configuration for text-query-based detection using ViT+."""
  config = ml_collections.ConfigDict()

  # Dataset.
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.input_size = 768
  config.dataset_configs.max_query_length = 16

  # Model.
  config.model_name = 'text_zero_shot_detection'

  config.model = ml_collections.ConfigDict()
  config.model.normalize = True

  config.model.body = ml_collections.ConfigDict()
  config.model.body.type = 'clip'
  config.model.body.variant = 'vit_b32'
  config.model.body.merge_class_token = 'mul-ln'
  config.model.box_bias = 'both'

  # Init.
  config.init_from = ml_collections.ConfigDict()
  config.init_from.checkpoint_path = 'gs://scenic-bucket/owl_vit/checkpoints/clip_vit_b32_b0203fc'

  return config
