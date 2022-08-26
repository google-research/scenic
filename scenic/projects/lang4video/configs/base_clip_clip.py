"""Base config for CLIP visual + CLIP text."""

import ml_collections
from scenic.projects.lang4video.configs import base_clip


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = base_clip.get_config(run_local)

  config.experiment_name = 'clip_clip'

  del config.model.encoder_name
  del config.model.encoder.config_name
  del config.model.encoder

  config.model.image_encoder_name = 'clip'
  config.model.image_encoder = ml_collections.ConfigDict()
  config.model.image_encoder.config_name = 'vit_b32'

  config.model.text_encoder_name = 'clip'
  config.model.text_encoder = ml_collections.ConfigDict()
  config.model.text_encoder.config_name = 'vit_b32'

  # The following configurations are meant to be used by DMVR datasets but also
  # as input to other parts of the config.

  config.dataset_configs.min_resize = base_clip.get_clip_image_size(
      config.model.image_encoder.config_name)
  config.dataset_configs.crop_size = base_clip.get_clip_image_size(
      config.model.image_encoder.config_name)

  return config
