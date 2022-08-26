"""Base config for CLIP w/ linear + BERT."""

import ml_collections
from scenic.projects.lang4video.configs import base_clip
from scenic.projects.lang4video.configs import base_clip_bert


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = base_clip_bert.get_config(run_local)

  config.experiment_name = 'clip_linear_bert'

  del config.model.image_encoder.config_name

  config.model.image_encoder_name = 'sequential'
  config.model.image_encoder.encoders = ml_collections.ConfigDict()
  config.model.image_encoder.encoders.a = ml_collections.ConfigDict()
  config.model.image_encoder.encoders.a.encoder_name = 'clip'
  config.model.image_encoder.encoders.a.encoder = ml_collections.ConfigDict()
  config.model.image_encoder.encoders.a.encoder.config_name = 'vit_b32'
  config.model.image_encoder.encoders.a.encoder.normalize = False
  config.model.image_encoder.encoders.b = ml_collections.ConfigDict()
  config.model.image_encoder.encoders.b.encoder_name = 'mlp'
  config.model.image_encoder.encoders.b.encoder = ml_collections.ConfigDict()
  config.model.image_encoder.encoders.b.encoder.num_layers = 0
  config.model.image_encoder.encoders.b.encoder.embedding_size = 768
  config.model.image_encoder.encoders.b.encoder.skip_connection = False

  # The following configurations are meant to be used by DMVR datasets but also
  # as input to other parts of the config.

  config.dataset_configs.min_resize = base_clip.get_clip_image_size(
      config.model.image_encoder.encoders.a.encoder.config_name)
  config.dataset_configs.crop_size = base_clip.get_clip_image_size(
      config.model.image_encoder.encoders.a.encoder.config_name)

  return config
