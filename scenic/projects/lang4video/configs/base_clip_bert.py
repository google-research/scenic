"""Base config for CLIP+BERT."""

import ml_collections
from scenic.projects.lang4video.configs import base_clip_clip

def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = base_clip_clip.get_config(run_local)

  config.experiment_name = 'clip_bert'

  config.model.text_encoder_name = 'bert'
  config.model.text_encoder = ml_collections.ConfigDict()
  config.model.text_encoder.config_name = 'base'

  config.dataset_configs.vocab_size = 30_522
  # TODO(sacastro): try cased. We'd need to change the DMVR BertTokenizer.
  config.dataset_configs.lowercase = True

  # The following configurations are meant to be used by DMVR datasets but also
  # as input to other parts of the config.

  config.dataset_configs.tokenizer.tokenizer_type = 'bert'

  # Add vocab path.
  config.dataset_configs.tokenizer.tokenizer_vocab = ''

  return config
