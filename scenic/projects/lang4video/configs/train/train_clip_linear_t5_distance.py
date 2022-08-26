r"""Training config for CLIP w/ linear + T5 with a distance loss.

"""  # pylint: disable=line-too-long

import ml_collections
from scenic.projects.lang4video.configs.train import train_clip_linear_t5


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = train_clip_linear_t5.get_config(run_local)
  config.model.loss = 'distance'
  config.model.gather_scores = False
  config.temperature = 1
  return config

