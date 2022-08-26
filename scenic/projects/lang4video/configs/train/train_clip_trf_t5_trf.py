r"""Training config for CLIP + T5 w Transformers.

"""  # pylint: disable=line-too-long

import ml_collections
from scenic.projects.lang4video.configs.train import train


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  return train.get_config(f'clip_trf_t5_trf+{run_local}')
