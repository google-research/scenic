r"""Training config for CLIP+T5 w/ MLPs.

"""  # pylint: disable=line-too-long

import ml_collections
from scenic.projects.lang4video.configs.train import train


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  return train.get_config(f'clip_mlp_t5_mlp+{run_local}')
