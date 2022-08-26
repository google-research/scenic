r"""Evaluation config for CLIP.

"""  # pylint: disable=line-too-long

import ml_collections
import scenic.projects.lang4video.configs.zero_shot.eval as eval_


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  return eval_.get_config(f'clip+{run_local}')
