"""Base training config for CLIP+BERT w/ MLPs."""

import ml_collections
from scenic.projects.lang4video.configs import base_clip_mlp_bert_mlp as general_base_clip_mlp_bert_mlp
from scenic.projects.lang4video.configs.train import mixin


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = general_base_clip_mlp_bert_mlp.get_config(run_local)
  config.update(mixin.get_config(run_local))

  config.optimizer_configs.params_to_freeze = (r'^\w+_encoder/encoders_0/',)

  return config
