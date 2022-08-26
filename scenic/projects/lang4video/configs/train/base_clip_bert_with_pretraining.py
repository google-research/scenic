"""Base training config for CLIP+BERT w/ pretraining."""

import ml_collections
from scenic.projects.lang4video.configs import base_clip_bert_with_pretraining as general_base_clip_bert_with_pretraining
from scenic.projects.lang4video.configs.train import mixin


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = general_base_clip_bert_with_pretraining.get_config(run_local)
  config.update(mixin.get_config(run_local))
  config.trainer_name = 'visual_text_with_text_pretraining_trainer'
  config.mlm_loss_weight = 1.0
  return config
