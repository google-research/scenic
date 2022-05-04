"""Utilities for loading a pretrained distance model.
"""

import jax
import ml_collections
from scenic.projects.func_dist import model as scenic_model
from scenic.projects.func_dist import train_utils
from scenic.train_lib_deprecated import pretrain_utils




def restore_model(config: ml_collections.ConfigDict, ckpt_path: str):
  """Restore model definition, weights and config from a checkpoint path."""
  rng = jax.random.PRNGKey(0)
  model_cls = scenic_model.get_model_cls(config.model_name)
  data_rng, rng = jax.random.split(rng)
  # Only used for metadata.
  dataset = train_utils.get_dataset(config, data_rng)
  train_state = pretrain_utils.restore_pretrained_checkpoint(ckpt_path)
  model = model_cls(config, dataset.meta_data)
  return model, train_state, config
