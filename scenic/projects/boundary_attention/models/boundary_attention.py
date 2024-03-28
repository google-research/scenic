# Copyright 2024 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Define Boundary Attention Model."""

import functools
from typing import Any, Dict, Optional

import flax
import flax.linen as nn
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models import base_model as scenic_base_model
from scenic.projects.boundary_attention.helpers import params2maps
from scenic.projects.boundary_attention.loss_lib import boundary_attention_loss
from scenic.projects.boundary_attention.loss_lib import metrics_dict
from scenic.projects.boundary_attention.models.model_lib import boundary_attention_model_base


_FASTFOJ_METRICS = metrics_dict.FASTFOJ_METRICS


class BoundaryAttention(scenic_base_model.BaseModel):
  """Boundary Attention model."""

  def __init__(self, config: ml_collections.ConfigDict,
               dataset_metadata: Dict[str, Dict[str, Any]]) -> None:
    self.config = config
    self.dataset_metadata = dataset_metadata
    self.params2maps = params2maps.Params2Maps(config.model.opts,
                                               config.model.input_opts)
    self.flax_model = self.build_flax_model()
    self.loss_fn = boundary_attention_loss.BoundaryAttentionLoss(
        config, params2maps=self.params2maps)

  def loss_function(self, model_outputs: Dict[str, jnp.ndarray],
                    batch: jnp.ndarray) -> Any:

    return self.loss_fn.get_loss(model_outputs, batch)

  def get_metrics_fn(self, split: Optional[str] = None) -> Any:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      batch)```
    """
    if split == 'test':
      return metrics_dict.metric_function_noop
    else:
      return functools.partial(
          metrics_dict.metric_function,
          metrics_dict=_FASTFOJ_METRICS,
          loss_fn=self.loss_fn)

  def build_flax_model(self) -> nn.Module:
    return boundary_attention_model_base.BoundaryAttentionModelBase(
        config=self.config, params2maps=self.params2maps)

  def init_from_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
    """Updates the train_state with data from `restored_train_state`."""
    del restored_model_cfg
    params = flax.core.unfreeze(train_state.params)
    restored_params = flax.core.unfreeze(restored_train_state.params)
    for pname, pvalue in restored_params.items():
      params[pname] = pvalue
    return train_state.replace(
        params=flax.core.freeze(params),
        model_state=restored_train_state.model_state)
