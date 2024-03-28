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

"""Base model with NCR regularisation losses."""

import functools
from typing import Dict, Optional, Tuple, Union

from flax.training import common_utils
from immutabledict import immutabledict
import jax
import jax.numpy as jnp
import numpy as np
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import classification_model
from scenic.model_lib.base_models import model_utils
from scenic.projects.ncr import loss

Array = Union[jnp.ndarray, np.ndarray]

_CLASSIFICATION_METRICS = immutabledict({
    'accuracy':
        (model_utils.weighted_correctly_classified, model_utils.num_examples),
    'loss_xentropy': (model_utils.weighted_unnormalized_softmax_cross_entropy,
                      model_utils.num_examples),
})


class NCRModel(base_model.BaseModel):
  """Abstract class for model with NCR losses.

  Supports both softmax-classification and multi-label classification models.
  """

  def loss_function(  # pytype: disable=signature-mismatch  # overriding-return-type-checks
      self,
      logits: Array,
      batch: base_model.Batch,
      use_ncr: bool = False,
      use_bootstrap: bool = False,
      features: Optional[Array] = None,
      memory_logits: Optional[Array] = None,
      memory_features: Optional[Array] = None,
      loss_weight: Optional[float] = 0.0,
      model_params: Optional[Array] = None) -> Tuple[float, Dict[str, Array]]:

    if use_ncr:
      return self.ncr_loss(logits, batch, features, memory_logits,
                           memory_features, loss_weight, model_params)
    else:
      return self.ce_loss(logits, batch, model_params, use_bootstrap,
                          loss_weight)

  def ce_loss(
      self,
      logits: Array,
      batch: base_model.Batch,
      model_params: Optional[Array] = None,
      use_bootstrap: bool = False,
      loss_weight: Optional[float] = 1.0) -> Tuple[float, Dict[str, Array]]:
    """Returns softmax cross entropy loss with an L2 penalty on the weights.

    Args:
      logits: Output of model in shape [batch, length, num_classes].
      batch: Batch of data that has 'label' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.
      use_bootstrap: Enable the bootstrap loss term
      loss_weight: Weight for the bootstrap loss term

    Returns:
      Total loss.
    """
    weights = batch.get('batch_mask')
    loss_metrics = {}

    if self.dataset_meta_data.get('target_is_onehot', False):
      one_hot_targets = batch['label']
    else:
      one_hot_targets = common_utils.onehot(batch['label'], logits.shape[-1])

    softmax_ce_loss = model_utils.weighted_softmax_cross_entropy(
        logits,
        one_hot_targets,
        weights,
        label_smoothing=self.config.get('label_smoothing'))
    loss_metrics['softmax_cross_entropy'] = softmax_ce_loss

    if self.config.get('l2_decay_factor') is None:
      total_loss = softmax_ce_loss
    else:
      l2_loss = model_utils.l2_regularization(model_params)
      total_loss = softmax_ce_loss + 0.5 * self.config.l2_decay_factor * l2_loss

    if use_bootstrap:
      bootstrap_labels = jax.nn.softmax(logits)
      bootstrap_loss = model_utils.weighted_softmax_cross_entropy(
          logits,
          bootstrap_labels,
          weights,
          label_smoothing=self.config.get('label_smoothing'))
      total_loss = (1.0 - loss_weight) * total_loss + (
          loss_weight * bootstrap_loss)

    # Add the dummy entry for the NCR loss
    loss_metrics['ncr_loss'] = 0.0
    loss_metrics['total_loss'] = total_loss

    return total_loss, loss_metrics  # pytype: disable=bad-return-type  # jax-ndarray

  def ncr_loss(
      self,
      logits: Array,
      batch: base_model.Batch,
      features: Array,
      batch_logits: Array,
      batch_features: Array,
      ncr_loss_weight: float,
      model_params: Optional[Array] = None) -> Tuple[float, Dict[str, Array]]:
    """Returns softmax cross entropy loss with an L2 penalty on the weights.

    Args:
      logits: Output of model in shape [batch, length, num_classes].
      batch: Batch of data that has 'label' and optionally 'batch_mask'.
      features: Feature embeddings of batch inputs,
      batch_logits: Logits corresponding to the batch items to be queried from
      batch_features: Features corresponding to the batch items to be queried
      from
      ncr_loss_weight: The weight of the NCR loss term
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """
    weights = batch.get('batch_mask')
    loss_metrics = {}

    if self.dataset_meta_data.get('target_is_onehot', False):
      one_hot_targets = batch['label']
    else:
      one_hot_targets = common_utils.onehot(batch['label'], logits.shape[-1])

    softmax_ce_loss = model_utils.weighted_softmax_cross_entropy(
        logits,
        one_hot_targets,
        weights,
        label_smoothing=self.config.get('label_smoothing'))

    softmax_ce_loss = (1.0 - ncr_loss_weight) * softmax_ce_loss
    loss_metrics['softmax_cross_entropy'] = softmax_ce_loss
    if self.config.get('l2_decay_factor') is None:
      total_loss = softmax_ce_loss
    else:
      l2_loss = model_utils.l2_regularization(model_params)
      total_loss = softmax_ce_loss + 0.5 * self.config.l2_decay_factor * l2_loss

    # Add NCR loss
    ncr_loss = loss.ncr_loss(
        logits, features, batch_logits, batch_features,
        number_neighbours=self.config.ncr.number_neighbours,
        smoothing_gamma=self.config.ncr.smoothing_gamma,
        temperature=self.config.ncr.temperature,
        example_weights=weights)
    total_loss += ncr_loss_weight * ncr_loss
    loss_metrics['ncr_loss'] = ncr_loss
    loss_metrics['total_loss'] = total_loss

    return total_loss, loss_metrics  # pytype: disable=bad-return-type  # jax-ndarray

  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      batch)```
    """
    del split  # For all splits, we return the same metric functions.

    return functools.partial(
        classification_model.classification_metrics_function,
        target_is_onehot=self.dataset_meta_data.get('target_is_onehot',
                                                    False),
        metrics=_CLASSIFICATION_METRICS)
