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

"""Base class for layout denoise model."""

from typing import Any, Dict, Optional, Tuple

from absl import logging
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils

ArrayDict = Dict[str, jnp.ndarray]
MetricsDict = Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]


class LayoutDenoiseBaseModel(base_model.BaseModel):
  """Base model for object detection with matching."""

  def __init__(self, config: ml_collections.ConfigDict,
               dataset_meta_data: Dict[str, Dict[str, Any]]):
    """Initialize Detection model.

    Args:
      config: Configurations of the model.
      dataset_meta_data: Dataset meta data specifies `target_is_onehot`, which
        is False by default, and a required `num_classes`, which is the number
        of object classes including background/unlabeled/padding. The padded
        objects have label 0. The first legitimate object has label 1, and so
        on.
    """
    del dataset_meta_data
    if config is not None:
      self.loss_terms_weights = {
          'loss_ce': config.class_loss_coef,
      }

    if config is None:
      logging.warning('You are creating the model with default config.')
      config = self.default_flax_model_config()
    self.config = config
    self.target_is_onehot = False
    self.flax_model = self.build_flax_model()

  def label_losses_and_metrics(self,
                               outputs: ArrayDict,
                               batch: ArrayDict,
                               log: bool = True) -> Tuple[Any, MetricsDict]:
    """Classification softmax cross entropy loss.

    Args:
      outputs: Model predictions. For the purpose of this loss, outputs must
        have key 'pred_logits'. outputs['pred_logits'] is a nd-array of the
        predicted pre-softmax logits of shape [batch-size, num-objects,
        num-classes].
      batch: Dict that has 'inputs', 'batch_mask' and, 'label' (ground truth).
        batch['label'] is a dict. For the purpose of this loss, label dict must
        have key 'labels', which the value is an int nd-array of labels. It may
        be one-hot if dataset_meta_data.target_is_onehot was set to True. If
        batch['batch_mask'] is provided it is used to weight the loss for
        different images in the current batch of examples.
      log: If true, return class_error as well.

    Returns:
      loss: Dict with keys 'loss_ce'.
      metrics: Dict with keys 'loss_ce' and 'class_error`.
    """
    assert 'pred_logits' in outputs
    assert 'label' in batch

    metrics = {}
    batch_weights = batch.get('batch_mask')
    targets = batch['label']['labels']

    # Shape: [batch, 101, 101]
    src_logits = jax.nn.log_softmax(outputs['pred_logits'])

    tgt_labels_weights = targets != 0
    tgt_labels_onehot = jax.nn.one_hot(targets,
                                       self.config.get('num_classes', 25))
    tgt_labels_onehot *= tgt_labels_weights[..., None]

    # For predicted bbox that don't have a matched parent id, the weight is 0.
    # Shape: [batch, num_matched]
    weights = tgt_labels_weights
    if batch_weights is not None:
      weights *= batch_weights[..., None]

    label_weights = [0.0] + [1.0] * (self.config.get('num_classes') - 1)
    label_weights = jnp.array(label_weights)

    unnormalized_loss_ce = model_utils.weighted_unnormalized_softmax_cross_entropy(
        src_logits,
        tgt_labels_onehot,
        weights=weights,
        logits_normalized=True,
        label_smoothing=self.config.label_smoothing)

    denom = tgt_labels_onehot.sum(axis=[1, 2])
    if batch_weights is not None:
      denom *= batch_weights

    norm_type = self.config.get('normalization', 'detr')
    if norm_type == 'detr':
      denom = denom.sum()
      normalized_loss_ce = unnormalized_loss_ce.sum() / jnp.maximum(denom, 1.)
    elif norm_type == 'global':
      denom = jax.lax.pmean(denom.sum(), axis_name='batch')
      normalized_loss_ce = unnormalized_loss_ce.sum() / jnp.maximum(denom, 1.)
    elif norm_type == 'per_example':
      normalized_loss_ce = unnormalized_loss_ce.sum(axis=1)
      normalized_loss_ce = (normalized_loss_ce / jnp.maximum(denom, 1.)).mean()
    else:
      raise ValueError(f'Unknown normalization {norm_type}.')

    metrics['loss_ce'] = (unnormalized_loss_ce.sum(), denom.sum())

    if log:
      # For normalization, we need to have number of inputs that we do
      # prediction for, which is number of predicted boxes that have a matched
      # parent id.
      batch_num_inputs = weights.sum()
      # We are not using the eos_coef for accuracy computation
      num_correct = model_utils.weighted_correctly_classified(
          src_logits, tgt_labels_onehot, weights=weights)
      # For the evaluation, we will globally (across replicas) normalize the
      # num_correct to get the accuracy. The caller will do the normalization,
      # so we dont normalize and simply collect the sums here.
      metrics['label_accuracy'] = (num_correct, batch_num_inputs)

    # Sum metrics and normalizers over all replicas.
    for k, v in metrics.items():
      metrics[k] = model_utils.psum_metric_normalizer(v)
    return normalized_loss_ce, metrics

  def loss_function(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self,
      outputs: ArrayDict,
      batch: ArrayDict,
      task: str,
      model_params: Optional[jnp.ndarray] = None
  ) -> Tuple[jnp.ndarray, MetricsDict]:
    """Loss and metrics function."""
    if task == 'layout_denoise':
      total_loss, metrics = self.label_losses_and_metrics(
          outputs=outputs, batch=batch)

      aux_outputs = outputs.get('aux_outputs', [])
      for i, aux_outputs in enumerate(aux_outputs):
        aux_loss_ce, aux_metrics = self.label_losses_and_metrics(
            outputs=aux_outputs, batch=batch)
        # add metrics for aux outputs
        metrics.update({k + f'_aux_{i}': v for k, v in aux_metrics.items()})
        total_loss += aux_loss_ce

      if self.config.get('l2_decay_factor') is not None:
        l2_loss = model_utils.l2_regularization(model_params)
        metrics['l2_loss'] = (l2_loss, 1)
        total_loss = total_loss + 0.5 * self.config.l2_decay_factor * l2_loss

      # Process metrics dictionary to generate final unnormalized metrics
      metrics['minibatch_object_detection_loss'] = (total_loss, 1)
      return total_loss, metrics  # pytype: disable=bad-return-type  # jax-ndarray
    else:
      raise ValueError('Unsupported task %s' % task)

  def build_flax_model(self):
    raise NotImplementedError('Subclasses must implement build_flax_module().')

  def default_flax_model_config(self):
    """Default config for the flax model that is built in `build_flax_model`.

    This function in particular serves the testing functions and supposed to
    provide config tha are passed to the flax_model when it's build in
    `build_flax_model` function, e.g., `model_dtype_str`.
    """
    raise NotImplementedError(
        'Subclasses must implement default_flax_model_config().')
