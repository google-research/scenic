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

"""ViT Classification model."""
from typing import Optional

from flax.training import common_utils
import jax.numpy as jnp
import numpy as np
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils
from scenic.model_lib.base_models.classification_model import ClassificationModel
from scenic.projects.baselines import vit

from topological_transformer.images import topvit


class ViTClassificationModel(ClassificationModel):
  """ViT model for classification task."""

  def build_flax_model(self):
    return vit.ViT(
        num_classes=self.dataset_meta_data['num_classes'],
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        representation_size=self.config.model.representation_size,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.1),
        dtype='float32',
    )

  def init_from_train_state(self, train_state, restored_train_state,
                            restored_model_cfg):
    """Updates the train_state with data from restored_train_state.

    This function is written to be used for 'fine-tuning' experiments. Here, we
    do some surgery to support larger resolutions (longer sequence length) in
    the transformer block, with respect to the learned pos-embeddings.

    Args:
      train_state: A raw TrainState for the model.
      restored_train_state: A TrainState that is loaded with parameters/state of
        a  pretrained model.
      restored_model_cfg: Configuration of the model from which the
        restored_train_state come from. Usually used for some asserts.

    Returns:
      Updated train_state.
    """
    return vit.init_vit_from_train_state(train_state, restored_train_state,
                                         self.config, restored_model_cfg)


class TopologicalViTClassificationModel(ClassificationModel):
  """TopologicalViT model for classification task."""

  def build_flax_model(self):
    return topvit.TopologicalViT(
        num_classes=self.dataset_meta_data['num_classes'],
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        representation_size=self.config.model.representation_size,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.1),
        dtype='float32',
    )

  def init_from_train_state(self, train_state, restored_train_state,
                            restored_model_cfg):
    """Updates the train_state with data from restored_train_state.

    This function is written to be used for 'fine-tuning' experiments. Here, we
    do some surgery to support larger resolutions (longer sequence length) in
    the transformer block, with respect to the learned pos-embeddings.

    Args:
      train_state: A raw TrainState for the model.
      restored_train_state: A TrainState that is loaded with parameters/state of
        a  pretrained model.
      restored_model_cfg: Configuration of the model from which the
        restored_train_state come from. Usually used for some asserts.

    Returns:
      Updated train_state.
    """
    return topvit.init_topvit_from_train_state(train_state,
                                               restored_train_state,
                                               self.config, restored_model_cfg)

  def loss_function(self,
                    logits: jnp.ndarray,
                    batch: base_model.Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Returns softmax cross entropy loss with an L2 penalty on the weights.

    The function overwrites/modifies loss_function of ClassificationModel.
    Args:
      logits: Output of model in shape [batch, length, num_classes].
      batch: Batch of data that has 'label' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """
    weights = batch.get('batch_mask')

    if self.dataset_meta_data.get('target_is_onehot', False):
      one_hot_targets = batch['label']
    else:
      one_hot_targets = common_utils.onehot(batch['label'], logits.shape[-1])
    if self.config.get('class_balancing'):
      sof_ce_loss = model_utils.weighted_softmax_cross_entropy(
          logits,
          one_hot_targets,
          weights,
          label_weights=self.get_label_weights(),
          label_smoothing=self.config.get('label_smoothing'))
    else:
      sof_ce_loss = model_utils.weighted_softmax_cross_entropy(
          logits,
          one_hot_targets,
          weights,
          label_smoothing=self.config.get('label_smoothing'))
    if self.config.get('l2_decay_factor') is None:
      total_loss = sof_ce_loss
    else:
      l2_loss = model_utils.l2_regularization(model_params)
      total_loss = sof_ce_loss + 0.5 * self.config.l2_decay_factor * l2_loss
    return total_loss  # pytype: disable=bad-return-type  # jax-ndarray

  def get_label_weights(self) -> jnp.ndarray:
    """Returns labels' weights to be used for computing weighted loss.

    This can be used for weighting the loss terms based on the amount of
    available data for each class, when we have unbalanced data.
    """
    if not self.config.dataset_configs.get('num_normal'):
      raise ValueError(
          'When `class_balancing` is True, `num_normal` must'
          ' be provided.')
    if not self.config.dataset_configs.get('num_normal'):
      raise ValueError(
          'When `class_balancing` is True, `num_abnormal` must'
          ' be provided.')
    bincount = np.array([
        self.config.dataset_configs.get('num_normal'),
        self.config.dataset_configs.get('num_abnormal')
    ])
    n_samples = self.config.dataset_configs.get(
        'num_normal') + self.config.dataset_configs.get('num_abnormal')
    n_classes = 2  # For binary classification.
    return n_samples / (n_classes * bincount)
