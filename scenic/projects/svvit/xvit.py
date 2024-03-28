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

"""X-ViT Classification model."""
from typing import Any
import ml_collections
from scenic.model_lib.base_models.classification_model import ClassificationModel
from scenic.projects.baselines import vit
from scenic.projects.fast_vit.xvit import XViT


class XViTClassificationModel(ClassificationModel):
  """X-ViT model for classification task."""

  def build_flax_model(self):
    return XViT(
        num_outputs=self.dataset_meta_data['num_classes'],
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        attention_configs=self.config.model.attention_configs,
        attention_fn=self.config.model.attention_fn,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        transformer_encoder_configs=self.config.model
        .transformer_encoder_configs,
        representation_size=self.config.model.representation_size,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.),
    )

  def default_flax_model_config(self):
    return ml_collections.ConfigDict(
        dict(
            model=dict(
                attention_fn='standard',
                attention_configs={'num_heads': 2},
                transformer_encoder_configs={'type': 'global'},
                num_layers=1,
                representation_size=16,
                mlp_dim=32,
                dropout_rate=0.,
                attention_dropout_rate=0.,
                hidden_size=16,
                patches={'size': (4, 4)},
                classifier='gap',
            ),
            data_dtype_str='float32'))

  def init_from_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
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
    return vit.init_vit_from_train_state(
        train_state,
        restored_train_state,
        self.config,
        restored_model_cfg,
    )
