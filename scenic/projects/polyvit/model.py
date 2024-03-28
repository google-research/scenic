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

"""PolyVit."""

from typing import Any, Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.projects.polyvit import layers
from scenic.projects.polyvit import model_utils
from scenic.projects.polyvit import polyvit_base_model


Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


class PolyVit(nn.Module):
  """PolyVit."""

  modalities_config: ml_collections.ConfigDict
  encoder_config: ml_collections.ConfigDict
  heads_config: ml_collections.ConfigDict
  stochastic_droplayer_config: Optional[ml_collections.ConfigDict]
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               targets: Optional[jnp.ndarray] = None,
               *,
               task: Optional[str] = None,
               modality: Optional[str] = 'image',
               dataset: Optional[str] = None,
               train: bool,
               debug: bool = False):

    if self.stochastic_droplayer_config is None:
      stochastic_droplayer_rate = None
    else:
      stochastic_droplayer_rate = self.stochastic_droplayer_config.get(dataset)

    x = layers.Tokenizer(
        config=self.modalities_config, dtype=self.dtype, name='tokenizer')(
            x,
            train=train,
            modality=modality,
            stochastic_droplayer_rate=stochastic_droplayer_rate,
            dataset=dataset)

    x = layers.PolyViTEncoder(
        mlp_dim=self.encoder_config.mlp_dim,
        num_tokenizer_layers=self.encoder_config.num_tokenizer_layers,
        num_layers=self.encoder_config.num_layers,
        num_heads=self.encoder_config.num_heads,
        dropout_rate=self.encoder_config.dropout_rate,
        attention_dropout_rate=self.encoder_config.attention_dropout_rate,
        dtype=self.dtype,
        name='vit_encoder')(
            x,
            train=train,
            stochastic_droplayer_rate=stochastic_droplayer_rate,
            dataset=dataset)

    if self.encoder_config.get('freeze_body', False):
      x = jax.lax.stop_gradient(x)

    elif task in [
        polyvit_base_model.Task.LABEL, polyvit_base_model.Task.MULTILABEL,
        polyvit_base_model.Task.MULTIHEADLABEL
    ]:

      head_config = self.heads_config.label.get(dataset)

      x = layers.ClassificationHead(
          num_outputs=head_config.num_classes,
          hid_sizes=head_config.hid_sizes,
          output_proj_zero_init=head_config.get('output_proj_zero_init', False),
          classifier=head_config.classifier,
          dtype=self.dtype,
          name='classification_head_' + dataset)(
              x, train=train)

      return x

    elif task == polyvit_base_model.Task.BOW:

      head_config = self.heads_config.bow.get(dataset)

      x = layers.ClassificationHead(
          num_outputs=head_config.vocab_size,
          hid_sizes=head_config.hid_sizes,
          output_proj_zero_init=head_config.get('output_proj_zero_init', False),
          classifier=head_config.classifier,
          dtype=self.dtype,
          name='bag_of_words_head_' + dataset)(
              x, train=train)

      return x

    elif task == polyvit_base_model.Task.FEWSHOT:

      x = layers.FewshotHead(self.heads_config.fewshot.pooling_type,
                             name='fewshot_head')(x, train=train)

      return x

    else:
      raise NotImplementedError(f'Task {task} is not supported yet.')


class PolyVitModel(polyvit_base_model.PolyVitBaseModel):
  """PolyVit model."""

  def build_flax_model(self):
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))

    add_cls_token, _ = model_utils.get_cls_token_and_video_frames(self.config)

    for dataset_name, meta_data in self.dataset_meta_data.items():
      if meta_data['task'] in [
          polyvit_base_model.Task.LABEL, polyvit_base_model.Task.MULTILABEL,
          polyvit_base_model.Task.MULTIHEADLABEL
      ]:
        with self.config.unlocked():
          self.config.model.heads.label.get(
              dataset_name).num_classes = meta_data['num_classes']
      if meta_data['task'] in [polyvit_base_model.Task.BOW]:
        with self.config.unlocked():
          self.config.model.heads.bow.get(
              dataset_name).vocab_size = meta_data['vocab_size']

    with self.config.unlocked():
      modalities_cfg = self.config.model.modalities
      encoder_cfg = self.config.model.encoder
      modalities_cfg.mlp_dim = encoder_cfg.mlp_dim
      if modalities_cfg.get('num_layers') is None:
        modalities_cfg.num_layers = 0
      modalities_cfg.num_heads = encoder_cfg.num_heads
      modalities_cfg.dropout_rate = encoder_cfg.dropout_rate
      modalities_cfg.attention_dropout_rate = encoder_cfg.attention_dropout_rate
      modalities_cfg.add_cls_token = add_cls_token
      encoder_cfg.num_tokenizer_layers = modalities_cfg.num_layers

    stochastic_droplayer_config = self.config.get('stochastic_droplayer_rates')

    return PolyVit(
        modalities_config=self.config.model.modalities,
        encoder_config=self.config.model.encoder,
        heads_config=self.config.model.heads,
        stochastic_droplayer_config=stochastic_droplayer_config,
        dtype=model_dtype,
    )

  def init_from_vit_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
    """Updates the train_state with data from restored_train_state (ViT)."""

    return model_utils.initialise_from_vit_train_state(self.config, train_state,
                                                       restored_train_state,
                                                       restored_model_cfg)

  def init_from_polyvit_train_state(
      self,
      train_state: Any,
      restored_train_state: Any,
      tokenizer_to_init_from: Optional[str] = None,
      tokenizer_to_init: Optional[str] = None,
      resolution_to_init: Optional[Any] = None,
      initialize_heads: bool = False) -> Any:
    """Updates the train_state with data from restored_train_state (PolyViT)."""

    return model_utils.initialize_from_polyvit_train_state(
        train_state,
        restored_train_state,
        tokenizer_to_init_from=tokenizer_to_init_from,
        tokenizer_to_init=tokenizer_to_init,
        resolution_to_init=resolution_to_init,
        initialize_heads=initialize_heads)

  def init_from_mbt_train_state(
      self,
      train_state: Any,
      restored_train_state: Any,
      tokenizer_to_init: str = 'tokenizer_spec',
      resolution_to_init: Optional[Any] = None,
      initialize_head: bool = False,
  ) -> Any:
    """Updates the train_state with data from restored_train_state (AViT)."""

    return model_utils.initialize_from_mbt_train_state(
        train_state,
        restored_train_state,
        tokenizer_to_init=tokenizer_to_init,
        resolution_to_init=resolution_to_init,
        initialize_head=initialize_head,
    )

  def init_from_vivit_train_state(self,
                                  train_state: Any,
                                  restored_train_state: Any,
                                  tokenizer_to_init: str = 'tokenizer3d',
                                  resolution_to_init: Optional[Any] = None,
                                  initialize_head: bool = False) -> Any:
    """Updates the train_state with data from restored_train_state (ViViT)."""

    return model_utils.initialize_from_vivit_train_state(
        train_state,
        restored_train_state,
        tokenizer_to_init=tokenizer_to_init,
        resolution_to_init=resolution_to_init,
        initialize_head=initialize_head)
