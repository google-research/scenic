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

"""ViViT model with object-guided training.
"""
from typing import Any, Optional

from absl import logging
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models import classification_model
from scenic.model_lib.layers import nn_layers
from scenic.projects.objectvivit import model_utils
from scenic.projects.vivit import model as vivit_model
from scenic.projects.vivit import model_utils as vivit_model_utils


class ObjectViViT(nn.Module):
  """ViViT model with object information."""

  mlp_dim: int
  num_layers: int
  num_heads: int
  num_classes: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  temporal_encoding_config: ml_collections.ConfigDict
  attention_config: ml_collections.ConfigDict
  dropout_rate: float = 0.
  attention_dropout_rate: float = 0.
  stochastic_droplayer_rate: float = 0.
  representation_size: Optional[int] = None
  classifier: str = 'gap'
  use_batch_norm_after_encoder: bool = True
  positional_embedding: str = 'sinusoidal_1d'
  normalise_encoder_output: bool = True
  use_approximate_gelu: bool = True
  dtype: jnp.dtype = jnp.float32
  object_config: ml_collections.ConfigDict = ml_collections.ConfigDict()
  detector_configs: ml_collections.ConfigDict = ml_collections.ConfigDict()
  attach_configs: ml_collections.ConfigDict = ml_collections.ConfigDict()

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, boxes: Optional[jnp.ndarray] = None,
      detections: Optional[jnp.ndarray] = None, *,
      train: bool = False, debug: bool = False):
    token_score_from_dataloader = self.attach_configs.get(
        'token_score_from_dataloader', False)
    data_has_detection = self.detector_configs.get('use_detector', False) or (
        self.attach_configs.get('enabled', False)
    ) or token_score_from_dataloader
    run_cross_frame_attention = self.attach_configs.get(
        'run_cross_frame_attention', False)
    random_object_baseline = self.attach_configs.get(
        'random_object_baseline', False)

    if data_has_detection:
      assert not random_object_baseline
      if token_score_from_dataloader:
        inputs, token_scores = inputs[..., :3], inputs[..., 3:]
        token_scores = model_utils.resize_token_score(
            token_scores, self.patches.size)  # batch x num_tokens x num_objs
        token_scores = token_scores.transpose(
            0, 2, 1)  #  batch x num_objs x num_tokens
      else:
        assert 0
    elif random_object_baseline:
      n_batch = inputs.shape[0]
      sp = inputs.shape
      sz = self.patches.size
      n_tokens = (sp[1] * sp[2] * sp[3]) // (sz[0] * sz[1] * sz[2])
      token_scores = jax.random.uniform(
          self.make_rng('dropout'), (n_batch, 1, n_tokens))
    else:
      token_scores = None

    # Shape is [batch, num_tokens, hidden_size]
    x_tokens, temporal_dims = vivit_model.temporal_encode(
        inputs, self.temporal_encoding_config, self.patches, self.hidden_size)

    n_batch, n_tokens, hidden_dim = x_tokens.shape
    height = width = int(np.sqrt(n_tokens // temporal_dims))
    if height * width * temporal_dims != n_tokens:
      raise ValueError('Input is assumed to be square.')
    num_tokens_per_frame = n_tokens // (inputs.shape[1] // self.patches.size[2])

    #  fg_inds is used to drop background tokens or attach more tokens.
    #  We need to process fg_inds (index of foreground tokens) outside of
    #    CustomEncoder to handle self.classifier == 'token'.
    fg_inds = None
    if self.attach_configs.get('enabled', False) or self.attach_configs.get(
        'drop_pixel_tokens', False):
      fg_inds = model_utils.get_object_inds(
          token_scores, num_tokens_per_frame, self.attach_configs
      )  # batch x num_attach_tokens

    # Add positional encodings.
    input_shape = None
    if self.positional_embedding not in ['learned_1d']:
      x_tokens = model_utils.add_positional_embeddings(
          x_tokens, self.positional_embedding, input_shape=input_shape)

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, hidden_dim),
                       x_tokens.dtype)
      cls = jnp.tile(cls, [n_batch, 1, 1])
      x_tokens = jnp.concatenate([cls, x_tokens], axis=1)
      if fg_inds is not None:
        cls_inds = jnp.zeros((n_batch, 1), dtype=fg_inds.dtype)
        fg_inds = jnp.concatenate([cls_inds, fg_inds + 1], axis=-1)

    x_tokens_encoded, _, aux = model_utils.CustomEncoder(
        temporal_dims=temporal_dims,
        hidden_size=self.hidden_size,
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        attention_config=self.attention_config,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_droplayer_rate=self.stochastic_droplayer_rate,
        dtype=self.dtype,
        positional_embedding='none' if self.positional_embedding not in [
            'learned_1d'] else self.positional_embedding,
        normalise_output=self.normalise_encoder_output,
        use_approximate_gelu=self.use_approximate_gelu,
        num_tokens_per_frame=num_tokens_per_frame,
        object_config=self.object_config,
        attach_configs=self.attach_configs,
        run_cross_frame_attention=run_cross_frame_attention,
        video_batch=n_batch,
        name='Transformer')(
            x_tokens, fg_inds=fg_inds,
            token_scores=token_scores, boxes=boxes,
            train=train, debug=debug)

    if self.classifier in ('token', '0'):
      x_tokens_encoded = x_tokens_encoded[:, 0]
    elif self.classifier in ('gap', 'gmp', 'gsp'):
      fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
      x_tokens_encoded = fn(x_tokens_encoded, axis=1)
      x_tokens_encoded = nn.LayerNorm(name='encoder_norm')(x_tokens_encoded)
    else:
      raise ValueError(f'Unknown classifier {self.classifier}')

    if self.representation_size is not None:
      x_tokens_encoded = nn.Dense(self.representation_size,
                                  name='pre_logits')(x_tokens_encoded)
      x_tokens_encoded = nn.tanh(x_tokens_encoded)
    else:
      x_tokens_encoded = nn_layers.IdentityLayer(name='pre_logits')(
          x_tokens_encoded)

    x_tokens_encoded = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(
            x_tokens_encoded)
    if debug:
      return x_tokens_encoded, aux
    return x_tokens_encoded


class ViViTModelWithObjects(classification_model.ClassificationModel):
  """Vision Video Transformer model for MAE finetuning."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    attention_type = self.config.model.attention_config.get(
        'type', 'spacetime')
    assert attention_type in ['spacetime']
    num_classes = self.dataset_meta_data['num_classes']

    return ObjectViViT(
        num_classes=num_classes,
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        temporal_encoding_config=self.config.model.temporal_encoding_config,
        attention_config=self.config.model.attention_config,
        representation_size=self.config.model.representation_size,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.0),
        attention_dropout_rate=self.config.model.get(
            'attention_dropout_rate', 0.0),
        stochastic_droplayer_rate=self.config.model.get(
            'stochastic_droplayer_rate', 0),
        dtype=model_dtype,
        normalise_encoder_output=self.config.model.get(
            'normalise_encoder_output',
            self.config.model.classifier == 'token'),
        use_approximate_gelu=self.config.model.get(
            'use_approximate_gelu', True),
        positional_embedding=self.config.model.get(
            'positional_embedding', 'sinusoidal_1d'),
        use_batch_norm_after_encoder=self.config.model.get(
            'use_batch_norm_after_encoder', True),
        object_config=self.config.model.get(
            'object_config', ml_collections.ConfigDict()),
        detector_configs=self.config.get(
            'detector_configs', ml_collections.ConfigDict()),
        attach_configs=self.config.get(
            'attach_configs', ml_collections.ConfigDict()),
    )

  def init_from_train_state(self,
                            train_state: Any,
                            restored_train_state: Any,
                            restored_model_cfg: ml_collections.ConfigDict,
                            restore_output_proj: bool = False) -> Any:
    """Updates the train_state with data from restored_train_state."""
    attention_type = self.config.model.attention_config.get(
        'type', 'spacetime')
    if attention_type in ['spacetime']:
      vivit_transformer_key = 'Transformer'
    else:
      raise ValueError(f'Attention type {attention_type} does not exist.')

    vivit_transformer_key = self.config.init_from.get(
        'vivit_transformer_key', vivit_transformer_key)

    logging.info('vivit_transformer_key: %s.', vivit_transformer_key)
    if vivit_transformer_key in restored_train_state.params and (
        'encoder_norm' in restored_train_state.params[vivit_transformer_key]):
      logging.info('fixing loading encoder_norm')
      restored_parameters = flax.core.unfreeze(restored_train_state.params)
      norm_parameters = restored_parameters[vivit_transformer_key].pop(
          'encoder_norm')
      restored_parameters['encoder_norm'] = norm_parameters
      restored_train_state = restored_train_state.replace(
          params=flax.core.freeze(restored_parameters))

    # If we restore from a non-MAE checkpoint and the positional embedding is
    # 'learned_1d', we have to move the positional embedding outside
    # the 'Transformer' block and also to drop the value of cls positional
    # embedding from the positonal embedding and add it to cls token.

    if (self.config.init_from.get('restore_from_non_mae_checkpoint', False)
        and self.config.model.get('positional_embedding',
                                  'sinusoidal_1d') == 'learned_1d'):
      # Move the positional embedding outside the 'Transformer' block.
      restored_parameters = flax.core.unfreeze(restored_train_state.params)
      restored_parameters['posembed_input'] = restored_parameters[
          vivit_transformer_key].pop('posembed_input')

      if restored_model_cfg.model.classifier == 'token':
        pos_embedding_params = restored_parameters[
            'posembed_input']['pos_embedding']
        # Drop the value of cls positional embedding.
        cls_pos_embedding = pos_embedding_params[:, 0]
        restored_parameters['posembed_input'][
            'pos_embedding'] = pos_embedding_params[:, 1:]
        # Add the value of cls positional embedding to cls token.
        if 'cls' in restored_parameters:
          restored_parameters['cls'] = restored_parameters[
              'cls'] + cls_pos_embedding

      restored_train_state = restored_train_state.replace(
          params=flax.core.freeze(restored_parameters))

    return vivit_model_utils.initialise_from_train_state(
        self.config,
        train_state,
        restored_train_state,
        restored_model_cfg,
        restore_output_proj,
        vivit_transformer_key=vivit_transformer_key)
