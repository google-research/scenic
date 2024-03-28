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

"""ViViT model for MAE pretraining."""

from typing import Any, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from scenic.model_lib.base_models import classification_model
from scenic.model_lib.layers import nn_layers

from scenic.projects.av_mae import base_model
from scenic.projects.av_mae import model_utils
from scenic.projects.vivit import model as vivit_model
from scenic.projects.vivit import model_utils as vivit_model_utils


class ViViT(nn.Module):
  """Vision Video Transformer model baseline for transfer learning.

    The differences to the scenic.project.vivit.model.ViViT are that:
    -- Posibility of freezing the backbone.
    -- The positional embedding is added before the encoder.
    -- The CLS token is added after the positional embeddings. This follows MAE.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_classes: Number of output classes.
    num_heads: Number of self-attention heads.
    num_layers: Number of layers.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    temporal_encoding_config: ConfigDict which defines the type of input
      encoding when tokenising the video.
    attention_config: ConfigDict which defines the type of spatio-temporal
      attention applied in the model.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_droplayer_rate: Probability of dropping a layer. Linearly
      increases from 0 to the provided value.
    representation_size: Size of the representation layer in the model's head.
      if None, we skip the extra projection + tanh activation at the end.
    classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
      'token'.
    freeze_backbone: If True, add a stop-gradient before the final
      classifier to only evaluate linear evaluation performance.
    use_batch_norm_after_encoder: Only applies when the backbone is frozen.
      In this case, an additional batch normalisation layer is applied before
      the linear classifier. This was done in MAE
      (https://arxiv.org/abs/2111.06377).
    positional_embedding: The type of positional embeddings to add to the
      tokens at the beginning of the transformer encoder. Options are
      {learned_1d, sinusoidal_3d, none}.
    normalise_encoder_output: If true, layer normalisation is applied to the
      output of the transformer encoder. This is typically not done when
      not using the token classifier.
    dtype: JAX data type for activations.
  """

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
  freeze_backbone: bool = False
  use_batch_norm_after_encoder: bool = True
  positional_embedding: str = 'sinusoidal_1d'
  normalise_encoder_output: bool = True
  dtype: jnp.dtype = jnp.float32
  modality: str = base_model.FeatureTargets.RGB
  use_modality_tokens: bool = False

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, train: bool, debug: bool = False):

    del debug

    # Shape is [batch, num_tokens, hidden_size]
    if self.modality == base_model.FeatureTargets.RGB:
      x_tokens, temporal_dims = vivit_model.temporal_encode(
          inputs, self.temporal_encoding_config, self.patches, self.hidden_size)
    elif self.modality == base_model.FeatureTargets.SPECTROGRAM:
      x_tokens = model_utils.embed_2d_patch(
          inputs, self.patches, self.hidden_size)
      temporal_dims = None
    else:
      raise ValueError(f'Unknown modality {self.modality}')

    n_batch, n_tokens, hidden_dim = x_tokens.shape
    if self.modality == base_model.FeatureTargets.RGB:
      height = width = int(np.sqrt(n_tokens // temporal_dims))
      if height * width * temporal_dims != n_tokens:
        raise ValueError('Input is assumed to be square.')

    if (self.modality == base_model.FeatureTargets.SPECTROGRAM
        and self.positional_embedding not in ['learned_1d', 'sinusoidal_1d']):
      raise ValueError(
          'Only 1d positional embdeddings are supported for spectograms.')

    # Add positional encodings.
    input_shape = None
    if self.positional_embedding == 'sinusoidal_3d':
      input_shape = [n_batch, temporal_dims, height, width, hidden_dim]
    elif self.positional_embedding == 'learned_space_time':
      input_shape = [n_batch, temporal_dims, height * width, hidden_dim]

    x_tokens = model_utils.add_positional_embeddings(
        x_tokens, self.positional_embedding, input_shape=input_shape)

    x_tokens = self.add_modality_token(x_tokens)

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, hidden_dim),
                       x_tokens.dtype)
      cls = jnp.tile(cls, [n_batch, 1, 1])
      x_tokens = jnp.concatenate([cls, x_tokens], axis=1)

    x_tokens_encoded = vivit_model.Encoder(
        temporal_dims=temporal_dims,
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        attention_config=self.attention_config,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_droplayer_rate=self.stochastic_droplayer_rate,
        dtype=self.dtype,
        positional_embedding='none',  # Has already been added.
        normalise_output=self.normalise_encoder_output,
        name='Transformer')(x_tokens, train=train)

    if self.classifier in ('token', '0'):
      x_tokens_encoded = x_tokens_encoded[:, 0]
    elif self.classifier in ('gap', 'gmp', 'gsp'):
      fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
      x_tokens_encoded = fn(x_tokens_encoded, axis=1)
      x_tokens_encoded = nn.LayerNorm(name='encoder_norm')(x_tokens_encoded)
    elif self.classifier in ('none'):
      return x_tokens_encoded
    else:
      raise ValueError(f'Unknown classifier {self.classifier}')

    if self.representation_size is not None:
      x_tokens_encoded = nn.Dense(self.representation_size,
                                  name='pre_logits')(x_tokens_encoded)
      x_tokens_encoded = nn.tanh(x_tokens_encoded)
    else:
      x_tokens_encoded = nn_layers.IdentityLayer(name='pre_logits')(
          x_tokens_encoded)

    if self.freeze_backbone:
      x_tokens_encoded = jax.lax.stop_gradient(x_tokens_encoded)

      if self.use_batch_norm_after_encoder:
        x_tokens_encoded = nn.BatchNorm(
            # Match PyTorch default and MAE PyTorch code
            # https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_linprobe.py#L222
            momentum=0.9,
            epsilon=1e-6,
            use_bias=False,
            use_scale=False,
            )(x_tokens_encoded, use_running_average=not train)

    x_tokens_encoded = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(
            x_tokens_encoded)
    return x_tokens_encoded

  def add_modality_token(self, x_tokens: jnp.ndarray, name: str = 'Encoder'
                         ) -> jnp.ndarray:
    """Add modality learned tokens."""
    if not self.use_modality_tokens:
      return x_tokens

    modality_token = self.param(f'{name}_modality_token_{self.modality}',
                                nn.initializers.zeros,
                                (1, 1, x_tokens.shape[-1]))
    x_tokens = x_tokens + modality_token
    return x_tokens


class ViViTMaskedAutoencoder(nn.Module):
  """Vision Video Transformer model for masked-autoencoding.

    The differences to the scenic.project.vivit.model.ViViT are that:
    -- Remove masked tokens from the encoder.
    -- Process all tokens with the decoder.
    -- The CLS token is added after the positional embeddings. This follows MAE.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_classes: Number of output classes.
    num_heads: Number of self-attention heads.
    num_layers: Number of layers.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    token_mask_probability: Probability of dropping out the input tokens
    during training.
    masking_strategy: Masking strategy used to mask the tokens.
    temporal_encoding_config: ConfigDict which defines the type of input
      encoding when tokenising the video.
    attention_config: ConfigDict which defines the type of spatio-temporal
      attention applied in the model.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_droplayer_rate: Probability of dropping a layer. Linearly
      increases from 0 to the provided value..
    classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
      'token'.
    normalise_encoder_output: If true, layer normalisation is applied to the
      output of the transformer encoder.
    dtype: JAX data type for activations.
  """

  mlp_dim: int
  num_layers: int
  num_heads: int
  num_classes: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  token_mask_probability: float
  masking_strategy: str
  temporal_encoding_config: ml_collections.ConfigDict
  decoder_config: ml_collections.ConfigDict
  attention_config: ml_collections.ConfigDict
  dropout_rate: float = 0.
  attention_dropout_rate: float = 0.
  stochastic_droplayer_rate: float = 0.
  classifier: str = 'gap'
  positional_embedding: str = 'sinusoidal_1d'
  positional_embedding_decoder: str = 'sinusoidal_1d'
  normalise_encoder_output: bool = True
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, train: bool, debug: bool = False):

    del debug
    # Shape is [batch, num_tokens, hidden_size].
    x_tokens, temporal_dims = vivit_model.temporal_encode(
        inputs, self.temporal_encoding_config, self.patches, self.hidden_size)

    n_batch, n_tokens, hidden_dim = x_tokens.shape
    height = width = int(np.sqrt(n_tokens // temporal_dims))
    if height * width * temporal_dims != n_tokens:
      raise ValueError('Input is assumed to be square.')

    # Add positional encodings.
    input_shape = None
    if self.positional_embedding == 'sinusoidal_3d':
      input_shape = [n_batch, temporal_dims, height, width, hidden_dim]
    elif self.positional_embedding == 'learned_space_time':
      input_shape = [n_batch, temporal_dims, height * width, hidden_dim]

    x_tokens = model_utils.add_positional_embeddings(
        x_tokens, self.positional_embedding, input_shape=input_shape
    )

    if train:
      if self.masking_strategy == 'random':
        # Generate mask indices by randomly masking the tokens.
        n_masked = int(self.token_mask_probability * n_tokens)
        mask_indices, unmasked_indices, token_mask = (
            model_utils.get_mask_indices(
                n_batch, n_tokens, n_masked, self.make_rng('dropout')
            )
        )

      elif self.masking_strategy == 'tube':
        # Generate mask indices by using tube masking.
        mask_indices, unmasked_indices, token_mask = (
            model_utils.get_tube_mask_indices(
                n_batch=n_batch,
                n_tokens=n_tokens,
                token_mask_probability=self.token_mask_probability,
                temporal_dims=temporal_dims,
                rng=self.make_rng('dropout'),
            )
        )
      else:
        raise ValueError(
            f'The masking strategy {self.masking_strategy} is not implemented.'
        )
      # Process only unmasked tokens with the encoder.
      batch_indices = jnp.arange(n_batch).reshape(n_batch, 1)
      x_unmasked = x_tokens[batch_indices, unmasked_indices]
    else:
      x_unmasked = x_tokens
      token_mask = jnp.zeros((n_batch, n_tokens))

    # If we want to add a class token, add it here.
    # Note that in MAE, positional encodings are not added to the CLS token.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, hidden_dim),
                       inputs.dtype)
      cls = jnp.tile(cls, [n_batch, 1, 1])
      x_unmasked = jnp.concatenate([cls, x_unmasked], axis=1)

    x_unmasked_encoded = vivit_model.Encoder(
        temporal_dims=temporal_dims,
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        attention_config=self.attention_config,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_droplayer_rate=self.stochastic_droplayer_rate,
        dtype=self.dtype,
        positional_embedding='none',  # Has already been added.
        normalise_output=self.normalise_encoder_output,
        name='Transformer')(x_unmasked, train=train)

    x_unmasked_encoded = nn_layers.IdentityLayer(name='encoder_output')(
        x_unmasked_encoded)
    if not train:
      x_representation = nn_layers.IdentityLayer(name='representation')(
          x_unmasked_encoded)
      return x_representation, token_mask

    # Process entire sequence with the decoder.
    mask_token = self.param('mask_token',
                            nn.initializers.zeros,
                            (1, 1, self.decoder_config.hidden_size))
    x_unmasked_proj = nn.Dense(
        self.decoder_config.hidden_size,
        use_bias=self.decoder_config.get('use_projection_bias', True),
        kernel_init=nn.initializers.xavier_uniform(),
        name='decoder_projection')(x_unmasked_encoded)
    if self.classifier == 'token':
      cls_encoded = x_unmasked_proj[:, :1, :]
      x_unmasked_proj = x_unmasked_proj[:, 1:, :]

    # This effectively "unshuffles" the tokens. This means that we can simply
    # add positional encodings in the decoder without having to worry about
    # their ordering.
    x_all = jnp.zeros((n_batch, n_tokens, self.decoder_config.hidden_size))
    x_all = x_all.at[batch_indices, unmasked_indices].set(x_unmasked_proj)
    x_all = x_all.at[batch_indices, mask_indices].set(mask_token)

    # Note. VideoMAE (Facebook) adds positional encodinggs to the CLS token at
    # the encoder as well. VideoMAE (Tong et al) don't use a CLS token.
    # This implementation does not add positional embeddings to the CLS token,
    # as in the original image MAE of He et al.
    if input_shape is not None:
      input_shape = input_shape[:-1] + [self.decoder_config.hidden_size]
    x_all = model_utils.add_positional_embeddings(
        x_all, self.positional_embedding_decoder,
        input_shape=input_shape,
        layer_name='posembed_decoder')

    if self.classifier == 'token':
      x_all = jnp.concatenate([cls_encoded, x_all], axis=1)

    x_decoded = vivit_model.Encoder(
        temporal_dims=temporal_dims,
        mlp_dim=self.decoder_config.mlp_dim,
        num_layers=self.decoder_config.num_layers,
        num_heads=self.decoder_config.num_heads,
        attention_config=self.decoder_config.attention_config,
        dropout_rate=self.decoder_config.dropout_rate,
        attention_dropout_rate=self.decoder_config.attention_dropout_rate,
        stochastic_droplayer_rate=self.decoder_config.stochastic_droplayer_rate,
        dtype=self.dtype,
        positional_embedding='none',  # Has already been added.
        normalise_output=self.normalise_encoder_output,
        name='Decoder')(x_all, train=train)

    if self.classifier == 'token':
      # Remove the CLS token for predicting reconstructions.
      x_decoded = x_decoded[:, 1:, :]

    x_prelogits = nn_layers.IdentityLayer(name='pre_logits')(x_decoded)

    x_logits = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(x_prelogits)

    return x_logits, token_mask


class ViViTMaskedAutoencoderModel(base_model.MaskedFeatureRegressionModel):
  """Vision Video Transformer model for masked autoencoder pretraining."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    num_classes = base_model.get_output_shapes(
        self.config.masked_feature_loss.target,
        tuple(self.config.model.patches.size),
        self.config.masked_feature_loss.select_central_frame)

    return ViViTMaskedAutoencoder(
        num_classes=num_classes,
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        token_mask_probability=(
            self.config.masked_feature_loss.token_mask_probability),
        masking_strategy=self.config.masked_feature_loss.get('masking_strategy',
                                                             'random'),
        temporal_encoding_config=self.config.model.temporal_encoding_config,
        attention_config=self.config.model.attention_config,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        attention_dropout_rate=self.config.model.get(
            'attention_dropout_rate', 0.1),
        stochastic_droplayer_rate=self.config.model.get(
            'stochastic_droplayer_rate', 0),
        dtype=model_dtype,
        decoder_config=self.config.model.get('decoder_config', None),
        positional_embedding=self.config.model.get('positional_embedding',
                                                   'sinusoidal_1d'),
        positional_embedding_decoder=self.config.model
        .get('positional_embedding_decoder', 'sinusoidal_1d'),
        normalise_encoder_output=self.config.model.get(
            'normalise_encoder_output', True),
    )

  def init_from_train_state(self,
                            train_state: Any,
                            restored_train_state: Any,
                            restored_model_cfg: ml_collections.ConfigDict,
                            restore_output_proj: bool = False) -> Any:
    """Updates the train_state with data from restored_train_state."""

    return vivit_model_utils.initialise_from_train_state(
        self.config,
        train_state,
        restored_train_state,
        restored_model_cfg,
        restore_output_proj)


class ViViTMAEClassificationFinetuningModel(
    classification_model.ClassificationModel):
  """Vision Video Transformer model for MAE finetuning."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    num_classes = self.dataset_meta_data['num_classes']

    return ViViT(
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
        positional_embedding=self.config.model.get('positional_embedding',
                                                   'sinusoidal_1d'),
        freeze_backbone=self.config.model.get('freeze_backbone', False),
        use_batch_norm_after_encoder=self.config.model.get(
            'use_batch_norm_after_encoder', True),
    )

  def init_from_train_state(self,
                            train_state: Any,
                            restored_train_state: Any,
                            restored_model_cfg: ml_collections.ConfigDict,
                            restore_output_proj: bool = False) -> Any:
    """Updates the train_state with data from restored_train_state."""
    attention_type = self.config.model.attention_config.get(
        'type', 'spacetime')
    if attention_type in [
        'spacetime', 'factorized_transformer_block',
        'factorized_self_attention_block', 'factorized_dot_product_attention'
    ]:
      vivit_transformer_key = 'Transformer'
    elif attention_type == 'factorized_encoder':
      vivit_transformer_key = 'SpatialTransformer'
    else:
      raise ValueError(f'Attention type {attention_type} does not exist.')

    # Move the encoder norm if it is there outside of the transformer encoder.
    if 'encoder_norm' in restored_train_state.params['Transformer']:
      restored_parameters = flax.core.unfreeze(restored_train_state.params)
      norm_parameters = restored_parameters['Transformer'].pop('encoder_norm')
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
          'Transformer'].pop('posembed_input')

      if restored_model_cfg.model.classifier == 'token':
        pos_embedding_params = restored_parameters[
            'posembed_input']['pos_embedding']
        # Drop the value of cls positional embedding.
        cls_pos_embedding = pos_embedding_params[:, 0]
        restored_parameters['posembed_input'][
            'pos_embedding'] = pos_embedding_params[:, 1:]
        # Add the value of cls positional embedding to cls token.
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
