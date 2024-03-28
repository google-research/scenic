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

"""Vision Transformer."""

from typing import Any, Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

from scenic.model_lib.base_models import classification_model
from scenic.model_lib.base_models import multilabel_classification_model
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scenic.projects.av_mae import base_model
from scenic.projects.av_mae import model_utils
from scenic.projects.baselines import vit

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


class ViT(nn.Module):
  """Vision Transformer model.

    This differs from scenic.projects.baselines.vit in that
    -- Positional embeddings are added before the transformer block. This makes
       it easier to load MAE-pretrained checkpoints.
    -- The CLS token is added after the positional embeddings. This follows MAE.
    -- Add support for linear evaluation by adding a stop-gradient.

    Attributes:
    num_classes: Number of output classes.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    positional_embedding: The type of positional embeddings to add to the
      tokens at the beginning of the transformer encoder. Options are
      {learned_1d, sinusoidal_2d, none}.
    representation_size: Size of the representation layer in the model's head.
      if None, we skip the extra projection + tanh activation at the end.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
      'token', 'none'.
    freeze_backbone: If True, add a stop-gradient before the final
      classifier to only evaluate linear evaluation performance.
    use_batch_norm_after_encoder: Only applies when the backbone is frozen.
      In this case, an additional batch normalisation layer is applied before
      the linear classifier. This was done in MAE
      (https://arxiv.org/abs/2111.06377).
    dtype: JAX data type for activations.
  """

  num_classes: int
  mlp_dim: int
  num_layers: int
  num_heads: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  positional_embedding: str = 'learned_1d'
  representation_size: Optional[int] = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  classifier: str = 'gap'
  freeze_backbone: bool = False
  use_batch_norm_after_encoder: bool = True
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, debug: bool = False):

    fh, fw = self.patches.size
    # Extracting patches and then embedding is in fact a single convolution.
    x = nn.Conv(
        self.hidden_size, (fh, fw),
        strides=(fh, fw),
        padding='VALID',
        name='embedding')(
            x)
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # Add positional embeddings to tokens.
    if self.positional_embedding == 'learned_1d':
      x = vit.AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='posembed_input')(x)
    elif self.positional_embedding == 'sinusoidal_1d':
      x = attention_layers.Add1DPositionEmbedding(posemb_init=None)(x)
    elif self.positional_embedding == 'sinusoidal_2d':
      x_reshape = x.reshape([n, h, w, c])
      x = attention_layers.AddFixedSinCosPositionEmbedding()(x_reshape)
      x = jnp.reshape(x, [n, h * w, c])
    elif self.positional_embedding == 'none':
      pass
    else:
      raise ValueError('Unknown positional embedding: '
                       f'{self.positional_embedding}')

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    x = vit.Encoder(
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        positional_embedding='none',
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
        dtype=self.dtype,
        name='Transformer')(
            x, train=train)

    if self.classifier in ('token', '0'):
      x = x[:, 0]
    elif self.classifier in ('gap', 'gmp', 'gsp'):
      fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
      x = fn(x, axis=1)
    elif self.classifier == 'none':
      pass
    else:
      raise ValueError(f'Unknown classifier {self.classifier}')

    if self.representation_size is not None:
      x = nn.Dense(self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = nn_layers.IdentityLayer(name='pre_logits')(x)

    if self.freeze_backbone:
      x = jax.lax.stop_gradient(x)

      if self.use_batch_norm_after_encoder:
        x = nn.BatchNorm(
            # Match PyTorch default and MAE PyTorch code
            # https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_linprobe.py#L222
            momentum=0.9,
            epsilon=1e-6,
            use_bias=False,
            use_scale=False,
        )(x, use_running_average=not train)

    if self.num_classes > 0:
      # If self.num_classes <= 0, we just return the backbone features.
      x = nn.Dense(
          self.num_classes,
          kernel_init=nn.initializers.zeros,
          name='output_projection')(
              x)
    return x


class ViTMaskedAutoencoder(nn.Module):
  """Vision Transformer model for masked-autoencoding.

    The differences to `scenic.baselines.vit` are that:
    -- Remove masked tokens from the encoder.
    -- Process all tokens with the decoder.

    Attributes:
      num_classes: Number of output classes.
      mlp_dim: Dimension of the mlp on top of attention block.
      num_layers: Number of layers.
      num_heads: Number of self-attention heads.
      patches: Configuration of the patches extracted in the stem of the model.
      hidden_size: Size of the hidden state of the output of model's stem.
      token_mask_probability: Probability of masking out the input tokens (with
        a learned mask token) during training.
      representation_size: Size of the representation layer in the model's head.
        if None, we skip the extra projection + tanh activation at the end.
      dropout_rate: Dropout rate.
      attention_dropout_rate: Dropout for attention heads.
      stochastic_depth: Probability of dropping out a layer during training.
      classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
        'token'.
      dtype: JAX data type for activations.
  """

  num_classes: int
  mlp_dim: int
  num_layers: int
  num_heads: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  token_mask_probability: float
  decoder_config: ml_collections.ConfigDict
  representation_size: Optional[int] = None
  positional_embedding: str = 'sinusoidal_1d'
  positional_embedding_decoder: str = 'sinusoidal_1d'
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  classifier: str = 'gap'
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, train: bool, debug: bool = False):
    """Forward pass of Vision Transformer."""

    # Extracting patches and embed via a convolution.
    fh, fw = self.patches.size
    x_tokens = nn.Conv(
        self.hidden_size, (fh, fw),
        strides=(fh, fw),
        padding='VALID',
        name='embedding')(inputs)
    n_batch, height, width, channels = x_tokens.shape
    n_tokens = height * width
    x_tokens = jnp.reshape(x_tokens, [n_batch, n_tokens, channels])

    # Add positional encodings.
    x_tokens = model_utils.add_positional_embeddings(
        x_tokens, self.positional_embedding, [n_batch, height, width, channels])

    if train:
      # Generate mask indices.
      n_masked = int(self.token_mask_probability * n_tokens)
      mask_indices, unmasked_indices, token_mask = model_utils.get_mask_indices(
          n_batch, n_tokens, n_masked, self.make_rng('dropout'))

      # Process only unmasked tokens with the encoder.
      batch_indices = jnp.arange(n_batch).reshape(n_batch, 1)
      x_unmasked = x_tokens[batch_indices, unmasked_indices]
    else:
      x_unmasked = x_tokens
      token_mask = jnp.zeros((n_batch, n_tokens))

    # If we want to add a class token, add it here.
    # Note that in MAE, positional encodings are not added to the CLS token.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros,
                       (1, 1, channels), x_unmasked.dtype)
      cls = jnp.tile(cls, [n_batch, 1, 1])
      x_unmasked = jnp.concatenate([cls, x_unmasked], axis=1)

    x_unmasked_encoded = vit.Encoder(
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
        dtype=self.dtype,
        positional_embedding='none',  # Has already been added.
        name='Transformer')(
            x_unmasked, train=train)

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

    # Add positional encodings to the decoder.
    x_all = model_utils.add_positional_embeddings(
        x_all, self.positional_embedding_decoder,
        [n_batch, height, width, self.decoder_config.hidden_size])

    if self.classifier == 'token':
      x_all = jnp.concatenate([cls_encoded, x_all], axis=1)

    x_decoded = vit.Encoder(
        mlp_dim=self.decoder_config.mlp_dim,
        num_layers=self.decoder_config.num_layers,
        num_heads=self.decoder_config.num_heads,
        dropout_rate=self.decoder_config.dropout_rate,
        attention_dropout_rate=self.decoder_config.attention_dropout_rate,
        stochastic_depth=self.decoder_config.stochastic_depth,
        dtype=self.dtype,
        positional_embedding='none',  # Has already been added.
        name='Decoder')(x_all, train=train)

    if self.classifier == 'token':
      # Remove the CLS token for predicting reconstructions.
      x_decoded = x_decoded[:, 1:, :]

    # Predict pixel reconstructions.
    if self.representation_size is not None:
      x_prelogits = nn.Dense(self.representation_size, name='pre_logits')(
          x_decoded)
      x_prelogits = nn.tanh(x_prelogits)
    else:
      x_prelogits = nn_layers.IdentityLayer(name='pre_logits')(x_decoded)
    x_logits = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(x_prelogits)

    return x_logits, token_mask


class ViTMaskedAutoencoderModel(base_model.MaskedFeatureRegressionModel):
  """Vision Transformer model for masked autoencoder pretraining."""

  def build_flax_model(self)-> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    num_classes = base_model.get_output_shapes(
        self.config.masked_feature_loss.target,
        tuple(self.config.model.patches.size))

    return ViTMaskedAutoencoder(
        num_classes=num_classes,
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        representation_size=self.config.model.representation_size,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        positional_embedding=self.config.model.get(
            'positional_embedding', 'sinusoidal_2d'),
        positional_embedding_decoder=self.config.model.get(
            'positional_embedding_decoder', 'sinusoidal_2d'),
        decoder_config=self.config.model.decoder_config,
        token_mask_probability=(
            self.config.masked_feature_loss.token_mask_probability),
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.1),
        stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
        dtype=model_dtype,
    )

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict({
        'model':
            dict(
                num_heads=2,
                num_layers=1,
                representation_size=None,
                mlp_dim=64,
                dropout_rate=0.,
                attention_dropout_rate=0.,
                hidden_size=16,
                patches={'size': (4, 4)},
                classifier='token',
                data_dtype_str='float32',
                decoder_config=dict(
                    num_heads=2,
                    num_layers=1,
                    mlp_dim=32,
                    hidden_size=8,
                    dropout_rate=0,
                    attention_dropout_rate=0,
                    stochastic_depth=0)),
        'masked_feature_loss':
            dict(target='rgb', token_mask_probability=0.75),
    })

  def init_from_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
    """Updates the train_state with data from restored_train_state.

    This function is writen to be used for 'fine-tuning' experiments. Here, we
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


class ViTMAEMultilabelFinetuning(
    multilabel_classification_model.MultiLabelClassificationModel):
  """Vision Transformer model for multi-label classification task."""

  def build_flax_model(self)-> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))

    return ViT(
        num_classes=self.dataset_meta_data['num_classes'],
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        positional_embedding=self.config.model.positional_embedding,
        representation_size=self.config.model.representation_size,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.0),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.0),
        stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
        freeze_backbone=self.config.model.get('freeze_backbone', False),
        use_batch_norm_after_encoder=self.config.model.get(
            'use_batch_norm_after_encoder', True),
        dtype=model_dtype,
    )

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict({
        'model':
            dict(
                num_heads=2,
                num_layers=1,
                representation_size=None,
                mlp_dim=64,
                dropout_rate=0.,
                attention_dropout_rate=0.,
                positional_embedding='learned_1d',
                hidden_size=16,
                patches={'size': (4, 4)},
                classifier='token',
                data_dtype_str='float32',
                freeze_backbone=False),
    })

  def init_from_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
    """Updates the train_state with data from restored_train_state.

    This function is writen to be used for 'fine-tuning' experiments. Here, we
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


class ViTMAEClassificationFinetuning(
    classification_model.ClassificationModel):
  """Vision Transformer model for multi-label classification task."""

  def build_flax_model(self)-> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))

    return ViT(
        num_classes=self.dataset_meta_data['num_classes'],
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        positional_embedding=self.config.model.positional_embedding,
        representation_size=self.config.model.representation_size,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.0),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.0),
        stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
        freeze_backbone=self.config.model.get('freeze_backbone', False),
        use_batch_norm_after_encoder=self.config.model.get(
            'use_batch_norm_after_encoder', True),
        dtype=model_dtype,
    )

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict({
        'model':
            dict(
                num_heads=2,
                num_layers=1,
                representation_size=None,
                mlp_dim=64,
                dropout_rate=0.,
                attention_dropout_rate=0.,
                positional_embedding='learned_1d',
                hidden_size=16,
                patches={'size': (4, 4)},
                classifier='token',
                data_dtype_str='float32',
                freeze_backbone=False,
                use_batch_norm_after_encoder=True),
    })

  def init_from_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
    """Updates the train_state with data from restored_train_state.

    This function is writen to be used for 'fine-tuning' experiments. Here, we
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
