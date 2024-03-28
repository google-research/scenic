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

"""TokenLearner model.

Includes the implementation of the paper: https://arxiv.org/abs/2106.11297
"""

import copy
import functools
import math
from typing import Any, Optional, Type, Union

from absl import logging
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import classification_model
from scenic.model_lib.base_models import multilabel_classification_model
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import vit
from scenic.projects.vivit import model as vivit_model
from scenic.projects.vivit import model_utils as vivit_model_utils

# JAX team is working on type annotation for PyTree:
# https://github.com/google/jax/issues/1555
Array = Union[jnp.ndarray, np.ndarray]
PyTree = Any


def get_model_cls(model_name: str) -> Type[base_model.BaseModel]:
  """Returns model class given its name."""
  if model_name == 'token_learner_multilabel_classification':
    return TokenLearnerMultilabelClassificationModel
  elif model_name == 'token_learner_classification':
    return TokenLearnerClassificationModel
  else:
    raise ValueError(f'Unrecognized model: {model_name}.')


class TokenLearnerModule(nn.Module):
  """TokenLearner module.

  This is the module used for the experiments in the paper.

  Attributes:
    num_tokens: Number of tokens.
  """
  num_tokens: int
  use_sum_pooling: bool = True

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies learnable tokenization to the 2D inputs.

    Args:
      inputs: Inputs of shape `[bs, h, w, c]` or `[bs, hw, c]`.

    Returns:
      Output of shape `[bs, n_token, c]`.
    """
    if inputs.ndim == 3:
      n, hw, c = inputs.shape
      h = int(math.sqrt(hw))
      inputs = jnp.reshape(inputs, [n, h, h, c])

      if h * h != hw:
        raise ValueError('Only square inputs supported.')

    feature_shape = inputs.shape

    selected = inputs
    selected = nn.LayerNorm()(selected)

    for _ in range(3):
      selected = nn.Conv(
          self.num_tokens,
          kernel_size=(3, 3),
          strides=(1, 1),
          padding='SAME',
          use_bias=False)(selected)  # Shape: [bs, h, w, n_token].

      selected = nn.gelu(selected)

    selected = nn.Conv(
        self.num_tokens,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        use_bias=False)(selected)  # Shape: [bs, h, w, n_token].

    selected = jnp.reshape(
        selected, [feature_shape[0], feature_shape[1] * feature_shape[2], -1
                  ])  # Shape: [bs, h*w, n_token].
    selected = jnp.transpose(selected, [0, 2, 1])  # Shape: [bs, n_token, h*w].
    selected = nn.sigmoid(selected)[..., None]  # Shape: [bs, n_token, h*w, 1].

    feat = inputs
    feat = jnp.reshape(
        feat, [feature_shape[0], feature_shape[1] * feature_shape[2], -1
              ])[:, None, ...]  # Shape: [bs, 1, h*w, c].

    if self.use_sum_pooling:
      inputs = jnp.sum(feat * selected, axis=2)
    else:
      inputs = jnp.mean(feat * selected, axis=2)

    return inputs


class TokenLearnerModuleV11(nn.Module):
  """TokenLearner module Version 1.1, using slightly different conv. layers.

  Instead of using 4 conv. layers with small channels to implement spatial
  attention, this version uses a MLP with gelu inbetween. It also uses softmax
  instead of sigmoid. We confirmed that this version works better in general.

  Attributes:
    num_tokens: Number of tokens.
    bottleneck_dim: The size of hidden units in the MLP for spatial attention.
    dropout_rate: Dropout rate.
  """
  num_tokens: int
  bottleneck_dim: int = 64
  dropout_rate: float = 0.

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
    """Applies learnable tokenization to the 2D inputs.

    Args:
      inputs: Inputs of shape `[bs, h, w, c]`.
      deterministic: Weather we are in the deterministic mode (e.g inference
        time) or not.

    Returns:
      Output of shape `[bs, n_token, c]`.
    """
    if inputs.ndim == 4:
      n, h, w, c = inputs.shape
      inputs = jnp.reshape(inputs, [n, h*w, c])

    feature_shape = inputs.shape

    selected = inputs

    selected = nn.LayerNorm()(selected)

    selected = attention_layers.MlpBlock(
        mlp_dim=self.bottleneck_dim,
        out_dim=self.num_tokens,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        name='token_masking')(
            selected, deterministic=deterministic)

    selected = jnp.reshape(
        selected,
        [feature_shape[0], -1, self.num_tokens])  # Shape: [bs, h*w, n_token].
    selected = jnp.transpose(selected, [0, 2, 1])  # Shape: [bs, n_token, h*w].
    selected = jax.nn.softmax(selected, axis=-1)

    feat = inputs
    feat = jnp.reshape(
        feat, [feature_shape[0], -1, feature_shape[-1]])  # Shape: [bs, h*w, c].

    feat = jnp.einsum('...si,...id->...sd', selected, feat)

    return feat


class TokenFuser(nn.Module):
  """Token fusion module.

  Attributes:
    use_normalization: Whether to use LayerNorm layers. This is needed when
      using sum pooling in the TokenLearner module.
    bottleneck_dim: The size of hidden units in the MLP for spatial attention.
    dropout_rate: Dropout rate.
  """

  use_normalization: bool = True
  bottleneck_dim: int = 64
  dropout_rate: float = 0.

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, original: jnp.ndarray,
               deterministic: bool) -> jnp.ndarray:
    """Applies token fusion to the generate 2D ouputs.

    Args:
      inputs: Inputs of shape `[bs, n_token, c]`.
      original: Inputs of shape `[bs, hw, c]` or `[bs, h, w, c]`.
      deterministic: Weather we are in the deterministic mode (e.g inference
        time) or not.

    Returns:
      Output tensor with the shape identical to `original'.
    """
    feature_shape = inputs.shape
    num_tokens = feature_shape[-2]

    if original.ndim == 4:
      n, h, w, c = original.shape
      original = jnp.reshape(original, [n, h*w, c])

    if self.use_normalization:
      inputs = nn.LayerNorm(name='fuser_mix_norm1')(inputs)

    inputs = jnp.transpose(inputs, axes=[0, 2, 1])  # Shape: [bs, c, n_token].
    inputs = nn.Dense(
        num_tokens,
        kernel_init=nn.initializers.zeros)(inputs)
    inputs = jnp.transpose(inputs, axes=[0, 2, 1])  # Shape: [bs, n_token, c].

    if self.use_normalization:
      inputs = nn.LayerNorm(name='fuser_mix_norm2')(inputs)

    original = nn.LayerNorm()(original)
    mix = attention_layers.MlpBlock(
        mlp_dim=self.bottleneck_dim,
        out_dim=num_tokens,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        name='token_masking')(
            original, deterministic=deterministic)  # Shape: [bs, h*w, n_token].
    mix = nn.sigmoid(mix)

    inputs = jnp.einsum('...sc,...hs->...hc',
                        inputs, mix)  # Shape: [bs, h*w, c].

    inputs = nn.Dropout(rate=self.dropout_rate)(
        inputs, deterministic=deterministic)

    if original.ndim == 4:
      inputs = jnp.reshape(inputs, [n, h, w, -1])

    return inputs


class EncoderMod(nn.Module):
  """Transformer Encoder modified, to use TokenLearner.

  Attributes:
    num_layers: Number of layers.
    mlp_dim: Dimension of the MLP on top of the attention block.
    num_heads: The number of self-attention heads.
    dropout_rate: Dropout rate in the transformer encoder.
    attention_dropout_rate: Dropout rate for multi-head dot-product attention.
    tokenizer_type: Which tokenizer to use. 'dynamic' or 'video' means using
      TokenLearner.
    temporal_dimensions: The number of temporal dimensions in the input. This
      is necessary for video models. Default is 1 for image models.
    num_tokens: Number of tokens to learn by TokenLearner. The total number of
      tokens learned is thus num_tokens * temporal_dimensions.
    tokenlearner_loc: The layer indices to add TokenLearner to.
    use_v11: whether to use version 1.1.
    dtype: Dtype of activations.
  """
  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  tokenizer_type: str = 'patch'
  temporal_dimensions: int = 1
  num_tokens: int = 8
  tokenlearner_loc: int = 12
  use_v11: bool = True
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, train: bool = False):
    """Applies Transformer model on the inputs."""

    assert inputs.ndim == 3  # Shape is `[batch, len, emb]`.
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)

    x = vit.AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),  # From BERT.
        name='posembed_input')(
            inputs)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    tl_locs = [self.tokenlearner_loc]
    tl_size = [self.num_tokens]
    # Input Encoder.
    for lyr in range(self.num_layers):
      if self.tokenizer_type in {'dynamic', 'video'} and lyr in tl_locs:
        tl_index = tl_locs.index(lyr)

        n, thw, c = x.shape
        hw = thw // self.temporal_dimensions
        x = jnp.reshape(x, [n * self.temporal_dimensions, hw, c])
        if self.use_v11:
          x = TokenLearnerModuleV11(
              tl_size[tl_index], dropout_rate=self.dropout_rate)(
                  x, deterministic=not train)  # Shape [n*t, n_tokens, c].
        else:
          x = TokenLearnerModule(tl_size[tl_index])(x)
        _, n_tokens, c = x.shape
        x = jnp.reshape(x, [n, self.temporal_dimensions * n_tokens, c])

        x = vit.Encoder1DBlock(
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            name=f'encoderblock_{lyr}',
            dtype=dtype)(
                x, deterministic=not train)

      else:
        x = vit.Encoder1DBlock(
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            name=f'encoderblock_{lyr}',
            dtype=dtype)(
                x, deterministic=not train)
    encoded = nn.LayerNorm(name='encoder_norm')(x)
    return encoded


class EncoderModFuser(nn.Module):
  """Transformer Encoder modified, to use TokenLearner + TokenFuser.

  Attributes:
    num_layers: Number of layers.
    mlp_dim: Dimension of the MLP on top of the attention block.
    num_heads: The number of self-attention heads.
    dropout_rate: Dropout rate in the transformer encoder.
    attention_dropout_rate: Dropout rate for multi-head dot-product attention.
    tokenizer_type: Which tokenizer to use. 'dynamic' or 'video' means using
      TokenLearner.
    temporal_dimensions: The number of temporal dimensions in the input. This
      is necessary for video models. Default is 1 for image models.
    num_tokens: Number of tokens to learn by TokenLearner. The total number of
      tokens learned is thus num_tokens * temporal_dimensions.
    tokenlearner_loc: The layer indices to add TokenLearner to.
    use_v11: whether to use version 1.1 of the TokenLearner
      module. Works better when the module is applied early in the network.
    dtype: Dtype of activations.

  """
  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  tokenizer_type: str = 'patch'
  temporal_dimensions: int = 1
  num_tokens: int = 8
  tokenlearner_loc: int = 12
  use_v11: bool = True
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, train: bool = False):
    """Applies Transformer model on the inputs."""

    assert inputs.ndim == 3  # Shape is `[batch, len, emb]`.
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)

    x = vit.AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),  # From BERT.
        name='posembed_input')(
            inputs)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder.
    for lyr in range(self.num_layers):
      if (self.tokenizer_type in {'dynamic', 'video'} and
          lyr >= self.tokenlearner_loc):
        n, thw, c = x.shape
        hw = thw // self.temporal_dimensions
        x = jnp.reshape(x, [n * self.temporal_dimensions, hw, c])
        residual = x
        if self.use_v11:
          x = TokenLearnerModuleV11(
              self.num_tokens, dropout_rate=self.dropout_rate)(
                  x, deterministic=not train)  # Shape [n*t, n_tokens, c].
        else:
          x = TokenLearnerModule(self.num_tokens)(x)
        _, n_tokens, c = x.shape
        x = jnp.reshape(x, [n, self.temporal_dimensions * n_tokens, c])

        x = vit.Encoder1DBlock(
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            name=f'encoderblock_{lyr}',
            dtype=dtype)(
                x, deterministic=not train)

        x = jnp.reshape(x, [n * self.temporal_dimensions, n_tokens, c])
        x = TokenFuser(dropout_rate=self.dropout_rate)(
            x, residual,
            deterministic=not train)  # [n * t, n_tokens, c], [n * t, hw, c]
        x = x + residual
        x = jnp.reshape(x, [n, self.temporal_dimensions * hw, c])

      else:
        x = vit.Encoder1DBlock(
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            name=f'encoderblock_{lyr}',
            dtype=dtype)(
                x, deterministic=not train)
      logging.info('Layer %d. Shape %s', lyr, x.shape)
    encoded = nn.LayerNorm(name='encoder_norm')(x)
    return encoded


class TokenLearnerViT(nn.Module):
  """Vision Transformer model with TokenLearner.

    Attributes:
    num_classes: Number of output classes.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    representation_size: Size of the representation layer in the model's head.
      if None, we skip the extra projection + tanh activation at the end.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
      'token'.
    dtype: JAX data type for activations.
  """

  num_classes: int
  mlp_dim: int
  num_layers: int
  num_heads: int
  tokenizer: ml_collections.ConfigDict
  hidden_size: int
  representation_size: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  classifier: str = 'gap'
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, debug: bool = False):

    temporal_dimensions = 1
    if self.tokenizer.type == 'patch':
      fh, fw = self.tokenizer.patches.size
      x = nn.Conv(
          self.hidden_size, (fh, fw),
          strides=(fh, fw),
          padding='VALID',
          name='embedding')(
              x)
      n, h, w, c = x.shape
      x = jnp.reshape(x, [n, h * w, c])
    elif self.tokenizer.type == 'dynamic':
      fh, fw = self.tokenizer.patches.size
      x = nn.Conv(
          self.hidden_size, (fh, fw),
          strides=(fh, fw),
          padding='VALID',
          name='embedding')(
              x)
      n, h, w, c = x.shape
      x = jnp.reshape(x, [n, h * w, c])
    elif self.tokenizer.type == 'video':
      x, temporal_dimensions = vivit_model.temporal_encode(
          x, self.tokenizer.temporal_encoding_config, self.tokenizer.patches,
          self.hidden_size)
    else:
      raise ValueError('Unknown tokenizer type')

    use_v11 = self.tokenizer.get('use_v11', True)

    if self.tokenizer.use_tokenfuse:
      x = EncoderModFuser(
          mlp_dim=self.mlp_dim,
          num_layers=self.num_layers,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          tokenizer_type=self.tokenizer.type,
          num_tokens=self.tokenizer.num_tokens,
          tokenlearner_loc=self.tokenizer.tokenlearner_loc,
          use_v11=use_v11,
          temporal_dimensions=temporal_dimensions,
          dtype=self.dtype,
          name='Transformer')(
              x, train=train)
    else:
      x = EncoderMod(
          mlp_dim=self.mlp_dim,
          num_layers=self.num_layers,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          tokenizer_type=self.tokenizer.type,
          num_tokens=self.tokenizer.num_tokens,
          tokenlearner_loc=self.tokenizer.tokenlearner_loc,
          use_v11=use_v11,
          temporal_dimensions=temporal_dimensions,
          dtype=self.dtype,
          name='Transformer')(
              x, train=train)

    fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
    x = fn(x, axis=1)

    if self.representation_size is not None:
      x = nn.Dense(self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = nn_layers.IdentityLayer(name='pre_logits')(x)

    x = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(
            x)
    return x


class TokenLearnerViTRepresentation(nn.Module):
  """Token Learner + ViT without classification head.

    Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    representation_size: Size of the representation layer in the model's head.
      if None, we skip the extra projection + tanh activation at the end.
    use_concat_final: Whether to use the concatenation instead of mean pooling
      at the end of the network.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    dtype: JAX data type for activations.
  """

  mlp_dim: int
  num_layers: int
  num_heads: int
  tokenizer: ml_collections.ConfigDict
  hidden_size: int
  target_channel_dim: int
  use_concat_final: bool = False
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, debug: bool = False):

    fh, fw = self.tokenizer.patches.size
    if len(x.shape) == 5:
      n, t, h, w, _ = x.shape
      x = jnp.reshape(x, [n * t, h, w, -1])
    else:
      n = x.shape[0]
      t = 1

    x = nn.Conv(
        self.hidden_size, (fh, fw),
        strides=(fh, fw),
        padding='VALID',
        name='embedding')(
            x)
    x = jnp.reshape(x, [n, -1, self.hidden_size])

    use_v11 = self.tokenizer.get('use_v11', True)

    if self.tokenizer.use_tokenfuse:
      x = EncoderModFuser(
          mlp_dim=self.mlp_dim,
          num_layers=self.num_layers,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          tokenizer_type=self.tokenizer.type,
          num_tokens=self.tokenizer.num_tokens,
          tokenlearner_loc=self.tokenizer.tokenlearner_loc,
          use_v11=use_v11,
          temporal_dimensions=t,
          dtype=self.dtype,
          name='Transformer')(
              x, train=train)
    else:
      x = EncoderMod(
          mlp_dim=self.mlp_dim,
          num_layers=self.num_layers,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          tokenizer_type=self.tokenizer.type,
          num_tokens=self.tokenizer.num_tokens,
          tokenlearner_loc=self.tokenizer.tokenlearner_loc,
          use_v11=use_v11,
          temporal_dimensions=t,
          dtype=self.dtype,
          name='Transformer')(
              x, train=train)

    return x


class TokenLearnerMultilabelClassificationModel(
    multilabel_classification_model.MultiLabelClassificationModel):
  """TokenLearner ViT model for multi-label image classification task."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return TokenLearnerViT(
        num_classes=self.dataset_meta_data['num_classes'],
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        tokenizer=self.config.model.tokenizer,
        hidden_size=self.config.model.hidden_size,
        representation_size=self.config.model.representation_size,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.1),
        dtype=model_dtype,
    )

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict({
        'model':
            dict(
                num_heads=2,
                num_layers=1,
                mlp_dim=32,
                dropout_rate=0.,
                attention_dropout_rate=0.,
                hidden_size=16,
                classifier='gap',
                data_dtype_str='float32',
                tokenizer=None,
            )
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
    return init_token_learner_from_train_state(train_state,
                                               restored_train_state,
                                               self.config, restored_model_cfg)


class TokenLearnerClassificationModel(classification_model.ClassificationModel):
  """TokenLearner ViT model for classification."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return TokenLearnerViT(
        num_classes=self.dataset_meta_data['num_classes'],
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        tokenizer=self.config.model.tokenizer,
        hidden_size=self.config.model.hidden_size,
        representation_size=self.config.model.representation_size,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.1),
        dtype=model_dtype,
    )

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict({
        'model':
            dict(
                num_heads=2,
                num_layers=1,
                mlp_dim=32,
                dropout_rate=0.,
                attention_dropout_rate=0.,
                hidden_size=16,
                classifier='gap',
                data_dtype_str='float32',
                tokenizer=None,
            )
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
    return init_token_learner_from_train_state(train_state,
                                               restored_train_state,
                                               self.config, restored_model_cfg)

  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one
        of the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      label, weights)```
    """
    del split  # For all splits, we return the same metric functions.

    return functools.partial(
        classification_model.classification_metrics_function,
        target_is_onehot=self.dataset_meta_data.get('target_is_onehot', False),
        metrics=vivit_model.ViViT_CLASSIFICATION_METRICS)


def init_token_learner_from_train_state(
    train_state: Any, restored_train_state: Any,
    model_cfg: ml_collections.ConfigDict,
    restored_model_cfg: ml_collections.ConfigDict) -> Any:
  """Updates the train_state with data from restored_train_state.

  This function is writen to be used for 'fine-tuning' experiments. Here, we
  do some surgery to support larger resolutions (longer sequence length) in
  the transformer block, with respect to the learned pos-embeddings.

  Args:
    train_state: A raw TrainState for the model.
    restored_train_state: A TrainState that is loaded with parameters/state of a
      pretrained model.
    model_cfg: Configuration of the model. Usually used for some asserts.
    restored_model_cfg: Configuration of the model from which the
      restored_train_state come from. Usually used for some asserts.

  Returns:
    Updated train_state.
  """
  params = flax.core.unfreeze(train_state.optimizer.target)
  restored_params = flax.core.unfreeze(restored_train_state.optimizer.target)

  # Start moving parameters, one-by-one and apply changes if needed.
  for m_key, m_params in restored_params.items():
    if m_key == 'output_projection':
      # For the classifier head, we use a the randomly initialized params and
      #   ignore the the one from pretrained model.
      pass

    elif m_key == 'pre_logits':
      if model_cfg.model.representation_size is None:
        # We don't have representation_size in the new model, so let's ignore
        #   it from the pretained model, in case it has it.
        # Note, removing the key from the dictionary is necessary to prevent
        #   obscure errors from the Flax optimizer.
        params.pop(m_key, None)
      else:
        assert restored_model_cfg.model.representation_size
        params[m_key] = m_params

    elif m_key == 'Transformer':
      for tm_key, tm_params in m_params.items():
        if tm_key == 'posembed_input':  # Might need resolution change.
          # TODO(aarnab): Adapt config as its different to ViVIT. Unify the two.
          if model_cfg.model.tokenizer.get('temporal_encoding_config'):
            with model_cfg.unlocked():
              model_cfg.model.temporal_encoding_config = copy.deepcopy(
                  model_cfg.model.tokenizer.temporal_encoding_config)
              model_cfg.model.patches = copy.deepcopy(
                  model_cfg.model.tokenizer.patches)
          vivit_model_utils.init_posemb(params[m_key], m_params, model_cfg,
                                        restored_model_cfg, is_temporal=False)
        else:  # Other parameters of the Transformer encoder.
          params[m_key][tm_key] = tm_params

    elif m_key == 'embedding':
      init_embedding(params, m_params, model_cfg)

    else:
      # Use the rest as they are in the pretrianed model.
      params[m_key] = m_params

  return train_state.replace(
      optimizer=train_state.optimizer.replace(target=flax.core.freeze(params)))


def init_embedding(to_params: PyTree, from_params: PyTree,
                   config: ml_collections.ConfigDict) -> None:
  """Initialize input embedding.

  Args:
    to_params: PyTree of model parameters that will be updated. This argument
      is modified by the function.
    from_params: PyTree of model parameters that are being restored from.
    config: Config of the model being restored.
  """
  if config.init_from.get('restore_input_embedding', True):
    input_kernel = to_params['embedding']['kernel']
    restored_kernel = from_params['kernel']
    restored_bias = from_params['bias']

    if input_kernel.shape != restored_kernel.shape:
      # This branch should only be entered if we are initialising a video model
      # from an image model.
      # Kernel dimensions for video model are [t, h, w, c_in, c_out].
      temporal_encoding_config = config.model.tokenizer.temporal_encoding_config
      if temporal_encoding_config.method != '3d_conv':
        raise ValueError(
            'Input kernel dimensions should only differ if 3d_conv is the'
            'temporal encoding method.')
      if input_kernel.shape[1:] != restored_kernel.shape:
        raise ValueError(
            'All filter dimensions besides the temporal dimension should be'
            f'equal. {input_kernel.shape} vs {restored_kernel.shape}')

      kernel_init_method = temporal_encoding_config.kernel_init_method
      if kernel_init_method == 'average_frame_initializer':
        # This corresponds to "filter inflation" in
        # J Carreira and A Zisserman. Quo vadis, action recognition?
        # A new model and the kinetics dataset. CVPR 2017".
        logging.info('Initializing input kernel with filter inflation.')
        t = input_kernel.shape[0]
        restored_kernel = np.expand_dims(restored_kernel, axis=0)
        restored_kernel = np.tile(restored_kernel, [t, 1, 1, 1, 1]) / t
      elif kernel_init_method == 'central_frame_initializer':
        logging.info('Initializing input kernel to select centre frame.')
        central_time_index = input_kernel.shape[0] // 2
        temp = np.zeros(input_kernel.shape)
        temp[central_time_index] = restored_kernel.copy()
        restored_kernel = temp
      else:
        raise AssertionError(
            'Unknown input kernel initialization {}'.format(kernel_init_method))

    to_params['embedding']['kernel'] = restored_kernel
    to_params['embedding']['bias'] = restored_bias
  else:
    logging.info('Not restoring input embedding parameters')
