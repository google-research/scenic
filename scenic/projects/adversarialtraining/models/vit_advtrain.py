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

from absl import logging
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models.multilabel_classification_model import MultiLabelClassificationModel
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
import scipy


Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


class SplitDropout(nn.Module):
  """Dropout with two paths."""

  rate: float = 0.1
  aux_rate: float = 0.2

  @nn.compact
  def __call__(self, x, deterministic: bool,
               use_aux_dropout: bool) -> jnp.ndarray:
    x1 = nn.Dropout(rate=self.rate)(x, deterministic)
    x2 = nn.Dropout(rate=self.aux_rate)(x, deterministic)
    if use_aux_dropout:
      return x2
    else:
      return x1


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    posemb_init: Positional embedding initializer.

  Returns:
    Output in shape `[bs, timesteps, in_dim]`.
  """
  posemb_init: Initializer = nn.initializers.normal(stddev=0.02)  # From BERT.

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    # Inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape,
                    inputs.dtype)
    return inputs + pe


class SplitStochasticDepth(nn.Module):
  """Stochastic depth with two paths."""
  stochastic_depth: float
  aux_stochastic_depth: float

  @nn.compact
  def __call__(self, x: jnp.ndarray, deterministic: bool,
               use_aux_dropout: bool) -> jnp.ndarray:
    """Generate the stochastic depth mask in order to apply layer-drop."""
    if not deterministic and use_aux_dropout:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      mask = jax.random.bernoulli(
          self.make_rng('dropout'), self.aux_stochastic_depth, shape)
      return x * (1.0 - mask)
    if not deterministic and not use_aux_dropout:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      mask = jax.random.bernoulli(
          self.make_rng('dropout'), self.stochastic_depth, shape)
      return x * (1.0 - mask)
    else:
      return x


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of self-attention heads.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value.

  Returns:
    output after transformer encoder block.
  """
  mlp_dim: int
  num_heads: int
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  aux_dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  aux_stochastic_depth: float = 0.0

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, deterministic: bool,
               use_aux_dropout: bool) -> jnp.ndarray:
    """Applies Encoder1DBlock module."""
    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate)(x, x)
    x = SplitDropout(
        rate=self.dropout_rate,
        aux_rate=self.aux_dropout_rate)(x, deterministic, use_aux_dropout)
    x = SplitStochasticDepth(self.stochastic_depth, self.aux_stochastic_depth)(
        x, deterministic, use_aux_dropout) + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = attention_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(
            y, deterministic=deterministic)

    return SplitStochasticDepth(self.stochastic_depth,
                                self.aux_stochastic_depth)(y, deterministic,
                                                           use_aux_dropout) + x


class Encoder(nn.Module):
  """Transformer Encoder.

  Attributes:
    num_layers: Number of layers.
    mlp_dim: Dimension of the mlp on top of attention block.
    inputs_positions: Input subsequence positions for packed examples.
    dropout_rate: Dropout rate.
    stochastic_depth: probability of dropping a layer linearly grows
      from 0 to the provided value. Our implementation of stochastic depth
      follows timm library, which does per-example layer dropping and uses
      independent dropping patterns for each skip-connection.
    dtype: Dtype of activations.
  """
  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  aux_dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  aux_stochastic_depth: float = 0.0
  sd_direction: str = 'drop_late'
  aux_sd_direction: str = 'drop_late'
  dtype: Any = jnp.float32

  def get_sd_coef(self, sd_str, lyr):
    mean_of_all_layers = np.mean(
        [x / max(self.num_layers - 1, 1) for x in range(self.num_layers)])
    if sd_str == 'drop_flat':
      sd_coef = mean_of_all_layers
    elif sd_str == 'drop_late':
      sd_coef = lyr / max(self.num_layers - 1, 1)
    elif sd_str == 'drop_early':
      sd_coef = max(self.num_layers - 1 - lyr, 0) / max(self.num_layers - 1, 1)
    else:
      raise NotImplementedError('not implemented sd_direction %s' % sd_str)
    return sd_coef

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               train: bool = False,
               use_aux_dropout: bool = False):
    """Applies Transformer model on the inputs."""

    assert inputs.ndim == 3  # Shape is `[batch, len, emb]`.
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)

    x = AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
        name='posembed_input')(
            inputs)
    x = SplitDropout(
        rate=self.dropout_rate, aux_rate=self.aux_dropout_rate)(
            x, deterministic=not train, use_aux_dropout=use_aux_dropout)

    # Input Encoder.
    for lyr in range(self.num_layers):

      clean_sd_coef = self.get_sd_coef(self.sd_direction, lyr)
      aux_sd_coef = self.get_sd_coef(self.aux_sd_direction, lyr)

      #      print('sd', self.sd_direction, lyr, sd_coef, flush=True)
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          aux_dropout_rate=self.aux_dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_depth=clean_sd_coef * self.stochastic_depth,
          aux_stochastic_depth=aux_sd_coef * self.aux_stochastic_depth,
          name=f'encoderblock_{lyr}',
          dtype=dtype)(
              x, deterministic=not train, use_aux_dropout=use_aux_dropout)

    encoded = nn.LayerNorm(name='encoder_norm')(x)
    return encoded


class ViT(nn.Module):
  """Vision Transformer model.

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
  patches: ml_collections.ConfigDict
  hidden_size: int
  representation_size: Optional[int] = None
  dropout_rate: float = 0.1
  aux_dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  aux_stochastic_depth: float = 0.0
  sd_direction: str = 'drop_flat'
  aux_sd_direction: str = 'drop_flat'
  classifier: str = 'gap'
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               *,
               train: bool,
               use_aux_dropout: bool = False,
               debug: bool = False):
    #    print('use_aux_dropout', use_aux_dropout)

    n, h, w, c = x.shape
    if self.patches.get('grid') is not None:
      gh, gw = self.patches.grid
      fh, fw = h // gh, w // gw
    else:
      fh, fw = self.patches.size
      gh, gw = h // fh, w // fw
    if self.hidden_size:  # We can merge s2d+emb into a single conv.
      x = nn.Conv(
          self.hidden_size, (fh, fw),
          strides=(fh, fw),
          padding='VALID',
          name='embedding')(
              x)
    else:
      # This path often results in excessive padding.
      x = jnp.reshape(x, [n, gh, fh, gw, fw, c])
      x = jnp.transpose(x, [0, 1, 3, 2, 4, 5])
      x = jnp.reshape(x, [n, gh, gw, -1])

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    x = Encoder(
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        aux_dropout_rate=self.aux_dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
        aux_stochastic_depth=self.aux_stochastic_depth,
        sd_direction=self.sd_direction,
        aux_sd_direction=self.aux_sd_direction,
        dtype=self.dtype,
        name='Transformer')(
            x, train=train, use_aux_dropout=use_aux_dropout)

    if self.classifier in ('token', '0'):
      x = x[:, 0]
    elif self.classifier in ('gap', 'gmp', 'gsp'):
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


class ViTMultiLabelClassificationModel(MultiLabelClassificationModel):
  """Vision Transformer model for multi-label classification task."""

  def build_flax_model(self)-> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return ViT(
        num_classes=self.dataset_meta_data['num_classes'],
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        representation_size=self.config.model.representation_size,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        aux_dropout_rate=self.config.advprop.get('aux_dropout_rate', 0.1),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.1),
        stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
        aux_stochastic_depth=self.config.advprop.get('aux_stochastic_depth',
                                                     0.0),
        sd_direction=self.config.advprop.get('sd_direction', 'drop_flat'),
        aux_sd_direction=self.config.advprop.get('aux_sd_direction',
                                                 'drop_flat'),
        dtype=model_dtype,
    )

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict({
        'model':
            dict(
                num_heads=2,
                num_layers=1,
                representation_size=16,
                mlp_dim=32,
                dropout_rate=0.,
                attention_dropout_rate=0.,
                hidden_size=16,
                patches={'grid': (4, 4)},
                classifier='gap',
                data_dtype_str='float32')
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
    return init_vit_from_train_state(train_state, restored_train_state,
                                     self.config, restored_model_cfg)


def init_vit_from_train_state(
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
          posemb = params[m_key]['posembed_input']['pos_embedding']
          restored_posemb = m_params['posembed_input']['pos_embedding']

          if restored_posemb.shape != posemb.shape:
            # Rescale the grid of pos, embeddings: param shape is (1, N, d).
            logging.info('Resized variant: %s to %s', restored_posemb.shape,
                         posemb.shape)
            ntok = posemb.shape[1]
            if restored_model_cfg.model.classifier == 'token':
              # The first token is the CLS token.
              cls_tok = restored_posemb[:, :1]
              restored_posemb_grid = restored_posemb[0, 1:]
              ntok -= 1
            else:
              cls_tok = restored_posemb[:, :0]
              restored_posemb_grid = restored_posemb[0]

            restored_gs = int(np.sqrt(len(restored_posemb_grid)))
            gs = int(np.sqrt(ntok))
            if restored_gs != gs:  # We need resolution change.
              logging.info('Grid-size from %s to %s.', restored_gs, gs)
              restored_posemb_grid = restored_posemb_grid.reshape(
                  restored_gs, restored_gs, -1)
              zoom = (gs / restored_gs, gs / restored_gs, 1)
              restored_posemb_grid = scipy.ndimage.zoom(
                  restored_posemb_grid, zoom, order=1)
              restored_posemb_grid = restored_posemb_grid.reshape(
                  1, gs * gs, -1)
              # Attache the CLS token again.
              restored_posemb = jnp.array(
                  np.concatenate([cls_tok, restored_posemb_grid], axis=1))

          params[m_key][tm_key]['pos_embedding'] = restored_posemb
        else:  # Other parameters of the Transformer encoder.
          params[m_key][tm_key] = tm_params

    else:
      # Use the rest as they are in the pretrianed model.
      params[m_key] = m_params

  return train_state.replace(
      optimizer=train_state.optimizer.replace(target=flax.core.freeze(params)))
