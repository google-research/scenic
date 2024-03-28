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

"""Plain Vision Transformer from https://arxiv.org/abs/2205.01580.

This implementation is forked from the big_vision codebase.
"""

import functools
from typing import Any, Optional

from absl import logging
import flax
import flax.linen as nn
from flax.training import common_utils
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import classification_model
from scenic.model_lib.base_models import model_utils
from scenic.model_lib.base_models import multilabel_classification_model
from scenic.model_lib.layers import nn_layers
import scipy


def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32):
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]

  assert width % 4 == 0, 'Width must be mult of 4 for sincos posemb'
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1. / (temperature**omega)
  y = jnp.einsum('m,d->md', y.flatten(), omega)
  x = jnp.einsum('m,d->md', x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
  if typ == 'learn':
    return self.param(name, nn.initializers.normal(stddev=1 / np.sqrt(width)),
                      (1, np.prod(seqshape), width), dtype)
  elif typ == 'sincos2d':
    return posemb_sincos_2d(*seqshape, width, dtype=dtype)
  else:
    raise ValueError(f'Unknown posemb type: {typ}')


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    """Applies Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )
    n, l, d = x.shape  # pylint: disable=unused-variable
    x = nn.Dense(self.mlp_dim or 4 * d, **inits)(x)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic)
    x = nn.Dense(d, **inits)(x)
    return x


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    y = nn.LayerNorm()(x)
    y = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=deterministic,
    )(y, y)
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = x + y

    y = nn.LayerNorm()(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim,
        dropout=self.dropout,
    )(y, deterministic)
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    return x + y


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  depth: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    for lyr in range(self.depth):
      x = Encoder1DBlock(
          name=f'encoderblock_{lyr}',
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout=self.dropout,
      )(x, deterministic)
    return nn.LayerNorm(name='encoder_norm')(x)


class MAPHead(nn.Module):
  """Multihead Attention Pooling."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12

  @nn.compact
  def __call__(self, x):
    n, l, d = x.shape  # pylint: disable=unused-variable
    probe = self.param('probe', nn.initializers.xavier_uniform(), (1, 1, d),
                       x.dtype)
    probe = jnp.tile(probe, [n, 1, 1])

    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads, kernel_init=nn.initializers.xavier_uniform()
    )(probe, x)

    y = nn.LayerNorm()(x)
    x = x + MlpBlock(mlp_dim=self.mlp_dim)(y)
    return x[:, 0]


class ViT(nn.Module):
  """Vision Transformer model.

    Attributes:
    num_classes: Number of output classes.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    positional_embedding: The type of positional embeddings to add to the tokens
      at the beginning of the transformer encoder. Options are {learned_1d,
      sinusoidal_2d, none}.
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
  positional_embedding: str = 'learn'
  representation_size: Optional[int] = None
  dropout_rate: float = 0.1
  classifier: str = 'gap'
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, debug: bool = False):

    fh, fw = self.patches.size
    # Extracting patches and then embedding is in fact a single convolution.
    x = nn.Conv(
        self.hidden_size,
        (fh, fw),
        strides=(fh, fw),
        padding='VALID',
        name='embedding',
    )(x)
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # Add posemb before adding extra token.
    x = x + get_posemb(self, self.positional_embedding,
                       (h, w), c, 'pos_embedding', x.dtype)

    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

    x = nn.Dropout(rate=self.dropout_rate)(x, not train)

    x = Encoder(
        depth=self.num_layers,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout_rate,
        name='Transformer',
    )(x, deterministic=not train)

    if self.classifier == 'map':
      x = MAPHead(num_heads=self.num_heads, mlp_dim=self.mlp_dim)(x)
    elif self.classifier == 'gap':
      x = jnp.mean(x, axis=1)
    elif self.classifier in ('token', '0'):
      x = x[:, 0]
    else:
      raise ValueError(f'Unknown classifier {self.classifier}')

    if self.representation_size is not None:
      x = nn.Dense(self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = nn_layers.IdentityLayer(name='pre_logits')(x)

    return nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection',
    )(x)


class PlainViT(classification_model.ClassificationModel):
  """Plain Vision Transformer."""

  def build_flax_model(self)-> nn.Module:
    dtype_str = self.config.get('model_dtype_str', 'float32')
    if dtype_str != 'float32':
      raise ValueError(
          '`dtype` argument is not propagated properly '
          'in the current implementation, so only '
          '`float32` is supported for now.'
      )
    return ViT(
        num_classes=self.dataset_meta_data['num_classes'],
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        positional_embedding=self.config.model.positional_embedding,
        representation_size=self.config.model.representation_size,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.dropout_rate,
        dtype=getattr(jnp, dtype_str),
    )

  def loss_function(
      self,
      logits: jnp.ndarray,
      batch: base_model.Batch,
      model_params: Optional[jnp.ndarray] = None,
  ) -> float:
    """Returns sigmoid or softmax cross entropy loss.

    Args:
      logits: Output of model in shape [batch, length, num_classes].
      batch: Batch of data that has 'label' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """
    weights = batch.get('batch_mask')
    loss_fn = self.config.get('loss', 'sigmoid_xent')

    if self.dataset_meta_data.get('target_is_onehot', False):
      one_hot_targets = batch['label']
    else:
      # This is to support running a multi-label classification model on
      # single-label classification tasks.
      one_hot_targets = common_utils.onehot(batch['label'], logits.shape[-1])

    if loss_fn == 'sigmoid_xent':
      total_loss = model_utils.weighted_sigmoid_cross_entropy(
          logits,
          one_hot_targets,
          weights,
          label_smoothing=self.config.get('label_smoothing'),
      )
    elif loss_fn == 'softmax_xent':
      total_loss = model_utils.weighted_softmax_cross_entropy(
          logits,
          one_hot_targets,
          weights,
          label_smoothing=self.config.get('label_smoothing'),
      )
    else:
      raise ValueError(f'Unknown loss function {loss_fn}.')

    if self.config.get('l2_decay_factor'):
      l2_loss = model_utils.l2_regularization(model_params)
      total_loss = total_loss + 0.5 * self.config.l2_decay_factor * l2_loss
    return total_loss  # pytype: disable=bad-return-type  # jax-ndarray

  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      batch)```
    """
    del split  # For all splits, we return the same metric functions.
    loss_fn = self.config.get('loss', 'sigmoid_xent')
    # pylint: disable=protected-access
    if loss_fn == 'sigmoid_xent':
      return functools.partial(
          multilabel_classification_model.multilabel_classification_metrics_function,
          target_is_multihot=self.dataset_meta_data.get(
              'target_is_onehot', False
          ),
          metrics=multilabel_classification_model._MULTI_LABEL_CLASSIFICATION_METRICS,
      )
    elif loss_fn == 'softmax_xent':
      return functools.partial(
          classification_model.classification_metrics_function,
          target_is_onehot=self.dataset_meta_data.get(
              'target_is_onehot', False
          ),
          metrics=classification_model._CLASSIFICATION_METRICS,
      )
    # pylint: enable=protected-access
    else:
      raise ValueError(f'Unknown loss function {loss_fn}.')

  def init_from_train_state(
      self,
      train_state: Any,
      restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict,
  ) -> Any:
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
    return init_vit_from_train_state(
        train_state, restored_train_state, self.config, restored_model_cfg
    )


def init_vit_from_train_state(
    train_state: Any,
    restored_train_state: Any,
    model_cfg: ml_collections.ConfigDict,
    restored_model_cfg: ml_collections.ConfigDict,
) -> Any:
  """Updates the init_params with data from restored_params."""
  del restored_model_cfg
  params_dict = flax.traverse_util.flatten_dict(
      flax.core.unfreeze(train_state.params), sep='/'
  )
  restored_params_dict = flax.traverse_util.flatten_dict(
      flax.core.unfreeze(restored_train_state.params), sep='/'
  )
  del restored_train_state
  # Copy parameters over:
  for pname, pvalue in restored_params_dict.items():
    if 'output_projection' in pname or 'head' in pname:
      # Don't copy the output projection weight and bias from the checkpoint.
      continue
    elif 'MAPHead' in pname and model_cfg.model.classifier != 'map':
      # Skip the MapHead parameters.
      continue
    elif 'pos_embedding' in pname:
      params_dict[pname] = resample_posemb(pvalue, params_dict[pname])
    else:
      params_dict[pname] = pvalue

  logging.info('Inspect missing keys from the restored params:\n%s',
               params_dict.keys() - restored_params_dict.keys())
  logging.info('Inspect extra keys the the restored params:\n%s',
               restored_params_dict.keys() - params_dict.keys())

  # Restore data format, then initialize embeddings
  params = flax.traverse_util.unflatten_dict(params_dict, sep='/')
  return train_state.replace(params=flax.core.freeze(params))


def resample_posemb(old, new):
  """Resampling posemb to finetune a ViT on different resolutions."""
  # Rescale the grid of position embeddings. Param shape is (1,N,1024)
  if old.shape == new.shape:
    return old

  logging.info('ViT: resize %s to %s', old.shape, new.shape)
  gs_old = int(np.sqrt(old.shape[1]))
  gs_new = int(np.sqrt(new.shape[1]))
  logging.info('ViT: grid-size from %s to %s', gs_old, gs_new)
  grid = old.reshape(gs_old, gs_old, -1)

  zoom = (gs_new / gs_old, gs_new / gs_old, 1)
  grid = scipy.ndimage.zoom(grid, zoom, order=1)
  grid = grid.reshape(1, gs_new * gs_new, -1)
  return jnp.array(grid)
