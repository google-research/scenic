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

"""MBT model for finetuning."""

import functools
import re
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from absl import logging
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.common_lib import debug_utils
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import classification_model
from scenic.model_lib.base_models import model_utils as base_model_utils
from scenic.model_lib.base_models.classification_model import ClassificationModel
from scenic.model_lib.layers import nn_layers
from scenic.projects.av_mae import model_utils as avmae_model_utils
from scenic.projects.baselines import vit
from scenic.projects.mbt import model as mbt_model
from scenic.projects.vivit import model_utils as vivit_utils
import scipy


Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]
PyTree = Any

# pylint: disable=protected-access
_MBT_CLASSIFICATION_METRICS = mbt_model._MBT_CLASSIFICATION_METRICS
_MODALITIES = mbt_model._MODALITIES
# pylint: enable=protected-access


def add_positional_embed(
    x: jnp.ndarray,
    feat_name: str,
    positional_embedding='sinusoidal_1d'):
  """Adds positional embedding."""

  if x.ndim != 3:  # (batch, len, emb)
    raise ValueError(f'Input should be 3 dimensional. Got {x.shape}')
  if positional_embedding != 'sinusoidal_1d':
    raise ValueError('Only sinusoidal_1d embedding is supported!')

  return avmae_model_utils.add_positional_embeddings(
      x, positional_embedding, input_shape=x.shape,
      layer_name=f'posembed_{feat_name}')


def _inflate_with_mean_channel(x):
  """Inflate tensor with an extra mean channel."""
  mean_channel = jnp.mean(x, axis=-1, keepdims=True)
  y = jnp.concatenate([x, mean_channel], axis=-1)
  return y


class Encoder(nn.Module):
  """Transformer Encoder.

  Attributes:
    inputs: nd-array, Input data
    modality_fusion: Tuple with modalities to combine.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of attention heads.
    attention_config: Has parameters for the type of attention.
    dropout_rate: Dropout rate.
    fusion_layer: Which layer to fuse modalities. fusion_layer == 0 provides
      early fusion.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_droplayer_rate: Probability of dropping a layer linearly
      grows from 0 to the provided value. Our implementation of stochastic
      depth follows timm library, which does per-example layer dropping and
      uses independent dropping patterns for each skip-connection.
    use_bottleneck: If True, adds self-attention bottleneck.
    test_with_bottlenecks: Whether to use bottlenecks at test time.
    share_encoder: If True, different modalities share the same encoder weights
      for the layers before fusion.
    add_pos_embedding: If True, positional embeddings are added to the input
      token embeddings.
    return_bottlenecks: If True, return bottleneck embeddings.
  """

  mlp_dim: int
  num_layers: int
  num_heads: int
  attention_config: Optional[ml_collections.ConfigDict] = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_droplayer_rate: float = 0.0
  modality_fusion: Tuple[str] = ('spectrogram',)
  fusion_layer: int = 0
  use_bottleneck: bool = False
  test_with_bottlenecks: bool = True
  share_encoder: bool = False
  add_pos_embedding: bool = True
  return_bottlenecks: bool = False
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: Dict[str, Any],
               bottleneck: jnp.ndarray, *,
               train: bool):
    """Applies Transformer model on the inputs."""

    def get_encoder_block(encoder_block, droplayer_p, name):
      """Returns the encoder block for a single layer."""
      dtype = jax.dtypes.canonicalize_dtype(self.dtype)
      return encoder_block(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          droplayer_p=droplayer_p,
          name=name,
          dtype=dtype)

    def get_context(target_modality, modality_fusion, x):
      """Returns list of context modalities."""
      context = []
      for modality in _MODALITIES:
        if modality != target_modality and modality in modality_fusion:
          context.append(x[modality])
      return context

    def combine_context(x, other_modalities):
      """Combine x with a list of other modalities."""
      t_x = x.shape[1]
      # Append x to the end of the list
      other_modalities.append(x)
      x_combined = jnp.concatenate(other_modalities, axis=1)
      return x_combined, t_x

    assert self.modality_fusion

    # Add positional embeddings
    if self.add_pos_embedding:
      for modality in self.modality_fusion:
        if modality == 'spectrogram':
          modality_name = 'spec'
        else:
          modality_name = modality
        if modality == 'rgb':
          name = ''
        else:
          name = '_' + modality_name
        x[modality] = add_positional_embed(x[modality], 'posembed_input' + name)

    if self.attention_config is None or self.attention_config.type in [  # pytype: disable=attribute-error
        'spacetime', 'factorized_encoder'
    ]:
      encoder_block = mbt_model.EncoderBlock
    else:
      raise ValueError(f'Unknown attention type {self.attention_config.type}')  # pytype: disable=attribute-error

    use_bottlenecks = train or self.test_with_bottlenecks
    x_combined = None
    # Input Encoder
    for lyr in range(self.num_layers):
      droplayer_p = (
          lyr / max(self.num_layers - 1, 1)) * self.stochastic_droplayer_rate
      encoders = {}
      first_modality = self.modality_fusion[0]
      encoders[first_modality] = get_encoder_block(encoder_block, droplayer_p,
                                                   f'encoderblock_{lyr}')
      for modality in self.modality_fusion:
        # This is important for loading old checkpoints, where we used 'spec'
        if modality == 'spectrogram':
          modality_name = 'spec'
        else:
          modality_name = modality
        if modality != first_modality:
          if self.share_encoder:
            encoders[modality] = encoders[first_modality]
          else:
            encoders[modality] = get_encoder_block(
                encoder_block, droplayer_p,
                f'encoderblock_{lyr}_' + modality_name)

      if (lyr < self.fusion_layer or len(self.modality_fusion) == 1 or
          (self.use_bottleneck and not use_bottlenecks)):
        for modality in self.modality_fusion:
          x[modality] = encoders[modality](x[modality], deterministic=not train)
      else:
        if self.use_bottleneck:
          bottle = []
          for modality in self.modality_fusion:
            t_mod = x[modality].shape[1]
            in_mod = jnp.concatenate([x[modality], bottleneck], axis=1)
            out_mod = encoders[modality](in_mod, deterministic=not train)
            x[modality] = out_mod[:, :t_mod]
            bottle.append(out_mod[:, t_mod:])
          bottleneck = jnp.mean(jnp.stack(bottle, axis=-1), axis=-1)
        else:
          if not self.share_encoder and len(self.modality_fusion) > 1:
            x_new = {}
            for modality in self.modality_fusion:
              other_modalities = get_context(modality, self.modality_fusion, x)
              combined_mods, t = combine_context(x[modality], other_modalities)
              combined_mods = encoders[modality](
                  combined_mods, deterministic=not train)
              x_new[modality] = combined_mods[:, -t:]
            x = x_new

          elif self.share_encoder and len(self.modality_fusion) > 1:
            if x_combined is None:
              x_combined = []
              for modality in self.modality_fusion:
                x_combined.append(x[modality])
              x_combined = jnp.concatenate(x_combined, axis=1)
            x_combined = encoders[first_modality](
                x_combined, deterministic=not train)
    if x_combined is not None:
      x_out = x_combined
    else:
      x_out = []
      for modality in self.modality_fusion:
        x_out.append(x[modality])
      x_out = jnp.concatenate(x_out, axis=1)
    encoded = nn.LayerNorm(name='encoder_norm')(x_out)

    if self.return_bottlenecks:
      assert self.use_bottleneck, ("`use_bottleneck' should be True to return "
                                   "bottlenecks")
      return encoded, bottleneck

    return encoded


class ProjectionHeadAndClassifier(nn.Module):
  """Projection Head and Classifier for MBT.

  Attributes:
    inputs: nd-array, Input data
    classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
      'token', 'onetoken'
    num_classes: Number of output classes.
    modality_fusion: Tuple with modalities to combine.
    representation_size: Size of the representation layer in the model's head.
      if None, we skip the extra projection + tanh activation at the end.
    num_layers: Number of extra projection + tanh activation layers before
    projection. Must set representation_size.
    return_prelogits: If true, return the final representation of the network
      before the classification head. Useful when using features for a
      downstream task.
    dtype: JAX data type for activations.
  """

  classifier: str
  num_classes: int
  modality_fusion: Tuple[str]
  representation_size: Optional[int] = None
  num_layers: int = 1  # default set for backwards compatibility
  return_prelogits: bool = False
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: Dict[str, Any],
               temporal_dims: Dict[str, Any],
               *,
               train: bool):
    """Applies projection and classifier on the inputs."""
    if self.num_layers > 1:
      assert (self.representation_size
              is not None), 'Please provide representation_size'
    x_out = {}
    counter = 0
    if self.classifier in ['onetoken', 'token', '0']:
      # Obtaining the CLS tokens for each modality.
      # Note when self.classifier is 'onetoken', counter remains 0.
      for modality in self.modality_fusion:
        x_out[modality] = x[:, counter]
        counter += temporal_dims[modality] + 1
    elif self.classifier in ('gap', 'gmp', 'gsp'):
      fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
      # Note here we pool each modality separately
      for modality in self.modality_fusion:
        modality_tokens = x[:, counter:counter + temporal_dims[modality]]
        x_out[modality] = fn(
            modality_tokens, axis=list(range(1, modality_tokens.ndim - 1)))
        counter += temporal_dims[modality]

    if self.representation_size is not None:
      for layer in range(self.num_layers):
        if self.num_layers == 1:  # backward compatibility with previous models
          name = 'pre_logits'
        else:
          name = 'pre_logits_fc_{}'.format(layer)
        pre_logits_fc = nn.Dense(
            self.representation_size, name=name)
        if isinstance(x_out, dict):
          for modality in x_out:
            x_out[modality] = pre_logits_fc(x_out[modality])
            x_out[modality] = nn.tanh(x_out[modality])
        else:
          x_out = pre_logits_fc(x_out)
          x_out = nn.tanh(x_out)
    else:
      if not isinstance(x_out, dict):
        x_out = nn_layers.IdentityLayer(name='pre_logits')(x_out)

    if self.return_prelogits:
      return x_out
    if isinstance(x_out, dict):
      output_projection_fc = nn.Dense(
          self.num_classes,
          kernel_init=nn.initializers.zeros,
          name='output_projection')
      x_pool = 0
      for modality in x_out:
        x_out[modality] = output_projection_fc(x_out[modality])
        x_pool += x_out[modality]
      x_pool /= len(x_out)
      # We always use the average CLS logits during inference.
      if not train:
        return x_pool
    else:
      x_out = nn.Dense(
          self.num_classes,
          kernel_init=nn.initializers.zeros,
          name='output_projection')(
              x_out)
      logging.info('Shape of final logits is %s', x_out.shape)
    return x_out


class MBT(nn.Module):
  """Audio-Visual Fusion Transformer model for Video.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_classes: Number of output classes.
    modality_fusion: Tuple with modalities to combine.
    fusion_layer: Which layer to fuse modalities.
    num_heads: Number of self-attention heads.
    num_layers: Number of layers.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    representation_size: Size of the representation layer in the model's head.
      if None, we skip the extra projection + tanh activation at the end.
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
    return_prelogits: If true, return the final representation of the network
      before the classification head. Useful when using features for a
      downstream task.
    return_preclassifier: If true, return a dict of all token embeddings.
      Useful when using token embeddings for a downstream task.
    return_as_dict: If true, return the token embeddings as a dictionary instead
      of a concatenated tensor.
    use_bottleneck: If True, adds self-attention bottleneck.
    n_bottlenecks: Number of bottleneck tokens.
    test_with_bottlenecks: Whether to use bottlenecks at test time.
    share_encoder: If True, different modalities share the same encoder weights
      for the layers before fusion.
    return_bottlenecks: If True, return bottleneck embeddings.
    use_modality_tokens: If True, modality tokens are used.
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
  representation_size: Optional[int] = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_droplayer_rate: float = 0.
  classifier: str = 'gap'
  modality_fusion: Tuple[str] = ('spectrogram',)
  fusion_layer: int = 0
  return_prelogits: bool = False
  return_preclassifier: bool = False
  return_as_dict: bool = True
  use_bottleneck: bool = False
  n_bottlenecks: int = 4
  test_with_bottlenecks: bool = True
  share_encoder: bool = False
  return_bottlenecks: bool = False
  use_modality_tokens: bool = False
  dtype: Any = jnp.float32

  def _temporal_encode(self, x: Dict[str, jnp.ndarray]):
    temporal_dims = {}
    is_single_modal = 'size' in self.patches
    for modality in self.modality_fusion:
      patch = self.patches if is_single_modal else self.patches[modality]
      if modality == 'flow':
        # Inflate from 2 channels to 3 channels with the mean of the first two.
        x[modality] = _inflate_with_mean_channel(x[modality])
      x[modality], _ = mbt_model.temporal_encode(
          x[modality], modality, self.temporal_encoding_config, patch,
          self.hidden_size)
      n, temporal_dims[modality], c = x[modality].shape
      # If we want to add a class token, add it here.
      if self.classifier in ['token']:
        if modality == 'rgb' or len(self.modality_fusion) == 1:
          name = ''
        else:
          name = modality
        cls = self.param('cls'+name, nn.initializers.zeros, (1, 1, c),
                         x[modality].dtype)
        cls = jnp.tile(cls, [n, 1, 1])
        x[modality] = jnp.concatenate([cls, x[modality]], axis=1)
    return x, temporal_dims

  def add_modality_token(self, x_tokens_dict, name: str = 'Encoder'):
    """Add modality learned tokens."""
    for key, x_tokens in x_tokens_dict.items():
      modality_token = self.param(f'{name}_modality_token_{key}',
                                  nn.initializers.zeros,
                                  (1, 1, x_tokens.shape[-1]))
      x_tokens = x_tokens + modality_token
      x_tokens_dict[key] = x_tokens

    return x_tokens_dict

  @nn.compact
  def __call__(self,
               x: Dict[str, jnp.ndarray],
               *,
               train: bool,
               debug: bool = False):
    assert self.fusion_layer <= self.num_layers and self.fusion_layer >= 0
    assert self.classifier in ['onetoken', 'token', '0', 'gap', 'gmp', 'gsp']
    attention_type = self.attention_config.get('type', 'spacetime')
    assert attention_type not in [
        'factorized_transformer_block', 'factorized_self_attention_block',
        'factorized_dot_product_attention'
    ], ('Factorised attention is not implemented')

    x, temporal_dims = self._temporal_encode(x)
    if self.use_modality_tokens:
      x = self.add_modality_token(x)

    bottleneck_dtype = x[self.modality_fusion[0]].dtype
    n, _, c = x[self.modality_fusion[0]].shape
    bottleneck = None
    if self.use_bottleneck:
      n_bottlenecks = self.n_bottlenecks
      if self.classifier in ['token']:
        n_bottlenecks += 1
      bottleneck = self.param('bottleneck',
                              nn.initializers.normal(stddev=0.02),  # From BERT.
                              (1, n_bottlenecks, c), bottleneck_dtype)
      bottleneck = jnp.tile(bottleneck, [n, 1, 1])

    token_lengths = {m: x[m].shape[1] for m in self.modality_fusion}
    output = Encoder(
        modality_fusion=self.modality_fusion,
        fusion_layer=self.fusion_layer,
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        attention_config=self.attention_config,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_droplayer_rate=self.stochastic_droplayer_rate,
        use_bottleneck=self.use_bottleneck,
        test_with_bottlenecks=self.test_with_bottlenecks,
        share_encoder=self.share_encoder,
        return_bottlenecks=self.return_bottlenecks,
        dtype=self.dtype,
        name='Transformer')(x, bottleneck, train=train)
    if self.return_bottlenecks:
      x, bottleneck = output
    else:
      x = output
    if self.return_preclassifier:
      if self.return_as_dict:
        x_dict = {}
        for m in self.modality_fusion:
          v = token_lengths[m]
          x_dict[m] = x[:, :v]
          x = x[:, v:]
        x = x_dict
      if self.return_bottlenecks:
        return x, bottleneck
      return x

    x_out = ProjectionHeadAndClassifier(
        classifier=self.classifier,
        num_classes=self.num_classes,
        modality_fusion=self.modality_fusion,
        representation_size=self.representation_size,
        return_prelogits=self.return_prelogits)(
            x, temporal_dims, train=train)
    return x_out


class MBTMultilabelClassificationModel(vit.ViTMultiLabelClassificationModel):
  """Video Transformer model for multi-class classification."""

  def build_flax_model(self) -> nn.Module:
    assert (self.config.model.attention_config.get('type', 'spacetime') !=
            'factorized_encoder'), (
                'Please add support for factorized_encoder for models with '
                'sigmoid loss.')
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    mbt_args = {
        'num_classes':
            self.dataset_meta_data['num_classes'],
        'modality_fusion':
            self.config.model.modality_fusion,
        'fusion_layer':
            self.config.model.fusion_layer,
        'use_bottleneck':
            self.config.model.get('use_bottleneck', False),
        'test_with_bottlenecks':
            self.config.model.get('test_with_bottlenecks', True),
        'n_bottlenecks':
            self.config.model.get('n_bottlenecks', 4),
        'share_encoder':
            self.config.model.get('share_encoder', False),
        'mlp_dim':
            self.config.model.mlp_dim,
        'num_layers':
            self.config.model.num_layers,
        'num_heads':
            self.config.model.num_heads,
        'representation_size':
            self.config.model.representation_size,
        'patches':
            self.config.model.patches,
        'hidden_size':
            self.config.model.hidden_size,
        'temporal_encoding_config':
            self.config.model.temporal_encoding_config,
        'attention_config':
            self.config.model.attention_config,
        'classifier':
            self.config.model.classifier,
        'dropout_rate':
            self.config.model.get('dropout_rate', 0.1),
        'attention_dropout_rate':
            self.config.model.get('attention_dropout_rate', 0.1),
        'stochastic_droplayer_rate':
            self.config.model.get('stochastic_droplayer_rate', 0),
        'return_prelogits':
            self.config.model.get('return_prelogits', False),
        'return_preclassifier':
            self.config.model.get('return_preclassifier', False),
        'dtype':
            model_dtype
    }
    return MBT(**mbt_args)

  def init_from_train_state(self,
                            train_state: Any,
                            restored_train_state: Any,
                            restored_model_cfg: ml_collections.ConfigDict,
                            restore_output_proj: bool = False) -> Any:
    """Updates the train_state with data from restored_train_state."""
    return initialise_from_train_state(
        self.config, train_state, restored_train_state, restored_model_cfg,
        restore_output_proj)

  def loss_function(
      self,
      logits: jnp.ndarray,
      batch: base_model.Batch,
      model_params: Optional[jnp.ndarray] = None,
  ) -> float:
    """Returns sigmoid cross entropy loss with an L2 penalty on the weights.

    Args:
      logits: Output of model in shape [batch, length, num_classes]. Optionally,
        this can also be a dictionary with logits for individual modalities.
      batch: Batch of data that has 'label' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """
    weights = batch.get('batch_mask')
    labels = batch['label']

    assert self.dataset_meta_data.get('target_is_onehot', False)

    label_weights = self.dataset_meta_data.get('class_weights', None)
    if self.config.model.classifier in ['onetoken']:
      if isinstance(labels, dict):
        assert 'all' in labels, 'mixmod must be turned off.'
        labels = labels['all']
      sig_ce_loss = base_model_utils.weighted_sigmoid_cross_entropy(
          logits['onetoken'],
          labels,
          weights,
          label_weights=label_weights,
          label_smoothing=self.config.get('label_smoothing'))
    elif isinstance(logits, dict):
      sig_ce_loss = []
      for modality in logits:
        sig_ce_loss.append(base_model_utils.weighted_sigmoid_cross_entropy(
            logits[modality],
            labels[modality],
            weights,
            label_weights=label_weights,
            label_smoothing=self.config.get('label_smoothing')))
      sig_ce_loss = jnp.mean(jnp.array(sig_ce_loss))
    else:
      if isinstance(labels, dict):
        assert 'all' in labels, 'mixmod must be turned off.'
        labels = labels['all']
      sig_ce_loss = base_model_utils.weighted_sigmoid_cross_entropy(
          logits,
          labels,
          weights,
          label_weights=label_weights,
          label_smoothing=self.config.get('label_smoothing'))
    if self.config.get('l2_decay_factor') is None:
      total_loss = sig_ce_loss
    else:
      l2_loss = base_model_utils.l2_regularization(model_params)
      total_loss = sig_ce_loss + 0.5 * self.config.l2_decay_factor * l2_loss
    return total_loss  # pytype: disable=bad-return-type  # jax-ndarray


class MBTClassificationModel(ClassificationModel):
  """Audio Video Transformer model for n-way classification."""

  def build_flax_model(self) -> nn.Module:
    assert (self.config.model.attention_config.get('type', 'spacetime') !=
            'factorized_encoder'), (
                'Other attention types not supported.')
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return MBT(
        num_classes=self.dataset_meta_data['num_classes'],
        modality_fusion=self.config.model.modality_fusion,
        fusion_layer=self.config.model.fusion_layer,
        use_bottleneck=self.config.model.get('use_bottleneck', False),
        test_with_bottlenecks=self.config.model.get(
            'test_with_bottlenecks', True),
        n_bottlenecks=self.config.model.get('n_bottlenecks', 4),
        share_encoder=self.config.model.get('share_encoder', False),
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        representation_size=self.config.model.representation_size,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        temporal_encoding_config=self.config.model.temporal_encoding_config,
        attention_config=self.config.model.attention_config,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.1),
        stochastic_droplayer_rate=self.config.model.get(
            'stochastic_droplayer_rate', 0),
        return_prelogits=self.config.model.get('return_prelogits', False),
        return_preclassifier=self.config.model.get(
            'return_preclassifier', False),
        use_modality_tokens=self.config.model.get(
            'use_modality_tokens', False),
        dtype=model_dtype)

  def loss_function(self,
                    logits: jnp.ndarray,
                    batch: base_model.Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Returns softmax cross entropy loss with an L2 penalty on the weights.

    Args:
      logits: Output of model in shape [batch, length, num_classes]. Optionally,
        this can also be a dictionary with logits for individual modalities.
      batch: Batch of data that has 'label' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """
    weights = batch.get('batch_mask')
    labels = batch['label']

    assert self.dataset_meta_data.get('target_is_onehot', False)

    if isinstance(logits, dict):
      sof_ce_loss = []
      for modality in logits:
        sof_ce_loss.append(base_model_utils.weighted_softmax_cross_entropy(
            logits[modality],
            labels[modality],
            weights,
            label_smoothing=self.config.get('label_smoothing')))
      sof_ce_loss = jnp.mean(jnp.array(sof_ce_loss))
    else:
      sof_ce_loss = base_model_utils.weighted_softmax_cross_entropy(
          logits,
          labels,
          weights,
          label_smoothing=self.config.get('label_smoothing'))
    if self.config.get('l2_decay_factor') is None:
      total_loss = sof_ce_loss
    else:
      l2_loss = base_model_utils.l2_regularization(model_params)
      total_loss = sof_ce_loss + 0.5 * self.config.l2_decay_factor * l2_loss
    return total_loss  # pytype: disable=bad-return-type  # jax-ndarray

  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one
        of the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      label, weights)```
    """
    del split  # for all splits, we return the same metric functions

    return functools.partial(
        classification_model.classification_metrics_function,
        target_is_onehot=self.dataset_meta_data.get('target_is_onehot', False),
        metrics=_MBT_CLASSIFICATION_METRICS)

  def init_from_train_state(self,
                            train_state: Any,
                            restored_train_state: Any,
                            restored_model_cfg: ml_collections.ConfigDict,
                            restore_output_proj: bool = False) -> Any:
    """Updates the train_state with data from restored_train_state."""
    return initialise_from_train_state(
        self.config, train_state, restored_train_state, restored_model_cfg,
        restore_output_proj)


class MBTMultiHeadClassificationModel(MBTClassificationModel):
  """Audio Visual Transformer model for multiple n-way classification."""

  def __init__(self, config, dataset_meta_data):
    super().__init__(config, dataset_meta_data)

    assert self.config.dataset_configs.get('class_splits'), (
        'dataset_configs.class_splits must be specified')
    self.class_splits = np.cumsum(self.config.dataset_configs.class_splits)
    if self.config.dataset_configs.get('split_names'):
      self.split_names = self.config.dataset_configs.split_names
    else:
      self.split_names = [str(x + 1) for x in range(len(self.class_splits))]

    assert not config.get('multicrop_softmax_logits', False), (
        'Returning softmaxed logits during multicrop evaluation is not '
        'supported for this model.')

  def loss_function(self,
                    logits: jnp.ndarray,
                    batch: base_model.Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Return softmax cross entropy loss with an L2 penalty on the weights."""
    weights = batch.get('batch_mask')
    labels = batch['label']

    assert self.dataset_meta_data.get('target_is_onehot', False)
    if not isinstance(logits, dict):
      all_logits = logits
      logits = {}
      logits['all'] = all_logits
      if isinstance(labels, dict):
        assert 'all' in labels, 'mixmod must be turned off.'
        labels = labels['all']
      else:
        all_labels = labels
        labels = {}
        labels['all'] = all_labels

    sof_ce_loss = []
    for modality in logits:
      if logits[modality].shape[-1] != self.class_splits[-1]:
        raise AssertionError(
            'Logit dimension must be equal to number of classes')

      logit_splits = jnp.split(logits[modality],
                               self.class_splits, axis=-1)[:-1]
      assert not isinstance(labels[modality], dict), labels.keys()
      labels_splits = jnp.split(
          labels[modality], self.class_splits, axis=-1)[:-1]
      label_smoothing = self.config.get('label_smoothing')

      sof_ce_losses = [
          base_model_utils.weighted_softmax_cross_entropy(
              logit_split, labels_split, weights, label_smoothing)
          for logit_split, labels_split in zip(logit_splits, labels_splits)
      ]
      sof_ce_loss.append(jnp.mean(jnp.array(sof_ce_losses)))
    sof_ce_loss = jnp.mean(jnp.array(sof_ce_loss))

    if self.config.get('l2_decay_factor') is None:
      total_loss = sof_ce_loss
    else:
      l2_loss = base_model_utils.l2_regularization(model_params)
      total_loss = sof_ce_loss + 0.5 * self.config.l2_decay_factor * l2_loss
    return total_loss  # pytype: disable=bad-return-type  # jnp-type

  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one
        of the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      label, weights)```
    """
    del split  # for all splits, we return the same metric functions

    def classification_metrics_function(logits, batch, metrics, class_splits,
                                        split_names):

      one_hot_targets = batch['label']
      weights = batch.get('batch_mask')  # batch_mask might not be defined

      logit_splits = jnp.split(logits, class_splits, axis=-1)[:-1]
      one_hot_target_splits = jnp.split(
          one_hot_targets, class_splits, axis=-1)[:-1]

      evaluated_metrics = {}
      total_loss = [0.0, 0.0]
      for logits_i, one_hot_targets_i, name in zip(logit_splits,
                                                   one_hot_target_splits,
                                                   split_names):
        for key, val in metrics.items():
          evaluated_metrics[
              f'{name}_{key}'] = base_model_utils.psum_metric_normalizer(
                  (val[0](logits_i, one_hot_targets_i,
                          weights), val[1](logits_i, one_hot_targets_i,
                                           weights)))
          if key == 'loss':
            total_loss[0] += evaluated_metrics[f'{name}_{key}'][0]
            total_loss[1] += evaluated_metrics[f'{name}_{key}'][1]
      evaluated_metrics['total_loss'] = total_loss

      if len(class_splits) == 2:
        pairwise_acc = base_model_utils.psum_metric_normalizer(
            (vivit_utils.joint_accuracy(logits, one_hot_targets, class_splits,
                                        weights),
             base_model_utils.num_examples(logits, one_hot_targets, weights)))
        pairwise_top_five = base_model_utils.psum_metric_normalizer(
            (vivit_utils.joint_top_k(
                logits, one_hot_targets, class_splits, k=5, weights=weights),
             base_model_utils.num_examples(logits, one_hot_targets, weights)))
        eval_name = f'{split_names[0]}-{split_names[1]}'
        evaluated_metrics[f'{eval_name}_accuracy'] = pairwise_acc
        evaluated_metrics[f'{eval_name}_accuracy_top_5'] = pairwise_top_five

      return evaluated_metrics

    return functools.partial(
        classification_metrics_function,
        metrics=_MBT_CLASSIFICATION_METRICS,
        class_splits=self.class_splits,
        split_names=self.split_names)


def initialise_from_train_state(
    config,
    train_state: Any,
    restored_train_state: Any,
    restored_model_cfg: ml_collections.ConfigDict,
    restore_output_proj: bool,
    mbt_transformer_key: str = 'Transformer',
    log_initialised_param_shapes: bool = True,
    one_config: bool = True,
    prefix_path: Any = None) -> Any:
  # TODO(aarnab): Deal with Optax and flax.Optim format train states.
  """Updates the train_state with data from restored_train_state.

  This function is written to be used for 'fine-tuning' experiments. Here, we
  do some surgery to support larger resolutions (longer sequence length) in
  the transformer block, with respect to the learned pos-embeddings.

  Args:
    config: Configurations for the model being updated, or tuple of configs.
    train_state: A raw TrainState for the model.
    restored_train_state: A TrainState that is loaded with parameters/state of a
      pretrained model.
    restored_model_cfg: Configuration of the model from which the
      restored_train_state come from. Usually used for some asserts.
    restore_output_proj: If true, load the final output projection. Set
      to False if finetuning to a new dataset.
    mbt_transformer_key: The key used for storing the subtree in the
      parameters that keeps Transformer weights, that are supposed to be
      initialized from the given pre-trained model.
    log_initialised_param_shapes: If true, print tabular summary of all the
      variables in the model once they have been initialised.
    one_config: If true, we have only a single config. If false, we get a tuple
      of configs in the order [init_config, model_config, dataset_config]. This
      is useful for works that build upon MBT and have different models in their
      config.
    prefix_path: If parameters are in a subtree.

  Returns:
    Updated train_state.
  """
  def _get_optimizer(train_state):
    if hasattr(train_state, 'optimizer'):
      if hasattr(train_state.optimizer, 'target'):
        return train_state.optimizer.target
      else:
        return train_state.optimizer['target']
    else:
      return train_state.params

  # Split up configs
  if one_config:
    init_config = config.init_from
    model_config = config.model
  else:
    init_config, model_config, _ = config

  params = flax.core.unfreeze(_get_optimizer(train_state))
  logging.info('Parameters in the target model are: %s', params.keys())
  restored_params = flax.core.unfreeze(_get_optimizer(restored_train_state))

  if init_config.model_type == 'vit':
    params = initialise_from_vit(params=params,
                                 restored_params=restored_params,
                                 config=config,
                                 restored_model_cfg=restored_model_cfg,
                                 mbt_transformer_key=mbt_transformer_key,
                                 restore_output_proj=restore_output_proj,
                                 prefix_path=prefix_path)

  elif init_config.model_type == 'multimae':
    encoder_strategy = restored_model_cfg.model.encoder_strategy
    if encoder_strategy == 'separate_encoders':
      params = initialise_from_separate_encoders(
          params, restored_params, config, restored_model_cfg)
    elif encoder_strategy == 'separate_encoders_and_concat':
      params = initialise_from_mid_fusion(
          params, restored_params, config, restored_model_cfg)
    elif encoder_strategy in ['concat_and_encode', 'same_encoder']:
      params = initialise_from_same_encoder(
          params, restored_params, config, restored_model_cfg)
    else:
      raise AssertionError(f'Unsupported encoder strategy {encoder_strategy}.')

  elif init_config.model_type == 'mbt':
    for m_key, m_params in restored_params.items():
      logging.info('mkey is: %s', m_key)
      if 'ProjectionHeadAndClassifier' in m_key:
        for tm_key, tm_params in m_params.items():
          if tm_key == 'output_projection':
            if restore_output_proj:
              params[m_key][tm_key] = tm_params
            else:
              logging.info('Skipping output projection in restoring weights')
              pass
          elif tm_key == 'pre_logits':
            if model_config.representation_size is None:
              # We don't have representation_size in the new model, so let's
              # ignore if from the pretained model, in case it has it.
              # Note, removing the key from the dictionary is necessary to
              # prevent obscure errors from the Flax optimizer.
              params.pop(tm_key, None)
            else:
              assert restored_model_cfg.model.representation_size
              params[m_key][tm_key] = tm_params
      else:
        if m_key in params:
          params[m_key] = m_params
        else:
          logging.info('Skipping %s. In restored model but not in target',
                       m_key)
  else:
    raise ValueError(
        f'Type of model initialising from unknown: {init_config.model_type}')

  if log_initialised_param_shapes:
    logging.info('Parameter summary after initialising from train state')
    debug_utils.log_param_shapes(params)
  if hasattr(train_state, 'optimizer'):
    return train_state.replace(
        optimizer=train_state.optimizer.replace(
            target=flax.core.freeze(params)))
  else:
    return train_state.replace(params=flax.core.freeze(params))


def _flatten(x):
  return flax.traverse_util.flatten_dict(x, sep='/')


def _unflatten(x):
  return flax.traverse_util.unflatten_dict(x, sep='/')


def initialise_from_vit(params: PyTree,
                        restored_params: PyTree,
                        config: ml_collections.ConfigDict,
                        restored_model_cfg: ml_collections.ConfigDict,
                        mbt_transformer_key: str,
                        restore_output_proj: bool,
                        prefix_path: Any) -> PyTree:
  """Initialize the parameters from a ViT like model."""
  init_config = config.init_from
  model_config = config.model
  dataset_config = config.dataset_configs

  if prefix_path:
    video_params = params[prefix_path]
  else:
    video_params = params

  # Start moving parameters, one-by-one and apply changes if needed
  for m_key, m_params in restored_params.items():
    if 'ProjectionHeadAndClassifier' in m_key:
      for tm_key, tm_params in m_params.items():
        if tm_key == 'output_projection':
          if restore_output_proj:
            video_params[m_key][tm_key] = tm_params
          else:
            logging.info('Skipping output projection in restoring weights')
            pass
        elif tm_key == 'pre_logits':
          if model_config.representation_size is None:
            # We don't have representation_size in the new model, so let's
            # ignore if from the pretained model, in case it has it.
            # Note, removing the key from the dictionary is necessary to
            # prevent obscure errors from the Flax optimizer.
            video_params.pop(tm_key, None)
          else:
            assert restored_model_cfg.model.representation_size
            video_params[m_key][tm_key] = tm_params

    elif m_key in ['Transformer']:
      for tm_key, tm_params in m_params.items():
        if tm_key == 'posembed_input':  # Might need resolution change
          init_posemb(
              video_params[mbt_transformer_key],
              m_params,
              init_config,
              model_config,
              dataset_config,
              restored_model_cfg,
              'posembed_input',
              prefix_path=prefix_path)
          init_posemb(
              video_params,
              m_params,
              init_config,
              model_config,
              dataset_config,
              restored_model_cfg,
              'bottleneck',
              prefix_path=prefix_path)
          for modality in model_config.modality_fusion:
            if modality == 'spectrogram':
              modality_name = 'spec'
            else:
              modality_name = modality
            name = '_' + modality_name
            init_posemb(
                video_params[mbt_transformer_key],
                m_params,
                init_config,
                model_config,
                dataset_config,
                restored_model_cfg,
                'posembed_input' + name,
                prefix_path=prefix_path)
        elif 'encoderblock' in tm_key:
          logging.info('Loading encoder parameters.')
          init_encoderblock(
              video_params[mbt_transformer_key], m_params,
              tm_key, model_config)
        else:  # Other parameters of the Transformer encoder
          video_params[mbt_transformer_key][tm_key] = tm_params
    elif m_key == 'embedding':
      init_embedding(video_params, m_params, init_config,
                     model_config, 'embedding')
      for modality in model_config.modality_fusion:
        if modality == 'spectrogram':
          modality_name = 'spec'
        else:
          modality_name = modality
        name = '_' + modality_name
        init_embedding(video_params, m_params, init_config,
                       model_config, 'embedding' + name)

    else:
      mkey_found = False
      if m_key in params:
        video_params[m_key] = m_params
        mkey_found = True
      for modality in model_config.modality_fusion:
        if modality == 'spectrogram':
          modality_name = 'spec'
        else:
          modality_name = modality
        mkey_name = '{}_'.format(m_key) + modality_name
        if mkey_name in params:
          video_params[mkey_name] = m_params
          mkey_found = True
      if not mkey_found:
        logging.info('Skipping %s. In restored model but not in target', m_key)

  return params


def initialise_from_separate_encoders(
    params: PyTree,
    restored_params: PyTree,
    config: ml_collections.ConfigDict,
    restored_model_cfg: ml_collections.ConfigDict) -> PyTree:
  """Initialise MBT parameters from MultiMAE with separate encoders.

  Args:
   params: PyTree of model parameters in the target model.
   restored_params: PyTree of model parameters to restore.
   config: Configuration of the target model.
   restored_model_cfg: Configuration of the restored model.

  Returns:
    Adapted parameters for MBT.
  """

  del restored_model_cfg  # Can be used for asserts.

  flattened_params = _flatten(params)
  flattened_restored = _flatten(restored_params)

  # We need to check if we are restoring RGB only, Spectogram only or
  # RGB and Spectogram
  restore_rgb_spec = False
  restore_rgb_only = False
  restore_spec_only = False

  if len(config.model.modality_fusion) > 2:
    raise AssertionError('Only support 1 or 2 modalities.')
  if len(config.model.modality_fusion) == 2:
    if config.model.modality_fusion[0] != 'rgb':
      raise AssertionError('We assume that rgb is the first listed modality.')
    restore_rgb_spec = True
  elif config.model.modality_fusion[0] == 'spectrogram':
    restore_spec_only = True
  elif config.model.modality_fusion[0] == 'rgb':
    restore_rgb_only = True
  else:
    raise AssertionError(f'Unknown modalities {config.model.modality_fusion}')

  # Use a regex to rename all the transformer variables.
  renamed_params = {}
  for name, value in flattened_restored.items():
    new_name = name
    if restore_rgb_spec or restore_rgb_only:
      new_name = re.sub('Transformer_rgb/encoderblock_([0-9]+)',
                        r'Transformer/encoderblock_\1', new_name)
    if restore_rgb_spec:
      new_name = re.sub('Transformer_spectrogram/encoderblock_([0-9]+)',
                        r'Transformer/encoderblock_\1_spec', new_name)
    if restore_spec_only:
      new_name = re.sub('Transformer_spectrogram/encoderblock_([0-9]+)',
                        r'Transformer/encoderblock_\1', new_name)

    renamed_params[new_name] = value

  # Now handle special cases.
  renamed_params['embedding/bias'] = renamed_params.pop(
      'embedding_rgb/bias')
  renamed_params['embedding/kernel'] = renamed_params.pop(
      'embedding_rgb/kernel')
  renamed_params['embedding_spec/bias'] = renamed_params.pop(
      'embedding_spectrogram/bias')
  renamed_params['embedding_spec/kernel'] = renamed_params.pop(
      'embedding_spectrogram/kernel')

  if ('embedding_spec/kernel' in flattened_params and
      flattened_params['embedding_spec/kernel'].shape[2] == 3 and
      renamed_params['embedding_spec/kernel'].shape[2] == 1):
    value = renamed_params['embedding_spec/kernel']
    value = np.tile(value, [1, 1, 3, 1])
    renamed_params['embedding_spec/kernel'] = value
  else:
    logging.info('Not inflating spectrogram embedding filter.')

  # Note: MAE has separate layer-norms per modality:
  # ie Transformer_rgb/encoder_norm/... and
  # Transformer_spectogram/encoder_norm/...
  # MBT has a single layer-norm. We do not rename any of the MAE layer-norms
  # and keep the standard identity initialisation here.

  # Assign transformed names.
  for name in flattened_params:
    if name in renamed_params:
      if flattened_params[name].shape == renamed_params[name].shape:
        flattened_params[name] = renamed_params[name]
      else:
        logging.warning(
            'Shapes for %s do not match. %s vs %s', name,
            flattened_params[name].shape, renamed_params[name].shape)
    else:
      logging.info('%s in target model not being initialised', name)

  for name in renamed_params:
    if name not in flattened_params:
      logging.info('%s not being restored.', name)

  params = _unflatten(flattened_params)
  return params


def initialise_from_same_encoder(
    params: PyTree,
    restored_params: PyTree,
    config: ml_collections.ConfigDict,
    restored_model_cfg: ml_collections.ConfigDict) -> PyTree:
  """Init MBT from "same_encoder" or "concat_and_encode" pretrained models.

  Because the model has a single encoder "Transformer", we can call the
  "initialise_from_vit" method then add the embeddings for both modalities.

  Args:
   params: PyTree of model parameters in the target model.
   restored_params: PyTree of model parameters to restore.
   config: Configuration of the target model.
   restored_model_cfg: Configuration of the restored model.

  Returns:
    Adapted parameters for MBT.
  """
  if len(config.model.modality_fusion) == 2:
    assert config.model.modality_fusion == ('rgb', 'spectrogram'), (
        'The modality fusion must be "rgb, spectrogram".')

  params = initialise_from_vit(params=params,
                               restored_params=restored_params,
                               config=config,
                               restored_model_cfg=restored_model_cfg,
                               mbt_transformer_key='Transformer',
                               restore_output_proj=False,
                               prefix_path=None)

  # Now handle special cases.
  if 'rgb' in  config.model.modality_fusion:
    params['embedding']['bias'] = restored_params['embedding_rgb']['bias']
    params['embedding']['kernel'] = restored_params['embedding_rgb']['kernel']

  if 'spectrogram' in config.model.modality_fusion:
    params['embedding_spec']['bias'] = (
        restored_params['embedding_spectrogram']['bias'])

    if ('embedding_spec' in params and
        params['embedding_spec']['kernel'].shape[2] == 3 and
        restored_params['embedding_spectrogram']['kernel'].shape[2] == 1):
      value = restored_params['embedding_spectrogram']['kernel']
      value = np.tile(value, [1, 1, 3, 1])
      params['embedding_spec']['kernel'] = value
    else:
      logging.info('Not inflating spectrogram embedding filter.')
      params['embedding_spec']['kernel'] = (
          restored_params['embedding_spectrogram']['kernel'])

  return params


def initialise_from_mid_fusion(
    params: PyTree,
    restored_params: PyTree,
    config: ml_collections.ConfigDict,
    restored_model_cfg: ml_collections.ConfigDict) -> PyTree:
  """Initialise MBT from "separate_encoders_and_concat" pretrained models.

  Here, we convert the model parameter names to the same format as
  "separate_encoders" by replicating the model parameters from the shared part
  to each of the individual encoders.
  And then, we initialise the way we would for a "separate_encoders" model.

  Args:
   params: PyTree of model parameters in the target model.
   restored_params: PyTree of model parameters to restore.
   config: Configuration of the target model.
   restored_model_cfg: Configuration of the restored model.

  Returns:
    Adapted parameters for MBT.
  """
  if (
      restored_model_cfg.model.encoder_strategy
      != 'separate_encoders_and_concat'
  ):
    raise AssertionError('Only support "separate_encoders_and_concat" models.')

  flattened_restored = _flatten(restored_params)

  ## Duplicate the shared parameters to each encoder.
  # First, we need to determine the number of layers
  num_separate_layers = 0
  num_shared_layers = 0
  for name in flattened_restored:
    for modality in restored_model_cfg.masked_feature_loss.target:
      match = re.match(f'Transformer_{modality}/encoderblock_([0-9]+)', name)
      if match:
        layer_id = int(match[1]) + 1
        num_separate_layers = max(num_separate_layers, layer_id)

    # Now, the number of shared layers
    match = re.match('Transformer_concat/encoderblock_([0-9]+)', name)
    if match:
      layer_id = int(match[1]) + 1
      num_shared_layers = max(num_shared_layers, layer_id)

  if num_shared_layers == 0 or num_separate_layers == 0:
    raise AssertionError(
        'num_shared_layers and num_separate_layers should both be > 0.'
        f'Got {num_shared_layers} and {num_separate_layers}')

  # Now, we rename the shared layers.
  flattened_restored_renamed = {}
  for name, value in flattened_restored.items():
    if 'Transformer_concat/' in name:
      if 'Transformer_concat/encoder_norm' in name:
        new_name = name.replace('Transformer_concat/', 'Transformer/')
        flattened_restored_renamed[new_name] = value
        continue

      layer_id = int(re.match(
          'Transformer_concat/encoderblock_([0-9]+)', name)[1])
      new_id = layer_id + num_separate_layers
      name_rgb = name.replace('Transformer_concat', 'Transformer_rgb')
      name_rgb = name_rgb.replace(f'encoderblock_{layer_id}',
                                  f'encoderblock_{new_id}')
      flattened_restored_renamed[name_rgb] = value

      name_spec = name.replace('Transformer_concat', 'Transformer_spectrogram')
      name_spec = name_spec.replace(f'encoderblock_{layer_id}',
                                    f'encoderblock_{new_id}')
      flattened_restored_renamed[name_spec] = value
    else:
      flattened_restored_renamed[name] = value

  restored_params = _unflatten(flattened_restored_renamed)
  logging.info('Restored parameters after renaming:')
  debug_utils.log_param_shapes(restored_params)
  return initialise_from_separate_encoders(
      params, restored_params, config, restored_model_cfg)


def interpolate_positional_embeddings(restored_posemb_grid, n_tokens):
  """Interpolate positional embeddings from one size to another.

  Args:
    restored_posemb_grid: Positional embeddings from restored model. Shape is
      [n_restored_tokens, d]. It is assumed that the restored model used square
      image patches.
    n_tokens: Number of tokens in the target model. Can be a scalar if the
      target image is square, otherwise should be a tuple of 2.

  Returns:
    positional embedding resized to match n_tokens. Shape is [1, n_tokens, d]
  """

  restored_gs = int(np.sqrt(len(restored_posemb_grid)))
  if isinstance(n_tokens, tuple):
    gh, gw = n_tokens
  else:
    if n_tokens == len(restored_posemb_grid):
      # No need to interpolate
      return np.expand_dims(restored_posemb_grid, axis=0)
    gh = int(np.sqrt(n_tokens))
    gw = n_tokens // gh
    assert gh * gw == n_tokens
  logging.info('Resizing grid-size from (%s, %s) to (%s, %s).',
               restored_gs, restored_gs, gh, gw)
  restored_posemb_grid = restored_posemb_grid.reshape(restored_gs, restored_gs,
                                                      -1)
  zoom = (gh / restored_gs, gw / restored_gs, 1)
  restored_posemb_grid = scipy.ndimage.zoom(restored_posemb_grid, zoom, order=1)
  restored_posemb_grid = restored_posemb_grid.reshape(1, gh * gw, -1)
  return restored_posemb_grid


def init_posemb(to_params, from_params, init_config, model_config,
                dataset_config, restored_model_cfg, name, prefix_path=None):
  """Initialize the positional embeddings."""
  if name not in to_params:
    logging.info('No %s in target model', name)
  elif init_config.restore_positional_embedding:
    if name == 'bottleneck':
      posemb = to_params[name]
    else:
      posemb = to_params[name]['pos_embedding']
    restored_posemb = from_params['posembed_input']['pos_embedding']
    if restored_posemb.shape != posemb.shape:
      # Rescale the grid of pos, embeddings.
      # Default parameter shape is (1, N, 768)
      logging.info('Adapting positional embeddings %s from %s to %s',
                   name, restored_posemb.shape, posemb.shape)
      ntok = posemb.shape[1]
      if prefix_path:
        # MBT is part of a larger model
        classifier = restored_model_cfg.mbt.model.classifier
      else:
        classifier = restored_model_cfg.model.classifier
      if classifier == 'token':
        # the first token is the CLS token
        cls_tok = restored_posemb[:, :1]
        restored_posemb_grid = restored_posemb[0, 1:]
      else:
        cls_tok = restored_posemb[:, :0]
        restored_posemb_grid = restored_posemb[0]
      if model_config.classifier == 'token':
        ntok -= 1

      size_change = init_config.positional_embed_size_change
      if name == 'bottleneck':
        restored_posemb_grid = interpolate_positional_embeddings(
            restored_posemb_grid, ntok)
      elif size_change == 'tile':
        restored_posemb_grid = vivit_utils.tile_positional_embeddings(
            restored_posemb_grid, ntok)
      elif size_change in ['resize_tile', 'resize']:
        temp_encoding = model_config.temporal_encoding_config
        if name.find('spec') > -1:
          gh = ((dataset_config.spec_shape[0] *
                 dataset_config.num_spec_frames) //
                model_config.patches.size[0])
          gw = (dataset_config.spec_shape[1] //
                model_config.patches.size[1])
          tokens_per_frame = (gh, gw)
        elif name.find('wave') > -1 or size_change == 'resize':
          tokens_per_frame = ntok
        elif temp_encoding.method == 'temporal_sampling':
          tokens_per_frame = int(ntok / temp_encoding.n_sampled_frames)
        elif temp_encoding.method == '3d_conv':
          # This is for RGB only.
          n_frames = (
              dataset_config.num_frames //
              model_config.patches.size[2])
          tokens_per_frame = ntok // n_frames
        else:
          raise AssertionError(
              f'Unknown temporal encoding {temp_encoding.method}')

        restored_posemb_grid = interpolate_positional_embeddings(
            restored_posemb_grid, tokens_per_frame)
        if size_change == 'resize_tile' and ntok != tokens_per_frame:
          restored_posemb_grid = restored_posemb_grid[0]
          restored_posemb_grid = vivit_utils.tile_positional_embeddings(
              restored_posemb_grid, ntok)
      else:
        raise AssertionError(
            'Unknown positional embedding size changing method')
      # attach the CLS token again
      if model_config.classifier == 'token':
        restored_posemb = jnp.array(
            np.concatenate([cls_tok, restored_posemb_grid], axis=1))
      else:
        restored_posemb = restored_posemb_grid

    if name == 'bottleneck':
      to_params[name] = restored_posemb
    else:
      to_params[name]['pos_embedding'] = restored_posemb
  else:
    logging.info('Not restoring positional encodings from pretrained model')


def init_embedding(to_params, from_params, init_config, model_config, name):
  """Initialize input embedding."""
  if name not in to_params:
    logging.info('No %s in target model', name)
  elif init_config.get('restore_input_embedding', True):
    input_kernel = to_params[name]['kernel']
    restored_kernel = from_params['kernel']
    restored_bias = from_params['bias']

    if input_kernel.shape != restored_kernel.shape:
      kernel_init_method = (
          model_config.temporal_encoding_config.kernel_init_method
      )
      if input_kernel.shape == restored_kernel.shape[1:]:
        # Deflates a ViViT 3D embedder to work with 2D spectrogram inputs.
        restored_kernel = np.mean(restored_kernel, axis=0)
      elif input_kernel.shape[1:] != restored_kernel.shape:
        # Kernel dimensions are [t, c_in, c_out]
        restored_kernel = np.reshape(restored_kernel, input_kernel.shape)
      elif input_kernel.shape[0] == 1:
        # Kernel dimensions are [t, h, w, c_in, c_out]
        restored_kernel = np.expand_dims(restored_kernel, axis=0)
      elif kernel_init_method == 'average_frame_initializer':
        # This corresponds to "filter inflation" in
        # J Carreira and A Zisserman. Quo vadis, action recognition?
        # A new model and the kinetics dataset. CVPR 2017"
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

    to_params[name]['kernel'] = restored_kernel
    to_params[name]['bias'] = restored_bias
  else:
    logging.info('Not restoring input embedding parameters')


def init_encoderblock(to_params, from_params, tm_key, model_config):
  """Initialize encoder_block_parameters."""
  # Explicitly enumerate over the keys in the encoder-block. Don't just
  # assign the dictionary. It is possible for the target model to
  # contain keys that are not in the restored model.
  attention_type = model_config.attention_config.type
  for enc_key in from_params[tm_key].keys():
    if attention_type in [
        'spacetime', 'factorized_encoder', 'factorized_dot_product_attention'
    ]:
      restoring_params = False
      if tm_key in to_params:
        assert enc_key in to_params[tm_key], '%s not in to_params[%s]' % (
            enc_key, tm_key)
        to_params[tm_key][enc_key] = from_params[tm_key][enc_key]
        restoring_params = True
      for modality in model_config.modality_fusion:
        if modality == 'spectrogram':
          modality_name = 'spec'
        else:
          modality_name = modality
        tmkey_name = '{}_'.format(tm_key) + modality_name
        if tmkey_name in to_params:
          assert enc_key in to_params[
              tmkey_name], '%s not in to_params[%s]' % (enc_key, tmkey_name)
          to_params[tmkey_name][enc_key] = from_params[tm_key][enc_key]
          restoring_params = True
      if not restoring_params:
        logging.info('Warning: Not restoring encoder parameters.')

    elif attention_type == 'factorized_transformer_block':
      raise NotImplementedError('Factorized attention not implemented.')
    else:
      raise ValueError(f'Unknown attention type {attention_type}')
