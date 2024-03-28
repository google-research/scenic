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

"""MBT: Multimodal Bottleneck Transformers."""

import functools
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from absl import logging
import flax.linen as nn
from flax.linen.linear import default_kernel_init
from immutabledict import immutabledict
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import classification_model
from scenic.model_lib.base_models import model_utils as base_model_utils
from scenic.model_lib.base_models.classification_model import ClassificationModel
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import vit
from scenic.projects.mbt import model_utils

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]

_MBT_CLASSIFICATION_METRICS = immutabledict({
    'accuracy': (base_model_utils.weighted_correctly_classified,
                 base_model_utils.num_examples),
    'accuracy_top_5': (functools.partial(
        base_model_utils.weighted_topk_correctly_classified,
        k=5), base_model_utils.num_examples),
    'loss': (base_model_utils.weighted_unnormalized_softmax_cross_entropy,
             base_model_utils.num_examples)
})

_MODALITIES = ['rgb', 'spectrogram']


def _reshape_to_time_space(x, temporal_dims):
  if x.ndim == 3:
    b, thw, d = x.shape
    assert thw % temporal_dims == 0
    hw = thw // temporal_dims
    x = jnp.reshape(x, [b, temporal_dims, hw, d])
  assert x.ndim == 4
  return x


def embed_2d_patch(x, patches, embedding_dim, name='embedding'):
  """Embedding input patches with 2D conv."""

  assert patches.get('size') is not None, ('patches.size is now the only way'
                                           'to define the patches')
  assert embedding_dim, 'embedding_dim must be specified'
  fh = patches.size[0]
  fw = patches.size[1]
  x = nn.Conv(
      embedding_dim, (fh, fw),
      strides=(fh, fw),
      padding='VALID',
      name=name)(x)

  return x


def embed_3d_patch(x,
                   patches,
                   embedding_dim,
                   kernel_init_method,
                   name='embedding'):
  """Embed 3D input patches into tokens."""

  assert patches.get('size') is not None, 'patches.size must be defined'
  assert len(patches.size) == 3, 'patches.size must have 3 elements'
  assert embedding_dim, 'embedding_dim must be specified'

  fh, fw, ft = patches.size

  if kernel_init_method == 'central_frame_initializer':
    kernel_initializer = model_utils.central_frame_initializer()
    logging.info('Using central frame initializer for input embedding')
  elif kernel_init_method == 'average_frame_initializer':
    kernel_initializer = model_utils.average_frame_initializer()
    logging.info('Using average frame initializer for input embedding')
  else:
    kernel_initializer = default_kernel_init
    logging.info('Using default initializer for input embedding')

  x = nn.Conv(
      embedding_dim, (ft, fh, fw),
      strides=(ft, fh, fw),
      padding='VALID',
      name=name,
      kernel_init=kernel_initializer)(x)

  return x


def temporal_encode(x,
                    modality,
                    temporal_encoding_config,
                    patches,
                    hidden_size,
                    return_1d=True):
  """Encode video for feeding into ViT."""
  if modality == 'spectrogram':
    # Spectrogram is treated as a big num_time_bins by num_mel_bins image.
    x = embed_2d_patch(x, patches, hidden_size, 'embedding_spectrogram')
    temporal_dims = 1
    if return_1d:
      n, h, w, c = x.shape
      x = jnp.reshape(x, [n, h * w, c])
  elif modality == 'rgb':
    if temporal_encoding_config.method == 'temporal_sampling':
      n, num_frames, in_h, in_w, c = x.shape
      n_sampled_frames = temporal_encoding_config.n_sampled_frames
      if n_sampled_frames < num_frames:
        t_start_idx = num_frames / (n_sampled_frames + 1)
        t_step = t_start_idx
      else:
        t_start_idx = 0
        t_step = 1
      t_end_idx = num_frames
      temporal_indices = jnp.arange(t_start_idx, t_end_idx, t_step)
      temporal_indices = jnp.round(temporal_indices).astype(jnp.int32)
      temporal_indices = jnp.minimum(temporal_indices, num_frames - 1)

      x = x[:, temporal_indices]  # [n, t_s, in_h, in_w, c]
      t_s = x.shape[1]
      x = jnp.reshape(x, [n, t_s * in_h, in_w, c])
      x = embed_2d_patch(x, patches, hidden_size)
      temporal_dims = t_s
      if return_1d:
        n, th, w, c = x.shape
        x = jnp.reshape(x, [n, th * w, c])
      else:
        n, th, w, c = x.shape
        x = jnp.reshape(x, [n, t_s, -1, w, c])
    if temporal_encoding_config.method == '3d_conv':
      kernel_init_method = temporal_encoding_config.get('kernel_init_method',
                                                        None)

      x = embed_3d_patch(x, patches, hidden_size, kernel_init_method)
      temporal_dims = x.shape[1]
      if return_1d:
        n, t, h, w, c = x.shape
        x = jnp.reshape(x, [n, t * h * w, c])
  else:
    raise AssertionError('Unknown temporal encoding method.')
  return x, temporal_dims


def add_positional_embed(x, feat_name):
  """Adds positional embedding."""
  assert x.ndim == 3  # (batch, len, emb)
  x = vit.AddPositionEmbs(
      posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
      name=feat_name)(x)
  return x


class EncoderBlock(nn.Module):
  """Transformer encoder block.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of heads.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    attention_kernel_initializer: Initializer to use for attention
      layers.
    droplayer_p: Probability of dropping a layer.



  Returns:
    Output after transformer encoder block.
  """
  mlp_dim: int
  num_heads: int
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  attention_kernel_initializer: Initializer = nn.initializers.xavier_uniform()
  droplayer_p: float = 0.0

  def get_drop_pattern(self, x, deterministic):
    if not deterministic and self.droplayer_p:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng('dropout'), self.droplayer_p, shape).astype('float32')
    else:
      return 0.0

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
    """Applies Encoder1DBlock module."""

    # Attention block.
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=self.attention_kernel_initializer,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        dtype=self.dtype)(x, x, deterministic=deterministic)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)

    drop_pattern = self.get_drop_pattern(x, deterministic)
    x = x * (1.0 - drop_pattern) + inputs

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

    drop_pattern = self.get_drop_pattern(x, deterministic)
    return y * (1.0 - drop_pattern) + x


class Encoder(nn.Module):
  """Transformer Encoder.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of attention heads.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_droplayer_rate: Probability of dropping a layer linearly
      grows from 0 to the provided value. Our implementation of stochastic
      depth follows timm library, which does per-example layer dropping and
      uses independent dropping patterns for each skip-connection.
    modality_fusion: Tuple with modalities to combine.
    fusion_layer: Which layer to fuse modalities. fusion_layer == 0 provides
      early fusion.
    use_bottleneck: If True, adds self-attention bottleneck.
    test_with_bottlenecks: Whether to use bottlenecks at test time.
    share_encoder: If True, different modalities share the same encoder weights
      for the layers before fusion.
    dtype: The dtype of the computation (default: float32).
  """
  mlp_dim: int
  num_layers: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_droplayer_rate: float = 0.0
  modality_fusion: Tuple[str] = ('spectrogram',)
  fusion_layer: int = 0
  use_bottleneck: bool = False
  test_with_bottlenecks: bool = True
  share_encoder: bool = False
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
      num_tokens = x.shape[1]
      # Append x to the end of the list
      other_modalities.append(x)
      x_combined = jnp.concatenate(other_modalities, axis=1)
      return x_combined, num_tokens

    assert self.modality_fusion

    # Add positional embeddings
    for modality in self.modality_fusion:
      if modality == 'rgb':
        name = ''
      else:
        name = '_' + modality
      x[modality] = add_positional_embed(x[modality], 'posembed_input' + name)

    use_bottlenecks = train or self.test_with_bottlenecks
    x_combined = None
    # Input Encoder
    for lyr in range(self.num_layers):
      droplayer_p = (
          lyr / max(self.num_layers - 1, 1)) * self.stochastic_droplayer_rate
      encoders = {}
      encoders['rgb'] = get_encoder_block(EncoderBlock, droplayer_p,
                                          f'encoderblock_{lyr}')

      for modality in self.modality_fusion:
        if modality != 'rgb':
          if self.share_encoder:
            encoders[modality] = encoders['rgb']
          else:
            encoders[modality] = get_encoder_block(
                EncoderBlock, droplayer_p,
                f'encoderblock_{lyr}_' + modality)

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
            x_combined = encoders['rgb'](x_combined, deterministic=not train)
    if x_combined is not None:
      x_out = x_combined
    else:
      x_out = []
      for modality in self.modality_fusion:
        x_out.append(x[modality])
      x_out = jnp.concatenate(x_out, axis=1)
    encoded = nn.LayerNorm(name='encoder_norm')(x_out)

    return encoded


class MBT(nn.Module):
  """Audio-Visual Fusion Transformer model for Video.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    num_classes: Number of output classes.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
      if None, we skip the extra projection + tanh activation at the end.
    temporal_encoding_config: ConfigDict which defines the type of input
      encoding when tokenising the video.
    attention_config: ConfigDict which defines the type of spatio-temporal
      attention applied in the model.
    representation_size: Size of the representation layer in the model's head.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_droplayer_rate: Probability of dropping a layer. Linearly
      increases from 0 to the provided value..
    classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
      'token'.
    modality_fusion: Tuple with modalities to combine.
    fusion_layer: Which layer to fuse modalities.
    return_prelogits: If true, return the final representation of the network
      before the classification head. Useful when using features for a
      downstream task.
    return_preclassifier: If true, return a dict of all token embeddings.
      Useful when using token embeddings for a downstream task.
    use_bottleneck: If True, adds self-attention bottleneck.
    n_bottlenecks: Number of bottleneck tokens.
    test_with_bottlenecks: Whether to use bottlenecks at test time.
    share_encoder: If True, different modalities share the same encoder weights
      for the layers before fusion.
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
  use_bottleneck: bool = False
  n_bottlenecks: int = 4
  test_with_bottlenecks: bool = True
  share_encoder: bool = False
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self,
               x,
               *,
               train: bool,
               debug: bool = False):
    assert self.fusion_layer <= self.num_layers and self.fusion_layer >= 0
    assert self.classifier in ['token', '0', 'gap', 'gmp', 'gsp']

    temporal_dims = {}
    for modality in self.modality_fusion:
      x[modality], _ = temporal_encode(
          x[modality], modality, self.temporal_encoding_config, self.patches,
          self.hidden_size)
      # If we want to add a class token, add it here.
      if self.classifier in ['token']:
        if modality == 'rgb' or len(self.modality_fusion) == 1:
          name = ''
        else:
          name = modality
        n, temporal_dims[modality], c = x[modality].shape
        cls = self.param('cls'+name, nn.initializers.zeros, (1, 1, c),
                         x[modality].dtype)
        cls = jnp.tile(cls, [n, 1, 1])
        x[modality] = jnp.concatenate([cls, x[modality]], axis=1)
        bottleneck_dtype = x[modality].dtype

    bottleneck = None
    if self.use_bottleneck:
      n_bottlenecks = self.n_bottlenecks
      if self.classifier in ['token']:
        n_bottlenecks += 1
      bottleneck = self.param('bottleneck',
                              nn.initializers.normal(stddev=0.02),  # From BERT.
                              (1, n_bottlenecks, c), bottleneck_dtype)
      bottleneck = jnp.tile(bottleneck, [n, 1, 1])

    x = Encoder(
        modality_fusion=self.modality_fusion,
        fusion_layer=self.fusion_layer,
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_droplayer_rate=self.stochastic_droplayer_rate,
        use_bottleneck=self.use_bottleneck,
        test_with_bottlenecks=self.test_with_bottlenecks,
        share_encoder=self.share_encoder,
        dtype=self.dtype,
        name='Transformer')(x, bottleneck, train=train)

    if self.return_preclassifier:
      return x

    if self.classifier in ['token', '0']:
      # Obtaining the CLS tokens for each modality.
      x_out = {}
      counter = 0
      for modality in self.modality_fusion:
        x_out[modality] = x[:, counter]
        counter += temporal_dims[modality] + 1
    elif self.classifier in ('gap', 'gmp', 'gsp'):
      fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
      x_out = fn(x, axis=list(range(1, x.ndim - 1)))

    if self.representation_size is not None:
      pre_logits_fc = nn.Dense(self.representation_size, name='pre_logits')
      if isinstance(x_out, dict):
        for modality in x_out:
          x_out[modality] = pre_logits_fc(x_out[modality])
          x_out[modality] = nn.tanh(x_out[modality])
      else:
        x_out = nn.Dense(self.representation_size, name='pre_logits')(x_out)
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
      if not train:
        return x_pool
    else:
      x_out = nn.Dense(
          self.num_classes,
          kernel_init=nn.initializers.zeros,
          name='output_projection')(
              x_out)
    return x_out


class MBTMultilabelClassificationModel(vit.ViTMultiLabelClassificationModel):
  """Video Transformer model for multi-class classification."""

  def build_flax_model(self) -> nn.Module:
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
        dtype=model_dtype)

  def init_from_train_state(self,
                            train_state: Any,
                            restored_train_state: Any,
                            restored_model_cfg: ml_collections.ConfigDict,
                            restore_output_proj: bool = False) -> Any:
    """Updates the train_state with data from restored_train_state."""
    return model_utils.initialise_from_train_state(self.config, train_state,
                                                   restored_train_state,
                                                   restored_model_cfg,
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

    if isinstance(logits, dict):
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
    return total_loss


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
    return total_loss

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
    return model_utils.initialise_from_train_state(self.config, train_state,
                                                   restored_train_state,
                                                   restored_model_cfg,
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
    return total_loss

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
            (model_utils.joint_accuracy(logits, one_hot_targets, class_splits,
                                        weights),
             base_model_utils.num_examples(logits, one_hot_targets, weights)))
        pairwise_top_five = base_model_utils.psum_metric_normalizer(
            (model_utils.joint_top_k(
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
