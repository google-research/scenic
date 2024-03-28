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

"""OmniNet model."""

from typing import Any, Optional, Sequence

from absl import logging
import flax.linen as nn
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models.multilabel_classification_model import MultiLabelClassificationModel
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import mixer
from scenic.projects.baselines import vit
from scenic.projects.fast_vit import model_utils as fast_vit_model_utils
from scenic.projects.omninet import model_utils


class OmnidirectionalEncoder1D(nn.Module):
  """Omnidirectional Encoder.

  Attributes:
    mlp_dim: Dimension of the MLP on top of attention block.
    num_layers: Number of layers.
    attention_configs: Configurations passed to the self-attention.
    attention_fn: Self-attention function used in the model.
    omninet: Configurations of the omninet (omnidirectional attention).
    dropout_rate:  Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
  """
  mlp_dim: int
  num_layers: int
  attention_configs: ml_collections.ConfigDict
  attention_fn: str
  omninet: ml_collections.ConfigDict
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_droplayer_rate: float = 0.0

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool,
  ) -> jnp.ndarray:
    """Applies Transformer model on the inputs."""
    assert inputs.ndim == 3  # Shape is `[batch, len, emb]`.
    x = vit.AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
        name='posembed_input')(
            inputs)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input encoder.
    partition_layers = []
    skip = False
    for lyr in range(self.num_layers):
      droplayer_p = (
          lyr / max(self.num_layers - 1, 1)) * self.stochastic_droplayer_rate
      if skip and self.omninet.skip_standard:
        logging.info('Skipping vanilla transformer at layer %d', lyr)
        skip = False
      else:
        x = fast_vit_model_utils.Encoder1DBlock(  # pytype: disable=wrong-arg-types  # jax-ndarray
            mlp_dim=self.mlp_dim,
            attention_fn=self.attention_fn,
            attention_configs=self.attention_configs,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            droplayer_p=droplayer_p,
            name=f'encoder_block_{lyr}')(
                x, inputs_kv=None, deterministic=not train)
        partition_layers.append(x)

      if len(partition_layers) == (self.omninet.partition - 1):
        # Every N-1th layer.
        #################### OmniNet ########################
        if self.omninet.query_type == 'full':
          if self.omninet.integrate == 'concat':
            xp = jnp.concatenate(partition_layers, 1)
          elif self.omninet.integrate == 'grid_stack':
            xp = model_utils.grid_restack(partition_layers)
          elif self.omninet.integrate == 'factorized':
            xp = jnp.stack(partition_layers, 1)  # Shape: (bs, L, N, d).
          else:
            raise ValueError(
                f'The integrate type {self.omninet.integrate} is not defined.')

          def _pool(xp):
            if self.omninet.pool == 'last':
              assert self.omninet.integrate == 'concat'
              return xp[:, -inputs.shape[1]:, :]
            elif self.omninet.pool == 'max':
              assert self.omninet.integrate in ['grid_stack', 'factorized']
              if self.omninet.integrate == 'grid_stack':
                return nn.max_pool(
                    xp, (len(partition_layers),),
                    strides=(len(partition_layers),),
                    padding='VALID')
              else:
                return jnp.max(xp, axis=1)
            else:
              raise ValueError(
                  f'The pool type {self.omninet.pool} is not defined.')

          if self.omninet.pool_after_sa:
            post_sa_fn = _pool
          else:
            post_sa_fn = None

          if self.omninet.encoder.type == '1d':
            xp = fast_vit_model_utils.Encoder1DBlock(  # pytype: disable=wrong-arg-types  # jax-ndarray
                mlp_dim=self.omninet.get('mlp_dim', self.mlp_dim),
                attention_fn=self.omninet.attention_fn,
                attention_configs=self.omninet.attention_configs,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                droplayer_p=droplayer_p,
                post_sa_fn=post_sa_fn,
                name=f'omni_encoder_block_{lyr}')(
                    xp, inputs_kv=None, deterministic=not train)
          elif self.omninet.encoder.type == 'factorized':
            if self.omninet.integrate != 'factorized':
              raise ValueError(
                  "omninet.encoder.type is 'factorized', "
                  f'but omninet.integrate is {self.omninet.integrate}.')
            xp = fast_vit_model_utils.EncoderAxialBlock(
                mlp_dim=self.omninet.get('mlp_dim', self.mlp_dim),
                attention_configs=self.omninet.attention_configs,
                attention_fn=self.omninet.attention_fn,
                factorization_axis=self.omninet.encoder.get(
                    'factorization_axis',
                    (1,)),  # Apply attention only in depth.
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                droplayer_p=droplayer_p,
                post_sa_fn=post_sa_fn,
                name=f'omni_encoder_block_{lyr}')(
                    xp, deterministic=not train)
          else:
            raise ValueError(
                f'Unknown omninet.encoder.type: {self.omninet.encoder.type}')

          if not self.omninet.pool_after_sa:
            xp = _pool(xp)
        #################### TopQuery ########################
        elif self.omninet.query_type == 'top':
          if self.omninet.integrate == 'concat':
            xp = jnp.concatenate(partition_layers, 1)
          else:
            raise ValueError

          xp = fast_vit_model_utils.Encoder1DBlock(
              mlp_dim=self.omninet.get('mlp_dim', self.mlp_dim),
              attention_fn=self.omninet.attention_fn,
              attention_configs=self.omninet.attention_configs,
              dropout_rate=self.dropout_rate,
              attention_dropout_rate=self.attention_dropout_rate,
              droplayer_p=droplayer_p,
              name=f'omni_encoder_block_{lyr}')(
                  x, xp, deterministic=not train)
        ########################################################
        x = x + xp
        partition_layers = []
        skip = True

    encoded = nn.LayerNorm(name='encoder_norm')(x)
    return encoded


class OmniNet(nn.Module):
  """OmniNet model.

  Attributes:
      num_classes: number of classes.
      mlp_dim: Dimension of the MLP on top of attention block.
      num_layers: Number of layers.
      attention_configs: Configurations passed to the self-attention.
      attention_fn: Self-attention function used in the model.
      patches: Configuration of the patches extracted in the stem of the model.
      hidden_size: Size of the hidden dimension on the stem of the model.
      omninet: Configurations of the omninet (omnidirectional attention).
      representation_size: Size of the final representation.
      dropout_rate: Dropout rate.
      attention_dropout_rate: Dropout rate for attention heads.
      classifier:  Type of the classifier.
  """
  num_classes: int
  mlp_dim: int
  num_layers: int
  attention_configs: ml_collections.ConfigDict
  attention_fn: ml_collections.ConfigDict
  patches: ml_collections.ConfigDict
  hidden_size: int
  omninet: ml_collections.ConfigDict
  representation_size: Optional[int] = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_droplayer_rate: float = 0.0
  classifier: str = 'gap'

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               train: bool,
               debug: Optional[bool] = False) -> jnp.ndarray:
    """OmniNet model."""
    patch_height, patch_width = self.patches.size
    patch_stride_height, patch_stride_width = self.patches.get(
        'strides', self.patches.size)
    x = nn.Conv(
        self.hidden_size, (patch_height, patch_width),
        strides=(patch_stride_height, patch_stride_width),
        padding='VALID',
        name='embedding')(
            inputs)

    # Flatten the input.
    bs, h, w, c = x.shape
    x = jnp.reshape(x, [bs, h * w, c])

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      cls = jnp.tile(cls, [bs, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    x = OmnidirectionalEncoder1D(
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        attention_configs=self.attention_configs,
        attention_fn=self.attention_fn,
        omninet=self.omninet,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_droplayer_rate=self.stochastic_droplayer_rate,
        name='Transformer')(
            x, train=train)

    if self.classifier in ('token', '0'):
      x = x[:, 0]
    elif self.classifier in ('gap', 'gmp', 'gsp'):
      fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
      x = fn(x, axis=1)

      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))

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


class OmniNetMultiLabelClassificationModel(MultiLabelClassificationModel):
  """OmniNet model for multi-label classification task."""

  def build_flax_model(self):
    return OmniNet(
        num_classes=self.dataset_meta_data['num_classes'],
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        attention_configs=self.config.model.attention_configs,
        attention_fn=self.config.model.attention_fn,
        hidden_size=self.config.model.hidden_size,
        patches=self.config.model.patches,
        representation_size=self.config.model.representation_size,
        classifier=self.config.model.classifier,
        omninet=self.config.model.omninet,
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.1),
    )

  def default_flax_model_config(self):
    return ml_collections.ConfigDict(
        dict(
            model=dict(
                attention_fn='standard',
                attention_configs={'num_heads': 2},
                num_layers=1,
                representation_size=16,
                mlp_dim=32,
                dropout_rate=0.,
                attention_dropout_rate=0.,
                hidden_size=16,
                patches={'size': (4, 4)},
                classifier='gap',
                omninet={
                    'skip_standard': True,
                    'partition': 1,
                    'integrate': 'concat',
                    'layer_type': 'self-attention',
                    'pool': 'last',
                },
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
    return vit.init_vit_from_train_state(train_state, restored_train_state,
                                         self.config, restored_model_cfg)


class OmniMixerEncoder(nn.Module):
  """Omnidirectional Mixer Encoder.

  Attributes:
    num_layers: Number of layers.
    channels_mlp_dim: Hidden dimension of the channel mixing MLP.
    sequence_mlp_dim: Hidden dimension of the token (sequence) mixing MLP.
    omnimixer: Configurations of the omnimixer (omnidirectional mixer).
    dropout_rate: Dropout rate.
    stochastic_depth: The layer dropout rate (= stochastic depth).

  Returns:
    Output after OmniMixer block.
  """
  num_layers: int
  channels_mlp_dim: int
  sequence_mlp_dim: int
  omnimixer: ml_collections.ConfigDict
  dropout_rate: float = 0.0
  stochastic_depth: float = 0.0

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      *,
      train: bool,
  ) -> jnp.ndarray:
    """Applies OmniMixer encoder model on the inputs."""
    partition_layers = []
    skip = False  # First layer should be always a standard mixer layer.
    for lyr in range(self.num_layers):
      if skip and self.omnimixer.skip_standard:
        logging.info('Skipping vanilla mixer at layer %d', lyr)
        skip = False
      else:
        p = (lyr / max(self.num_layers - 1, 1)) * self.stochastic_depth
        x = mixer.MixerBlock(
            channels_mlp_dim=self.channels_mlp_dim,
            sequence_mlp_dim=self.sequence_mlp_dim,
            dropout_rate=self.dropout_rate,
            stochastic_depth=p,
            name=f'mixerblock_{lyr}')(
                x, deterministic=not train)
        partition_layers.append(x)
      if len(partition_layers) == self.omnimixer.partition - 1:
        # Mixing in depth every N-1th layers.
        xp = jnp.stack(partition_layers, 3)  # Shape: (bs, N, d, L).
        xp = attention_layers.MlpBlock(
            mlp_dim=self.omnimixer.depth_mlp_dim,
            dropout_rate=self.dropout_rate,
            activation_fn=nn.gelu,
            name=f'depth_mixing_{lyr}')(
                xp, deterministic=not train)
        if self.omnimixer.pool == 'last':
          xp = xp[:, :, :, -1]  # Shape: (bs, N, d).
        elif self.omnimixer.pool == 'max':
          xp = jnp.max(xp, axis=3)  # Shape: (bs, N, d).
        else:
          raise ValueError(
              f'The pool type {self.omnimixer.pool} is not defined.')
        x += xp
        partition_layers = []
        skip = True
    return nn.LayerNorm(name='encoder_norm')(x)


# TODO(dehghani, yitay): Tune the LR a bit and write a paper for the OmniMixer.
class OmniMixer(nn.Module):
  """OmniMixer model.

  Attributes:
    num_classes: Number of output classes.
    patch_size: Patch size of the stem.
    hidden_size: Size of the hidden state of the output of model's stem.
    num_layers: Number of layers.
    channels_mlp_dim: hidden dimension of the channel mixing MLP.
    sequence_mlp_dim: hidden dimension of the token (sequence) mixing MLP.
    omnimixer: Configurations of the omnimixer (omnidirectional mixer).
    dropout_rate: Dropout rate.
    stochastic_depth: overall stochastic depth rate.
  """
  num_classes: int
  patch_size: Sequence[int]
  hidden_size: int
  num_layers: int
  channels_mlp_dim: int
  sequence_mlp_dim: int
  omnimixer: ml_collections.ConfigDict
  dropout_rate: float = 0.0
  stochastic_depth: float = 0.0

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               *,
               train: bool,
               debug: Optional[bool] = False) -> jnp.ndarray:
    """OmniMixer model."""
    x = nn.Conv(
        self.hidden_size,
        self.patch_size,
        strides=self.patch_size,
        padding='VALID',
        name='embedding')(
            x)
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    x = OmniMixerEncoder(
        num_layers=self.num_layers,
        channels_mlp_dim=self.channels_mlp_dim,
        sequence_mlp_dim=self.sequence_mlp_dim,
        omnimixer=self.omnimixer,
        dropout_rate=self.dropout_rate,
        stochastic_depth=self.stochastic_depth,
        name='omnimixer_encoder')(
            x, train=train)

    # Use global average pooling for our classifier, dim (1,) or (1,2).
    x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))

    x = nn_layers.IdentityLayer(name='pre_logits')(x)
    x = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(
            x)
    return x


class OmniMixerMultiLabelClassificationModel(MultiLabelClassificationModel):
  """OmniMixer model for multi-label classification task."""

  def build_flax_model(self):
    return OmniMixer(
        num_classes=self.dataset_meta_data['num_classes'],
        patch_size=self.config.model.patch_size,
        hidden_size=self.config.model.hidden_size,
        num_layers=self.config.model.num_layers,
        channels_mlp_dim=self.config.model.channels_mlp_dim,
        sequence_mlp_dim=self.config.model.sequence_mlp_dim,
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
        omnimixer=self.config.model.omnimixer,
    )
