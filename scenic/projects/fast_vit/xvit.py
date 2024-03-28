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

"""X-ViT model.

Todo(dehghani, yitay): Write a paper on the results of XViT.
"""

from typing import Any, Optional

import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models.multilabel_classification_model import MultiLabelClassificationModel
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
from scenic.model_lib.layers import nn_ops
from scenic.projects.baselines import vit
from scenic.projects.fast_vit import model_utils


class Encoder1D(nn.Module):
  """XViT 1D Encoder.

  Attributes:
    mlp_dim: Dimension of the MLP on top of attention block.
    num_layers: Number of layers.
    attention_configs: Configurations passed to the self-attention.
    attention_fn: Self-attention function used in the model.
    dropout_rate:  Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    num_kernel_features: Number of kernel features used.
    redraw: Whether to redraw (applicabl only if random features are used).
  """
  mlp_dim: int
  num_layers: int
  attention_configs: ml_collections.ConfigDict
  attention_fn: str
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  num_kernel_features: int = 256
  redraw: bool = True

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool,
  ) -> jnp.ndarray:
    """Applies Transformer model on the inputs.

    Args:
      inputs: Input data.
      train: If it is training.

    Returns:
    """
    assert inputs.ndim == 3  # Shape is `[batch, len, emb]`.
    x = attention_layers.Add1DPositionEmbedding(
        posemb_init=nn.initializers.normal(stddev=0.02),  # From BERT.
        name='posembed_input')(
            inputs)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input encoder.
    for lyr in range(self.num_layers):
      x = model_utils.Encoder1DBlock(  # pytype: disable=wrong-arg-types  # jax-ndarray
          mlp_dim=self.mlp_dim,
          attention_fn=self.attention_fn,
          attention_configs=self.attention_configs,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          num_kernel_features=self.num_kernel_features,
          redraw=self.redraw,
          name=f'encoder_block_{lyr}')(
              x, inputs_kv=None, deterministic=not train)

    return nn.LayerNorm(name='encoder_norm')(x)


class Encoder1DPyramid(nn.Module):
  """Pyramid Transformer Encoder.

  Attributes:
    mlp_dim: Dimension of the MLP on top of attention block.
    num_layers: Number of layers.
    attention_configs: Configurations passed to the self-attention.
    attention_fn: Self-attention function used in the model.
    transformer_encoder_configs: Transformer encoder configurations.
    dropout_rate:  Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    classifier: Type of classifier used.
  """
  mlp_dim: int
  num_layers: int
  attention_configs: ml_collections.ConfigDict
  attention_fn: str
  transformer_encoder_configs: ml_collections.ConfigDict
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  classifier: str = 'gap'

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool,
  ) -> jnp.ndarray:
    """Applies Transformer model on the inputs in Pyramid style.

    Args:
      inputs: Input data.
      train: If it is training.

    Returns:
      Output of a transformer encoder.
    """
    assert inputs.ndim == 3  # Shape is `[batch, len, emb]`.
    x = attention_layers.Add1DPositionEmbedding(
        posemb_init=nn.initializers.normal(stddev=0.02),  # From BERT.
        name='posembed_input')(
            inputs)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    pyramid_lens = self.transformer_encoder_configs['pyramid_lens']
    pyramid_mode = self.transformer_encoder_configs['pyramid_mode']
    pyramid_attn_func = self.transformer_encoder_configs.get(
        'pyramid_attn_func', [])

    if len(pyramid_lens) != self.num_layers:
      raise ValueError('Does not support pyramid lens!=num_layers!')

    # Input encoder.
    for lyr in range(self.num_layers):
      if not pyramid_attn_func:
        layer_attn_func = self.attention_fn
      else:
        assert len(pyramid_attn_func) == self.num_layers
        layer_attn_func = pyramid_attn_func[lyr]

      # TODO(yitay) Support upsampling.
      target_down_scale = abs(pyramid_lens[lyr])
      if target_down_scale != 1:  # Only scale if scaling is more than 1.

        if pyramid_mode == 'pool':
          length = x.shape[1]
          assert length % target_down_scale == 0, (
              f'Current length ({length}) must be divisible by scale '
              f'({target_down_scale})')
          x = nn.max_pool(
              x, (target_down_scale,),
              strides=(target_down_scale,),
              padding='VALID')

        elif pyramid_mode == '2d_pool':
          # Here `-n` means we down-sample by (n, n) on the grid.
          if self.classifier == 'token':
            x_cls_token, x = x[:, :1, :], x[:, 1:, :]
          else:
            x_cls_token, x = x[:, :0, :], x[:, 0:, :]

          bs, l, d = x.shape
          window_size = int(np.sqrt(l))
          assert window_size * window_size == l
          x = x.reshape(bs, window_size, window_size, d)
          assert window_size % target_down_scale == 0, (
              f'Current window size ({window_size}) must be divisible by scale '
              f'({target_down_scale})')

          x = nn.max_pool(
              x, (target_down_scale, target_down_scale),
              strides=(target_down_scale, target_down_scale),
              padding='VALID')
          x = x.reshape(bs, -1, d)
          x = jnp.concatenate([x_cls_token, x], axis=1)

        elif pyramid_mode == 'cnn':
          # Here `-n` means we down-sample by (n, n) on the grid
          if self.classifier == 'token':
            x_cls_token, x = x[:, :1, :], x[:, 1:, :]
          else:
            x_cls_token, x = x[:, :0, :], x[:, 0:, :]

          bs, l, d = x.shape
          window_size = int(np.sqrt(l))
          assert window_size * window_size == l
          x = x.reshape(bs, window_size, window_size, d)
          assert window_size % target_down_scale == 0, (
              f'Current window size ({window_size}) must be divisible by scale '
              f'({target_down_scale})')
          x = nn.Conv(
              x.shape[-1],  # TODO(dehghani): make this configurable
              (target_down_scale, target_down_scale),
              strides=(target_down_scale, target_down_scale),
              padding='VALID')(
                  x)
          x = x.reshape(bs, -1, d)
          x = jnp.concatenate([x_cls_token, x], axis=1)

        elif pyramid_mode == 'memory':
          # TODO(yitay) Inducing set point method in Set Transformer.
          raise NotImplementedError

        elif pyramid_mode == 'linformer':
          # Linformer based downsampling.
          cur_length = x.shape[-2]
          assert cur_length % target_down_scale == 0, (
              f'Current length ({cur_length}) must be divisible by scale '
              f'({target_down_scale})')
          new_length = cur_length // target_down_scale
          x = nn.LayerNorm(name=f'pool_norm_{lyr}')(x)
          x = model_utils.LinformerEncoderAttention(
              num_heads=self.attention_configs['num_heads'],
              kernel_init=nn.initializers.xavier_uniform(),
              broadcast_dropout=False,
              proj_mode='linear',
              low_rank_features=new_length,
              downsample=True,
              dropout_rate=self.attention_dropout_rate)(
                  x, deterministic=not train)

          # Input encoder.
      x = model_utils.Encoder1DBlock(  # pytype: disable=wrong-arg-types  # jax-ndarray
          mlp_dim=self.mlp_dim,
          attention_fn=layer_attn_func,
          attention_configs=self.attention_configs,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoder_block_{lyr}')(
              x, inputs_kv=None, deterministic=not train)

    return nn.LayerNorm(name='encoder_norm')(x)


class EncoderAxial(nn.Module):
  """Transformer Axial Encoder.

  EncoderAxial replaces each Transformer block, i.e. `self-attention + MLP` with
  two axial attention block: `[self-row-attention + MLP] + [self-col-attention
  + MLP]`.
  Attributes:
    mlp_dim: Dimension of the MLP on top of attention block.
    num_layers: Number of layers.
    transformer_encoder_type: Type of the transformer encoder, one of
      ['axial', 'axial_attention'].
    attention_configs: Configurations passed to the self-attention.
    attention_fn: Self-attention function used in the model.
    dropout_rate:  Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
  """
  mlp_dim: int
  num_layers: int
  transformer_encoder_type: ml_collections.ConfigDict
  attention_configs: ml_collections.ConfigDict
  attention_fn: str
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, train: bool = False) -> jnp.ndarray:
    """Applies Axial Transformer model on the inputs.

    Args:
      inputs: Input data.
      train: If it is training.

    Returns:
      Output of a transformer encoder.
    """
    assert inputs.ndim == 4  # Shape is `[batch, h, w, emb]`.
    x = attention_layers.Add2DPositionEmbedding(
        posemb_init=nn.initializers.normal(stddev=0.02),  # From BERT.
        name='posembed_input')(
            inputs)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder
    for lyr in range(self.num_layers):
      if self.transformer_encoder_type == 'axial':
        two_d_shape = x.shape
        # Row attention.
        x = model_utils.get_axial_1d_input(x, axis=1)
        x = model_utils.Encoder1DBlock(  # pytype: disable=wrong-arg-types  # jax-ndarray
            mlp_dim=self.mlp_dim,
            attention_fn=self.attention_fn,
            attention_configs=self.attention_configs,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            name=f'encoder_block_row_{lyr}')(
                x, inputs_kv=None, deterministic=not train)
        x = model_utils.get_axial_2d_input(x, axis=1, two_d_shape=two_d_shape)

        # Column attention.
        x = model_utils.get_axial_1d_input(x, axis=2)
        x = model_utils.Encoder1DBlock(  # pytype: disable=wrong-arg-types  # jax-ndarray
            mlp_dim=self.mlp_dim,
            attention_fn=self.attention_fn,
            attention_configs=self.attention_configs,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            name=f'encoder_block_col_{lyr}')(
                x, inputs_kv=None, deterministic=not train)
        x = model_utils.get_axial_2d_input(x, axis=2, two_d_shape=two_d_shape)

      elif self.transformer_encoder_type == 'axial_attention':
        x = model_utils.EncoderAxialBlock(
            mlp_dim=self.mlp_dim,
            attention_fn=self.attention_fn,
            attention_configs=self.attention_configs,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            name=f'axial_encoder_fblock_{lyr}')(
                x, deterministic=not train)

      else:
        raise ValueError('Undefined transformer encoder type: '
                         f'{self.transformer_encoder_type}.')

    return nn.LayerNorm(x, name='encoder_norm')  # type: ignore  # jnp-type


class XViT(nn.Module):
  """XViT model.

  Attributes:
    num_outputs: number of classes.
    mlp_dim: Dimension of the MLP on top of attention block.
    num_layers: Number of layers.
    attention_configs: Configurations passed to the self-attention.
    attention_fn: Self-attention function used in the model.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden dimension on the stem of the model.
    fast_vit: Configurations of the fast_vit (omnidirectional attention).
    representation_size: Size of the final representation.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout rate for attention heads.
    classifier:  Type of the classifier.
  """
  num_outputs: int
  mlp_dim: int
  num_layers: int
  attention_configs: ml_collections.ConfigDict
  attention_fn: ml_collections.ConfigDict
  patches: ml_collections.ConfigDict
  hidden_size: int
  transformer_encoder_configs: ml_collections.ConfigDict
  representation_size: Optional[int] = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  classifier: str = 'gap'

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               train: bool,
               debug: Optional[bool] = False) -> jnp.ndarray:
    """X-ViT model.

    Args:
      inputs: Input data.
      train: If it is training.
      debug: If we are running the model in the debug mode.

    Returns:
      Output of the model.
    """
    _, height, width, _ = inputs.shape
    patch_height, patch_width = self.patches.size
    grid_height, grid_width = height // patch_height, width // patch_width
    patch_stride_height, patch_stride_width = self.patches.get(
        'strides', self.patches.size)
    x = nn.Conv(
        self.hidden_size, (patch_height, patch_width),
        strides=(patch_stride_height, patch_stride_width),
        padding='VALID',
        name='embedding')(
            inputs)

    transformer_encoder_type = self.transformer_encoder_configs.type
    if transformer_encoder_type in ['global', 'pyramid']:
      bs, h, w, c = x.shape
      x = jnp.reshape(x, [bs, h * w, c])
      if self.classifier == 'token':
        # Only when we do flattening here, we can add the extra CLS token, in
        #  other cases, we use the first token as CLS token.
        cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
        cls = jnp.tile(cls, [bs, 1, 1])
        x = jnp.concatenate([cls, x], axis=1)

    if transformer_encoder_type == 'global':
      x = Encoder1D(
          mlp_dim=self.mlp_dim,
          num_layers=self.num_layers,
          attention_configs=self.attention_configs,
          attention_fn=self.attention_fn,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name='Transformer')(
              x, train=train)
      cls_token = x[:, 0]

    elif transformer_encoder_type == 'pyramid':
      if self.transformer_encoder_configs['pyramid_mode'] in ['2d_pool', 'cnn']:
        assert grid_height == grid_width, (
            'For now, only square grid is supported for pyramid with 2d '
            'pooling or cnn pooling.')
      x = Encoder1DPyramid(
          mlp_dim=self.mlp_dim,
          num_layers=self.fnum_layers,
          attention_configs=self.attention_configs,
          attention_fn=self.attention_fn,
          transformer_encoder_configs=self.transformer_encoder_configs,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          classifier=self.classifier,
          name='Transformer')(
              x, train=train)
      cls_token = x[:, 0]

    elif transformer_encoder_type in [
        'axial', 'axial_attention', 'grid', 'grid_attention'
    ]:
      if transformer_encoder_type in ['grid', 'grid_attention']:
        # First put patches of patches in rows. (we already projected pixels to
        # patches in the stem and at this level, the input tokens are patches).
        # TODO(dehghani): Check if nn_ops.extract_image_patches is faster.
        pp_size = self.transformer_encoder_configs.patches_of_patches_size
        x = nn_ops.extract_patches(lhs=x, rhs_shape=pp_size, strides=pp_size)
        # TODO(dehghani): Check if we can output a 4D tensor directly and get
        #  rid of this reshape.
        bs, ph, pw, h, w, c = x.shape
        x = x.reshape(bs, ph * pw, h * w, c)
        # Now, we simply need to run axial/axial_attention to get
        # inter-patch and intra-patch attention.
        if transformer_encoder_type == 'grid':
          transformer_encoder_type = 'axial'
        elif transformer_encoder_type == 'grid_attention':
          transformer_encoder_type = 'axial_attention'

      x = EncoderAxial(
          mlp_dim=self.mlp_dim,
          num_layers=self.num_layers,
          transformer_encoder_type=transformer_encoder_type,
          attention_configs=self.attention_configs,
          attention_fn=self.attention_fn,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name='Transformer')(
              x, train=train)
      cls_token = x[:, 0, 0]

    else:
      raise ValueError(
          f'Transformer encoder type {transformer_encoder_type} is not defined!'
      )

    if self.classifier in ('token', '0'):
      x = cls_token

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
        self.num_outputs,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(
            x)
    return x


class XViTMultiLabelClassificationModel(MultiLabelClassificationModel):
  """X-ViT model for multi-label classification task."""

  def build_flax_model(self):
    return XViT(
        num_outputs=self.dataset_meta_data['num_classes'],
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        attention_configs=self.config.model.attention_configs,
        attention_fn=self.config.model.attention_fn,
        patches=self.config.model.patches,
        hidden_size=self.config.model.hidden_size,
        transformer_encoder_configs=self.config.model
        .transformer_encoder_configs,
        representation_size=self.config.model.representation_size,
        classifier=self.config.model.classifier,
        dropout_rate=self.config.model.get('dropout_rate', 0.),
        attention_dropout_rate=self.config.model.get('attention_dropout_rate',
                                                     0.),
    )

  def default_flax_model_config(self):
    return ml_collections.ConfigDict(
        dict(
            model=dict(
                attention_fn='standard',
                attention_configs={'num_heads': 2},
                transformer_encoder_configs={'type': 'global'},
                num_layers=1,
                representation_size=16,
                mlp_dim=32,
                dropout_rate=0.,
                attention_dropout_rate=0.,
                hidden_size=16,
                patches={'size': (4, 4)},
                classifier='gap',
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
