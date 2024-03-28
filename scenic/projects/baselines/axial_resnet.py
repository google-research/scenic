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

"""Implementation of AxialResNet with group norm and weight standardization.

Ported from:
https://arxiv.org/abs/2003.07853
based on:
https://github.com/csrhddlam/axial-deeplab/tree/optimize
"""

from typing import Dict, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models.multilabel_classification_model import MultiLabelClassificationModel
from scenic.model_lib.layers import nn_layers
from scenic.model_lib.layers import nn_ops
from scenic.projects.baselines import bit_resnet


class SelfAttentionWith1DRelativePos(nn.Module):
  """Multi-head dot-product self-attention with reltive positional encodeing.

  Attributes:
    num_heads: Number of attention heads.
  """

  num_heads: int

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies multi-head dot product self-attention on the input data.

    Args:
      x: input of shape `[bs, len, features]`.

    Returns:
      output of shape `[bs, len, features]`.
    """
    bs, input_len, features = x.shape
    if features % (2 * self.num_heads) != 0:
      raise ValueError(
          f'Inputs feature dimension {features} must be divisible by 2 * number '
          f'of heads {2*self.num_heads}.')
    head_features = features // self.num_heads

    # Apply qkv transformation.
    qkv = nn.Dense(
        features=features * 2,
        kernel_init=nn.initializers.normal(stddev=1.0 / features),
        use_bias=False,
        dtype=jnp.float32)(
            x)

    # Normalize.
    qkv = nn.GroupNorm(epsilon=1e-4, name='gn_qkv')(qkv)
    # Split to multi-headed.
    qkv = qkv.reshape(bs, input_len, self.num_heads,
                      head_features * 2).transpose((0, 2, 3, 1))  # To bhdl.
    # Following the reference implementation, we set feature-size per head
    #  of query, key, to half of the feature size per head for value:
    query, key, value = jnp.split(
        qkv, [head_features // 2, head_features], axis=2)

    # Compute relative positional attention logits.
    length = query.shape[-1]  # Shape: `[bs, heads, depth, len]`.
    relative_emb = self.param(
        'relative_pos_emb', nn.initializers.normal(stddev=1.0 / head_features),
        (head_features * 2, 2 * length - 1), jnp.float32)
    relative_pos_emb = jnp.take(
        relative_emb,
        nn_ops.compute_1d_relative_distance(length, length),
        axis=-1)
    relative_pos_emb_q, relative_pos_emb_k, relative_pos_emb_v = jnp.split(
        relative_pos_emb, [head_features // 2, head_features], axis=0)

    # When computing the similarity of keys and queries, we in fact have
    # (Q + P_q)*(K + P_k), which is Q*K + K*P_q + Q*P_k + P_q*P_k.
    # The last term, i.e. P_q*P_k, is fixed for all Q, K, but but we compute
    # the attention logits for other three terms:
    # 1) We attend from content of queries to the relative position of keys,
    # i.e. Q*P_k:
    qr_attn_logits = jnp.einsum('bhdi,dij->bhij', query, relative_pos_emb_k)
    # 2) We attend from content of keys to the relative position of queries,
    # i.e. K*P_q
    kr_attn_logits = jnp.einsum('bhdi,dij->bhij', key, relative_pos_emb_q)
    # 3) We attend from content of queries to the content of keys, i.e. Q*K:
    qk_attn_logits = jnp.einsum('bhdi,bhdj->bhij', query, key)
    # Finally we combine all these attention logits:
    attn_weights = qk_attn_logits + qr_attn_logits + kr_attn_logits

    # Normalize the attention weights with softmax.
    attn_weights = jax.nn.softmax(attn_weights, axis=3).astype(jnp.float32)
    # Weighted sum over values for each query position.
    wv = jnp.einsum('bhij,bhdj->bhdi', attn_weights, value)
    wve = jnp.einsum('bhij,dij->bhdi', attn_weights, relative_pos_emb_v)
    # From bhdi to bihd and smash head dim.

    x = (wv + wve).transpose((0, 3, 1, 2))  # From bhdi to bihd.
    # Smash head dim and back to the original inputs dimensions.
    out = x.reshape(bs, input_len, features)
    return nn.GroupNorm(epsilon=1e-4, name='gn_out')(out)


class AxialSelfAttention(nn.Module):
  """Axial Self Attention module.

  Attributes:
    attention_axis: Axis of the attention
    axial_attention_configs: Configurations of the axial attention.
  """

  attention_axis: int
  axial_attention_configs: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies Axial Self Attention module.

    Args:
      x: Input data.

    Returns:
      Output after axial attention applied on it.
    """
    bs, height, width, channel = x.shape
    if self.attention_axis == 1:  # Row attention.
      x = x.transpose((0, 2, 1, 3)).reshape(bs * width, height, channel)
    elif self.attention_axis == 2:  # Column attention.
      x = x.reshape(bs * height, width, channel)
    else:
      raise ValueError('Only attention over rows or columns is supported.')

    x = SelfAttentionWith1DRelativePos(self.axial_attention_configs.num_heads)(
        x)

    if self.attention_axis == 1:
      return x.reshape((bs, width, height, channel)).transpose((0, 2, 1, 3))
    else:
      return x.reshape((bs, height, width, channel))


class AxialResidualUnit(nn.Module):
  """Bottleneck AxialResNet block.

  Attributes:
    nout: Number of output features.
    axial_attention_configs: Configurations of the axial attention.
    strides: Down-sampling stride.
    bottleneck: If True, the block is a bottleneck block.
  """
  nout: int
  axial_attention_configs: ml_collections.ConfigDict
  strides: Tuple[int, ...] = (1, 1)
  bottleneck: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    features = self.nout
    nout = self.nout * 4 if self.bottleneck else self.nout
    needs_projection = x.shape[-1] != nout or self.strides != (1, 1)
    residual = x
    if needs_projection:
      residual = bit_resnet.StdConv(
          nout, (1, 1), self.strides, use_bias=False, name='conv_proj')(
              residual)
      residual = nn.GroupNorm(epsilon=1e-4, name='gn_proj')(residual)

    if self.bottleneck:
      x = bit_resnet.StdConv(features, (1, 1), use_bias=False, name='conv1')(x)
      x = nn.GroupNorm(epsilon=1e-4, name='gn1')(x)
      x = nn.relu(x)

    # Axial block that is replacing the 3x3 Convs in the ResNet residual unit.
    # Row  attention:
    x = AxialSelfAttention(
        attention_axis=1, axial_attention_configs=self.axial_attention_configs)(
            x)
    # Column attention:
    x = AxialSelfAttention(
        attention_axis=2, axial_attention_configs=self.axial_attention_configs)(
            x)
    if self.strides == (2, 2):
      x = nn.avg_pool(x, (2, 2), strides=(2, 2), padding='SAME')
    x = nn.relu(x)

    last_kernel = (1, 1) if self.bottleneck else (3, 3)
    x = bit_resnet.StdConv(nout, last_kernel, use_bias=False, name='conv3')(x)
    x = nn.GroupNorm(
        epsilon=1e-4, name='gn3', scale_init=nn.initializers.zeros)(
            x)
    x = nn.relu(residual + x)

    return x


class AxialResNetStage(nn.Module):
  """ResNet Stage: one or more stacked ResNet blocks.

  Attributes:
    block_size: Number of ResNet blocks to stack.
    nout: Number of features.
    first_stride: Downsampling stride.
    bottleneck: If True, the bottleneck block is used.
  """

  block_size: int
  nout: int
  axial_attention_configs: ml_collections.ConfigDict
  first_stride: Tuple[int, ...]
  bottleneck: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x = AxialResidualUnit(
        self.nout,
        axial_attention_configs=self.axial_attention_configs,
        strides=self.first_stride,
        bottleneck=self.bottleneck,
        name='unit1')(
            x)
    for i in range(1, self.block_size):
      x = AxialResidualUnit(
          self.nout,
          axial_attention_configs=self.axial_attention_configs,
          strides=(1, 1),
          bottleneck=self.bottleneck,
          name=f'unit{i + 1}')(
              x)
    return x


class AxialResNet(nn.Module):
  """Axial ResNet.

  Attributes:
    num_outputs: Num output classes. If None, a dict of intermediate feature
      maps is returned.
    width_factor: Width multiplier for each of the ResNet stages.
    num_layers: Number of layers (see `BLOCK_SIZE_OPTIONS` for stage
      configurations).
  """
  num_outputs: int
  axial_attention_configs: ml_collections.ConfigDict
  width_factor: int = 1
  num_layers: int = 50

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      *,
      train: bool = True,
      debug: bool = False) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Applies the AxialResNet model to the inputs.

    Args:
      x: Inputs to the model.
      train: Unused.
      debug: Unused.

    Returns:
       Un-normalized logits if `num_outputs` is provided, a dictionary with
       representations otherwise.
    """
    del train
    del debug
    blocks, bottleneck = bit_resnet.BLOCK_SIZE_OPTIONS[self.num_layers]
    width = int(64 * self.width_factor)

    # Root block.
    x = bit_resnet.StdConv(
        width, (7, 7), (2, 2), use_bias=False, name='conv_root')(
            x)
    x = nn.GroupNorm(epsilon=1e-4, name='gn_root')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

    # Stages.
    x = AxialResNetStage(
        blocks[0],
        width,
        axial_attention_configs=self.axial_attention_configs,
        first_stride=(1, 1),
        bottleneck=bottleneck,
        name='block1')(
            x)
    for i, block_size in enumerate(blocks[1:], 1):
      x = AxialResNetStage(
          block_size,
          width * 2**i,
          axial_attention_configs=self.axial_attention_configs,
          first_stride=(2, 2),
          bottleneck=bottleneck,
          name=f'block{i + 1}')(
              x)

    # Head.
    x = jnp.mean(x, axis=(1, 2))
    x = nn_layers.IdentityLayer(name='pre_logits')(x)
    return nn.Dense(
        self.num_outputs,
        kernel_init=nn.initializers.zeros,
        name='output_projection')(
            x)


class AxialResNetMultiLabelClassificationModel(MultiLabelClassificationModel):
  """Implements the AxialResNet model for multi-label classification."""

  def build_flax_model(self) -> nn.Module:
    return AxialResNet(
        num_outputs=self.dataset_meta_data['num_classes'],
        axial_attention_configs=self.config.axial_attention_configs,
        width_factor=self.config.get('width_factor', 1),
        num_layers=self.config.num_layers,
    )

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict(
        dict(
            width_factor=1,
            num_layers=5,
            axial_attention_configs=ml_collections.ConfigDict({'num_heads': 2}),
        ))
