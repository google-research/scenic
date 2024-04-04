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

"""Implementation of PCT model layers."""

from typing import Any

import flax.linen as nn
from flax.linen.initializers import zeros
import jax
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models.multilabel_classification_model import MultiLabelClassificationModel
from scenic.projects.performer import performer

NUM_FT_PARAMS_PER_HEAD = 25
NUM_POINT_DIMS = 3


def _pairwise_distances(coordinates):
  xx = jnp.sum(coordinates**2, axis=-1)
  xy = jnp.einsum('...xy,...zy->...xz', coordinates, coordinates)

  x2 = jnp.expand_dims(xx, axis=1)
  y2 = jnp.expand_dims(xx, axis=1)
  sq_distances = x2 + y2 - 2 * xy
  return sq_distances


def top_k_hot(distances, k):
  """Returns the one-hot mask for k-nearest neighbours."""
  _, idx = jax.lax.top_k(-distances, k)
  tops = jax.nn.one_hot(idx, distances.shape[-1], dtype=jnp.int32)
  return tops.sum(axis=-2)


# The applied attention type is described in attention_fn_configs which
# has the following fields:
#
# <attention_kind>: str; possible values: 'regular', 'performer'
# <performer>: Dict[Any, Any]; a dictionary of the following fields:
#
#   <kernel_transformation>: str; possible values: 'softmax', 'relu', 'quartet'
#   <use_random_projections>: boolean
#   <num_fetures>: int
#   <seed>: int
#   <rpe_method>: str
#   <num_realizations>: int
#   <num_sines>: int
#   <favor_type>: str; possible values: 'favorplus', 'favorplusplus',
#                                       'favorsharp'
#   <masking_type>: str; possible values: 'nomask', 'fftmasked', 'sharpmasked'


class SelfAttentionLayer(nn.Module):
  """Self Attention Layer."""
  in_channels: Any
  out_channels: Any
  key_channels: Any | None = None
  kernel_size: int | None = 1
  mask_function: str | None = 'linear'
  attention_type: str | None = 'naive'
  attention_fn_configs: dict[Any, Any] | None = None

  @nn.compact
  def __call__(self,
               inputs,
               coords=None,
               mask=None,
               numerical_stabilizer=1e-9,
               train: bool = False):
    """Applies self attention on the input data.

    Args:
      inputs: Input tensor of shape [batch_size, num_points, feature_dim]
      coords: Input tensor of point positions of the shape [batch_size,
        num_points, 3]
      mask: Binary array of shape broadcastable to `inputs` tensor, indicating
        the positions for which self attention should be computed.
      numerical_stabilizer: takes into account stability when inputs are zero.
      train: Whether it is training or not.

    Returns:
      output: Tensor of shape [batch_size, num_points, feature_dim]
    """
    key_channels = self.key_channels or self.out_channels
    input_q = nn.Conv(
        key_channels,
        kernel_size=(self.kernel_size,),
        use_bias=True)(
            inputs)
    input_k = nn.Conv(
        key_channels,
        kernel_size=(self.kernel_size,),
        use_bias=True)(
            inputs)
    input_v = nn.Conv(
        self.out_channels,
        kernel_size=(self.kernel_size,),
        use_bias=True)(
            inputs)

    if (
        self.attention_fn_configs is None
        or self.attention_fn_configs['attention_kind'] == 'regular'
    ):
      attention = jnp.einsum('...MC,...NC->...MN', input_q, input_k)
      if mask is not None:
        mask = nn.make_attention_mask(mask, mask)
        mask = mask.squeeze(axis=-3)
        big_neg = jnp.finfo(attention.dtype).min
        attention = jnp.where(mask, attention, big_neg)
      attention = nn.softmax(attention, axis=-1)
      attention = attention / (
          numerical_stabilizer + jnp.sum(attention, axis=1, keepdims=True))
      output = jnp.einsum('...MN,...NC->...NC', attention, input_v)
    else:
      query = jnp.expand_dims(input_q, axis=-2)
      key = jnp.expand_dims(input_k, axis=-2)
      value = jnp.expand_dims(input_v, axis=-2)
      # TODO(kchoro): Include point cloud masking in performer attention
      if self.attention_fn_configs['performer']['masking_type'] == 'nomask':
        output = performer.regular_performer_dot_product_attention(
            query,
            key,
            value,
            kernel_config=self.attention_fn_configs['performer'],
        )
      elif (
          self.attention_fn_configs['performer']['masking_type'] == 'fftmasked'
      ):
        toeplitz_params = self.param(
            'toeplitz_params', zeros, (query.shape[-2], 2 * query.shape[-3] - 1)
        )
        output = performer.masked_performer_dot_product_attention(
            query,
            key,
            value,
            toeplitz_params=toeplitz_params,
            kernel_config=self.attention_fn_configs['performer'],
        )
      elif (
          self.attention_fn_configs['performer']['masking_type']
          == 'sharpmasked'
      ):
        toeplitz_params = self.param(
            'toeplitz_params',
            zeros,
            (query.shape[-2], 5 * NUM_FT_PARAMS_PER_HEAD),
        )
        output = performer.sharp_masked_performer_dot_product_attention(
            query,
            key,
            value,
            coords,
            toeplitz_params=toeplitz_params,
            kernel_config=self.attention_fn_configs['performer'],
        )
      elif (
          self.attention_fn_configs['performer']['masking_type']
          == 'pseudolocal'
      ):
        sigma = self.attention_fn_configs['performer']['sigma']
        base_aniso_matrix = (1.0 / sigma) * jnp.identity(3)
        output = performer.pseudolocal_subquadratic_attention(
            query,
            key,
            value,
            coords,
            aniso_matrix=(base_aniso_matrix),
            rf_type=self.attention_fn_configs['performer']['rf_type'],
            nb_rfs=self.attention_fn_configs['performer']['num_features'],
        )
      elif (
          self.attention_fn_configs['performer']['masking_type']
          == 'pseudolocal_learnable'
      ):
        inner_dim = 3
        sigma = self.attention_fn_configs['performer']['sigma']
        base_aniso_matrix = (1.0 / sigma) * jnp.identity(3)
        aniso_matrix_delta_params = (1.0 / sigma) * self.param(
            'aniso_matrix_delta_params', zeros, (inner_dim, 3)
        )
        output = performer.pseudolocal_subquadratic_attention(
            query,
            key,
            value,
            coords,
            aniso_matrix=(base_aniso_matrix + aniso_matrix_delta_params),
            rf_type=self.attention_fn_configs['performer']['rf_type'],
            nb_rfs=self.attention_fn_configs['performer']['num_features'],
        )
      else:
        raise ValueError(
            'Unsupported masking type: %s'
            % self.attention_fn_configs['performer']['masking_type']
        )
      output = jnp.squeeze(output, axis=-2)

    output = (inputs - output) if self.attention_type == 'offset' else output
    output = nn.Conv(
        self.out_channels,
        kernel_size=(self.kernel_size,),
        use_bias=True,
    )(output)
    output = nn.LayerNorm(reduction_axes=-2)(output)
    output = nn.relu(output)
    return output + inputs


class PointCloudTransformerEncoder(nn.Module):
  """Point Cloud Transformer Encoder."""
  in_dim: int
  feature_dim: int
  kernel_size: int | None = 1
  encoder_feature_dim: int | None = 1024
  num_attention_layers: int | None = 4
  num_pre_conv_layers: int = 2
  num_heads: int | None = 1
  attention_fn_configs: dict[Any, Any] | None = None
  use_attention_masking: bool | None = False
  use_knn_mask: bool | None = False
  nearest_neighbour_count: int | None = 256
  mask_function: str | None = 'linear'
  out_dim: int | None = None

  @nn.compact
  def __call__(
      self,
      inputs,
      mask: jnp.ndarray | None = None,
      train: bool = False,
      debug: bool = False,
      coords: jnp.ndarray | None = None,
  ):
    output = inputs
    if mask is not None and (jnp.ndim(mask) < jnp.ndim(inputs)):
      layer_norm_mask = jnp.expand_dims(mask, axis=-1)
    else:
      layer_norm_mask = mask
    if coords is None:
      coords = inputs
    for _ in range(self.num_pre_conv_layers):
      output = nn.Conv(
          self.feature_dim,
          kernel_size=(self.kernel_size,),
          use_bias=True,
      )(output)
      output = nn.LayerNorm(reduction_axes=-2)(output, mask=layer_norm_mask)

    # Self-attention blocks, input_shape= [B, N, D]
    attention_outputs = []
    for _ in range(self.num_attention_layers):
      output = SelfAttentionLayer(
          in_channels=self.feature_dim,
          key_channels=self.feature_dim,
          out_channels=self.feature_dim,
          attention_fn_configs=self.attention_fn_configs)(
              output, coords, mask=mask)
      attention_outputs.append(output)

    output = jnp.concatenate(attention_outputs, axis=-1)

    output = nn.Conv(
        self.encoder_feature_dim,
        kernel_size=(self.kernel_size,),
        use_bias=True)(output)

    if self.out_dim is not None:
      output = nn.LayerNorm(reduction_axes=-2)(output, mask=layer_norm_mask)
      output = nn.leaky_relu(output, negative_slope=0.2)
      output = nn.Conv(
          self.out_dim,
          kernel_size=(self.kernel_size,),
          use_bias=True)(output)
    return output


class PointCloudTransformerClassifier(nn.Module):
  """Point Cloud Transformer Classifier."""
  in_dim: int
  feature_dim: int
  kernel_size: int | None = 1
  num_classes: int | None = 40
  dropout_rate: float | None = 0.5
  attention_type: str | None = 'standard'
  attention_fn_configs: dict[Any, Any] | None = None
  use_attention_masking: bool | None = False
  use_knn_mask: bool | None = False
  nearest_neighbour_count: int | None = 256
  mask_function: str | None = 'linear'

  @nn.compact
  def __call__(self, inputs, train: bool = False, debug: bool = False):
    if self.attention_type == 'standard':
      output = PointCloudTransformerEncoder(
          in_dim=self.in_dim,
          feature_dim=self.feature_dim,
          kernel_size=self.kernel_size,
          attention_fn_configs=self.attention_fn_configs,
          use_attention_masking=self.use_attention_masking,
          use_knn_mask=self.use_knn_mask,
          nearest_neighbour_count=self.nearest_neighbour_count,
          mask_function=self.mask_function)(
              inputs, train=train, debug=debug)

    # Max Pooling
    output = jnp.max(output, axis=1, keepdims=False)
    # LBR Block 1
    output = nn.Dense(4 * self.feature_dim, use_bias=True)(output)
    output = nn.LayerNorm(reduction_axes=-2)(output)
    output = nn.leaky_relu(output, negative_slope=0.2)
    output = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(output)
    # LBR Block 2
    output = nn.Dense(2 * self.feature_dim, use_bias=True)(output)
    output = nn.LayerNorm(reduction_axes=-2)(output)
    output = nn.leaky_relu(output, negative_slope=0.2)
    output = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(output)
    # Classification head
    output = nn.Dense(self.num_classes, use_bias=True)(output)
    return output


class PointCloudTransformerClassificationModel(MultiLabelClassificationModel):
  """Implements the PCT model for multi-label classification."""

  def build_flax_model(self) -> nn.Module:
    return PointCloudTransformerClassifier(
        in_dim=self.config.in_dim,
        feature_dim=self.config.feature_dim,
        kernel_size=self.config.kernel_size,
        num_classes=self.config.dataset_configs.num_classes,
        dropout_rate=self.config.dropout_rate,
        attention_type=self.config.attention_type,
        attention_fn_configs=self.config.attention_fn_configs,
        use_attention_masking=self.config.use_attention_masking,
        use_knn_mask=self.config.attention_masking_configs.use_knn_mask,
        nearest_neighbour_count=self.config.attention_masking_configs
        .nearest_neighbour_count,
        mask_function=self.config.attention_masking_configs.mask_function)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return _get_default_configs_for_testing()


def _get_default_configs_for_testing() -> ml_collections.ConfigDict:
  return ml_collections.ConfigDict(
      dict(
          in_dim=3,
          feature_dim=128,
          kernel_size=1,
          num_classes=40,
      ))
