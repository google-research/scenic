"""Implementation of PCT model layers."""

from typing import Any, Optional, Dict

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
  kernel_size: Optional[int] = 1
  mask_function: Optional[str] = 'linear'
  attention_type: Optional[str] = 'naive'
  attention_fn_configs: Optional[Dict[Any, Any]] = None

  @nn.compact
  def __call__(self,
               inputs,
               coords=None,
               numerical_stabilizer=1e-9,
               train: bool = False):
    """Applies self attention on the input data.

    Args:
      inputs: Input tensor of shape [batch_size, num_points, feature_dim]
      coords: Input tensor of point positions of the shape [batch_size,
        num_points, 3]
      numerical_stabilizer: takes into account stability when inputs are zero.
      train: Whether it is training or not.

    Returns:
      output: Tensor of shape [batch_size, num_points, feature_dim]
    """
    input_q = nn.Conv(
        self.out_channels,
        kernel_size=(self.kernel_size, self.kernel_size),
        use_bias=True)(
            inputs)
    input_k = nn.Conv(
        self.out_channels,
        kernel_size=(self.kernel_size, self.kernel_size),
        use_bias=True)(
            inputs)
    input_v = nn.Conv(
        self.out_channels,
        kernel_size=(self.kernel_size, self.kernel_size),
        use_bias=True)(
            inputs)

    if self.attention_fn_configs is None or self.attention_fn_configs[
        'attention_kind'] == 'regular':
      attention = jnp.einsum('...MC,...NC->...MN', input_q, input_k)
      attention = nn.softmax(attention, axis=-1)
      attention = attention / (
          numerical_stabilizer + jnp.sum(attention, axis=1, keepdims=True))
      output = jnp.einsum('...MN,...NC->...NC', attention, input_v)
    else:
      query = jnp.expand_dims(input_q, axis=-2)
      key = jnp.expand_dims(input_k, axis=-2)
      value = jnp.expand_dims(input_v, axis=-2)
      if self.attention_fn_configs['performer']['masking_type'] == 'nomask':
        output = performer.regular_performer_dot_product_attention(
            query,
            key,
            value,
            kernel_config=self.attention_fn_configs['performer'])
      elif self.attention_fn_configs['performer'][
          'masking_type'] == 'fftmasked':
        toeplitz_params = self.param('toeplitz_params', zeros,
                                     (query.shape[-2], 2 * query.shape[-3] - 1))
        output = performer.masked_performer_dot_product_attention(
            query,
            key,
            value,
            toeplitz_params=toeplitz_params,
            kernel_config=self.attention_fn_configs['performer'])
      elif self.attention_fn_configs['performer'][
          'masking_type'] == 'sharpmasked':
        toeplitz_params = self.param(
            'toeplitz_params', zeros,
            (query.shape[-2], 5 * NUM_FT_PARAMS_PER_HEAD))
        output = performer.sharp_masked_performer_dot_product_attention(
            query,
            key,
            value,
            coords,
            toeplitz_params=toeplitz_params,
            kernel_config=self.attention_fn_configs['performer'])
      output = jnp.squeeze(output, axis=-2)

    output = (inputs - output) if self.attention_type == 'offset' else output
    output = nn.Conv(
        self.out_channels,
        kernel_size=(self.kernel_size, self.kernel_size),
        use_bias=True)(
            output)
    output = nn.BatchNorm(use_running_average=not train)(output)
    output = nn.relu(output)
    return output + inputs


class PointCloudTransformerEncoder(nn.Module):
  """Point Cloud Transformer Encoder."""
  in_dim: int
  feature_dim: int
  kernel_size: Optional[int] = 1
  encoder_feature_dim: Optional[int] = 1024
  num_attention_layers: Optional[int] = 4
  num_heads: Optional[int] = 1
  attention_fn_configs: Optional[Dict[Any, Any]] = None
  use_attention_masking: Optional[bool] = False
  use_knn_mask: Optional[bool] = False
  nearest_neighbour_count: Optional[int] = 256
  mask_function: Optional[str] = 'linear'

  @nn.compact
  def __call__(self, inputs, train: bool = False, debug: bool = False):
    output = nn.Conv(
        self.feature_dim,
        kernel_size=(self.kernel_size, self.kernel_size),
        use_bias=True)(
            inputs)
    output = nn.BatchNorm(use_running_average=not train)(output)
    output = nn.Conv(
        self.feature_dim,
        kernel_size=(self.kernel_size, self.kernel_size),
        use_bias=True)(
            output)
    output = nn.BatchNorm(use_running_average=not train)(output)

    # Self-attention blocks, input_shape= [B, N, D]
    attention_outputs = []
    for _ in range(self.num_attention_layers):
      output = SelfAttentionLayer(
          in_channels=self.feature_dim,
          out_channels=self.feature_dim,
          attention_fn_configs=self.attention_fn_configs)(
              output, inputs)
      attention_outputs.append(output)

    output = jnp.concatenate(attention_outputs, axis=-1)

    # conv-batchnorm-relu block
    output = nn.Conv(
        self.encoder_feature_dim,
        kernel_size=(self.kernel_size, self.kernel_size),
        use_bias=True)(
            output)
    output = nn.BatchNorm(use_running_average=not train)(output)
    output = nn.leaky_relu(output, negative_slope=0.2)
    return output


class PointCloudTransformerClassifier(nn.Module):
  """Point Cloud Transformer Classifier."""
  in_dim: int
  feature_dim: int
  kernel_size: Optional[int] = 1
  num_classes: Optional[int] = 40
  dropout_rate: Optional[float] = 0.5
  attention_type: Optional[str] = 'standard'
  attention_fn_configs: Optional[Dict[Any, Any]] = None
  use_attention_masking: Optional[bool] = False
  use_knn_mask: Optional[bool] = False
  nearest_neighbour_count: Optional[int] = 256
  mask_function: Optional[str] = 'linear'

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
    output = nn.BatchNorm(use_running_average=not train)(output)
    output = nn.leaky_relu(output, negative_slope=0.2)
    output = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(output)
    # LBR Block 2
    output = nn.Dense(2 * self.feature_dim, use_bias=True)(output)
    output = nn.BatchNorm(use_running_average=not train)(output)
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
