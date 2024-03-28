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

"""Implementation of PCT segmentation model."""

import functools
from typing import Any, Dict, Optional, Tuple

import flax.linen as nn
from flax.training import common_utils
from immutabledict import immutabledict
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils
from scenic.model_lib.base_models.segmentation_model import SegmentationModel
from scenic.projects.pointcloud.models import PointCloudTransformerEncoder


def point_count(logits: jnp.ndarray,
                one_hot_targets: jnp.ndarray,
                weights: Optional[jnp.ndarray] = None) -> float:
  """Computes number of pixels in the target to be used for normalization.

  It needs to have the same API as other defined metrics.

  Args:
    logits: Unused.
    one_hot_targets: Targets, in form of one-hot vectors.
    weights: Input weights (can be used for accounting the padding in the
      input).

  Returns:
    Number of (non-padded) pixels in the input.
  """
  del logits
  if weights is None:
    return np.prod(one_hot_targets.shape[:3])
  assert weights.ndim == 2, (
      'For segmentation task, the weights should be a point level mask.')
  return weights.sum()  # pytype: disable=bad-return-type  # jax-ndarray


# Standard default metrics for the semantic segmentation models.
_POINTCLOUD_SEGMENTATION_METRICS = immutabledict({
    'accuracy': (model_utils.weighted_correctly_classified, point_count),

    # The loss is already normalized, so we set num_pixels to 1.0:
    'loss': (model_utils.weighted_softmax_cross_entropy, lambda *a, **kw: 1.0)
})


def semantic_segmentation_metrics_function(
    logits: jnp.ndarray,
    batch: base_model.Batch,
    target_is_onehot: bool = False,
    metrics: base_model
    .MetricNormalizerFnDict = _POINTCLOUD_SEGMENTATION_METRICS,
) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
  """Calculates metrics for the semantic segmentation task.

  Currently we assume each metric_fn has the API:
    ```metric_fn(logits, targets, weights)```
  and returns an array of shape [batch_size]. We also assume that to compute
  the aggregate metric, one should sum across all batches, then divide by the
  total samples seen. In this way we currently only support metrics of the 1/N
  sum f(inputs, targets). Note, the caller is responsible for dividing by
  the normalizer when computing the mean of each metric.

  Args:
   logits: Output of model in shape [batch, length, num_classes].
   batch: Batch of data that has 'label' and optionally 'batch_mask'.
   target_is_onehot: If the target is a one-hot vector.
   metrics: The semantic segmentation metrics to evaluate. The key is the name
    of the metric, and the value is the metrics function.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """
  if target_is_onehot:
    one_hot_targets = batch['label']
  else:
    one_hot_targets = common_utils.onehot(batch['label'], logits.shape[-1])
  weights = batch.get('batch_mask')  # batch_mask might not be defined

  # This psum is required to correctly evaluate with multihost. Only host 0
  # will report the metrics, so we must aggregate across all hosts. The psum
  # will map an array of shape [n_global_devices, batch_size] -> [batch_size]
  # by summing across the devices dimension. The outer sum then sums across the
  # batch dim. The result is then we have summed across all samples in the
  # sharded batch.
  evaluated_metrics = {}
  for key, val in metrics.items():
    evaluated_metrics[key] = model_utils.psum_metric_normalizer(  # pytype: disable=wrong-arg-types  # jax-ndarray
        (val[0](logits, one_hot_targets, weights),  # pytype: disable=wrong-arg-types  # jax-types
         val[1](logits, one_hot_targets, weights)))  # pytype: disable=wrong-arg-types  # jax-types
  return evaluated_metrics


class PointCloudTransformerSegmentation(nn.Module):
  """Point Cloud Transformer Segmentation Model."""
  in_dim: int
  feature_dim: int
  self_attention: str = 'standard'
  kernel_size: Optional[int] = 1
  num_class: Optional[int] = 50
  dropout_rate: Optional[float] = 0.5
  attention_fn_configs: Optional[Dict[Any, Any]] = None
  use_attention_masking: Optional[bool] = False
  use_knn_mask: Optional[bool] = False
  nearest_neighbour_count: Optional[int] = 256
  mask_function: Optional[str] = 'linear'

  @nn.compact
  def __call__(self,
               inputs,
               cls_label=None,
               train: bool = False,
               debug: bool = False):

    _, num_points, _ = inputs.shape

    # [B, N, D]
    if self.self_attention == 'standard':
      pointwise_features = PointCloudTransformerEncoder(
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
    max_features = jnp.max(pointwise_features, axis=1, keepdims=True)
    max_features = jnp.repeat(max_features, repeats=num_points, axis=1)
    # Mean Pooling
    mean_features = jnp.mean(pointwise_features, axis=1, keepdims=True)
    mean_features = jnp.repeat(mean_features, repeats=num_points, axis=1)

    # concatenate along feature dim
    global_features = jnp.concatenate(
        [pointwise_features, max_features, mean_features],
        axis=-1)

    if cls_label is not None:
      # class label features
      cls_label_feature = jnp.expand_dims(cls_label, axis=1)
      cls_label_feature = nn.Conv(
          self.feature_dim // 2,
          kernel_size=(self.kernel_size, self.kernel_size),
          use_bias=True)(cls_label_feature)
      cls_label_feature = nn.BatchNorm(use_running_average=not train)(
          cls_label_feature)
      cls_label_feature = nn.leaky_relu(cls_label_feature, negative_slope=0.2)
      cls_label_feature = jnp.repeat(
          cls_label_feature, repeats=num_points, axis=1)
      global_features = jnp.concatenate([global_features, cls_label_feature],
                                        axis=-1)

    # LBR Block 1
    output = nn.Conv(4 * self.feature_dim,
                     kernel_size=(self.kernel_size, self.kernel_size),
                     use_bias=True)(global_features)
    output = nn.BatchNorm(use_running_average=not train)(output)
    output = nn.leaky_relu(output, negative_slope=0.2)
    output = nn.Dropout(
        rate=self.dropout_rate, deterministic=not train)(output)
    # LBR Block 2 w/o dropout
    output = nn.Conv(2 * self.feature_dim,
                     kernel_size=(self.kernel_size, self.kernel_size),
                     use_bias=True)(output)
    output = nn.BatchNorm(use_running_average=not train)(output)
    output = nn.leaky_relu(output, negative_slope=0.2)
    # Classification head
    output = nn.Dense(self.num_class, use_bias=True)(output)
    return output


class PointCloudTransformerSegmentationModel(SegmentationModel):
  """Implemets the PCT model for part segmentation."""

  def build_flax_model(self) -> nn.Module:
    return PointCloudTransformerSegmentation(
        in_dim=self.config.in_dim,
        feature_dim=self.config.feature_dim,
        kernel_size=self.config.kernel_size,
        num_class=self.config.dataset_configs.num_classes,
        dropout_rate=self.config.dropout_rate,
        attention_fn_configs=self.config.attention_fn_configs,
        use_attention_masking=self.config.use_attention_masking,
        use_knn_mask=self.config.attention_masking_configs.use_knn_mask,
        nearest_neighbour_count=self.config.attention_masking_configs
        .nearest_neighbour_count,
        mask_function=self.config.attention_masking_configs.mask_function
        )

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return _get_default_configs_for_testing()

  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      batch)```
    """
    del split  # For all splits, we return the same metric functions.
    return functools.partial(
        semantic_segmentation_metrics_function,
        target_is_onehot=self.dataset_meta_data.get('target_is_onehot', False),
        metrics=_POINTCLOUD_SEGMENTATION_METRICS)


def _get_default_configs_for_testing() -> ml_collections.ConfigDict:
  return ml_collections.ConfigDict(
      dict(
          in_dim=3,
          feature_dim=128,
          kernel_size=1,
          sequence_length=2048,
      ))
