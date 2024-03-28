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

"""The ROI box head in Faster R-CNN.

Modified from
https://github.com/google-research/google-research/blob/master/fvlm/
modeling/heads.py
"""
from typing import Optional
import einops
from flax import linen as nn
import jax
from jax.nn import initializers
import jax.numpy as jnp


class ROIBoxHead(nn.Module):
  """A standard conv + fc + classification/regression box head in Faster R-CNN.

  Attributes:
    num_classes: Number of classes (not including additional "background"
      class).
    conv_dims: Number of filters in each of the conv layer.
    conv_norm: Either None or "BN", "LN", the norm layer after each conv.
    fc_dims: Number of filters in each of the fc layer.
    class_box_regression: if do class specific box regression
    use_zeroshot_cls: if using open-vocabulary classifier
    zs_weight_dim: feature dimension of open-vocabulary classifier
    zs_weight: array in shape (zs_weight_dim, num_classes + 1)
    bias_init_prob: if init bias prob
  """
  num_classes: int
  conv_dims: tuple[int, ...] = (256, 256, 256, 256)
  conv_norm: Optional[str] = None
  fc_dims: tuple[int, ...] = (1024,)
  class_box_regression: bool = True
  add_box_pred_layers: bool = False
  use_zeroshot_cls: bool = False
  zs_weight_dim: int = 512
  zs_weight: Optional[jnp.float32] = None
  norm_temp: float = 50.0
  bias_init_prob: Optional[float] = None

  def __call__(self, roi_features: jnp.ndarray, *,
               training: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
    """The forward logic.

    Args:
      roi_features: Per-roi feature of shape (Batch, T, h, w, C), where T is the
        number of RoIs, h and w are shape of the feature.
      training: Training mode.

    Returns:
      class_outputs of shape (Batch, T, cls + 1)
      box_outputs of shape (Batch, T, cls + 1, 4) or (Batch, T, 4)
    """
    batch_size = roi_features.shape[0]
    # TODO(yuxinw): This reshape is not friendly to pjit. Use nn.vmap.
    roi_features = einops.rearrange(roi_features, 'B T H W C -> (B T) H W C')
    class_outputs, box_outputs = self.predict(roi_features, training=training)

    class_outputs = einops.rearrange(
        class_outputs, '(B T) C -> B T C', B=batch_size)
    if self.class_box_regression:
      box_outputs = einops.rearrange(
          box_outputs, '(B T) (C b) -> B T C b', B=batch_size, b=4)
    else:
      box_outputs = einops.rearrange(
          box_outputs, '(B T) (b) -> B T b', B=batch_size, b=4)
    return class_outputs, box_outputs

  @nn.compact
  def predict(self, roi_features: jnp.ndarray, *,
              training: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Runs the head and returns raw layer outputs."""
    del training
    use_bias = self.conv_norm is None
    x = roi_features
    for dim in self.conv_dims:
      x = nn.Conv(
          features=dim,
          kernel_size=(3, 3),
          kernel_init=initializers.variance_scaling(
              scale=2, mode='fan_out', distribution='normal'),
          use_bias=use_bias,
          bias_init=initializers.zeros,
          padding='same')(
              x)
      if self.conv_norm is not None:
        raise NotImplementedError
      x = jax.nn.relu(x)

    x = einops.rearrange(x, 'N H W C -> N (H W C)')
    for i, dim in enumerate(self.fc_dims):
      x = nn.Dense(
          features=dim,
          kernel_init=initializers.variance_scaling(
              1, mode='fan_in', distribution='uniform'),
          name=f'fc{i+1}')(
              x)
      x = jax.nn.relu(x)

    if self.add_box_pred_layers:
      box_outputs = nn.Dense(
          features=self.fc_dims[-1],
          name='bbox_pred.0',
          kernel_init=initializers.variance_scaling(
              1, mode='fan_in', distribution='uniform'),
          bias_init=initializers.zeros,
      )(x)
      box_outputs = jax.nn.relu(box_outputs)
      box_outputs = nn.Dense(
          features=4,
          name='bbox_pred.2',
          kernel_init=initializers.normal(stddev=0.001),
          bias_init=initializers.zeros,
      )(box_outputs)
    else:
      box_outputs = nn.Dense(
          features=(
              self.num_classes + 1) * 4 if self.class_box_regression else 4,
          name='bbox_pred',
          kernel_init=initializers.normal(stddev=0.001),
          bias_init=initializers.zeros,
      )(x)

    if self.use_zeroshot_cls:
      assert not self.class_box_regression
      class_outputs = nn.Dense(
          features=self.zs_weight_dim,
          name='cls_score.linear',
          kernel_init=initializers.normal(stddev=0.01),
          bias_init=initializers.zeros,
      )(x)
      if self.zs_weight is not None:
        class_outputs = class_outputs / (
            (class_outputs ** 2).sum(axis=0)[None, :] ** 0.5 + 1e-8)  # l2 norm
        class_outputs = self.norm_temp * jnp.dot(class_outputs, self.zs_weight)
    else:
      cls_bias_init = initializers.zeros
      if self.bias_init_prob is not None:
        bias = -jnp.log((1 - self.bias_init_prob) / self.bias_init_prob)
        cls_bias_init = initializers.constant(bias, dtype=jnp.float32)
      class_outputs = nn.Dense(
          features=self.num_classes + 1,
          name='cls_score',
          kernel_init=initializers.normal(stddev=0.01),
          bias_init=cls_bias_init,
      )(x)
    return class_outputs, box_outputs
