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

"""Backbone for DeformableDETR."""

from typing import List, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from scenic.projects.baselines import resnet


class InputPosEmbeddingSine(nn.Module):
  """Creates sinusoidal positional embeddings for inputs."""

  hidden_dim: int
  dtype: jnp.dtype = jnp.float32
  scale: Optional[float] = None
  temperature: float = 10000

  @nn.compact
  def __call__(self, padding_mask: jnp.ndarray) -> jnp.ndarray:
    """Creates the positional embeddings for transformer inputs.

    This is slightly different from the one used in DETR in that an offset of
    -0.5 is added when calculating `x_embed` and `y_embed`.

    Args:
      padding_mask: Binary matrix with 0 at padded image regions. Shape is
        [batch, height, width]

    Returns:
      Positional embedding for inputs.

    Raises:
      ValueError if `hidden_dim` is not an even number.
    """
    if self.hidden_dim % 2:
      raise ValueError('`hidden_dim` must be an even number.')

    mask = padding_mask.astype(jnp.float32)
    y_embed = jnp.cumsum(mask, axis=1)
    x_embed = jnp.cumsum(mask, axis=2)

    # Normalization:
    eps = 1e-6
    scale = self.scale if self.scale is not None else 2 * jnp.pi
    y_embed = (y_embed - 0.5)/ (y_embed[:, -1:, :] + eps) * scale
    x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * scale

    num_pos_feats = self.hidden_dim // 2
    dim_t = jnp.arange(num_pos_feats, dtype=jnp.float32)
    dim_t = self.temperature**(2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, jnp.newaxis] / dim_t
    pos_y = y_embed[:, :, :, jnp.newaxis] / dim_t
    pos_x = jnp.stack([
        jnp.sin(pos_x[:, :, :, 0::2]),
        jnp.cos(pos_x[:, :, :, 1::2]),
    ],
                      axis=4).reshape(padding_mask.shape + (-1,))
    pos_y = jnp.stack([
        jnp.sin(pos_y[:, :, :, 0::2]),
        jnp.cos(pos_y[:, :, :, 1::2]),
    ],
                      axis=4).reshape(padding_mask.shape + (-1,))

    pos = jnp.concatenate([pos_y, pos_x], axis=3)
    b, h, w = padding_mask.shape
    pos = jnp.reshape(pos, [b, h * w, self.hidden_dim])
    return jnp.asarray(pos, self.dtype)


def mask_for_shape(shape, pad_mask: Optional[jnp.ndarray] = None):
  """Create boolean mask by resizing from given mask or set all True."""
  bs, h, w, _ = shape
  if pad_mask is None:
    resized_pad_mask = jnp.ones((bs, h, w), dtype=jnp.bool_)
  else:
    resized_pad_mask = jax.image.resize(
        pad_mask.astype(jnp.float32), shape=[bs, h, w],
        method='nearest').astype(jnp.bool_)
  return resized_pad_mask


class DeformableDETRBackbone(nn.Module):
  """Backbone CNN for multi-scale feature extraction for DeformableDETR.

  Attributes:
    num_filters: Number of Resnet filters.
    num_layers: Number of Resnet layers.
    embed_dim: Position embedding dimension.
    num_feature_levels: Number of feature levels to output.
    dtype: Data type of the computation (default: float32).
  """
  embed_dim: int
  num_filters: int
  num_layers: int
  num_feature_levels: int
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      train: bool = False,
      *,
      padding_mask: Optional[jnp.ndarray] = None,
      update_batch_stats: Optional[bool] = None) -> Sequence[List[jnp.ndarray]]:
    """Perform multi-scale feature extraction, padding and position embedding.

    Args:
      inputs: [bs, h, w, c] input data.
      train:  Whether it is training.
      padding_mask: [bs, h, w] of bools with 0 at padded image regions.
      update_batch_stats: Whether update the batch statistics for the BatchNorms
        in the backbone. if None, the value of `train` flag will be used, i.e.
        we update the batch stat if we are in the train mode.

    Returns:
      Output: Features [bs, h, w, c], pad masks [bs, h, w], and
        position_embeddings [bs, h * w, embed_dim] each in a list ordered by
        scale ordered from smallest to largest stride (highest resolution at
        fist index).
    """
    assert 0 < self.num_feature_levels < 4
    if update_batch_stats is None:
      update_batch_stats = train

    backbone_features = resnet.ResNet(
        num_outputs=None,
        num_filters=self.num_filters,
        num_layers=self.num_layers,
        dtype=self.dtype,
        name='resnet')(
            inputs, train=update_batch_stats)

    # Highest resolution first strides=[8, 16, 32]
    feature_keys = ['stage_2', 'stage_3', 'stage_4'][-self.num_feature_levels:]
    backbone_features = [backbone_features[k] for k in feature_keys]

    # Interpolate pad_mask for each feature level.
    pad_masks = [
        mask_for_shape(x.shape, padding_mask) for x in backbone_features
    ]
    pos_embeds = [
        InputPosEmbeddingSine(hidden_dim=self.embed_dim)(m) for m in pad_masks
    ]

    return backbone_features, pad_masks, pos_embeds
