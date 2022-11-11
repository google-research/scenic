"""Backbone for DeformableDETR."""

from typing import List, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from scenic.projects.baselines import resnet
from scenic.projects.baselines.detr.model import InputPosEmbeddingSine


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
