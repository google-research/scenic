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

r"""Sam prompt encoder.

Pytorch reference:

https://github.com/facebookresearch/segment-anything/blob/HEAD/\
segment_anything/modeling/prompt_encoder.py

"""

from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class PromptEncoder(nn.Module):
  """Sam prompt encoder for points and boxes."""

  embed_dim: int = 256
  image_embedding_size: Tuple[int, int] = (1024 // 16, 1024 // 16)
  input_image_size: Tuple[int, int] = (1024, 1024)
  num_point_embeddings: int = 4  # pos/neg point + 2 box corners
  mask_in_chans: int = 16

  def setup(self):
    self.pe_layer = PositionEmbeddingRandom(
        self.embed_dim // 2, name='pe_layer')
    point_embeddings = []
    # TODO(zhouxy): check if `nn.initializers.normal(stddev=1.)` is the same as
    # pytorch nn.Embedding default initialization.
    for i in range(self.num_point_embeddings):
      point_embeddings.append(self.param(
          f'point_embeddings.{i}.weight',
          nn.initializers.normal(stddev=1.),
          (1, self.embed_dim)))
    self.point_embeddings = point_embeddings
    del point_embeddings
    self.not_a_point_embed = self.param(
        'not_a_point_embed.weight',
        nn.initializers.normal(stddev=1.),
        (1, self.embed_dim))
    self.no_mask_embed = self.param(
        'no_mask_embed.weight',
        nn.initializers.normal(stddev=1.),
        (1, self.embed_dim))
    self.mask_downscaling = MaskDownScaling(
        mask_in_chans=self.mask_in_chans, embed_dim=self.embed_dim,
        name='mask_downscaling')

  def get_dense_pe(self, image_embedding_size=None):
    if image_embedding_size is None:
      image_embedding_size = self.image_embedding_size
    return self.pe_layer(image_embedding_size)

  def _embed_points(self, points, labels, pad, image_size=None):
    """Embed points.

    Args:
      points: (num_prompts, num_points, 2). In absolute coordinates.
      labels: (num_prompts, num_points)
      pad: bool
      image_size: Tuple[int, int] or None
    Returns:
      point_embeddings: (num_prompts, num_points, embed_dim)
    """
    # Shift to center of pixel following:
    # https://github.com/facebookresearch/segment-anything/blob/main/\
    # segment_anything/modeling/prompt_encoder.py#L80
    points = points + 0.5
    if pad:
      padding_point = jnp.zeros((points.shape[0], 1, 2), dtype=jnp.float32)
      padding_label = -jnp.ones((labels.shape[0], 1), dtype=jnp.float32)
      points = jnp.concatenate([points, padding_point], axis=1)
      labels = jnp.concatenate([labels, padding_label], axis=1)
    point_embedding = self.pe_layer.forward_with_coords(
        points, self.input_image_size if image_size is None else image_size
    )  # (num_prompts, num_points, embed_dim)
    ignored_points = labels[..., None] == -1  # (num_prompts, num_points, 1)
    point_embedding = point_embedding * (1 - ignored_points) + (
        self.not_a_point_embed[None] * ignored_points
    )
    neg_points = labels[..., None] == 0  # (num_prompts, num_points, 1)
    point_embedding += neg_points * self.point_embeddings[0][None]
    pos_points = labels[..., None] == 1  # (num_prompts, num_points, 1)
    point_embedding += pos_points * self.point_embeddings[1][None]
    return point_embedding

  def _embed_boxes(self, boxes, image_size=None):
    boxes = boxes + 0.5
    coords = boxes.reshape(-1, 2, 2)
    corner_embedding = self.pe_layer.forward_with_coords(
        coords, self.input_image_size if image_size is None else image_size
    )  # (num_prompts, 2, embed_dim)
    lt_emb = corner_embedding[:, 0, :] + self.point_embeddings[2]
    rb_emb = corner_embedding[:, 1, :] + self.point_embeddings[3]
    corner_embedding = jnp.stack(
        [lt_emb, rb_emb], axis=1
    )  # (num_prompts, 2, embed_dim)
    return corner_embedding

  def _embed_masks(self, masks):
    mask_embedding = self.mask_downscaling(masks)
    return mask_embedding

  @nn.compact
  def __call__(
      self,
      points,
      point_labels,
      boxes=None,
      masks=None,
      image_size=None,
      image_embedding_size=None,
  ):
    """Forward pass. Currently only supports points prompt.

    Args:
      points: (num_prompts, num_points, 2)
      point_labels: (num_prompts, num_points): labels of each point. 1 means
        positive points, 0 means negative points (shouldn't be included in the
        mask), and -1 means padded/ ignored points.
      boxes: (num_prompts, 4) or None
      masks: (num_prompts, height, width) or None
      image_size: Tuple[int, int] or None
      image_embedding_size: Tuple[int, int] or None
    Returns:
      sparse_embeddings: (num_prompts, num_points, embed_dim)
      dense_embeddings: (num_prompts, H, W, embed_dim)
    """
    num_prompts = points.shape[0] if points is not None else (
        boxes.shape[0] if boxes is not None else masks.shape[0])
    sparse_embeddings = jnp.zeros(
        (num_prompts, 0, self.embed_dim), dtype=jnp.float32)
    if points is not None:
      point_embeddings = self._embed_points(
          points, point_labels, pad=(boxes is None), image_size=image_size)
      sparse_embeddings = jnp.concatenate(
          [sparse_embeddings, point_embeddings], axis=1)
    if boxes is not None:
      box_embeddings = self._embed_boxes(boxes, image_size=image_size)
      sparse_embeddings = jnp.concatenate(
          [sparse_embeddings, box_embeddings], axis=1)
    if masks is not None:
      dense_embeddings = self._embed_masks(masks)
    else:
      if image_embedding_size is None:
        image_embedding_size = self.image_embedding_size
      dense_embeddings = jnp.broadcast_to(
          self.no_mask_embed[:, None, None],
          (num_prompts, image_embedding_size[0],
           image_embedding_size[1], self.embed_dim,)
      )
    return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
  """Positional encoding using random spatial frequencies."""

  num_pos_feats: int
  scale: Optional[float] = None

  def setup(self):
    scale = 1.0 if self.scale is None or self.scale <= 0.0 else self.scale
    self.positional_encoding_gaussian_matrix = self.param(
        'positional_encoding_gaussian_matrix',
        nn.initializers.normal(stddev=scale),
        (2, self.num_pos_feats)
    )

  def _pe_encoding(self, coords):
    """PE encoding."""
    # Assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
    coords = 2 * coords - 1
    coords = coords @ jax.lax.stop_gradient(
        self.positional_encoding_gaussian_matrix)
    coords = 2 * jnp.pi * coords
    # outputs d_1 x ... x d_n x C shape
    return jnp.concatenate([jnp.sin(coords), jnp.cos(coords)], axis=-1)

  @nn.compact
  def __call__(self, size):
    """Forward pass.

    Args:
      size: 2
    Returns:
      pe: H x W x D
    """
    h, w = size
    grid = jnp.ones((h, w), dtype=jnp.float32)
    y_embed = jnp.cumsum(grid, axis=0) - 0.5
    x_embed = jnp.cumsum(grid, axis=1) - 0.5
    y_embed = y_embed / h
    x_embed = x_embed / w
    pe = self._pe_encoding(jnp.stack([x_embed, y_embed], axis=-1))
    return pe

  def forward_with_coords(self, coords_input, image_size):
    """Forward with points.

    Args:
      coords_input: (num_prompts, num_points, 2)
      image_size: (2,)
    Returns:
      embedding: (num_prompts, num_points, self.num_pos_feats * 2)
    """
    x = coords_input[:, :, 0] / image_size[1]
    y = coords_input[:, :, 1] / image_size[0]
    return self._pe_encoding(jnp.stack([x, y], axis=-1))


class MaskDownScaling(nn.Module):
  """Mask downscaling."""
  mask_in_chans: int = 16
  embed_dim: int = 256

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(
        self.mask_in_chans // 4, kernel_size=(2, 2), strides=(2, 2),
        name='0')(x)
    x = nn.LayerNorm(name='1')(x)
    x = nn.gelu(x, approximate=False)
    x = nn.Conv(
        self.mask_in_chans, kernel_size=(2, 2), strides=(2, 2),
        name='3')(x)
    x = nn.LayerNorm(name='4')(x)
    x = nn.gelu(x, approximate=False)
    x = nn.Conv(
        self.embed_dim, kernel_size=(1, 1), strides=(1, 1),
        name='6')(x)
    return x
