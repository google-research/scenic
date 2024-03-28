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

"""Embedding utils."""
from typing import Any, Callable, Dict, Optional

import flax.linen as nn
import jax
from jax.nn import initializers
import jax.numpy as jnp
from scenic.projects.layout_denoise.layers import common


class TokenEmbedding(nn.Module):
  """Creates learned embeddings for text.

  Attributes:
    hidden_dim: Hidden dimension for the pos embeddings.
    vocab_size: Number of unique tokens.
    token_emb_init: Positional embeddings initializer.
    dtype: Jax dtype; The dtype of the computation (default: float32).
  """
  hidden_dim: int
  vocab_size: int
  token_emb_init: Callable[..., Any] = initializers.normal(stddev=1.0)
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, tokens) -> jnp.ndarray:
    """Creates the token embeddings.

    Args:
      tokens: the tokens to be embeded.

    Returns:
      Embedding for tokens with rank=token_rank + 1.
    """
    embs = self.param('token_emb', self.token_emb_init,
                      (self.vocab_size, self.hidden_dim))
    embds = jnp.take(embs, tokens, axis=0)
    return jnp.asarray(embds, self.dtype)


class InputPosEmbeddingSine(nn.Module):
  """Creates sinusoidal positional embeddings for inputs."""

  hidden_dim: int
  dtype: jnp.dtype = jnp.float32
  scale: Optional[float] = None
  temperature: float = 10000
  normalize: bool = True

  @nn.compact
  def __call__(self, padding_mask: jnp.ndarray) -> jnp.ndarray:
    """Creates the positional embeddings for transformer inputs.

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

    if self.normalize:
      eps = 1e-6
      scale = self.scale if self.scale is not None else 2 * jnp.pi
      y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
      x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

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


class ImageEmbedding(nn.Module):
  """Creates learned embeddings for images.

  Attributes:
    hidden_dim: Hidden dimension for the pos embeddings.
    backbone_num_filters: Num filters in the ResNet backbone.
    backbone_num_layers: Num layers in the ResNet backbone.
    dtype: Jax dtype; The dtype of the computation (default: float32).
  """
  hidden_dim: int
  backbone_num_filters: int
  backbone_num_layers: int
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self,
               cnn,
               images: jnp.ndarray,
               train: bool,
               *,
               padding_mask: Optional[jnp.ndarray] = None,
               update_batch_stats: bool = False) -> Dict[str, Any]:
    """Creates the image embeddings.

    Args:
      cnn: Conv Net for processing the image.
      images: The images to be embedded.
      train:  Whether it is training.
      padding_mask: Binary matrix with 0 at padded image regions.
      update_batch_stats: Whether update the batch statistics for the BatchNorms
        in the backbone. if None, the value of `train` flag will be used, i.e.
        we update the batch stat if we are in the train mode.

    Returns:
      Output: dict; that has 'content_emb' and 'pos_emb'.
    """
    if update_batch_stats is None:
      update_batch_stats = train

    backbone_features = cnn(images, train=update_batch_stats)
    x = backbone_features['stage_4']

    bs, h, w, _ = x.shape

    if padding_mask is None:
      padding_mask_downsampled = jnp.ones((bs, h, w), dtype=jnp.bool_)
    else:
      padding_mask_downsampled = jax.image.resize(
          padding_mask.astype(jnp.float32), shape=[bs, h, w],
          method='nearest').astype(jnp.bool_)
    pos_emb = InputPosEmbeddingSine(hidden_dim=self.hidden_dim)(
        padding_mask_downsampled)

    # Project and reshape to 3 dimensions and project.
    x = nn.Conv(features=self.hidden_dim, kernel_size=(1, 1), strides=(1, 1))(x)
    x = x.reshape(bs, h * w, self.hidden_dim)
    mask = jnp.reshape(padding_mask_downsampled, [bs, h * w])
    output = {}
    output['content_emb'] = x
    output['pos_emb'] = pos_emb
    output['mask'] = mask
    output['backbone_features'] = backbone_features
    output['shapes'] = (bs, h, w)
    return output


class QueryPosEmbedding(nn.Module):
  """Creates learned positional embeddings for object queries.

  Attributes:
    hidden_dim: Hidden dimension for the pos embeddings.
    num_queries: Number of object queries.
    posemb_init: Positional embeddings initializer.
    dtype: Jax dtype; The dtype of the computation (default: float32).
  """
  hidden_dim: int
  num_queries: int
  posemb_init: Callable[..., Any] = initializers.normal(stddev=1.0)
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self) -> jnp.ndarray:
    """Creates the positional embeddings for queries.

    Returns:
      Positional embedding for object queries.
    """
    query_pos = self.param('query_emb', self.posemb_init,
                           (self.num_queries, self.hidden_dim))
    query_pos = jnp.expand_dims(query_pos, 0)
    return jnp.asarray(query_pos, self.dtype)


class StructureEmbedding(nn.Module):
  """Creates learned embeddings for structures.

  Attributes:
    hidden_dim: Hidden dimension for the pos embeddings.
    num_queries: The number of queries.
    dtype: Jax dtype; The dtype of the computation (default: float32).
  """
  hidden_dim: int
  num_queries: int
  txt_pool_method: str = 'max'
  num_types: int = 30
  coordinate_emb_depth: int = 256
  dtype: jnp.dtype = jnp.float32
  aggregation: str = 'concat'
  dropout_rate: float = 0.2

  @nn.compact
  def __call__(
      self,
      obj_mask: jnp.ndarray,
      desc_id: jnp.ndarray,
      resource_id: jnp.ndarray,
      name_id: jnp.ndarray,
      boxes: jnp.ndarray,
      task: str,
      token_embder,
      pos_pattern,
      train) -> Dict[str, Any]:
    """Creates the structure embeddings."""
    # Recover coordinates in absolute values.
    # h = jnp.sum(padding_mask, axis=1)[:, 0]
    # w = jnp.sum(padding_mask, axis=2)[:, 0]
    # [bs, 1, 4]
    # sizes = jnp.expand_dims(jnp.stack([w, h, w, h], axis=1), axis=1)
    # [bs, num_objs, 1]
    bcx, bcy, bw, bh = jnp.split(boxes, 4, axis=2)
    # x1, y1, x2, y2.
    boxes = jnp.concatenate(
        [bcx - bw / 2, bcy - bh / 2, bcx + bw / 2, bcy + bh / 2], axis=2)

    pos_embs = self.embed_pos(
        obj_mask=obj_mask, obj_boxes=boxes, pos_pattern=pos_pattern)

    obj_embds = self.embed_layout(
        obj_mask=obj_mask,
        obj_desc_id=desc_id,
        obj_resource_id=resource_id,
        obj_name_id=name_id,
        token_embder=token_embder)

    obj_embds = nn.Dropout(rate=self.dropout_rate)(
        obj_embds, deterministic=not train)
    pos_embs = nn.Dropout(rate=self.dropout_rate)(
        pos_embs, deterministic=not train)

    output = {}
    output['content_emb'] = obj_embds
    output['mask'] = jnp.asarray(jnp.minimum(obj_mask, 1), self.dtype)
    output['pos_emb'] = pos_embs
    return output

  def embed_layout(self, obj_mask, obj_desc_id, obj_resource_id, obj_name_id,
                   token_embder):
    """Prepares the input for the screen encoder."""
    # [bs, num_objs, tokens] -> [bs, num_objs, depth]
    # jax.experimental.host_callback.id_print(
    #     (obj_txt, obj_type, obj_boxes, obj_targets), what='input')
    # Embed types.
    # [bs, num_objs, tokens] -> [bs, num_objs, depth]

    obj_desc_embs = pool_txt_embs(
        obj_desc_id,
        token_embder(obj_desc_id),
        method=self.txt_pool_method,
        valid_token_start=4,
        dtype=self.dtype)
    obj_resource_id_embs = pool_txt_embs(
        obj_resource_id,
        token_embder(obj_resource_id),
        method=self.txt_pool_method,
        valid_token_start=4,
        dtype=self.dtype)
    obj_name_embs = pool_txt_embs(
        obj_name_id,
        token_embder(obj_name_id),
        method=self.txt_pool_method,
        valid_token_start=4,
        dtype=self.dtype)

    if self.aggregation == 'concat':
      obj_embds = jnp.concatenate(
          [obj_desc_embs, obj_resource_id_embs, obj_name_embs], axis=-1)
      obj_embds = common.dense(obj_embds, self.hidden_dim, self.dtype)
    elif self.aggregation == 'sum':
      obj_embds = (obj_desc_embs + obj_resource_id_embs + obj_name_embs)
    else:
      raise ValueError('Unrecognized aggregation method: %s' % self.aggregation)
    obj_non_paddings = jnp.asarray(jnp.minimum(obj_mask, 1), self.dtype)
    obj_embds *= jnp.expand_dims(obj_non_paddings, 2)
    return obj_embds

  def embed_pos(self, obj_mask, obj_boxes, pos_pattern='1/4'):
    """Prepares the input for the screen encoder."""
    # Embed positions.
    # [bs, num_objs, 4] -> [bs, num_objs, depth]
    if self.aggregation == 'sum':
      coordinate_emb_depth = self.hidden_dim
    else:
      coordinate_emb_depth = self.coordinate_emb_depth

    pos_embds = encode_coordinate(
        obj_boxes, coordinate_emb_depth, self.dtype, pattern=pos_pattern)
    obj_non_paddings = jnp.asarray(jnp.minimum(obj_mask, 1), self.dtype)
    pos_embds *= jnp.expand_dims(obj_non_paddings, 2)
    return pos_embds


def encode_coordinate(obj_boxes, depth, dtype, freq_depth=64, pattern='1/4'):
  """Encodes positions using random features-based encoder."""
  # positions: [batch, length, group, dim]
  if pattern == '4/1':
    obj_boxes = jnp.expand_dims(obj_boxes, 3)
    num_groups = 4
  elif pattern == '1/4':
    obj_boxes = jnp.expand_dims(obj_boxes, 2)
    num_groups = 1
  elif pattern == '2/2':
    obj_boxes = jnp.reshape(obj_boxes, obj_boxes.shape[:2] + (2, 2))
    num_groups = 2
  else:
    raise ValueError('Unrecognized coord encoding pattern: %s' % pattern)
  kernel_init = nn.initializers.normal(stddev=1e-6)
  # [batch, length, group, freq_depth]
  freqs = common.dense(obj_boxes, freq_depth, dtype, kernel_init=kernel_init)
  # [batch, length, group, freq_depth * 2]
  features = jnp.concatenate([jnp.cos(freqs), jnp.sin(freqs)], axis=-1)
  coord_embds = common.dense(features, depth // num_groups, dtype)
  coord_embds = nn.relu(coord_embds)
  coord_embds = common.dense(coord_embds, depth // num_groups, dtype)
  coord_embds = jnp.reshape(coord_embds, features.shape[:2] + (-1,))
  return coord_embds


def pool_txt_embs(token_ids,
                  text_embeddings,
                  method,
                  valid_token_start=4,
                  dtype=jnp.float32):
  """Aggregate text embedding for a UI element."""
  # [batch, #nodes, #tokens]
  non_tokens = jnp.asarray(jnp.less(token_ids, valid_token_start), dtype)
  if method == 'max':
    assert len(token_ids.shape) == 3
    embed_bias = non_tokens * -1e7
    # Max value for each dimension
    text_embeddings = jnp.max(
        text_embeddings + jnp.expand_dims(embed_bias, 3), axis=-2)
    # Find locations still with very large negative values.
    non_paddings = jnp.asarray(jnp.greater(text_embeddings, -1e6), dtype)
    # For padded location, use 0.
    embeddings = text_embeddings * non_paddings
  elif method == 'sum':
    embeddings = jnp.sum(
        text_embeddings * jnp.expand_dims(1 - non_tokens, 4), axis=-2)
  elif method == 'mean':
    sum_embeddings = jnp.sum(
        text_embeddings * jnp.expand_dims(1 - non_tokens, 4), axis=-2)
    token_counts = jnp.maximum(jnp.sum(1 - non_tokens, axis=-1), 1)
    embeddings = sum_embeddings / token_counts
  else:
    raise ValueError('Unrecognized token aggregation %s' % method)
  return embeddings
