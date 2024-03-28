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

"""Transformer modules for DeformableDETR.

Note that most names and variables are chose for staying closely aligned to
the official pytorch implementation [1].

[1] https://github.com/fundamentalvision/Deformable-DETR.
"""

import functools
from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.projects.baselines.deformable_detr.attention import MultiScaleDeformableAttention
from scenic.projects.baselines.detr.model import MultiHeadDotProductAttention


def inverse_sigmoid(x: jnp.ndarray, eps: float = 1e-5):
  x = x.clip(min=0, max=1)
  x1 = x.clip(min=eps)
  x2 = (1 - x).clip(min=eps)
  return jnp.log(x1) - jnp.log(x2)


pytorch_kernel_init = functools.partial(jax.nn.initializers.variance_scaling,
                                        1. / 3., 'fan_in', 'uniform')


def uniform_initializer(minval, maxval, dtype=jnp.float32):

  def init(key, shape, dtype=dtype):
    return jax.random.uniform(key, shape, dtype, minval=minval, maxval=maxval)

  return init


class BBoxCoordPredictor(nn.Module):
  """FFN block for predicting bounding box coordinates."""
  mlp_dim: int
  num_layers: int
  use_sigmoid: bool
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies FFN MLP block to inputs.

    Args:
      x: Input tensor.

    Returns:
      Output of FFN MLP block.
    """
    for _ in range(self.num_layers - 1):
      # This is like pytorch initializes biases in linear layers.
      bias_range = 1 / np.sqrt(x.shape[-1])
      x = nn.Dense(
          self.mlp_dim,
          kernel_init=pytorch_kernel_init(dtype=self.dtype),
          bias_init=uniform_initializer(
              -bias_range, bias_range, dtype=self.dtype),
          dtype=self.dtype)(
              x)
      x = nn.relu(x)

    bias_range = 1 / np.sqrt(x.shape[-1])
    x = nn.Dense(
        4,
        kernel_init=pytorch_kernel_init(dtype=self.dtype),
        bias_init=uniform_initializer(
            -bias_range, bias_range, dtype=self.dtype))(
                x)
    if self.use_sigmoid:
      x = nn.sigmoid(x)
    return x


class DeformableDETREncoderLayer(nn.Module):
  """Layer of DETR encoder."""
  spatial_shapes: Tuple[Tuple[int, int], ...]
  embed_dim: int
  num_heads: int
  num_levels: int
  num_reference_points: int
  ffn_dim: int
  dropout: float
  compiler_config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      src: jnp.ndarray,
      pos_embed: jnp.ndarray,
      ref_points: jnp.ndarray,
      pad_mask: jnp.ndarray,
      train: bool,
  ) -> jnp.ndarray:
    """Single encoder layer using MultiScaleDeformableAttention.

    Args:
      src: [bs, len_qkv, embed_dim]-ndarray of values. This is self-attention so
        these will also be used as queries after position embedding added.
      pos_embed: [bs, len_qkv, embed_dim]-ndarray of position embedding for each
        src position.
      ref_points: [bs, len_qkv, num_levels, box_dim]-ndarray of reference points
        for each query. box_dim is in {2, 4} as ref_points can either be box
        cxcy or cxcywh.
      pad_mask: [bs, len_qkv]-ndarray of boolean values, where 1 indicates pad.
      train: Whether we are in training mode.

    Returns:
      [bs, len_qkv, embed_dim]-ndarray of encoding from layer.
    """
    query = src + pos_embed
    x = MultiScaleDeformableAttention(
        spatial_shapes=self.spatial_shapes,
        embed_dim=self.embed_dim,
        num_heads=self.num_heads,
        num_levels=self.num_levels,
        num_points=self.num_reference_points,
        compiler_config=self.compiler_config,
        dtype=self.dtype,
        name='self_attn')(query, ref_points, src, pad_mask, train)

    x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
    x = src + x
    x = nn.LayerNorm(name='norm1')(x)

    # FeedForward Network.
    residual = x
    bias_range = 1 / np.sqrt(x.shape[-1])
    x = nn.Dense(
        self.ffn_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=uniform_initializer(
            -bias_range, bias_range, dtype=self.dtype),
        name='linear1')(
            x)
    x = nn.relu(x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
    bias_range = 1 / np.sqrt(x.shape[-1])
    x = nn.Dense(
        self.embed_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=uniform_initializer(
            -bias_range, bias_range, dtype=self.dtype),
        name='linear2')(
            x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
    x = residual + x
    x = nn.LayerNorm(name='norm2')(x)

    # TODO(tonysherbondy): Consider clamping values between [-max, max].
    return x


class DeformableDETREncoder(nn.Module):
  """Sequence of DeformableDETREncoderLayer."""
  spatial_shapes: Tuple[Tuple[int, int]]
  embed_dim: int
  num_layers: int
  num_heads: int
  num_levels: int
  num_reference_points: int
  ffn_dim: int
  dropout: float
  compiler_config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      src: jnp.ndarray,
      pos_embed: jnp.ndarray,
      ref_points: jnp.ndarray,
      pad_mask: jnp.ndarray,
      train: bool,
  ) -> jnp.ndarray:
    """Compute encoding with stack of encoder layers.

    Args:
      src: [bs, len_qkv, embed_dim]-ndarray of values. This is self-attention so
        these will also be used as queries after position embedding added.
      pos_embed: [bs, len_qkv, embed_dim]-ndarray of position embedding for each
        src position.
      ref_points: [bs, len_qkv, num_levels, box_dim]-ndarray of reference points
        for each query. box_dim is in {2, 4} as ref_points can either be box
        cxcy or cxcywh.
      pad_mask: [bs, len_qkv]-ndarray of boolean values, where 1 indicates pad.
      train: Whether we are in training mode.

    Returns:
      [bs, len_qkv, embed_dim]-ndarray of encoding from layer.
    """
    x = src
    for i in range(self.num_layers):
      # TODO(tonysherbondy): Consider layerdrop (see
      # https://arxiv.org/abs/1909.11556)
      x = DeformableDETREncoderLayer(
          spatial_shapes=self.spatial_shapes,
          embed_dim=self.embed_dim,
          num_heads=self.num_heads,
          num_levels=self.num_levels,
          num_reference_points=self.num_reference_points,
          ffn_dim=self.ffn_dim,
          dropout=self.dropout,
          compiler_config=self.compiler_config,
          dtype=self.dtype,
          name=f'layer{i}')(
              x, pos_embed, ref_points, pad_mask, train=train)

    return x


class DeformableDETRDecoderLayer(nn.Module):
  """Layer of DeformableDETR decoder.

  Uses MultiScaleDeformableAttention for cross-attention and typical DETR dense
  attention for the self-attention.

  Attributes:
    spatial_shapes: (h, w) for each feature level.
    embed_dim: Size of the hidden embedding dimension, used for query, value,
      embeddings, and outputs.
    num_heads: Number of heads.
    num_levels: Number of feature levels.
    num_points: Number of points in deformable attention.
    dropout: Dropout rate.
    ffn_dim: Hidden dimension for feed-forward/MLP network.
    dtype: Data type of the computation (default: float32).
  """
  spatial_shapes: Sequence[Tuple[int, int]]
  embed_dim: int
  num_heads: int
  num_levels: int
  num_reference_points: int
  ffn_dim: int
  dropout: float
  compiler_config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      query: jnp.ndarray,
      query_pos: jnp.ndarray,
      ref_points: jnp.ndarray,
      value: jnp.ndarray,
      pad_mask: jnp.ndarray,
      train: bool,
  ) -> jnp.ndarray:
    """Compute decoder layer.

    Args:
      query: [bs, len_q, embed_dim] of queries.
      query_pos: [bs, len_q, embed_dim] of position embedding for each query
        position.
      ref_points: [bs, len_q, num_levels, box_dim] of reference points for each
        query. box_dim is in {2, 4} as ref_points can either be box cxcy or
        cxcywh.
      value: [bs, len_v, embed_dim] of values to be applied in cross-attention.
      pad_mask: [bs, len_v] of boolean values, where 0 indicates padding in
        value array.
      train: Whether we are in training mode.

    Returns:
      [bs, len_q, embed_dim] of decoder-encoded queries.
    """
    x = MultiHeadDotProductAttention(
        name='self_attn',
        # Scaling is necessary to match pytorch official model that combines
        # the kernel into one init with larger fan_in.
        # qkv_kernel_init=jax.nn.initializers.variance_scaling(
        #     0.5, 'fan_avg', 'uniform'),
        num_heads=self.num_heads)(
            inputs_q=query, pos_emb_q=query_pos, pos_emb_k=query_pos)

    query = query + nn.Dropout(rate=self.dropout)(x, deterministic=not train)
    # TODO(tonysherbondy): Reverse layer norm naming and just fix in
    # torch_param map.
    query = nn.LayerNorm(dtype=self.dtype, name='norm2')(query)

    # cross attention
    x = MultiScaleDeformableAttention(
        spatial_shapes=self.spatial_shapes,
        num_levels=self.num_levels,
        num_heads=self.num_heads,
        num_points=self.num_reference_points,
        embed_dim=self.embed_dim,
        compiler_config=self.compiler_config,
        dtype=self.dtype,
        name='cross_attn')(query + query_pos, ref_points, value, pad_mask,
                           train)

    query = query + nn.Dropout(rate=self.dropout)(x, deterministic=not train)
    query = nn.LayerNorm(dtype=self.dtype, name='norm1')(query)

    # FeedForward Network.
    # TODO(tonysherbondy): Extract as module since its the same as other FFN.
    bias_range = 1 / np.sqrt(x.shape[-1])
    x = nn.Dense(
        self.ffn_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=uniform_initializer(
            -bias_range, bias_range, dtype=self.dtype),
        name='linear1')(
            query)

    x = nn.relu(x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
    bias_range = 1 / np.sqrt(x.shape[-1])
    x = nn.Dense(
        self.embed_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=uniform_initializer(
            -bias_range, bias_range, dtype=self.dtype),
        name='linear2')(
            x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
    query = query + x
    query = nn.LayerNorm(name='norm3')(query)

    # TODO(tonysherbondy): Consider clamping values between [-max, max].
    return query


class DeformableDETRDecoder(nn.Module):
  """Sequence of DeformableDETRDecoderLayers.

  Attributes:
    embed_dim: Size of the hidden embedding dimension, used for query, value,
      embeddings, and outputs.
    num_heads: Number of heads. num_levels  : Number of feature levels.
    num_layers: Number of decoder layers.
    dropout: Dropout rate.
    dtype: Data type of the computation (default: float32).
  """
  spatial_shapes: Sequence[Tuple[int, int]]
  embed_dim: int
  num_heads: int
  num_levels: int
  num_layers: int
  num_reference_points: int
  ffn_dim: int
  bbox_embeds: Sequence[nn.Module]
  dropout: float
  compiler_config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      query: jnp.ndarray,
      query_pos: jnp.ndarray,
      ref_points: jnp.ndarray,
      value: jnp.ndarray,
      pad_mask: jnp.ndarray,
      valid_ratios: jnp.ndarray,
      train: bool,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply decoder.

    Args:
      query: [bs, len_q, embed_dim] of queries.
      query_pos: [bs, len_q, embed_dim] of position embedding for each query
        position.
      ref_points: [bs, len_q, box_dim] of reference points for each query.
        box_dim is in {2, 4} as ref_points can either be box cxcy or cxcywh.
      value: [bs, len_v, embed_dim] of values to be applied in cross-attention.
      pad_mask: [bs, len_v] of boolean values, where 0 indicates padding in
        value array.
      valid_ratios: [bs, num_levels, 2] of ratios of actual shape dim to padded
        dim. Range from (0, 1]; 0 is all padding (impossible) and 1 is no
        padding.
      train: Whether we are in training mode.

    Returns:
      [bs, len_q, embed_dim] of decoder-encoded queries and [bs, len_q, box_dim]
        of output reference points.
    """
    assert ref_points.shape[-1] in {2, 4}

    output = query
    output_by_layer = []
    ref_points_by_layer = []
    for i in range(self.num_layers):
      # Use valid shape ratios to keep reference points in bounds at each level.
      vratios = valid_ratios
      if ref_points.shape[-1] == 4:
        vratios = jnp.concatenate([vratios] * 2, -1)
      ref_points_input = ref_points[:, :, None] * vratios[:, None]

      output = DeformableDETRDecoderLayer(
          spatial_shapes=self.spatial_shapes,
          embed_dim=self.embed_dim,
          num_heads=self.num_heads,
          num_levels=self.num_levels,
          num_reference_points=self.num_reference_points,
          ffn_dim=self.ffn_dim,
          dropout=self.dropout,
          compiler_config=self.compiler_config,
          dtype=self.dtype,
          name=f'layer{i}')(
              query=output,
              query_pos=query_pos,
              ref_points=ref_points_input,
              value=value,
              pad_mask=pad_mask,
              train=train)

      bbox_offset_embed = self.bbox_embeds[i](output)

      if ref_points.shape[-1] == 4:
        new_ref_points = bbox_offset_embed + inverse_sigmoid(ref_points)
      else:
        new_ref_points = bbox_offset_embed
        xy = bbox_offset_embed[..., :2] + inverse_sigmoid(ref_points)
        # Here is where ref_points goes to 4d.
        new_ref_points = jnp.concatenate([xy, bbox_offset_embed[..., 2:]],
                                         axis=-1)
      ref_points = nn.sigmoid(new_ref_points)
      # To satisfy deformable detr iterative refinement must stop gradient here.
      ref_points = jax.lax.stop_gradient(ref_points)

      output_by_layer.append(output)
      ref_points_by_layer.append(ref_points)

    output, ref_points = jnp.stack(output_by_layer), jnp.stack(
        ref_points_by_layer)

    return output, ref_points


def get_mask_valid_ratio(mask: jnp.ndarray) -> jnp.ndarray:
  """Get non-padded:padded ratio for width/height for each mask."""
  _, h, w = mask.shape
  valid_h = jnp.sum(mask[:, :, 0], 1)
  valid_w = jnp.sum(mask[:, 0, :], 1)
  valid_ratio_h = valid_h / h
  valid_ratio_w = valid_w / w
  valid_ratio = jnp.stack([valid_ratio_w, valid_ratio_h], -1)
  return valid_ratio


def prepare_encoder_input(
    inputs: Sequence[jnp.ndarray], masks: Sequence[jnp.ndarray],
    pos_embeds: Sequence[jnp.ndarray], level_embeds: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Flatten all image dimensions and add level embed to position embed."""
  input_flattened = []
  mask_flattened = []
  level_pos_embed_flattened = []
  for inp, mask, pos_embed, level_embed in zip(inputs, masks, pos_embeds,
                                               level_embeds):
    bs, h, w, c = inp.shape
    inp = inp.reshape(bs, h * w, c)
    mask = mask.reshape(bs, h * w)
    pos_embed = pos_embed.reshape(bs, h * w, c)
    lvl_pos_embed = pos_embed + level_embed

    level_pos_embed_flattened.append(lvl_pos_embed)
    input_flattened.append(inp)
    mask_flattened.append(mask)

  input_flattened = jnp.concatenate(input_flattened, 1)
  mask_flattened = jnp.concatenate(mask_flattened, 1)
  level_pos_embed_flattened = jnp.concatenate(level_pos_embed_flattened, 1)
  valid_ratios = jnp.stack([get_mask_valid_ratio(m) for m in masks], 1)
  return (input_flattened, valid_ratios, level_pos_embed_flattened,
          mask_flattened)


@functools.partial(jax.jit, static_argnums=0)
def get_encoder_reference_points(spatial_shapes: Sequence[Tuple[int, int]],
                                 valid_ratios: jnp.ndarray,
                                 dtype=jnp.float32) -> jnp.ndarray:
  """Return grid of 2D reference points within valid range by feature level.

  Args:
    spatial_shapes: [h, w] for each feature map level.
    valid_ratios: [bs, num_levels, 2] fraction of non-pad pixels in x and y.
    dtype: The dtype of the computation.

  Returns:
    [bs, len_v, num_levels, 2] of reference point positions in range [0, 1],
      where len_v is the sum of all feature map areas.
  """
  reference_points_list = []
  for lvl, (h, w) in enumerate(spatial_shapes):
    ref_y, ref_x = jnp.meshgrid(
        jnp.linspace(0.5, h - 0.5, h, dtype=dtype),
        jnp.linspace(0.5, w - 0.5, w, dtype=dtype))
    ref_y, ref_x = ref_y.T, ref_x.T
    ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h)
    ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w)
    ref = jnp.stack((ref_x, ref_y), -1)
    reference_points_list.append(ref)
  reference_points = jnp.concatenate(reference_points_list, 1)
  # TODO(tonysherbondy): The reference implementation does this multiplication
  # of the valid_ratios again, which basically removes normalizing by
  # valid_ratios in the loop. It should be equivalent to just normalizing by h
  # and w, but for some reason it does not reproduce the results if we do that.
  reference_points = reference_points[:, :, None] * valid_ratios[:, None]
  return reference_points


class DeformableDETRTransformer(nn.Module):
  """DeformableDETR Transformer.

  Attributes:
    embed_dim: Size of the hidden embedding dimension.
    enc_embed_dim: Size of the hidden embedding dimension for encoder.
    num_heads: Number of heads.
    num_queries: Number of object queries.
    num_enc_layers: Number of encoder layers.
    num_dec_layers: Number of decoder layers.
    num_enc_points: Number of encoder points in deformable attention.
    num_dec_points: Number of decoder points in deformable attention.
    ffn_dim: Size of feed-forward network embedding.
    dropout: Dropout rate.
    compiler_config: Compiler configuration.
    dtype: Data type of the computation (default: float32).
  """

  embed_dim: int
  enc_embed_dim: int
  num_heads: int
  num_queries: int
  num_enc_layers: int
  num_dec_layers: int
  num_enc_points: int
  num_dec_points: int
  bbox_embeds: Sequence[nn.Module]
  ffn_dim: int
  dropout: float
  compiler_config: ml_collections.ConfigDict
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      inputs: Sequence[jnp.ndarray],
      pad_masks: Sequence[jnp.ndarray],
      pos_embeds: Sequence[jnp.ndarray],
      train: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Prepare inputs for encoder.
    spatial_shapes = tuple([inp.shape[1:3] for inp in inputs])
    num_levels = len(spatial_shapes)

    level_embeds = jnp.asarray(
        self.param('level_embed', nn.initializers.normal(stddev=1.0),
                   (num_levels, self.enc_embed_dim)))
    (x, valid_ratios, pos_embeds, pad_masks) = prepare_encoder_input(
        inputs=inputs,
        masks=pad_masks,
        pos_embeds=pos_embeds,
        level_embeds=level_embeds)

    enc_ref_points = get_encoder_reference_points(
        tuple(spatial_shapes), valid_ratios)

    # Encoder.
    encoder = DeformableDETREncoder(
        spatial_shapes=spatial_shapes,
        embed_dim=self.enc_embed_dim,
        num_heads=self.num_heads,
        num_layers=self.num_enc_layers,
        num_levels=num_levels,
        ffn_dim=self.ffn_dim,
        num_reference_points=self.num_enc_points,
        dropout=self.dropout,
        compiler_config=self.compiler_config,
        dtype=self.dtype,
        name='encoder')

    x = encoder(
        src=x,
        pos_embed=pos_embeds,
        ref_points=enc_ref_points,
        pad_mask=pad_masks,
        train=train)

    # Project encoder output to decoder embedding. Note that this layer does
    # not exist in the reference implementation. However, this greatly reduces
    # the amount of memory required while training and seems to have no
    # noticeable effect in COCO mAP.
    if self.enc_embed_dim != self.embed_dim:
      x = nn.Conv(
          features=self.embed_dim,
          kernel_size=(1,),
          name='enc_to_dec_proj_conv')(
              x)
      x = nn.GroupNorm(num_groups=32, name='enc_to_dec_proj_groupnorm')(x)

    query_embed = jnp.asarray(
        self.param('query_embed', nn.initializers.normal(stddev=1.0),
                   (self.num_queries, self.embed_dim * 2)))

    # Prepare decoder input.
    bs = x.shape[0]
    query_embed = query_embed[None, ...].repeat(bs, 0)
    query_embed, query = jnp.split(query_embed, indices_or_sections=2, axis=-1)
    dec_init_ref_points = nn.Dense(2, name='ref_embed')(query_embed)
    dec_init_ref_points = nn.sigmoid(dec_init_ref_points)

    x, ref_points = DeformableDETRDecoder(
        spatial_shapes=spatial_shapes,
        embed_dim=self.embed_dim,
        num_heads=self.num_heads,
        num_layers=self.num_dec_layers,
        num_levels=num_levels,
        ffn_dim=self.ffn_dim,
        num_reference_points=self.num_dec_points,
        bbox_embeds=self.bbox_embeds,
        dropout=self.dropout,
        compiler_config=self.compiler_config,
        dtype=self.dtype,
        name='decoder')(
            value=x,
            query=query,
            query_pos=query_embed,
            ref_points=dec_init_ref_points,
            pad_mask=pad_masks,
            valid_ratios=valid_ratios,
            train=train)

    return x, ref_points, dec_init_ref_points
