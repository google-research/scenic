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

"""Util functions for PixelLLM models."""

import jax
import jax.numpy as jnp


def preprocess(
    inputs, pixel_mean, pixel_std, padding_mask=None, image_size=None
):
  """Proprocess images. Normalize pixels for non-padded pixels."""
  mean = jnp.asarray(pixel_mean, dtype=jnp.float32).reshape(1, 1, 1, 3)
  std = jnp.asarray(pixel_std, dtype=jnp.float32).reshape(1, 1, 1, 3)
  inputs = (inputs - mean) / std
  if padding_mask is not None:
    inputs = inputs * padding_mask[..., None]  # Padded pixels remain 0
  if image_size is not None and inputs.shape[1:3] != image_size:
    assert tuple(image_size) <= inputs.shape[1:3], (
        f'image_size={image_size} should be less than'
        f' inputs.shape[1:3]={inputs.shape[1:3]}'
    )
    inputs = jax.image.resize(
        inputs,
        (inputs.shape[0], image_size[0], image_size[1], inputs.shape[3]),
        method='bilinear',
    )
  return inputs


def build_solid_grid(points_per_side, with_offset=True):
  """Generates a 2D grid of points evenly spaced in [0, 1] x [0, 1]."""
  # return corner points instead
  if points_per_side < 1:
    points = jnp.stack(
        [jnp.zeros((2,), dtype=jnp.float32), jnp.ones((2,), dtype=jnp.float32)],
        axis=0,
    )
    return points
  if with_offset:
    offset = 1.0 / (2 * points_per_side)
  else:
    offset = 0.0
  points_one_side = jnp.linspace(offset, 1 - offset, points_per_side)
  points_x = jnp.tile(points_one_side[None, :], (points_per_side, 1))
  points_y = jnp.tile(points_one_side[:, None], (1, points_per_side))
  points = jnp.stack([points_x, points_y], axis=-1).reshape(-1, 2)
  return points  # (points_per_side ** 2, 2)


def build_donut_grid(
    points_per_side: int, thickness: int = 1, with_offset: bool = True
):
  """Create a dense grid of x,y points that form the border of a square.

  Points are spaced in [0, 1] x [0, 1]

  Args:
    points_per_side (int): The size of the grid (N x N).
    thickness (int): The width of the border.
    with_offset (bool): with offset for solid grid

  Returns:
      - x, y coordinates of the border.
  """
  dense_grid = build_solid_grid(points_per_side, with_offset)
  # TODO(jiaruixu): I found when this function output is incorrect inside pmap
  # [[1, 0], [1, 0], [0, 0], [0, 0]] instead of [[0, 1], [1, 0], [1, 1], [0, 0]]
  # not sure why. The workaround it to return early.
  if points_per_side <= 2:
    return dense_grid

  dense_grid = dense_grid.reshape(points_per_side, points_per_side, 2)

  top_grid = dense_grid[:thickness, :-thickness].reshape(-1, 2)
  right_grid = dense_grid[:-thickness, -thickness:].reshape(-1, 2)

  bottom_grid = dense_grid[-thickness:, thickness:].reshape(-1, 2)
  left_grid = dense_grid[thickness:, :thickness].reshape(-1, 2)

  dense_grid = jnp.concatenate(
      [top_grid, right_grid, bottom_grid, left_grid], axis=0
  )

  return dense_grid


def boxes_to_points(boxes, grids):
  """Sample points from boxes."""
  x0, y0, x1, y1 = jnp.split(boxes, 4, axis=-1)
  # [..., 4]
  boxes_xywh = jnp.concatenate([x0, y0, x1-x0, y1-y0], axis=-1)
  # [..., 1, 4]
  boxes_xywh = jnp.expand_dims(boxes_xywh, axis=-2)

  # [..., num_points, 2]
  points = boxes_xywh[...,:2] + grids * boxes_xywh[..., 2:]

  return points


def points_to_boxes(points, points_per_side=0, valid_mask=None):
  """Convert (x,y) points to XYXY boxes."""
  x = points[..., 0]
  y = points[..., 1]
  if valid_mask is None:
    x_min = jnp.min(x, axis=-1)
    y_min = jnp.min(y, axis=-1)
    x_max = jnp.max(x, axis=-1)
    y_max = jnp.max(y, axis=-1)
  else:
    x_min = jnp.min(
        jnp.where(valid_mask, x, x.max(axis=-1, keepdims=True) + 1), axis=-1
    )
    y_min = jnp.min(
        jnp.where(valid_mask, y, y.max(axis=-1, keepdims=True) + 1), axis=-1
    )
    x_max = jnp.max(
        jnp.where(valid_mask, x, x.min(axis=-1, keepdims=True) - 1), axis=-1
    )
    y_max = jnp.max(
        jnp.where(valid_mask, y, y.min(axis=-1, keepdims=True) - 1), axis=-1
    )

  if points_per_side > 1:
    w = x_max - x_min
    h = y_max - y_min
    offset_w = w / (points_per_side - 1) / 2
    offset_h = h / (points_per_side - 1) / 2

    x_min = x_min - offset_w
    y_min = y_min - offset_h
    x_max = x_max + offset_w
    y_max = y_max + offset_h

  boxes = jnp.stack([x_min, y_min, x_max, y_max], axis=-1)

  return boxes


def points_to_absolute(points, image_shape):
  """Convert (x,y) coords from [0, 1] relative coord to absolute coord.

  Args:
    points: (batch_size, ..., 2), xy
    image_shape: (batch_size, 2), hw

  Returns:
    (batch_size, ..., 2) in absolute coords
  """

  batch_size = points.shape[0]
  h, w = jnp.split(image_shape, 2, axis=-1)
  scaler = jnp.concatenate([w, h], axis=-1)
  scaler = jnp.reshape(scaler, (batch_size,) + (1,) * (points.ndim - 2) + (2,))

  points = points * scaler

  return points


def points_to_relative(points, image_shape):
  """Convert (x,y) coords from absolute coord  to [0, 1] relative coord.

  Args:
    points: (batch_size, ..., 2), xy
    image_shape: (batch_size, 2), hw

  Returns:
    (batch_size, ..., 2) in absolute coords
  """

  batch_size = points.shape[0]
  h, w = jnp.split(image_shape, 2, axis=-1)
  scaler = jnp.concatenate([w, h], axis=-1)
  scaler = jnp.reshape(scaler, (batch_size,) + (1,) * (points.ndim - 2) + (2,))

  points = points / scaler

  return points


def boxes_to_relative(boxes, image_shape):
  """Convert x0y0x1y1 boxes from absolute coord  to [0, 1] relative coord.

  Args:
    boxes: (batch_size, ..., 4), x0xyx1y1
    image_shape: (batch_size, 2), hw

  Returns:
    (batch_size, ..., 2) in absolute coords
  """

  batch_size = boxes.shape[0]
  h, w = jnp.split(image_shape, 2, axis=-1)
  scaler = jnp.concatenate([w, h, w, h], axis=-1)
  scaler = jnp.reshape(scaler, (batch_size,) + (1,) * (boxes.ndim - 2) + (4,))

  boxes = boxes / scaler

  return boxes


def get_image_shape(padding_mask, images):
  """Get image shape from padding mask."""

  if padding_mask is not None:
    valid_h = padding_mask.max(axis=2).sum(axis=-1)
    valid_w = padding_mask.max(axis=1).sum(axis=-1)
    # [batch_size, 2]
    image_shape = jnp.stack([valid_h, valid_w], axis=1)
  else:
    image_shape = jnp.concatenate(
        [
            jnp.ones((images.shape[0], 1), jnp.float32) * images.shape[1],
            jnp.ones((images.shape[0], 1), jnp.float32) * images.shape[2],
        ],
        axis=1,
    )  # B x 2, in order (height, width)

  return image_shape


def get_token_valid_mask(
    text_tokens, ignore_types, begin_token_id, end_token_id
):
  """Get valid mask for text tokens."""
  # negation
  if ignore_types.startswith('^'):
    negation = True
    ignore_types = ignore_types[1:]
  else:
    negation = False
  if isinstance(ignore_types, str):
    ignore_types = ignore_types.split(',')
  valid_mask = jnp.ones_like(text_tokens)
  if 'pad' in ignore_types:
    valid_mask = valid_mask * (text_tokens > 0)
  if 'begin' in ignore_types:
    valid_mask = valid_mask * (text_tokens != begin_token_id)
  if 'end' in ignore_types:
    valid_mask = valid_mask * (text_tokens != end_token_id)
  if 'end-1' in ignore_types:
    # shift to right
    shifted_text_tokens = jnp.pad(
        text_tokens[..., 1:],
        [(0, 0), (0, 0), (0, 1)],
        constant_values=0,
    )
    valid_mask = valid_mask * (shifted_text_tokens != end_token_id)
  if 'text' in ignore_types:
    valid_mask = valid_mask * (
        (text_tokens == begin_token_id)
        + (text_tokens == end_token_id)
        + (text_tokens == 0)
    )
  if negation:
    return valid_mask == 0
  else:
    return valid_mask > 0


def generate_point_label(
    rng,
    point_coords,
    valid_mask=None,
    prompt_drop_rate=0.0,
    train=False,
):
  """Sample point labels.

  Args:
    rng: PRNG Keys
    point_coords: (batch, num_prompts, num_points, 2)
    valid_mask: (batch, num_prompts)
    prompt_drop_rate: float
    train: bool

  Returns:
    point_labels: (...), 1 for positive, 0 for negative, -1 for ignored
  """
  if valid_mask is None:
    valid_mask = jnp.ones(point_coords.shape[:-2], dtype=jnp.uint8)
  # [..., max_text_tokens, num_points_per_token]
  valid_mask = jnp.broadcast_to(
      valid_mask[..., None], point_coords.shape[:-1]
  )

  if train and rng is not None:
    label_probs = jax.random.uniform(
        rng, shape=point_coords.shape[:-1]
    )
    point_labels = jnp.where(label_probs > prompt_drop_rate, 1, -1)
    point_labels = jnp.where(valid_mask > 0, point_labels, -1)

  else:
    point_labels = jnp.where(valid_mask > 0, 1, -1)

  # [batch_size, num_prompts, num_points]
  return point_labels


def concat_visual_features(visual_features_dict, feature_keys):
  """Resize and concat visual features."""
  seperator = ','
  resize_to_max_size = False
  if '+' in feature_keys:
    seperator = '+'
    resize_to_max_size = True
  feature_keys = feature_keys.split(seperator)
  if resize_to_max_size:
    image_embedding_size = max(
        visual_features_dict[feature_key].shape[1:3]
        for feature_key in feature_keys
    )
  else:
    image_embedding_size = visual_features_dict[feature_keys[0]].shape[1:3]
  image_embeddings = []
  for k in feature_keys:
    curr_embedding = visual_features_dict[k]
    if curr_embedding.shape[1:3] != image_embedding_size:
      curr_embedding = jax.image.resize(
          curr_embedding,
          (
              curr_embedding.shape[0],
              image_embedding_size[0],
              image_embedding_size[1],
              curr_embedding.shape[-1],
          ),
          method='bicubic',
      )
    image_embeddings.append(curr_embedding)
  image_embeddings = jnp.concatenate(image_embeddings, axis=-1)
  return image_embeddings


def get_image_size(visual_features_dict, feature_keys):
  """Get image size from visual features."""
  seperator = ','
  resize_to_max_size = False
  if '+' in feature_keys:
    seperator = '+'
    resize_to_max_size = True
  feature_keys = feature_keys.split(seperator)
  if resize_to_max_size:
    image_size = max(
        visual_features_dict[
            feature_key.replace('visual_features', 'image_size')
        ]
        for feature_key in feature_keys
    )
  else:
    image_size = visual_features_dict[
        feature_keys[0].replace('visual_features', 'image_size')
    ]
  return image_size


def get_first_possible_value(key, dic_list, default_value=None):
  """Get first possible value from a list of dictionaries."""
  for dic in dic_list:
    if key in dic:
      return dic[key]
  return default_value


def fuse_out_feat(all_hidden_states, fuse_method):
  """Fuse out features.

  Support concat and sum, e.g. concat:3,4, sum:-1,-2.


  Args:
    all_hidden_states: (List[jnp.array])
    fuse_method: str

  Returns:
    jnp.array
  """
  hidden_state_indices = [
      int(i) for i in fuse_method.split(':')[-1].split(',')
  ]
  features = [all_hidden_states[i] for i in hidden_state_indices]
  if fuse_method.startswith('concat:'):
    out_feature = jnp.concatenate(features, axis=-1)
  elif fuse_method.startswith('sum:'):
    out_feature = jnp.stack(features, axis=0).sum(axis=0)
  else:
    raise ValueError(f'Unsupported fuse: {fuse_method}')

  return out_feature
