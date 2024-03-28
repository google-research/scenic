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

"""Transform functions for preprocessing."""

from typing import Any, Optional, Sequence, Tuple, Union

import tensorflow as tf


SizeTuple = Tuple[tf.Tensor, tf.Tensor]  # (height, width).
Self = Any

PADDING_VALUE = -1
PADDING_VALUE_STR = b""


def get_padding_value(dtype):
  """Helper function for determing datatype-appropriate padding values."""
  if dtype in (tf.int32, tf.int64):
    return PADDING_VALUE
  elif dtype == tf.string:
    return PADDING_VALUE_STR
  return None


def get_box_area(
    boxes: tf.Tensor, image_size: Optional[SizeTuple] = None) -> tf.Tensor:
  """Calculate area using box coordinates.

  Arguments:
    boxes: Relative box coordinates.
    image_size: If provided, size of the image to use for calculating the
      absolute box area.

  Returns:
    Box area.
  """
  box_height = boxes[..., 2] - boxes[..., 0]
  box_width = boxes[..., 3] - boxes[..., 1]
  area = box_height * box_width

  if image_size is not None:
    image_height, image_width = image_size
    image_height = tf.cast(image_height, area.dtype)
    image_width = tf.cast(image_width, area.dtype)
    area *= image_height * image_width
  return area


def box_iou(boxes1: tf.Tensor, boxes2: tf.Tensor, eps=1e-6) -> Tuple[
    tf.Tensor, tf.Tensor]:
  """Computes IoU (Intesection over Union) between two sets of boxes.

  See https://en.wikipedia.org/wiki/Jaccard_index for definition and visual
  example of IoU for bounding boxes.

  Boxes are in [xmin, ymin, xmax, ymax] format.

  Args:
    boxes1: Bounding-boxes in shape [bs, n, 4].
    boxes2: Bounding-boxes in shape [bs, m, 4].
    eps: Small floating point number used to avoid division by zero.

  Returns:
    Pairwise IoU and union matrices of shape [bs, n, m].
  """

  # First, compute box areas. These will be used later for computing the
  # union.
  width_height1 = boxes1[..., 2:] - boxes1[..., :2]
  area1 = width_height1[..., 0] * width_height1[..., 1]  # [bs, n]

  width_height2 = boxes2[..., 2:] - boxes2[..., :2]
  area2 = width_height2[..., 0] * width_height2[..., 1]  # [bs, m]

  # Compute pairwise top-left and bottom-right corners of the intersection
  # of the boxes.
  left_top = tf.maximum(boxes1[..., :, None, :2],
                        boxes2[..., None, :, :2])  # [bs, n, m, 2].
  right_bottom = tf.minimum(boxes1[..., :, None, 2:],
                            boxes2[..., None, :, 2:])  # [bs, n, m, 2].

  # Intersection = area of the box defined by [left_top, right_bottom].
  width_height = tf.maximum(right_bottom - left_top, 0.0)  # [bs, n, m, 2].
  intersection = width_height[..., 0] * width_height[..., 1]  # [bs, n, m].

  # Union = sum of areas - intersection.
  union = area1[..., :, None] + area2[..., None, :] - intersection

  iou = intersection / (union + eps)

  return iou, union


def get_within_bounds_crop_slice(
    begin: tf.Tensor,
    size: tf.Tensor,
    image_shape: Union[tf.TensorShape, tf.Tensor]
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes a within bounds crop slice from slice & image shape.

  Given a potentially outside of image bound crop slice, return a crop slice
  that strictly fall within the bound of the image.

  Args:
    begin: Beginning of the slice. Assumed to have 3 elements (W, H, C) with
      the channel slice starting at 0.
    size: Size of the slice. Assumed to have 3 elemets (W, H, C) with the
      channel slice covering the entire shape (i.e. equal to -1).
    image_shape: Size of the image being sliced.

  Returns:
    Updated begin and size that strictly fall within the image.
  """
  crop_ymin, crop_xmin, _ = tf.unstack(begin, axis=0)
  crop_height, crop_width, _ = tf.unstack(size, axis=0)
  crop_ymax = crop_ymin + crop_height
  crop_xmax = crop_xmin + crop_width
  ymax = tf.minimum(crop_ymax, image_shape[0])
  xmax = tf.minimum(crop_xmax, image_shape[1])
  ymin, xmin, _ = tf.unstack(tf.maximum(begin, 0), axis=0)
  begin = tf.stack([ymin, xmin, 0], axis=0)
  size = tf.stack([ymax - ymin, xmax - xmin, -1], axis=0)
  return begin, size


def get_padding_params_from_crop_slice(
    begin: tf.Tensor, size: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes padding parameters for (possibly out of bounds) crop slice.

  Given a crop slice that potentially falls outside of the image, calculates
  offset and size for the two spatial dimensions.

  Args:
    begin: Beginning of the slice. Assumed to have 3 elements (W, H, C) with
      the channel slice starting at 0.
    size: Size of the slice. Assumed to have 3 elemets (W, H, C) with the
      channel slice covering the entire shape (i.e. equal to -1).

  Returns:
    Offset and size for height and width.
  """
  crop_ymin, crop_xmin, _ = tf.unstack(begin, axis=0)
  crop_height, crop_width, _ = tf.unstack(size, axis=0)
  ymin, xmin, _ = tf.unstack(tf.maximum(begin, 0), axis=0)
  offset_y = tf.maximum(0, ymin - crop_ymin)
  offset_x = tf.maximum(0, xmin - crop_xmin)
  return offset_y, offset_x, crop_height, crop_width


def get_dynamic_size(feature: tf.Tensor,
                     dtype=tf.int32,
                     has_channel_dim: bool = True) -> SizeTuple:
  """Returns dynamic size (height, width) of image, video, or mask."""
  if feature.dtype.name == "string":  # Encoded jpeg.
    shape = tf.io.extract_jpeg_shape(feature)
  else:
    shape = tf.shape(feature)
    if has_channel_dim:
      assert shape.shape[0] >= 3, "Expected: [..., height, width, channels]"
      shape = shape[-3:-1]
    else:
      assert shape.shape[0] >= 2, "Expected: [..., height, width]"
      shape = shape[-2:]
  h = tf.cast(shape[0], dtype=dtype)
  w = tf.cast(shape[1], dtype=dtype)
  return h, w


def crop_or_pad_boxes(boxes: tf.Tensor, top: int, left: int, height: int,
                      width: int, h_orig: tf.Tensor, w_orig: tf.Tensor):
  """Transforms the relative box coordinates according to the frame crop.

  Note that, if height/width are larger than h_orig/w_orig, this function
  implements the equivalent of padding.

  Args:
    boxes: Tensor of bounding boxes with shape (..., 4).
    top: Top of crop box in absolute pixel coordinates.
    left: Left of crop box in absolute pixel coordinates.
    height: Height of crop box in absolute pixel coordinates.
    width: Width of crop box in absolute pixel coordinates.
    h_orig: Original image height in absolute pixel coordinates.
    w_orig: Original image width in absolute pixel coordinates.
  Returns:
    Boxes tensor with same shape as input boxes but updated values.
  """
  # Bounding boxes: [num_instances, 4]
  assert len(boxes.shape) == 2
  assert boxes.shape[-1] == 4
  seq_len = tf.shape(boxes)[0]
  not_padding = tf.reduce_any(boxes != PADDING_VALUE, axis=-1)

  # Transform the box coordinates.
  a = tf.cast(tf.stack([h_orig, w_orig]), tf.float32)
  b = tf.cast(tf.stack([top, left]), tf.float32)
  c = tf.cast(tf.stack([height, width]), tf.float32)
  boxes = tf.reshape((tf.reshape(boxes, (seq_len, 1, 2, 2)) * a - b) / c,
                     (seq_len, 1, 4))

  # Filter the valid boxes.
  boxes = tf.minimum(tf.maximum(boxes, 0.0), 1.0)
  boxes = tf.reshape(boxes, (seq_len, 4))
  boxes = tf.where(not_padding[..., tf.newaxis], boxes, PADDING_VALUE)

  return boxes


def crop_or_pad_sequence(seq, length, allow_crop=True):
  """Crops or pads a sequence of scalars."""
  paddings = [[0, length - tf.shape(seq)[0]]] + [(0, 0)] * (len(seq.shape) - 1)
  if allow_crop:
    paddings = tf.maximum(paddings, 0)
  if seq.dtype == tf.string:
    padded = tf.pad(seq, paddings, constant_values="")
  elif seq.dtype == tf.bool:
    padded = tf.pad(seq, paddings, constant_values=False)
  else:
    padded = tf.pad(seq, paddings, constant_values=-1)
  if allow_crop:
    padded = padded[:length]
    padded.set_shape([length] + seq.shape[1:])
  return padded


def get_paddings(image_shape: tf.Tensor,
                 size: Union[int, Tuple[int, int], Sequence[int]],
                 pre_spatial_dim: Optional[int] = None,
                 allow_crop: bool = True,
                 mode: str = "bottom_right") -> tf.Tensor:
  """Returns paddings tensors for tf.pad operation.

  Args:
    image_shape: The shape of the Tensor to be padded. The shape can be
      [..., N, H, W, C] or [..., H, W, C]. The paddings are computed for H, W
      and optionally N dimensions.
    size: The total size for the H and W dimensions to pad to.
    pre_spatial_dim: Optional, additional padding dimension before the spatial
      dimensions. It is only used if given and if len(shape) > 3.
    allow_crop: If size is bigger than requested max size, padding will be
      negative. If allow_crop is true, negative padding values will be set to 0.
    mode: Padding mode, "bottom_right" or "central".

  Returns:
    Paddings the given tensor shape.
  """
  assert image_shape.shape.rank == 1
  if isinstance(size, int):
    size = (size, size)
  h, w = image_shape[-3], image_shape[-2]
  if mode == "bottom_right":
    top, left = 0, 0
  elif mode == "central":
    top, left = (size[0] - h) // 2, (size[1] - w) // 2
  else:
    raise ValueError(f"Unknown padding mode: {mode}")
  # Spatial padding.
  paddings = [
      tf.stack([top, size[0] - h - top]),
      tf.stack([left, size[1] - w - left]),
      tf.stack([0, 0])
  ]
  ndims = image_shape.shape[0]  # pytype: disable=wrong-arg-types
  # Prepend padding for temporal dimension or number of instances.
  if pre_spatial_dim is not None and ndims > 3:
    paddings = [[0, pre_spatial_dim - image_shape[-4]]] + paddings
  # Prepend with non-padded dimensions if available.
  if ndims > len(paddings):
    paddings = [[0, 0]] * (ndims - len(paddings)) + paddings
  if allow_crop:
    paddings = tf.maximum(paddings, 0)
  return tf.stack(paddings)


def assert_boxes_are_relative(boxes: tf.Tensor):
  """Checks that boxes conform to the relative (normalized) format."""
  tf.debugging.assert_type(
      boxes, tf.float32,
      f"Expected boxes to be relative (float dtype), got {boxes.dtype}.")
  not_padding = tf.reduce_any(boxes != -1, axis=-1)
  tf.debugging.assert_greater_equal(
      boxes[not_padding], 0., "Relative boxes must be >= 0.")
  tf.debugging.assert_less_equal(
      boxes[not_padding], 1., "Relative boxes must be <= 1.")
