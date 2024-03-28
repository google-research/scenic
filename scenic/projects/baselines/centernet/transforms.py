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

"""Data augmentation transforms for data loading.

Forked and modified from
https://github.com/google-research/scenic/blob/main/scenic/projects/baselines/
detr/transforms.py
"""

from typing import Any, Dict
import tensorflow as tf


class FixedSizeCrop:
  """Crop a random sized region from the image."""

  def __init__(self, crop_size):
    self.crop_size = crop_size

  def __call__(self, features):
    h, w = get_hw(features, dtype=tf.int32)
    wcrop = tf.cast(tf.minimum(w, self.crop_size), tf.int32)
    hcrop = tf.cast(tf.minimum(h, self.crop_size), tf.int32)
    i = tf.random.uniform([], 0, h - hcrop + 1, dtype=tf.int32)
    j = tf.random.uniform([], 0, w - wcrop + 1, dtype=tf.int32)
    region = (i, j, hcrop, wcrop)
    return crop(features, region)


class RandomRatioResize:
  """EfficientNet data augmentation. First resize than crop a fixed size."""

  def __init__(self, scale_range, target_size):
    self.min_scale = scale_range[0]
    self.max_scale = scale_range[1]
    self.target_size = target_size

  def __call__(self, features):
    ratio = tf.random.uniform(
        [], self.min_scale, self.max_scale, dtype=tf.float32)
    size = tf.cast(tf.cast(self.target_size, tf.float32) * ratio, tf.int32)
    return resize(features, size, max_size=size)


class InitPaddingMask:
  """Create a `padding_mask` of `ones` to match the current unpadded image."""

  def __call__(self, features):
    h, w = get_hw(features, dtype=tf.int32)
    # padding_mask is initialized as ones. It will later be padded with zeros.
    features['padding_mask'] = tf.ones((h, w), dtype=tf.float32)
    return features


class RandomHorizontalFlip:
  """Horizontally flip image and boxes [cxcywh format] with probability `p`."""

  def __init__(self, p: float = 0.5):
    self.p = p

  def __call__(self, features):
    rnd = tf.random.uniform([], minval=0.0, maxval=1.0, dtype=tf.float32)
    if rnd < self.p:
      return hflip(identity(features))  # Identity helps avoid autograph errors.
    else:
      return identity(features)


class Resize:
  """Resizes image so that min side is of provided size."""

  def __init__(self, size, max_size=None):
    assert isinstance(size, int)
    self.size = tf.constant(size, dtype=tf.int32)
    self.max_size = max_size  # Max side after resize should be < max_size.

  def __call__(self, features):
    return resize(features, self.size, self.max_size)


class Compose:
  """Compose several transforms together.

  Attributes:
    transforms (list of ``Transform`` objects): list of transforms to compose.

  """

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, features):
    for t in self.transforms:
      features = t(features)
      if 'masks' in features['label']:
        tf.debugging.assert_shapes(
            shapes=(
                (features['label']['masks'], ['n', 'w', 'h', 1]),
                (features['inputs'], [..., 'w', 'h', 3]),
                (features['label']['labels'], ['n']),
            ),
            message=f'Shape mismatch after transformation {t.__class__}')

    return features

  def __repr__(self):
    format_string = self.__class__.__name__ + '('
    for t in self.transforms:
      format_string += '\n'
      format_string += '    {0}'.format(t)
    format_string += '\n)'
    return format_string


def tf_float(t):
  return tf.cast(t, tf.float32)


def tf_int32(t):
  return tf.cast(t, tf.int32)


def identity(features: Dict[str, Any]) -> Dict[str, Any]:
  """tf.identity for nested dictionary of Tensors."""
  out = {}
  for k, v in features.items():
    if isinstance(v, tf.Tensor):
      out[k] = tf.identity(v)
    elif isinstance(v, dict):
      out[k] = identity(v)
    else:
      raise NotImplementedError(f'{v}\'s type that is unsupported by identity.')

  return out


def hflip(features):
  """Flip an image, boxes [xyxy un-normalized] (, and masks) horizontally."""
  image = features['inputs']
  target = features['label']

  flipped_image = tf.image.flip_left_right(image)

  if 'boxes' in target:
    # Flips the boxes.
    _, w = get_hw(image, dtype=tf.float32)
    x0, y0, x1, y1 = tf.split(target['boxes'], 4, axis=-1)
    # Converts as [w - x1, y0, w - x0, y1] not [w - x1 - 1, w - x0 - 1, y1]
    # because these are float coordinates not pixel indices.
    target['boxes'] = tf.concat([w - x1, y0, w - x0, y1], axis=-1)

  if 'masks' in target:
    target['masks'] = tf.image.flip_left_right(target['masks'])

  features['inputs'] = flipped_image
  features['label'] = target
  return features


def get_hw(features, dtype=tf.int32):
  """Return the height, width of image as float32 tf.Tensors."""
  if isinstance(features, dict):
    sz = tf.shape(features['inputs'])
  elif isinstance(features, tf.Tensor):
    sz = tf.shape(features)
  else:
    raise ValueError(f'Unknown type of object: {features}')

  h = tf.cast(sz[0], dtype=dtype)
  w = tf.cast(sz[1], dtype=dtype)
  return h, w


def get_size_with_aspect_ratio(image_size, size, max_size=None):
  """Output (h, w) such that smallest side in image_size resizes to size."""
  h, w = image_size[0], image_size[1]
  if max_size is not None:
    max_size = tf_float(max_size)
    min_original_size = tf_float(tf.minimum(w, h))
    max_original_size = tf_float(tf.maximum(w, h))
    if max_original_size / min_original_size * tf_float(size) > max_size:
      size = tf_int32(tf.floor(
          max_size * min_original_size / max_original_size))

  if (w <= h and tf.equal(w, size)) or (h <= w and tf.equal(h, size)):
    return (h, w)

  if w < h:
    ow = size
    oh = tf_int32(size * h / w)
  else:
    oh = size
    ow = tf_int32(size * w / h)

  return (oh, ow)


def resize(features, size, max_size=None):
  """Resize the image to min-side = size and adjust target boxes, area, mask.

  Args:
    features: dict; 'inputs' contains tf.Tensor image unbatched. 'label' is
      a dictionary of label information such a boxes, area, etc.
    size: tf.Tensor; Scalar for size of smallest sized after resize.
    max_size: int[Optional]; Scalar upper bound on resized image dimensions.

  Returns:
    Resized and adjusted features. Also features['size'] = (w, h) tuple.
  """
  image = features['inputs']
  target = features['label']

  # Resize the image while preserving aspect ratio.
  original_size = tf.shape(image)[0:2]
  new_size = get_size_with_aspect_ratio(original_size, size, max_size)
  rescaled_image = tf.image.resize(image, new_size)

  # Compute resize ratios for each dimension to be used for scaling boxes, area.
  r_height = tf_float(new_size[0] / original_size[0])
  r_width = tf_float(new_size[1] / original_size[1])

  if 'boxes' in target:
    x0, y0, x1, y1 = tf.split(target['boxes'], 4, axis=-1)
    target['boxes'] = tf.concat([x0 * r_width, y0 * r_height,
                                 x1 * r_width, y1 * r_height], axis=-1)

  if 'area' in target:
    area = target['area']
    scaled_area = tf_float(area) * (r_width * r_height)
    target['area'] = scaled_area

  target['size'] = tf.stack(new_size)

  if 'masks' in target:
    dtype = target['masks'].dtype
    rescaled_masks = tf.image.resize(
        tf_float(target['masks']),
        new_size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    tf.debugging.assert_shapes((
        (rescaled_image, [..., 'w', 'h', 3]),
        (rescaled_masks, [..., 'w', 'h', 1])))
    target['masks'] = tf.cast(rescaled_masks, dtype)

  features['inputs'] = rescaled_image
  features['label'] = target
  return features


def crop(features, region):
  """Crop the image + bbox (+ mask) to region.

  WARNING! Only use during train. In eval mode the original_size would need to
  be updated somehow.

  Args:
    features: DETR decoded input features.
    region: (i, j, h, w) tuple of the region to be cropped.

  Returns:
    Cropped features dictionary.
  """
  image = features['inputs']
  target = features['label']
  i, j, h, w = region

  cropped_image = image[i:i+h, j:j+w, :]
  features['inputs'] = cropped_image

  target['size'] = tf.stack([h, w])

  fields = ['labels', 'area', 'is_crowd', 'objects/id']

  if 'boxes' in target:
    boxes = target['boxes']
    cropped_boxes = boxes - tf_float(tf.expand_dims(
        tf.stack([j, i, j, i]), axis=0))
    cropped_boxes = tf.minimum(
        tf.reshape(cropped_boxes, [-1, 2, 2]),
        tf.reshape(tf_float(tf.stack([w, h])), [1, 1, 2]))
    cropped_boxes = tf.clip_by_value(cropped_boxes, 0, 1000000)
    target['boxes'] = tf.reshape(cropped_boxes, [-1, 4])
    fields.append('boxes')

    if 'area' in target:
      area = tf.reduce_prod(cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :],
                            axis=1)
      target['area'] = area

  if 'masks' in target:
    target['masks'] = target['masks'][..., i:i+h, j:j+w, :]
    fields.append('masks')

  # Removes elements for which the boxes or masks that have zero area.
  if 'boxes' in target or 'masks' in target:
    if 'boxes' in target:
      cropped_boxes = tf.reshape(target['boxes'], [-1, 2, 2])
      keep = tf.logical_and(cropped_boxes[:, 1, 0] > cropped_boxes[:, 0, 0],
                            cropped_boxes[:, 1, 1] > cropped_boxes[:, 0, 1])
    else:
      keep = tf.reduce_any(tf.not_equal(target['masks'], 0), axis=[1, 2, 3])

    for field in fields:
      if field in target:
        target[field] = target[field][keep]

  features['label'] = target
  return features
