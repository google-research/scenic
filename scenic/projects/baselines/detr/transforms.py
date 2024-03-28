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

"""Data augmentation transforms for data loading."""

from typing import Any, Dict
import tensorflow as tf

# TODO(aravindhm): Control randomness by passing Rndkey and splitting, etc.


def tf_float(t):
  return tf.cast(t, tf.float32)


def tf_int32(t):
  return tf.cast(t, tf.int32)


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


class RandomSelect:
  """Randomly selects between transforms1 and transforms2 ~ [p, 1 - p] ."""

  def __init__(self, transforms1, transforms2, p: float = 0.5):
    self.transforms1 = transforms1
    self.transforms2 = transforms2
    self.p = p

  def __call__(self, features):
    rnd = tf.random.uniform([], minval=0.0, maxval=1.0, dtype=tf.float32)
    if rnd < self.p:
      return self.transforms1(identity(features))
    else:
      return self.transforms2(identity(features))


class Resize:
  """Resizes image so that min side is of provided size."""

  def __init__(self, size, max_size=None):
    assert isinstance(size, int)
    self.size = tf.constant(size, dtype=tf.int32)
    self.max_size = max_size  # Max side after resize should be < max_size.

  def __call__(self, features):
    return resize(features, self.size, self.max_size)


class RandomResize:
  """Randomly resizes image so that min side is one of provided sizes."""

  def __init__(self, sizes, max_size=None):
    assert isinstance(sizes, (list, tuple))
    self.sizes = tf.constant(sizes, dtype=tf.int32)
    self.max_size = max_size  # Max side after resize should be < max_size.

  def __call__(self, features):
    # Randomly picks a size.
    logits = tf.zeros([1, len(self.sizes)])
    idx = tf.random.categorical(logits, 1)[0, 0]
    size = self.sizes[idx]
    return resize(features, size, self.max_size)


class NormalizeBoxes:
  """Map boxes from xyxy to cxcywh and normalize to [0,1]."""

  def __call__(self, features):
    h, w = get_hw(features, dtype=tf.float32)
    if 'boxes' in features['label']:
      boxes = features['label']['boxes']

      # Maps boxes from xyxy to cxcywh.
      x0, y0, x1, y1 = tf.split(boxes, 4, axis=-1)
      boxes = tf.concat([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)],
                        axis=-1)

      # Normalizes boxes to [0, 1].
      boxes = boxes / tf.reshape(tf.stack([w, h, w, h]), shape=[1, 4])
      features['label']['boxes'] = boxes

    return features


class GetCocoInstanceMasks:
  """Constructs the instance mask for each object in the original image."""

  def __call__(self, features):
    """Constructs the instance mask for each object in the original image.

    Args:
      features: dict; Contains a single unbatched input example with keys
        `inputs` and `label`. `label` contains the key `masks`, which is a COCO
        panoptic image (with object IDs encoded in RGB).

    Returns:
      An updated feature dict with a mask tensor of shape [num_objects, H, W].
    """

    if 'masks' not in features['label']:
      raise ValueError('masks are required when using this transformation.')

    panoptic_image = features['label']['masks']
    object_ids = features['label']['objects/id']

    # Number of objects in the image.
    num_objects = tf.shape(object_ids)[0]
    object_ids_map = tf.reshape(object_ids, (num_objects, 1, 1))
    object_ids_map = tf.cast(object_ids_map, dtype=tf.int32)

    # Reconstruct object ids from the panoptic image per-pixel segment id.
    panoptic_image = tf.cast(panoptic_image, dtype=tf.int32)
    # convert RGB from panoptic_image to object id
    object_id_mask = (
        panoptic_image[:, :, 0] + panoptic_image[:, :, 1] * 256 +
        panoptic_image[:, :, 2] * 256 * 256)

    # `instance_masks` contains values in {0,1} to indicate presence of this
    # object for each pixel. instance_masks.shape = [num_objects, height, width]
    instance_masks = tf_int32(tf.equal(object_id_mask, object_ids_map))

    # Add channel dimension. This makes things like hflip easier because it will
    # treat the leading dimension (num_objects) as batch dimension:
    instance_masks = instance_masks[..., tf.newaxis]

    features['label']['masks'] = tf.identity(instance_masks, 'masks')

    return features


class GetCocoBboxFromMasks:
  """Compute the bounding boxes and object classes from segmentation masks."""

  def __init__(self, keep_masks):
    self.keep_masks = keep_masks

  def __call__(self, features):
    """Compute the bounding boxes and object classes.

    based on:
    https://github.com/facebookresearch/detr/blob/6a608d3c3a9e10403379f7e7f65f48e26d03f645/util/box_ops.py#L64

    Args:
      features: dict; Contains a single unbatched input example with keys
        `inputs` and `label`. `label` contains the field `masks` with a tensor
        of shape [num_objects, H, W, 1].

    Returns:
      An updated feature dict with box coordinates and labels.
    """
    instance_masks = features['label']['masks'][..., 0]  # Remove channel dim.
    object_labels = features['label']['labels']
    h = tf.shape(instance_masks)[-2]
    w = tf.shape(instance_masks)[-1]
    x, y = tf.meshgrid(tf.range(w), tf.range(h))

    def get_axis_min_max(axis_grid):
      """Calculates the min and max pixel along the given axis for each object.

      Args:
        axis_grid: tensor; grid of axis over which min and max is calculated.

      Returns:
        tuple(int): axis_min and axis_max
      """
      axis_mask = instance_masks * tf.expand_dims(axis_grid, axis=0)
      axis_max = tf.math.reduce_max(axis_mask, axis=(1, 2))
      axis_min = tf.math.reduce_min(
          tf.where(instance_masks == 0, int(1e8), axis_mask), axis=(1, 2))
      return axis_min, axis_max

    x_min, x_max = get_axis_min_max(x)
    y_min, y_max = get_axis_min_max(y)

    # Stack objects to form the shape [num_objects, 4].
    bbox = tf.stack([x_min, y_min, x_max, y_max], 1)

    # Filter out objects that do not fall into the resized/cropped image
    # non-existing objects are those with:
    #  [x_min, y_min, x_max, y_max] = [int(1e8), int(1e8), 0, 0]
    existing_objects_bool = tf.reduce_all(
        tf.not_equal(bbox, [int(1e8), int(1e8), 0, 0]), axis=-1)
    existing_objects = tf.where(existing_objects_bool)
    existing_objects = tf.squeeze(existing_objects, -1)

    # Gather rows that correspond to the existing objects.
    instance_masks = tf.gather(instance_masks, existing_objects)
    bbox = tf.gather(bbox, existing_objects)
    object_labels = tf.gather(object_labels, existing_objects)
    object_labels = object_labels + 1  # for the padded objects

    # Maps boxes to cxcywh format (the loss function expects to receive cxcywh).
    bbox = tf.cast(bbox, tf.float32)
    x0, y0, x1, y1 = tf.split(bbox, 4, axis=-1)
    bbox = tf.concat([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)],
                     axis=-1)

    # Normalizes boxes to [0, 1].
    size = tf.reshape(tf.stack([w, h, w, h]), shape=[1, 4])
    size = tf.cast(size, tf.float32)
    bbox = bbox / size

    # Add channel dim back:
    instance_masks = instance_masks[..., tf.newaxis]

    tf.debugging.assert_shapes((
        (bbox, ['num_objects', 4]),
        (object_labels, ['num_objects']),
        (instance_masks, ['num_objects', 'h', 'w', 1])))

    if self.keep_masks:
      features['label']['masks'] = tf.identity(instance_masks, 'masks')
    else:
      del features['label']['masks']
    features['label']['boxes'] = tf.identity(bbox, 'boxes')
    features['label']['labels'] = tf.identity(object_labels, 'labels')

    return features


class InitPaddingMask:
  """Create a `padding_mask` of `ones` to match the current unpadded image."""

  def __call__(self, features):
    h, w = get_hw(features, dtype=tf.int32)
    # padding_mask is initialized as ones. It will later be padded with zeros.
    features['padding_mask'] = tf.ones((h, w), dtype=tf.float32)
    return features


class RandomSizeCrop:
  """Crop a random sized region from the image."""

  def __init__(self, min_size, max_size):
    self.min_size = min_size
    self.max_size = max_size

  def __call__(self, features):
    h, w = get_hw(features, dtype=tf.int32)
    wcrop = tf.random.uniform([], self.min_size, tf.minimum(w, self.max_size),
                              dtype=tf.int32)
    hcrop = tf.random.uniform([], self.min_size, tf.minimum(h, self.max_size),
                              dtype=tf.int32)

    i = tf.random.uniform([], 0, h - hcrop + 1, dtype=tf.int32)
    j = tf.random.uniform([], 0, w - wcrop + 1, dtype=tf.int32)
    region = (i, j, hcrop, wcrop)

    return crop(features, region)


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
    # TODO(aravindhm): should we update the area here if there are no boxes?
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
