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

"""Data augmentation transforms for data loading that support transforming points."""

from scenic.projects.baselines.centernet import transforms
import tensorflow as tf

identity = transforms.identity
get_hw = transforms.get_hw
get_size_with_aspect_ratio = transforms.get_size_with_aspect_ratio
tf_float = transforms.tf_float


def resize(features, size, max_size=None):
  """Resize the image to min-side = size and adjust target boxes, area, mask.

  Args:
    features: dict; 'inputs' contains tf.Tensor image unbatched. 'label' is a
      dictionary of label information such a boxes, area, etc.
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

  for key in ['boxes', 'prompt_boxes']:
    if key in target:
      x0, y0, x1, y1 = tf.split(target[key], 4, axis=-1)
      target[key] = tf.concat(
          [x0 * r_width, y0 * r_height, x1 * r_width, y1 * r_height], axis=-1
      )
  if 'points' in target:
    x, y = tf.split(target['points'], 2, axis=-1)
    target['points'] = tf.concat([x * r_width, y * r_height], axis=-1)

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
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    # tf.debugging.assert_shapes((
    #     (rescaled_image, [..., 'w', 'h', 3]),
    #     (rescaled_masks, [..., 'w', 'h', 1]),
    # ))
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

  cropped_image = image[i : i + h, j : j + w, :]
  features['inputs'] = cropped_image

  target['size'] = tf.stack([h, w])

  fields = [
      'labels',
      'area',
      'is_crowd',
      'objects/id',
      'text_tokens',
      'refexp_ids',
      'text_features',
  ]

  for key in ['boxes', 'prompt_boxes']:
    if key in target:
      boxes = target[key]
      cropped_boxes = boxes - tf_float(
          tf.expand_dims(tf.stack([j, i, j, i]), axis=0)
      )
      cropped_boxes = tf.minimum(
          tf.reshape(cropped_boxes, [-1, 2, 2]),
          tf.reshape(tf_float(tf.stack([w, h])), [1, 1, 2]),
      )
      cropped_boxes = tf.clip_by_value(cropped_boxes, 0, 1000000)
      target[key] = tf.reshape(cropped_boxes, [-1, 4])

      if key == 'boxes':
        fields.append('boxes')

        if 'area' in target:
          area = tf.reduce_prod(
              cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :], axis=1
          )
          target['area'] = area

  if 'points' in target:
    points = target['points']
    cropped_points = points - tf_float(tf.expand_dims(tf.stack([j, i]), axis=0))
    cropped_points = tf.minimum(
        cropped_points, tf.reshape(tf_float(tf.stack([w, h])), [1, 2])
    )
    cropped_points = tf.clip_by_value(cropped_points, 0, 1000000)
    target['points'] = cropped_points
    # NOTE: we skip adding points into the filed because we don't wanto filter
    # on points based on point index
    # fields.append('points')

  if 'masks' in target:
    # TODO(aravindhm): should we update the area here if there are no boxes?
    target['masks'] = target['masks'][..., i : i + h, j : j + w, :]
    fields.append('masks')

  # Removes elements for which the boxes or masks that have zero area.
  if 'boxes' in target or 'masks' in target:
    if 'boxes' in target:
      cropped_boxes = tf.reshape(target['boxes'], [-1, 2, 2])
      keep = tf.logical_and(
          cropped_boxes[:, 1, 0] > cropped_boxes[:, 0, 0],
          cropped_boxes[:, 1, 1] > cropped_boxes[:, 0, 1],
      )
    else:
      keep = tf.reduce_any(tf.not_equal(target['masks'], 0), axis=[1, 2, 3])

    for field in fields:
      if field in target:
        target[field] = target[field][keep]

  features['label'] = target
  return features


def hflip(features):
  """Flip an image, boxes [xyxy un-normalized] (, and masks) horizontally."""
  image = features['inputs']
  target = features['label']

  flipped_image = tf.image.flip_left_right(image)

  for key in ['boxes', 'prompt_boxes']:
    if key in target:
      # Flips the boxes.
      _, w = get_hw(image, dtype=tf.float32)
      x0, y0, x1, y1 = tf.split(target[key], 4, axis=-1)
      # Converts as [w - x1, y0, w - x0, y1] not [w - x1 - 1, w - x0 - 1, y1]
      # because these are float coordinates not pixel indices.
      target[key] = tf.concat([w - x1, y0, w - x0, y1], axis=-1)

  if 'points' in target:
    _, w = get_hw(image, dtype=tf.float32)
    x, y = tf.split(target['points'], 2, axis=-1)
    target['points'] = tf.concat([w - x, y], axis=-1)

  if 'masks' in target:
    target['masks'] = tf.image.flip_left_right(target['masks'])

  features['inputs'] = flipped_image
  features['label'] = target
  return features
