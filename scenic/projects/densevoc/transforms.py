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

This is modified from scenic.projects.baseline.detr.transforms.
The original crop transform filtered out empty objects and have fixed keys.
This version adds customized keys in the filtering.
"""
from scenic.projects.baselines.centernet import transforms as centernet_transforms
import tensorflow as tf
INF = 1000000


def tf_float(t):
  return tf.cast(t, tf.float32)


def crop(features, region, additional_keys=()):
  """The same as detr crop, with additional keys (e.g., object captioning.)."""
  image = features['inputs']
  target = features['label']
  i, j, h, w = region

  cropped_image = image[i:i+h, j:j+w, :]
  features['inputs'] = cropped_image

  target['size'] = tf.stack([h, w])

  fields = ['labels', 'area', 'is_crowd', 'objects/id'] + list(additional_keys)

  boxes = target['boxes']
  cropped_boxes = boxes - tf_float(tf.expand_dims(
      tf.stack([j, i, j, i]), axis=0))
  cropped_boxes = tf.minimum(
      tf.reshape(cropped_boxes, [-1, 2, 2]),
      tf.reshape(tf_float(tf.stack([w, h])), [1, 1, 2]))

  cropped_boxes = tf.clip_by_value(cropped_boxes, 0, INF)
  target['boxes'] = tf.reshape(cropped_boxes, [-1, 4])
  fields.append('boxes')

  if 'area' in target:
    area = tf.reduce_prod(cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :],
                          axis=1)
    target['area'] = area

  # Removes elements for which the boxes or masks that have zero area.
  cropped_boxes = tf.reshape(target['boxes'], [-1, 2, 2])
  keep = tf.logical_and(cropped_boxes[:, 1, 0] > cropped_boxes[:, 0, 0],
                        cropped_boxes[:, 1, 1] > cropped_boxes[:, 0, 1])

  for field in fields:
    if field in target:
      target[field] = target[field][keep]

  features['label'] = target
  return features


class FixedSizeCropWithAdditionalKeys:
  """Crop a random sized region from the image."""

  def __init__(self, crop_size, additional_keys=('text_tokens',)):
    self.crop_size = crop_size
    self.additional_keys = additional_keys

  def __call__(self, features):
    h, w = centernet_transforms.get_hw(features, dtype=tf.int32)
    wcrop = tf.cast(tf.minimum(w, self.crop_size), tf.int32)
    hcrop = tf.cast(tf.minimum(h, self.crop_size), tf.int32)
    i = tf.random.uniform([], 0, h - hcrop + 1, dtype=tf.int32)
    j = tf.random.uniform([], 0, w - wcrop + 1, dtype=tf.int32)
    region = (i, j, hcrop, wcrop)
    return crop(
        features, region, additional_keys=self.additional_keys)

