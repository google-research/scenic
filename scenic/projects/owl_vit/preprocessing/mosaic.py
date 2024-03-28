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

"""Tools for creating mosaic datasets."""
import dataclasses
from typing import Tuple

from scenic.projects.owl_vit.preprocessing import image_ops
from scenic.projects.owl_vit.preprocessing import modalities
import tensorflow as tf


@dataclasses.dataclass
class CreateMosaic:
  """Batch processing op that assembles mosaic images.

  The op expects three batch dimensions:
  [device count, local batch size, (mosaic size) ** 2].

  The op then concatenates the examples from the third batch dim into mosaic
  images and merges labels.

  Mosaic approach:
  1. Run deterministic_data.create_dataset with standard preprocessing and
     batch to num_tiles. But do not apply GV-to-Scenic conversion.
  2. Convert each batch into one single image.
  3. Apply another round of pad-or-crop preprocessing.
  4. Apply GV-to-Sceinic conversion.
  5. Batch to final batch size as in deterministic_data.create_dataset.
  """

  mosaic_size: int
  instance_feature_keys: Tuple[str, ...] = (
      modalities.INSTANCE_LABELS, modalities.INSTANCE_TEXT_LABELS,
      modalities.ANNOTATION_ID, modalities.AREA)

  def __call__(self, features: image_ops.Features) -> image_ops.Features:

    # Merge images:
    features[modalities.IMAGE] = _image_tiles_to_mosaic(
        features[modalities.IMAGE], self.mosaic_size)

    # Merge scalar features:
    instance_padding_mask = tf.not_equal(
        features[modalities.INSTANCE_LABELS], -1)

    for k in self.instance_feature_keys:
      if k in features:
        features[k] = _merge_instances(features[k], instance_padding_mask)

    # Special cases:
    if modalities.NEGATIVE_LABELS in features:
      features[modalities.NEGATIVE_LABELS] = _merge_instances(
          features[modalities.NEGATIVE_LABELS],
          tf.not_equal(features[modalities.NEGATIVE_LABELS], -1))

      # Update negative labels to account for instances in all tiles:
      features[modalities.NEGATIVE_LABELS] = tf.sparse.to_dense(
          tf.sets.difference(
              features[modalities.NEGATIVE_LABELS][tf.newaxis, ...],
              tf.cast(features[modalities.INSTANCE_LABELS][tf.newaxis, ...],
                      features[modalities.NEGATIVE_LABELS].dtype)))[0]  # pytype: disable=attribute-error  # allow-recursive-types

    if modalities.NEGATIVE_TEXT_LABELS in features:
      features[modalities.NEGATIVE_TEXT_LABELS] = _merge_instances(
          features[modalities.NEGATIVE_TEXT_LABELS],
          tf.not_equal(features[modalities.NEGATIVE_TEXT_LABELS], ''))

      # Update negative labels to account for instances in all tiles:
      features[modalities.NEGATIVE_TEXT_LABELS] = tf.sparse.to_dense(
          tf.sets.difference(
              features[modalities.NEGATIVE_TEXT_LABELS][tf.newaxis, ...],
              features[modalities.INSTANCE_TEXT_LABELS][tf.newaxis, ...]))[0]

    if modalities.NOT_EXHAUSTIVE_LABELS in features:
      features[modalities.NOT_EXHAUSTIVE_LABELS] = _merge_instances(
          features[modalities.NOT_EXHAUSTIVE_LABELS],
          tf.not_equal(features[modalities.NOT_EXHAUSTIVE_LABELS], -1))

    if modalities.CROWD in features:
      features[modalities.CROWD] = _merge_instances(
          features[modalities.CROWD],
          tf.not_equal(features[modalities.CROWD], -1))

    # Merge boxes:
    if modalities.BOXES in features:
      features[modalities.BOXES] = _box_tiles_to_mosaic(
          features[modalities.BOXES], self.mosaic_size, instance_padding_mask)

    if image_ops.SEED_KEY in features:
      features[image_ops.SEED_KEY] = features[image_ops.SEED_KEY][0]

    return features


def _image_tiles_to_mosaic(image: tf.Tensor, mosaic_size: int) -> tf.Tensor:
  """Reshapes a batch of image tiles into a mosaic."""
  assert len(image.shape) == 4, (
      f'Expect shape [num_tiles, h, w, c], got {image.shape}.')

  # Get dynamic image shape:
  shape = tf.shape(image)
  num_tiles, h, w, c = shape[0], shape[1], shape[2], shape[3]

  tf.debugging.assert_equal(
      num_tiles, mosaic_size**2,
      'The first dimension must contain exactly self.mosaic_size ** 2 '
      f'elements, but got images of shape {image.shape}')

  image = tf.reshape(image, [mosaic_size, mosaic_size, h, w, c])
  image = tf.transpose(image, [0, 2, 1, 3, 4])
  image = tf.reshape(image, [h * mosaic_size, w *mosaic_size, c])
  return image


def _box_tiles_to_mosaic(boxes: tf.Tensor, mosaic_size: int,
                         padding_mask: tf.Tensor) -> tf.Tensor:
  """Reshapes a batch of per-tile boxes into boxes for a mosaic."""
  assert len(boxes.shape) == 3, (
      'Expect shape [num_tiles, n, 4], got {image.shape}.')

  num_tiles = boxes.shape[0]
  assert num_tiles == mosaic_size**2, (
      'The first dimension must contain exactly self.mosaic_size ** 2 '
      f'elements, but got boxes of shape {boxes.shape}')

  # Get offsets by which boxes must be shifted at each tile:
  x_offset, y_offset = tf.meshgrid(
      range(mosaic_size), range(mosaic_size), indexing='xy')
  x_offset = tf.cast(tf.reshape(x_offset, [num_tiles, 1, 1]), tf.float32)
  y_offset = tf.cast(tf.reshape(y_offset, [num_tiles, 1, 1]), tf.float32)

  # Shift boxes by the tile offsets:
  y0, x0, y1, x1 = tf.split(boxes, 4, axis=-1)
  boxes = tf.concat(
      [y0 + y_offset, x0 + x_offset, y1 + y_offset, x1 + x_offset], axis=-1)

  # Reshape from [num_tiles, n, 4] to [num_tiles * n, 4]:
  boxes = tf.reshape(boxes, [-1, 4])
  padding_mask = tf.reshape(padding_mask, [-1])

  # Remove padding instances:
  boxes = boxes[padding_mask]

  # Renormalize boxes:
  return boxes / mosaic_size


def _merge_instances(labels: tf.Tensor, padding_mask: tf.Tensor) -> tf.Tensor:
  """Combine a batch of padded instance labels into a single-example label."""
  tf.debugging.assert_equal(tf.shape(labels), tf.shape(padding_mask))
  labels = tf.reshape(labels, [-1])
  padding_mask = tf.reshape(padding_mask, [-1])
  return labels[padding_mask]
