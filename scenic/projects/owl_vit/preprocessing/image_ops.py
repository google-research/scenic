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

"""Preprocessing ops for RGB images.

Unless otherwise mentioned images should be [h, w, 3] with pixel values as
tf.float32 in range [0, 1]. Bounding boxes are [num_objects, 4] in tf.float32 in
range [0, 1].
"""
import abc
import dataclasses
from typing import Optional, Sequence, Tuple, Union

from clu import preprocess_spec
from scenic.projects.owl_vit.preprocessing import modalities
from scenic.projects.owl_vit.preprocessing import transforms
import tensorflow as tf

Features = preprocess_spec.Features
SEED_KEY = preprocess_spec.SEED_KEY

all_ops = lambda: preprocess_spec.get_all_ops(__name__)

FEATURES_WITH_FIRST_INSTANCE_AXIS = [
    modalities.ANNOTATION_ID,
    modalities.AREA,
    modalities.BOXES,
    modalities.CROWD,
    modalities.INSTANCE_LABELS,
    modalities.INSTANCE_MULTI_LABELS,
    modalities.INSTANCE_TEXT_LABELS,
    modalities.INSTANCE_TEXT_MULTI_LABELS,
]

FEATURES_WITH_NO_INSTANCE_AXIS = [
    modalities.IMAGE_ID,
    modalities.IMAGE,
    modalities.NEGATIVE_LABELS,
    modalities.NEGATIVE_TEXT_LABELS,
    modalities.NOT_EXHAUSTIVE_LABELS,
    modalities.ORIGINAL_SIZE,
    modalities.TEXT_QUERIES_TOKENIZED,
    modalities.TEXT_QUERIES,
    SEED_KEY,
]


class ImagePreprocessOp(abc.ABC):
  """Base class for all image preprocess ops."""

  image_key: str = modalities.IMAGE  # tf.float32 in [0, 1]
  boxes_key: str = modalities.BOXES  # tf.float32 in [0, 1]

  def __call__(self, features: Features) -> Features:

    # Copy input features to ensure that they are not accidentally modified in
    # place:
    features = dict(features)

    # Apply to images:
    image_size = None
    if self.image_key in features:
      image_size = transforms.get_dynamic_size(features[self.image_key])
      features[self.image_key] = self.apply(features[self.image_key])

    # Apply to boxes:
    if self.boxes_key in features:
      if image_size is None:
        raise ValueError(
            "When providing box features, image features are also required, so "
            "that the image size can be computed.")
      features[self.boxes_key] = self.apply_boxes(features[self.boxes_key],
                                                  image_size)

    return features

  @abc.abstractmethod
  def apply(self, image: tf.Tensor) -> tf.Tensor:
    """Returns transformed image."""
    pass

  def apply_boxes(self, boxes: tf.Tensor,
                  image_size: transforms.SizeTuple) -> tf.Tensor:
    """Returns transformed boxes."""
    raise NotImplementedError(
        f"{self.__class__.__name__} is not implemented for bounding boxes.")

  @staticmethod
  def _get_instance_axis(feature_key):
    """Looks up the axis storing object instances for a given feature."""
    if feature_key in FEATURES_WITH_FIRST_INSTANCE_AXIS:
      return 0
    elif feature_key in FEATURES_WITH_NO_INSTANCE_AXIS:
      return None
    else:
      raise ValueError(
          f"Please specify instance axis for feature key {feature_key}.")


class RandomImagePreprocessOp(ImagePreprocessOp):
  """Base class for image ops that require a random seed."""

  def __call__(self, features: Features) -> Features:
    # Copy input features to ensure that they are not accidentally modified in
    # place:
    features = dict(features)

    if SEED_KEY not in features:
      raise ValueError(
          f"Random image preprocess op {type(self)} requires a random seed.")
    image_size = transforms.get_dynamic_size(features[self.image_key])
    rngs = tf.random.experimental.stateless_split(features[SEED_KEY])
    features[SEED_KEY] = rngs[0]
    op_seed = rngs[1]
    features[self.image_key] = self.apply(features[self.image_key], op_seed)
    if self.boxes_key in features:
      features[self.boxes_key] = self.apply_boxes(
          features[self.boxes_key], image_size=image_size, seed=op_seed)
    return features

  @abc.abstractmethod
  def apply(self, image: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:  # pytype: disable=signature-mismatch
    """Return the transformed image. The op can consume the seed."""
    pass

  def apply_boxes(self, boxes: tf.Tensor, *, image_size: transforms.SizeTuple,  # pytype: disable=signature-mismatch
                  seed: tf.Tensor) -> tf.Tensor:
    """Returns transformed boxes."""
    raise NotImplementedError(
        f"{self.__class__.__name__} is not implemented for bounding boxes.")


def _stateless_bernoulli_trial(seed: tf.Tensor, p: float = 0.5) -> tf.Tensor:
  return tf.greater(tf.random.stateless_uniform([], seed), p)


@dataclasses.dataclass(frozen=True)
class RandomFlipLeftRight(RandomImagePreprocessOp):
  """Randomly flips an image horizontally (left to right)."""

  def apply(self, image: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    return tf.cond(
        _stateless_bernoulli_trial(seed, p=0.5),
        lambda: tf.identity(image),
        lambda: tf.image.flip_left_right(image),
    )

  def apply_boxes(self, boxes: tf.Tensor, *, image_size: transforms.SizeTuple,
                  seed: tf.Tensor) -> tf.Tensor:
    del image_size
    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=-1)
    # To flip the boxes, swap the x coordinates and subtract them from 1:
    x_min, x_max = 1.0 - x_max, 1.0 - x_min
    flipped_boxes = tf.concat([y_min, x_min, y_max, x_max], axis=-1)
    return tf.cond(
        _stateless_bernoulli_trial(seed, p=0.5),
        lambda: tf.identity(boxes),
        lambda: flipped_boxes,
    )


class RandomCropBase(RandomImagePreprocessOp):
  """Randomly crops an image based on self._sample_random_crop_region."""

  def __call__(self, features: Features) -> Features:
    new_features = super().__call__(dict(features))
    return self._drop_degenerate_features(new_features, features)

  def apply(self, image: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    image_shape = tf.shape(image)
    begin, size = self._sample_random_crop_region(
        image_shape=image_shape, seed=seed)

    (offset_y, offset_x, crop_height, crop_width
     ) = transforms.get_padding_params_from_crop_slice(begin, size)
    begin, size = transforms.get_within_bounds_crop_slice(
        begin, size, image_shape)

    image = tf.slice(image, begin, size)  # Slice.
    image = tf.image.pad_to_bounding_box(  # Maybe pad.
        image, offset_height=offset_y, offset_width=offset_x,
        target_height=crop_height, target_width=crop_width)

    return tf.ensure_shape(image, (None, None, 3))

  def apply_boxes(self, boxes: tf.Tensor, *, image_size: transforms.SizeTuple,
                  seed: tf.Tensor) -> tf.Tensor:
    image_shape = tf.concat([image_size, [1]], axis=0)
    begin, size = self._sample_random_crop_region(
        image_shape=image_shape, seed=seed)
    top, left, _ = tf.unstack(begin)
    h, w, _ = tf.unstack(size)
    return transforms.crop_or_pad_boxes(
        boxes,
        top=top,
        left=left,
        height=h,
        width=w,
        h_orig=image_size[0],
        w_orig=image_size[1])

  def _drop_degenerate_features(
      self, features: Features, orig_features: Optional[Features] = None
    ) -> Features:
    """Drops degenerate (e.g. cropped out) boxes."""
    # Find degenerate boxes (i.e. boxes which have been cropped out of the
    # image).
    keep = []
    if self.boxes_key in features:
      # Keep boxes whose area is greater than 0:
      rel_area = transforms.get_box_area(features[self.boxes_key])
      keep.append(rel_area > 0.0)

      if (hasattr(self, "min_area_fraction") and orig_features is not None):
        area = transforms.get_box_area(
            features[self.boxes_key],
            transforms.get_dynamic_size(features[self.image_key]))
        orig_area = transforms.get_box_area(
            orig_features[self.boxes_key],
            transforms.get_dynamic_size(orig_features[self.image_key]))

        area_left = area / (orig_area + 1e-8)
        keep.append(area_left >= self.min_area_fraction)

    # If there are no boxes there are no degenerate objects to filter out.
    if not keep:
      return features

    # Keep instances for which all features are non-degenerate:
    keep = tf.reduce_all(tf.stack(keep, axis=0), axis=0)

    # Only keep non-degenerate instances:
    for key, feature in features.items():
      axis = self._get_instance_axis(key)
      if axis is not None:
        if axis == 0:
          features[key] = feature[keep]
        else:
          raise NotImplementedError("Instances must be along leading axis.")

    return features

  def _sample_random_crop_region(
      self,
      *,
      image_shape: tf.TensorShape,
      seed: tf.Tensor,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Randomly samples a crop region (bounding box) for random cropping.

    The region is represented as two tensors, `begin` and `size`, which can be
    passed directly to tf.slice to slice the sampled region from the original
    image.

    Args:
      image_shape: Shape of the image (height, width, channels) from which crops
        will be taken.
      seed: Random seed for tf.image.stateless_sample_distorted_bounding_box.

    Returns:
      Two tensors, begin and size, which can be passed directly to tf.slice.
    """
    raise NotImplementedError(f"{self.__class__.__name__} is not implemented.")


@dataclasses.dataclass(frozen=True)
class RandomCrop(RandomCropBase):
  """Randomly crops an image.

  Attr:
    aspect_ratio_range: An optional tuple of `floats`. The cropped area of the
      image must have an aspect `ratio = width / height` within this range.
    area_range: An optional tuple of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within this range.
    min_area_fraction: Bounding boxes will be removed if their area is less than
      min_area_fraction after cropping.
  """

  aspect_ratio_range: Tuple[float, float] = (0.75, 1.33)
  area_range: Tuple[float, float] = (0.3, 1.0)
  min_area_fraction: float = 0.0

  def _sample_random_crop_region(
      self,
      *,
      image_shape: tf.TensorShape,
      seed: tf.Tensor,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Randomly samples a crop region (bounding box)."""
    return tf.image.stateless_sample_distorted_bounding_box(
        image_shape,
        tf.zeros([0, 0, 4], tf.float32),
        seed=seed,
        area_range=self.area_range,
        aspect_ratio_range=self.aspect_ratio_range,
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)[:2]


@dataclasses.dataclass(frozen=True)
class Keep():
  """Keeps only the given keys."""

  keys: Sequence[str]

  def __call__(self, features: Features) -> Features:
    return {k: v for k, v in features.items() if k in self.keys}


@dataclasses.dataclass(frozen=True)
class Drop():
  """Drops the given keys."""

  keys: Sequence[str]
  ignore_missing_features: bool = False

  def __call__(self, features: Features) -> Features:
    if not self.ignore_missing_features:
      for k in self.keys:
        if k not in features:
          raise ValueError(
              f"Could not drop features '{k}'. Available features:"
              f" {list(features)}"
          )
    return {k: v for k, v in features.items() if k not in self.keys}


@dataclasses.dataclass(frozen=True)
class ResizeWithPad(ImagePreprocessOp):
  """Resizes image to a given size, adding padding to perserve the aspect ratio.

  Padding is added on the bottom and right.

  Attr:
    size: The new size of the image: either an integer H, where H is both the
      new height and width, or a tuple or list [H, W] of integers, where H and W
      are new image's height and width respectively.
    pad_value: Value to use for padding.
    antialias: Whether to use an anti-aliasing filter when downsampling an
      image.
  """

  size: Union[int, Tuple[int, int], Sequence[int]]
  pad_value: float = 0.5
  antialias: bool = False

  def _resize_image(self,
                    image,
                    pad_value,
                    method=tf.image.ResizeMethod.BILINEAR):
    """Applies the appropriate TF resizing function."""
    size = self.size
    if isinstance(size, int):
      size = (size, size)
    dtype = image.dtype
    num_channels = image.shape[-1]

    # Resize the image to fit into target size, while keeping aspect ratio:
    in_height, in_width = transforms.get_dynamic_size(image, tf.float32)
    ratio = tf.maximum(in_height / float(size[0]), in_width / float(size[1]))
    fit_height = tf.cast(tf.minimum(in_height / ratio, size[0]), tf.int32)
    fit_width = tf.cast(tf.minimum(in_width / ratio, size[1]), tf.int32)
    image = tf.image.resize(
        image, [fit_height, fit_width], method=method, antialias=self.antialias)

    # Pad to the same aspect ratio as the desired size:
    paddings = transforms.get_paddings(tf.shape(image), size)
    image = tf.pad(image, paddings, mode="CONSTANT", constant_values=pad_value)
    image.set_shape(image.shape[:-3] + tuple(size) + (num_channels,))
    return tf.cast(image, dtype)

  def apply(self, image: tf.Tensor) -> tf.Tensor:
    return self._resize_image(image, pad_value=self.pad_value)

  def apply_boxes(self, boxes: tf.Tensor,
                  image_size: transforms.SizeTuple) -> tf.Tensor:
    transforms.assert_boxes_are_relative(boxes)

    # Get relative image size before padding (w.r.t. padded output image):
    h_orig, w_orig = image_size
    long_edge = tf.maximum(h_orig, w_orig)
    h_rel = h_orig / long_edge
    w_rel = w_orig / long_edge

    # Pad the boxes to the output size:
    padded_size = self.size
    if isinstance(padded_size, int):
      padded_size = (padded_size, padded_size)
    return transforms.crop_or_pad_boxes(
        boxes=boxes,
        top=0,
        left=0,
        height=padded_size[0],
        width=padded_size[1],
        h_orig=tf.cast(h_rel * padded_size[0], h_orig.dtype),
        w_orig=tf.cast(w_rel * padded_size[1], w_orig.dtype))


@dataclasses.dataclass(frozen=True)
class CropOrPad(ImagePreprocessOp):
  """Crops or pads the features to have uniform shapes."""

  # Height and width. Only does spatial cropping or padding if set.
  size: Optional[int]
  num_instances: int
  allow_crop: bool = True

  def apply(self, image: tf.Tensor) -> tf.Tensor:
    if self.size is None:
      return image

    c = image.shape[-1]
    paddings = transforms.get_paddings(
        tf.shape(image), self.size, allow_crop=self.allow_crop)
    image = tf.pad(image, paddings, constant_values=0)
    if self.allow_crop:
      image = image[:self.size, :self.size, :]
      image.set_shape((self.size, self.size, c))
      return image
    return image

  def apply_boxes(self, boxes: tf.Tensor,
                  image_size: transforms.SizeTuple) -> tf.Tensor:
    if self.size is not None:
      # Pad and crop the spatial dimensions:
      if self.allow_crop:
        # After cropping, the image shape is always [self.size, self.size]:
        processed_image_size = [self.size, self.size]
      else:
        # If only padding is performed, the image size is at least self.size:
        processed_image_size = tf.maximum(image_size, self.size)
      boxes = transforms.crop_or_pad_boxes(
          boxes,
          top=0,
          left=0,
          height=processed_image_size[0],
          width=processed_image_size[1],
          h_orig=image_size[0],
          w_orig=image_size[1])

    # Pad or crop the number of instances:
    paddings = [[0, self.num_instances - tf.shape(boxes)[0]], [0, 0]]
    if self.allow_crop:
      paddings = tf.maximum(paddings, 0)
    boxes = tf.pad(boxes, tf.stack(paddings), constant_values=-1.0)
    if self.allow_crop:
      boxes = boxes[:self.num_instances]
      boxes.set_shape((self.num_instances, 4))
    return boxes


@dataclasses.dataclass(frozen=True)
class CropOrPadMetaData():
  """Crops or pads all label and meta-data features to have uniform shapes."""

  num_instances: int
  image_multilabels: int
  allow_crop: bool = True

  negative_labels_key: str = modalities.NEGATIVE_LABELS
  negative_text_labels_key: str = modalities.NEGATIVE_TEXT_LABELS
  not_exhaustive_labels_key: str = modalities.NOT_EXHAUSTIVE_LABELS
  instance_labels_key: str = modalities.INSTANCE_LABELS
  instance_multi_labels_key: str = modalities.INSTANCE_MULTI_LABELS
  instance_text_labels_key: str = modalities.INSTANCE_TEXT_LABELS
  crowds_key: str = modalities.CROWD
  annotation_id_key: str = modalities.ANNOTATION_ID
  area_key: str = modalities.AREA

  def __call__(self, features: Features) -> Features:
    image_scalar_sequences = [
        self.negative_labels_key, self.negative_text_labels_key,
        self.not_exhaustive_labels_key
    ]
    instance_scalar_sequences = [
        self.instance_text_labels_key, self.area_key, self.crowds_key,
        self.annotation_id_key, self.instance_labels_key,
        self.instance_multi_labels_key
    ]
    for key in image_scalar_sequences:
      if key in features:
        features[key] = transforms.crop_or_pad_sequence(features[key],
                                                        self.image_multilabels,
                                                        self.allow_crop)
    for key in instance_scalar_sequences:
      if key in features:
        features[key] = transforms.crop_or_pad_sequence(features[key],
                                                        self.num_instances,
                                                        self.allow_crop)
    return features


@dataclasses.dataclass(frozen=True)
class MergeOverlappingInstances:
  """Merge labels of instances with similar bounding boxes.

  This is useful when data contains multiple non-disjoint labels per instance
  (e.g. in federated datasets, when bounding boxes for specified labels are
  annotated independently of each other).

  Box similarity is assessed using the IoU (Intersetion over Union) metric. See
  `transforms.box_iou` for details.

  Attributes:
    iou_threshold: Box IoU threshold used to determine whether two instances
      should be merged.
    eps: Small float number used for numerical stability when computing IoU.
    label_feature_keys: Sequence of instance label modalities that should be
      merged. Note that these should be multi-label modalities.
  """
  iou_threshold: float = 0.95
  eps: float = 1e-6

  label_feature_keys: Sequence[str] = (
      modalities.INSTANCE_TEXT_MULTI_LABELS, modalities.INSTANCE_MULTI_LABELS)

  def __call__(self, features: Features) -> Features:
    """Iteratively merge instances with bbox IoU above a preset threshold."""

    def _compute_masked_iou(boxes):
      iou, _ = transforms.box_iou(boxes, boxes, eps=self.eps)
      # Mask comparison to self.
      iou = tf.where(tf.eye(tf.shape(iou)[0], dtype=tf.bool), 0., iou)
      return iou

    def _has_boxes_to_merge(boxes, labels, iou, rng):
      del boxes, labels, rng
      max_iou = tf.reduce_max(iou)
      return tf.greater_equal(max_iou, self.iou_threshold)

    def _merge_boxes(boxes, labels, iou, rng):
      ind = tf.unravel_index(  # Indices [i, j] of the two most similar boxes.
          tf.cast(tf.argmax(tf.reshape(iou, (-1,))), tf.int32),
          tf.shape(iou))

      # Merge labels of the two boxes; do not pay attention to possible
      # label collisions.
      labels_merged = tf.nest.map_structure(
          lambda lab: tf.gather(lab, ind).flat_values, labels)

      # Pick one box randomly.
      rngs = tf.random.experimental.stateless_split(rng)
      rng, new_rng = rngs[0], rngs[1]
      boxes_merged = tf.gather(boxes, ind)
      choice = tf.greater(tf.random.stateless_uniform([], rng), .5)
      boxes_merged = tf.where(choice, boxes_merged[0], boxes_merged[1])

      # Updated labels and boxes based on the merge. Tensors will be merged
      # using a gather. Prepare gather indices below.
      num_instances = tf.shape(boxes)[0]
      i, j = tf.reduce_min(ind), tf.reduce_max(ind)
      ind_all = tf.range(num_instances)
      ind_no_j = ind_all[ind_all != j]
      ind = tf.where(ind_no_j == i, num_instances, ind_no_j)
      def _merge_tensors(tensor, merged_value):
        tensor = tf.concat([tensor, tf.expand_dims(merged_value, 0)], axis=0)
        return tf.gather(tensor, ind)

      boxes = _merge_tensors(boxes, boxes_merged)
      labels = tf.nest.map_structure(_merge_tensors, labels, labels_merged)

      # Finally, update iou for the next iteration.
      iou = _compute_masked_iou(boxes)

      return (boxes, labels, iou, new_rng)

    if SEED_KEY not in features:
      raise ValueError("Merged box choice requires a random seed.")

    rng = features[preprocess_spec.SEED_KEY]

    boxes = features[modalities.BOXES]
    max_instances = tf.shape(boxes)[0]
    not_padding = tf.reduce_any(boxes != -1, axis=-1)
    boxes = boxes[not_padding]

    # Prepare (ragged) label tensors. This simplifies label merging logic.
    def to_ragged(x: tf.Tensor) -> tf.RaggedTensor:
      x = x[not_padding]
      lengths = tf.reduce_sum(
          tf.cast(tf.not_equal(x, transforms.get_padding_value(x.dtype)),
                  tf.int32), axis=-1)
      return tf.RaggedTensor.from_tensor(x, lengths)

    orig_labels = {}
    for name in self.label_feature_keys:
      if name in features:
        orig_labels[name] = features[name]
    labels = tf.nest.map_structure(to_ragged, orig_labels)

    state = (boxes, labels, _compute_masked_iou(boxes), rng)
    boxes, labels, _, rng = tf.while_loop(
        _has_boxes_to_merge,
        _merge_boxes,
        state,
        parallel_iterations=1,
        back_prop=False)

    features_new = dict(features)

    features_new[preprocess_spec.SEED_KEY] = rng

    # Pad boxes to original shape.
    features_new[modalities.BOXES] = tf.pad(
        boxes,
        [(0, max_instances - tf.shape(boxes)[0]), (0, 0)],
        constant_values=-1)

    # Convert ragged to normal tensor.
    def to_tensor(labels, orig_labels):
      return labels.to_tensor(
          default_value=transforms.get_padding_value(orig_labels.dtype),
          shape=tf.shape(orig_labels))
    labels = tf.nest.map_structure(to_tensor, labels, orig_labels)
    features_new.update(labels)
    return features_new


@dataclasses.dataclass(frozen=True)
class DecodeImage:
  """Decodes image feature and scales to [0, 1] range."""

  image_key: str = modalities.IMAGE
  input_image_key: str = "image"
  channels: Optional[int] = 3  # Set to 0 or None for adaptive.

  def __call__(self, features: Features) -> Features:
    image = features[self.input_image_key]

    # Some TFDS input pipeline configurations don't decode images by default.
    # pytype: disable=attribute-error  # allow-recursive-types
    if image.dtype == tf.string:
      # Decodes common image formats into uint8.
      image = tf.image.decode_image(image, channels=self.channels)

    if image.dtype == tf.uint8:
      image = tf.cast(image, tf.float32) / 255.0
    elif image.dtype == tf.float32:
      tf.debugging.assert_greater_equal(image, 0.)
      tf.debugging.assert_less_equal(image, 1.)
    else:
      raise ValueError(f"Unsupported dtype for image feature: {image.dtype}")
    # pytype: enable=attribute-error  # allow-recursive-types

    features[self.image_key] = image
    return features


@dataclasses.dataclass(frozen=True)
class DecodeCocoExample(DecodeImage):
  """Given a COCO TFDS example, creates features with boxes.

  The processing in this class includes:
  1. Converting images from uint8 to float32 with range [0, 1.]. Note that TFDS
     already parses the serialized protos and decodes jpeg images into uint8.
  2. Renaming keys to modality names.
  """

  boxes_key: str = modalities.BOXES
  instance_labels_key: str = modalities.INSTANCE_LABELS
  area_key: str = modalities.AREA
  orig_size_key: str = modalities.ORIGINAL_SIZE
  image_id_key: str = modalities.IMAGE_ID
  annotation_id_key: str = modalities.ANNOTATION_ID
  crowd_key: str = modalities.CROWD
  instance_text_labels_key: str = modalities.INSTANCE_TEXT_LABELS
  remove_crowd_annotations: bool = False

  def __call__(self, features: Features) -> Features:
    features = super().__call__(features)
    image_size = transforms.get_dynamic_size(features[self.image_key])
    boxes = features["objects"]["bbox"]  # float32, in range [0, 1].
    instance_labels = tf.cast(features["objects"]["label"], tf.int32)

    features_new = {
        self.image_key: features[self.image_key],
        self.boxes_key: boxes,
        self.instance_labels_key: instance_labels,
        self.area_key: features["objects"]["area"],
        self.orig_size_key: tf.cast(image_size, tf.int32),
        self.image_id_key: features["image/id"],
        self.annotation_id_key: features["objects"]["id"]
    }

    # We optionally fetch `is_crowd` as it is only present in Coco, and this
    # prevents breakage of inheritors e.g. LVIS, OpenImagesV5 and Coco Panoptic.
    if "is_crowd" in features["objects"]:
      features_new[self.crowd_key] = tf.cast(features["objects"]["is_crowd"],
                                             tf.int32)

      if self.remove_crowd_annotations:
        # Remove categories for which any instance is labeled as a crowd. These
        # categories violate the assumption that every instance of the category
        # is exhaustively annotated, which is made in some losses.
        crowd_labels = tf.boolean_mask(instance_labels,
                                       features_new[self.crowd_key] == 1)
        not_exhaustively_annotated = tf.reduce_any(
            instance_labels[..., None] == crowd_labels[..., None, :], axis=-1)
        mask = tf.logical_not(not_exhaustively_annotated)

        instance_keys = [
            self.boxes_key, self.instance_labels_key, self.area_key,
            self.annotation_id_key, self.crowd_key
        ]
        for k in instance_keys:
          features_new[k] = tf.boolean_mask(features_new[k], mask)

    if "rng" in features:
      features_new[SEED_KEY] = features["rng"]

    return features_new


@dataclasses.dataclass(frozen=True)
class DecodeLvisExample(DecodeCocoExample):
  """Given an LVIS TFDS example, creates features with boxes.

  The processing in this class includes:
  1. Converting images from uint8 to float32 with range [0, 1]. Note that TFDS
     already parses the serialized protos and decodes jpeg images into uint8.
  2. Renaming keys to modality names.
  """

  negative_labels_key: str = modalities.NEGATIVE_LABELS
  not_exhaustive_labels_key: str = modalities.NOT_EXHAUSTIVE_LABELS
  instance_text_labels_key: str = modalities.INSTANCE_TEXT_LABELS
  negative_text_labels_key: str = modalities.NEGATIVE_TEXT_LABELS

  def __call__(self, features: Features) -> Features:
    new_features = super().__call__(features)
    new_features[self.negative_labels_key] = tf.cast(
        features["neg_category_ids"], tf.int32)
    # A non-standard feature representing category ids not fully covered
    # by instance labels.
    new_features[self.not_exhaustive_labels_key] = tf.cast(
        features["not_exhaustive_category_ids"], tf.int32)
    return new_features
