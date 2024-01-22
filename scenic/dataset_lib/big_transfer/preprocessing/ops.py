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

"""Implementation of data preprocessing ops.

All preprocessing ops should return a data processing functors. A data
is represented as a dictionary of tensors, where field "image" is reserved
for 3D images (height x width x channels). The functors output dictionary with
field "image" being modified. Potentially, other fields can also be modified
or added.
"""
from typing import Optional, Tuple
import numpy as np

from scenic.dataset_lib.big_transfer.preprocessing import autoaugment
from scenic.dataset_lib.big_transfer.preprocessing import utils
from scenic.dataset_lib.big_transfer.registry import Registry
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

from tensorflow_addons import image as image_utils


@Registry.register("preprocess_ops.color_distort", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_color_distortion():
  """Applies random brigthness/saturation/hue/contrast transformations."""

  def _color_distortion(image):
    image = tf.image.random_brightness(image, max_delta=128. / 255.)
    image = tf.image.random_saturation(image, lower=0.1, upper=2.0)
    image = tf.image.random_hue(image, max_delta=0.5)
    image = tf.image.random_contrast(image, lower=0.1, upper=2.0)
    return image

  return _color_distortion


@Registry.register("preprocess_ops.random_brightness", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_random_brightness(max_delta=0.1):
  """Applies random brigthness transformations."""

  # A random value in [-max_delta, +max_delta] is added to the image values.
  # Small max_delta <1.0 assumes that the image values are within [0, 1].
  def _random_brightness(image):
    return tf.image.random_brightness(image, max_delta)

  return _random_brightness


@Registry.register("preprocess_ops.random_saturation", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_random_saturation(lower=0.5, upper=2.0):
  """Applies random saturation transformations."""

  # Multiplies saturation channel in HSV (with converting from/to RGB) with a
  # random float value in [lower, upper].
  def _random_saturation(image):
    return tf.image.random_saturation(image, lower=lower, upper=upper)

  return _random_saturation


@Registry.register("preprocess_ops.random_hue", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_random_hue(max_delta=0.1):
  """Applies random hue transformations."""

  # Adds to hue channel in HSV (with converting from/to RGB) a random offset
  # in [-max_delta, +max_delta].
  def _random_hue(image):
    return tf.image.random_hue(image, max_delta=max_delta)

  return _random_hue


@Registry.register("preprocess_ops.random_contrast", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_random_contrast(lower=0.5, upper=2.0):
  """Applies random contrast transformations."""

  # Stretches/shrinks value stddev (per channel) by multiplying with a random
  # value in [lower, upper].
  def _random_contrast(image):
    return tf.image.random_contrast(image, lower=lower, upper=upper)

  return _random_contrast


@Registry.register("preprocess_ops.decode", "function")
@utils.InKeyOutKey()
def get_decode(channels=3):
  """Decode an encoded image string, see tf.io.decode_image."""

  def _decode(image):  # pylint: disable=missing-docstring
    # tf.io.decode_image does not set the shape correctly, so we use
    # tf.io.deocde_jpeg, which also works for png, see
    # https://github.com/tensorflow/tensorflow/issues/8551
    return tf.io.decode_jpeg(image, channels=channels)

  return _decode


@Registry.register("preprocess_ops.decode_grayscale", "function")
@utils.InKeyOutKey()
def get_decode_grayscale(channels=1):
  """Decode an encoded image string, see tf.io.decode_image."""

  def _decode_gray(image):  # pylint: disable=missing-docstring
    # tf.io.decode_image does not set the shape correctly, so we use
    # tf.io.deocde_jpeg, which also works for png, see
    # https://github.com/tensorflow/tensorflow/issues/8551
    return tf.io.decode_jpeg(image, channels=channels)

  return _decode_gray


@Registry.register("preprocess_ops.pad", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_pad(pad_size):
  """Pads an image.

  Args:
    pad_size: either an integer u giving verticle and horizontal pad sizes u, or
      a list or tuple [u, v] of integers where u and v are vertical and
      horizontal pad sizes.

  Returns:
    A function for padding an image.

  """
  pad_size = utils.maybe_repeat(pad_size, 2)

  def _pad(image):
    return tf.pad(
        image, [[pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [0, 0]])

  return _pad


@Registry.register("preprocess_ops.resize", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_resize(resize_size, method=tf2.image.ResizeMethod.BILINEAR,
               antialias=False):
  """Resizes image to a given size.

  Args:
    resize_size: either an integer H, where H is both the new height and width
      of the resized image, or a list or tuple [H, W] of integers, where H and W
      are new image"s height and width respectively.
    method: The type of interpolation to apply when resizing.
    antialias: Whether to use an anti-aliasing filter when downsampling an
      image.

  Returns:
    A function for resizing an image.

  """
  resize_size = utils.maybe_repeat(resize_size, 2)

  def _resize(image):
    """Resizes image to a given size."""
    # Note: use TF-2 version of tf.image.resize as the version in TF-1 is
    # buggy: https://github.com/tensorflow/tensorflow/issues/6720.
    # In particular it was not equivariant with rotation and lead to the network
    # to learn a shortcut in self-supervised rotation task, if rotation was
    # applied after resize.
    dtype = image.dtype
    image = tf2.image.resize(
        images=image, size=resize_size, method=method, antialias=antialias)
    return tf.cast(image, dtype)

  return _resize


@Registry.register("preprocess_ops.resize_small", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_resize_small(smaller_size, method="area", antialias=True):
  """Resizes the smaller side to `smaller_size` keeping aspect ratio.

  Args:
    smaller_size: an integer, that represents a new size of the smaller side of
      an input image.
    method: the resize method. `area` is a meaningful, bwd-compat default.
    antialias: See TF's image.resize method.

  Returns:
    A function, that resizes an image and preserves its aspect ratio.

  """

  def _resize_small(image):  # pylint: disable=missing-docstring
    h, w = tf.shape(image)[0], tf.shape(image)[1]

    # Figure out the necessary h/w.
    ratio = (
        tf.cast(smaller_size, tf.float32) /
        tf.cast(tf.minimum(h, w), tf.float32))
    h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
    w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)

    dtype = image.dtype
    image = tf2.image.resize(image, (h, w), method, antialias)
    return tf.cast(image, dtype)

  return _resize_small


@Registry.register("preprocess_ops.inception_crop", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_inception_crop(resize_size=None, area_min=5, area_max=100,
                       resize_method=tf2.image.ResizeMethod.BILINEAR):
  """Makes inception-style image crop.

  Inception-style crop is a random image crop (its size and aspect ratio are
  random) that was used for training Inception models, see
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

  Args:
    resize_size: Resize image to [resize_size, resize_size] after crop.
    area_min: minimal crop area.
    area_max: maximal crop area.
    resize_method: The type of interpolation to apply when resizing. Valid
      values those accepted by tf.image.resize.

  Returns:
    A function, that applies inception crop.
  """

  def _inception_crop(image):  # pylint: disable=missing-docstring
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(area_min / 100, area_max / 100),
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)
    crop = tf.slice(image, begin, size)
    # Unfortunately, the above operation loses the depth-dimension. So we need
    # to restore it the manual way.
    crop.set_shape([None, None, image.shape[-1]])
    if resize_size:
      crop = get_resize([resize_size, resize_size], resize_method)(
          {"image": crop})["image"]
    return crop

  return _inception_crop


@Registry.register("preprocess_ops.decode_jpeg_and_inception_crop", "function")
@utils.InKeyOutKey()
def get_decode_jpeg_and_inception_crop(
    resize_size=None,
    area_min=5,
    area_max=100,
    aspect_ratio_range=None,
    resize_method=tf2.image.ResizeMethod.BILINEAR):
  """Decode jpeg string and make inception-style image crop.

  Inception-style crop is a random image crop (its size and aspect ratio are
  random) that was used for training Inception models, see
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

  Args:
    resize_size: Resize image to [resize_size, resize_size] after crop.
    area_min: minimal crop area.
    area_max: maximal crop area.
    aspect_ratio_range: An optional list of floats. Defaults to [0.75, 1.33].
      The cropped area of the image must have an aspect ratio = width / height
      within this range.
    resize_method: The type of interpolation to apply when resizing. Valid
      values those accepted by tf.image.resize.

  Returns:
    A function, that applies inception crop.
  """

  def _inception_crop(image_data):  # pylint: disable=missing-docstring
    shape = tf.image.extract_jpeg_shape(image_data)
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        shape,
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(area_min / 100, area_max / 100),
        min_object_covered=0,  # Don't enforce a minimum area.
        aspect_ratio_range=aspect_ratio_range,
        use_image_if_no_bounding_boxes=True)

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(begin)
    target_height, target_width, _ = tf.unstack(size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_data, crop_window, channels=3)

    if resize_size:
      image = get_resize([resize_size, resize_size], resize_method)(
          {"image": image})["image"]

    return image

  return _inception_crop


@Registry.register("preprocess_ops.decode_jpeg_and_center_crop", "function")
@utils.InKeyOutKey()
def get_decode_jpeg_and_center_crop(crop_size=None):
  """Decode jpeg string and make a center image crop.

  Args:
    crop_size: Crop image to [crop_size, crop_size].

  Returns:
    A function that applies center crop.
  """

  crop_size = utils.maybe_repeat(crop_size, 2)

  def _decode_and_center_crop(image_data):  # pylint: disable=missing-docstring
    shape = tf.image.extract_jpeg_shape(image_data)
    target_height, target_width = crop_size

    offset_y = (shape[0] - target_height) // 2
    offset_x = (shape[1] - target_width) // 2

    # Crop the image to the specified bounding box.
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_data, crop_window, channels=3)
    image.set_shape([target_height, target_width, 3])
    return image

  return _decode_and_center_crop


@Registry.register("preprocess_ops.decode_jpeg_and_random_crop", "function")
@utils.InKeyOutKey()
def get_decode_jpeg_and_random_crop(crop_size=None):
  """Decode jpeg string and make a center image crop.

  Args:
    crop_size: Crop image to [crop_size, crop_size].

  Returns:
    A function that applies center crop.
  """

  crop_size = utils.maybe_repeat(crop_size, 2)

  def _decode_and_random_crop(image_data):  # pylint: disable=missing-docstring
    shape = tf.image.extract_jpeg_shape(image_data)[:2]
    target_height, target_width = crop_size
    limit = shape - crop_size + 1
    offset = tf.random.uniform([2], 0, tf.int32.max, dtype=tf.int32) % limit

    # Crop the image to the specified bounding box.
    crop_window = tf.stack([offset[0], offset[1], target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_data, crop_window, channels=3)
    image.set_shape([target_height, target_width, 3])
    return image

  return _decode_and_random_crop


@Registry.register("preprocess_ops.random_crop", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_random_crop(crop_size):
  """Makes a random crop of a given size.

  Args:
    crop_size: either an integer H, where H is both the height and width of the
      random crop, or a list or tuple [H, W] of integers, where H and W are
      height and width of the random crop respectively.

  Returns:
    A function, that applies random crop.
  """
  crop_size = utils.maybe_repeat(crop_size, 2)

  def _crop(image):
    return tf.random_crop(image, [crop_size[0], crop_size[1], image.shape[-1]])

  return _crop


@Registry.register("preprocess_ops.central_crop", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_central_crop(crop_size):
  """Makes central crop of a given size.

  Args:
    crop_size: either an integer H, where H is both the height and width of the
      central crop, or a list or tuple [H, W] of integers, where H and W are
      height and width of the central crop respectively.

  Returns:
    A function, that applies central crop.
  """
  crop_size = utils.maybe_repeat(crop_size, 2)

  def _crop(image):
    h, w = crop_size[0], crop_size[1]
    dy = (tf.shape(image)[0] - h) // 2
    dx = (tf.shape(image)[1] - w) // 2
    return tf.image.crop_to_bounding_box(image, dy, dx, h, w)

  return _crop


@Registry.register("preprocess_ops.central_crop_longer", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_central_crop_longer():
  """Center crop the longer side so that the image becomes a square.

  Args:

  Returns:
    A function, that applies central crop.
  """

  def _crop(image):
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    crop_fn = tf.image.crop_to_bounding_box
    return tf.cond(
        h > w,
        lambda: crop_fn(image, h // 2 - w // 2, 0, w, w),
        lambda: crop_fn(image, 0, w // 2 - h // 2, h, h))

  return _crop


@Registry.register("preprocess_ops.flip_lr", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_random_flip_lr():
  """Flips an image horizontally with probability 50%."""

  def _random_flip_lr_pp(image):
    return tf.image.random_flip_left_right(image)

  return _random_flip_lr_pp


@Registry.register("preprocess_ops.flip_ud", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_random_flip_ud():
  """Flips an image vertically with probability 50%."""

  def _random_flip_ud_pp(image):
    return tf.image.random_flip_up_down(image)

  return _random_flip_ud_pp


@Registry.register("preprocess_ops.random_rotate", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_random_rotation(min_angle=0, max_angle=360):
  """Randomly rotate an image."""
  if min_angle > max_angle:
    raise ValueError("min_angle (%f) must be lower than max_angle (%f)" %
                     (min_angle, max_angle))
  # Convert to radians.
  min_angle = np.radians(min_angle)
  max_angle = np.radians(max_angle)

  def _random_rotation(image):
    """Rotation function."""
    num_dims = len(image.shape)
    if num_dims in [3, 4]:
      batch_size = tf.shape(image)[0] if num_dims == 4 else 1
    else:
      raise ValueError("Tensor \"image\" should have 3 or 4 dimensions.")
    random_angles = tf.random.uniform(
        shape=(batch_size,), minval=min_angle, maxval=max_angle)
    return image_utils.rotate(images=image, angles=random_angles)

  return _random_rotation


@Registry.register("preprocess_ops.random_rotate90", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_random_rotation90():
  """Randomly rotate an image by multiples of 90 degrees."""

  def _random_rotation90(image):
    """Rotation function."""
    num_rotations = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(image, k=num_rotations)

  return _random_rotation90


@Registry.register("preprocess_ops.rotate", "function")
def get_rotate(create_labels=None):
  """Returns a function that does 90deg rotations and sets according labels.

  Args:
    create_labels: create new labels to the default label field in the input
      dictionary. It should be set to one of ['rotation', 'supervised', None].

  Returns:
    A function, that applies rotation preprocess.
  """

  def _four_rots(img):
    """Rotates an image four times, with 90 degrees between each rotation."""
    return tf.stack([
        img,
        tf.transpose(tf.reverse_v2(img, [1]), [1, 0, 2]),
        tf.reverse_v2(img, [0, 1]),
        tf.reverse_v2(tf.transpose(img, [1, 0, 2]), [1]),
    ])

  def _rotate_pp(data):
    """Rotate preprocessing function applied on data dictionary input."""
    assert create_labels in [
        "rotation", "supervised", None
    ], ("create_labels:{} must be one of ['rotation', 'supervised', None]."
        .format(create_labels))

    # Creates labels in the same structure as images.
    if create_labels == "rotation":
      data["label"] = tf.constant([0, 1, 2, 3])
    # Duplicates the original supervised label four times.
    elif create_labels == "supervised":
      if "label" in data:
        data["label"] = tf.stack(tf.tile([data["label"]], [4]))
    # Creates rotated images and rot labels.
    data["image"] = _four_rots(data["image"])
    data["rot_label"] = tf.constant([0, 1, 2, 3])

    return data

  return _rotate_pp


@Registry.register("preprocess_ops.value_range", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing(output_dtype=tf.float32)
def get_value_range(vmin=-1, vmax=1, in_min=0, in_max=255.0, clip_values=False):
  """Transforms a [in_min,in_max] image to [vmin,vmax] range.

  Input ranges in_min/in_max can be equal-size lists to rescale the invidudal
  channels independently.

  Args:
    vmin: A scalar. Output max value.
    vmax: A scalar. Output min value.
    in_min: A scalar or a list of input min values to scale. If a list, the
      length should match to the number of channels in the image.
    in_max: A scalar or a list of input max values to scale. If a list, the
      length should match to the number of channels in the image.
    clip_values: Whether to clip the output values to the provided ranges.

  Returns:
    A function to rescale the values.
  """

  def _value_range(image):
    """Scales values in given range."""
    in_min_t = tf.constant(in_min, tf.float32)
    in_max_t = tf.constant(in_max, tf.float32)
    image = tf.cast(image, tf.float32)
    image = (image - in_min_t) / (in_max_t - in_min_t)
    image = vmin + image * (vmax - vmin)
    if clip_values:
      image = tf.clip_by_value(image, vmin, vmax)
    return image

  return _value_range


@Registry.register("preprocess_ops.value_range_mc", "function")
def get_value_range_mc(vmin, vmax, *args):
  """Independent multi-channel rescaling."""
  if len(args) % 2:
    raise ValueError("Additional args must be list of even length giving "
                     "`in_max` and `in_min` concatenated")
  num_channels = len(args) // 2
  in_min = args[:num_channels]
  in_max = args[-num_channels:]

  return get_value_range(vmin, vmax, in_min, in_max)


@Registry.register("preprocess_ops.delete_field", "function")
def get_delete_field(key):

  def _delete_field(datum):
    if key in datum:
      del datum[key]
    return datum

  return _delete_field


@Registry.register("preprocess_ops.replicate", "function")
@utils.InKeyOutKey()
def get_replicate(num_replicas=2):
  """Replicates an image `num_replicas` times along a new batch dimension."""

  def _replicate(image):
    tiles = [num_replicas] + [1] * len(image.shape)
    return tf.tile(image[None], tiles)

  return _replicate


@Registry.register("preprocess_ops.standardize", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing(output_dtype=tf.float32)
def get_standardize(mean, std):
  """Standardize an image with the given mean and standard deviation."""

  def _standardize(image):
    image = tf.cast(image, dtype=tf.float32)
    return (image - mean) / std

  return _standardize


@Registry.register("preprocess_ops.select_channels", "function")
@utils.InKeyOutKey()
@utils.BatchedImagePreprocessing()
def get_select_channels(channels):
  """Returns function to select specified channels."""

  def _select_channels(image):
    """Returns a subset of available channels."""
    return tf.gather(image, channels, axis=-1)

  return _select_channels


@Registry.register("preprocess_ops.extract_patches", "function")
@utils.InKeyOutKey()
def get_extract_patches(patch_size, stride):
  """Extracts image patches.

  Args:
    patch_size: patch size.
    stride: patches stride.

  Returns:
     A function for extracting patches.
  """

  def _extract_patches(image):
    """Extracts image patches."""
    h, w, c = image.get_shape().as_list()

    count_h = h // stride
    count_w = w // stride

    # pyformat: disable
    image = tf.extract_image_patches(image[None],
                                     [1, patch_size, patch_size, 1],
                                     [1, stride, stride, 1],
                                     [1, 1, 1, 1],
                                     padding="VALID")
    # pyformat: enable

    return tf.reshape(image, [count_h * count_w, patch_size, patch_size, c])

  return _extract_patches


@Registry.register("preprocess_ops.onehot", "function")
def get_onehot(depth,
               key="labels",
               key_result=None,
               multi=True,
               on=1.0,
               off=0.0):
  """One-hot encodes the input.

  Args:
    depth: Length of the one-hot vector (how many classes).
    key: Key of the data to be one-hot encoded.
    key_result: Key under which to store the result (same as `key` if None).
    multi: If there are multiple labels, whether to merge them into the same
      "multi-hot" vector (True) or keep them as an extra dimension (False).
    on: Value to fill in for the positive label (default: 1).
    off: Value to fill in for negative labels (default: 0).

  Returns:
    Data dictionary.
  """

  def _onehot(data):
    # When there's more than one label, this is significantly more efficient
    # than using tf.one_hot followed by tf.reduce_max; we tested.
    labels = data[key]
    if labels.shape.rank > 0 and multi:
      # Currently, the assertion below is only used for datasets with single
      # labels. In a multi-label dataset either `on` or `off` should be computed
      # dynamically to yield the correct sum, when the number of labels varies.
      x = tf.scatter_nd(labels[:, None], tf.ones(tf.shape(labels)[0]), (depth,))
      x = tf.clip_by_value(x, 0, 1) * (on - off) + off
    else:
      assert np.isclose(on + off * (depth - 1), 1), (
          "All on and off values must sum to 1")
      x = tf.one_hot(labels, depth, on_value=on, off_value=off)
    data[key_result or key] = x
    return data

  return _onehot


@Registry.register("preprocess_ops.keep", "function")
def get_keep(*keys):
  """Keeps only the given keys."""

  def _keep(data):
    return {k: v for k, v in data.items() if k in keys}

  return _keep


@Registry.register("preprocess_ops.drop", "function")
def get_drop(*keys):
  """Drops the given keys."""

  def _drop(data):
    return {k: v for k, v in data.items() if k not in keys}

  return _drop


@Registry.register("preprocess_ops.copy", "function")
def get_copy(inkey, outkey):
  """Copies value of `inkey` into `outkey`."""

  def _copy(data):
    data[outkey] = data[inkey]
    return data

  return _copy


@Registry.register("preprocess_ops.randaug", "function")
@utils.InKeyOutKey()
def get_randaug(num_layers: int = 2, magnitude: int = 10):
  """Creates a function that applies RandAugment.

  RandAugment is from the paper https://arxiv.org/abs/1909.13719,

  Args:
    num_layers: Integer, the number of augmentation transformations to apply
      sequentially to an image. Represented as (N) in the paper. Usually best
      values will be in the range [1, 3].
    magnitude: Integer, shared magnitude across all augmentation operations.
      Represented as (M) in the paper. Usually best values are in the range [5,
      30].

  Returns:
    A function that applies RandAugment.
  """

  def _randaug(image):
    return autoaugment.distort_image_with_randaugment(
        image=image,
        num_layers=num_layers,
        magnitude=magnitude,
    )

  return _randaug


@Registry.register("preprocess_ops.patchify", "function")
@utils.InKeyOutKey()
def patchify(patch_size: Tuple[int, int], stride: Tuple[int, int]):
  """Patchifies image.

  If image is of size (h, w, c), patchify it into (h//p*w//p, p*p*c)

  Args:
    patch_size: Integer.
    stride: Integer.

  Returns:
    A function that applies RandAugment.
  """

  def _extract_patches(image):
    """Extracts image patches."""
    h, w, _ = image.get_shape().as_list()

    count_h = h // stride[0]
    count_w = w // stride[1]

    # pyformat: disable
    image = tf.extract_image_patches(image[None],
                                     [1, patch_size[0], patch_size[1], 1],
                                     [1, stride[0], stride[1], 1],
                                     [1, 1, 1, 1],
                                     padding="VALID")
    # pyformat: enable
    return tf.reshape(image, [count_h * count_w, -1])

  return _extract_patches


