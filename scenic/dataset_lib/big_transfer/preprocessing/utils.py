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

"""Preprocessing utils.

"""

from collections import abc
import functools
import tensorflow.compat.v1 as tf


def maybe_repeat(arg, n_reps):
  if not isinstance(arg, abc.Sequence):
    arg = (arg,) * n_reps
  return arg


def tf_apply_to_image_or_images(fn, image_or_images, **map_kw):
  """Applies a function to a single image or each image in a batch of them.

  Args:
    fn: the function to apply, receives an image, returns an image.
    image_or_images: Either a single image, or a batch of images.
    **map_kw: Arguments passed through to tf.map_fn if called.

  Returns:
    The result of applying the function to the image or batch of images.

  Raises:
    ValueError: if the input is not of rank 3 or 4.
  """
  static_rank = len(image_or_images.get_shape().as_list())
  if static_rank == 3:  # A single image: HWC
    return fn(image_or_images)
  elif static_rank == 4:  # A batch of images: BHWC
    return tf.map_fn(fn, image_or_images, **map_kw)
  elif static_rank > 4:  # A batch of images: ...HWC
    input_shape = tf.shape(image_or_images)
    h, w, c = image_or_images.get_shape().as_list()[-3:]
    image_or_images = tf.reshape(image_or_images, [-1, h, w, c])
    image_or_images = tf.map_fn(fn, image_or_images, **map_kw)
    return tf.reshape(image_or_images, input_shape)
  else:
    raise ValueError("Unsupported image rank: %d" % static_rank)


class BatchedImagePreprocessing(object):
  """Decorator for preprocessing ops, which adds support for image batches.

  Note: Doesn't support decorating ops which add new fields in data.
  """

  def __init__(self, output_dtype=None):
    self.output_dtype = output_dtype

  def __call__(self, get_pp_fn):

    def get_batch_pp_fn(*args, **kwargs):
      """Preprocessing function that supports batched images."""

      def _batch_pp_fn(image, *a, **kw):
        orig_image_pp_fn = get_pp_fn(*args, **kwargs)
        orig_image_pp_fn = functools.partial(orig_image_pp_fn, *a, **kw)
        return tf_apply_to_image_or_images(
            orig_image_pp_fn, image, dtype=self.output_dtype)

      return _batch_pp_fn

    return get_batch_pp_fn


class InKeyOutKey(object):
  """Decorator for preprocessing ops, which adds `inkey` and `outkey` arguments.

  Note: Only supports single-input single-output ops.
  """

  def __init__(self, uses_rngkey=False, indefault="image", outdefault="image"):
    self.uses_rngkey = uses_rngkey
    self.indefault = indefault
    self.outdefault = outdefault

  def __call__(self, orig_get_pp_fn):

    def get_ikok_pp_fn(*args,
                       key=None,
                       inkey=self.indefault,
                       outkey=self.outdefault,
                       **kw):

      # Support legacy arg from BatchedPreprocessing
      key = kw.pop("data_key", key)

      orig_pp_fn = orig_get_pp_fn(*args, **kw)

      def _ikok_pp_fn(data):
        if not self.uses_rngkey:
          data[key or outkey] = orig_pp_fn(data[key or inkey])
        else:
          data[key or
               outkey], data["_rngkey"] = orig_pp_fn(data[key or inkey],
                                                     data["_rngkey"])
        return data

      return _ikok_pp_fn

    return get_ikok_pp_fn
