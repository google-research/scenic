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

"""Implementation of data preprocessing ops for VTAB.

All preprocessing ops should return a data processing functors. A data
is represented as a dictionary of tensors, where field "image" is reserved
for 3D images (height x width x channels). The functors output dictionary with
field "image" being modified. Potentially, other fields can also be modified
or added.
"""
import numpy as np
from scenic.dataset_lib.big_transfer.registry import Registry
import tensorflow.compat.v1 as tf


@Registry.register("preprocess_ops.dsprites_pp", "function")
def get_dsprites_pp(predicted_attribute, num_classes=None):
  """Data preprocess function for dsprites dataset."""

  attribute_to_classes = {
      "label_shape": 3,
      "label_scale": 6,
      "label_orientation": 40,
      "label_x_position": 32,
      "label_y_position": 32,
  }

  def _dsprites_pp(data):
    # For consistency with other datasets, image needs to have three channels
    # and be in [0, 255).   # pylint: disable=unused-argument
    # data["image"] = tf.tile(data["image"], [1, 1, 3]) * 255
    data["image"] = data["image"] * 255

    # If num_classes is set, we group together nearby integer values to arrive
    # at the desired number of classes. This is useful for example for grouping
    # together different spatial positions.
    num_original_classes = attribute_to_classes[predicted_attribute]
    n_cls = num_original_classes if num_classes is None else num_classes
    if not isinstance(n_cls, int) or n_cls <= 1 or n_cls > num_original_classes:
      raise ValueError(
          "The number of classes should be None or in [2, ..., num_classes].")
    class_division_factor = float(num_original_classes) / n_cls

    data["label"] = tf.cast(
        tf.math.floordiv(
            tf.cast(data[predicted_attribute], tf.float32),
            class_division_factor), data[predicted_attribute].dtype)
    return data

  return _dsprites_pp


@Registry.register("preprocess_ops.clevr_pp", "function")
def get_clevr_pp(task, outkey="label"):
  """Data preprocess function for clevr dataset."""

  def _count_preprocess_fn(data):
    data[outkey] = tf.size(data["objects"]["size"]) - 3
    return data

  def _closest_object_preprocess_fn(data):
    dist = tf.reduce_min(data["objects"]["pixel_coords"][:, 2])
    # These thresholds are uniformly spaced and result in more or less balanced
    # distribution of classes.
    thrs = np.array([0.0, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0])
    data[outkey] = tf.reduce_max(tf.where((thrs - dist) < 0))
    return data

  task_to_preprocess = {
      "count_all": _count_preprocess_fn,
      "closest_object_distance": _closest_object_preprocess_fn,
  }

  return task_to_preprocess[task]


@Registry.register("preprocess_ops.kitti_pp", "function")
def get_kitti_pp(task):
  """Data preprocess function for kitti dataset."""

  def _closest_vehicle_distance_pp(data):
    """Predict the distance to the closest vehicle."""
    # Location feature contains (x, y, z) in meters w.r.t. the camera.
    vehicles = tf.where(data["objects"]["type"] < 3)  # Car, Van, Truck.
    vehicle_z = tf.gather(
        params=data["objects"]["location"][:, 2], indices=vehicles)
    vehicle_z = tf.concat([vehicle_z, tf.constant([[1000.0]])], axis=0)
    dist = tf.reduce_min(vehicle_z)
    # Results in a uniform distribution over three distances, plus one class for
    # "no vehicle".
    thrs = np.array([-100.0, 8.0, 20.0, 999.0])
    label = tf.reduce_max(tf.where((thrs - dist) < 0))
    return {"image": data["image"], "label": label}

  task_to_preprocess = {
      "closest_vehicle_distance": _closest_vehicle_distance_pp,
  }

  return task_to_preprocess[task]
