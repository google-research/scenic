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

"""COCO testing utilities."""
import numpy as np
import tensorflow as tf


# Width, height and ID are taken from real COCO annotations. This is required to
# make the COCO evaluator work (for the trainer test).
FAKE_W = [640, 352, 640, 640, 640, 640, 427, 375, 640, 351, 640, 640, 500, 375,
          640, 640, 640, 640, 640, 640, 640, 480, 640, 427, 427, 480, 461, 640,
          640, 634, 640, 640, 640, 640, 640, 480, 640, 640, 500, 640, 640, 640,
          500, 640, 425, 581, 640, 427, 640, 640]

FAKE_H = [427, 230, 428, 480, 388, 511, 640, 500, 426, 500, 427, 480, 423, 500,
          500, 462, 480, 480, 425, 480, 480, 640, 427, 640, 640, 640, 500, 428,
          426, 640, 480, 480, 400, 432, 480, 640, 427, 371, 375, 480, 427, 480,
          375, 449, 640, 640, 480, 640, 360, 640]

FAKE_ID = [397133, 37777, 252219, 87038, 174482, 403385, 6818, 480985, 458054,
           331352, 296649, 386912, 502136, 491497, 184791, 348881, 289393,
           522713, 181666, 17627, 143931, 303818, 463730, 460347, 322864,
           226111, 153299, 308394, 456496, 58636, 41888, 184321, 565778, 297343,
           336587, 122745, 219578, 555705, 443303, 500663, 418281, 25560,
           403817, 85329, 329323, 239274, 286994, 511321, 314294, 233771]


def generate_fake_example(w: int, h: int, identifier: int):
  """Generate a random COCO example."""
  num_objects = 8
  return {
      'image': np.random.randint(0, 256, size=(w, h, 3), dtype=np.uint8),
      'image/filename': f'{identifier:012}.jpg',
      'image/id': identifier,
      'objects': {
          'area': np.arange(num_objects, dtype=np.int64) * 50,
          'bbox': np.stack([np.array([0., 0., 1., 1.])] * num_objects),
          'id': np.arange(num_objects),
          'is_crowd': np.full((num_objects,), False, dtype=bool),
          'label': np.random.randint(
              0, 81, size=(num_objects,), dtype=np.int32),
      }
  }


def generate_fake_dataset(num_examples: int):
  """Constructs a dataset generator object."""

  def _generator(self, *args, **kwargs):
    """Generate a fake dataset for testing."""
    del args
    del kwargs

    def gen():
      n_annotations = len(FAKE_ID)
      for i in range(num_examples):
        yield generate_fake_example(
            w=FAKE_W[i % n_annotations],
            h=FAKE_H[i % n_annotations],
            identifier=FAKE_ID[i % n_annotations])

    return tf.data.Dataset.from_generator(
        gen,
        output_types=self.info.features.dtype,
        output_shapes=self.info.features.shape,
    )
  return _generator
