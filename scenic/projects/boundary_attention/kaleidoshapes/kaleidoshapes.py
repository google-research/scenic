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

"""Dataset builder for Kaleidoshapes dataset."""

import functools

import numpy as np
from scenic.projects.boundary_attention.kaleidoshapes import make_kaleido_image
import tensorflow as tf
import tensorflow_datasets as tfds

_BASE_DIR = ''
_MIN_OBJECTS = 15
_MAX_OBJECTS = 20
_BOUNDARY_WIDTH = 0.001
_MIN_RADIUS = 0.0
_MAX_RADIUS = 0.2
_MIN_TRIANGLE_BASE = 0.02
_MAX_TRIANGLE_BASE = 0.5
_MIN_TRIANGLE_HEIGHT = 0.02
_MAX_TRIANGLE_HEIGHT = 0.3
_MIN_VISIBILITY = 0.005
_IMAGE_HEIGHT = 240
_IMAGE_WIDTH = 320
_PROB_CIRCLE = 0.4
_NUM_IMAGES = 100_000


class ShapesConfig:
  """Configuration for Kaleidoshapes dataset."""

  def __init__(self):

    # bounds on number of objects in an image
    self.min_objects = _MIN_OBJECTS
    self.max_objects = _MAX_OBJECTS

    # bounds on object sizes and locations, as percentage of maximum image
    # dimension
    # all objects must be at least this far from the image boundary
    self.boundary_width = _BOUNDARY_WIDTH

    # circle bounds
    self.min_radius = _MIN_RADIUS
    self.max_radius = _MAX_RADIUS

    # triangle bounds
    self.min_triangle_height = _MIN_TRIANGLE_HEIGHT
    self.max_triangle_height = _MAX_TRIANGLE_HEIGHT

    self.min_triangle_base = _MIN_TRIANGLE_BASE
    self.max_triangle_base = _MAX_TRIANGLE_BASE

    # threshold for a shape being "visible", as fraction of total (normalized)
    # image area
    self.min_visibility = _MIN_VISIBILITY

    self.image_height = _IMAGE_HEIGHT
    self.image_width = _IMAGE_WIDTH

    self.prob_circle = _PROB_CIRCLE

    self.num_images = _NUM_IMAGES


_DESCRIPTION = """
The KaleidoShapes dataset.
"""

_CITATION = """
@inproceedings{KaleidoShapes,
  author    = {Todd Zickler, Mia Polansky},
  title     = {KaleidoShapes: A repository of colorful multi-object images
               segmentation and boundary detection},
  year      = {2023}
}
"""


def flatten(nested_list):
  for item in nested_list:
    yield item


class Kaleidoshapes(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Kaleidoshapes dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    self.config = ShapesConfig()

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image_index': tf.int64,
            'image': tfds.features.Image(shape=(self.config.image_height,
                                                self.config.image_width, 3)),
            'boundaries': tfds.features.Image(shape=(self.config.image_height,
                                                     self.config.image_width,
                                                     1)),
            'segments': tfds.features.Image(shape=(self.config.image_height,
                                                   self.config.image_width,
                                                   1)),
            'distances': tfds.features.Tensor(shape=(self.config.image_height,
                                                     self.config.image_width),
                                              dtype=tf.float32),
            'num_shapes': tf.int64,
            'shapes': tfds.features.Sequence(feature={
                'type': tfds.features.Text(),
                'color': tfds.features.Tensor(shape=(3,), dtype=tf.uint8),
                'triangle_params': tfds.features.Tensor(shape=(3, 2),
                                                        dtype=tf.float32),
                'circle_params': tfds.features.Tensor(shape=(3,),
                                                      dtype=tf.float32),
            }),
            'basecolor': tfds.features.Tensor(shape=(3,), dtype=tf.uint8),
            'num_intersections': tf.int64,
            'intersections': tfds.features.Tensor(shape=(2700, 2),
                                                  dtype=tf.float32),
            'vertices': tfds.features.Tensor(shape=(75, 2), dtype=tf.float32),
            'num_vertices': tf.int64,
        }),
        supervised_keys=None,  # Set to `None` to disable
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    num_images = self.config.num_images

    return {
        'train': self._generate_examples(
            num_images=int(num_images*.9),
            config=self.config),
        'test': self._generate_examples(
            num_images=num_images - int(num_images*.9), config=self.config)
    }

  def _generate_examples(self, num_images: int, config: ShapesConfig):
    """Yields examples."""
    beam = tfds.core.lazy_imports.apache_beam

    image_indices = range(num_images)

    # split image_patchs into shards
    images_per_shard = 10
    total_shards = len(image_indices) // images_per_shard

    if total_shards > 0:
      split_indices = np.array(np.array_split(np.array(image_indices),
                                              total_shards, axis=0)).tolist()
    else:
      split_indices = [image_indices]

    create_fn = functools.partial(self._generate_multiple_images, config=config)

    return (
        'Create' >> beam.Create(split_indices)
        | 'Process Images' >> beam.Map(create_fn)
        | 'Flatten' >> beam.FlatMap(flatten)
        | 'Add Keys' >> beam.Map(lambda x: (x['image_index'], x))
        )

  def _generate_multiple_images(self, image_indices, config):

    all_images = []

    for image_idx in image_indices:
      image_dict = make_kaleido_image.generate_image(image_idx, config)
      all_images.append(image_dict)

    return all_images
