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

"""Processes Kaleidoshape images."""
import tensorflow as tf


def process_kaleido_images(example, dataset_config, nmodel):
  """Processes Kaleido images by adding noise, cropping, etc.

  Args:
    example: A dictionary of a kaleidoshape image and corresponding features.
    dataset_config: A ml_collections containing the dataset configuration.
    nmodel: A function that takes an image and returns a noisy image.

  Returns:
    A dictionary of the processed image.
  """
  del example['shapes']['type']
  del example['tfds_id']

  example['image'] = tf.cast(example['image'], dtype=tf.float32) / 255.0
  example['boundaries'] = (
      tf.cast(example['boundaries'], dtype=tf.float32) / 255.0
  )
  example['distances'] = tf.expand_dims(example['distances'], -1)

  input_image_size = tf.cast(
      dataset_config.get('image_size', (240, 320, 3)), dtype=tf.int32
  )

  if dataset_config.get('crop', True):
    crop_size = tf.cast(
        dataset_config.get('crop_size', (100, 100, 3)), dtype=tf.int32
    )
    margin = 10  # in pixels

    y0 = tf.random.uniform(
        [],
        margin,
        input_image_size[0] - crop_size[0] + 1 - margin,
        dtype=tf.int32,
    )
    x0 = tf.random.uniform(
        [], margin, input_image_size[1] - crop_size[1] - margin, dtype=tf.int32
    )

    example['image'] = example['image'][
        y0 : y0 + crop_size[0], x0 : x0 + crop_size[1], :
    ]
    example['boundaries'] = example['boundaries'][
        y0 : y0 + crop_size[0], x0 : x0 + crop_size[1], :
    ]
    example['distances'] = example['distances'][
        y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]
    ]
    example['segments'] = example['segments'][
        y0 : y0 + crop_size[0], x0 : x0 + crop_size[1], :
    ]

    image_size = crop_size
  else:
    x0, y0 = 0, 0
    image_size = input_image_size

  example['crop_start'] = tf.cast([x0, y0], dtype=tf.int32)

  # Make importance mask:
  if dataset_config.get('make_iv_mask', True):
    centers = tf.cast(
        example['intersections'][: example['num_intersections']], tf.float32
    )
    vertices = tf.cast(
        example['vertices'][: example['num_vertices']], tf.float32
    )
    cv_all = tf.concat([centers, vertices], axis=0)

    x, y = tf.meshgrid(
        tf.range(x0, x0 + image_size[0]), tf.range(y0, y0 + image_size[1])
    )

    x = tf.cast(tf.expand_dims(x, 2), tf.float32)
    y = tf.cast(tf.expand_dims(y, 2), tf.float32)

    cv_x = tf.expand_dims(tf.expand_dims(cv_all[:, 0], 0), 1) * tf.cast(
        tf.math.maximum(input_image_size[0], input_image_size[1]), tf.float32
    )
    cv_y = tf.expand_dims(tf.expand_dims(cv_all[:, 1], 0), 1) * tf.cast(
        tf.math.maximum(input_image_size[0], input_image_size[1]), tf.float32
    )

    radius = dataset_config.get('iv_radius', 10.0)

    iv_mask = (
        tf.exp(-((x - cv_x) ** 2 + (y - cv_y) ** 2) / (2 * radius**2)) + 1e-4
    )
    iv_mask = tf.math.reduce_max(iv_mask, -1)
    # iv_mask = iv_mask/tf.math.reduce_max(iv_mask)

    example['iv_mask'] = tf.expand_dims(iv_mask, -1)
  else:
    example['iv_mask'] = tf.ones_like(example['boundaries'])

  ############### Add noise ###################################################

  example['clean_image'] = example['image']
  example['image'] = nmodel(example['image'], image_size)

  #############################################################################

  # If inject greyscale images:
  if dataset_config.get('add_greyscale_samples', True):
    if tf.random.uniform([], 0, 1, dtype=tf.float32) < dataset_config.get(
        'prop_grey', 0.2
    ):
      example['image'] = tf.repeat(
          tf.math.reduce_mean(example['image'], axis=-1, keepdims=True),
          3,
          axis=-1,
      )
      example['clean_image'] = tf.repeat(
          tf.math.reduce_mean(example['clean_image'], axis=-1, keepdims=True),
          3,
          axis=-1,
      )

  # Transpose all image matrices
  example['image'] = tf.transpose(example['image'], (2, 0, 1))
  example['clean_image'] = tf.transpose(example['clean_image'], (2, 0, 1))
  example['boundaries'] = tf.transpose(example['boundaries'], (2, 0, 1))
  example['distances'] = tf.transpose(example['distances'], (2, 0, 1))
  example['segments'] = tf.transpose(example['segments'], (2, 0, 1))

  return example
