# Copyright 2023 The Scenic Authors.
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

"""Returns default config for kaleidoshapes dataset."""

import ml_collections


def get_config_kaleidoshapes(
    dataset_dir: str,
) -> ml_collections.ConfigDict:
  """Returns default config for kaleidoshapes dataset."""

  dataset_config = ml_collections.ConfigDict()
  dataset_config.name = 'kaleidoshapes'
  dataset_config.train_batchsize = 1
  dataset_config.eval_batchsize = 1
  dataset_config.prefetch_to_host = 0
  dataset_config.num_train_images = 90_000
  dataset_config.num_eval_images = 10_000
  dataset_config.image_size = (240, 320, 3)
  dataset_config.crop = True
  dataset_config.crop_size = (125, 125, 3)
  dataset_config.min_noise_level = .3
  dataset_config.max_noise_level = .8
  dataset_config.iv_radius = 7
  dataset_config.add_greyscale_samples = True
  dataset_config.prop_grey = .10
  dataset_config.max_num_shapes = 15
  dataset_config.input_size = dataset_config.crop_size if (
      dataset_config.crop) else dataset_config.image_size
  dataset_config.dataset_dir = dataset_dir

  return dataset_config
