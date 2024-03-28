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

"""Base config for CLIP."""

import ml_collections
from scenic.projects.lang4video.configs import base

CLIP_CONFIG_NAMES = [
    'resnet_50', 'resnet_101', 'vit_b32', 'resnet_50x4', 'vit_b16',
    'resnet_50x16', 'resnet_50x64', 'vit_l14', 'vit_l14_336px'
]

CLIP_IMAGENET_TEMPLATES = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

CLIP_KINETICS_TEMPLATES = [
    'a photo of {}.',
    'a photo of a person {}.',
    'a photo of a person using {}.',
    'a photo of a person doing {}.',
    'a photo of a person during {}.',
    'a photo of a person performing {}.',
    'a photo of a person practicing {}.',
    'a video of {}.',
    'a video of a person {}.',
    'a video of a person using {}.',
    'a video of a person doing {}.',
    'a video of a person during {}.',
    'a video of a person performing {}.',
    'a video of a person practicing {}.',
    'a example of {}.',
    'a example of a person {}.',
    'a example of a person using {}.',
    'a example of a person doing {}.',
    'a example of a person during {}.',
    'a example of a person performing {}.',
    'a example of a person practicing {}.',
    'a demonstration of {}.',
    'a demonstration of a person {}.',
    'a demonstration of a person using {}.',
    'a demonstration of a person doing {}.',
    'a demonstration of a person during {}.',
    'a demonstration of a person performing {}.',
    'a demonstration of a person practicing {}.',
]

CLIP_UCF_TEMPLATES = [
    'a photo of a person {}.',
    'a video of a person {}.',
    'a example of a person {}.',
    'a demonstration of a person {}.',
    'a photo of the person {}.',
    'a video of the person {}.',
    'a example of the person {}.',
    'a demonstration of the person {}.',
    'a photo of a person using {}.',
    'a video of a person using {}.',
    'a example of a person using {}.',
    'a demonstration of a person using {}.',
    'a photo of the person using {}.',
    'a video of the person using {}.',
    'a example of the person using {}.',
    'a demonstration of the person using {}.',
    'a photo of a person doing {}.',
    'a video of a person doing {}.',
    'a example of a person doing {}.',
    'a demonstration of a person doing {}.',
    'a photo of the person doing {}.',
    'a video of the person doing {}.',
    'a example of the person doing {}.',
    'a demonstration of the person doing {}.',
    'a photo of a person during {}.',
    'a video of a person during {}.',
    'a example of a person during {}.',
    'a demonstration of a person during {}.',
    'a photo of the person during {}.',
    'a video of the person during {}.',
    'a example of the person during {}.',
    'a demonstration of the person during {}.',
    'a photo of a person performing {}.',
    'a video of a person performing {}.',
    'a example of a person performing {}.',
    'a demonstration of a person performing {}.',
    'a photo of the person performing {}.',
    'a video of the person performing {}.',
    'a example of the person performing {}.',
    'a demonstration of the person performing {}.',
    'a photo of a person practicing {}.',
    'a video of a person practicing {}.',
    'a example of a person practicing {}.',
    'a demonstration of a person practicing {}.',
    'a photo of the person practicing {}.',
    'a video of the person practicing {}.',
    'a example of the person practicing {}.',
    'a demonstration of the person practicing {}.',
]


def get_clip_image_size(config_name: str) -> int:
  return {
      'debug': 224,
      'resnet_50': 224,
      'resnet_101': 224,
      'resnet_50x4': 288,
      'resnet_50x16': 384,
      'resnet_50x64': 448,
      'vit_b32': 224,
      'vit_b16': 224,
      'vit_l14': 224,
      'vit_l14_336px': 336,
  }[config_name]


def get_config(run_local: str = '') -> ml_collections.ConfigDict:
  """Returns the experiment configuration."""
  config = base.get_config(run_local)

  config.experiment_name = 'clip'

  config.model_name = 'image_text'

  config.model.encoder_name = 'clip'
  config.model.encoder.config_name = 'vit_b32'

  # The following configurations are meant to be used by DMVR datasets but also
  # as input to other parts of the config.

  config.dataset_configs.min_resize = get_clip_image_size(
      config.model.encoder.config_name)
  config.dataset_configs.crop_size = get_clip_image_size(
      config.model.encoder.config_name)
  config.dataset_configs.normalization_mean = (0.48145466, 0.4578275,
                                               0.40821073)
  config.dataset_configs.normalization_std = (0.26862954, 0.26130258,
                                              0.27577711)
  config.dataset_configs.tokenizer.tokenizer_type = 'clip'

  config.dataset_configs.tokenizer_type = 'dmvr'
  config.dataset_configs.vocab_size = 49_408

  return config
