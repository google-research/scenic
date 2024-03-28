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

"""Contains config utility functions."""

from typing import List

import ml_collections

MODEL_SIZE_ORDER = ['Ti', 'S', 'B', 'L', 'H']
HIDDEN_SIZES = {'Ti': 192, 'S': 384, 'B': 768, 'L': 1024, 'H': 1280}
MLP_DIMS = {'Ti': 768, 'S': 1536, 'B': 3072, 'L': 4096, 'H': 5120}
NUM_HEADS = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16, 'H': 16}
NUM_LAYERS = {'Ti': 12, 'S': 12, 'B': 12, 'L': 24, 'H': 32}
# Default patch sizes and they can be overridden by the config.
PATCH_SIZES = {
    'Ti': (16, 16),
    'S': (16, 16),
    'B': (16, 16),
    'L': (16, 16),
    'H': (14, 14),
}


def parse_view_configs(variant: str) -> List[ml_collections.ConfigDict]:
  """Parse per-view model configs from an encoded text.

  Each view is encoded in the format of 'vit_version/HxWxT' where H, W, and T
  are the height, width, and temporal dimension of the tubelets, respectively.
  H and W are optional and they are set to 16 for Tiny, Small, Base, and
  Large models and 14 to Huge model by default.

  We use '+' to put views together. For example, 'S/8+B/4+L/2' is a three-view
  model composed of a Vit-S (tubelet size=[16, 16, 8]), a ViT-B (tubelet
  size=[16, 16, 4]), and a ViT-L (tubelet size=[16, 16, 2]).

  Args:
    variant: a str encoding the model structure.

  Returns:
    a list of per-view model configs.
  """
  view_configs = []
  views = variant.split('+')
  for view_variant in views:
    version, tubelet_size = view_variant.split('/')
    shape = tubelet_size.split('x')
    view_config = ml_collections.ConfigDict()
    view_config.hidden_size = HIDDEN_SIZES[version]
    view_config.patches = ml_collections.ConfigDict()
    view_config.num_heads = NUM_HEADS[version]
    view_config.mlp_dim = MLP_DIMS[version]
    view_config.num_layers = NUM_LAYERS[version]
    if len(shape) == 1:
      num_frames = int(shape[0])
      view_config.patches.size = PATCH_SIZES[version] + (num_frames,)
    elif len(shape) == 2:
      view_config.patches.size = (int(shape[0]), int(shape[0]), int(shape[1]))
    elif len(shape) == 3:
      view_config.patches.size = (int(shape[0]), int(shape[1]), int(shape[2]))
    else:
      raise ValueError(f'Model variant {variant} is invalid.')

    view_configs.append(view_config)
  return view_configs
