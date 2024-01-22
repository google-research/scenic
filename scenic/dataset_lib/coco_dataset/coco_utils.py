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

"""Common utils for coco dataset."""

import collections
import json
from typing import Dict, Optional


OBJECTS365_LABEL_MAP_PATH = (
    'scenic/dataset_lib/coco_dataset/data/objects365_class_names.txt')
LVIS_LABEL_MAP_PATH = (
    'scenic/dataset_lib/coco_dataset/data/lvis_label_map.json')
OI_LABEL_MAP_PATH = {
    'open_images_v4': (
        'scenic/dataset_lib/coco_dataset/data/open_images_v4-classes.csv'),
    # For open_images_v5_boxes_with_masks, we use the subset of classes that has
    # segmentations.
    'open_images_v5_boxes_with_masks': (
        'scenic/dataset_lib/coco_dataset/data/'
        'open_images_v5-classes-segmentation.csv'),
    'open_images_v5': (
        'scenic/dataset_lib/coco_dataset/data/open_images_v5-classes.csv'),
}

COCO_2017_THINGS = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    12: 'stop sign',
    13: 'parking meter',
    14: 'bench',
    15: 'bird',
    16: 'cat',
    17: 'dog',
    18: 'horse',
    19: 'sheep',
    20: 'cow',
    21: 'elephant',
    22: 'bear',
    23: 'zebra',
    24: 'giraffe',
    25: 'backpack',
    26: 'umbrella',
    27: 'handbag',
    28: 'tie',
    29: 'suitcase',
    30: 'frisbee',
    31: 'skis',
    32: 'snowboard',
    33: 'sports ball',
    34: 'kite',
    35: 'baseball bat',
    36: 'baseball glove',
    37: 'skateboard',
    38: 'surfboard',
    39: 'tennis racket',
    40: 'bottle',
    41: 'wine glass',
    42: 'cup',
    43: 'fork',
    44: 'knife',
    45: 'spoon',
    46: 'bowl',
    47: 'banana',
    48: 'apple',
    49: 'sandwich',
    50: 'orange',
    51: 'broccoli',
    52: 'carrot',
    53: 'hot dog',
    54: 'pizza',
    55: 'donut',
    56: 'cake',
    57: 'chair',
    58: 'couch',
    59: 'potted plant',
    60: 'bed',
    61: 'dining table',
    62: 'toilet',
    63: 'tv',
    64: 'laptop',
    65: 'mouse',
    66: 'remote',
    67: 'keyboard',
    68: 'cell phone',
    69: 'microwave',
    70: 'oven',
    71: 'toaster',
    72: 'sink',
    73: 'refrigerator',
    74: 'book',
    75: 'clock',
    76: 'vase',
    77: 'scissors',
    78: 'teddy bear',
    79: 'hair drier',
    80: 'toothbrush',
}

COCO_2017_STUFF = {
    81: 'banner',
    82: 'blanket',
    83: 'bridge',
    84: 'cardboard',
    85: 'counter',
    86: 'curtain',
    87: 'door-stuff',
    88: 'floor-wood',
    89: 'flower',
    90: 'fruit',
    91: 'gravel',
    92: 'house',
    93: 'light',
    94: 'mirror-stuff',
    95: 'net',
    96: 'pillow',
    97: 'platform',
    98: 'playingfield',
    99: 'railroad',
    100: 'river',
    101: 'road',
    102: 'roof',
    103: 'sand',
    104: 'sea',
    105: 'shelf',
    106: 'snow',
    107: 'stairs',
    108: 'tent',
    109: 'towel',
    110: 'wall-brick',
    111: 'wall-stone',
    112: 'wall-tile',
    113: 'wall-wood',
    114: 'water-other',
    115: 'window-blind',
    116: 'window-other',
    117: 'tree-merged',
    118: 'fence-merged',
    119: 'ceiling-merged',
    120: 'sky-other-merged',
    121: 'cabinet-merged',
    122: 'table-merged',
    123: 'floor-other-merged',
    124: 'pavement-merged',
    125: 'mountain-merged',
    126: 'grass-merged',
    127: 'dirt-merged',
    128: 'paper-merged',
    129: 'food-other-merged',
    130: 'building-other-merged',
    131: 'rock-merged',
    132: 'wall-other-merged',
    133: 'rug-merged',
}

# Obtained from https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip.
# (Also see dataset site at https://github.com/lichengunc/refer.)
REF_COCO = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush',
}


def get_label_map(tfds_name: str) -> Dict[int, str]:
  """Returns a {label: name} dict for the COCO dataset."""

  if tfds_name == 'coco/2017':
    return {0: 'padding', **COCO_2017_THINGS}
  elif tfds_name == 'coco/2017_panoptic':
    return {0: 'padding', **COCO_2017_THINGS, **COCO_2017_STUFF}
  elif tfds_name == 'ref_coco':
    return {0: 'padding', **REF_COCO}
  elif tfds_name == 'lvis':
    return get_lvis_label_map()
  elif tfds_name in ['objects365', 'scenic:objects365']:
    return get_objects365_label_map()
  elif tfds_name.startswith('open_images'):
    return get_openimages_label_map(tfds_name)
  else:
    raise ValueError(f'Unsupported TFDS name: {tfds_name}')


