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

"""Class and splits information for the datasets.

Datasets:

ade20k_ind: Similar to ADE20K dataset without ADE20K_OOD_CLASSES
ade20k_ood_open: ADE20K dataset including only ADE20K_OOD_CLASSES
"""

import collections
import dataclasses
from typing import Dict, Any, List, Optional


# ADEK OOD Classes to ignore
ADE20K_OOD_CLASSES = ['chair', 'armchair', 'swivel chair']
# pylint: disable=line-too-long
ADE20K_CORRUPTED_DIR = 'gs://ub-ekb/tensorflow_datasets/ad_e20k_corrupted/tfrecords/v.0.0'

ADE20K_TFDS_NAME = 'ade20k'
ADE20K_CORRUPTED_DIR = 'gs://ub-ekb/tensorflow_datasets/ade20k/tfrecords/v.0.0'
# pylint: enable=line-too-long

# ADE20K-C
ADE20K_C_SEVERITIES = range(1, 6)
ADE20K_C_CORRUPTIONS = [
    'gaussian_noise',
]


@dataclasses.dataclass(frozen=True)
class DatasetInfo:
  tfds_name: str
  image_key: str
  label_key: str
  classes: List[Any]
  pixels_per_class: Optional[Dict[int, int]] = None
  ood_classes: Optional[List[Any]] = None
  data_dir: Optional[str] = None

# Information for Cityscapes dataset.
# Based on https://github.com/mcordts/cityscapesScripts
CityscapesClass = collections.namedtuple(
    'CityscapesClass',
    ['name', 'id', 'train_id', 'category', 'category_id', 'has_instances',
     'ignore_in_eval', 'color'])

CITYSCAPES_CLASSES = [
    CityscapesClass(
        'unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass(
        'ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass(
        'rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass(
        'out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass(
        'static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass(
        'dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass(
        'ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass(
        'road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass(
        'sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass(
        'parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass(
        'rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass(
        'building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass(
        'wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass(
        'fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass(
        'guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass(
        'bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass(
        'tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass(
        'pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass(
        'polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass(
        'traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass(
        'traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass(
        'vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass(
        'terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass(
        'sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass(
        'person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass(
        'rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass(
        'car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass(
        'truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass(
        'bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass(
        'caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass(
        'trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass(
        'train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass(
        'motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass(
        'bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass(
        'license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

# Number of pixels per Cityscapes class ID in the training set:
CITYSCAPES_PIXELS_PER_CID = {
    7: 3806423808,
    8: 629490880,
    11: 2354443008,
    12: 67089092,
    13: 91210616,
    17: 126753000,
    19: 21555918,
    20: 57031712,
    21: 1647446144,
    22: 119165328,
    23: 415038624,
    24: 126403824,
    25: 13856368,
    26: 725164864,
    27: 27588982,
    28: 24276994,
    31: 24195352,
    32: 10207740,
    33: 42616088
}

CITYSCAPES = DatasetInfo(
    tfds_name='cityscapes/semantic_segmentation',
    image_key='image_left',
    label_key='segmentation_label',
    classes=CITYSCAPES_CLASSES,
    pixels_per_class=CITYSCAPES_PIXELS_PER_CID)

# Information for ADE20k dataset
ADE20KClass = collections.namedtuple(
    'ADE20KClass', ['name', 'id', 'train_id', 'ignore_in_eval', 'color'])

ADE20K_CLASS_NAMES = [
    'background', 'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
    'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
    'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water',
    'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field',
    'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub',
    'railing', 'cushion', 'base', 'box', 'column', 'signboard',
    'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace',
    'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case',
    'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge',
    'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
    'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer',
    'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel',
    'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth',
    'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land',
    'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage',
    'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
    'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag',
    'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name',
    'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen',
    'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray',
    'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board',
    'shower', 'radiator', 'glass', 'clock', 'flag'
]

ADE20K_CLASS_COLORS = [
    [0, 0, 0], [120, 120, 120], [180, 120, 120], [6, 230, 230],
    [80, 50, 50], [4, 200, 3], [120, 120, 80], [140, 140, 140],
    [204, 5, 255], [230, 230, 230], [4, 250, 7], [224, 5, 255],
    [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51],
    [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7],
    [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51],
    [11, 102, 255], [255, 7, 71], [255, 9, 224], [9, 7, 230],
    [220, 220, 220], [255, 9, 92], [112, 9, 255], [8, 255, 214],
    [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10],
    [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6],
    [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41],
    [255, 5, 153], [6, 51, 255], [235, 12, 255], [160, 150, 20],
    [0, 163, 255], [140, 140, 140], [250, 10, 15], [20, 255, 0],
    [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0],
    [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255],
    [31, 0, 255], [11, 200, 200], [255, 82, 0], [0, 255, 245],
    [0, 61, 255], [0, 255, 112], [0, 255, 133], [255, 0, 0],
    [255, 163, 0], [255, 102, 0], [194, 255, 0], [0, 143, 255],
    [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173],
    [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0],
    [255, 0, 255], [255, 0, 245], [255, 0, 102], [255, 173, 0],
    [255, 0, 20], [255, 184, 184], [0, 31, 255], [0, 255, 61],
    [0, 71, 255], [255, 0, 204], [0, 255, 194], [0, 255, 82],
    [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255],
    [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10],
    [255, 112, 0], [143, 255, 0], [82, 0, 255], [163, 255, 0],
    [255, 235, 0], [8, 184, 170], [133, 0, 255], [0, 255, 92],
    [184, 0, 255], [255, 0, 31], [0, 184, 255], [0, 214, 255],
    [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255],
    [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0],
    [255, 0, 163], [255, 204, 0], [255, 0, 143], [0, 255, 235],
    [133, 255, 0], [255, 0, 235], [245, 0, 255], [255, 0, 122],
    [255, 245, 0], [10, 190, 212], [214, 255, 0], [0, 204, 255],
    [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255],
    [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255],
    [0, 245, 255], [71, 0, 255], [122, 0, 255], [0, 255, 184],
    [0, 92, 255], [184, 255, 0], [0, 133, 255], [255, 214, 0],
    [25, 194, 194], [102, 255, 0], [92, 0, 255]]

# -------------------------
# Construct ade20k dataset:
# -------------------------
ADE20K_CLASSES = []

for i in range(151):
  c = ADE20KClass(ADE20K_CLASS_NAMES[i], i,
                  255 if ADE20K_CLASS_NAMES[i] == 'background' else i - 1,
                  True if ADE20K_CLASS_NAMES[i] == 'background' else False,
                  ADE20K_CLASS_COLORS[i])
  ADE20K_CLASSES.append(c)

ADE20K = DatasetInfo(
    tfds_name=ADE20K_TFDS_NAME,
    image_key='image',
    label_key='segmentation',
    classes=ADE20K_CLASSES,
    pixels_per_class=None,
    data_dir=ADE20K_DIR)

# -----------------------------
# Construct ade20k_ind dataset:
# -----------------------------
# Construct a subset ade20k dataset which assigns the classes in
# ADE20K_OOD_CLASSES as background.
# ignore the background and the OOD classes during eval
# TODO(kellybuchanan): put this as function.
train_class = 0
ADE20KSUBSET_CLASSES = []
for i in range(151):
  name = ADE20K_CLASS_NAMES[i]
  train_id = 255 if name == 'background' else train_class
  ignore_in_eval = True if name == 'background' else False
  if name in ADE20K_OOD_CLASSES:
    train_id = 255
    ignore_in_eval = True
  else:
    train_class += 1
  c = ADE20KClass(name, i, train_id, ignore_in_eval, ADE20K_CLASS_COLORS[i])
  ADE20KSUBSET_CLASSES.append(c)

ADE20KSUBSET = DatasetInfo(
    tfds_name=ADE20K_TFDS_NAME,
    image_key='image',
    label_key='segmentation',
    classes=ADE20KSUBSET_CLASSES,
    pixels_per_class=None,
    data_dir=ADE20K_DIR)

# ----------------------------------
# Construct ade20k_odd_open dataset:
# ----------------------------------
# Generate openset dataset, where all classes except for ADE20K_OOD classes
# are considered background or class 0 and ADE20K_OOD_CLASSES are class 1.
# ignore the background during eval, other OOD classes are set to
ADE20KOPEN_CLASSES = []
for i in range(151):
  name = ADE20K_CLASS_NAMES[i]
  if name in ADE20K_OOD_CLASSES:
    train_id = 1
    ignore_in_eval = False
  elif name == 'background':
    train_id = 255
    ignore_in_eval = True
  else:
    train_id = 0
    ignore_in_eval = False

  c = ADE20KClass(name, i, train_id, ignore_in_eval, ADE20K_CLASS_COLORS[i])
  ADE20KOPEN_CLASSES.append(c)

# Classes defined as Background/InD/OOD sets.
c255 = ADE20KClass('background', 0, 255, True, [0, 0, 0])
c0 = ADE20KClass('ind', 1, 0, False, [0, 0, 1])
c1 = ADE20KClass('ood', 2, 1, False, [1, 0, 0])
ADE20KOPEN_3CLASSES = [c255, c0, c1]

ADE20KOPEN = DatasetInfo(
    tfds_name=ADE20K_TFDS_NAME,
    image_key='image',
    label_key='segmentation',
    classes=ADE20KOPEN_3CLASSES,
    ood_classes=ADE20KOPEN_CLASSES,
    pixels_per_class=None,
    data_dir=ADE20K_DIR)


def build_datasets():
  """Build datasets."""

  local_dataset = {
      'cityscapes': CITYSCAPES,
      'ade20k': ADE20K,
      'ade20k_ind': ADE20KSUBSET,
      'ade20k_ood_open': ADE20KOPEN,
  }

  # -----------------------------
  # Construct ade20k_c dataset:
  # -----------------------------
  for severity in ADE20K_C_SEVERITIES:
    for corruption in ADE20K_C_CORRUPTIONS:
      tfds_dataset_name = f'ade20k_corrupted/ade20k_{corruption}_{severity}'
      temp_dataset_name = f'ade20k_c_{corruption}_{severity}'
      local_dataset[temp_dataset_name] = DatasetInfo(
          tfds_name=tfds_dataset_name,
          image_key='image',
          label_key='annotations',
          classes=ADE20K_CLASSES,
          pixels_per_class=None,
          data_dir=ADE20K_CORRUPTED_DIR,
      )

  # -------------------------------
  # Construct ade20k_ind_c dataset:
  # -------------------------------
  for severity in ADE20K_C_SEVERITIES:
    for corruption in ADE20K_C_CORRUPTIONS:
      tfds_dataset_name = f'ade20k_corrupted/ade20k_{corruption}_{severity}'
      temp_dataset_name = f'ade20k_ind_c_{corruption}_{severity}'
      local_dataset[temp_dataset_name] = DatasetInfo(
          tfds_name=tfds_dataset_name,
          image_key='image',
          label_key='annotations',
          classes=ADE20KSUBSET_CLASSES,
          pixels_per_class=None,
          data_dir=ADE20K_CORRUPTED_DIR,
      )

  # ------------------------------------
  # Construct ade20k_ood_open_c dataset:
  # ------------------------------------
  for severity in ADE20K_C_SEVERITIES:
    for corruption in ADE20K_C_CORRUPTIONS:
      tfds_dataset_name = f'ade20k_corrupted/ade20k_{corruption}_{severity}'
      temp_dataset_name = f'ade20k_ood_open_c_{corruption}_{severity}'
      local_dataset[temp_dataset_name] = DatasetInfo(
          tfds_name=tfds_dataset_name,
          image_key='image',
          label_key='annotations',
          classes=ADE20KOPEN_3CLASSES,
          ood_classes=ADE20KOPEN_CLASSES,
          pixels_per_class=None,
          data_dir=ADE20K_CORRUPTED_DIR,
      )

  return local_dataset

# ------------------
# BUILD all datasets
# ------------------
DATASETS = build_datasets()


def get_info(dataset: str) -> DatasetInfo:
  """Returns dataset information for a dataset."""
  info = DATASETS.get(dataset)
  if not info:
    raise ValueError(f'{dataset} is not available')
  return info
