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

"""Data generators for Scenic."""

import functools
import importlib
from typing import Callable, List

from absl import logging
from scenic.dataset_lib import dataset_utils

# The dict below hardcodes import that define datasets. This is necessary for
# several reasons:
# 1) Datasets are only registered once they are defined (have been imported).
# 2) We don't want the user code (e.g. trainers / projects) to have to import
#    the dataset modules. Instead we'd like to do it for them.
# 3) And finally we don't want to import all datasets available to unless if the
#    the user code does not need them.
# TODO(b/186631707): This routing table is not a great solution because it
#  requires every new dataset to modify this import routing table. Going forward
#  we should find a way to avoid that.
_IMPORT_TABLE = {
    'cifar10': 'scenic.dataset_lib.cifar10_dataset',
    'cityscapes': 'scenic.dataset_lib.cityscapes_dataset',
    'imagenet': 'scenic.dataset_lib.imagenet_dataset',
    'fashion_mnist': 'scenic.dataset_lib.fashion_mnist_dataset',
    'mnist': 'scenic.dataset_lib.mnist_dataset',
    'bair': 'scenic.dataset_lib.bair_dataset',
    'oxford_pets': 'scenic.dataset_lib.oxford_pets_dataset',
    'svhn': 'scenic.dataset_lib.svhn_dataset',
    'video_tfrecord_dataset': (
        'scenic.projects.vivit.data.video_tfrecord_dataset'
    ),
    'av_asr_tfrecord_dataset': (
        'scenic.projects.avatar.datasets.av_asr_tfrecord_dataset'
    ),
    'bit': 'scenic.dataset_lib.big_transfer.bit',
    'bert_wikibooks': (
        'scenic.projects.baselines.bert.datasets.bert_wikibooks_dataset'
    ),
    'bert_glue': 'scenic.projects.baselines.bert.datasets.bert_glue_dataset',
    'coco_detr_detection': (
        'scenic.projects.baselines.detr.input_pipeline_detection'
    ),
    'cityscapes_variants': (
        'scenic.projects.robust_segvit.datasets.cityscapes_variants'
    ),
    'robust_segvit_segmentation': (
        'scenic.projects.robust_segvit.datasets.segmentation_datasets'
    ),
    'robust_segvit_variants': (
        'scenic.projects.robust_segvit.datasets.segmentation_variants'
    ),
    'flexio': 'scenic.dataset_lib.flexio.flexio',
}


class DatasetRegistry(object):
  """Static class for keeping track of available datasets."""
  _REGISTRY = {}

  @classmethod
  def add(cls, name: str, builder_fn: Callable[..., dataset_utils.Dataset]):
    """Add a dataset to the registry, i.e. register a dataset.

    Args:
      name: Dataset name (must be unique).
      builder_fn: Function to be called to construct the datasets. Must accept
        dataset-specific arguments and return a dataset description.

    Raises:
      KeyError: If the provided name is not unique.
    """
    if name in cls._REGISTRY:
      raise KeyError(f'Dataset with name ({name}) already registered.')
    cls._REGISTRY[name] = builder_fn

  @classmethod
  def get(cls, name: str) -> Callable[..., dataset_utils.Dataset]:
    """Get a dataset from the registry by its name.

    Args:
      name: Dataset name.

    Returns:
      Dataset builder function that accepts dataset-specific parameters and
      returns a dataset description.

    Raises:
      KeyError: If the dataset is not found.
    """
    if name not in cls._REGISTRY:
      if name in _IMPORT_TABLE:
        module = _IMPORT_TABLE[name]
        importlib.import_module(module)
        logging.info(
            'On-demand import of dataset (%s) from module (%s).', name, module)
        if name not in cls._REGISTRY:
          raise KeyError(f'Imported module ({module}) did not register dataset'
                         f'({name}). Please check that dataset names match.')
      else:
        raise KeyError(f'Unknown dataset ({name}). Did you import the dataset '
                       f'module explicitly?')
    return cls._REGISTRY[name]

  @classmethod
  def list(cls) -> List[str]:
    """List registered datasets."""
    return list(cls._REGISTRY.keys())


def add_dataset(name: str, *args, **kwargs):
  """Decorator for shorthand dataset registdation."""
  def inner(builder_fn: Callable[..., dataset_utils.Dataset]
           ) -> Callable[..., dataset_utils.Dataset]:
    DatasetRegistry.add(name, functools.partial(builder_fn, *args, **kwargs))
    return builder_fn
  return inner


def get_dataset(dataset_name: str) -> Callable[..., dataset_utils.Dataset]:
  """Maps dataset name to a dataset_builder.

  API kept for compatibility of existing code with the DatasetRegistry.

  Args:
    dataset_name: Dataset name.

  Returns:
    A dataset builder.
  """
  return DatasetRegistry.get(dataset_name)
