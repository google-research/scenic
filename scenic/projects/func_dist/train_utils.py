"""Extend scenic.train_lib_deprecated.train_utils with custom datasets."""

from scenic.projects.func_dist.datasets import ssv2_regression  # pylint: disable=unused-import
from scenic.train_lib_deprecated import train_utils

get_dataset = train_utils.get_dataset
TrainState = train_utils.TrainState
