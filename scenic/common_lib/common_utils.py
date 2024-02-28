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

"""Utility functions that do not fit into other common_lib modules."""

import importlib
import types

from absl import logging
import jax
import ml_collections


def recursive_reload(module: types.ModuleType, package_restrict: str):
  """Recursively reload a module and the modules it imports.

  Args:
    module: The module to reload.

    package_restrict: Only modules with this prefix will be reloaded. For
      example, if package_restrict is "scenic.projects", only modules under
      scenic.projects will be reloaded. package_restrict must always be set to
      avoid reloading of built-in or unrelated packages that should not be
      reloaded (e.g. Numpy).

  Returns:
    The reloaded module object.

  Raises:
    ValueError if package_restrict is empyt.
  """
  reloaded = set()
  if not package_restrict:
    raise ValueError('package_restrict must be non-empty.')

  def reload(m):
    if m in reloaded:
      return m
    reloaded.add(m)
    for attribute_name in dir(m):
      attribute = getattr(m, attribute_name)
      if (isinstance(attribute, types.ModuleType) and
          attribute.__name__.startswith(package_restrict)):
        reload(attribute)
    logging.info('Reloading %s', m.__name__)
    return importlib.reload(m)

  return reload(module)


def to_config_dict_heuristic(
    config: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Heuristically converts dicts inside a ConfigDict into ConfigDicts.

  This function detects lists and tuples with dicts and converts those dicts
  into ConfigDicts. This will address most failure cases, but the function will
  not be able resolve nested cases (e.g. list(dict(list(....)))).

  Arguments:
    config: Config to attempt fixing.

  Returns:
    Probably fixed config.
  """
  def maybe_config_dict(x):
    if isinstance(x, dict):
      return ml_collections.ConfigDict(x)
    return x

  def maybe_config_dict_in_list(x):
    if isinstance(x, (list, tuple)):
      return jax.tree_util.tree_map(
          maybe_config_dict, x, is_leaf=lambda y: isinstance(y, dict)
      )
    return x

  config = jax.tree_util.tree_map(
      maybe_config_dict_in_list,
      config.to_dict(),
      is_leaf=lambda x: isinstance(x, list),
  )
  return ml_collections.ConfigDict(config)
