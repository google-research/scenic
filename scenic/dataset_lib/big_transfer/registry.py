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

"""Global Registry for the task adaptation framework.

"""

import ast
import functools
import logging


def partialclass(cls, *base_args, **base_kwargs):
  """Builds a subclass with partial application of the given args and keywords.

  Equivalent to functools.partial performance, base_args are preprended to the
  positional arguments given during object initialization and base_kwargs are
  updated with the kwargs given later.

  Args:
    cls: The base class.
    *base_args: Positional arguments to be applied to the subclass.
    **base_kwargs: Keyword arguments to be applied to the subclass.

  Returns:
    A subclass of the input class.

  Author: Joan Puigcerver (jpuigcerver@)
  """

  class _NewClass(cls):

    def __init__(self, *args, **kwargs):
      bound_args = base_args + args
      bound_kwargs = base_kwargs.copy()
      bound_kwargs.update(kwargs)
      super(_NewClass, self).__init__(*bound_args, **bound_kwargs)

  return _NewClass


def partialfactory(cls, lookup_string, *base_args, **base_kwargs):
  """Builds a factory with partial application of given args and keywords.

  Equivalent to functools.partial, but with a warning to prevent potential
  future headaches when overwriting keyword arguments. base_args are prepended
  to the positional arguments given during initialization and base_kwargs are
  updated with the kwargs given later.

  Args:
    cls: The base class.
    lookup_string: String for logging purposes.
    *base_args: Positional arguments to be preprended to all factory calls.
    **base_kwargs: Keyword arguments to be applied to all factory calls. These
      may be overwritten by kwargs in factory calls.

  Returns:
    A factory function creating objects of class `cls`.

  Author: Joan Puigcerver (jpuigcerver@),
    modifications by Paul Rubenstein (paulrubenstein@)
  """

  def _factory_fn(*args, **kwargs):
    """A factory function that creates objects of the registered class."""
    args = base_args + args
    for k, v in base_kwargs.items():
      if k in kwargs:
        # Warning to prevent future headaches.
        logging.warning(
            "The default kwarg %r=%r, used in the lookup string %r, "
            "is overridden by the call to the resulting factory. "
            "Notice that this may lead to some unexpected behavior.", k, v,
            lookup_string)
      else:
        kwargs[k] = v

    return cls(*args, **kwargs)

  return _factory_fn


def parse_name(string_to_parse):
  """Parses input to the registry's lookup function.

  Args:
    string_to_parse: can be either an arbitrary name or function call
      (optionally with positional and keyword arguments). e.g. "multiclass",
      "resnet50_v2(filters_factor=8)".

  Returns:
    A tuple of input name, argument tuple and a keyword argument dictionary.
    Examples:
      "multiclass" -> ("multiclass", (), {})
      "resnet50_v2(9, filters_factor=4)" ->
          ("resnet50_v2", (9,), {"filters_factor": 4})

  Author: Joan Puigcerver (jpuigcerver@)
  """
  expr = ast.parse(string_to_parse, mode="eval").body  # pytype: disable=attribute-error
  if not isinstance(expr, (ast.Attribute, ast.Call, ast.Name)):
    raise ValueError(
        "The given string should be a name or a call, but a {} was parsed from "
        "the string {!r}".format(type(expr), string_to_parse))

  # Notes:
  # name="some_name" -> type(expr) = ast.Name
  # name="module.some_name" -> type(expr) = ast.Attribute
  # name="some_name()" -> type(expr) = ast.Call
  # name="module.some_name()" -> type(expr) = ast.Call

  if isinstance(expr, ast.Name):
    return string_to_parse, (), {}
  elif isinstance(expr, ast.Attribute):
    return string_to_parse, (), {}

  def _get_func_name(expr):
    if isinstance(expr, ast.Attribute):
      return _get_func_name(expr.value) + "." + expr.attr
    elif isinstance(expr, ast.Name):
      return expr.id
    else:
      raise ValueError(
          "Type {!r} is not supported in a function name, the string to parse "
          "was {!r}".format(type(expr), string_to_parse))

  def _get_func_args_and_kwargs(call):
    args = tuple([ast.literal_eval(arg) for arg in call.args])
    kwargs = {
        kwarg.arg: ast.literal_eval(kwarg.value) for kwarg in call.keywords
    }
    return args, kwargs

  func_name = _get_func_name(expr.func)
  func_args, func_kwargs = _get_func_args_and_kwargs(expr)

  return func_name, func_args, func_kwargs


class Registry(object):
  """Implements global Registry.

  Authors: Joan Puigcerver (jpuigcerver@), Alexander Kolesnikov (akolesnikov@)
  """

  _GLOBAL_REGISTRY = {}

  @staticmethod
  def global_registry():
    return Registry._GLOBAL_REGISTRY

  @staticmethod
  def register(name, item_type, replace=False):
    """Creates a function that registers its input."""

    if item_type not in ["object", "function", "factory", "class"]:
      raise ValueError("Unknown item type: %s" % item_type)

    def _register(item):
      if name in Registry.global_registry() and not replace:
        raise KeyError(
            "The name {!r} was already registered in with type {!r}".format(
                name, item_type))

      Registry.global_registry()[name] = (item, item_type)
      return item

    return _register

  @staticmethod
  def lookup(lookup_string, kwargs_extra=None):
    """Lookup a name in the registry."""

    name, args, kwargs = parse_name(lookup_string)
    if kwargs_extra:
      kwargs.update(kwargs_extra)
    item, item_type = Registry.global_registry()[name]
    if item_type == "function":
      return functools.partial(item, *args, **kwargs)
    elif item_type == "object":
      return item(*args, **kwargs)
    elif item_type == "factory":
      return partialfactory(item, lookup_string, *args, **kwargs)
    elif item_type == "class":
      return partialclass(item, *args, **kwargs)
