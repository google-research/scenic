# Copyright 2022 The Scenic Authors.
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

"""Utilities for logging, debugging, profiling, testing, and visualization."""

import collections
import json
import operator
from typing import Any, Callable, Optional, Sequence, Set, Tuple, Union

from absl import logging
from clu import parameter_overview
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import ml_collections

PyTree = Any


def enable_jax_debugging_flags():
  """Enables some of the global JAX flags for debugging."""

  # Enable the NaN-checker behavior to cause JAX to hard-break on the first
  # occurrence of a NaN.
  jax.config.update('jax_debug_nans', True)

  # Enable the compilation logger to check whether or not we're accidentally
  # causing a lot of re-compilation (inspect logs for excessive jitting).
  jax.config.update('jax_log_compiles', True)

  # Detect numpy-style automatic rank promotion and force strict, explicit
  # casts. We can use `raise` instead of warn to raise an error.
  jax.config.update('jax_numpy_rank_promotion', 'warn')

  # Print global JAX flags in logs.
  logging.info('Global JAX flags: %s', jax.config.values)


def log_param_shapes(params: Any,
                     print_params_nested_dict: bool = False) -> int:
  """Prints out shape of parameters and total number of trainable parameters.

  Args:
    params: PyTree of model parameters.
    print_params_nested_dict: If True, it prints parameters in shape of a nested
      dict.

  Returns:
    int; Total number of trainable parameters.
  """
  if print_params_nested_dict:
    shape_dict = tree_map(lambda x: str(x.shape), params)
    # We use json.dumps for pretty printing nested dicts.
    logging.info('Printing model param shape:/n%s',
                 json.dumps(shape_dict, sort_keys=True, indent=4))
  parameter_overview.log_parameter_overview(params)
  total_params = jax.tree_util.tree_reduce(operator.add,
                                           tree_map(lambda x: x.size, params))
  logging.info('Total params: %d', total_params)
  return total_params


def input_spec_to_jax_shape_dtype_struct(
    spec: Union[Tuple[Tuple[int, ...], jnp.dtype], Tuple[int, ...]],
    batch_size: Optional[int] = None) -> jax.ShapeDtypeStruct:
  """Parse an input specs into a jax.ShapeDtypeStruct."""
  spec = tuple(spec)
  if len(spec) == 2 and isinstance(spec[0], collections.abc.Iterable):
    shape = (batch_size,) + tuple(spec[0][1:]) if batch_size else spec[0]
    dtype = spec[1]
  else:
    shape = (batch_size,) + tuple(spec[1:]) if batch_size else spec
    dtype = jnp.float32
  return jax.ShapeDtypeStruct(shape, dtype)


def compute_flops(flax_model_apply_fn: Callable[[jnp.ndarray], Any],
                  input_spec: Sequence[Union[Tuple[Tuple[int, ...], jnp.dtype],
                                             Tuple[int, ...], None]],
                  fuse_multiply_add: bool) -> float:
  """Performs static analysis of the graph to compute theoretical FLOPs.

  One can also use the XProf profiler to get the actual FLOPs at runtime
  based on device counters. Theoretical FLOPs are more useful for comparing
  models across different library implementations and is hardware-agnostic.

  Args:
    flax_model_apply_fn: Apply function of the flax model to be analysed.
    input_spec: An iterable of (shape, dtype) pairs specifying the shape and
      dtype of the inputs. If unspecified the dtype is float32.
    fuse_multiply_add: Bool; If true, count a multiply and add (also known as
      "multiply-accumulate" or "MAC") as 1 FLOP rather than 2 (as done by the
      HLO analysis). This is commonly used in literature.

  Returns:
    flops: The total number of flops.
  """
  dummy_input = []
  for spec in input_spec:
    if spec is not None:
      in_st = input_spec_to_jax_shape_dtype_struct(spec, batch_size=1)
      dummy_input.append(jnp.zeros(in_st.shape, in_st.dtype))
    else:
      dummy_input.append(None)

  m = jax.xla_computation(flax_model_apply_fn)(*dummy_input).as_hlo_module()
  client = jax.lib.xla_bridge.get_backend()
  analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, m)  # pylint: disable=protected-access

  flops = analysis['flops']
  if fuse_multiply_add:
    flops = flops / 2
  logging.info('GFLOPs %0.3f for input spec: %s', flops / 10**9, input_spec)
  return flops


def compute_flops_with_pytree(flax_model_apply_fn: Callable[[jnp.ndarray], Any],
                              input_spec: PyTree,
                              fuse_multiply_add: bool) -> float:
  """Performs static analysis of the graph to compute theoretical FLOPs.

  One can also use the XProf profiler to get the actual FLOPs at runtime
  based on device counters. Theoretical FLOPs are more useful for comparing
  models across different library implementations and is hardware-agnostic.

  Args:
    flax_model_apply_fn: Apply function of the flax model to be analysed.
    input_spec: A PyTree whose leaves are (shape, dtype) pairs specifying the
      shape and dtype of the inputs. If unspecified the dtype is float32.
    fuse_multiply_add: Bool; If true, count a multiply and add (also known as
      "multiply-accumulate" or "MAC") as 1 FLOP rather than 2 (as done by the
      HLO analysis). This is commonly used in literature.

  Returns:
    flops: The total number of flops.
  """

  def check_leaf_spec(spec: Sequence[PyTree]) -> bool:
    return ((len(spec) == 2 and isinstance(spec[0], collections.Sequence) and
             all(isinstance(i, int) for i in spec[0]) and
             isinstance(spec[1], jnp.dtype)) or
            (all(isinstance(i, int) for i in spec[0])))

  def create_dummy_input(spec: PyTree) -> PyTree:
    if isinstance(spec, dict):
      return {k: create_dummy_input(v) for k, v in spec.items()}
    elif isinstance(spec, collections.Sequence):
      if check_leaf_spec(spec):
        in_st = input_spec_to_jax_shape_dtype_struct(spec, batch_size=1)
        return jnp.zeros(in_st.shape, in_st.dtype)
      else:
        return tuple(create_dummy_input(child) for child in spec)
    elif spec is None:
      return None
    else:
      raise NotImplementedError('Unsupported spec type.', type(spec))

  dummy_input = create_dummy_input(input_spec)

  m = jax.xla_computation(flax_model_apply_fn)(*dummy_input).as_hlo_module()
  client = jax.lib.xla_bridge.get_backend()
  analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, m)  # pylint: disable=protected-access

  flops = analysis['flops']
  if fuse_multiply_add:
    flops = flops / 2
  logging.info('GFLOPs %0.3f for input spec: %s', flops / 10**9, input_spec)
  return flops


class ConfigDictWithAccessRecord(ml_collections.ConfigDict):
  """A wrapper for ConfigDicts that records access of any config field.

  ConfigDictWithAccessRecord behaves like a standard ConfigDict, except that it
  records access to any config field (including nested instances of
  ConfigDictWithAccessRecord). This allows testing for unused config fields.

  Example usage:

    def test_config_access(self):
      with mock.patch('configs.my_config.ml_collections.ConfigDict',
                      test_utils.ConfigDictWithAccessRecord):
        config = config_module.get_config()
      config.reset_access_record()  # Resets previous access records.
      ...  # Code that uses config.
      self.assertEmpty(config.get_not_accessed())
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.reset_access_record()

  def __getitem__(self, key: str):
    self._access_record.add(key)
    return super().__getitem__(key)

  def reset_access_record(self):
    """Resets the record of config field accesses."""
    for value in self._fields.values():
      if isinstance(value, type(self)):
        value.reset_access_record()
    # object.__setattr__ avoids triggering ConfigDict's __getattr__:
    object.__setattr__(self, '_access_record', set())

  def get_not_accessed(self, prefix: str = 'config') -> Set[str]:
    """Returns the set of fields that were not accessed since the last reset."""
    not_accessed = set()
    for key, value in self._fields.items():
      path = f'{prefix}.{key}'
      if isinstance(value, type(self)):
        not_accessed |= value.get_not_accessed(prefix=path)
      else:
        if key not in self._access_record and key != '_access_record':
          not_accessed.add(path)
    return not_accessed
