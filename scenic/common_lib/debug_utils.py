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

"""Utilities for logging, debugging, profiling, testing, and visualization."""

from collections import abc
from concurrent import futures
import json
import operator
import threading
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


def log_param_shapes(
    params: Any,
    print_params_nested_dict: bool = False,
    description: Optional[str] = None,
    include_stats: bool = True,
) -> int:
  """Prints out shape of parameters and total number of trainable parameters.

  Args:
    params: PyTree of model parameters.
    print_params_nested_dict: If True, it prints parameters in shape of a nested
      dict.
    description: Optional description to print out before logging the parameter
      summary.
    include_stats: Include parameter stats if True.

  Returns:
    int; Total number of trainable parameters.
  """
  if print_params_nested_dict:
    shape_dict = tree_map(lambda x: str(x.shape), params)
    # We use json.dumps for pretty printing nested dicts.
    logging.info(
        'Printing model param shape:/n%s',
        json.dumps(shape_dict, sort_keys=True, indent=4),
    )
  parameter_overview.log_parameter_overview(
      params, include_stats=include_stats, msg=description
  )
  total_params = jax.tree_util.tree_reduce(
      operator.add, tree_map(lambda x: x.size, params)
  )
  logging.info('Total params: %d', total_params)
  return total_params


def input_spec_to_jax_shape_dtype_struct(
    spec: Union[Tuple[Tuple[int, ...], jnp.dtype], Tuple[int, ...]],
    batch_size: Optional[int] = None,
) -> jax.ShapeDtypeStruct:
  """Parse an input specs into a jax.ShapeDtypeStruct."""
  spec = tuple(spec)
  if batch_size and len(spec) == 1:
    raise ValueError('batch_size unsupported when len(spec) is 1.')
  if len(spec) == 2 and isinstance(spec[0], abc.Iterable):
    shape = (batch_size,) + tuple(spec[0][1:]) if batch_size else spec[0]
    dtype = spec[1]
  else:
    shape = (batch_size,) + tuple(spec[1:]) if batch_size else spec
    dtype = jnp.float32
  return jax.ShapeDtypeStruct(shape, dtype)


def compute_flops(
    flax_model_apply_fn: Callable[[jnp.ndarray], Any],
    input_spec: Sequence[
        Union[Tuple[Tuple[int, ...], jnp.dtype], Tuple[int, ...], None]
    ],
    fuse_multiply_add: bool,
) -> float:
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

  analysis = jax.jit(flax_model_apply_fn).lower(*dummy_input).cost_analysis()
  flops = analysis['flops']
  if fuse_multiply_add:
    flops = flops / 2
  logging.info('GFLOPs %0.3f for input spec: %s', flops / 10**9, input_spec)
  return flops


def compute_flops_with_pytree(
    flax_model_apply_fn: Callable[[jnp.ndarray], Any],
    input_spec: PyTree,
    unpack_input: bool = True,
    fuse_multiply_add: bool = True,
) -> float:
  """Performs static analysis of the graph to compute theoretical FLOPs.

  One can also use the XProf profiler to get the actual FLOPs at runtime
  based on device counters. Theoretical FLOPs are more useful for comparing
  models across different library implementations and is hardware-agnostic.

  Args:
    flax_model_apply_fn: Apply function of the flax model to be analysed.
    input_spec: A PyTree whose leaves are (shape, dtype) pairs specifying the
      shape and dtype of the inputs. If unspecified the dtype is float32.
    unpack_input: Unpack the pytree when feeding it to the model.
    fuse_multiply_add: Bool; If true, count a multiply and add (also known as
      "multiply-accumulate" or "MAC") as 1 FLOP rather than 2 (as done by the
      HLO analysis). This is commonly used in literature.

  Returns:
    flops: The total number of flops.
  """

  def check_leaf_spec(spec: Sequence[PyTree]) -> bool:
    return (
        len(spec) == 2
        and isinstance(spec[0], abc.Sequence)
        and all(isinstance(i, int) for i in spec[0])
        and isinstance(spec[1], jnp.dtype)
    ) or (all(isinstance(i, int) for i in spec[0]))

  def create_dummy_input(spec: PyTree) -> PyTree:
    if isinstance(spec, dict):
      return {k: create_dummy_input(v) for k, v in spec.items()}
    elif isinstance(spec, abc.Sequence):
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

  if isinstance(dummy_input, dict) and unpack_input:
    analysis = jax.jit(flax_model_apply_fn).lower(**dummy_input).cost_analysis()
  elif isinstance(dummy_input, abc.Sequence) and unpack_input:
    analysis = jax.jit(flax_model_apply_fn).lower(*dummy_input).cost_analysis()
  else:
    analysis = jax.jit(flax_model_apply_fn).lower(dummy_input).cost_analysis()

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


class DummyExecutor(futures.Executor):
  """A mock executor that operates serially.

  Useful for debugging.

  Example usage:

  # Runs concurrently, difficult to debug:
  pool = futures.ThreadPoolExecutor(max_workers=max_workers)
  pool.submit(my_function)

  # For debugging:
  pool = DummyExecutor()
  pool.submit(my_function)  # Will block and run serially.
  """

  def __init__(self):
    self._shutdown = False
    self._shutdown_lock = threading.Lock()

  def submit(self, fn: Callable[..., Any], *args, **kwargs) -> futures.Future:  # pylint: disable=g-bare-generic
    with self._shutdown_lock:
      if self._shutdown:
        raise RuntimeError('Cannot schedule new futures after shutdown.')

      future = futures.Future()
      try:
        result = fn(*args, **kwargs)
      except BaseException as e:  # pylint: disable=broad-except
        future.set_exception(e)
      else:
        future.set_result(result)
      return future

  def shutdown(self, wait: bool = True):  # pytype: disable=signature-mismatch  # overriding-parameter-name-checks
    with self._shutdown_lock:
      self._shutdown = True


class StepTraceContextHelper:
  """Helper class to use jax.profiler.StepTraceAnnotation.

  This will cause a "name" event to show up on the trace timeline if the
  event occurs while the process is being traced by TensorBoard. In addition,
  if using accelerators, the device trace timeline will also show a "name"
  event. Note that "step_num" can be set as a keyword argument to pass the
  global step number to the profiler. See jax.profiler.StepTraceAnnotation.

  """

  def __init__(self, name: str, init_step_num: int):
    self.name = name
    self.step_num = init_step_num
    self.context = None

  def __enter__(self):
    self.context = jax.profiler.StepTraceAnnotation(
        self.name, step_num=self.step_num
    )
    self.step_num += 1
    self.context.__enter__()
    return self

  def __exit__(self, exc_type, exc_value, tb):
    assert self.context is not None, 'Exited context without entering.'
    self.context.__exit__(exc_type, exc_value, tb)
    self.context = None

  def next_step(self):
    if self.context is None:
      raise ValueError('Must call next_step() within a context.')
    self.__exit__(None, None, None)
    self.__enter__()
