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

"""Utility functions for training."""

import functools
from typing import Any, Callable, Tuple, Optional, Mapping, Union
from absl import logging

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.common_lib import debug_utils
from scenic.train_lib_deprecated import optimizers

PyTree = Union[Mapping[str, Mapping], Any]
PRNGKey = jnp.ndarray


def compute_flops(flax_model_apply_fn: Callable[[jnp.ndarray], Any],
                  input_spec: Mapping[str, Tuple[Tuple[int, ...], jnp.dtype]],
                  fuse_multiply_add: bool) -> float:
  """Performs static analysis of the graph to compute theoretical FLOPs.

  This function is branched from scenic/common_lib/debug_utils.py.
  The difference is that here the input_spec is a dictionary while it is a
  sequence in the original implementation.

  Args:
    flax_model_apply_fn: Apply function of the flax model to be analysed.
    input_spec: An mapping of modality names to (shape, dtype) pairs specifying
      the shape and dtype of the inputs.
    fuse_multiply_add: Bool; If true, count a multiply and add (also known as
      "multiply-accumulate" or "MAC") as 1 FLOP rather than 2 (as done by the
      HLO analysis). This is commonly used in literature.

  Returns:
    flops: The total number of flops.
  """
  input_placeholder = {}
  for modality, spec in input_spec.items():
    in_st = debug_utils.input_spec_to_jax_shape_dtype_struct(spec, batch_size=1)
    input_placeholder[modality] = jnp.zeros(in_st.shape, in_st.dtype)

  analysis = (
      jax.jit(flax_model_apply_fn).lower(input_placeholder).cost_analysis()
  )

  flops = analysis['flops']
  if fuse_multiply_add:
    flops = flops / 2
  logging.info('GFLOPs %0.3f for input spec: %s', flops / 10**9, input_spec)
  return flops


def initialize_model(
    *,
    model_def: nn.Module,
    input_spec: Mapping[str, Tuple[Tuple[int, ...], jnp.dtype]],
    config: ml_collections.ConfigDict,
    rngs: Union[jnp.ndarray, Mapping[str, jnp.ndarray]],
) -> Tuple[PyTree, PyTree, int, Optional[float]]:
  """Initializes parameters and model state.

  This function is branched from scenic/train_lib_deprecated/train_utils.py.
  The difference is that here the input_spec is a dictionary while it is a
  sequence in the original implementation.

  Args:
    model_def: Definition of a model.
    input_spec: An mapping of modality name to (shape, dtype) pairs specifying
      the shape and dtype of the inputs.
    config: Configurations of the initialization.
    rngs: Jax rng keys.

  Returns:
    Initial params, initial model_state, and number of trainable_params.
  """
  batch_size = (config.batch_size //
                jax.device_count()) if config.get('batch_size') else None
  input_placeholder = {}
  for modality_name, spec in input_spec.items():
    in_st = debug_utils.input_spec_to_jax_shape_dtype_struct(
        spec, batch_size=batch_size)
    input_placeholder[modality_name] = jnp.zeros(in_st.shape, in_st.dtype)

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def _initialize_model(rngs):
    """Initialization function to be jitted."""
    init_model_state, init_params = flax.core.pop(
        model_def.init(rngs, input_placeholder, train=False, debug=False),
        'params',
    )
    # Set bias in the head to low value, such that loss is small initially.
    if config.get('init_head_bias', None) is not None:
      init_params = flax.core.unfreeze(init_params)
      init_params['output_projection'] = optimizers.tree_map_with_names(
          lambda p: jnp.full_like(p, config.init_head_bias),
          init_params['output_projection'],
          match_name_fn=lambda name: 'bias' in name)
      init_params = flax.core.freeze(init_params)
    return init_params, init_model_state

  if not isinstance(rngs, dict):
    rngs = {'params': rngs}
  init_params, init_model_state = _initialize_model(rngs)
  # Pop out params rng:
  rngs.pop('params')

  # Count number of trainable parameters:
  num_trainable_params = debug_utils.log_param_shapes(init_params)

  # Count gflops:
  count_flops = config.get('count_flops',
                           ml_collections.ConfigDict({'count_flops': True}))
  if count_flops:
    variables = {'params': init_params, **init_model_state}
    flops = compute_flops(
        flax_model_apply_fn=functools.partial(
            model_def.apply, variables, train=False, debug=False, rngs=rngs),
        input_spec=count_flops.get('input_spec', input_spec),
        fuse_multiply_add=count_flops.get('fuse_multiply_add', True))
    gflops = flops / (10**9)
  else:
    gflops = None

  return init_params, init_model_state, num_trainable_params, gflops
