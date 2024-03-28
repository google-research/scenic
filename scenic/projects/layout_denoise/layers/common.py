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

"""Common utils."""
import functools
import flax.linen as nn
import jax
from jax.nn import initializers
import jax.numpy as jnp
import numpy as np

pytorch_kernel_init = functools.partial(initializers.variance_scaling,
                                        1. / 3., 'fan_in', 'uniform')


def uniform_initializer(minval, maxval, dtype=jnp.float32):
  def init(key, shape, dtype=dtype):
    return jax.random.uniform(key, shape, dtype, minval=minval, maxval=maxval)
  return init


def dense(inputs, output_dim, dtype, kernel_init=None):
  bias_range = 1. / np.sqrt(inputs.shape[-1])
  if kernel_init is None:
    kernel_init = pytorch_kernel_init(dtype=dtype)
  return nn.Dense(
      output_dim,
      kernel_init=kernel_init,
      bias_init=uniform_initializer(
          -bias_range, bias_range, dtype),
      dtype=dtype)(inputs)


def create_output(output_model, params, aux_loss=False, layout_model_pamp=None):
  """Creates the output dict."""
  output = {}
  multimodal_outputs = params['multimodal_outputs']

  if not aux_loss:
    output.update(output_model(params))
    return output

  # Currently only layout has intermediate losses
  layout_model_pamp_partial = functools.partial(
      layout_model_pamp, train=params['train'])
  pred_dict = jax.vmap(layout_model_pamp_partial)(multimodal_outputs)
  for key in pred_dict:
    output[key] = pred_dict[key][-1]

  # Append intermediate layer logits.
  output['aux_outputs'] = []
  num_layers = multimodal_outputs.shape[0]
  for layer in range(num_layers - 1):
    lgt_dict = {}
    for key in pred_dict:
      logts = pred_dict[key][layer]
      lgt_dict.update({key: logts})
    output['aux_outputs'].append(lgt_dict)
  return output
