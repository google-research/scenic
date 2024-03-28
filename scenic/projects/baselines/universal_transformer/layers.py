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

"""Adaptive Computation Time layers."""

from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections


class Identity(nn.Module):
  """Identity layer (used for shunting)."""

  @nn.compact
  def __call__(self, *args):
    # Inputs and outputs must maintain same tree structure.
    return args[0] if len(args) == 1 else args


class ActStep(nn.Module):
  """Takes an ACT step."""
  ac_config: ml_collections.ConfigDict
  layer: nn.Module

  @nn.compact
  def __call__(self, inputs: Any) -> Any:
    """An act step (which either adaptivaly appies the layer or skips).

    Args:
      inputs: A tuple of: - state: An array of shape `[batch_size, length,
        channel]`. - halting_probability: An array containing the halting probs.
        - remainders: An array containing the act remainders. - n_updates: An
        array containing the act n_updates. - previous_state: An array that has
        the previous state. - layer_call_args: Arguments to be passed to the
        self.layer.

    Returns:
      A tupe of (output_state, new halting_probabilities,
        updated remainders, updated n_updates, new_state).
    """
    threshold = 1.0 - self.ac_config.act_epsilon
    act_type = self.ac_config.act_type
    halting_bias_init = self.ac_config.act_halting_bias_init
    act_level = self.ac_config.act_level

    (state, halting_probability, remainders, n_updates, previous_state,
     *layer_call_args) = inputs
    if act_type == 'random':
      # Random as halting probability, to be used as a baseline.
      rng = jax.random.PRNGKey(0)  # bind rng to step?
      # TODO(dehghani): currently, it gives the error of:
      # ScanACTFunction_0 needs PRNG for "dropout"!
      p = jax.random.uniform(rng, shape=halting_probability.shape)

    else:
      p = nn.sigmoid(
          nn.Dense(
              features=1,
              use_bias=True,
              kernel_init=nn.initializers.zeros,
              bias_init=lambda k, s, *_: jnp.full(s, halting_bias_init),
              dtype=jnp.float32,
              name='step_halting_prob')(state))

      if act_level == 'per_example':
        # Average over all tokens:
        p = jnp.mean(p, axis=1)
      p = jnp.squeeze(p, axis=-1)

    # Create a mask for inputs which have not halted yet.
    still_running = jnp.less(halting_probability, 1.0).astype(jnp.float32)

    # Create a mask for inputs which halted at this step.
    new_halted = jnp.greater(halting_probability + p * still_running,
                             threshold).astype(jnp.float32) * still_running

    # Crerate mask of inputs which haven't halted and didn't halt this step.
    still_running = jnp.less_equal(halting_probability + p * still_running,
                                   threshold).astype(
                                       jnp.float32) * still_running

    # Add the halting probability for this step to the halting
    # probabilities for those inputs which haven't halted yet.
    halting_probability += p * still_running

    # Compute remainders for the inputs which halted at this step.
    remainders += new_halted * (1 - halting_probability)

    # Add the remainders to those inputs which halted at this step.
    halting_probability += new_halted * remainders

    # Increment n_updates for all inputs which are still running.
    n_updates += still_running + new_halted

    # Compute the weight to be applied to the new state and output:
    # 0: when the input has already halted.
    # p: when the input hasn't halted yet.
    # remainders: when it halted this step.
    update_weights = jnp.expand_dims(
        p * still_running + new_halted * remainders, -1)
    if act_level == 'per_example':
      update_weights = jnp.expand_dims(update_weights, -1)

    # Apply the layer on the state.
    output_state = self.layer(state, *layer_call_args)

    if act_type in ['basic', 'random']:
      # Update running part in the weighted state and keep the rest
      new_state = ((output_state * update_weights) + (previous_state *
                                                      (1 - update_weights)))
    elif act_type == 'accumulated':
      # Add in the weighted state.
      new_state = (output_state * update_weights) + previous_state
    else:
      raise ValueError(f'Unknown act_type {act_type}!')

    return (output_state, halting_probability, remainders, n_updates, new_state,
            *layer_call_args)


class ACTFunction(nn.Module):
  """Adaptive Computation Time Function to help we use nn.scan on ACT."""
  ac_config: ml_collections.ConfigDict
  layer: nn.Module
  stop_fn: Any

  def setup(self):
    self.act_step = ActStep(
        ac_config=self.ac_config, layer=self.layer, name='act_step')

  def take_a_step(self, x) -> Any:
    return self.act_step(x)

  def skip_a_step(self, x) -> Any:  # Shunt
    return x

  @nn.compact
  def __call__(self, x, _) -> Any:
    if self.is_mutable_collection('params'):  # Init-mode
      out = self.take_a_step(x)
    else:
      decision = self.stop_fn(x)
      out = nn.cond(decision, self.skip_a_step, self.take_a_step, self, x)
    return out, None


class AdaptiveComputationTime(nn.Module):
  """Adaptive Computation Time module, based on: arxiv.org/abs/1807.03819."""

  ac_config: ml_collections.ConfigDict
  layer: nn.Module
  share_parameters: bool

  @nn.compact
  def __call__(self, x: jnp.ndarray, *layer_call_args):

    threshold = 1.0 - self.ac_config.act_epsilon
    max_steps = self.ac_config.act_max_steps

    state = x
    original_state_shape = state.shape

    if self.ac_config.act_level == 'per_example':
      state_slice = slice(0, 1)
    elif self.ac_config.act_level == 'per_token':
      state_slice = slice(0, 2)
    else:
      raise ValueError(f'Unknown act_level {self.ac_config.act_level}')

    # Dynamic shape for update tensors below.
    update_shape = state.shape[state_slice]
    # Halting probabilities (p_t^n in the paper).
    halting_probability = jnp.zeros(update_shape)
    # Remainders (R(t) in the paper).
    remainders = jnp.zeros(update_shape)
    # Number of updates performed (N(t) in the paper).
    n_updates = jnp.zeros(update_shape)
    # Previous cell states (s_t in the paper).
    previous_state = jnp.zeros_like(state)

    # Define one stop function to decide the routing result.
    def stop_fn(inputs: Any) -> jnp.ndarray:
      # Returns True if all of halting probability >= 1-eps.
      _, halting_probability, _, _, _, *_ = inputs
      return jnp.all(jnp.greater_equal(halting_probability, threshold))

    # Run max_steps, for each sample/token, when the decision is True,
    # go to the shunt_layer.
    intermedia_output = (state, halting_probability, remainders, n_updates,
                         previous_state, *layer_call_args)

    if self.share_parameters:
      # Scan over `ACTFunction` while broadcasing (sharing) the params.
      act_fn = nn.scan(
          ACTFunction,
          variable_broadcast='params',
          split_rngs={
              'params': False,
              'dropout': True
          },
          length=max_steps)

    else:
      # When we want to have different parameters for different layers, if we
      # use simple nn.scan and set "variable_broadcast=None", we also get
      # different parameters for the haulting mechaniems (the dense layer the
      # predicts the halting probs) for different layers, however we want to
      # have the same mdoule make the halting decision across all layers.
      # To do that we need to map variables to two collections: `params` and
      # `shared_params` before sending it to the scan, then set
      # variable_broadcast='shared_params', and then map them back to a single
      # collection.
      def trans_in_fn(target):
        return {
            'params':
                dict(
                    target.get('params', {}), **target.get('shared_params', {}))
        }

      def trans_out_fn(target):
        params = target.get('params', {})
        shared_params = {}
        if 'act_step' in params:
          shared_params['act_step'] = params.pop('act_step')
        return {'params': params, 'shared_params': shared_params}

      act_fn_two_collections = nn.scan(
          # Map  params to a two collections.
          nn.map_variables(
              ACTFunction, ['params', 'shared_params'],
              trans_in_fn=trans_in_fn,
              trans_out_fn=trans_out_fn,
              mutable=True),
          variable_broadcast='shared_params',
          variable_axes={'params': 0},
          split_rngs={
              'params': False,
              'dropout': True
          },
          length=max_steps)

      # Map all params back to a single collection.
      act_fn = nn.map_variables(
          act_fn_two_collections, ['params', 'shared_params'],
          trans_in_fn=trans_out_fn,
          trans_out_fn=trans_in_fn,
          mutable=True)

    output, _ = act_fn(self.ac_config, self.layer, stop_fn)(intermedia_output,
                                                            None)

    (output_state, halting_probability, remainders, ponder_times, new_state,
     *layer_call_args) = output

    # Check some shapes
    assert output_state.shape == new_state.shape == original_state_shape
    for x in [halting_probability, remainders, n_updates]:
      assert x.shape == original_state_shape[state_slice]
    return new_state, (ponder_times, remainders)
