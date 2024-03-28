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


class PonderStep(nn.Module):
  """Takes an ACT step."""
  ac_config: ml_collections.ConfigDict
  layer: nn.Module

  @nn.compact
  def __call__(self, carry_inputs: Any) -> Any:
    """An pondernet act step.

    Args:
      carry_inputs: A tuple of: - state: An array of shape `[batch_size, length,
        channel]`. - unhalting_probability: An array containing the unhalting
        probabilities. - halted: An array containing halted flag of each token.
        - layer_id: An int denotes the layer id - layer_call_args: Arguments to
        be passed to the self.layer.

    Returns:
      A tupe of (output_state, unhalting_probability, halted).
    """
    halting_bias_init = self.ac_config.act_halting_bias_init

    # Unpack the inputs.
    (output_state, unhalting_probability, halted, layer_id, _, current_state,
     all_states, all_p, n_updates, *layer_call_args) = carry_inputs

    p = nn.sigmoid(
        nn.Dense(
            features=1,
            use_bias=True,
            kernel_init=nn.initializers.zeros,
            bias_init=lambda k, s, *_: jnp.full(s, halting_bias_init),
            dtype=jnp.float32,
            name='step_halting_prob')(current_state))

    # Average over all tokens:
    p = jnp.mean(p, axis=1)
    p = jnp.squeeze(p, axis=-1)

    # Update p if this is the last layer.
    # if layer_id == self.act_config.act_max_steps - 1:
    #  p = 1 - jnp.sum(all_p, axis=0)
    p = jax.lax.cond(layer_id == self.ac_config.act_max_steps - 1,
                     lambda: 1 - jnp.sum(all_p, axis=0), lambda: p)

    # Init the prob of halting here.
    prob_halt_here = unhalting_probability * p

    # Update unhalting_probability according to the new p.
    unhalting_probability = unhalting_probability * (1 - p)

    # Init the halting decision by sampling from bernoulli distribution.
    rng = self.make_rng('ponder')
    halt_decision = (1 - halted) * jax.random.bernoulli(rng, p=p, shape=p.shape)

    # Apply the layer on the state. And update output_state by the halted mask.
    new_state = self.layer(current_state, *layer_call_args)

    update_halted_decision = jnp.expand_dims(halt_decision, -1)
    update_halted_decision = jnp.expand_dims(update_halted_decision, -1)
    output_state = output_state + new_state * update_halted_decision

    # Update the all states and all p.
    all_states = all_states.at[layer_id].set(all_states[layer_id] + new_state)
    all_p = all_p.at[layer_id].set(all_p[layer_id] + prob_halt_here)

    # Update n_updates.
    n_updates += (1 - halted)

    # Update the halted.
    halted = halted + halt_decision

    return (output_state, unhalting_probability, halted, layer_id + 1,
            prob_halt_here, new_state, all_states, all_p, n_updates,
            *layer_call_args)


class PonderFunction(nn.Module):
  """Adaptive Computation Time Function to help we use nn.scan on ACT."""
  ac_config: ml_collections.ConfigDict
  layer: nn.Module
  stop_fn: Any
  deterministic: bool

  def setup(self):
    self.ponder_step = PonderStep(
        ac_config=self.ac_config, layer=self.layer, name='ponder_step')

  def take_a_step(self, x) -> Any:
    return self.ponder_step(x)

  def skip_a_step(self, x) -> Any:  # Shunt
    return x

  @nn.compact
  def __call__(self, x, _) -> Any:
    # We only consider take_a_step here, since for PonderNet, the skip only
    # happens during inference.
    if self.is_mutable_collection('params'):  # Init-mode
      out = self.take_a_step(x)
    else:
      decision = self.stop_fn(x) * self.deterministic
      out = nn.cond(decision, self.skip_a_step, self.take_a_step, self, x)
    return out, None


class AdaptiveComputationTime(nn.Module):
  """Adaptive Computation Time module, based on: arxiv.org/abs/1807.03819."""

  ac_config: ml_collections.ConfigDict
  layer: nn.Module
  share_parameters: bool

  @nn.compact
  def __call__(self, x: jnp.ndarray, *layer_call_args):

    max_steps = self.ac_config.act_max_steps
    deterministic = layer_call_args[0]

    state = x
    original_state_shape = state.shape

    state_slice = slice(0, 1)

    # Dynamic shape for update tensors below.
    update_shape = state.shape[state_slice]
    # Unhalting probabilities.
    unhalting_probability = jnp.ones(update_shape)
    # Halted mask.
    halted = jnp.zeros(update_shape)
    # All states from different steps.
    all_states = jnp.zeros((max_steps,) + original_state_shape)
    # All p from different steps.
    all_p = jnp.zeros((max_steps,) + update_shape)
    # Count how many updates we did, we use this to log and debug
    n_updates = jnp.zeros(update_shape)

    # Define one stop function to decide the routing result.
    def stop_fn(inputs: Any) -> jnp.ndarray:
      # Returns True if all of halting probability >= 1-eps.
      _, _, halted, _, _, _, *_ = inputs
      return jnp.all(halted)

    # Create one empty_probability with unhalting_probability
    empty_probability = jnp.zeros_like(unhalting_probability)
    # empty_state = jnp.zeros_like(state)

    # Run max_steps, for each sample/token, when the decision is True,
    # go to the shunt_layer.
    scan_carry_input = (state, unhalting_probability, halted, 0,
                        empty_probability, state, all_states, all_p, n_updates,
                        *layer_call_args)

    if self.share_parameters:
      # Scan over `PonderFunction` while broadcasing (sharing) the params.
      act_fn = nn.scan(
          PonderFunction,
          variable_broadcast='params',
          split_rngs={
              'params': False,
              'dropout': True,
              'ponder': True
          },
          length=max_steps)
    else:
      # Scan over PonderFunction while only broadcasing the shared param.
      def trans_in_fn(target):
        return {
            'params':
                dict(
                    target.get('params', {}), **target.get('shared_params', {}))
        }

      def trans_out_fn(target):
        params = target.get('params', {})
        shared_params = {}
        if 'ponder_step' in params:
          shared_params['ponder_step'] = params.pop('ponder_step')
        return {'params': params, 'shared_params': shared_params}

      act_fn_two_collections = nn.scan(
          # Map params to a two collections.
          nn.map_variables(
              PonderFunction, ['params', 'shared_params'],
              trans_in_fn=trans_in_fn,
              trans_out_fn=trans_out_fn,
              mutable=True),
          variable_broadcast='shared_params',
          variable_axes={'params': 0},
          split_rngs={
              'params': False,
              'dropout': True,
              'ponder': True
          },
          length=max_steps)

      # Map all params back to a single collection.
      act_fn = nn.map_variables(
          act_fn_two_collections, ['params', 'shared_params'],
          trans_in_fn=trans_out_fn,
          trans_out_fn=trans_in_fn,
          mutable=True)

    scan_carry_output, _ = act_fn(
        self.ac_config,
        self.layer,
        stop_fn,
        deterministic,
    )(scan_carry_input, None)

    (final_state, unhalting_probability, halted, _, _, _, all_states, all_p,
     n_updates, *layer_call_args) = scan_carry_output

    # Check some shapes
    assert final_state.shape == original_state_shape
    for x in [unhalting_probability, halted]:
      assert x.shape == original_state_shape[state_slice]
    return final_state, (all_states, all_p, n_updates)
