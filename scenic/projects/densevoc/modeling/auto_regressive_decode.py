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

"""Auto-regressive generate caption.

This is simplified from t5 decoding
https://github.com/google-research/t5x/blob/main/t5x/decoding.py.
We don't use beam search (always take the top-1) and don't use model cache.
"""
import functools

import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class State:
  """Holds beam search state data."""
  cur_index: jax.Array  # int
  predictions: jax.Array  # int array of (batch_size, max_steps)
  sum_log_prob: jax.Array  # float array of (batch_size,)


def scatter_min(inp, index, src):
  """Jax implementation of torch.scatter(inp, 1, index, src)."""
  # from https://github.com/google/jax/issues/8487
  dnums = jax.lax.ScatterDimensionNumbers(
      update_window_dims=(), inserted_window_dims=(0,),
      scatter_dims_to_operand_dims=(0,))
  scatter = functools.partial(jax.lax.scatter_min, dimension_numbers=dnums)
  scatter = jax.vmap(scatter, in_axes=(0, 0, 0), out_axes=0)
  return scatter(inp, jnp.expand_dims(index, axis=-1), src)


def auto_regressive_decode(
    begin_token, tokens_to_logits,
    max_steps=40, eos_index=102, vocab_size=30522):
  """Autoregressively generate a single caption.

  Args:
    begin_token: int array (batch_size, max_steps)
    tokens_to_logits: a function that converts input
        (batch_size, max_steps) to (batch_size, max_steps, vocab_size)
    max_steps: int
    eos_index: int. Default from bert tokenizer.
    vocab_size: int. Default from bert tokenizer.

  Returns:
    predictions: (batch_size, max_steps)
    log_prob: (batch_size,)
  """
  batch_size = begin_token.shape[0]
  logits_after_end = jnp.full(
      (batch_size, vocab_size), float("-inf"), dtype=jnp.float32)
  logits_after_end = logits_after_end.at[:, eos_index].set(0)

  def cond_fn(state: State) -> bool:
    return state.cur_index < max_steps - 1  # pytype: disable=bad-return-type  # jax-devicearray

  def body_fn(state: State) -> State:
    logits = tokens_to_logits(
        state.predictions)[:, state.cur_index - 1]  # (batch_size, vocab_size)
    # Avoid predicting repeating words following:
    #   https://github.com/JialianW/GRiT/blob/master/grit/modeling/text/
    #   text_decoder.py#L450
    last_prediction = state.predictions[:, state.cur_index - 1]  # (batch_size,)
    logits = scatter_min(
        logits, last_prediction,
        jnp.full((logits.shape[0],), -10000., dtype=jnp.float32))
    logits = jnp.where(
        jnp.broadcast_to(
            last_prediction[:, None], (batch_size, vocab_size)) == eos_index,
        logits_after_end, logits)  # (batch_size, vocab_size)
    log_prob = jax.nn.log_softmax(logits)  # (batch_size, vocab_size)
    inds = jnp.argmax(log_prob, axis=-1)  # (batch_size,)
    predictions = state.predictions.at[:, state.cur_index].set(
        inds)  # (batch_size, max_steps)
    max_log_prob = jnp.max(log_prob, axis=-1)  # (batch_size,)
    new_log_prob = state.sum_log_prob + max_log_prob  # (batch_size,)
    return State(
        cur_index=state.cur_index + 1,
        predictions=predictions,
        sum_log_prob=new_log_prob)

  init_state = State(  # pytype: disable=wrong-arg-types  # jax-devicearray
      cur_index=1,
      predictions=begin_token,
      sum_log_prob=jnp.zeros((begin_token.shape[0],), dtype=jnp.float32))
  final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
  predictions = final_state.predictions  # (batch_size, max_steps)
  sum_log_prob = final_state.sum_log_prob
  num_valid = (predictions != eos_index).sum(axis=-1) - 1  # (batch_size,)
  num_valid = jnp.maximum(num_valid, 1)
  log_probs = sum_log_prob / num_valid
  return predictions, log_probs
