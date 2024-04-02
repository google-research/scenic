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
https://github.com/google-research/t5x/blob/main/t5x/decoding.py

We don't use model cache for now.
"""
import functools

from typing import Any, Callable, Tuple
import flax
import jax
from jax import lax
import jax.numpy as jnp

NEG_INF = -1.0e7
PyTreeDef = Any
Array = jax.Array


@flax.struct.dataclass
class State:
  """Holds beam search state data."""
  cur_index: int
  predictions: Array  # int array of (batch_size, max_steps)
  sum_log_prob: Array  # float array of (batch_size,)


def scatter_min(inp, index, src):
  """Jax implementation of torch.scatter(inp, 1, index, src)."""
  # from https://github.com/google/jax/issues/8487
  dnums = jax.lax.ScatterDimensionNumbers(
      update_window_dims=(), inserted_window_dims=(0,),
      scatter_dims_to_operand_dims=(0,))
  scatter = functools.partial(jax.lax.scatter_min, dimension_numbers=dnums)
  scatter = jax.vmap(scatter, in_axes=(0, 0, 0), out_axes=0)
  return scatter(inp, jnp.expand_dims(index, axis=-1), src)


def greedy_decode(
    begin_token, tokens_to_logits,
    max_steps=40, eos_index=102, vocab_size=30522, **kwargs):
  """Autoregressively generate a single caption.

  Args:
    begin_token: int array (batch_size, max_steps)
    tokens_to_logits: a function that converts input
        (batch_size, max_steps) to (batch_size, max_steps, vocab_size)
    max_steps: int
    eos_index: int
    vocab_size: int
    **kwargs: args for other decoder

  Returns:
    predictions: (batch_size, max_steps)
    log_prob: (batch_size,)
  """
  del kwargs
  batch_size = begin_token.shape[0]
  logits_after_end = jnp.full(
      (batch_size, vocab_size), float('-inf'), dtype=jnp.float32)
  logits_after_end = logits_after_end.at[:, eos_index].set(0)

  def cond_fn(state: State) -> bool:
    return state.cur_index < max_steps - 1

  def body_fn(state: State) -> State:
    logits = tokens_to_logits(
        state.predictions)[:, state.cur_index - 1]  # (batch_size, vocab_size)
    # Avoid predicting repeating words following:
    #   https://github.com/JialianW/GRiT/blob/master/grit/modeling/text/
    #   text_decoder.py#L450
    last_prediction = state.predictions[
        :, state.cur_index - 1]  # (batch_size,)
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

  init_state = State(
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


def brevity_penalty(alpha, length):
  return jnp.power(((5.0 + length) / 6.0), alpha)


@flax.struct.dataclass
class BeamState:
  """Holds beam search state data."""
  # The position of the decoding loop in the length dimension.
  cur_index: Array  # scalar int32: current decoded length index
  # The active sequence log probabilities and finished sequence scores.
  live_logprobs: Array  # float32: [batch_size, beam_size]
  finished_scores: Array  # float32: [batch_size, beam_size]
  # The current active-beam-searching and finished sequences.
  live_seqs: Array  # int32: [batch_size, beam_size, max_decode_len]
  finished_seqs: Array  # int32: [batch_size, beam_size,
  #                                         max_decode_len]
  # Records which of the 'finished_seqs' is occupied and not a filler slot.
  finished_flags: Array  # bool: [batch_size, beam_size]
  # The current state of the autoregressive decoding caches.


def flatten_beam_dim(x: jnp.ndarray, offset: int = 0) -> jnp.ndarray:
  """Flattens the first two dimensions of a non-scalar array."""
  xshape = list(x.shape)
  b_sz = xshape.pop(offset)
  xshape[offset] *= b_sz
  return x.reshape(xshape)


def unflatten_beam_dim(x: jnp.ndarray,
                       batch_size: int,
                       beam_size: int,
                       offset: int = 0) -> jnp.ndarray:
  """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
  assert batch_size * beam_size == x.shape[offset]
  xshape = list(x.shape)
  newshape = xshape[:offset] + [batch_size, beam_size] + xshape[offset + 1:]
  return x.reshape(newshape)


def beam_init(batch_size: int,
              beam_size: int,
              max_decode_len: int,
              inputs: jnp.ndarray) -> BeamState:
  """Initializes the beam search state data structure."""
  cur_index0 = jnp.array(1)
  live_logprobs0 = jnp.tile(
      jnp.array([0.0] + [NEG_INF] * (beam_size - 1)), [batch_size, 1])
  finished_scores0 = jnp.ones((batch_size, beam_size)) * NEG_INF
  live_seqs0 = jnp.broadcast_to(
      inputs[:, None], (batch_size, beam_size, inputs.shape[-1]))
  finished_seqs0 = jnp.zeros((batch_size, beam_size, max_decode_len), jnp.int32)
  finished_flags0 = jnp.zeros((batch_size, beam_size), jnp.bool_)
  # add beam dimension to attention cache pytree elements
  return BeamState(
      cur_index=cur_index0,
      live_logprobs=live_logprobs0,
      finished_scores=finished_scores0,
      live_seqs=live_seqs0,
      finished_seqs=finished_seqs0,
      finished_flags=finished_flags0,
      )


def gather_beams(nested: PyTreeDef,
                 beam_indices: jnp.ndarray,
                 batch_size: int,
                 old_beam_size: int,
                 new_beam_size: int) -> jnp.ndarray:
  """Gathers the beam slices indexed by beam_indices into new beam array.

  Args:
    nested: pytree of arrays or scalars (the latter ignored).
    beam_indices: array of beam_indices
    batch_size: size of batch.
    old_beam_size: size of _old_ beam dimension.
    new_beam_size: size of _new_ beam dimension.

  Returns:
    New pytree with new beam arrays.
    [batch_size, old_beam_size, ...] --> [batch_size, new_beam_size, ...]
  """
  del batch_size
  del new_beam_size
  # Gather via one-hot contraction, needed for SPMD partitioning.
  oh_beam_indices = jax.nn.one_hot(
      beam_indices, old_beam_size, dtype=jnp.int32)

  def gather_fn(x):
    return jnp.einsum('beo,bo...->be...', oh_beam_indices, x).astype(x.dtype)

  return jax.tree_util.tree_map(gather_fn, nested)


def gather_topk_beams(nested: PyTreeDef, score_or_log_prob: jnp.ndarray,
                      batch_size: int, new_beam_size: int) -> jnp.ndarray:
  """Gathers the top-k beam slices given by score_or_log_prob array.

  Args:
    nested: pytree of arrays or scalars (the latter ignored).
    score_or_log_prob: [batch_size, old_beam_size] array of values to sort by
      for top-k selection of beam slices.
    batch_size: int: size of batch.
    new_beam_size: int: size of _new_ top-k selected beam dimension

  Returns:
    New pytree with new beam arrays containing top k new_beam_size slices.
    [batch_size, old_beam_size, ...] --> [batch_size, new_beam_size, ...]
  """
  _, topk_indices = lax.top_k(score_or_log_prob, k=new_beam_size)
  topk_indices = jnp.flip(topk_indices, axis=1)
  return gather_beams(nested, topk_indices, batch_size,
                      score_or_log_prob.shape[1], new_beam_size)


def beam_search(inputs: jnp.ndarray,
                tokens_to_logits: Callable[[jnp.ndarray], jnp.ndarray],
                eos_index: int,
                beam_size: int = 4,
                alpha: float = 0.6,
                max_steps: int = 40,
                **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Beam search for transformer machine translation.

  If `inputs` has non-zero entries, those values are not modified, i.e.,
  the sampled values for those positions are discarded. This simulates the
  teacher forcing on the prefix positions.

  Args:
    inputs: array: (batch_size, max_steps)
    tokens_to_logits: a function that converts input
        (batch_size, max_steps) to (batch_size, max_steps, vocab_size)
    eos_index: int: id of end-of-sentence token for target vocabulary.
    beam_size: number of decoded sequences to be returned. This is equivalent
      to the number of beams used in the beam search.
    alpha: float: scaling factor for brevity penalty.
    max_steps: int: an optional maximum length of decoded sequence. If
      None, it uses `inputs.shape[1]` as `max_decode_len`.
    **kwargs: args for other decoder

  Returns:
     Tuple of:
       [batch_size, beam_size, max_decode_len] top-scoring sequences
       [batch_size, beam_size] beam-search scores.
  """
  del kwargs
  batch_size = inputs.shape[0]
  end_marker = jnp.array(eos_index)

  # initialize beam search state
  beam_search_init_state = beam_init(batch_size, beam_size, max_steps, inputs)

  def beam_search_loop_cond_fn(state: BeamState):
    """Beam search loop termination condition."""
    not_at_end = (state.cur_index < max_steps - 1)
    # Is no further progress in the beam search possible?
    # Get the best possible scores from alive sequences.
    min_brevity_penalty = brevity_penalty(alpha, max_steps)
    best_live_scores = state.live_logprobs[:, -1:] / min_brevity_penalty
    # Get the worst scores from finished sequences.
    worst_finished_scores = jnp.min(
        state.finished_scores, axis=1, keepdims=True)
    # Mask out scores from slots without any actual finished sequences.
    worst_finished_scores = jnp.where(state.finished_flags,
                                      worst_finished_scores, NEG_INF)
    # If no best possible live score is better than current worst finished
    # scores, the search cannot improve the finished set further.
    search_terminated = jnp.all(worst_finished_scores > best_live_scores)

    # If we're not at the max decode length, and the search hasn't terminated,
    # continue looping.
    return not_at_end & (~search_terminated)

  def beam_search_loop_body_fn(state: BeamState) -> BeamState:
    """Beam search loop state update function."""
    # Flatten beam dimension into batch to be compatible with model.
    test_input = flatten_beam_dim(state.live_seqs)
    flat_logits = tokens_to_logits(test_input)[:, state.cur_index - 1]
    # [batch * beam, vocab] --> [batch, beam, vocab]
    logits = unflatten_beam_dim(flat_logits, batch_size, beam_size)
    candidate_log_probs = jax.nn.log_softmax(logits)  # [batch, beam, vocab]
    log_probs = (
        candidate_log_probs + jnp.expand_dims(
            state.live_logprobs, axis=2))  # [batch, beam, vocab]
    vocab_size = log_probs.shape[-1]

    beams_to_keep = 2 * beam_size
    flat_log_probs = log_probs.reshape(
        (batch_size, beam_size * vocab_size))  # [batch, beams * vocab]
    topk_log_probs, topk_indices = lax.top_k(
        flat_log_probs, k=beams_to_keep)  # [batch, 2*beams]

    topk_ids = topk_indices % vocab_size
    topk_ids = jnp.expand_dims(topk_ids, axis=2)

    # Recover the beam index by floor division.
    topk_beam_indices = topk_indices // vocab_size
    # Gather 2*k top beams.
    # --> [batch, 2*beams, length]
    topk_seq = gather_beams(state.live_seqs, topk_beam_indices, batch_size,
                            beam_size, beams_to_keep)
    # Update sequences for the 2*K top-k new sequences.
    # --> [batch, 2*beams, length]
    topk_seq = lax.dynamic_update_slice(
        topk_seq, topk_ids, (0, 0, state.cur_index))

    # Update LIVE (in-progress) sequences:
    # Did any of these sequences reach an end marker?
    # --> [batch, 2*beams]
    newly_finished = (topk_seq[:, :, state.cur_index] == end_marker)
    # To prevent these newly finished sequences from being added to the LIVE
    # set of active beam search sequences, set their log probs to a very large
    # negative value.
    new_log_probs = topk_log_probs + newly_finished * NEG_INF
    # Determine the top k beam indices (from top 2*k beams) from log probs.
    # --> [batch, beams]
    _, new_topk_indices = lax.top_k(new_log_probs, k=beam_size)
    new_topk_indices = jnp.flip(new_topk_indices, axis=1)
    # Gather the top k beams (from top 2*k beams).
    # --> [batch, beams, length], [batch, beams]
    top_alive_seq, top_alive_log_probs = gather_beams([topk_seq, new_log_probs],
                                                      new_topk_indices,
                                                      batch_size, 2 * beam_size,
                                                      beam_size)

    # Update FINISHED (reached end of sentence) sequences:
    # Calculate new seq scores from log probabilities.
    new_scores = topk_log_probs / brevity_penalty(alpha, state.cur_index)
    # Mask out the still unfinished sequences by adding large negative value.
    # --> [batch, 2*beams]
    new_scores += (~newly_finished) * NEG_INF

    # Combine sequences, scores, and flags along the beam dimension and compare
    # new finished sequence scores to existing finished scores and select the
    # best from the new set of beams.
    finished_seqs = jnp.concatenate(  # --> [batch, 3*beams, length]
        [state.finished_seqs, topk_seq],
        axis=1)
    finished_scores = jnp.concatenate(  # --> [batch, 3*beams]
        [state.finished_scores, new_scores], axis=1)
    finished_flags = jnp.concatenate(  # --> [batch, 3*beams]
        [state.finished_flags, newly_finished], axis=1)
    # --> [batch, beams, length], [batch, beams], [batch, beams]
    top_finished_seq, top_finished_scores, top_finished_flags = (
        gather_topk_beams([finished_seqs, finished_scores, finished_flags],
                          finished_scores, batch_size, beam_size))

    return BeamState(
        cur_index=state.cur_index + 1,
        live_logprobs=top_alive_log_probs,
        finished_scores=top_finished_scores,
        live_seqs=top_alive_seq,
        finished_seqs=top_finished_seq,
        finished_flags=top_finished_flags,
        )

  # Run while loop and get final beam search state.
  final_state = lax.while_loop(beam_search_loop_cond_fn,
                               beam_search_loop_body_fn, beam_search_init_state)

  # Account for the edge-case where there are no finished sequences for a
  # particular batch item. If so, return live sequences for that batch item.
  # --> [batch]
  none_finished = jnp.any(final_state.finished_flags, axis=1)
  # --> [batch, beams, length]
  finished_seqs = jnp.where(
      none_finished[:, None, None],
      final_state.finished_seqs, final_state.live_seqs)
  # --> [batch, beams]
  finished_scores = jnp.where(
      none_finished[:, None],
      final_state.finished_scores, final_state.live_logprobs)
  return finished_seqs, finished_scores


def autoregressive_predict(
    flax_model, params, outputs, method='beam', beam_size=4,
    brevity_penalty_alpha=0.6,
    feature_key='visual_features'):
  """Generate caption from object features in an auto-agressive way.

  Args:
    flax_model: flax model.
    params: pytree of network parameters.
    outputs: dict with keys:
        'visual_features': (batch_size, num_tokens, hidden_size)
        'begin_tokens': (batch_size, max_caption_length)
        'context_tokens': (batch_size, num_tokens) or None
    method: 'greedy' or 'beam'
    beam_size: int
    brevity_penalty_alpha: float
    feature_key: str
  Returns:
    Updated outputs with updated keys:
      'text_tokens': int array (batch_size, max_caption_length),
          whose values are in range vocab_size
  """
  batch_size = outputs[feature_key].shape[0]
  visual_features = outputs[feature_key]
  begin_tokens = outputs['begin_tokens']
  context_tokens = outputs['context_tokens'] if (
      'context_tokens' in outputs) else None
  if method == 'beam' and beam_size > 1:
    assert method == 'beam', 'Beam size must be 1 for greedy decoding'
    visual_features = jnp.broadcast_to(
        visual_features[:, None],
        (batch_size, beam_size,) + visual_features.shape[1:]).reshape(
            (batch_size * beam_size,) + visual_features.shape[1:]
        )
    if context_tokens is not None:
      context_tokens = jnp.broadcast_to(
          context_tokens[:, None],
          (batch_size, beam_size, context_tokens.shape[1])).reshape(
              batch_size * beam_size, context_tokens.shape[1])
  tokens_to_logits_kwargs = {}
  if context_tokens is not None:
    tokens_to_logits_kwargs['context_tokens'] = context_tokens
  # pylint: disable=g-long-lambda
  # (text_batch_size, max_caption_length) ->
  #   (text_batch_size, max_caption_length, vocab_size)
  tokens_to_logits = lambda x: flax_model.apply(
      variables={'params': params},
      text_tokens=x,
      visual_features=visual_features,
      method=flax_model.decode_text,
      **tokens_to_logits_kwargs,
  )
  assert method in ['greedy', 'beam']
  decode_fn = greedy_decode if method == 'greedy' else beam_search
  kwargs = {}
  if method == 'beam':
    kwargs['beam_size'] = beam_size
    kwargs['alpha'] = brevity_penalty_alpha
  text_tokens, _ = decode_fn(
      begin_tokens, tokens_to_logits,
      max_steps=flax_model.max_caption_length,
      eos_index=flax_model.end_token_id,
      vocab_size=flax_model.vocab_size,
      **kwargs)
  outputs['text_tokens'] = text_tokens.reshape(
      batch_size, beam_size, flax_model.max_caption_length)
  # output of beam search scores are in increasing order.
  outputs['text_tokens'] = outputs['text_tokens'][:, -1]
  return outputs
