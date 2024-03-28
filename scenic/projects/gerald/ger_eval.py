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

"""Utilities for evaluating GER model on the OVEN benchmark."""

from typing import Any, Dict, Optional

from absl import logging
import flax
import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.projects.gerald import utils
from scenic.train_lib import train_utils


NEG_INF = -1.0e7
PyTreeDef = Any
Array = jax.Array


def get_predicted_labels(seqs, code2id):
  out = - np.ones((seqs.shape[0], seqs.shape[1])).astype(np.int32)
  for bs in range(seqs.shape[0]):
    for be in range(seqs.shape[1]):
      code_str = '-'.join([str(int(c))for c in seqs[bs, be]])
      if code_str in code2id:
        out[bs, be] = code2id[code_str]
  return out


def get_first_valid_label(predicted_labels):
  out = - np.ones(predicted_labels.shape[0]).astype(np.int32)
  for b in range(predicted_labels.shape[0]):
    valid_indexes = np.where(predicted_labels[b] != -1)[0]
    if len(valid_indexes):  # pylint: disable=g-explicit-length-test
      out[b] = predicted_labels[b, valid_indexes[0]]
  return out[:, None]


def eval_loss_and_accuracy_step(
    train_state,
    batch,
    *,
    flax_model: nn.Module,
    loss_and_metrics_fn: Any,
    entity2code: Any = None,):
  """Runs a single step of evaluation."""
  code_tokens = entity2code(batch['label']['entity/id'][..., 0, 0])
  variables = {'params': train_state.params, **train_state.model_state}
  predictions = flax_model.apply(
      variables,
      batch['inputs'],
      code_tokens=code_tokens,
      context_text_tokens=batch.get('context', None),
      preprocess=True,
      train=False
  )
  _, metrics = loss_and_metrics_fn(
      predictions,
      {**{k: v for k, v in batch.items()}, 'code_tokens': code_tokens})
  # adapt to normalization API in log_train_summary
  metrics = {k: (v, 1.) for k, v in metrics.items()}
  return metrics


def evaluate_oven_ger(train_state, dataset_iter, eval_step_pmapped,
                      code2id, total_eval_steps, save_predictions=''):
  """Evaluates a Generative Entity Recognition model on OVEN."""
  eval_metrics = {}
  all_seqs = []
  all_masks = []
  all_labels = []
  all_ids = []
  for eval_step_i in range(total_eval_steps):
    if eval_step_i % 100 == 0:
      logging.info('eval step [%d/%d]', eval_step_i, total_eval_steps)
    eval_batch = next(dataset_iter)
    seqs, batch_mask, labels, ids = eval_step_pmapped(train_state, eval_batch)
    all_seqs.append(utils.to_cpu(seqs))
    all_masks.append(utils.to_cpu(batch_mask))
    all_labels.append(utils.to_cpu(labels))
    all_ids.append(utils.to_cpu(ids))
  seqs = np.concatenate(all_seqs, axis=0)
  masks = np.concatenate(all_masks, axis=0)
  labels = np.concatenate(all_labels, axis=0)
  ids = np.concatenate(all_ids, axis=0)
  idx = masks == 1
  seqs = seqs[idx]
  labels = labels[idx]
  ids = ids[idx]
  predicted_labels = get_predicted_labels(seqs, code2id)
  logging.info('predicted labels shape: %s', predicted_labels.shape)
  logging.info('gt labels shape: %s', labels.shape)
  non_valid = np.sum(predicted_labels[:, 0] == -1)
  lab = predicted_labels[:, 0].reshape((-1,))
  eval_metrics.update({'existing_entities_prop': (
      100. * (lab.shape[0] - non_valid) / lab.shape[0], 1)})
  prec_dict = compute_precision(predicted_labels, labels, (1,))
  # cheap constrained beam search
  predicted_labels = get_first_valid_label(predicted_labels)
  eval_metrics.update({
      'cheaply_constrained_' + k: v for (
          k, v) in compute_precision(predicted_labels, labels, (1,)).items()})
  eval_metrics.update(prec_dict)
  eval_metrics = {k: v[0] for k, v in eval_metrics.items()}
  return eval_metrics


def compute_precision(pred, gt, ks=(1, 2, 4, 8)):
  """Computes precision."""
  assert pred.shape[0] == gt.shape[0]
  res = {}
  for k in ks:
    topk_preds = pred[:, :k]
    true_positive = 1.0 * np.sum(topk_preds == gt[:, None], axis=-1)
    precision = np.mean(true_positive / k)
    res['prec@' + str(k)] = (precision, 1)
  return res


def eval_step(
    train_state: train_utils.TrainState,
    batch: Dict[str, jnp.ndarray],
    *,
    flax_model: nn.Module,
    config: ml_collections.ConfigDict,
    gather_to_host: Optional[bool] = True,
    ) -> Any:
  """Runs a single step of test."""
  variables = {'params': train_state.params, **train_state.model_state}
  predictions = flax_model.apply(
      variables,
      batch['inputs'],
      context_text_tokens=batch.get('context', None),
      preprocess=True,
      train=False,
      mutable=False,
      debug=False)

  batch_size = predictions['visual_features'].shape[0]
  visual_features = predictions['visual_features']
  begin_tokens = predictions['begin_tokens']
  context_tokens = predictions['context_tokens'] if (
      'context_tokens' in predictions) else None
  beam_size = config.model.decode_beam_size
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
  tokens_to_logits = lambda x: flax_model.apply(
      variables={'params': train_state.params},
      code_tokens=x,
      visual_features=visual_features,
      method=flax_model.decode_text,
      **tokens_to_logits_kwargs,
  )
  # live_seqs
  live_seqs, _ = beam_search_routine_with_different_code_lengths(
      begin_tokens, tokens_to_logits,
      max_steps=config.code_length + 1, eos_index=config.get('ger_eos', 102),
      beam_size=beam_size)
  live_seqs = live_seqs.reshape(
      batch_size, beam_size, config.code_length + 1)
  live_seqs = live_seqs[:, -config.get('topk', beam_size):, 1:]
  # The output of beam search scores are in increasing order.
  live_seqs = live_seqs[:, ::-1]

  batch_mask = batch['batch_mask']
  labels = batch['label']['entity/id'][..., 0, 0]
  question_id = batch['label']['image/id']

  if gather_to_host:
    live_seqs = jax.lax.all_gather(live_seqs, 'batch')
    batch_mask = jax.lax.all_gather(batch_mask, 'batch')
    labels = jax.lax.all_gather(labels, 'batch')
    question_id = jax.lax.all_gather(question_id, 'batch')
  return live_seqs, batch_mask, labels, question_id


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


def beam_search_routine_with_different_code_lengths(
    inputs, tokens_to_logits, eos_index=1, beam_size=4, max_steps=5):
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
    max_steps: int: an optional maximum length of decoded sequence. If
      None, it uses `inputs.shape[1]` as `max_decode_len`.

  Returns:
     Tuple of:
       [batch_size, beam_size, max_decode_len] top-scoring sequences
       [batch_size, beam_size] beam-search scores.
  """
  batch_size = inputs.shape[0]
  end_marker = jnp.array(eos_index)

  # initialize beam search state
  beam_search_init_state = beam_init(batch_size, beam_size, max_steps, inputs)

  def beam_search_loop_cond_fn(state: BeamState):
    """Beam search loop termination condition."""
    not_at_end = (state.cur_index < max_steps)
    # Is no further progress in the beam search possible?
    # Get the best possible scores from alive sequences.
    best_live_scores = state.live_logprobs[:, -1:]
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
    topk_seq = gather_beams(
        state.live_seqs, topk_beam_indices, batch_size, beam_size,
        beams_to_keep)
    # Update sequences for the 2*K top-k new sequences.
    # --> [batch, 2*beams, length]
    topk_seq = lax.dynamic_update_slice(
        topk_seq, topk_ids, (0, 0, state.cur_index))

    # Update LIVE (in-progress) sequences:
    # Did any of these sequences reach an end marker?
    # --> [batch, 2*beams]
    newly_finished = (topk_seq[:, :, state.cur_index] == end_marker)
    # If we're at the final step, then all seqs are considered finished.
    final_decoding_step = state.cur_index == max_steps - 1
    newly_finished = final_decoding_step | newly_finished
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
    top_alive_seq, top_alive_log_probs = gather_beams(
        [topk_seq, new_log_probs], new_topk_indices, batch_size, 2 * beam_size,
        beam_size)

    # Update FINISHED (reached end of sentence) sequences:
    # Calculate new seq scores from log probabilities.
    new_scores = topk_log_probs
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
        gather_topk_beams(
            [finished_seqs, finished_scores, finished_flags],
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
