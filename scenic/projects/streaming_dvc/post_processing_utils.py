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

"""Util functions for Streaming DVC post-processing."""

import jax
import jax.numpy as jnp


def remove_padding_and_concate_and_pad_tokens(
    tokens, bos_id, eos_id, max_text_tokens):
  """Remove padding and concate and pad tokens.

    Removing padding tokens at the end of each caption, concat them into
    a single paragraph caption, and add paddings to the paragraph captions.
    Padding is always 0.

  Args:
    tokens: List, each element in the list has shape (batch, max_text_tokens).
    bos_id: int
    eos_id: int
    max_text_tokens: int
  Returns:
    merged_token: (batch, max_text_tokens)
  """
  stack_tokens = jnp.concatenate(
      [x[:, None] for x in tokens],
      axis=1)  # (batch, num_seqs, max_text_tokens)
  def impl(single_batch_tokens):
    single_batch_tokens = single_batch_tokens.reshape(-1)
    inds = jnp.nonzero(
        (single_batch_tokens != 0) & (single_batch_tokens != bos_id) & (
            single_batch_tokens != eos_id),
        size=max_text_tokens - 1, fill_value=-1)[0]  # (max_text_tokens - 1,)
    concate_tokens = jnp.take_along_axis(
        single_batch_tokens, inds,
        axis=0) * (inds >= 0)  # (max_text_tokens - 1,)
    concate_tokens_with_bos = jnp.concatenate(
        [jnp.zeros((1), jnp.int32) + bos_id, concate_tokens],
        axis=0)  # (max_text_tokens,)
    eos_position = jnp.minimum((inds >= 0).sum() + 1, inds.shape[0])
    eos_position_onehot = jnp.arange(inds.shape[0] + 1) == eos_position
    concate_tokens_with_bos_eos = concate_tokens_with_bos * (
        1 - eos_position_onehot) +  eos_position_onehot * eos_id
    return concate_tokens_with_bos_eos
  return jax.vmap(impl)(stack_tokens)


def remove_segments_from_wrong_checkpoint(
    text_tokens, max_end_time, ori_vocab_size, bos_id, eos_id):
  """Remove segments that is out-of-range of the current checkpoint.

  Example: text_token represents segments including: [0, 10] S1 [10, 60] S2;
    max_end_time being 50. The returning text_token should be [0, 10] S1.

  We assume text_token to be always in the correct format, i.e., 2 time tokens
  followed by the sentence. We also assume text_tokens starts with BOS.

  Args:
    text_tokens: (batch_size, max_cap_len).
    max_end_time: int, in range [0, num_bins]
    ori_vocab_size: int
    bos_id: int
    eos_id: int
  Returns:
    valid_text_tokens: (batch_size, max_cap_len)
  """
  max_cap_len = text_tokens.shape[1]
  max_num_segments = max_cap_len // 2
  def impl(single_batch_tokens):
    is_timetoken = single_batch_tokens >= ori_vocab_size  # (max_cap_len,)
    is_segment_start = is_timetoken & jnp.concatenate(
        [is_timetoken[1:], jnp.zeros((1,), dtype=bool)],
        axis=0)  # (max_cap_len,)
    # Index i is the start of a segment if token[i] and token[i+1] are both
    # time tokens.

    segment_id = jnp.cumsum(
        is_segment_start.astype(jnp.int32)) - 1  # (max_cap_len,)
    time_token_inds = jnp.nonzero(
        is_timetoken, size=max_num_segments * 2,
        fill_value=-1)[0]
    time_tokens = jnp.take_along_axis(
        single_batch_tokens, time_token_inds, axis=0)
    time_tokens = (time_tokens - ori_vocab_size) * (time_token_inds > 0) + (
        (max_end_time + 1) * (time_token_inds <= 0))
    time_tokens = time_tokens.reshape(max_num_segments, 2)
    is_valid = (time_tokens[:, 1] < max_end_time)  # (max_num_segments,)
    is_valid_token = jnp.take_along_axis(
        is_valid, segment_id, axis=0)  # (max_cap_len,)
    valid_tokens = single_batch_tokens * is_valid_token
    return valid_tokens
  valid_text_tokens = jax.vmap(impl)(text_tokens)
  # We only need to "remove padding" here.
  valid_text_tokens = remove_padding_and_concate_and_pad_tokens(
      [valid_text_tokens], bos_id, eos_id, max_cap_len)
  return valid_text_tokens


def remove_timestamps(tokens, ori_vocab_size):
  """Remove times tokens.

  Args:
    tokens: (batch_size, max_cap_len)
    ori_vocab_size: int
  Returns:
    tokens_without_timestamp: (batch_size, max_cap_len). If a token=0, it is
      assumed to be padding.
  """
  max_cap_len = tokens.shape[1]
  def impl(single_batch_tokens):
    is_caption_token = single_batch_tokens < ori_vocab_size  # (max_cap_len,)
    caption_token_inds = jnp.nonzero(
        is_caption_token, size=max_cap_len, fill_value=-1)[0]
    caption_tokens = jnp.take_along_axis(
        single_batch_tokens, caption_token_inds, axis=0)
    caption_tokens = caption_tokens * (caption_token_inds >= 0)
    return caption_tokens
  tokens_without_timestamp = jax.vmap(impl)(tokens)
  return tokens_without_timestamp
