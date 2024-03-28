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

"""Losses used in Verf-Focused Contrastive learning."""
from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from scenic.projects.verbs_in_action import utils


def baseline_nce(encoded_video: jnp.ndarray,
                 encoded_text: jnp.ndarray,
                 temperature: float = 0.05,
                 v2t_weight: float = 1.0,
                 t2v_weight: float = 1.0,
                 beta: float = 0.) -> float:
  """Returns HN-NCE loss when *not* considering verb hard negatives."""
  # Logits are [VIDEO x TEXT]
  logits = utils.compute_inners(encoded_video, encoded_text) / temperature

  # Video to text loss
  loss_v2t = hn_nce_loss_without_hardnegs(logits, beta=beta)
  loss_v2t = jnp.mean(loss_v2t)

  # Text to video loss
  logits_t = jnp.transpose(logits)  # [TEXT x VIDEO]
  n = logits_t.shape[-1] - 1
  loss_t2v = hn_nce_loss_without_hardnegs(logits_t, beta=beta, n=n)
  loss_t2v = jnp.mean(loss_t2v)

  loss_final = v2t_weight * loss_v2t + t2v_weight * loss_t2v
  return loss_final


def verb_hard_neg_nce(encoded_video: jnp.ndarray,
                      encoded_text: jnp.ndarray,
                      mask_text: jnp.ndarray,
                      temperature: float = 0.05,
                      v2t_weight: float = 1.0,
                      t2v_weight: float = 1.0,
                      beta: float = 0.) -> float:
  """Returns HN-NCE loss when including verb hard negatives.

  Args:
    encoded_video: The encoded videos.
    encoded_text: The encoded text.
    mask_text: 0 where caption (pos or neg), 1 where padding,
      shape [total_num_texts, 1]
    temperature: Temperature for scaling softmax distribution.
    v2t_weight: weight on vid2text loss.
    t2v_weight: weight on text2vid loss
    beta: beta parameter used for HN-NCE: see https://arxiv.org/abs/2301.02280
  """
  logits = utils.compute_inners(encoded_video, encoded_text)
  labels, masking, inverse_mask_hn_other_vid = get_contrastive_labels(
      logits, mask_text)
  logits = logits / temperature

  inverse_mask_hn_other_vid = jnp.sum(inverse_mask_hn_other_vid, axis=-1,
                                      keepdims=True)
  # Step 1): Video-to-text loss
  v2t_loss = hn_nce_loss_with_hardnegs(
      logits, labels, inverse_mask_hn_other_vid, beta=beta, masking=masking)
  # Actually it's logits_masked.shape[0] + num neg cap per video.
  v2t_loss = v2t_loss / jnp.log(inverse_mask_hn_other_vid)
  v2t_loss = jnp.mean(v2t_loss)

  # Step 2): Text-to-video loss
  logits_t = jnp.transpose(logits)
  labels_t = jnp.transpose(labels)
  t2v_loss = jnp.sum(hn_nce_loss_with_hardnegs(logits_t, labels_t, beta=beta))
  # divide by the number of pos_texts, i.e. num_vids
  t2v_loss = t2v_loss / logits.shape[0]
  # divide by log(K) where K=num_vids
  t2v_loss = t2v_loss / jnp.log(logits.shape[0])

  # Step 3): Add both loss terms.
  loss_final = v2t_weight * v2t_loss + t2v_weight * t2v_loss
  return loss_final


def hn_nce_loss_without_hardnegs(logits, alpha=1, beta=0, n=None):
  """Returns HN-NCE loss when *not* considering verb hard negatives."""
  # The weights will compare how similar is each text compared to the average
  ws = jnp.exp(beta * logits - jnp.max(beta * logits, axis=-1, keepdims=True))
  # Zeroing the diagonal.
  ws = (1 != jnp.eye(logits.shape[0], logits.shape[1])) * ws
  # Substracting the mean of each row.
  if n is None:
    n = ws.shape[-1] - 1
  ws = n * ws / jnp.sum(ws, axis=-1, keepdims=True)
  exp_logits = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
  pos = jnp.diag(exp_logits)
  ws = ws * exp_logits
  loss = - jnp.log(pos / (alpha * pos + jnp.sum(ws, axis=-1)))
  return loss


def hn_nce_loss_with_hardnegs(logits, labels, n=None, beta=0, masking=None):
  """Returns HN-NCE loss when considering verb hard negatives."""
  # The weights will compare how similar is each text compared to the average
  ws = jnp.exp(beta * logits - jnp.max(beta * logits, axis=-1, keepdims=True))
  # Zeroing the positive elements
  ws = (1 != labels) * ws
  # and the masked elements...
  if masking is not None:
    ws = (-1000000 != masking) * ws
  # Substracting the mean of each row.
  if n is None:
    n = ws.shape[-1]
  ws = (n - 1) * ws / jnp.sum(ws, axis=-1, keepdims=True)

  # Setting to 1 the positive elements so that they participate in the sum in
  # the denominator.
  ws = ws + labels

  exp_logits = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
  ws = ws * exp_logits
  loss = - jnp.sum(labels * jnp.log(exp_logits / jnp.sum(
      ws, axis=-1, keepdims=True)), axis=-1)
  return loss


def get_contrastive_labels(inners: jnp.ndarray,
                           text_mask: jnp.ndarray,
                           loss_exclude_hn_other_vid: bool = True,
                           ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Labels the inner products compute from `compute_inners`.

  Given an output from `_compute_inners` and text_mask, returns:
  -labels: indicates where is positive by 1.
  -masking: indicates where is masking by large neg number.

  Args:
    inners: The inner products returned from `compute_inners`.
    text_mask: This is 1 where we need to mask it out (padding), of
      shape D x 1 where D = batch_size x max_num_captions
    loss_exclude_hn_other_vid: Hard negative for element i of the batch is
      ignored (ie. used neither as negative not positive) for batch elements j.

  Returns:
  -labels: 0 where negative or padding, 1 where positive
  -masking: 0 where positive or negative caption, -100000 where padding
  """
  # labels 1 for positive, 0 for negative
  labels = jnp.zeros(inners.shape, dtype=inners.dtype)
  bs = inners.shape[0]
  max_num_captions = int(text_mask.shape[0]/bs)
  mask_hn_other_vid = jnp.ones(max_num_captions, dtype=inners.dtype)
  if loss_exclude_hn_other_vid:
    mask_hn_other_vid = mask_hn_other_vid.at[0].set(0)
    mask_hn_other_vid = jnp.transpose(jnp.tile(jnp.expand_dims(
        jnp.tile(mask_hn_other_vid, bs), 1), bs))
  inverse_mask_hn_other_vid = jnp.ones(inners.shape, dtype=inners.dtype)
  # 0 for those we want in logsoftmax, -1000000 for those we want to ignore
  large_neg_number = -1000000
  for idx in range(bs):
    labels = labels.at[idx, idx*max_num_captions].set(1)
    if loss_exclude_hn_other_vid:
      start = idx * max_num_captions
      end = idx * max_num_captions + max_num_captions
      mask_hn_other_vid = mask_hn_other_vid.at[
          idx, start:end].set(jnp.squeeze(text_mask[start:end]))
  masking = jnp.transpose(jnp.tile(text_mask, bs)) * large_neg_number
  if loss_exclude_hn_other_vid:
    inverse_mask_hn_other_vid = inverse_mask_hn_other_vid - mask_hn_other_vid
    masking = mask_hn_other_vid * large_neg_number
  return labels, masking, inverse_mask_hn_other_vid


def verb_phrase_nce(encoded_video: jnp.ndarray,
                    encoded_text: jnp.ndarray,
                    batch_mask_verb: jnp.ndarray,
                    temperature: float = 0.05,) -> float:
  """Returns NCE loss for video to verb loss."""
  logits = utils.compute_inners(encoded_video, encoded_text)
  labels, masking = get_labels_verbs(logits, batch_mask_verb)
  logits = logits + masking
  logits = logits / temperature
  loss = -jnp.sum(labels * nn.log_softmax(logits, axis=-1), axis=-1)
  loss = jnp.sum(loss * jnp.squeeze(batch_mask_verb)) / jnp.sum(batch_mask_verb)
  loss_final = loss / jnp.log(jnp.sum(batch_mask_verb))
  return loss_final


def get_labels_verbs(
    inners: jnp.ndarray, batch_mask_verb: jnp.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
  """Labels the inner products computed from `compute_inners` for verb loss."""
  bs = inners.shape[0]
  labels = jnp.zeros(inners.reshape((bs, bs, -1)).shape, dtype=inners.dtype)
  for i in range(inners.shape[0]):
    labels = labels.at[i, i, 0].set(jnp.squeeze(batch_mask_verb[i]))
  masking = jnp.ones(batch_mask_verb.shape, dtype=batch_mask_verb.dtype)
  masking = masking - batch_mask_verb
  masking = jnp.transpose(jnp.tile(masking, bs))*-1000000
  return labels.reshape(inners.shape), masking


def get_labels(inners: jnp.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Labels the inner products compute dfrom `compute_inners`."""
  bs = inners.shape[0]
  labels = np.zeros(inners.reshape((bs, bs, -1)).shape, dtype=inners.dtype)
  labels_all = np.zeros_like(labels)
  for i in range(inners.shape[0]):
    labels[i, i, 0] = 1
    labels_all[i, i, :] = 1
  return labels.reshape(inners.shape), labels_all.reshape(inners.shape)
