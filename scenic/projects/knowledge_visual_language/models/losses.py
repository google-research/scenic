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

"""Loss functions."""

from absl import logging
from flax.training import common_utils
import jax
import jax.numpy as jnp
from scenic.model_lib.base_models import model_utils as base_model_utils


def nll_loss(targets, pred, target_masks=None, label_smoothing=0):
  """Negative Log-loglikelihood loss (perplexity).

  Args:
    targets: ground-truth labels
    pred: predicted logits
    target_masks: mask that don't count
    label_smoothing: factor to smooth label.

  Returns:
    loss value
  """

  vocab_size = pred.shape[-1]
  onehot_targets = common_utils.onehot(targets, vocab_size)

  return base_model_utils.weighted_softmax_cross_entropy(
      pred, onehot_targets, target_masks, label_smoothing=label_smoothing)


def contrastive_loss(query_emb: jnp.ndarray,
                     key_emb: jnp.ndarray,
                     temperature: float = 1.0):
  """Contrastive loss with hard negative samples & other in-batch negatives.

  Args:
    query_emb: An array of shape [bsz, n_dim].
    key_emb: An array of shape [bsz, n_knowledge, n_dim]. Only the first one is
      true positive sample, and the others are hard negatives.
    temperature: A scalar that the temprature is divided by it.

  Returns:
    Computed loss value.
  """
  if query_emb.shape[0] != key_emb.shape[0]:
    raise ValueError('query_emb and key_emb should have the same batch size.')
  if query_emb.shape[-1] != key_emb.shape[-1]:
    raise ValueError(
        'query_emb and key_emb should have the same embedding size.')
  per_device_bsz, k = query_emb.shape[0], key_emb.shape[1]
  global_key_emb = jnp.concatenate(jax.lax.all_gather(key_emb, 'batch'), 0)
  labels = jax.lax.axis_index(
      axis_name='batch') * per_device_bsz * k + jnp.arange(per_device_bsz)

  # bsz×d @ (bsz*n_device)×K×d -> bsz×(bsz * k * n_device)
  # positive pairs are on first diagonal.
  score_matrix = jnp.reshape(
      jnp.einsum('bd,nkd->bkn', query_emb, global_key_emb),
      [per_device_bsz, -1])

  loss = nll_loss(pred=score_matrix / temperature, targets=labels)
  accs = jnp.equal(jnp.argmax(score_matrix, axis=1), labels)
  s0, s1 = score_matrix[0][0], score_matrix[0][1]  # debug purpose
  logging.info('backward host_id : %d', jax.process_index())
  logging.info(jax.lax.axis_index(axis_name='batch'))
  return loss, (jnp.mean(accs), s0, s1)
