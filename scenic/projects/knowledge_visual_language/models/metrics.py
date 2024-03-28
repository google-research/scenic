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

"""Metric functions."""

from typing import Dict, Tuple
from flax.training import common_utils
from scenic.model_lib.base_models import model_utils as base_model_utils
from scenic.projects.knowledge_visual_language.models import constants

JTensor = constants.JTensor
JTensorDict = constants.JTensorDict


def token_accuracy(logits, batch: JTensorDict) -> Dict[str, Tuple[float, int]]:
  """Return the accuracy for LM prediction.

  Args:
    logits: Output of model in shape [B, L, C].
    batch: Batch of data that has 'decoder_outputs' as ground-truth.

  Returns:
    Accuracy stored as Dict.
  """
  targets = batch['decoder_target_tokens']
  vocab_size = logits.shape[-1]
  onehot_targets = common_utils.onehot(targets, vocab_size)
  masks = targets > 0
  n_corrects = base_model_utils.weighted_correctly_classified(
      logits, onehot_targets, masks)
  n_valids = base_model_utils.num_examples(logits, onehot_targets, masks)
  return {  # pytype: disable=bad-return-type  # jax-ndarray
      'token_accuracy':
          base_model_utils.psum_metric_normalizer((n_corrects, n_valids))
  }

