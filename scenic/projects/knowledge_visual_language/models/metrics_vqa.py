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

"""Metric functions for VQA."""
import string
from typing import Any

from clu import metrics
import clu.values as clu_values
import flax
import jax.numpy as jnp
import numpy as np
from scenic.projects.knowledge_visual_language.data import data_utils
from scenic.projects.t5 import tokenizer as t5_tokenizer
from scenic.projects.vit_vqa.models import qa_utils


_PUNCTUATION = string.punctuation + '‘’´`_'
# Recheck the total number of ground truth answers
# received from the data loader.
MAX_GT_ANSWERS = 10


def normalize_answer(prediction: str) -> str:
  answer = qa_utils.normalize_answer(
      prediction, punc_chars=_PUNCTUATION, punc_repl=''
  )
  return answer.strip()


@flax.struct.dataclass
class VQAMetrics(
    metrics.CollectingMetric.from_outputs(('predictions', 'targets'))
):
  """Computes VQA accuracy and F1 metrics.

  Since jax does not support strings, during the compute phase, which is done
  outside of pmap and other jax functions, converts the integer word/text
  predictions into words/strings which are used to compute the scores.

  Attributes:
    predictions: jnp.ndarray of integers representing word/text predictions.
    targets: jnp.ndarray of integers representing ground truth words/text.
  """

  def compute(self) -> dict[str, Any]:
    values = super().compute()
    tokenizer = t5_tokenizer.build_dmvr_sp_model()
    tokenizer.initialize()
    # Moves all values into the batch dimension when run with pmap.
    predictions = jnp.reshape(
        values['predictions'], (-1, values['predictions'].shape[-1])
    )
    # Targets has multiple answers per-question, so we keep that structure when
    # reshaping.
    targets = values['targets']
    targets = jnp.reshape(targets, (-1, targets.shape[-2], targets.shape[-1]))
    assert len(targets.shape) == 3
    assert len(predictions.shape) == 2

    # Convert the values into text.
    prediction_tokens = []
    for p in predictions.tolist():
      if data_utils.EOS_ID in p:
        p = p[: p.index(data_utils.EOS_ID)]
      prediction_tokens += [tokenizer.indices_to_string(p)]
    # Each question has multiple answers, so we need to decode a list-of-list.

    target_tokens = []
    for target_answers in targets.tolist():
      tokens = []
      for answer in target_answers:
        if data_utils.EOS_ID in answer:
          answer = answer[: answer.index(data_utils.EOS_ID)]
        tokens += [tokenizer.indices_to_string(answer)]
      target_tokens += [tokens]
    # Normalize answers, which expects a string.
    predictions = [normalize_answer(p) for p in prediction_tokens]
    targets = [[normalize_answer(a) for a in t] for t in target_tokens]

    if len(targets[0]) > 1:
      # VizWiz and VQA2.0 style metric with multiple GT answers.
      qa = qa_utils.vqa_metrics(targets, predictions)
    else:
      # SNLI-VE, GQA, NLVR single GT answer metric.
      qa = qa_utils.qa_metrics(targets, predictions)
    qa['acc'] = 100 * np.average(
        [p in ans for p, ans in zip(predictions, targets)]
    )
    return qa

  def compute_value(self) -> dict[str, clu_values.Value]:
    metric_values = self.compute()
    metric_results = {
        key: clu_values.Scalar(value) for key, value in metric_values.items()
    }
    return metric_results
