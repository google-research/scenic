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

r"""Densecap evaluator wrapper of Vid2Seq."""
import json
from typing import Any, Dict, Optional, Union

from absl import logging
from dmvr import tokenizers
import numpy as np
from scenic.projects.streaming_dvc import caption_evaluator
from scenic.projects.t5 import tokenizer as t5_tokenizer
# pylint: disable=g-import-not-at-top
from scenic.projects.vid2seq import dvc_eval
from tensorflow.io import gfile
# pylint: enable=g-import-not-at-top

BERT_VOCAB_SIZE = 30522
SP_VOCAB_SIZE = 32128

TOKENIZER = Union[tokenizers.BertTokenizer, t5_tokenizer.SentencePieceTokenizer]


def remove_nonascii(text):
  return ''.join([i if ord(i) < 128 else ' ' for i in text])


def load_data(path):
  """Load data from dumped json files."""
  if path.endswith('.json'):
    files = [path]
  else:
    # NOTE: we need to make sure there is only one copy of prediction file.
    files = [
        path + x for x in sorted(gfile.listdir(path)) if x.endswith('.json')]
  ret = {}
  for x in files:
    data = json.load(gfile.GFile(x, 'r'))
    ret.update(data)
  logging.info('Number of examples in %s: %d', path, len(ret))
  return ret


class DenseCapEvaluator(caption_evaluator.CaptionEvaluator):
  """Densecap evaluator following Vid2Seq."""

  def __init__(
      self, annotations_loc, tokenizer: TOKENIZER, num_bins,
      step: Optional[int] = None, iou_thresholds=(0.3, 0.5, 0.7, 0.9),
      **kwargs):
    del kwargs
    logging.info('Initializing evaluator.')
    self.annotations_loc = annotations_loc
    self.tokenizer = tokenizer
    assert isinstance(tokenizer, TOKENIZER)
    self.vocab_size = BERT_VOCAB_SIZE if isinstance(
        tokenizer, tokenizers.BertTokenizer) else SP_VOCAB_SIZE
    self.predictions = {}
    self.annotations = {}
    self.pred_image_set = set()
    self.gt_image_set = set()
    self._num_examples_added = 0
    self._num_captions_added = 0
    self.iou_thresholds = iou_thresholds
    self.step = step
    self.num_bins = num_bins

  def add_example(self, prediction: Any, target: Dict[str, np.ndarray]):
    """Add a single example to the evaluator.

    Args:
      prediction: string.
      target: Target dictionary with keys and 'image/id'.
    """
    media_id = ''.join([chr(x) for x in target['media_id'] if x])
    times, captions, abs_times = self._decode_time_and_caption(
        prediction['text_tokens'][1:].tolist(), duration=target['duration'][0])
    pred = {
        'pred_captions': captions,
        'pred_timestamps': times,
        'pred_abs_timestamps': abs_times,
    }
    if media_id not in self.predictions:
      self._num_examples_added += 1
    self.predictions[media_id] = pred

  def compute_metrics(
      self,
      save_dir: str,
      clear_annotations: Optional[bool] = True,
      skip_evaluate=False):
    """Computes the metrics for all added predictions."""
    self.write_pred_annotations_to_file(save_dir)
    gt_data = load_data(self.annotations_loc)
    pred_data = self.predictions
    keys = [k for k in gt_data.keys()]
    gt_segments = [np.asarray(gt_data[k]['gt_timestamps']) for k in keys]
    gt_captions = [gt_data[k]['gt_captions'] for k in keys]
    splits = [
        np.asarray(gt_data[k]['splits']) if len(gt_data[k]['splits']) else
        np.ones(len(gt_data[k]['gt_captions']), dtype=np.int32) for k in keys]
    for k in keys:
      if k not in pred_data:
        logging.info('Example %s not in prediction', k)
    predicted_segments = [
        np.asarray(pred_data[k]['pred_timestamps'])
        if k in pred_data else np.zeros((0, 2)) for k in keys]
    predicted_captions = [
        pred_data[k]['pred_captions'] if k in pred_data else [''] for k in keys]
    eval_res = dvc_eval.evaluate_dense_captions(  # pytype: disable=attribute-error
        predicted_segments=predicted_segments,
        gt_segments=gt_segments,
        predicted_captions=predicted_captions,
        gt_captions=gt_captions,
        splits=splits,
        keys=keys,
        iou_thresholds=self.iou_thresholds,
        max_workers=1,
        soda=True,
        tmponly=False,
    )
    full_res = {
        x: np.array(eval_res[x]) for x in eval_res.keys() if x != 'key'
    }

    # compute averaged statistics
    avg_res = {}
    for x in full_res:
      if x == 'SODA_c_old_1' or x == 'SODA_c_old_2':
        avg_res[x] = float(np.mean(full_res[x][full_res[x] != -1]))
      else:
        avg_res[x] = float(np.mean(full_res[x]))
    if 'SODA_c_old_1' in avg_res and 'SODA_c_old_2' in avg_res:
      avg_res['SODA_c_old'] = (
          avg_res['SODA_c_old_2'] + avg_res['SODA_c_old_1']) / 2
      del avg_res['SODA_c_old_2'], avg_res['SODA_c_old_1']
    return avg_res

  def clear(self):
    self.predictions = {}
    self.annotations = {}
    self._num_examples_added = 0
    self._num_captions_added = 0

  def _decode_time_and_caption(self, seq, duration):
    times = []
    captions = []
    abs_times = []
    j = 0
    while j < len(seq):
      x = seq[j]
      if x < self.vocab_size:
        if len(captions) <= 0:
          captions.append([])
          times.append([0, 0])
          abs_times.append([0, 0])
        captions[-1].append(x)
        j += 1
      else:
        # TODO(zhouxy): handle when the model does not predict end time.
        st = seq[j] - self.vocab_size
        ed = seq[j + 1] - self.vocab_size
        times.append([
            int(st * duration / (self.num_bins - 1)),
            int(ed * duration / (self.num_bins - 1))])
        abs_times.append([int(st), int(ed)])
        captions.append([])
        j += 2
    captions = [remove_nonascii(
        self.tokenizer.indices_to_string(x)).strip() for x in captions]
    return times, captions, abs_times
