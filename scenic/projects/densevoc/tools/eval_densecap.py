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

r"""Re-evaluate dense-caption mAP given ground truth json and predictions.

The evaluation follows the official lua script:
https://github.com/jcjohnson/densecap/blob/maste*/eval/eval_utils.lua


python eval_densecap.py --gt_json /path/to/vg/annotations/test.json \
--pred_json /path/to/predictions.json
"""

import json

from absl import app
from absl import flags
from absl import logging

from scenic.projects.densevoc import densevoc_evaluator
from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'gt_json', '', 'path to the json annotations.')
flags.DEFINE_string(
    'pred_json', '', 'path to the prediction json.')
flags.DEFINE_string(
    'score_key',
    'score',
    'score key.')


def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  logging.info('Loading %s', FLAGS.gt_json)
  densecap_eval = densevoc_evaluator.DensecapEval(
      FLAGS.gt_json, score_key=FLAGS.score_key)
  preds = json.load(gfile.GFile(FLAGS.pred_json, 'r'))
  results = densecap_eval.compute_metrics(preds)
  print(results)

if __name__ == '__main__':
  app.run(main)
