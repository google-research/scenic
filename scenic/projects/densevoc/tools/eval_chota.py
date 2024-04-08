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

# pylint: disable=line-too-long
r"""Re-evaluate tracking metrics given ground truth json and predictions.


python eval_chota.py --gt_json /path/to/vidstg_max200f_val_coco_format.json \
--pred_json /path/to/predictions.json

"""
import json
import os
import tempfile

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers

from scenic.projects.densevoc import chota
from tensorflow.io import gfile

# replace with the path to your JAVA bin location
JRE_BIN_JAVA = path_to_jre_bin_java

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', '/tmp/', '')
flags.DEFINE_string(
    'gt_json', '', 'path to the json annotations.')
flags.DEFINE_string(
    'pred_json', '', 'path to the prediction json.')
flags.DEFINE_string(
    'caption_metric', 'cider,meteor,spice', '')


def main(unused_argv):
  java_jre = get_java_bin_path()
  os.environ['JRE_BIN_JAVA'] = JRE_BIN_JAVA
  logging.set_verbosity(logging.INFO)
  logging.info('Loading %s', FLAGS.gt_json)
  gt_data = json.load(gfile.GFile(FLAGS.gt_json, 'r'))
  pred_data = json.load(gfile.GFile(FLAGS.pred_json, 'r'))

  chota_evaluator = chota.CHOTA(caption_metric=FLAGS.caption_metric.split(','))
  results = chota_evaluator.compute_metrics(gt_data, pred_data)
  writer = metric_writers.create_default_writer(FLAGS.workdir)
  for k, v in results.items():
    logging.info('%s: %f', k, v)
    writer.write_scalars(0, {k: v})
  json.dump(results, gfile.GFile(FLAGS.workdir + '/results.json', 'w'))

if __name__ == '__main__':
  app.run(main)
