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

r"""Build Visual Genome tfrecord from GRiT preprocessed json and raw images.

Downloaded GRiT preprocessed annotations `train.json` and `test.json` from:
https://github.com/JialianW/GRiT/blob/master/datasets/DATASETS.md#vg-dataset

Download Visual Genome 1.0 images from
https://homes.cs.washington.edu/~ranjay/visualgenome/api.html and put them in
the same folder `VG_100K`

Run with the corresponding path to the image folder and json file on both splits

```
mkdir ~/Datasets/VisualGenome/tfrecords/

python build_vg_tfrecord.py -- \
--input_json ~/Datasets/VisualGenome/annotations/test.json \
--image_path ~/Datasets/VisualGenome/VG_100K/ \
--output_path ~/Datasets/VisualGenome/tfrecords/test.tfrecord

python build_vg_tfrecord -- \
--input_json ~/Datasets/VisualGenome/annotations/train.json \
--image_path ~/Datasets/VisualGenome/VG_100K/ \
--output_path ~/Datasets/VisualGenome/tfrecords/train.tfrecord \
--num_shards 128
```


"""

import json

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
from tensorflow.io import gfile


FLAGS = flags.FLAGS

flags.DEFINE_string('input_json', '', 'path to the json annotations.')
flags.DEFINE_string(
    'image_path', '', 'path to images, should have 108249 images.')
flags.DEFINE_string('output_path', '', 'Output path.')
flags.DEFINE_integer('num_samples', -1, '')
flags.DEFINE_integer('num_shards', -1, '')


def str_to_bytes(string):
  return string.encode('utf-8')


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def process_record(image_info, anns, image_path):
  """Creates a sequence example from a list of dict."""
  file_name = image_info['file_name']
  img_path = image_path + file_name
  img_string = gfile.GFile(img_path, 'rb').read()
  width, height = image_info['width'], image_info['height']
  bbox = np.asarray(
      [x['bbox'] for x in anns], dtype=np.float32).reshape(-1, 4)
  bbox[:, 2:] = bbox[:, 2:] + bbox[:, :2]  # [x0, y0, w, h] -> [x0, y0, x1, y1]
  bbox[:, [0, 2]] /= width
  bbox[:, [1, 3]] /= height
  # tfds builder use format [y0, x0, y1, x1]
  bbox[:, [0, 1]], bbox[:, [2, 3]] = bbox[:, [1, 0]], bbox[:, [3, 2]]

  feature = {
      'image': _bytes_feature([img_string]),
      'img_id': _int64_feature([image_info['id']]),
      'regions/bbox': _float_feature(bbox.flatten()),
      'regions/id': _int64_feature(np.asarray(
          [x['id'] for x in anns], dtype=np.int64)),
      'regions/phrase': _bytes_feature(
          [str_to_bytes(x['caption']) for x in anns]),
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  example = example.SerializeToString()
  return example


def main(unused_argv):
  logging.info('Loading %s', FLAGS.input_json)
  data = json.load(gfile.GFile(FLAGS.input_json, 'r'))
  images = data['images']
  annotations = {x['id']: [] for x in images}
  for x in data['annotations']:
    annotations[x['image_id']].append(x)

  if FLAGS.num_samples > 0:
    images = images[:FLAGS.num_samples]

  output_path = FLAGS.output_path
  num_examples_per_shard = 0
  shard_id = 0
  output_path_pattern = output_path
  if FLAGS.num_shards > 0:
    output_path_pattern = output_path + '-{:05d}-of-{:05d}'
    output_path = output_path_pattern.format(shard_id, FLAGS.num_shards)
    num_examples_per_shard = (len(images) - 1) // FLAGS.num_shards
  print('Writing to', output_path)
  writer = tf.io.TFRecordWriter(output_path)
  num_exampels = 0
  for i, image_info in enumerate(images):
    if i % 1000 == 0:
      print(i)
    anns = annotations[image_info['id']]
    record = process_record(image_info, anns, FLAGS.image_path)
    writer.write(record)
    num_exampels += 1
    if FLAGS.num_shards > 0 and (
        num_exampels % num_examples_per_shard == 0) and (
            shard_id < FLAGS.num_shards - 1):
      writer.close()
      shard_id += 1
      output_path = output_path_pattern.format(shard_id, FLAGS.num_shards)
      print('Writing to', output_path)
      writer = tf.io.TFRecordWriter(output_path)
  writer.close()

if __name__ == '__main__':
  app.run(main)
