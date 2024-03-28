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

r"""Build LLaVa data tfrecords from json files.


python scenic/projects/pixel_llm/tools/build_llava_tfrecord.py \
    --output_dir ~/Datasets/PixelLLM/llava/LLaVA-Instruct-150K \
    --input_json ~/Datasets/PixelLLM/LLaVA-Instruct-150K/llava_v1_5_mix665k.json \
    --image_root ~/Datasets/PixelLLM/llava_images

"""

import io
import json
import os

from absl import app
from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.io import gfile
import tqdm


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'output_dir',
    '',
    'Output path of TFRecord',
)
flags.DEFINE_string(
    'input_json',
    '',
    'Input path of json file',
)
flags.DEFINE_string(
    'image_root',
    '',
    'image root',
)
flags.DEFINE_boolean('to_jpg', False, 'convert image to jpg format')


def convert_image_to_jpg_bytestring(image_bytestring):
  # Load the image from the bytestring
  input_image_stream = io.BytesIO(image_bytestring)
  image = Image.open(input_image_stream)
  image = image.convert('RGB')

  # Convert the image to JPEG format
  output_image_stream = io.BytesIO()
  image.save(output_image_stream, format='JPEG')
  jpg_bytestring = output_image_stream.getvalue()

  return jpg_bytestring


def str_to_bytes(string):
  return string.encode('utf-8')


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def process_record(data_dic):
  """Creates a example from a dict."""
  img_path = os.path.join(FLAGS.image_root, data_dic['image'])
  img_string = gfile.GFile(img_path, 'rb').read()
  if FLAGS.to_jpg:
    img_string = convert_image_to_jpg_bytestring(img_string)
  img_id = str(data_dic['id'])

  conv_human = []
  conv_agent = []
  for conv in data_dic['conversations']:
    if conv['from'] == 'human':
      conv_human.append(conv['value'])
    else:
      conv_agent.append(conv['value'])
  assert len(conv_human) == len(conv_agent)

  feature = {
      'image/encoded': _bytes_feature([img_string]),
      'image/id': _bytes_feature([str_to_bytes(img_id)]),
      'conversations/human': _bytes_feature(
          [str_to_bytes(x) for x in conv_human]
      ),
      'conversations/agent': _bytes_feature(
          [str_to_bytes(x) for x in conv_agent]
      ),
  }

  example = tf.train.Example(features=tf.train.Features(feature=feature))
  example = example.SerializeToString()
  return example


def main(unused_argv):
  raw_data = json.load(gfile.GFile(FLAGS.input_json, 'r'))
  data = []
  for dic in raw_data:
    if 'image' in dic:
      data.append(dic)
  print('====', len(data), len(raw_data), '====')
  num_shards = 2 ** (int(np.log2(len(data)) - 10))
  num_shards = max(num_shards, 64)

  output_path = os.path.join(
      FLAGS.output_dir,
      os.path.basename(FLAGS.input_json).replace('.json', '.tfrecord'),
  )
  num_examples_per_shard = (len(data) - 1) // num_shards + 1
  output_path_pattern = output_path + '-{:05d}-of-{:05d}'

  shard_id = 0
  shard_output_path = output_path_pattern.format(shard_id, num_shards)
  gfile.makedirs(os.path.dirname(shard_output_path))
  print('Writing to', shard_output_path)
  writer = tf.io.TFRecordWriter(shard_output_path)

  for i, dic in tqdm.tqdm(enumerate(data), total=len(data)):
    record = process_record(dic)
    writer.write(record)
    if ((i+1) % num_examples_per_shard == 0):
      writer.close()
      shard_id += 1
      shard_output_path = output_path_pattern.format(shard_id, num_shards)
      print('Writing to', shard_output_path)
      writer = tf.io.TFRecordWriter(shard_output_path)
  writer.close()

if __name__ == '__main__':
  app.run(main)
