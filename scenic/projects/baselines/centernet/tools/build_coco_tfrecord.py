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

r"""Convert COCO json annotation and images into tfrecord.

Example usage:


python scenic/projects/baselines/centernet/tools/build_coco_tfrecord.py \
--input_json ~/Datasets/COCO/annotations/instances_train2017.json \
--image_path ~/Datasets/COCO/train2017/ \
--output_path ~/Datasets/COCO/tfrecords/instances_train2017.tfrecord \
--num_shards 256
"""
import io
import json

from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
from pycocotools import mask as mask_api
import tensorflow as tf
from tensorflow.io import gfile


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_json',
    '',
    'path to the json annotations.')
flags.DEFINE_string(
    'image_path',
    '',
    'path to images.')
flags.DEFINE_string(
    'output_path',
    '',
    'Output path of SSTable of bounding boxes')
flags.DEFINE_integer('num_samples', -1, '')
flags.DEFINE_integer('num_shards', 32, '')


def str_to_bytes(string):
  return string.encode('utf-8')


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def polygons_to_bitmask(polygons, height, width):
  """Convert polygons to bitmask."""
  if len(polygons) <= 0:
    # COCOAPI does not support empty polygons
    return np.zeros((height, width)).astype(bool)
  rles = mask_api.frPyObjects(polygons, height, width)
  rle = mask_api.merge(rles)
  return mask_api.decode(rle).astype(bool)


def numpy_to_encoded(image_nps):
  image_bytes = []
  for image_np in image_nps:
    image_pil = Image.fromarray(image_np)
    buffer = io.BytesIO()
    image_pil.save(buffer, format='PNG')
    buffer.seek(0)
    image_byte = buffer.getvalue()
    image_bytes.append(image_byte)
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=image_bytes))


def process_record(image_info, anns, image_path, clsid2contid):
  """Creates a sequence example from a list of dict."""
  if 'file_name' in image_info:
    file_name = image_info['file_name']
  else:
    file_name = image_info['coco_url'][30:]
  img_path = image_path + file_name
  img_string = gfile.Open(img_path, 'rb').read()
  width, height = image_info['width'], image_info['height']
  bbox = np.asarray(
      [x['bbox'] for x in anns], dtype=np.float32).reshape(-1, 4)

  with_mask = False
  for x in anns:
    if 'segmentation' in x:
      with_mask = True
      if isinstance(x['segmentation'], list):
        x['mask'] = polygons_to_bitmask(x['segmentation'], height, width)
      elif isinstance(x['segmentation'], dict):
        if isinstance(x['segmentation']['counts'], list):
          rle = mask_api.frPyObjects([x['segmentation']], height, width)
        else:
          rle = [x['segmentation']]
        x['mask'] = mask_api.decode(rle)
      else:
        assert 0, type(x['segmentation'])
      if len(x['mask'].shape) == 3:
        assert x['mask'].shape[2] == 1, x['mask'].shape
        x['mask'] = x['mask'][:, :, 0]

  if with_mask:
    mask = np.asarray([x['mask'] * 255 for x in anns], dtype=np.uint8)
  else:
    mask = None
  areas = bbox[:, 2] * bbox[:, 3]
  bbox[:, 2:] = bbox[:, 2:] + bbox[:, :2]
  bbox[:, [0, 2]] /= width
  bbox[:, [1, 3]] /= height
  # tfds builder use format [y0, x0, y1, x1]
  bbox[:, [0, 1]], bbox[:, [2, 3]] = bbox[:, [1, 0]], bbox[:, [3, 2]]

  feature = {
      'image/encoded': _bytes_feature([img_string]),
      'image/filename': _bytes_feature([str_to_bytes(file_name)]),
      'image/height': _int64_feature([image_info['height']]),
      'image/width': _int64_feature([image_info['width']]),
      'image/id': _int64_feature([image_info['id']]),
      'objects/bbox': _float_feature(bbox.flatten()),
      'objects/area': _int64_feature(np.asarray(areas, dtype=np.int64)),
      'objects/id': _int64_feature(np.asarray(
          [x['id'] for x in anns], dtype=np.int64)),
      'objects/is_crowd': _int64_feature(np.asarray(
          [x['iscrowd'] if 'iscrowd' in x else 0 for x in anns],
          dtype=np.int64)),
      'objects/label': _int64_feature(np.asarray(
          [clsid2contid[x['category_id']] for x in anns], dtype=np.int64)),
  }
  if with_mask:
    feature['objects/segmentation'] = numpy_to_encoded(mask)
    for x in anns:
      del x['mask']
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  example = example.SerializeToString()
  return example


def main(unused_argv):
  logging.info('Loading %s', FLAGS.input_json)
  data = json.load(gfile.Open(FLAGS.input_json, 'r'))
  images = data['images']
  annotations = {x['id']: [] for x in images}
  clsid2contid = {x['id']: i for i, x in enumerate(
      sorted(data['categories'], key=lambda x: x['id']))}

  for x in data['annotations']:
    annotations[x['image_id']].append(x)

  if FLAGS.num_samples > 0:
    images = images[:FLAGS.num_samples]

  output_path = FLAGS.output_path
  shard_id = 0
  output_path_pattern = output_path + '-{:05d}-of-{:05d}'
  output_path = output_path_pattern.format(shard_id, FLAGS.num_shards)
  num_examples_per_shard = (len(images) - 1) // FLAGS.num_shards + 1
  print('Writing to', output_path)
  writer = tf.io.TFRecordWriter(output_path)
  num_exampels = 0
  for i, image_info in enumerate(images):
    if i % 1000 == 0:
      print(i)
    anns = annotations[image_info['id']]
    record = process_record(
        image_info, anns, FLAGS.image_path, clsid2contid)
    writer.write(record)
    num_exampels += 1
    if (num_exampels % num_examples_per_shard == 0):
      writer.close()
      shard_id += 1
      output_path = output_path_pattern.format(shard_id, FLAGS.num_shards)
      print('Writing to', output_path)
      writer = tf.io.TFRecordWriter(output_path)
  writer.close()

if __name__ == '__main__':
  app.run(main)
