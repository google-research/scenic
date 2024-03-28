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

r"""Add bounding box predictions from ORViT paper to TFRecords."""
import os

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

FILE_SUFFIX_PATTERN = '-{:05d}-of-{:05d}'
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_tfrecord',
    '/path/to/ssv2/tfrecord@xxx',
    'Input path')
flags.DEFINE_string(
    'bbox_folder',
    '/path/to/orvid/box/folder/',
    'path to the json annotations.')
flags.DEFINE_string(
    'output_tfrecord',
    '/path/to/ssv2.orvit_box/tfrecord',
    'Output path')


def process(example, ann=None, box_key='orvit'):
  """Add bounding boxes as additional fields to TFRecord."""
  height = example.context.feature['image/height'].int64_list.value[0]
  width = example.context.feature['image/width'].int64_list.value[0]
  num_frames = example.context.feature['clip/frames'].int64_list.value[0]
  feature_list = example.feature_lists.feature_list

  if ann is None:
    annotated_frames = 0
  else:
    xmin = ann[..., 0] / width  # T x O
    ymin = ann[..., 1] / height  # T x O
    xmax = ann[..., 2] / width  # T x O
    ymax = ann[..., 3] / height  # T x O
    boxes = np.stack([xmin, ymin, xmax, ymax], axis=2)  # T x O x 4
    scores = ann[..., 4]
    annotated_frames = ann.shape[0]

  feature_list[f'{box_key}/bbox/xmin'].Clear()
  feature_list[f'{box_key}/bbox/xmax'].Clear()
  feature_list[f'{box_key}/bbox/ymin'].Clear()
  feature_list[f'{box_key}/bbox/ymax'].Clear()
  feature_list[f'{box_key}/bbox/score'].Clear()

  for frame_idx in range(num_frames):
    j = frame_idx - num_frames + annotated_frames
    if j < 0 or j >= annotated_frames:
      boxes_t = np.zeros((0, 4), dtype=np.float32)
      scores_t = np.zeros((0), dtype=np.float32)
    else:
      boxes_t = boxes[j]
      scores_t = scores[j]
    feature_list[f'{box_key}/bbox/score'].feature.add().float_list.value.extend(
        scores_t.tolist())
    feature_list[f'{box_key}/bbox/xmin'].feature.add().float_list.value.extend(
        boxes_t[:, 0].tolist())
    feature_list[f'{box_key}/bbox/ymin'].feature.add().float_list.value.extend(
        boxes_t[:, 1].tolist())
    feature_list[f'{box_key}/bbox/xmax'].feature.add().float_list.value.extend(
        boxes_t[:, 2].tolist())
    feature_list[f'{box_key}/bbox/ymax'].feature.add().float_list.value.extend(
        boxes_t[:, 3].tolist())
  return example


def read_and_convert_boxes(video_box_path, num_boxes=4):
  """Convert annotation in tracking orders."""
  if not os.path.exists(video_box_path):
    return None
  num_frames = len(os.listdir(video_box_path))
  box_tensors = np.zeros((num_frames, num_boxes, 5), dtype=np.float32)
  for i in range(num_frames):
    frame_name = f'{video_box_path}/{i + 1:04d}.npz'
    frame_data = dict(np.load(open(frame_name, 'rb')))
    hand_idx, obj_idx = 0, 2
    for ibox in range(len(frame_data['boxes'])):
      standard_category = frame_data['pred_classes'][ibox]
      assert standard_category in [0, 1]
      global_box_id = standard_category
      if global_box_id == 0:
        global_box_id = hand_idx
        hand_idx += 1
      elif global_box_id == 1:
        global_box_id = obj_idx
        obj_idx += 1
      if global_box_id < num_boxes:
        box_tensors[i, global_box_id, :4] = frame_data['boxes'][ibox]
        box_tensors[i, global_box_id, 4] = frame_data['scores'][ibox]
  return box_tensors


def decode_sharded_names(paths):
  """Convert sharded file names into a list."""
  ret = []
  paths = paths.split(',')
  for name in paths:
    if '@' in name:
      idx = name.find('@')
      num_shards = int(name[idx + 1:])
      names = [('{}' + FILE_SUFFIX_PATTERN).format(
          name[:idx], i, num_shards) for i in range(num_shards)]
      ret.extend(names)
    else:
      ret.append(name)
  return ret


def main(unused_argv):
  shard_names = decode_sharded_names(FLAGS.input_tfrecord)
  for shard_name in shard_names:
    print('Processing', shard_name)
    raw_ds = tf.data.TFRecordDataset(shard_name)
    raw_data_iter = iter(raw_ds)
    shard_appendix = shard_name[-len(FILE_SUFFIX_PATTERN.format(0, 0)):]
    writer = tf.io.TFRecordWriter(
        f'{FLAGS.output_tfrecord}{shard_appendix}')
    while True:
      try:
        raw_data = next(raw_data_iter)
        example = tf.train.SequenceExample.FromString(raw_data.numpy())
      except:  # pylint: disable=bare-except
        break
      video_data_path = (
          example.context.feature['data_path'].bytes_list.value[0]
      ).decode('utf-8')
      key = int(video_data_path[video_data_path.rfind('/') + 1:-len('.webm')])
      video_box_path = f'{FLAGS.bbox_folder}/{key}/'
      box_tensors = read_and_convert_boxes(video_box_path)
      example = process(example, ann=box_tensors)
      writer.write(example.SerializeToString())
    writer.close()
if __name__ == '__main__':
  app.run(main)
