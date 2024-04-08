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

r"""Build VidSTG tfrecord.

VidSTG dataset (https://github.com/Guaranteer/VidSTG-Dataset) is built on top
of VidOR dataset (https://xdshang.github.io/docs/vidor.html). Before running
this script to process VidSTG datasets, please follow the instruction in
'./build_vidor_tfrecord.py' to process VidOR. This script will use the resulting
tfrecord files of VidOR datasets.

Download the VidSTG annotation jsons from
`https://github.com/Guaranteer/VidSTG-Dataset/tree/master/annotations`.
Note both VidSTG train and val are from VidOR training set.

Run the following command with path to the downloaded VidSTG json, the VidOR
json folder path from VidOR, and the VidOR tfrecord created in
'./build_vidor_tfrecord.py'.

```
mkdir ~/Datasets/VidSTG/tfrecords/

python build_vidstg_tfrecord.py \
--vidstg_json ~/Datasets/VidSTG/annotations/val_annotations.json \
--vidor_json_path ~/Datasets/VidOR/annotations/training/ \
--vidor_tfrecord_path ~/Datasets/VidOR/tfrecords/vidor.training.tfrecord@256 \
--output_path ~/Datasets/VidSTG/tfrecords/vidstg.video.max200f.caption.val.tfrecord@32 \
--video_max_len 200

python build_vidstg_tfrecord.py \
--vidstg_json ~/Datasets/VidSTG/annotations/train_annotations.json \
--vidor_json_path ~/Datasets/VidOR/annotations/training/ \
--vidor_tfrecord_path ~/Datasets/VidOR/tfrecords/vidor.training.tfrecord@256 \
--output_path ~/Datasets/VidSTG/tfrecords/vidstg.video.caption.train.tfrecord@256
```


NOTE(zhouxy): this script is for external reproducibility only, and is NOT the
exact script we run for the paper. Our original script uses internal
tools which run faster but can't be released. Users may integrate
this script with multi-threaded tools for speedup.
"""

import json
import os

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string('vidstg_json', '', 'path to the VidSTG which has captions.')
flags.DEFINE_string(
    'vidor_json_path', '',
    'path to the VidOR annotations which has boxes.')
flags.DEFINE_string(
    'vidor_tfrecord_path', '', 'path to vidor tfrecords.')
flags.DEFINE_string(
    'output_path', '', 'Output path.')
flags.DEFINE_float('output_fps', 5, '')
flags.DEFINE_integer('video_max_len', -1, '')

MAX_FRAMES_PER_VIDEO = 6000
MAX_TRACKS_PER_VIDEO = 100


def decode_sharded_names(path):
  """Convert sharded file names into a list."""
  ret = []
  path = path.split(',')
  for name in path:
    if '@' in name:
      num_shards = int(name.split('@')[1].split('.')[0])
      suffix = name.split(f'@{num_shards}')[-1]
      prefix = name.split('@')[0]
      names = [
          f'{prefix}-{i:05d}-of-{num_shards:05d}{suffix}'
          for i in range(num_shards)
      ]
      ret.extend(names)
    else:
      ret.append(name)
  return ret


def str_to_bytes(string):
  return string.encode('utf-8')


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def process_video_record(
    all_images, width, height, video_id, all_objects,
    video_captions, video_questions, all_ids):
  """Creates a sequence example from a list of dict."""
  boxes = []
  track_ids, caption_ids, captions, questions = [], [], [], []
  for anns in all_objects:
    bbox = np.asarray(
        [x['bbox'] for x in anns], dtype=np.float32).reshape(-1, 4)
    # [x0, y0, w, h] -> [x0, y0, x1, y1]
    bbox[:, 2:] = bbox[:, 2:] + bbox[:, :2]
    bbox[:, [0, 2]] /= width
    bbox[:, [1, 3]] /= height
    # tfds builder uses format [y0, x0, y1, x1]
    bbox[:, [0, 1]], bbox[:, [2, 3]] = bbox[:, [1, 0]], bbox[:, [3, 2]]
    boxes.append(_float_feature(bbox.flatten()))
    track_ids.append(_int64_feature(
        np.asarray([x['track_id'] for x in anns], dtype=np.int64)))
    caption_ids.append(_int64_feature(
        np.asarray([x['caption_id'] for x in anns], dtype=np.int64)))
    captions.append(_bytes_feature(
        [str_to_bytes(x['caption']) for x in anns]))
    questions.append(_bytes_feature(
        [str_to_bytes(x['question']) for x in anns]))

  feature = {
      'video_id': _int64_feature([video_id]),
      'video/captions': _bytes_feature(
          [str_to_bytes(x) for x in video_captions]),
      'video/questions': _bytes_feature(
          [str_to_bytes(x) for x in video_questions]),
      'image_ids': _int64_feature(all_ids),
  }
  seq_feature = {
      'image/encoded': tf.train.FeatureList(feature=all_images),
      'objects/bbox': tf.train.FeatureList(feature=boxes),
      'objects/track_id': tf.train.FeatureList(feature=track_ids),
      'objects/caption_id': tf.train.FeatureList(feature=caption_ids),
      'objects/caption': tf.train.FeatureList(feature=captions),
      'objects/question': tf.train.FeatureList(feature=questions),
  }
  example = tf.train.SequenceExample(
      context=tf.train.Features(feature=feature),
      feature_lists=tf.train.FeatureLists(feature_list=seq_feature))
  return example


class ConvertSeqExamples(object):
  """Read images in tfrecord and add annotations from json files."""

  def __init__(
      self, vidor_json_path, vid2stg_anns, output_fps, video_id_to_cont_id,
      video_max_len):
    self.vidor_json_path = vidor_json_path
    self.vid2stg_anns = vid2stg_anns
    self.output_fps = output_fps
    self.video_id_to_cont_id = video_id_to_cont_id
    self.video_max_len = video_max_len

  def process(self, example):
    """Process a single example."""
    vid = str(int(example.context.feature[
        'video_id'].bytes_list.value[0].decode('utf-8')))
    if vid not in self.vid2stg_anns:
      return []

    full_data_path = example.context.feature[
        'data_path'].bytes_list.value[0].decode('utf-8')
    height = example.context.feature['height'].int64_list.value[0]
    width = example.context.feature['width'].int64_list.value[0]
    # FPS of the annotation
    fps = example.context.feature['fps'].float_list.value[0]
    # FPS of the decoded images
    decode_fps = example.context.feature['image/frame_rate'].float_list.value[0]
    assert abs(fps - decode_fps) < 1, f'{fps}, {decode_fps}'

    frame_count = example.context.feature['frame_count'].int64_list.value[0]
    num_frames = example.context.feature['clip/frames'].int64_list.value[0]
    assert frame_count == num_frames, f'{frame_count}, {num_frames}'
    feature_list = example.feature_lists.feature_list
    assert len(feature_list['image/encoded'].feature) == num_frames, (
        '{} {}'.format(len(feature_list['image/encoded'].feature), num_frames))

    data_path = full_data_path[full_data_path[
        :full_data_path.rfind('/')].rfind('/') + 1:].replace('mp4', 'json')
    vidor_path = os.path.join(self.vidor_json_path, data_path)
    vidor_anns = json.load(gfile.GFile(vidor_path, 'r'))
    vidstg_anns = self.vid2stg_anns[vid]
    video_captions = set()
    video_questions = set()
    for (_, traj) in vidstg_anns:
      for cap in traj['captions']:
        video_captions.add(cap['description'])
      for q in traj['questions']:
        video_questions.add(q['description'])

    sampling_rate = self.output_fps / fps
    assert sampling_rate <= 1, f'{sampling_rate}'
    frame_ids = [0]
    for frame_id in range(num_frames):
      # Filtering rule following TubeDETR:
      # https://github.com/antoyang/TubeDETR/blob/main/datasets/
      # vidstg_eval.py#L62
      if int(frame_ids[-1] * sampling_rate) < int(frame_id * sampling_rate):
        frame_ids.append(frame_id)
    if len(frame_ids) > self.video_max_len:  # subsample at video_max_len
      frame_ids = [
          frame_ids[(j * len(frame_ids)) // self.video_max_len]
          for j in range(self.video_max_len)]
    video_id = self.video_id_to_cont_id[int(vid)]
    all_images, all_objects, all_ids = [], [], []
    for frame_id in frame_ids:
      image_encoded = feature_list['image/encoded'].feature[frame_id]
      image_id = video_id * MAX_FRAMES_PER_VIDEO + frame_id
      vidor_anns_frame = vidor_anns['trajectories'][frame_id]
      objs = {}
      for x in vidor_anns_frame:
        b = x['bbox']
        bbox = [
            b['xmin'], b['ymin'],
            b['xmax'] - b['xmin'], b['ymax'] - b['ymin']]  # (x0, y0, w, h)
        objs[x['tid']] = {
            'caption': '', 'question': '', 'bbox': bbox,
            'track_id': video_id * MAX_TRACKS_PER_VIDEO + int(x['tid']),
            'caption_id': 0,
            'question_id': 0,
        }
      for (traj_id, traj) in vidstg_anns:
        if frame_id >= traj['temporal_gt']['begin_fid'] and frame_id < traj[
            'temporal_gt']['end_fid']:
          for cap in traj['captions']:
            cap_tid = cap['target_id']
            assert cap_tid in objs
            objs[cap_tid]['caption'] = cap['description']
            objs[cap_tid]['caption_id'] = traj_id
          for q in traj['questions']:
            q_tid = q['target_id']
            assert q_tid in objs
            objs[q_tid]['question'] = q['description']
            objs[q_tid]['question_id'] = traj_id
      objs = list(objs.values())
      all_images.append(image_encoded)
      all_objects.append(objs)
      all_ids.append(image_id)
    out_example = process_video_record(
        all_images, width, height, video_id, all_objects,
        video_captions, video_questions, all_ids)
    return [out_example]


def main(unused_argv):
  vidstg_data = json.load(gfile.GFile(FLAGS.vidstg_json, 'r'))
  video_ids = set(x['vid'] for x in vidstg_data)
  video_id_to_cont_id = {int(x): i + 1 for i, x in enumerate(sorted(video_ids))}
  print('num vidstg videos', len(video_ids))
  vid2stg_anns = {x: [] for x in video_ids}
  for i, x in enumerate(vidstg_data):
    vid2stg_anns[x['vid']].append((i + 1, x))

  convertor = ConvertSeqExamples(
      vidor_json_path=FLAGS.vidor_json_path,
      vid2stg_anns=vid2stg_anns,
      output_fps=FLAGS.output_fps,
      video_id_to_cont_id=video_id_to_cont_id,
      video_max_len=FLAGS.video_max_len)

  # init output files
  assert '@' in FLAGS.output_path
  output_path_base, num_shards = FLAGS.output_path.split('@')
  num_shards = int(num_shards)
  shard_id = 0
  output_path_pattern = output_path_base + '-{:05d}-of-{:05d}'
  output_path = output_path_pattern.format(shard_id, num_shards)
  num_examples_per_shard = (len(video_ids) - 1) // num_shards
  print('num_examples_per_shard', num_examples_per_shard)
  print('Writing to', output_path)
  writer = tf.io.TFRecordWriter(output_path)

  raw_ds = tf.data.TFRecordDataset(
      decode_sharded_names(FLAGS.vidor_tfrecord_path))
  raw_data_iter = iter(raw_ds)
  count = 0
  count_in_shard = 0
  while True:
    try:
      raw_data = next(raw_data_iter)
    except StopIteration:
      break
    data = tf.train.SequenceExample.FromString(raw_data.numpy())
    new_data_list = convertor.process(data)
    for new_data in new_data_list:
      writer.write(new_data.SerializeToString())
    count += len(new_data_list)
    count_in_shard += len(new_data_list)
    if count_in_shard >= num_examples_per_shard and (shard_id < num_shards - 1):
      writer.close()
      shard_id += 1
      output_path = output_path_pattern.format(shard_id, num_shards)
      print(f'Shard {shard_id - 1} done. Processed examples', count_in_shard)
      print('Writing to', output_path)
      count_in_shard = 0
      writer = tf.io.TFRecordWriter(output_path)
  writer.close()
  print('Num processed examples', count)


if __name__ == '__main__':
  app.run(main)
