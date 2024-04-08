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

r"""Creates TFRecord VidOR dataset. Required for the VidSTG dataset.

Download the VidOR videos and annotations from the website:
https://xdshang.github.io/docs/vidor.html

Run the following commands with the path to the downloaded annotation json
files and the video paths:

```
mkdir ~/Datasets/VidOR/tfrecords/

python build_vidor_tfrecord.py -- \
--ann_path ~/Datasets/VidOR/annotations/validation.json \
--video_dir ~/Datasets/VidOR/videos/validation/ \
--output_path ~/Datasets/VidOR/tfrecords/vidor.validation.tfrecord@32

python build_vidor_tfrecord.py -- \
--ann_path ~/Datasets/VidOR/annotations/training.json \
--video_dir ~/Datasets/VidOR/videos/training/ \
--output_path ~/Datasets/VidOR/tfrecords/vidor.training.tfrecord@256
```

NOTE(zhouxy): this script is for external reproducibility only, and is NOT the
exact script we run for the paper. Our original script uses internal
tools which run faster but can't be released. Users may integrate
this script with multi-threaded tools for speedup.

"""
import io
import json
import os

from absl import app
from absl import flags

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# from tensorflow.core.example import example_pb2
from tensorflow.io import gfile

flags.DEFINE_string('ann_path', '', 'Path to input json directory.')
flags.DEFINE_string('video_dir', '', 'Path to videos.')
flags.DEFINE_string('output_path', '', '')

FLAGS = flags.FLAGS


def numpy_to_encoded(image_np):
  image_pil = Image.fromarray(image_np)
  buffer = io.BytesIO()
  image_pil.save(buffer, format='jpeg')
  buffer.seek(0)
  image_bytes = buffer.getvalue()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))


def read_video(path, max_frames=-1):
  """Read a video numpy array from a path."""
  cap = cv2.VideoCapture(path)
  num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  print(path, num_frames)
  if max_frames > 0 and num_frames > max_frames:
    inds = set(np.linspace(0, num_frames - 1, max_frames).astype(
        np.int32).tolist())
  else:
    inds = set(np.arange(num_frames).tolist())
  frames = []
  frame_idx = 0
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    if frame_idx in inds:
      frames.append(frame[..., ::-1])  # OpenCV loads video in BGR order
    frame_idx += 1
  frames = np.asarray(frames)
  return frames


def construct_example(feat):
  """Creates a single tf.SequenceExample proto."""
  example = tf.train.SequenceExample()
  feature = example.context.feature
  feature['data_path'].bytes_list.value.append(
      bytes(feat['video_path'], 'utf-8'))
  feature['video_id'].bytes_list.value.append(
      bytes(feat['video_id'], 'utf-8'))
  feature['dataset_name'].bytes_list.value.append(
      bytes(feat['dataset_name'], 'utf-8'))
  feature['fps'].float_list.value.append(feat['fps'])
  feature['frame_count'].int64_list.value.append(feat['frame_count'])
  feature['width'].int64_list.value.append(feat['width'])
  feature['height'].int64_list.value.append(feat['height'])
  video = read_video(feat['video_path'])
  image_encodeds = []
  for frame in video:
    image_encoded = numpy_to_encoded(frame)
    image_encodeds.append(image_encoded)
  example.feature_lists.feature_list['image/encoded'].feature.extend(
      image_encodeds)
  return example


def create_dataset(ann_path, video_dir, output_path):
  """Creates a tf.SequenceExample TFRecord."""
  print('reading inputs.')
  anns = json.load(gfile.GFile(ann_path, 'r'))
  num_examples = len(anns)
  print(f'constructing {num_examples} examples.')

  assert '@' in output_path
  output_path_base, num_shards = output_path.split('@')
  num_shards = int(num_shards)
  shard_id = 0
  output_path_pattern = output_path_base + '-{:05d}-of-{:05d}'
  output_path = output_path_pattern.format(shard_id, num_shards)
  num_examples_per_shard = (num_examples - 1) // num_shards
  print('Writing to', output_path)
  writer = tf.io.TFRecordWriter(output_path)

  count = 0
  for _, feat in anns.items():
    count += 1
    video_path = os.path.join(video_dir, feat['video_path'])
    feat['video_path'] = video_path
    feat['dataset_name'] = 'vidor'
    example = construct_example(feat)
    writer.write(example.SerializeToString())
    if (count % num_examples_per_shard == 0) and shard_id < num_shards - 1:
      writer.close()
      shard_id += 1
      output_path = output_path_pattern.format(shard_id, num_shards)
      print('Writing to', output_path)
      writer = tf.io.TFRecordWriter(output_path)
  writer.close()
  print('Num processed examples', count)


def main(unused_argv):
  create_dataset(FLAGS.ann_path, FLAGS.video_dir, FLAGS.output_path)


if __name__ == '__main__':
  app.run(main)
