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

r"""Creates TFRecord Spoken moments-in-time dataset.

Download the Spoken moments-in-time videos and annotations from the website:
http://moments.csail.mit.edu/#download

Run the following commands with the path to the downloaded split file, caption
files, and the video paths:

```
mkdir ~/Datasets/S-MiT/tfrecords/

python build_smit_tfrecord.py -- \
--video_dir ~/Datasets/S-MiT/videos \
--transcription_dir ~/Datasets/S-MiT/transcriptions \
--split_file ~/Datasets/S-MiT/train_set.txt \
--output_path ~/Datasets/S-MiT/tfrecords/smit_train.tfrecord@1024
```


NOTE(zhouxy): this script is for external reproducibility only, and is NOT the
exact script we run during the training. Our original script uses internal
tools which run faster but can't be released. Users may integrate
this script with multi-threaded tools for speedup.

"""
import io
import os

from absl import app
from absl import flags

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow.io import gfile

flags.DEFINE_string('split_file', '', 'Path to split file.')
flags.DEFINE_string('video_dir', '', 'Path to videos.')
flags.DEFINE_string('transcription_dir', '', 'Path to transcriptions.')
flags.DEFINE_string('output_path', '', '')
flags.DEFINE_integer('max_frames', -1, '')

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


def construct_example(video_path, caption):
  """Creates a single tf.SequenceExample proto."""
  example = tf.train.SequenceExample()
  feature = example.context.feature
  feature['clip/data_path'].bytes_list.value.append(
      bytes(video_path, 'utf-8'))
  feature['video_id'].bytes_list.value.append(
      bytes(video_path, 'utf-8'))
  feature['caption/string'].bytes_list.value.append(
      bytes(caption, 'utf-8'))
  video = read_video(video_path, FLAGS.max_frames)
  image_encodeds = []
  for frame in video:
    image_encoded = numpy_to_encoded(frame)
    image_encodeds.append(image_encoded)
  example.feature_lists.feature_list['image/encoded'].feature.extend(
      image_encodeds)
  return example


def main(unused_argv):
  """Creates a tf.SequenceExample TFRecord."""
  print('reading split file.')
  split_file = gfile.GFile(FLAGS.split_file, 'r')
  videos = []
  for line in split_file:
    videos.append(line.strip())
  num_examples = len(videos)
  print('num_videos', num_examples)

  output_path = FLAGS.output_path
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
  for video_name in videos:
    count += 1
    video_path = os.path.join(FLAGS.video_dir, f'{video_name}.mp4')
    caption_path = os.path.join(FLAGS.transcription_dir, f'{video_name}.txt')
    caption = gfile.GFile(caption_path, 'r').readline().strip()
    example = construct_example(video_path, caption)
    writer.write(example.SerializeToString())
    if count % num_examples_per_shard == 0 and shard_id < num_shards - 1:
      writer.close()
      shard_id += 1
      output_path = output_path_pattern.format(shard_id, num_shards)
      print('Num processed examples', count)
      print('Writing to', output_path)
      writer = tf.io.TFRecordWriter(output_path)
  writer.close()
  print('Num processed examples', count)

if __name__ == '__main__':
  app.run(main)
