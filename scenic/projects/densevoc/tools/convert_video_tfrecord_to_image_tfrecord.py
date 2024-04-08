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

r"""Create json file for evaluation from tfrecord.

This scripts create Visual-Genome format dense object captioning (in images)
TFRecord from a TFrecord for videos. This is used for mAP evaluation or
training the per-frame captioning + tracker baseline.

Before running this script, please follow the instructions in
`tools/build_vidstg_tfrecord.py` to build the video TFrecord.

Run:

```
python convert_video_tfrecord_to_image_tfrecord.py \
--input_tfrecord ~/Datasets/VidSTG/tfrecords/vidstg.video.max200f.caption.val.tfrecord@32 \
--output_tfrecord ~/Datasets/VidSTG/tfrecords/vidstg.image.max200f.caption.val.tfrecord@32 \
--total_videos 603
```

"""

import io

from absl import app
from absl import flags
import numpy as np
from PIL import Image
from scenic.projects.densevoc import input_utils
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('input_tfrecord', '', 'path to the tfrecord data.')
flags.DEFINE_string('output_tfrecord', '', 'Output path of tfrecord data')
flags.DEFINE_integer('total_videos', -1, 'total videos')


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


def process_record(image_encoded, width, height, image_id, anns):
  """Creates a sequence example from a list of dict."""
  bbox = np.asarray(
      [x['bbox'] for x in anns], dtype=np.float32).reshape(-1, 4)
  bbox[:, 2:] = bbox[:, 2:] + bbox[:, :2]  # [x0, y0, w, h] -> [x0, y0, x1, y1]
  bbox[:, [0, 2]] /= width
  bbox[:, [1, 3]] /= height
  # tfds builder use format [y0, x0, y1, x1]
  bbox[:, [0, 1]], bbox[:, [2, 3]] = bbox[:, [1, 0]], bbox[:, [3, 2]]

  feature = {
      'image': image_encoded,
      'img_id': _int64_feature([image_id]),
      'regions/bbox': _float_feature(bbox.flatten()),
      'regions/id': _int64_feature(np.asarray(
          [0 for _ in anns], dtype=np.int64)),
      'regions/track_id': _int64_feature(np.asarray(
          [x['track_id'] for x in anns], dtype=np.int64)),
      'regions/phrase': _bytes_feature(
          [str_to_bytes(x['caption']) for x in anns]),
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  return example


def numpy_to_encoded(image_np):
  # Convert the NumPy array image to a PIL.Image object
  image_pil = Image.fromarray(image_np)

  # Save the PIL.Image object to a BytesIO buffer in PNG format
  buffer = io.BytesIO()
  image_pil.save(buffer, format='PNG')
  buffer.seek(0)

  # Convert the image buffer to a byte string
  image_bytes = buffer.getvalue()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))


def main(unused_argv):
  ds = tf.data.TFRecordDataset(decode_sharded_names(FLAGS.input_tfrecord))
  ds = ds.map(
      lambda x: tf.io.parse_sequence_example(  # pylint: disable=g-long-lambda
          x,
          sequence_features=input_utils.densecap_sequence_feature_description,
          context_features=input_utils.densecap_context_feature_description))
  ds = ds.map(
      lambda x, y, _:  # pylint: disable=g-long-lambda
      input_utils.decode_and_sample_video_example(
          x, y, _, num_frames=-1, temporal_stride=1))
  data_iter = iter(ds)
  ann_count = 0
  num_videos = 0

  num_shards = int(FLAGS.output_tfrecord[FLAGS.output_tfrecord.find('@') + 1:])
  output_path = FLAGS.output_tfrecord[:FLAGS.output_tfrecord.find('@')]
  shard_id = 0
  output_path_pattern = output_path + '-{:05d}-of-{:05d}'
  output_path = output_path_pattern.format(shard_id, num_shards)
  num_examples_per_shard = (FLAGS.total_videos - 1) // num_shards + 1
  print('Writing to', output_path)
  writer = tf.io.TFRecordWriter(output_path)

  while True:
    try:
      num_videos += 1
      if num_videos % 100 == 0:
        print(num_videos)
      data = next(data_iter)
    except:  # pylint: disable=bare-except
      break
    images = data['images'].numpy()
    image_ids = data['image_ids'].numpy()
    num_frames, height, width = images.shape[:3]
    video_boxes = data['boxes'].numpy()
    video_track_ids = data['track_ids'].numpy()
    video_captions = data['captions'].numpy()
    for i in range(num_frames):
      image_id = image_ids[i]
      image = images[i]
      boxes = video_boxes[i]
      phrases = video_captions[i]
      track_ids = video_track_ids[i]
      objs = []
      for box, phrase, track_id in zip(boxes, phrases, track_ids):
        if box.max() == 0:
          break
        y0, x0, y1, x1 = box
        bbox = [x0 * width, y0 * height, (x1 - x0) * width, (y1 - y0) * height]
        ann_count += 1
        ann = {
            'id': ann_count,
            'image_id': int(image_id),
            'bbox': bbox,
            'caption': phrase.decode('utf-8'),
            'track_id': int(track_id),
        }
        objs.append(ann)
      image_encoded = numpy_to_encoded(image)
      record = process_record(
          image_encoded, width, height, image_id, objs)
      writer.write(record.SerializeToString())
    if num_videos % num_examples_per_shard == 0:
      writer.close()
      shard_id += 1
      output_path = output_path_pattern.format(shard_id, num_shards)
      print('Writing to', output_path)
      writer = tf.io.TFRecordWriter(output_path)
  writer.close()


if __name__ == '__main__':
  app.run(main)
