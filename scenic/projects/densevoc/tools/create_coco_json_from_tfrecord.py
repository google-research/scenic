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

r"""Create coco format json files for evaluation from tfrecords.

This scripts create COCO-format annotation jsons from a TFrecord.
The json is used for mAP and CHOTA evaluation.

Before running this script, please follow the instructions in
`tools/build_vidstg_tfrecord.py` and `tools/build_vln_tfrecord.py` to build
the video TFrecord.

```
mkdir ~/Datasets/VidSTG/annotations

python create_coco_json_from_tfrecord.py -- \
--input_tfrecord ~/Datasets/VidSTG/tfrecords/vidstg.video.max200f.caption.val.tfrecord@32 \
--output_json ~/Datasets/VidSTG/annotations/vidstg_max200f_val_coco_format.json

mkdir ~/Datasets/VLN/annotations

python create_coco_json_from_tfrecord.py -- \
--input_tfrecord ~/Datasets/VLN/tfrecords/vng_uvo_sparse_val.tfrecord@32 \
--output_json  ~/Datasets/VLN/annotations/vng_uvo_sparse_val_coco_format.json
```

"""

import json

from absl import app
from absl import flags
import numpy as np
from PIL import Image
from scenic.projects.densevoc import input_utils
import tensorflow as tf
from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string('input_tfrecord', '', 'path to the tfrecord data.')
flags.DEFINE_string('output_json', '', '')
flags.DEFINE_string('output_image_path', '', '')


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
  image_infos = []
  categories = [{'id': 1, 'name': 'object'}]
  annotations = []
  ann_count = 0
  num_videos = 0
  while True:
    try:
      num_videos += 1
      if num_videos % 100 == 0:
        print(num_videos)
      data = next(data_iter)
    except StopIteration:
      break
    if len(image_infos) % 1000 == 0:
      print(f'processed {len(image_infos)} images.')
    images = data['images'].numpy()
    image_ids = data['image_ids'].numpy()
    num_frames, height, width = images.shape[:3]
    video_boxes = data['boxes'].numpy()
    video_track_ids = data['track_ids'].numpy()
    video_captions = data['captions'].numpy()
    try:
      video_id = int(data['video_id'])
    except ValueError:
      video_id = int(num_videos)
    for i in range(num_frames):
      image_id = image_ids[i]
      image = images[i]
      if FLAGS.output_image_path:
        file_name = f'{FLAGS.output_image_path}/{image_id}.jpg'
        image_pil = Image.fromarray(image.astype(np.uint8)).convert('RGB')
        image_pil.save(file_name)
      image_info = {
          'file_name': f'{image_id}.jpg',
          'id': int(image_id),
          'height': height,
          'width': width,
          'video_id': video_id,
      }
      image_infos.append(image_info)
      boxes = video_boxes[i]
      phrases = video_captions[i]
      track_ids = video_track_ids[i]
      for box, phrase, track_id in zip(boxes, phrases, track_ids):
        if box.max() == 0:
          break
        bbox = [
            float(box[0]), float(box[1]),
            float(box[2] - box[0]), float(box[3] - box[1])]
        ann_count += 1
        ann = {
            'id': ann_count,
            'iscrowd': 0,
            'area': bbox[2] * bbox[3],
            'image_id': int(image_id),
            'category_id': 1,
            'bbox': bbox,
            'caption': phrase.decode('utf-8'),
            'track_id': int(track_id),
        }
        annotations.append(ann)
  ret = {
      'images': image_infos,
      'categories': categories,
      'annotations': annotations}
  for k, v in ret.items():
    print(k, len(v))
  json.dump(ret, gfile.GFile(FLAGS.output_json, 'w'))

if __name__ == '__main__':
  app.run(main)
