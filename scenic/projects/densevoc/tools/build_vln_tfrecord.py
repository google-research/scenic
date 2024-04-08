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

r"""Creates TFRecord files for Video Localized Narratives.

Download the "Video Narrative Grounding" annotation from the Video Localized
Narrative homepage: https://google.github.io/video-localized-narratives/

Download the UVO videos and annotations from its homepage:
https://sites.google.com/corp/view/unidentified-video-object/dataset#h.kpl7yyecv660

Follow the UVO instruction to run the `video2frame.py` tool to extract frames.
Assuming the frames are extracted in
~/Datasets/UVO/uvo_videos_sparse_frames_annotated_*/

Run the follow command with the corresponding paths to create UVO TFRecords:

```
mkdir ~/Datasets/VLN/tfrecords
python build_vln_tfrecord.py -- \
--ann_path ~/Datasets/VLN/vng/UVO_VNG/meta_expressions/sparse_val/meta_expressions.json \
--uvo_extra_ann_path ~/Datasets/VLN/vng/UVO_VNG/extra_masks/sparse_val/extra_masks.json \
--uvo_ann_path  ~/Datasets/UVO/UVOv1.0/VideoSparseSet/UVO_sparse_val_video.json \
--image_dir ~/Datasets/UVO/uvo_videos_sparse_frames_annotated_val/ \
--output_path ~/Datasets/VLN/tfrecords/vng_uvo_sparse_val.tfrecord@32

python build_vln_tfrecord.py -- \
--ann_path ~/Datasets/VLN/vng/UVO_VNG/meta_expressions/sparse_train/meta_expressions.json \
--uvo_extra_ann_path ~/Datasets/VLN/vng/UVO_VNG/extra_masks/sparse_train/extra_masks.json \
--uvo_ann_path ~/Datasets/UVO/UVOv1.0/VideoSparseSet/UVO_sparse_train_video.json \
--image_dir ~/Datasets/UVO/uvo_videos_sparse_frames_annotated_train/ \
--output_path ~/Datasets/VLN/tfrecords/vng_uvo_sparse_train.tfrecord@32
```

"""

import json

from absl import app
from absl import flags
import numpy as np
from pycocotools import mask as mask_utils
import tensorflow as tf
from tensorflow.io import gfile


FLAGS = flags.FLAGS

flags.DEFINE_string('ann_path', '', 'Path to input json.')
flags.DEFINE_string('uvo_ann_path', '', 'Path to uvo json.')
flags.DEFINE_string('uvo_extra_ann_path', '', 'Path to uvo json.')
flags.DEFINE_string('image_dir', '', 'Path to images.')
flags.DEFINE_string('output_path', '', 'output sstable')

NUM_IMAGES = 3


def str_to_bytes(string):
  return string.encode('utf-8')


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def process_video_record(
    video_info, video_key, annotations, objid2ann, id2contid, image_dir):
  """Creates a sequence example from a list of dict."""
  expressions = annotations['expressions']
  actor_narratives = annotations['actor_narratives']
  actorid2caption = {x['actor_idx']: x['description'] for x in actor_narratives}
  actorid2name = {x['actor_idx']: x['actor_name'] for x in actor_narratives}
  isbackground = {
      x['actor_idx']: x['actor_name'] == 'background' for x in actor_narratives}
  video_id = video_info['id']
  file_names = video_info['file_names']
  height, width = video_info['height'], video_info['width']

  images = []
  boxes, track_ids, captions = [], [], []
  for i in range(NUM_IMAGES):
    file_name = image_dir + '/' + file_names[i]
    img_string = gfile.GFile(file_name, 'rb').read()
    images.append(_bytes_feature([img_string]))
    boxes_frame, ids_frame, caption_frame = [], [], []
    used_caption = {x: False for x in actorid2caption}
    appeared = set()
    for k in sorted(expressions):
      ann = expressions[k]
      obj_id = ann['obj_id']
      obj_ann = objid2ann[obj_id]
      actor_id = ann['narrative_actor_idx']
      if not used_caption[actor_id] and not isbackground[actor_id] and (
          ann['noun_phrase_end_idx'] <= len(actorid2name[actor_id]) + 5
      ):
        caption = actorid2caption[actor_id]
        used_caption[actor_id] = True
      else:
        caption = ''
      if actor_id in appeared:
        continue
      appeared.add(actor_id)
      if 'bboxes' in obj_ann:
        bbox = obj_ann['bboxes'][i]
        if bbox is None:
          continue
      else:
        mask = obj_ann['segmentations'][i]
        if mask is None:
          continue
        bbox = mask_utils.toBbox(mask)
      boxes_frame.append(bbox)
      ids_frame.append(id2contid[obj_id])
      caption_frame.append(caption)
    bbox = np.asarray(boxes_frame).reshape(-1, 4)
    # [x0, y0, w, h] -> [x0, y0, x1, y1]
    bbox[:, 2:] = bbox[:, 2:] + bbox[:, :2]
    bbox[:, [0, 2]] /= width
    bbox[:, [1, 3]] /= height
    # tfds builder use format [y0, x0, y1, x1]
    bbox[:, [0, 1]], bbox[:, [2, 3]] = bbox[:, [1, 0]], bbox[:, [3, 2]]
    boxes.append(_float_feature(bbox.flatten()))
    track_ids.append(_int64_feature(np.asarray(ids_frame, dtype=np.int64)))
    captions.append(_bytes_feature([str_to_bytes(x) for x in caption_frame]))

  feature = {
      'video_id': _int64_feature([video_id]),
      'ytid': _bytes_feature([str_to_bytes(video_key)]),
      'image_ids': _int64_feature(
          [video_id * NUM_IMAGES + i for i in range(NUM_IMAGES)]),
  }
  seq_feature = {
      'image/encoded': tf.train.FeatureList(feature=images),
      'objects/bbox': tf.train.FeatureList(feature=boxes),
      'objects/track_id': tf.train.FeatureList(feature=track_ids),
      'objects/caption': tf.train.FeatureList(feature=captions),
  }
  example = tf.train.SequenceExample(
      context=tf.train.Features(feature=feature),
      feature_lists=tf.train.FeatureLists(feature_list=seq_feature))
  return example


def main(unused_argv):
  print('read inputs.')
  anns = json.load(gfile.GFile(FLAGS.ann_path, 'r'))
  uvo_anns = json.load(gfile.GFile(FLAGS.uvo_ann_path, 'r'))
  uvo_extra_anns = json.load(gfile.GFile(FLAGS.uvo_extra_ann_path, 'r'))
  print('original annotations', len(uvo_anns['annotations']))
  uvo_anns['annotations'].extend(uvo_extra_anns['annotations'])
  print('with extra annotations', len(uvo_anns['annotations']))
  objid2ann = {x['id']: x for x in uvo_anns['annotations']}
  id2contid = {x['id']: i + 1 for i, x in enumerate(
      sorted(uvo_anns['annotations'], key=lambda x: x['id']))}
  ytid2videoid = {x['ytid']: x['id'] for x in uvo_anns['videos']}
  videoid2ann = {x['id']: x for x in uvo_anns['videos']}
  num_shards = int(FLAGS.output_path[FLAGS.output_path.find('@') + 1:])
  output_path = FLAGS.output_path[:FLAGS.output_path.find('@')]
  num_examples = 0
  shard_id = 0
  output_path_pattern = output_path + '-{:05d}-of-{:05d}'
  output_path = output_path_pattern.format(shard_id, num_shards)
  num_examples_per_shard = (len(anns['videos']) - 1) // num_shards + 1
  print('Writing to', output_path)
  writer = tf.io.TFRecordWriter(output_path)
  for k in sorted(anns['videos']):
    ann = anns['videos'][k]
    num_examples += 1
    video_id = ytid2videoid[k]
    video_info = videoid2ann[video_id]
    record = process_video_record(
        video_info, k, ann, objid2ann, id2contid, FLAGS.image_dir)
    writer.write(record.SerializeToString())
    if num_examples % num_examples_per_shard == 0:
      writer.close()
      shard_id += 1
      output_path = output_path_pattern.format(shard_id, num_shards)
      print('Writing to', output_path)
      writer = tf.io.TFRecordWriter(output_path)
  writer.close()

if __name__ == '__main__':
  app.run(main)
