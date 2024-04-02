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

r"""Create ground truth json file for evaluation.

For each tfrecord of the validation set (e.g., anet_val), run:

python scenic/projects/streaming_dvc/tools/create_densecap_json_from_tfrecord.py \
--input /path/to/anet_val.tfrecord@64 \
--output /path/to/output/anet_val_vid2seqformat.json
"""

from collections.abc import Sequence
import json
from absl import app
from absl import flags
import tensorflow as tf
from tensorflow.io import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string('input', '', 'Input path')
flags.DEFINE_string('output', '', 'Output path')


def decode_sharded_names(paths, end=''):
  """Convert sharded file names into a list."""
  ret = []
  paths = paths.split(',')
  for name in paths:
    if '@' in name:
      idx = name.find('@')
      if end:
        num_shards = int(name[idx + 1:-len(end)])
      else:
        num_shards = int(name[idx + 1:])
      names = ['{}-{:05d}-of-{:05d}{}'.format(
          name[:idx], i, num_shards, end) for i in range(num_shards)]
      ret.extend(names)
    else:
      ret.append(name)
  return ret


def nonempty(x):
  return x and (x != ' ') and (x != '.') and (x != '\n')

sequence_feature_description = {
    'image/timestamp': tf.io.FixedLenSequenceFeature([], tf.int64),
}

context_feature_description = {
    'caption/string': tf.io.VarLenFeature(tf.string),
    'video/timestamps/start': tf.io.VarLenFeature(tf.int64),
    'video/timestamps/end': tf.io.VarLenFeature(tf.int64),
    'video/duration': tf.io.VarLenFeature(tf.int64),
    'split': tf.io.VarLenFeature(tf.int64),
    'media_id': tf.io.VarLenFeature(tf.string),
}


def decode_annotations(context_features, sequence_features, _):
  """Convert custom tfrecord into tfds builder format."""
  caption = tf.sparse.to_dense(context_features['caption/string'])
  start = tf.sparse.to_dense(context_features['video/timestamps/start'])
  end = tf.sparse.to_dense(context_features['video/timestamps/end'])
  duration = tf.sparse.to_dense(context_features['video/duration'])
  timestamps = sequence_features['image/timestamp']
  split = tf.sparse.to_dense(context_features['split'])
  # split = tf.ones((tf.shape(caption)[0]), dtype=tf.int64)
  media_id = tf.sparse.to_dense(context_features['media_id'])[0]
  return {
      'timestamps': timestamps, 'caption': caption,
      'start': start, 'end': end, 'duration': duration,
      'split': split,
      'media_id': media_id}


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  dataset_files = decode_sharded_names(FLAGS.input)
  ds = tf.data.TFRecordDataset(dataset_files)
  ds = ds.map(
      lambda x: tf.io.parse_sequence_example(
          x,
          context_features=context_feature_description,
          sequence_features=sequence_feature_description))
  ds = ds.map(decode_annotations)
  data_iter = iter(ds)

  annotations = {}

  i = 0
  while True:
    try:
      data = next(data_iter)
      i += 1
    except:  # pylint: disable=bare-except
      break
    media_id = data['media_id'].numpy().decode('utf-8')
    gt_timestamps = [
        [int(st), int(ed)]
        for st, ed in zip(data['start'].numpy(), data['end'].numpy())]
    gt_captions = [x.decode('utf-8') for x in data['caption'].numpy()]
    non_empty_inds = [i for i, x in enumerate(gt_captions) if nonempty(x)]
    gt_timestamps = [gt_timestamps[x] for x in non_empty_inds]
    gt_captions = [gt_captions[x] for x in non_empty_inds]
    splits = [int(x) for x in data['split'].numpy().tolist()]
    duration = int(data['duration'].numpy()[0])
    timestamps = [int(x) for x in data['timestamps'].numpy().tolist()]
    ann = {
        'gt_timestamps': gt_timestamps,
        'gt_captions': gt_captions,
        'splits': splits,
        'duration': duration,
        'timestamps': timestamps,
    }
    if media_id in annotations:
      print('Duplicate media_id', media_id)
      print('Annotation 1:', annotations[media_id])
      print('Annotation 2:', ann)
    if i % 100 == 0:
      print(f'Processed {i} examples.')
    annotations[media_id] = ann

  print(f'Processed {i} examples. {len(annotations)} valid annotations.')
  json.dump(annotations, gfile.GFile(FLAGS.output, 'w'))

if __name__ == '__main__':
  app.run(main)
