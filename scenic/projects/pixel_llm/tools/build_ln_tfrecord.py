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

r"""Build Localized Narrative dataset tfrecord from jsonl.


python scenic/projects/pixel_llm/tools/build_ln_tfrecord.py \
--output_dir ~/Datasets/LN \
--ln_anno_path ~/Datasets/LN/annotations \
--coco_path ~/Datasets/coco

"""
import collections
import json
import os
from typing import NamedTuple, List

from absl import app
from absl import flags
import numpy as np
from scipy import interpolate
import tensorflow as tf
from tensorflow.io import gfile
import tqdm


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'output_dir',
    '',
    'Output path of TFRecords',
)
flags.DEFINE_string(
    'ln_anno_path',
    '',
    'Localized Narratives annotation path',
)
flags.DEFINE_string(
    'coco_path',
    '',
    'COCO dataset path',
)
flags.DEFINE_string('folder_name', 'tfrecords_with_bbox', '')
flags.DEFINE_integer('num_samples_per_trace', 16, '')


class TimedPoint(NamedTuple):
  x: float
  y: float
  t: float


class TimedUtterance(NamedTuple):
  utterance: str
  start_time: float
  end_time: float


class LocalizedNarrative(NamedTuple):
  """Represents a Localized Narrative annotation.

  Visit https://google.github.io/localized-narratives/index.html?file-formats=1
  for the documentation of each field.
  """
  dataset_id: str
  image_id: str
  annotator_id: int
  caption: str
  timed_caption: List[TimedUtterance]
  traces: List[List[TimedPoint]]
  voice_recording: str

  def __repr__(self):
    truncated_caption = self.caption[:60] + '...' if len(
        self.caption) > 63 else self.caption
    truncated_timed_caption = self.timed_caption[0].__str__()
    truncated_traces = self.traces[0][0].__str__()
    return (f'{{\n'
            f' dataset_id: {self.dataset_id},\n'
            f' image_id: {self.image_id},\n'
            f' annotator_id: {self.annotator_id},\n'
            f' caption: {truncated_caption},\n'
            f' timed_caption: [{truncated_timed_caption}, ...],\n'
            f' traces: [[{truncated_traces}, ...], ...],\n'
            f' voice_recording: {self.voice_recording}\n'
            f'}}')


def annotations_in_file(filename: str):
  """Yields all `LocalizedNarrative` dic in a given file.

  Args:
    filename: File to load the Localized Narratives from.

  Yields:
    LN dic.
  """
  with gfile.GFile(filename, 'rb') as file_handler:
    for line in file_handler:
      yield LocalizedNarrative(**json.loads(line))


def decode_sharded_names(paths):
  """Convert sharded file names into a list."""
  ret = []
  paths = paths.split(',')
  for name in paths:
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


def trace2coord(traces, timed_caption):
  """Computing the average location with intergral."""

  t_arr = np.array([trace['t'] for trace in traces])
  x_arr = np.array([trace['x'] for trace in traces])
  y_arr = np.array([trace['y'] for trace in traces])

  # get the indices that would sort t_arr
  sort_indices = np.argsort(t_arr)

  # sort t_arr, x_arr, y_arr using the indices
  t_arr = t_arr[sort_indices]
  x_arr = x_arr[sort_indices]
  y_arr = y_arr[sort_indices]

  num_points = FLAGS.num_samples_per_trace
  for dic in timed_caption:
    start_time = dic['start_time']
    end_time = dic['end_time']

    x_interpolator = interpolate.interp1d(
        t_arr, x_arr, fill_value='extrapolate'
    )
    y_interpolator = interpolate.interp1d(
        t_arr, y_arr, fill_value='extrapolate'
    )

    t_values = np.linspace(start_time, end_time, num=num_points)
    x_values = x_interpolator(t_values)
    y_values = y_interpolator(t_values)

    if t_values[-1] - t_values[0] < 1e-5:
      integral_x = np.mean(x_values)
      integral_y = np.mean(y_values)
    else:
      # calculate integral (average) x and y values
      integral_x = np.trapz(x_values, t_values) / (t_values[-1] - t_values[0])
      integral_y = np.trapz(y_values, t_values) / (t_values[-1] - t_values[0])

    dic['integral_x'] = integral_x
    dic['integral_y'] = integral_y
    dic['min_x'] = np.min(x_values)
    dic['min_y'] = np.min(y_values)
    dic['max_x'] = np.max(x_values)
    dic['max_y'] = np.max(y_values)
    dic['sampled_x'] = x_values.tolist()
    dic['sampled_y'] = y_values.tolist()

  return timed_caption


def str_to_bytes(string):
  return string.encode('utf-8')


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def is_valid_img_infos(image_id, img_infos):
  is_valid = False
  for img_info in img_infos:
    traces = []
    for trace in img_info.traces:
      traces.extend(trace)
    if len(traces) <= 1:
      print(f'no traces {len(traces)} for {image_id}', '=' * 10)
      continue
    is_valid = True
  return is_valid


def process_record(img_string, image_id, img_infos):
  """Creates a sequence example from a list of dict."""
  # captions = [img_info.caption for img_info in img_infos]

  timed_captions = []
  for img_info in img_infos:
    timed_caption = img_info.timed_caption
    traces = []
    for trace in img_info.traces:
      traces.extend(trace)
    if len(traces) <= 1:
      print(f'no traces {len(traces)} for {image_id}', '=' * 10)
      continue
    timed_caption = trace2coord(traces, timed_caption)
    # [N, 2]
    center_np = np.array(
        [[dic['integral_x'], dic['integral_y']] for dic in timed_caption]
    )
    # [N, 4]
    bbox_np = np.array([
        [dic['min_x'], dic['min_y'], dic['max_x'], dic['max_y']]
        for dic in timed_caption
    ])

    point_np = np.array([
        np.stack([dic['sampled_x'], dic['sampled_y']], axis=-1)
        for dic in timed_caption
    ])
    timed_captions.append({
        'center': center_np.flatten(),
        'bbox': bbox_np.flatten(),
        'point': point_np.flatten(),
        'utterance': [dic['utterance'] for dic in timed_caption],
        'string': img_info.caption
    })
  assert timed_captions, 'no timed captions'

  dataset_id = img_infos[0].dataset_id
  feature = {
      'image/encoded': _bytes_feature([img_string]),
      'image/id': _bytes_feature([str_to_bytes(image_id)]),
      'meta/dataset_id': _bytes_feature([str_to_bytes(dataset_id)]),
      'caption/string': _bytes_feature(
          [str_to_bytes(dic['string']) for dic in timed_captions]
      ),
  }
  feature_list = {
      'caption/center': tf.train.FeatureList(
          feature=[_float_feature(dic['center']) for dic in timed_captions]
      ),
      'caption/bbox': tf.train.FeatureList(
          feature=[_float_feature(dic['bbox']) for dic in timed_captions]
      ),
      'caption/point': tf.train.FeatureList(
          feature=[_float_feature(dic['point']) for dic in timed_captions]
      ),
      'caption/utterance': tf.train.FeatureList(
          feature=[
              _bytes_feature([str_to_bytes(u) for u in dic['utterance']])
              for dic in timed_captions
          ]
      ),
  }
  example = tf.train.SequenceExample(
      context=tf.train.Features(feature=feature),
      feature_lists=tf.train.FeatureLists(feature_list=feature_list),
  )
  example = example.SerializeToString()
  return example


def main(unused_argv):

  annotation_files = {
      'coco_train': [
          os.path.join(
              FLAGS.ln_anno_path,
              f'coco_train_localized_narratives-{i:05d}-of-00004.jsonl',
          )
          for i in range(4)
      ],
      'coco_val': [
          os.path.join(
              FLAGS.ln_anno_path, 'coco_val_localized_narratives.jsonl'
          )
      ],
  }

  image_dirs = {
      'coco_val': os.path.join(FLAGS.coco_path, 'val2017/'),
      'coco_train': os.path.join(FLAGS.coco_path, 'train2017/'),
  }

  for dataset_name, image_dir in image_dirs.items():
    shard_id = 0
    id2anno = collections.defaultdict(list)
    loco_ds = annotation_files[dataset_name]

    for annotation_file in tqdm.tqdm(
        loco_ds, desc=f'dataset: {dataset_name}', position=0
    ):
      annotations = annotations_in_file(annotation_file)
      for annotation in tqdm.tqdm(
          annotations, desc=f'process file: {annotation_file}', position=1
      ):
        image_id = annotation.image_id
        # num_images += 1
        id2anno[image_id].append(annotation)
    num_images = len(id2anno)

    num_shards = 2**(int(np.log2(num_images) - 10))
    num_shards = max(num_shards, 64)
    num_examples_per_shard = (num_images - 1) // num_shards + 1

    output_path = os.path.join(
        FLAGS.output_dir,
        dataset_name,
        FLAGS.folder_name,
        f'{dataset_name}.tfrecord-{shard_id:05d}-of-{num_shards:05d}',
    )
    gfile.makedirs(os.path.dirname(output_path))

    writer = tf.io.TFRecordWriter(output_path)
    num_exampels = 0

    pbar = tqdm.tqdm(
        id2anno.items(),
        desc=f'writing to {output_path}',
        total=len(id2anno),
    )
    for image_id, anno_list in pbar:
      image_path = os.path.join(image_dir, f'{int(image_id):012d}.jpg')
      img_string = gfile.GFile(image_path, 'rb').read()
      if not is_valid_img_infos(image_id, anno_list):
        continue
      record = process_record(img_string, image_id, anno_list)
      writer.write(record)
      num_exampels += 1
      if num_exampels % num_examples_per_shard == 0:
        shard_id += 1
        writer.close()
        output_path = os.path.join(
            FLAGS.output_dir,
            dataset_name,
            FLAGS.folder_name,
            f'{dataset_name}.tfrecord-{shard_id:05d}-of-{num_shards:05d}',
        )
        pbar.set_description(f'writing to {output_path}')
        writer = tf.io.TFRecordWriter(output_path)
    writer.close()
    print(f'Wrote {dataset_name} with {num_exampels} examples')
    if num_exampels != num_images:
      print(f'num_example {num_exampels} != num_images {num_images}')


if __name__ == '__main__':
  app.run(main)
