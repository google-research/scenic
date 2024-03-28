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

r"""Build RefCOCO tfrecord from UNINEXT json files.

The UNINEXT preprocessed annotations are downloaded from here:
https://github.com/MasterBin-IIAU/UNINEXT/blob/master/assets/DATA.md#rec--res


python scenic/projects/pixel_llm/tools/build_uninext_ref_tfrecord.py \
--output_dir ~/Datasets/PixelLLM/uninext_data \
--ann_output_dir ~/Datasets/PixelLLM/uninext_data/annotations \
--coco_path ~/Projects/PixelLLM/coco/ \
--vg_path ~/Projects/PixelLLM/MDETR/GQA/images \
--flickr_path ~/Projects/PixelLLM/MDETR/flickr30k/flickr30k-images
--ref_anno_path ~/Projects/PixelLLM/UNINEXT/annotations

"""

import io
import json
import os

from absl import app
from absl import flags
import numpy as np
from PIL import Image
from pycocotools import mask as mask_api
from pycocotools.coco import COCO
import tensorflow as tf
from tensorflow.io import gfile
import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'output_dir',
    '',
    'Output path of TFRecord',
)
flags.DEFINE_string(
    'ann_output_dir',
    '',
    'Output path of annotations',
)
flags.DEFINE_string(
    'ref_anno_path',
    '',
    'Refcoco annotation path',
)
flags.DEFINE_string(
    'vg_path',
    '',
    'path to VG images',
)
flags.DEFINE_string(
    'coco_path',
    '',
    'path to COCO dataset',
)
flags.DEFINE_string(
    'flickr_path',
    '',
    'path to Flickr images',
)
flags.DEFINE_boolean(
    'dryrun', False, 'print stats only'
)


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


def process_mask(annos, height, width):
  """Process mask."""
  for x in annos:
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
  mask = np.asarray([x['mask'] * 255 for x in annos], dtype=np.uint8)
  for x in annos:
    del x['mask']
  return numpy_to_encoded(mask)


def process_record(image_info, anns, image_path, clsid2contid):
  """Creates a sequence example from a list of dict."""
  file_name = image_info['file_name']
  img_path = os.path.join(image_path, file_name)
  img_string = gfile.GFile(img_path, 'rb').read()
  width, height = image_info['width'], image_info['height']
  bbox = np.asarray(
      [x['bbox'] for x in anns], dtype=np.float32).reshape(-1, 4)

  if 'segmentation' in anns[0]:
    mask = process_mask(anns, height, width)
  else:
    mask = None
  areas = bbox[:, 2] * bbox[:, 3]
  bbox[:, 2:] = bbox[:, 2:] + bbox[:, :2]
  bbox[:, [0, 2]] /= width
  bbox[:, [1, 3]] /= height
  # tfds builder use format [y0, x0, y1, x1]
  bbox[:, [0, 1]], bbox[:, [2, 3]] = bbox[:, [1, 0]], bbox[:, [3, 2]]

  refexp_id = []
  refexp_sent = []
  for ann in anns:
    refexp_id.extend(ann['refexp_id'])
    refexp_sent.extend(ann['refexp_sent'])
  ragged_row_lengths = [len(x['refexp_id']) for x in anns]

  feature = {
      'image': _bytes_feature([img_string]),
      'image/id': _int64_feature([image_info['id']]),
      'objects/id': _int64_feature(
          np.asarray([x['id'] for x in anns], dtype=np.int64)
      ),
      'objects/area': _int64_feature(np.asarray(areas, dtype=np.int64)),
      'objects/bbox': _float_feature(bbox.flatten()),
      'objects/label': _int64_feature(
          np.asarray(
              [clsid2contid[x['category_id']] for x in anns], dtype=np.int64
          )
      ),
      'objects/refexp/refexp_id/ragged_row_lengths_0': _int64_feature(
          np.asarray(ragged_row_lengths, dtype=np.int64)
      ),
      'objects/refexp/refexp_id/ragged_flat_values': _int64_feature(
          np.asarray(refexp_id, dtype=np.int64)
      ),
      'objects/refexp/sent/ragged_row_lengths_0': _int64_feature(
          np.asarray(ragged_row_lengths, dtype=np.int64)
      ),
      'objects/refexp/sent/ragged_flat_values': _bytes_feature(
          [str_to_bytes(x) for x in refexp_sent]
      ),
  }

  if mask is not None:
    feature['objects/mask'] = mask

  example = tf.train.Example(features=tf.train.Features(feature=feature))
  example = example.SerializeToString()
  return example


def convert_to_coco(data, data_source='coco'):
  """Converts to COCO format."""
  categories = [{'supercategory': 'object', 'id': 1, 'name': 'object'}]
  out_data = {'categories': categories, 'info': [], 'licenses': []}
  images = []
  annotations = []
  image_id_set = set()
  refexp_count = 0
  for dic in data:
    image_id = dic['image_id']
    if 'data_source' in dic:
      if dic['data_source'] != data_source:
        continue
    if image_id not in image_id_set:
      if data_source == 'coco':
        file_name = f'COCO_train2014_{image_id:012d}.jpg'
      elif data_source == 'vg' or data_source == 'flickr':
        file_name = f'{image_id}.jpg'
      else:
        raise ValueError('Unknown data source: %s' % data_source)
      image_id_set.add(image_id)
      img = {
          'file_name': file_name,
          'id': image_id,
          'width': dic['width'],
          'height': dic['height'],
      }
      images.append(img)
    if 'category_id' not in dic:
      category_id = 1
    else:
      category_id = categories[dic['category_id']]['id']
    ann = {
        'id': len(annotations),
        'image_id': image_id,
        'bbox': dic['bbox'],
        'category_id': category_id,
    }
    refexp_sent = dic['expressions']
    refexp_id = (refexp_count + np.arange(len(refexp_sent))).tolist()
    ann['refexp_id'] = refexp_id
    ann['refexp_sent'] = refexp_sent

    if 'mask' in dic:
      ann['segmentation'] = dic['mask']

    refexp_count += len(refexp_sent)

    annotations.append(ann)

  assert len(set(img['id'] for img in images)) == len(images)
  assert len(set(ann['image_id'] for ann in annotations)) == len(images)
  assert len(set(ann['id'] for ann in annotations)) == len(annotations)

  out_data['images'] = images
  out_data['annotations'] = annotations

  return out_data


def main(unused_argv):

  # pylint: disable=line-too-long
  ann_files = {
      'mixed_coco_train': os.path.join(FLAGS.ref_anno_path, 'mixed/instances.json'),
      'mixed_vg_train': os.path.join(FLAGS.ref_anno_path, 'mixed/instances.json'),
      'flickr_train': os.path.join(FLAGS.ref_anno_path, 'flickr30k/instances.json'),
  }
  # pylint: enable=line-too-long

  for dataset_name, json_file in ann_files.items():

    if 'flickr' in dataset_name:
      image_path = FLAGS.flickr_path
      data_source = 'flickr'
    elif 'mixed' in dataset_name:
      if 'coco' in dataset_name:
        data_source = 'coco'
        image_path = os.path.join(FLAGS.coco_path, 'train2014')
      elif 'vg' in dataset_name:
        data_source = 'vg'
        image_path = FLAGS.vg_path
      else:
        raise ValueError(f'Unknown data source: {dataset_name}')
    else:
      raise ValueError(f'Unknown data source: {dataset_name}')

    if 'train' in dataset_name:
      subset_name = 'train'
    elif 'val' in dataset_name:
      subset_name = 'val'
    else:
      raise ValueError(f'Unknown subset name: {dataset_name}')

    output_path = os.path.join(
        FLAGS.output_dir, dataset_name, f'{dataset_name}.tfrecord'
    )
    gfile.makedirs(os.path.dirname(output_path))

    print(f'Loadding {dataset_name} from {json_file}')
    data = json.load(gfile.GFile(json_file, 'r'))[subset_name]
    print('Load finished')
    data = convert_to_coco(data, data_source)

    coco = COCO(data)
    print(f'Finish loadding {dataset_name} from {json_file}')
    clsid2contid = {x: i for i, x in enumerate(sorted(coco.getCatIds()))}

    num_exampels = 0
    image_ids = coco.getImgIds()

    # filter out image ids without ann
    nonempty_image_ids = []
    for image_id in image_ids:
      if coco.getAnnIds(image_id):
        nonempty_image_ids.append(image_id)
    print(
        f'{dataset_name}: {len(nonempty_image_ids)} nonempty images of out'
        f' {len(image_ids)}'
    )
    image_ids = nonempty_image_ids

    num_shards = 2 ** (int(np.log2(len(image_ids)) - 10))
    num_shards = max(num_shards, 8)

    if FLAGS.dryrun:
      print('=' * 40)
      print(
          f'{dataset_name}: size: {len(image_ids)}, path:'
          f' {output_path}@{num_shards}'
      )
      print('=' * 40)
      continue

    shard_id = 0
    output_path_pattern = output_path + '-{:05d}-of-{:05d}'
    output_path = output_path_pattern.format(shard_id, num_shards)
    num_examples_per_shard = (len(image_ids) - 1) // num_shards + 1

    writer = tf.io.TFRecordWriter(output_path)

    for image_id in tqdm.tqdm(image_ids):
      img = coco.loadImgs(image_id)[0]
      ann_ids = coco.getAnnIds(image_id)
      anns = coco.loadAnns(ann_ids)
      assert anns, f'No anns found for image {image_id}'
      record = process_record(img, anns, image_path, clsid2contid)
      writer.write(record)
      num_exampels += 1
      if (num_exampels % num_examples_per_shard == 0):
        writer.close()
        shard_id += 1
        output_path = output_path_pattern.format(shard_id, num_shards)
        print('Writing to', output_path)
        writer = tf.io.TFRecordWriter(output_path)
    writer.close()

if __name__ == '__main__':
  app.run(main)
