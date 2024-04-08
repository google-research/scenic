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

"""Input pipeline for videos."""

import jax
import jax.numpy as jnp
from scenic.projects.baselines.centernet import input_pipeline as centernet_input_pipeline
from scenic.projects.baselines.centernet import transforms
from scenic.projects.densevoc import transforms as custom_transforms
import tensorflow as tf
import tensorflow_datasets as tfds


PRNGKey = jnp.ndarray
ALL_LOSSES = ['det', 'objcap', 'track', 'trackcap', 'vidcap', 'imagecap']
# pylint: disable=g-long-lambda


def make_resize_crop_transforms(
    image_set,
    scale_range=(0.1, 2.0),
    crop_size=1024):
  """EfficientNet style resize and crop augmentation.

  Different from the default detection augmentation, this function in addition
    processes text tokens using 'FixedSizeCropWithAdditionalKeys'.

  Args:
    image_set: 'train' or 'validation'
    scale_range: list of integers. Sizes of the shorter edge.
    crop_size: integer. Size of the longer edge.
  Returns:
    The data-augmentation functions.
  """
  init_padding_mask = transforms.InitPaddingMask()
  if image_set == 'train':
    return transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomRatioResize(scale_range, crop_size),
         custom_transforms.FixedSizeCropWithAdditionalKeys(
             crop_size, additional_keys=('text_tokens',)),
         init_padding_mask])
  elif image_set == 'validation':
    return transforms.Compose(
        [transforms.Resize(crop_size, max_size=crop_size),
         init_padding_mask])
  else:
    raise ValueError(f'Unknown image_set: {image_set}')

vg_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'img_id': tf.io.FixedLenFeature([], tf.int64),
    'regions/bbox': tf.io.VarLenFeature(dtype=tf.float32),
    'regions/id': tf.io.VarLenFeature(dtype=tf.int64),
    'regions/phrase': tf.io.VarLenFeature(dtype=tf.string),
}

# We use different eval and train pipeline as eval sometimes contain additional
#   information, e.g., grounding sentence.
eval_sequence_feature_description = {
    'image/encoded': tf.io.FixedLenSequenceFeature([], tf.string),
    'objects/bbox': tf.io.VarLenFeature(tf.float32),
    'objects/track_id': tf.io.VarLenFeature(tf.int64),
    'objects/caption': tf.io.VarLenFeature(tf.string),
}

eval_context_feature_description = {
    'bbox': tf.io.VarLenFeature(dtype=tf.float32),
    'caption': tf.io.VarLenFeature(dtype=tf.string),
    'video_id': tf.io.VarLenFeature(dtype=tf.int64),
    'frame_ids': tf.io.VarLenFeature(dtype=tf.int64),
    'image_ids': tf.io.VarLenFeature(dtype=tf.int64),
}


def decode_eval_video_example(
    context_feature, seq_feature, _, with_objects=True):
  """Convert custom tfrecord into tfds builder format."""
  images = tf.map_fn(
      lambda x: tf.image.decode_jpeg(x, channels=3),
      seq_feature['image/encoded'], back_prop=False, dtype=tf.uint8)
  ret = {
      'images': images,
      'caption': tf.sparse.to_dense(context_feature['caption']),  # video cap.
      'video_id': tf.sparse.to_dense(context_feature['video_id']),
      'image_ids': tf.sparse.to_dense(context_feature['image_ids']),
      'frame_ids': tf.sparse.to_dense(context_feature['frame_ids']),
  }
  if with_objects:
    bbox = tf.map_fn(
        tf.sparse.to_dense,
        seq_feature['objects/bbox'], back_prop=False, dtype=tf.float32)
    bbox = tf.reshape(bbox, [tf.shape(seq_feature['image/encoded'])[0], -1, 4])
    track_id = tf.map_fn(
        tf.sparse.to_dense,
        seq_feature['objects/track_id'], back_prop=False, dtype=tf.int64)
    object_caption = tf.map_fn(
        tf.sparse.to_dense,
        seq_feature['objects/caption'], back_prop=False, dtype=tf.string)
    ret.update({
        'boxes': bbox,
        'track_ids': track_id,
        'captions': object_caption,  # object caption.
    })
  return ret


def decode_eval_annotations(
    example,
    tokenizer,
    max_boxes=100,
    max_text_tokens=40,
    max_frames=200,
    with_caption_tokens=True,
    with_objects=True,
    temporal_stride=1,
    ):
  """Convert custom tfrecord into training pipeline builder format."""
  images = example['images'][::temporal_stride][:max_frames]
  size = tf.cast(tf.shape(images)[1:3], dtype=tf.int32)
  annotations = {
      'inputs': images,
      'label': {
          'video_id': example['video_id'][0],
          'orig_size': size,
          'size': tf.identity(size),
          'frame_ids': example['frame_ids'][::temporal_stride][:max_frames],
          'image_ids': example['image_ids'][::temporal_stride][:max_frames],
      },
  }
  if with_objects:
    boxes = tf.map_fn(
        lambda x: centernet_input_pipeline.decode_boxes(
            x, tf.shape(images)[1:3]),
        example['boxes'], back_prop=False, dtype=tf.float32)
    boxes = boxes[::temporal_stride][:max_frames, :max_boxes]
    annotations['label'].update({
        'boxes': boxes,
        'track_ids': example['track_ids'][
            ::temporal_stride][:max_frames, :max_boxes],
        'text_tokens': tf.map_fn(
            lambda x: tokenizer.string_tensor_to_indices(
                x, prepend_bos=True, append_eos=True,
                max_num_tokens=max_text_tokens),
            example['captions'][::temporal_stride][:max_frames, :max_boxes],
            back_prop=False, fn_output_signature=tf.int32),
    })
  if with_caption_tokens:
    annotations['caption_tokens'] = tokenizer.string_tensor_to_indices(
        example['caption'],
        prepend_bos=True, append_eos=True,
        max_num_tokens=max_text_tokens)[0]
  else:
    annotations['caption_tokens'] = tf.zeros(
        (max_text_tokens), dtype=tf.int64)
  return annotations


def video_resize_max_size(features, size=256):
  """Resize video to a fixed max-size."""
  image = features['inputs']
  original_size = tf.shape(image)[1:3]
  new_size = transforms.get_size_with_aspect_ratio(
      original_size, size, max_size=size)
  rescaled_image = tf.image.resize(image, new_size)
  features['inputs'] = rescaled_image
  features['label']['size'] = tf.stack(new_size)
  if 'boxes' in features['label']:
    r_height = tf.cast((new_size[0] / original_size[0]), tf.float32)
    r_width = tf.cast((new_size[1] / original_size[1]), tf.float32)
    x0, y0, x1, y1 = tf.split(features['label']['boxes'], 4, axis=-1)
    features['label']['boxes'] = tf.concat(
        [x0 * r_width, y0 * r_height,
         x1 * r_width, y1 * r_height], axis=-1)
  return features


def load_video_val_tfds(
    batch_size,
    *,
    dataset_path,
    tokenizer,
    max_size=256,
    max_text_tokens=40,
    max_boxes=100,
    max_frames=200,
    temporal_stride=1,
    with_objects=True,
    dataset_format='full',
    ):
  """Loads a split of a video dataset using TensorFlow Datasets.

  Args:
    batch_size: int; The batch size returned by the data pipeline.
    dataset_path: string; path of the dataset; by default load from tfds
    tokenizer: tokenizer
    max_size: int; Maximum image size.
    max_text_tokens: int; max number of text tokens.
    max_boxes: int; used in padding bounding box shape
    max_frames: int max number of frames.
    temporal_stride: int;
    with_objects: bool; for compatibility with some datasets
    dataset_format: str. Currently support 'full' or 'videocap'

  Returns:
    A `tf.data.Dataset`, and dataset info.
  """
  assert dataset_format == 'full'
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48

  if 'tfrecord' in dataset_path:
    ds = tf.data.TFRecordDataset(
        centernet_input_pipeline.decode_sharded_names(dataset_path))
    ds = ds.shard(jax.process_count(), jax.process_index())
    ds = ds.map(
        lambda x: tf.io.parse_sequence_example(
            x,
            sequence_features=eval_sequence_feature_description,
            context_features=eval_context_feature_description))
  else:
    raise ValueError('Unsupported dataset format: %s' % dataset_path)
  with_caption_tokens = 'tubedetr' in dataset_path  # For grounding

  ds = ds.with_options(options)
  ds = ds.map(
      lambda x, y, _: decode_eval_video_example(
          x, y, _, with_objects=with_objects),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(
      lambda x: decode_eval_annotations(  # pylint: disable=g-long-lambda
          x, tokenizer, max_frames=max_frames, max_boxes=max_boxes,
          with_caption_tokens=with_caption_tokens,
          with_objects=with_objects,
          temporal_stride=temporal_stride))

  padded_shapes = {
      'inputs': [max_frames, max_size, max_size, 3],
      'label': {
          'video_id': [],
          'orig_size': [2,],
          'size': [2,],
          'frame_ids': [max_frames,],
          'image_ids': [max_frames,],
      },
      'caption_tokens': [max_text_tokens,],
  }
  if with_objects:
    padded_shapes['label'].update({
        'boxes': [max_frames, max_boxes, 4],
        'text_tokens': [max_frames, max_boxes, max_text_tokens],
        'track_ids': [max_frames, max_boxes],
    })
  preprocess_fn = lambda x: video_resize_max_size(x, max_size)
  ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # First batch then repeat.
  ds = ds.padded_batch(
      batch_size, padded_shapes=padded_shapes, drop_remainder=False)
  ds = ds.repeat()

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds, {}


densecap_sequence_feature_description = {
    'image/encoded': tf.io.FixedLenSequenceFeature([], tf.string),
    'objects/bbox': tf.io.VarLenFeature(tf.float32),
    'objects/track_id': tf.io.VarLenFeature(tf.int64),
    'objects/segment_id': tf.io.VarLenFeature(tf.int64),
    'objects/caption': tf.io.VarLenFeature(tf.string),
}

densecap_context_feature_description = {
    'video_id': tf.io.VarLenFeature(dtype=tf.int64),
    'video/captions': tf.io.VarLenFeature(dtype=tf.string),
    'image_ids': tf.io.VarLenFeature(dtype=tf.int64),
}


def decode_and_sample_video_example(
    context_feature, seq_feature, _,
    num_frames=4, temporal_stride=1,
    ensure_sample_has_objects=True,
    track_id_key='objects/track_id'):
  """Convert custom tfrecord into tfds builder format."""
  images = seq_feature['image/encoded']
  bbox = tf.map_fn(
      tf.sparse.to_dense,
      seq_feature['objects/bbox'], back_prop=False, dtype=tf.float32)
  bbox = tf.reshape(bbox, [tf.shape(bbox)[0], -1, 4])
  track_id = tf.map_fn(
      tf.sparse.to_dense,
      seq_feature[track_id_key], back_prop=False, dtype=tf.int64)
  caption = tf.map_fn(
      tf.sparse.to_dense,
      seq_feature['objects/caption'], back_prop=False, dtype=tf.string)

  if num_frames > 0:
    max_frames = tf.shape(bbox)[0]
    sample_stride = tf.maximum(
        tf.minimum(temporal_stride, max_frames // num_frames), 1)
    max_offset = tf.maximum(max_frames - num_frames * sample_stride, 1)
    def local_get_inds(_):
      if temporal_stride > 0:  # sample a window with fixed stride
        offset = tf.random.uniform(
            (), maxval=max_offset, dtype=tf.int32)
        return tf.minimum(tf.range(
            offset, offset + num_frames * sample_stride,
            delta=temporal_stride), max_frames - 1)
      else:  # global uniform sample
        return tf.sort(tf.random.shuffle(tf.range(max_frames))[:num_frames])
    inds = local_get_inds(())
    if ensure_sample_has_objects:
      # Make sure the sampled video has at least one annotated object.
      inds = tf.while_loop(
          lambda x: tf.equal(tf.reduce_max(tf.gather(track_id, x)), 0),
          local_get_inds, [inds])[0]
    bbox = tf.gather(bbox, inds)
    track_id = tf.gather(track_id, inds)
    caption = tf.gather(caption, inds)
    images = tf.gather(images, inds)
  else:
    inds = tf.range(tf.shape(bbox)[0])
  images = tf.map_fn(
      lambda x: tf.image.decode_jpeg(x, channels=3),
      images, back_prop=False, dtype=tf.uint8)
  bbox = tf.map_fn(
      lambda x: centernet_input_pipeline.decode_boxes(x, tf.shape(images)[1:3]),
      bbox, back_prop=False, dtype=tf.float32)
  return {
      'images': images,
      'boxes': bbox,
      'track_ids': tf.cast(track_id, tf.int32),
      'captions': caption,
      'video_id': tf.sparse.to_dense(context_feature['video_id']),
      'video_captions': tf.sparse.to_dense(context_feature['video/captions']),
      'frame_inds': inds,
      'image_ids': tf.sparse.to_dense(context_feature['image_ids']),
  }


def decode_densecap_annotations(
    example,
    tokenizer,
    max_boxes=100,
    max_video_captions=100,
    max_image_captions=1,
    max_text_tokens=40,
    ):
  """Convert custom tfrecord into training pipeline builder format."""
  images = tf.cast(example['images'], tf.float32)
  boxes = example['boxes']
  boxes = boxes[:, :max_boxes]
  size = tf.cast(tf.shape(images)[1:3], dtype=tf.int32)
  video_caption_tokens = tokenizer.string_tensor_to_indices(
      example['video_captions'],
      prepend_bos=True, append_eos=True,
      max_num_tokens=max_text_tokens)
  video_caption_inds = tf.random.shuffle(
      tf.range(tf.shape(video_caption_tokens)[0]))[:max_video_captions]
  video_caption_tokens = tf.gather(video_caption_tokens, video_caption_inds)
  if 'image_captions' in example:
    image_captions_tokens = tf.map_fn(
        # pylint:disable=g-long-lambda
        lambda x: tokenizer.string_tensor_to_indices(
            x, prepend_bos=True, append_eos=True,
            max_num_tokens=max_text_tokens),
        # pylint:enable=g-long-lambda
        example['image_captions'][:, :max_image_captions],
        back_prop=False, fn_output_signature=tf.int32)
  else:
    image_captions_tokens = tf.zeros(
        (tf.shape(images)[0], max_image_captions, max_text_tokens), tf.int32)
  target = {
      'boxes': boxes,
      'text_tokens': tf.map_fn(
          # pylint:disable=g-long-lambda
          lambda x: tokenizer.string_tensor_to_indices(
              x, prepend_bos=True, append_eos=True,
              max_num_tokens=max_text_tokens),
          # pylint:enable=g-long-lambda
          example['captions'][:, :max_boxes],
          back_prop=False, fn_output_signature=tf.int32),
      'orig_size': size,
      'track_ids': tf.cast(example['track_ids'][:, :max_boxes], tf.int32),
      'size': tf.identity(size),
      'labels': tf.zeros(tf.shape(boxes)[:-1], dtype=tf.int32),
      'video_caption_tokens': video_caption_tokens,
      'frame_inds': example['frame_inds'],
      'image_caption_tokens': image_captions_tokens,
  }
  return {
      'inputs': images,
      'label': target,
      'loss_masks': {
          f'{k}_loss_mask': example[f'{k}_loss_mask'] for k in ALL_LOSSES}
  }

videocap_sequence_feature_description = {
    'image/encoded': tf.io.FixedLenSequenceFeature([], tf.string),
}

videocap_context_feature_description = {
    'caption/string': tf.io.VarLenFeature(tf.string),
    'video_id': tf.io.VarLenFeature(tf.string),
    'clip/data_path': tf.io.VarLenFeature(tf.string),
}


def get_inds(num_frames, max_frames, temporal_stride):
  sample_stride = tf.maximum(
      tf.minimum(temporal_stride, max_frames // num_frames), 1)
  max_offset = tf.maximum(max_frames - num_frames * sample_stride, 1)
  if temporal_stride > 0:  # sample a window with fixed stride
    offset = tf.random.uniform(
        (), maxval=max_offset, dtype=tf.int32)
    return tf.minimum(tf.range(
        offset, offset + num_frames * sample_stride,
        delta=temporal_stride), max_frames - 1)
  else:  # global uniform sample
    return tf.sort(tf.random.shuffle(tf.range(max_frames))[:num_frames])


def decode_videocap(
    context_feature, seq_feature, _,
    num_frames=6, temporal_stride=1, train=False):
  """Convert custom tfrecord into tfds builder format."""
  images = seq_feature['image/encoded']
  if num_frames > 0:
    max_frames = tf.shape(images)[0]
    if train:
      inds = get_inds(num_frames, max_frames, temporal_stride)
    else:
      stride = max_frames // num_frames
      inds = tf.range(num_frames) * stride
  else:
    inds = tf.range(tf.shape(images)[0])
  images = tf.gather(images, inds)
  images = tf.map_fn(
      lambda x: tf.image.decode_jpeg(x, channels=3),
      images, back_prop=False, dtype=tf.uint8)
  video_captions = tf.sparse.to_dense(context_feature['caption/string'])
  return {
      'images': images,
      'boxes': tf.zeros((num_frames, 0, 4), dtype=tf.float32),
      'captions': tf.constant([[] for _ in range(num_frames)], dtype=tf.string),
      'track_ids': tf.zeros((num_frames, 0), dtype=tf.int32),
      'video_id': tf.zeros([], dtype=tf.int32),
      'video_captions': video_captions,
      'image_ids': tf.zeros((num_frames,), dtype=tf.int32),
      'frame_inds': inds,
  }


def tf_float(t):
  return tf.cast(t, tf.float32)


def tf_int32(t):
  return tf.cast(t, tf.int32)


def get_aug_param(aug_ratio, h, w, size):
  ratio = tf.random.uniform(
      [], aug_ratio[0], aug_ratio[1], dtype=tf.float32)
  h = tf.cast(tf.cast(h, tf.float32) * ratio, tf.int32)
  w = tf.cast(tf.cast(w, tf.float32) * ratio, tf.int32)
  i = tf.random.uniform([], 0, h - tf.minimum(h, size) + 1, dtype=tf.int32)
  j = tf.random.uniform([], 0, w - tf.minimum(w, size) + 1, dtype=tf.int32)
  return tf.stack([h, w, i, j], axis=0)


def augment_image_annotation(image, bbox, h, w, size, param):
  """Apply data augmentation."""
  # resize
  new_size = tf_int32(param[:2])
  new_image = tf.image.resize(image, new_size)
  r_height = tf_float(new_size[0]) / tf_float(h)
  r_width = tf_float(new_size[1]) / tf_float(w)
  new_boxes = tf.stack(
      [bbox[:, 0] * r_width, bbox[:, 1] * r_height,
       bbox[:, 2] * r_width, bbox[:, 3] * r_height], axis=-1)
  # crop
  i, j = tf_int32(param[2]), tf_int32(param[3])
  hcrop = tf.minimum(new_size[0], size)
  wcrop = tf.minimum(new_size[1], size)
  new_image = new_image[i: i + hcrop, j: j + wcrop]
  new_image = tf.image.pad_to_bounding_box(
      new_image, 0, 0, size, size)
  new_boxes = new_boxes - tf_float(tf.expand_dims(
      tf.stack([j, i, j, i]), axis=0))
  new_boxes = tf.minimum(
      tf.reshape(new_boxes, [-1, 2, 2]),
      tf.reshape(tf_float(tf.stack([wcrop, hcrop])), [1, 1, 2]))
  new_boxes = tf.clip_by_value(new_boxes, 0, 1000000)
  new_boxes = tf.reshape(new_boxes, [-1, 4])
  area = tf.reduce_prod(new_boxes[:, 2:] - new_boxes[:, :2], axis=-1)
  # Mark invalid boxes as all zero. Otherwise training goes NaN.
  new_boxes = new_boxes * tf_float(area[:, None] > 0)
  return new_image, new_boxes


def pad_to(image, size):
  input_shape = tf.shape(image)
  padding = tf.math.maximum(size - input_shape, 0)
  paddings = tf.stack([[0, padding[i]] for i in range(len(padding))], axis=0)
  return tf.pad(image, paddings)


def pad_string_tensor(tensor, target_shape):
  input_shape = tf.shape(tensor)
  padding = tf.math.maximum(target_shape - input_shape, 0)
  paddings = tf.stack([[0, padding[i]] for i in range(len(padding))], axis=0)
  return tf.pad(tensor, paddings, constant_values='')


def decode_and_pad_vg_image(data, max_size, max_boxes, scale_range):
  """Augment an image into a video."""
  image = tf.io.decode_jpeg(data['image'])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  bbox = tf.reshape(
      tf.sparse.to_dense(data['regions/bbox']), [-1, 4])
  num_boxes = tf.shape(bbox)[0]
  bbox = centernet_input_pipeline.decode_boxes(bbox, (h, w))
  caption = tf.sparse.to_dense(data['regions/phrase'])
  param = get_aug_param(scale_range, h, w, max_size)
  image, bbox = augment_image_annotation(image, bbox, h, w, max_size, param)
  inds = tf.random.shuffle(tf.range(num_boxes))[:max_boxes]
  bbox = tf.gather(bbox, inds)
  caption = tf.gather(caption, inds)
  return {
      'images': pad_to(image, [max_size, max_size, 3]),
      'boxes': pad_to(bbox, [max_boxes, 4]),
      'captions': pad_string_tensor(caption, [max_boxes]),
      'track_ids': pad_to(
          tf.zeros((num_boxes,), tf.int32)[:max_boxes], [max_boxes]),
      'frame_inds': tf.zeros((), dtype=tf.int32),
      'image_ids': tf.zeros((), dtype=tf.int32),
  }


def add_zero_video_keys(data):
  data['video_id'] = tf.zeros((), dtype=tf.int32)
  data['video_captions'] = tf.constant([''], dtype=tf.string)
  return data


def convert_coco_format(data, max_size, max_boxes, scale_range):
  """Augment an image into a video."""
  image = data['image']
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  bbox = tf.reshape(data['objects']['bbox'], [-1, 4])
  bbox = centernet_input_pipeline.decode_boxes(bbox, (h, w))
  param = get_aug_param(scale_range, h, w, max_size)
  image, bbox = augment_image_annotation(image, bbox, h, w, max_size, param)
  keep = tf.where(
      tf.logical_and(
          tf.logical_not(data['objects']['is_crowd']),
          tf.logical_and(
              bbox[:, 2] > bbox[:, 0], bbox[:, 3] > bbox[:, 1])
      )
  )[:, 0]
  bbox = tf.gather(bbox, keep)
  num_boxes = tf.shape(bbox)[0]
  inds = tf.random.shuffle(tf.range(num_boxes))[:max_boxes]
  bbox = tf.gather(bbox, inds)
  caption = tf.zeros((max_boxes,), dtype=tf.string)
  return {
      'images': pad_to(image, [max_size, max_size, 3]),
      'boxes': pad_to(bbox, [max_boxes, 4]),
      'captions': pad_string_tensor(caption, [max_boxes]),
      'track_ids': pad_to(
          tf.zeros((num_boxes,), tf.int32)[:max_boxes], [max_boxes]),
      'frame_inds': tf.zeros((), dtype=tf.int32),
      'image_ids': tf.zeros((), dtype=tf.int32),
  }


def convert_coco_as_video(
    data, num_frames=6, size=384, aug_ratio=(0.1, 2.0)):
  """Augment an image into a video."""
  image_id = tf.zeros((), dtype=tf.int32)
  image = data['image']
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  bbox = tf.reshape(data['objects']['bbox'], [-1, 4])
  num_boxes = tf.shape(bbox)[0]
  bbox = centernet_input_pipeline.decode_boxes(bbox, (h, w))

  caption = tf.zeros((num_boxes,), dtype=tf.string)
  param1 = get_aug_param(aug_ratio, h, w, size)
  param2 = get_aug_param(aug_ratio, h, w, size)

  images, bboxes = [], []
  params = tf.linspace(tf_float(param1), tf_float(param2), num_frames)
  for t in range(num_frames):
    new_image, new_boxes = augment_image_annotation(
        image, bbox, h, w, size, params[t])
    new_boxes = new_boxes * tf_float(
        tf.logical_not(data['objects']['is_crowd']))[:, None]
    images.append(new_image)
    bboxes.append(new_boxes)
  images = tf.stack(images, axis=0)  # (T, H, W, 3)
  bboxes = tf.stack(bboxes, axis=0)  # (T, N, 4)
  track_ids = tf.broadcast_to(
      tf.range(num_boxes)[None], (num_frames, num_boxes)) + 1
  track_ids = track_ids * tf_int32(tf.reduce_max(bboxes, axis=-1) > 0)
  captions = tf.stack([caption for _ in range(num_frames)], axis=0)
  return {
      'images': images,
      'boxes': bboxes,
      'captions': captions,
      'track_ids': track_ids,
      'video_id': image_id,
      'video_captions': tf.constant([''], dtype=tf.string),
      'frame_inds': tf.zeros((num_frames), dtype=tf.int32),
      'image_ids': tf.stack([image_id for _ in range(num_frames)], axis=0),
  }


def convert_cococaption_format(data, max_size, max_boxes, max_image_captions):
  """Convert COCO captioning format to detection by adding zero objects."""
  image = data['image']
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  bbox = tf.zeros((0, 4), tf.float32)
  param = get_aug_param((1.0, 1.0), h, w, max_size)
  image, _ = augment_image_annotation(image, bbox, h, w, max_size, param)
  caption = tf.zeros((max_boxes,), dtype=tf.string)
  return {
      'images': pad_to(image, [max_size, max_size, 3]),
      'boxes': tf.zeros((max_boxes, 4), dtype=tf.float32),
      'captions': pad_string_tensor(caption, [max_boxes]),
      'track_ids': tf.zeros((max_boxes,), dtype=tf.float32),
      'frame_inds': tf.zeros((), dtype=tf.int32),
      'image_ids': tf.zeros((), dtype=tf.int32),
      'image_captions': data['captions']['text'][:max_image_captions],
  }


def add_loss_mask(data, losses):
  assert set(losses).issubset(ALL_LOSSES), losses
  for loss in ALL_LOSSES:
    data[f'{loss}_loss_mask'] = tf.ones(
        (1,), dtype=tf.float32) * float(loss in losses)
  return data


def build_detection_ds(dataset_path):
  """Build a detection dataset."""
  if dataset_path == 'coco/2017':
    builder = tfds.builder('coco/2017')
    data_range = tfds.even_splits(
        'train', jax.process_count())[jax.process_index()]
    ds = builder.as_dataset(split=data_range)
  else:
    ds = tf.data.TFRecordDataset(
        centernet_input_pipeline.decode_sharded_names(dataset_path))
    ds = ds.shard(jax.process_count(), jax.process_index())
    ds = ds.map(
        lambda x: tf.io.parse_single_example(
            x, centernet_input_pipeline.coco_feature_description))
    ds = ds.map(centernet_input_pipeline.coco_decode_example)
  return ds


def load_video_train_tfds(
    batch_size,
    *,
    dataset_path,
    tokenizer,
    max_size=256,
    max_boxes=100,
    max_text_tokens=40,
    shuffle_buffer_size=1000,
    shuffle_seed=0,
    max_frames=200,
    max_video_captions=100,
    temporal_stride=1,
    scale_range=(0.5, 1.5),  # pylint: disable=unused-argument
    ensure_sample_has_objects=True,
    dataset_format='full',
    track_id_key='objects/track_id',
    max_image_captions=1):
  """Loads a split of a video dataset using TensorFlow Datasets.

  Args:
    batch_size: int; The batch size returned by the data pipeline.
    dataset_path: string; path of the dataset; by default load from tfds
    tokenizer: tokenizer
    max_size: int; Maximum image size.
    max_boxes: int; Maximum number of boxes.
    max_text_tokens: int; max number of text tokens.
    shuffle_buffer_size: int; Size of the shuffle buffer.
    shuffle_seed: int; Seed for shuffling the training data.
    max_frames: int max number of frames.
    max_video_captions: int.
    temporal_stride: stride to downsample frames.
    scale_range: spatial data augmentation for image datasets.
    ensure_sample_has_objects: bool.
    dataset_format: str
    track_id_key: str
    max_image_captions: int

  Returns:
    A `tf.data.Dataset`, and dataset info.
  """
  ds_info = {}
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48

  if dataset_format in ['full', 'videotracking']:
    ds = tf.data.TFRecordDataset(
        centernet_input_pipeline.decode_sharded_names(dataset_path))
    ds = ds.shard(jax.process_count(), jax.process_index())
    ds = ds.map(
        lambda x: tf.io.parse_sequence_example(
            x,
            sequence_features=densecap_sequence_feature_description,
            context_features=densecap_context_feature_description))
    ds = ds.with_options(options)
    ds = ds.repeat()
    ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    ds = ds.map(
        lambda x, y, _:
        decode_and_sample_video_example(
            x, y, _, max_frames, temporal_stride, ensure_sample_has_objects,
            track_id_key=track_id_key),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    losses = ['det', 'track']
    if dataset_format == 'full':
      losses.extend(['objcap', 'trackcap'])
    ds = ds.map(lambda x: add_loss_mask(x, losses))
  elif dataset_format == 'videocap':
    if 'tfrecord' in dataset_path:
      ds = tf.data.TFRecordDataset(
          centernet_input_pipeline.decode_sharded_names(dataset_path))
      ds = ds.shard(jax.process_count(), jax.process_index())
      ds = ds.map(
          lambda x: tf.io.parse_sequence_example(
              x,
              sequence_features=videocap_sequence_feature_description,
              context_features=videocap_context_feature_description))
    else:
      raise ValueError('Unsupported dataset format: %s' % dataset_path)
    ds = ds.with_options(options)
    ds = ds.repeat()
    ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    ds = ds.map(
        lambda x, y, _:
        decode_videocap(
            x, y, _, max_frames, temporal_stride),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda x: add_loss_mask(x, ['vidcap']))
  elif dataset_format in ['imagedensecap', 'imagedensecap-nodet']:
    ds = tf.data.TFRecordDataset(
        centernet_input_pipeline.decode_sharded_names(dataset_path))
    ds = ds.shard(jax.process_count(), jax.process_index())
    ds = ds.map(
        lambda x: tf.io.parse_single_example(x, vg_feature_description))
    ds = ds.with_options(options)
    ds = ds.repeat()
    ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    ds = ds.map(
        lambda x: decode_and_pad_vg_image(x, max_size, max_boxes, scale_range),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(max_frames, drop_remainder=True)
    ds = ds.map(add_zero_video_keys)
    losses = ['objcap'] if dataset_format == 'imagedensecap-nodet' else [
        'det', 'objcap']
    ds = ds.map(lambda x: add_loss_mask(x, losses))
  elif dataset_format == 'imagedetection':
    ds = build_detection_ds(dataset_path)
    ds = ds.with_options(options)
    ds = ds.repeat()
    ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    ds = ds.map(
        lambda x: convert_coco_format(x, max_size, max_boxes, scale_range),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(max_frames, drop_remainder=True)
    ds = ds.map(add_zero_video_keys)
    ds = ds.map(lambda x: add_loss_mask(x, ['det']))
  elif dataset_format == 'imagedetection-augment':
    ds = build_detection_ds(dataset_path)
    ds = ds.with_options(options)
    ds = ds.repeat()
    ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    ds = ds.map(
        lambda x: convert_coco_as_video(
            x, num_frames=max_frames, size=max_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda x: add_loss_mask(x, ['det', 'track']))
  elif dataset_format == 'imagecap':
    assert dataset_path == 'coco_captions'
    builder = tfds.builder('coco_captions')
    split = 'train+restval'
    data_range = tfds.even_splits(
        split, jax.process_count())[jax.process_index()]
    ds = builder.as_dataset(split=data_range)
    ds = ds.with_options(options)
    ds = ds.repeat()
    ds = ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
    ds = ds.map(
        lambda x: convert_cococaption_format(
            x, max_size, max_boxes, max_image_captions),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(max_frames, drop_remainder=True)
    ds = ds.map(add_zero_video_keys)
    ds = ds.map(lambda x: add_loss_mask(x, ['imagecap']))
  else:
    raise NotImplementedError(dataset_format)
  ds = ds.map(
      lambda x: decode_densecap_annotations
      (x, tokenizer, max_boxes=max_boxes,
       max_video_captions=max_video_captions,
       max_image_captions=max_image_captions))

  padded_shapes = {
      'inputs': [max_frames, max_size, max_size, 3],
      'label': {
          'boxes': [max_frames, max_boxes, 4],
          'text_tokens': [max_frames, max_boxes, max_text_tokens],
          'labels': [max_frames, max_boxes],
          'track_ids': [max_frames, max_boxes],
          'orig_size': [2,],
          'size': [2,],
          'video_caption_tokens': [max_video_captions, max_text_tokens],
          'frame_inds': [max_frames,],
          'image_caption_tokens': [
              max_frames, max_image_captions, max_text_tokens],
      },
      'loss_masks': {
          'det_loss_mask': [1,],
          'objcap_loss_mask': [1,],
          'track_loss_mask': [1,],
          'trackcap_loss_mask': [1,],
          'vidcap_loss_mask': [1,],
          'imagecap_loss_mask': [1,],
      }
  }
  preprocess_fn = lambda x: video_resize_max_size(x, max_size)
  ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.padded_batch(
      batch_size, padded_shapes=padded_shapes, drop_remainder=True)

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds, ds_info
