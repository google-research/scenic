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

"""Utils for adding modalities with bounding boxes.

Forked from https://github.com/deepmind/dmvr/blob/master/dmvr/modalities.py

"""


from typing import Optional, Union

from absl import logging
from dmvr import builders
from dmvr import processors
import ml_collections
import tensorflow as tf


def add_image_and_boxes(
    parser_builder: builders.BaseParserBuilder,
    sampler_builder: builders.SamplerBuilder,
    decoder_builder: builders.DecoderBuilder,
    preprocessor_builder: builders.PreprocessorBuilder,
    postprocessor_builder: builders.PostprocessorBuilder,
    input_feature_name: str = 'image/encoded',
    output_feature_name: str = builders.IMAGE_FEATURE_NAME,
    is_training: bool = True,
    # Video related parameters.
    num_frames: int = 32,
    stride: int = 1,
    num_test_clips: int = 1,
    min_resize: int = 224,
    resize_method: str = tf.image.ResizeMethod.BILINEAR,
    crop_size: int = 200,
    use_crop_and_resize_video_mae: bool = False,
    train_frame_sampling_mode: Optional[str] = None,
    zero_centering_image: bool = False,
    random_flip: bool = True,
    normalization_mean: Union[tf.Tensor, float] = 0,
    normalization_std: Union[tf.Tensor, float] = 1,
    object_configs: ml_collections.ConfigDict = ml_collections.ConfigDict(),
) -> None:
  """Same as add_image with additional support boxes."""
  with_boxes = object_configs.get('with_boxes', False)
  keep_full_frames = object_configs.get('keep_full_frames', 0)
  bbox_key = object_configs.get('bbox_key', 'gt')
  bbox_num = object_configs.get('bbox_num', 100)
  return_boxes = object_configs.get('return_boxes', -1)
  mask_radius = object_configs.get('mask_radius', -1.)
  min_box_size = object_configs.get('min_box_size', 0)
  concat_mask = object_configs.get('concat_mask', False)
  add_detections = object_configs.get('add_detections', False)
  detection_stride_spatial = object_configs.get('detection_stride_spatial', 1)
  detection_stride_temporal = object_configs.get('detection_stride_temporal', 1)
  tracked_objects = object_configs.get('tracked_objects', False)
  add_global_box = object_configs.get('add_global_box', False)

  if is_training and num_test_clips != 1:
    logging.info('`num_test_clips` %d is ignored since `is_training` is true.',
                 num_test_clips)
  # Parse frames or single image.
  assert isinstance(parser_builder, builders.SequenceExampleParserBuilder)
  parser_builder.parse_feature(
      feature_name=input_feature_name,
      feature_type=tf.io.FixedLenSequenceFeature((), dtype=tf.string),
      output_name=output_feature_name)

  if with_boxes:
    for coord in ['xmax', 'xmin', 'ymax', 'ymin', 'score']:
      parser_builder.parse_feature(
          feature_name=f'{bbox_key}/bbox/{coord}',
          feature_type=tf.io.VarLenFeature(dtype=tf.float32),
          output_name=coord)
      sampler_builder.add_fn(
          fn=tf.sparse.to_dense,
          feature_name=coord,
          fn_name='{}_sparse_to_dense'.format(coord))
    preprocessor_builder.add_fn(
        fn=aggregate_bbox_coords,
        feature_name=None,
        fn_name='aggregate_bbox_coords')

  # pylint: disable=g-long-lambda
  # Temporal sampler.
  if is_training:
    if train_frame_sampling_mode == 'segment_sampling':
      # Sample random clip.
      sampler_builder.add_fn(
          fn=lambda x, s: sample_sequence_uniformly_with_state(
              x, num_frames, True, s),
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_random_sample',
          stateful=True)
      if with_boxes:
        for coord in ['xmax', 'xmin', 'ymax', 'ymin', 'score']:
          sampler_builder.add_fn(
              fn=lambda x, s: sample_sequence_uniformly_with_state(
                  x, num_frames, True, s),
              feature_name=coord,
              fn_name=f'{coord}_random_sample',
              stateful=True)
    else:
      # Sample random clip.
      sampler_builder.add_fn(
          fn=lambda x, s=None: processors.sample_sequence(
              x, num_frames, True, stride, state=s),
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_random_sample',
          stateful=True)
      if with_boxes:
        for coord in ['xmax', 'xmin', 'ymax', 'ymin', 'score']:
          sampler_builder.add_fn(
              fn=lambda x, s=None: processors.sample_sequence(
                  x, num_frames, True, stride, state=s),
              feature_name=coord,
              fn_name=f'{coord}_random_sample',
              stateful=True)
  else:
    if num_test_clips > 1:
      if train_frame_sampling_mode == 'segment_sampling':
        if num_test_clips != 2:
          raise ValueError(
              'For segment_sampling only 2 video clips are allowed.')
        sampler_builder.add_fn(
            fn=lambda x: sample_two_sequences_uniformly(
                x, num_frames),
            feature_name=output_feature_name,
            fn_name=f'{output_feature_name}_segment_sampling_test')
        if with_boxes:
          for coord in ['xmax', 'xmin', 'ymax', 'ymin', 'score']:
            sampler_builder.add_fn(
                fn=lambda x: sample_two_sequences_uniformly(
                    x, num_frames),
                feature_name=coord,
                fn_name=f'{coord}_segment_sampling_test')
      else:
        # Sample linspace clips.
        sampler_builder.add_fn(
            fn=lambda x: processors.sample_linspace_sequence(
                x, num_test_clips, num_frames, stride),
            feature_name=output_feature_name,
            fn_name=f'{output_feature_name}_linspace_sample')
        if with_boxes:
          for coord in ['xmax', 'xmin', 'ymax', 'ymin', 'score']:
            sampler_builder.add_fn(
                fn=lambda x: processors.sample_linspace_sequence(
                    x, num_test_clips, num_frames, stride),
                feature_name=coord,
                fn_name=f'{coord}_linspace_sample')
    else:
      if train_frame_sampling_mode == 'segment_sampling':
        sampler_builder.add_fn(
            fn=lambda x: sample_sequence_uniformly(
                x, num_frames, is_training=False),
            feature_name=output_feature_name,
            fn_name=f'{output_feature_name}_segment_sampling_train')
        if with_boxes:
          for coord in ['xmax', 'xmin', 'ymax', 'ymin', 'score']:
            sampler_builder.add_fn(
                fn=lambda x: sample_sequence_uniformly(
                    x, num_frames, is_training=False),
                feature_name=coord,
                fn_name=f'{coord}_segment_sampling_train')
      else:
        # Sample middle clip.
        sampler_builder.add_fn(
            fn=lambda x: processors.sample_sequence(
                x, num_frames, False, stride),
            feature_name=output_feature_name,
            fn_name=f'{output_feature_name}_middle_sample')
        if with_boxes:
          for coord in ['xmax', 'xmin', 'ymax', 'ymin', 'score']:
            sampler_builder.add_fn(
                fn=lambda x: processors.sample_sequence(
                    x, num_frames, False, stride),
                feature_name=coord,
                fn_name=f'{coord}_middle_sample')

  decoder_builder.add_fn(
      fn=lambda x: processors.decode_jpeg(x, channels=3),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_decode_jpeg')

  # Resize images (resize happens only if necessary to save compute).
  preprocessor_builder.add_fn(
      fn=lambda x: processors.resize_smallest(
          x, min_resize, is_flow=False, method=resize_method),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_resize_smallest')
  # We don't need to do this for boxes as boxes are relative coordinates

  if is_training:
    # Standard image data augmentation: random crop and random flip.
    if use_crop_and_resize_video_mae:
      preprocessor_builder.add_fn(
          fn=lambda x, s=None: crop_and_resize_image_tong(x),
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_crop_and_resize',
          stateful=True)
      assert not with_boxes
    else:
      preprocessor_builder.add_fn(
          fn=lambda x, s=None: custom_crop_image(
              x, crop_size, crop_size, True, state=s),
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_random_crop',
          stateful=True)
      if with_boxes:
        preprocessor_builder.add_fn(
            fn=lambda x, s: transform_box(x, crop_size, crop_size, state=s),
            feature_name='bboxes',
            fn_name='bboxes_random_crop',
            stateful=True)
    if random_flip:
      preprocessor_builder.add_fn(
          fn=lambda x, s=None: processors.random_flip_left_right(
              x, state=s, is_flow=False),
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_random_flip',
          stateful=True)
      if with_boxes:
        preprocessor_builder.add_fn(
            fn=lambda x, s: random_flip_boxes_left_right(x, state=s),
            feature_name='bboxes',
            fn_name='bboxes_random_flip',
            stateful=True)
  else:
    if with_boxes:
      # Central crop of the frames.
      preprocessor_builder.add_fn(
          fn=lambda x, s: custom_crop_image(x, crop_size, crop_size, False, s),
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_central_crop',
          stateful=True)
      preprocessor_builder.add_fn(
          fn=lambda x, s: transform_box(x, crop_size, crop_size, state=s),
          feature_name='bboxes',
          fn_name='bboxes_central_crop',
          stateful=True)
    else:
      preprocessor_builder.add_fn(
          fn=lambda x: processors.crop_image(x, crop_size, crop_size, False),
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_central_crop')

  # Cast the frames to `tf.float32`, normalizing according to
  # `zero_centering_image`.
  preprocessor_builder.add_fn(
      fn=lambda x: processors.normalize_image(x, zero_centering_image),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_normalize')

  preprocessor_builder.add_fn(
      fn=lambda x: x - normalization_mean,
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_subtract_given_mean')

  preprocessor_builder.add_fn(
      fn=lambda x: x / normalization_std,
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_divide_by_given_std')

  if with_boxes:
    preprocessor_builder.add_fn(
        fn=lambda x: append_or_add_object_mask(
            x, num_frames, keep_full_frames, bbox_num, mask_radius,
            concat_mask,
            min_box_size=min_box_size,
            add_detections=add_detections,
            detection_stride_spatial=detection_stride_spatial,
            detection_stride_temporal=detection_stride_temporal,
            tracked_objects=tracked_objects,
            add_global_box=add_global_box),
        feature_name=None,
        fn_name='mask_bbox_region_fn')
    if return_boxes > 0:
      preprocessor_builder.add_fn(
          fn=lambda x: pad_bboxes(x, return_boxes),
          feature_name=None,
          fn_name='pad_bboxes')
    else:
      preprocessor_builder.add_fn(
          fn=remove_bboxes,
          feature_name=None,
          fn_name='remove_bboxes')

  if num_test_clips > 1 and not is_training:
    # In this case, multiple clips are merged together in batch dimension which
    # will be `B * num_test_clips`.
    postprocessor_builder.add_fn(
        fn=lambda x: tf.reshape(
            x, (-1, num_frames, x.shape[2], x.shape[3], x.shape[4])),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_reshape')
    if add_detections:
      postprocessor_builder.add_fn(
          fn=lambda x: tf.reshape(
              x,
              (-1, num_frames // detection_stride_temporal * (
                  crop_size // detection_stride_spatial) ** 2)),
          feature_name='detections',
          fn_name='detections_reshape')
    if return_boxes > 0:
      postprocessor_builder.add_fn(
          fn=lambda x: tf.reshape(
              x, (-1, num_frames, return_boxes, 4)),
          feature_name='bboxes',
          fn_name='bboxes_reshape')
  # pylint: enable=g-long-lambda


def append_or_add_object_mask(
    feature_dict,
    num_frames,
    keep_full_frames,
    bbox_num=100,
    mask_radius=-1.,
    concat_mask=False,
    min_box_size=0,
    add_detections=False,
    detection_stride_spatial=1,
    detection_stride_temporal=1,
    tracked_objects=False,
    add_global_box=False):
  """Generate object heatmaps.

  Args:
    feature_dict: dict of tensors from the dataloader
    num_frames: int; frames of a single model.
    keep_full_frames: number of paired frames that will allways have full masks.
      We use this to provide global features. Note the actual number of full
      frames will be 2 * keep_full_frames, where 2 is the temporal token size
      in ViViT.
    bbox_num: int; the max number of boxes in one frame.
    mask_radius: hyperparameter that controls the size of the heatmap peaks.
    concat_mask: bool: if True, it concatenane the heatmap to RPB pixels so
      that the output channels will be 4. If False, it multiplied the heatmap
      values to the pixels.
    min_box_size: threshold to filter out small boxes.
    add_detections: bool: if True, add mask as an additional key "detections".
    detection_stride_spatial: int: spatial stride of the mask
    detection_stride_temporal: int: temporal stride of the mask
    tracked_objects: bool; if True, return object-specific heatmaps
    add_global_box: if True, add a box for the whole image.
  Returns:
    the updated feature_dict.
  """
  assert not tracked_objects or bbox_num < 100
  bboxes = feature_dict['bboxes']  # T x N x 4
  images = feature_dict['image']  # T x H x W x 3
  scores = feature_dict['score']  # T x N

  if add_global_box:
    t = bboxes.shape[0]
    new_box = tf.constant([0, 0, 1., 1.], dtype=tf.float32, shape=[1, 1, 4])
    new_box = tf.broadcast_to(new_box, (t, 1, 4))
    bboxes = tf.concat([new_box, bboxes], axis=1)
    new_score = tf.constant([1.], dtype=tf.float32, shape=[1, 1])
    new_score = tf.broadcast_to(new_score, (scores.shape[0], 1))
    scores = tf.concat([new_score, scores], axis=1)

  if bbox_num < 100:
    n = tf.shape(bboxes)[0]
    scores = scores[:, :bbox_num]
    bboxes = bboxes[:, :bbox_num]
    input_length = tf.shape(bboxes)[1]
    input_length = tf.clip_by_value(input_length, 0, bbox_num)
    padding_length = tf.maximum(0, bbox_num - input_length)
    paddings_boxes = tf.zeros((n, padding_length, 4), tf.float32)
    bboxes = tf.concat([bboxes, paddings_boxes], axis=1)

  t, h, w = images.shape[0], images.shape[1], images.shape[2]

  if keep_full_frames == 0:  # skip every frame-group
    skip = tf.convert_to_tensor(value=[False] * t, dtype=tf.bool)
    skip = tf.reshape(skip, (t, 1))
    skip = tf.broadcast_to(skip, [t, h * w])
  elif keep_full_frames == 1:  # keep middle frame
    value = ([False, False] * (num_frames // 4) + [True, True] + [
        False, False] * (num_frames // 4 - 1)) * (t // num_frames)
    skip = tf.convert_to_tensor(value=value, dtype=tf.bool)
    skip = tf.reshape(skip, (t, 1))
    skip = tf.broadcast_to(skip, [t, h * w])
  elif keep_full_frames == 2:  # keep first and last
    value = ([True, True] + [False, False] * (
        (num_frames - 4) // 2) + [True, True]) * (t // num_frames)
    skip = tf.convert_to_tensor(value=value, dtype=tf.bool)
    skip = tf.reshape(skip, (t, 1))
    skip = tf.broadcast_to(skip, [t, h * w])
  else:
    assert False, keep_full_frames
  # skip: t x h x w

  dw, dh = w // detection_stride_spatial, h // detection_stride_spatial
  xs = tf.range(0, dw) * detection_stride_spatial + (
      detection_stride_spatial // 2)
  ys = tf.range(0, dh) * detection_stride_spatial + (
      detection_stride_spatial // 2)
  grid_x, grid_y = tf.meshgrid(xs, ys)
  grid_x = tf.cast(tf.reshape(grid_x, (dh * dw,)) / w, tf.float32)
  grid_y = tf.cast(tf.reshape(grid_y, (dh * dw,)) / h, tf.float32)
  grid = tf.stack([grid_x, grid_y], axis=1)  # HW x 2
  grid = tf.broadcast_to(grid, [images.shape[0], dh * dw, 2])
  # grid: T x HW x 2

  bboxes = tf.clip_by_value(bboxes, 0.0, 1.0)
  x1y1 = tf.minimum(bboxes[:, :, :2], bboxes[:, :, 2:])
  x2y2 = tf.maximum(bboxes[:, :, :2], bboxes[:, :, 2:])
  bboxes = tf.concat([x1y1, x2y2], axis=2)  # fix bugs in Epic box format

  if mask_radius <= 0.0:
    inside_box = grid[:, None, :, 0] >= bboxes[:, :, None, 0]  # T x N x HW
    inside_box = tf.logical_and(
        inside_box, grid[:, None, :, 0] <= bboxes[:, :, None, 2])
    inside_box = tf.logical_and(
        inside_box, grid[:, None, :, 1] >= bboxes[:, :, None, 1])
    inside_box = tf.logical_and(
        inside_box, grid[:, None, :, 1] <= bboxes[:, :, None, 3])  # T x N x HW
    if tracked_objects:
      mask = tf.cast(
          tf.transpose(inside_box, (0, 2, 1)), tf.float32)  # T x HW x N
      mask_channels = bbox_num
    else:
      mask = tf.cast(
          tf.reduce_any(inside_box, 1), tf.float32)  # T x HW
      mask_channels = 1
  else:
    centers = tf.stack(
        [(bboxes[:, :, 0] + bboxes[:, :, 2]) / 2,
         (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2], axis=2)  # T x N x 2
    dist2 = tf.reduce_sum(
        (grid[:, None, :] - centers[:, :, None])**2, axis=3)  # T x N x HW
    area = (bboxes[:, :, 2] - bboxes[:, :, 0]) * (
        bboxes[:, :, 3] - bboxes[:, :, 1])  # T x N
    filtered = tf.cast(area * h * w <= min_box_size ** 2, tf.float32)  # T x N
    dist2 = dist2 * (
        1. - filtered[:, :, None]) + filtered[:, :, None] * 10000000.
    delta = (1. - mask_radius) / (1. + mask_radius)  # scalar
    radius2 = delta ** 2 * 2 * area + 1e-6  # T x N
    if tracked_objects:
      weighted_dist = dist2 / radius2[:, :, None]  # T x N x HW
      gaussian_mask = tf.exp(-weighted_dist)  # T x N x HW
      gaussian_mask = tf.transpose(gaussian_mask, (0, 2, 1))  # T x HW x N
      mask_channels = bbox_num
    else:
      weighted_dist = tf.reduce_min(
          dist2 / radius2[:, :, None], axis=1)  # T x HW
      gaussian_mask = tf.exp(-weighted_dist)  # T x HW
      mask_channels = 1
    mask = gaussian_mask

  if concat_mask:
    mask = tf.reshape(mask, (t, h, w, mask_channels))
    mask = tf.maximum(
        tf.cast(
            tf.broadcast_to(
                tf.reshape(skip, (t, h, w, 1)), (t, h, w, mask_channels)),
            tf.float32),
        mask)
    images = tf.concat([images, tf.cast(mask, tf.float32)], axis=3)
  elif add_detections:
    mask = tf.reshape(
        mask,
        (t // detection_stride_temporal, detection_stride_temporal, dh, dw))
    mask = tf.reduce_max(mask, axis=1)  # t' x h x w
    mask = tf.reshape(
        mask,
        (t // detection_stride_temporal * dh * dw,))
    feature_dict['detections'] = tf.cast(mask, tf.float32)
  feature_dict['image'] = images
  feature_dict['bboxes'] = bboxes
  return feature_dict


def aggregate_bbox_coords(feature_dict):
  """"Aggregate coordinates into boxes."""
  feature_dict['bboxes'] = tf.stack([
      feature_dict['xmin'],
      feature_dict['ymin'],
      feature_dict['xmax'],
      feature_dict['ymax']
  ], axis=2)
  # Remove these temporary fields.
  for coord_name in {'xmax', 'xmin', 'ymax', 'ymin'}:
    feature_dict.pop(coord_name)
  return feature_dict


def remove_bboxes(feature_dict):
  feature_dict.pop('bboxes', None)
  feature_dict.pop('score', None)
  return feature_dict


def pad_bboxes(feature_dict, num_pad_boxes):
  """Pad boxes to the same size."""
  bboxes = feature_dict['bboxes']  # T x N x 4

  num_frames = tf.shape(bboxes)[0]
  input_length = tf.shape(bboxes)[1]
  input_length = tf.clip_by_value(input_length, 0, num_pad_boxes)
  bboxes = bboxes[:, :input_length]

  padding_length = tf.maximum(0, num_pad_boxes - input_length)

  paddings_boxes = tf.cast(
      tf.zeros((num_frames, padding_length, 4)), tf.float32)
  padded_bboxes = tf.concat([bboxes, paddings_boxes], axis=1)
  feature_dict['bboxes'] = padded_bboxes
  feature_dict.pop('score', None)
  return feature_dict


def custom_crop_image(
    frames: tf.Tensor,
    height: int,
    width: int,
    random: bool,
    state: Optional[builders.ProcessorState] = None,
    ) -> tf.Tensor:
  """Add more metadata to processors.crop_image."""
  if random:
    # Random spatial crop. tf.image.random_crop is not used since the offset is
    # needed to ensure consistency between crops on different modalities.
    shape = tf.shape(input=frames)
    # If a static_shape is available (e.g. when using this method from add_image
    # method), it will be used to have an output tensor with static shape.
    static_shape = frames.shape.as_list()
    seq_len = shape[0] if static_shape[0] is None else static_shape[0]
    channels = shape[3] if static_shape[3] is None else static_shape[3]
    size = tf.convert_to_tensor(value=(seq_len, height, width, channels))

    if state and 'crop_offset_proportion' in state:
      # Use offset set by a previous cropping: [0, offset_h, offset_w, 0].
      offset = state['crop_offset_proportion'] * tf.cast(shape, tf.float32)
      offset = tf.cast(tf.math.round(offset), tf.int32)
    else:
      # Limit of possible offset in order to fit the entire crop:
      # [1, input_h - target_h + 1, input_w - target_w + 1, 1].
      limit = shape - size + 1
      offset = tf.random.uniform(
          shape=(4,),
          dtype=tf.int32,
          maxval=tf.int32.max,
          ) % limit  # [0, offset_h, offset_w, 0]

      if state is not None:
        # Update state.
        offset_proportion = tf.cast(offset, tf.float32) / tf.cast(
            shape, tf.float32)
        state['crop_offset_proportion'] = offset_proportion  # 4
        state['crop_ori_shape'] = shape  # 4
        state['crop_offset_abs'] = offset  # 4

    frames = tf.slice(frames, offset, size)
  else:
    # Central crop or pad.
    shape = tf.shape(input=frames)
    static_shape = frames.shape.as_list()
    seq_len = shape[0] if static_shape[0] is None else static_shape[0]
    channels = shape[3] if static_shape[3] is None else static_shape[3]
    size = tf.convert_to_tensor(value=(seq_len, height, width, channels))
    offset = tf.convert_to_tensor(
        value=(0, (shape[1] - height) // 2, (shape[2] - width) // 2, 0))
    frames = tf.slice(frames, offset, size)
    if state is not None:
      state['crop_ori_shape'] = shape
      state['crop_offset_abs'] = offset
  return frames


def transform_box(boxes: tf.Tensor,
                  height: int,
                  width: int,
                  state: builders.ProcessorState) -> tf.Tensor:
  """Transform boxes given the state of image crop."""
  assert (state and 'crop_offset_abs' in state and
          'crop_ori_shape' in state)
  offset = state['crop_offset_abs']
  ori_shape = tf.cast(state['crop_ori_shape'], tf.float32)
  pixel_offset = tf.reshape(tf.stack(
      [offset[2], offset[1], offset[2], offset[1]]), (1, 1, 4))
  ori_size = tf.stack([ori_shape[2], ori_shape[1], ori_shape[2], ori_shape[1]])
  ori_size = tf.reshape(ori_size, (1, 1, 4))
  ori_boxes = boxes * tf.cast(ori_size, tf.float32)
  crop_boxes = ori_boxes - tf.cast(pixel_offset, tf.float32)
  new_shape = tf.constant([width, height, width, height],
                          dtype=tf.float32, shape=[1, 1, 4])
  out_boxes = crop_boxes / new_shape
  return out_boxes


def three_spatial_crops_with_state(images, state, crop_size):
  """Returns three spatial crops of the same frame, as done by SlowFast."""
  height, width = tf.shape(images)[1], tf.shape(images)[2]

  result = []
  for spatial_idx in range(3):
    y_offset = tf.cast(tf.math.ceil((height - crop_size) / 2), tf.int32)
    x_offset = tf.cast(tf.math.ceil((width - crop_size) / 2), tf.int32)
    if height > width:
      if spatial_idx == 0:
        y_offset = 0
      elif spatial_idx == 2:
        y_offset = height - crop_size
    else:
      if spatial_idx == 0:
        x_offset = 0
      elif spatial_idx == 2:
        x_offset = width - crop_size
    images_cropped = tf.slice(
        images, [0, y_offset, x_offset, 0], [-1, crop_size, crop_size, -1])
    if state is not None:
      offset = tf.convert_to_tensor(value=(0, y_offset, x_offset, 0))
      shape = tf.convert_to_tensor(value=(-1, height, width, -1))
      offset_proportion = tf.cast(offset, tf.float32) / tf.cast(
          shape, tf.float32)
      state[f'crop_offset_proportion_{spatial_idx}'] = offset_proportion  # 4
      state[f'crop_ori_shape_{spatial_idx}'] = shape  # 4
      state[f'crop_offset_abs_{spatial_idx}'] = offset  # 4
    result.append(images_cropped)

  return tf.concat(result, axis=0)


def three_spatial_transform_box(boxes, state, crop_size):
  """Transform boxes given the state of image crop."""
  result = []
  for spatial_idx in range(3):
    assert (state and f'crop_offset_abs_{spatial_idx}' in state and
            f'crop_ori_shape_{spatial_idx}' in state)
    offset = state[f'crop_offset_abs_{spatial_idx}']
    ori_shape = tf.cast(state[f'crop_ori_shape_{spatial_idx}'], tf.float32)
    pixel_offset = tf.reshape(tf.stack(
        [offset[2], offset[1], offset[2], offset[1]]), (1, 1, 4))
    ori_size = tf.stack(
        [ori_shape[2], ori_shape[1], ori_shape[2], ori_shape[1]])
    ori_size = tf.reshape(ori_size, (1, 1, 4))
    ori_boxes = boxes * tf.cast(ori_size, tf.float32)
    crop_boxes = ori_boxes - tf.cast(pixel_offset, tf.float32)
    new_shape = tf.constant([crop_size, crop_size, crop_size, crop_size],
                            dtype=tf.float32, shape=[1, 1, 4])
    out_boxes = crop_boxes / new_shape
    result.append(out_boxes)
  return tf.concat(result, axis=0)


def random_flip_boxes_left_right(boxes: tf.Tensor,
                                 state: Optional[
                                     builders.ProcessorState] = None,
                                 ) -> tf.Tensor:
  """Flip boxes given the state of image flip."""
  assert state and 'flip_left_right_is_flipped' in state
  is_flipped = state['flip_left_right_is_flipped']
  boxes = tf.cond(
      pred=tf.equal(is_flipped, 1),
      # pylint: disable=g-long-lambda
      true_fn=lambda: tf.constant([1, 0, 1, 0], dtype=tf.float32) + tf.constant(
          [-1, 1, -1, 1], dtype=tf.float32) * boxes,
      # pylint: enable=g-long-lambda
      false_fn=lambda: boxes)
  return boxes


def sample_sequence_uniformly_with_state(
    sequence: tf.Tensor,
    num_steps: int,
    is_training: bool = True,
    state: Optional[builders.ProcessorState] = None) -> tf.Tensor:
  """mfp sample_sequence_uniformly with state."""

  sequence_length = tf.shape(input=sequence)[0]
  sequence_length = tf.cast(sequence_length, tf.int32)
  stride = tf.cast(sequence_length // num_steps, tf.int32)

  if state and 'sample_sequence_uniformly_indices' in state:
    indices = state['sample_sequence_uniformly_indices']
  else:
    if stride > 0:
      indices = tf.math.multiply(tf.range(num_steps), stride)
      if is_training:
        indices = indices + tf.random.uniform(shape=(1, num_steps), minval=0,
                                              maxval=stride, dtype=tf.int32)
    else:
      if is_training:
        indices = tf.sort(tf.random.uniform(shape=(1, num_steps),
                                            minval=0, maxval=sequence_length,
                                            dtype=tf.int32))
      else:
        stride_float = tf.cast(sequence_length / num_steps, tf.float32)
        indices = tf.cast(tf.range(num_steps, dtype=tf.float32) * stride_float,
                          tf.int32)
    if state is not None:
      state['sample_sequence_uniformly_indices'] = indices

  if is_training:
    indices = indices[0]

  indices.set_shape((num_steps,))
  output = tf.gather(sequence, indices)
  return output


def sample_fixed_offset(image_w: int, image_h: int, crop_w: int, crop_h: int,
                        more_fix_crop: bool = True) -> tf.Tensor:
  """Sample offset of the crop out of 13 fixed offsets.

  The sampling strategy is taken from: https://arxiv.org/abs/2203.12602, Github:
  https://github.com/MCG-NJU/VideoMAE.

  Args:
    image_w: The width of the image.
    image_h: The height of the image.
    crop_w: The width of the crop.
    crop_h: The height of the crop.
    more_fix_crop: Add another 8 fixed crops to the sampling.

  Returns:
    A tensor of shape [1, 2] with the corresponding offset
    [[offset_w, offset_h]].
  """
  w_step = (image_w - crop_w) // 4
  h_step = (image_h - crop_h) // 4

  ret = list()
  ret.append((tf.constant(0), tf.constant(0)))  # upper left
  ret.append((4 * w_step, 0))  # upper right
  ret.append((0, 4 * h_step))  # lower left
  ret.append((4 * w_step, 4 * h_step))  # lower right
  ret.append((2 * w_step, 2 * h_step))  # center

  if more_fix_crop:
    ret.append((0, 2 * h_step))  # center left
    ret.append((4 * w_step, 2 * h_step))  # center right
    ret.append((2 * w_step, 4 * h_step))  # lower center
    ret.append((2 * w_step, 0 * h_step))  # upper center

    ret.append((1 * w_step, 1 * h_step))  # upper left quarter
    ret.append((3 * w_step, 1 * h_step))  # upper right quarter
    ret.append((1 * w_step, 3 * h_step))  # lower left quarter
    ret.append((3 * w_step, 3 * h_step))  # lower right quarter

  ret_index = tf.random.uniform((1, 1), minval=0, maxval=len(ret),
                                dtype=tf.int32)[0, 0]
  ret = tf.stack(ret)

  ret_pair = tf.slice(ret, [ret_index, 0], [1, 2])
  return ret_pair


def sample_crop_size(image_h: int, image_w: int,
                     resized_size: tuple[int, int], scales: tf.Tensor,
                     max_distort: int = 1) -> tuple[int, int, int, int]:
  """Sample a crop size and the offset out of fixed choices.

  Args:
    image_h: The height of the image.
    image_w: The width of the image.
    resized_size: The size of the resized image.
    scales: The scales for the resize operation.
    max_distort: How many adjact possitions in the scales array to combine in
    order to get the pairs for the resize options.

  Returns:
    A tuple of 4 elements -> [crop_h, crop_w, offset_h, offset_w].

  """

  if len(scales) != 4:
    raise NotImplementedError('Only 4 values are supported for the scale.')

  base_size = tf.cast(tf.minimum(image_w, image_h), tf.float32)

  crop_sizes = [tf.cast(base_size * scales[0], tf.int32),
                tf.cast(base_size * scales[1], tf.int32),
                tf.cast(base_size * scales[2], tf.int32),
                tf.cast(base_size * scales[3], tf.int32)]
  rsize_h, rsize_w = resized_size

  crop_h = [
      rsize_h if abs(crop_sizes[0] - rsize_h) < 3 else crop_sizes[0],
      rsize_h if abs(crop_sizes[1] - rsize_h) < 3 else crop_sizes[1],
      rsize_h if abs(crop_sizes[2] - rsize_h) < 3 else crop_sizes[2],
      rsize_h if abs(crop_sizes[3] - rsize_h) < 3 else crop_sizes[3]]

  crop_w = [
      rsize_w if abs(crop_sizes[0] - rsize_w) < 3 else crop_sizes[0],
      rsize_w if abs(crop_sizes[1] - rsize_w) < 3 else crop_sizes[1],
      rsize_w if abs(crop_sizes[2] - rsize_w) < 3 else crop_sizes[2],
      rsize_w if abs(crop_sizes[3] - rsize_w) < 3 else crop_sizes[3]]

  # Get the resized pairs.
  pairs = []
  for i, h in enumerate(crop_h):
    for j, w in enumerate(crop_w):
      if abs(i - j) <= max_distort:
        pairs.append((w, h))

  # Implement random.choice.
  crop_pair_index = tf.random.uniform((1, 1), minval=0, maxval=len(pairs),
                                      dtype=tf.int32)[0, 0]
  pairs = tf.stack(pairs)
  crop_pair = tf.slice(pairs, [crop_pair_index, 0], [1, 2])

  offset = sample_fixed_offset(image_w=image_w, image_h=image_h,
                               crop_w=crop_pair[0][0], crop_h=crop_pair[0][1])
  return crop_pair[0][1], crop_pair[0][0], offset[0][1], offset[0][0]


def crop_and_resize_image_tong(frames: tf.Tensor,
                               resized_size: tuple[int, int] = (224, 224),
                               scales: tf.Tensor = tf.constant(
                                   [1, .875, .75, .66])) -> tf.Tensor:
  """Crops and resizes the images in the given sequence of images.

  Args:
    frames: A tensor of dimension [timesteps, input_h, input_w, channels].
    resized_size: The size for the resize operation.
    scales: The scales for the resize operation. Must be a tensor with 4 values.
  Returns:
    A tensor of shape [timesteps, output_h, output_w, channels] of same type as
    input with the cropped and resized images.
  """

  shape = tf.shape(input=frames)
  timesteps = shape[0]
  image_h = shape[1]
  image_w = shape[2]
  channels = shape[3]

  crop_h, crop_w, offset_h, offset_w = sample_crop_size(
      image_h=image_h, image_w=image_w, resized_size=resized_size,
      scales=scales)

  offset = tf.convert_to_tensor(value=(0, offset_h, offset_w, 0))
  size = tf.convert_to_tensor(value=(timesteps, crop_h, crop_w, channels))
  frames = tf.slice(frames, offset, size)
  frames = tf.image.resize(frames, resized_size)

  return frames


def sample_sequence_uniformly(
    sequence: tf.Tensor,
    num_steps: int,
    is_training: bool = True) -> tf.Tensor:
  """Uniform frame sampling.

  Sample frames based on uniform sampling following TSN (Wang et al., 2019)
  used by Tong et al. in VideoMAE. The stride is automatically computed based on
  the length of the sequence and the number of frames to take (`num_steps`). If
  `is_training` is set to False, a deterministic sequence will be returned.

  Args:
    sequence: Any tensor where the first dimension is timesteps.
    num_steps: Number of steps (e.g. frames) to take.
    is_training: If is called during training or not.
  Returns:
     A single tensor with first dimension `num_steps` with the sampled segment.
  """

  sequence_length = tf.shape(input=sequence)[0]
  sequence_length = tf.cast(sequence_length, tf.int32)
  stride = tf.cast(sequence_length // num_steps, tf.int32)

  if stride > 0:
    indices = tf.math.multiply(tf.range(num_steps), stride)
    if is_training:
      indices = indices + tf.random.uniform(shape=(1, num_steps), minval=0,
                                            maxval=stride, dtype=tf.int32)
  else:
    if is_training:
      indices = tf.sort(tf.random.uniform(shape=(1, num_steps),
                                          minval=0, maxval=sequence_length,
                                          dtype=tf.int32))
    else:
      stride_float = tf.cast(sequence_length / num_steps, tf.float32)
      indices = tf.cast(tf.range(num_steps, dtype=tf.float32) * stride_float,
                        tf.int32)
  if is_training:
    indices = indices[0]

  indices.set_shape((num_steps,))
  output = tf.gather(sequence, indices)
  return output


def sample_two_sequences_uniformly(sequence: tf.Tensor, num_steps: int):
  """Uniform sampling two non-overlapping sequences.

  Sample frames based on uniform sampling following TSN (Wang et al., 2019)
  used by Tong et al. in VideoMAE. The stride is automatically computed based on
  the length of the sequence and the number of frames to take (`num_steps`)

  Args:
    sequence: Any tensor where the first dimension is timesteps.
    num_steps: Number of steps (e.g. frames) to take.
  Returns:
     A single tensor with first dimension `2 * num_steps` with the sampled
     segment.
  """

  sequence_length = tf.shape(input=sequence)[0]
  sequence_length = tf.cast(sequence_length, tf.int32)
  average_duration = tf.cast(sequence_length / num_steps, tf.float32)

  index_1 = tf.cast(tf.range(num_steps, dtype=tf.float32)
                    * average_duration + average_duration / 2.0, tf.int32)

  index_2 = tf.cast(tf.range(num_steps, dtype=tf.float32)
                    * average_duration, tf.int32)
  indices = tf.concat((index_1, index_2), axis=0)

  indices.set_shape((2 * num_steps,))
  output = tf.gather(sequence, indices)
  return output


def deterministic_crop(images, size, spatial_idx):
  """Takes a deterministic crop of input images.

  Args:
    images: `Tensor` of shape shape [t, h, w, c]
    size: Integer ; size of height and width to crop the images.
    spatial_idx: 0, 1, or 2 for left, center, and right crop if width is larger
      than height. Or 0, 1, or 2 for top, center, and bottom crop if height is
      larger than width.

  Returns:
    cropped: `Tensor` of shape [t, crop_size, crop_size, c]
  """
  assert spatial_idx in [0, 1, 2]
  height, width = tf.shape(images)[1], tf.shape(images)[2]

  y_offset = tf.cast(tf.math.ceil((height - size) / 2), tf.int32)
  x_offset = tf.cast(tf.math.ceil((width - size) / 2), tf.int32)

  if height > width:
    if spatial_idx == 0:
      y_offset = 0
    elif spatial_idx == 2:
      y_offset = height - size
  else:
    if spatial_idx == 0:
      x_offset = 0
    elif spatial_idx == 2:
      x_offset = width - size

  cropped = tf.slice(images, [0, y_offset, x_offset, 0], [-1, size, size, -1])

  return cropped


def three_spatial_crops(images, crop_size):
  """Returns three spatial crops of the same frame, as done by SlowFast.

  This enables testing using the same protocol as prior works. ie
  (https://arxiv.org/abs/1812.03982, https://arxiv.org/abs/1904.02811,
   https://arxiv.org/abs/2004.04730)
  If width > height, takes left, centre and right crop.
  If height > width, takes top, middle and bottom crop.

  Args:
    images: `Tensor` of shape [t, h, w, c]
    crop_size: The size to crop from the images

  Returns:
    `Tensor` of shape [3 * t, h, w, c]
  """

  result = []
  for spatial_index in range(3):
    images_cropped = deterministic_crop(images, crop_size, spatial_index)
    result.append(images_cropped)

  return tf.concat(result, axis=0)
