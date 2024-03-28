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

"""Contains postprocessing utility functions."""

from typing import Any, List, Optional, Tuple, Union

import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.projects.unloc import model_utils
import tensorflow as tf

PyModule = Any
Array = Union[jnp.ndarray, np.ndarray]


def dedup_by_vid(
    logits: np.ndarray,
    labels: np.ndarray,
    batch_masks: np.ndarray,
    vids: np.ndarray,
    frame_masks: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Dedups by video ids.

  Args:
    logits: Predicted class logits in shape (num_videos, num_classes) if
      frame_masks is None or in shape (num_videos, num_frames, num_classes)
      otherwise.
    labels: Multihot vectors representing the ground truth labels in shape
      (num_videos, num_classes) if frame_masks is None or in shape (num_videos,
      num_frames, num_classes) otherwise.
    batch_masks: Batch masks in shape (num_videos,).
    vids: Video ids in shape (num_videos,).
    frame_masks: Frame masks in shape (num_videos, num_frames).

  Returns:
    deduped logits in shape (N, num_classes).
    deduped labels in shape (N, num_classes).
    deduped video ids in shape (N,).
  """

  batch_masks = batch_masks.astype(bool)
  vids = vids[batch_masks]
  logits = logits[batch_masks]
  labels = labels[batch_masks]
  if frame_masks is not None:
    frame_masks = frame_masks.astype(bool)
    frame_masks = frame_masks[batch_masks]
  vid_set = set()
  deduped_logits, deduped_labels, deduped_vids = [], [], []
  for idx, vid in enumerate(vids):
    if vid in vid_set:
      continue
    if frame_masks is None:
      deduped_logits.append(logits[idx][np.newaxis, :])
      deduped_labels.append(labels[idx][np.newaxis, :])
    else:
      frame_mask = frame_masks[idx]
      deduped_logits.append(logits[idx][frame_mask])
      deduped_labels.append(labels[idx][frame_mask])
    vid_set.add(vid)
    deduped_vids.append(vid)
  return (np.concatenate(deduped_logits, axis=0),
          np.concatenate(deduped_labels, axis=0), np.array(deduped_vids))


def make_2d_boxes(segments: Array, np_backend: PyModule = np) -> Array:
  """Make 2D boxes out of 1D segments.

  We reuse tf.image.non_max_suppression_with_scores() for non-maximal
  suppression, which takes 2D boxes.

  Args:
    segments: Temporal segments in shape (N, 2).
    np_backend: Numpy backend.

  Returns:
    2D boxes in shape (N, 4).
  """
  n = segments.shape[0]
  return np_backend.stack([
      np_backend.zeros((n,), dtype=np_backend.float32),
      segments[:, 0],
      np_backend.ones((n,), dtype=np_backend.float32),
      segments[:, 1],
  ],
                          axis=1)


def non_max_suppression(
    class_indices: np.ndarray, scores: np.ndarray, segments: np.ndarray,
    config: ml_collections.ConfigDict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Performs class-agnostic non-maximal suppression.

  Args:
    class_indices: Predicted class indices in shape (N,).
    scores: Predicted class scores in shape (N,).
    segments: Predicted temporal segments in shape (N, 2).
    config: NMS configs.

  Returns:
    class indices, scores, and segments after NMS.
  """

  out_class_indices = []
  out_scores = []
  out_segments = []

  if class_indices.size:
    selected_indices, selected_scores = (
        tf.image.non_max_suppression_with_scores(
            make_2d_boxes(segments),
            scores,
            max_output_size=config.get('max_detections', 100),
            iou_threshold=config.get('iou_threshold', 0.5),
            score_threshold=config.get('score_threshold', 0.001),
            soft_nms_sigma=config.get('soft_nms_sigma', 0.3),
        )
    )
    selected_indices = selected_indices.numpy()
    selected_scores = selected_scores.numpy()
    out_class_indices.append(class_indices[selected_indices])
    out_scores.append(selected_scores)
    out_segments.append(segments[selected_indices])

  out_class_indices = (
      np.concatenate(out_class_indices, axis=0)
      if out_class_indices else np.array([], dtype=np.int32))
  out_scores = (
      np.concatenate(out_scores, axis=0)
      if out_scores else np.array([], dtype=np.float32))
  out_segments = (
      np.concatenate(out_segments, axis=0)
      if out_segments else np.array([], dtype=np.float32))
  return out_class_indices, out_scores, out_segments


def non_max_suppression_multiclass(
    class_indices: np.ndarray, scores: np.ndarray, segments: np.ndarray,
    config: ml_collections.ConfigDict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Performs multiclass non-maximal suppression.

  Args:
    class_indices: Predicted class indices in shape (N,).
    scores: Predicted class scores in shape (N,).
    segments: Predicted temporal segments in shape (N, 2).
    config: NMS configs.

  Returns:
    class indices, scores, and segments after NMS.
  """

  out_class_indices = []
  out_scores = []
  out_segments = []

  for cls_idx in range(config.dataset_configs.num_classes):
    mask = class_indices == cls_idx
    cur_class_indices = class_indices[mask]
    if cur_class_indices.size:
      cur_segments = segments[mask]
      cur_scores = scores[mask]
      selected_indices, selected_scores = (
          tf.image.non_max_suppression_with_scores(
              make_2d_boxes(cur_segments),
              cur_scores,
              max_output_size=config.get('max_detections', 100),
              iou_threshold=config.get('iou_threshold', 0.5),
              score_threshold=config.get('score_threshold', 0.001),
              soft_nms_sigma=config.get('soft_nms_sigma', 0.3),
          )
      )
      selected_indices = selected_indices.numpy()
      selected_scores = selected_scores.numpy()
      out_class_indices.append(cur_class_indices[selected_indices])
      out_scores.append(selected_scores)
      out_segments.append(cur_segments[selected_indices])
  out_class_indices = (
      np.concatenate(out_class_indices, axis=0)
      if out_class_indices else np.array([], dtype=np.int32))
  out_scores = (
      np.concatenate(out_scores, axis=0)
      if out_scores else np.array([], dtype=np.float32))
  out_segments = (
      np.concatenate(out_segments, axis=0)
      if out_segments else np.array([], dtype=np.float32))
  return out_class_indices, out_scores, out_segments


def non_max_suppression_mr(
    scores: np.ndarray, segments: np.ndarray,
    config: ml_collections.ConfigDict
    ) -> Tuple[List[Array], List[Array]]:
  """Performs class-agnostic non-maximal suppression for each caption.

  Args:
    scores: Predicted class scores in shape (num_captions, N).
    segments: Predicted segments in shape (num_captions, N, 2).
    config: NMS configs.

  Returns:
    A List of scores in shape (M, ) and segments in shape (M, 2) after NMS,
    where 0 <= M <= `max_detections`.
  """

  out_scores = []
  out_segments = []
  num_captions = scores.shape[0]
  for i in range(num_captions):
    selected_indices, selected_scores = (
        tf.image.non_max_suppression_with_scores(
            make_2d_boxes(segments[i]),
            scores[i],
            max_output_size=config.get('max_detections', 100),
            iou_threshold=config.get('iou_threshold', 0.5),
            score_threshold=config.get('score_threshold', 0.001),
            soft_nms_sigma=config.get('soft_nms_sigma', 0.3),
        )
    )
    out_scores.append(selected_scores.numpy())
    out_segments.append(segments[i][selected_indices])

  return out_scores, out_segments


def get_segments_from_frame_predictions(
    class_probs: np.ndarray,
    displacements: np.ndarray,
    input_mask: np.ndarray,
    total_frames: int,
    stride: int,
    sampling_strategy: str = 'random',
    displacement_normalizer: str = 'duration',
    secs_per_timestep: float = 1.0,
    score_threshold: float = float('-inf'),
    feature_pyramid_config: Optional[ml_collections.ConfigDict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes predicted segments based on frame predictions.

  We assume that all frame predictions are from one video.

  Args:
    class_probs: A float array representing the class probabilities in shape
      (num_frames, num_classes).
    displacements: A float array representing the start/end time displacements
      in shape (num_frames, num_classes, 2). All values are greater or equal to
      zero.
    input_mask: A int array representing the input mask in shape (num_frames,).
    total_frames: The total number of frames in this video.
    stride: Temporal stride used in sampling the frames
    sampling_strategy: 'random' or 'linspace'.
    displacement_normalizer: 'none', 'duration', or 'sampled_span'.
    secs_per_timestep: Separation in seconds between two consecutive timesteps.
    score_threshold: Score threshold to filter the detections.
    feature_pyramid_config: Feature pyramid configs.

  Returns:
    Class indices in shape (N,).
    Class probabilities in shape (N,).
    Predicted segment start and end times in shape (N, 2).
  """
  assert sampling_strategy in {'random', 'linspace'}

  if feature_pyramid_config is None:
    num_frames = class_probs.shape[0]
  else:
    num_frames = feature_pyramid_config.num_features_level0
  num_classes = class_probs.shape[-1]
  if sampling_strategy == 'random':
    # The default setting is to sample the center clip at test time.
    start_frame_offset = np.maximum(0, (total_frames - num_frames * stride) //
                                    2).astype(np.float32)
  else:  # 'linspace'
    start_frame_offset = 0.0
  if displacement_normalizer == 'duration':
    displacement_multiplier = total_frames
  elif displacement_normalizer == 'sampled_span':
    displacement_multiplier = num_frames * stride
  else:
    displacement_multiplier = 1
  displacements[..., 0] *= -1.0

  if feature_pyramid_config is not None:
    feature_pyramid_downsample_stride = (
        feature_pyramid_config.feature_pyramid_downsample_stride
    )
    displacements_per_level = model_utils.split_pyramid_features(
        displacements,
        feature_pyramid_config.num_features_level0,
        len(feature_pyramid_config.feature_pyramid_levels),
        feature_pyramid_downsample_stride,
        axis=0,
        np_backend=np)
    normalize_displacements_by_downsample_stride = feature_pyramid_config.get(
        'normalize_displacements_by_downsample_stride', False)
  else:
    displacements_per_level = [displacements]
    feature_pyramid_downsample_stride = 1
    normalize_displacements_by_downsample_stride = False

  segments = []
  linspace_frame_indices = np.arange(0, num_frames, dtype=np.float32)
  for level, cur_displacements in enumerate(displacements_per_level):
    cur_downsample_stride = feature_pyramid_downsample_stride**level
    if normalize_displacements_by_downsample_stride:
      cur_displacements *= cur_downsample_stride
    cur_stride = stride * cur_downsample_stride
    if sampling_strategy == 'random':
      frame_indices = np.arange(
          0, num_frames * stride, cur_stride, dtype=np.float32
      )[:, None, None]
      cur_segments = (
          frame_indices
          + cur_displacements * displacement_multiplier
          + start_frame_offset
      )
    else:  # 'linspace'
      frame_indices = linspace_frame_indices[
          range(0, num_frames, cur_downsample_stride)
      ][:, None, None]
      cur_segments = (
          (frame_indices + cur_displacements)
          * (total_frames - 1)
          / (num_frames - 1)
      )
    segments.append(cur_segments)
  segments = np.concatenate(segments, axis=0)
  input_mask = input_mask.astype(bool)
  total_frames = np.full((segments.shape[0], num_classes), total_frames)[
      input_mask
  ]
  segments = segments[input_mask]
  segments[..., 0] = np.maximum(segments[..., 0], 0)
  segments[..., 1] = np.minimum(segments[..., 1], total_frames)
  segments = segments * secs_per_timestep
  class_probs = class_probs[input_mask]
  mask = class_probs >= score_threshold
  class_indices = mask.nonzero()[1]

  return class_indices, class_probs[mask], segments[mask]


# TODO(shenyan): remove code duplication for different tasks.
def get_segments_from_frame_predictions_mr(
    class_probs: np.ndarray,
    displacements: np.ndarray,
    input_mask: np.ndarray,
    caption_mask: np.ndarray,
    total_frames: int,
    stride: int,
    sampling_strategy: str = 'random',
    displacement_normalizer: str = 'duration',
    secs_per_timestep: float = 1.0,
    feature_pyramid_config: Optional[ml_collections.ConfigDict] = None,
) -> Tuple[Array, Array]:
  """Computes predicted segments based on frame predictions.

  We assume that all frame predictions are from one video.

  Args:
    class_probs: A float array representing the class probabilities in shape
      (max_num_captions, num_frames).
    displacements: A float array representing the start/end time displacements
      in shape (max_num_captions, num_frames, 2). All values are greater or
      equal to zero.
    input_mask: A int array representing the input mask in shape (num_frames,).
    caption_mask: A int array representing the caption mask in shape
      (max_num_captions,).
    total_frames: The total number of frames in this video.
    stride: Temporal stride used in sampling the frames
    sampling_strategy: 'random' or 'linspace'.
    displacement_normalizer: 'none', 'duration', or 'sampled_span'.
    secs_per_timestep: Separation in seconds between two consecutive timesteps.
    feature_pyramid_config: Feature pyramid configs.

  Returns:
    Class probabilities in shape (num_captions, N) after masking.
    Predicted segment start and end frame indices in shape
      (num_captions, N, 2) after masking.
  """
  assert sampling_strategy in {'random', 'linspace'}

  if feature_pyramid_config is None:
    num_frames = class_probs.shape[1]
  else:
    num_frames = feature_pyramid_config.num_features_level0
  if sampling_strategy == 'random':
    # The default setting is to sample the center clip at test time.
    start_frame_offset = np.maximum(0, (total_frames - num_frames * stride) //
                                    2).astype(np.float32)
  else:  # 'linspace'
    start_frame_offset = 0.0
  if displacement_normalizer == 'duration':
    displacement_multiplier = total_frames
  elif displacement_normalizer == 'sampled_span':
    displacement_multiplier = num_frames * stride
  else:
    displacement_multiplier = 1
  displacements[..., 0] *= -1.0

  if feature_pyramid_config is not None:
    feature_pyramid_downsample_stride = (
        feature_pyramid_config.feature_pyramid_downsample_stride
    )
    displacements_per_level = model_utils.split_pyramid_features(
        displacements,
        feature_pyramid_config.num_features_level0,
        len(feature_pyramid_config.feature_pyramid_levels),
        feature_pyramid_downsample_stride,
        axis=1,
        np_backend=np)
    normalize_displacements_by_downsample_stride = feature_pyramid_config.get(
        'normalize_displacements_by_downsample_stride', False)
  else:
    displacements_per_level = [displacements]
    feature_pyramid_downsample_stride = 1
    normalize_displacements_by_downsample_stride = False

  segments = []
  linspace_frame_indices = np.arange(0, num_frames, dtype=np.float32)
  for level, cur_displacements in enumerate(displacements_per_level):
    cur_downsample_stride = feature_pyramid_downsample_stride**level
    if normalize_displacements_by_downsample_stride:
      cur_displacements *= cur_downsample_stride
    cur_stride = stride * cur_downsample_stride
    if sampling_strategy == 'random':
      frame_indices = (
          np.arange(0, num_frames * stride, cur_stride,
                    dtype=np.float32)[None, :, None])
      cur_segments = (
          frame_indices + cur_displacements * displacement_multiplier +
          start_frame_offset)
    else:  # 'linspace'
      frame_indices = (
          linspace_frame_indices[range(0, num_frames,
                                       cur_downsample_stride)][None, :, None])
      cur_segments = ((frame_indices + cur_displacements) * (total_frames - 1) /
                      (num_frames - 1))
    segments.append(cur_segments)
  segments = np.concatenate(segments, axis=1)
  caption_mask = caption_mask.astype(bool)
  input_mask = input_mask.astype(bool)
  segments = segments[caption_mask]
  segments = segments[:, input_mask]
  segments[..., 0] = np.maximum(segments[..., 0], 0)
  segments[..., 1] = np.minimum(segments[..., 1], total_frames)
  segments = segments * secs_per_timestep
  class_probs = class_probs[caption_mask]
  class_probs = class_probs[:, input_mask]

  return class_probs, segments


def get_segments_from_frame_predictions_jax(
    class_probs: jnp.ndarray,
    displacements: jnp.ndarray,
    input_mask: jnp.ndarray,
    total_frames: int,
    stride: int,
    sampling_strategy: str = 'random',
    displacement_normalizer: str = 'duration',
    secs_per_timestep: float = 1.0,
    score_threshold: float = float('-inf'),
    feature_pyramid_config: Optional[ml_collections.ConfigDict] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Computes predicted segments based on frame predictions in jnp format.

  We assume that all frame predictions are from one video.

  Args:
    class_probs: A float array representing the class probabilities in shape
      (num_frames, num_classes).
    displacements: A float array representing the start/end time displacements
      in shape (num_frames, num_classes, 2). All values are greater or equal to
      zero.
    input_mask: A int array representing the input mask in shape (num_frames,).
    total_frames: The total number of frames in this video.
    stride: Temporal stride used in sampling the frames
    sampling_strategy: 'random' or 'linspace'.
    displacement_normalizer: 'none', 'duration', or 'sampled_span'.
    secs_per_timestep: Separation in seconds between two consecutive timesteps.
    score_threshold: Score threshold to filter the detections.
    feature_pyramid_config: Feature pyramid configs.

  Returns:
    Class indices in shape (num_frames,), padded ones are filled with -1.
    Class probabilities in shape (num_frames,), padded ones are filled with -1.
    Predicted segment start and end times in shape (num_frames, 2), padded ones
    are filled with -1.
  """
  assert sampling_strategy in {'random', 'linspace'}

  if feature_pyramid_config is None:
    num_frames = class_probs.shape[0]
  else:
    num_frames = feature_pyramid_config.num_features_level0
  num_classes = class_probs.shape[-1]
  if sampling_strategy == 'random':
    # The default setting is to sample the center clip at test time.
    start_frame_offset = jnp.maximum(
        0, (total_frames - num_frames * stride) // 2
    ).astype(jnp.float32)
  else:  # 'linspace'
    start_frame_offset = 0.0
  if displacement_normalizer == 'duration':
    displacement_multiplier = total_frames
  elif displacement_normalizer == 'sampled_span':
    displacement_multiplier = num_frames * stride
  else:
    displacement_multiplier = 1
  displacements = displacements.at[..., 0].multiply(-1.0)

  if feature_pyramid_config is not None:
    feature_pyramid_downsample_stride = (
        feature_pyramid_config.feature_pyramid_downsample_stride
    )
    displacements_per_level = model_utils.split_pyramid_features(
        displacements,
        feature_pyramid_config.num_features_level0,
        len(feature_pyramid_config.feature_pyramid_levels),
        feature_pyramid_downsample_stride,
        axis=0,
        np_backend=jnp,
    )
    normalize_displacements_by_downsample_stride = feature_pyramid_config.get(
        'normalize_displacements_by_downsample_stride', False
    )
  else:
    displacements_per_level = [displacements]
    feature_pyramid_downsample_stride = 1
    normalize_displacements_by_downsample_stride = False

  segments = []
  linspace_frame_indices = jnp.arange(0, num_frames, dtype=jnp.float32)
  for level, cur_displacements in enumerate(displacements_per_level):
    cur_downsample_stride = feature_pyramid_downsample_stride**level
    if normalize_displacements_by_downsample_stride:
      cur_displacements *= cur_downsample_stride
    cur_stride = stride * cur_downsample_stride
    if sampling_strategy == 'random':
      frame_indices = jnp.arange(
          0, num_frames * stride, cur_stride, dtype=jnp.float32
      )[:, None, None]
      cur_segments = (
          frame_indices
          + cur_displacements * displacement_multiplier
          + start_frame_offset
      )
    else:  # 'linspace'
      frame_indices = linspace_frame_indices[
          jnp.arange(0, num_frames, cur_downsample_stride)
      ][:, None, None]
      cur_segments = (
          (frame_indices + cur_displacements)
          * (total_frames - 1)
          / (num_frames - 1)
      )
    segments.append(cur_segments)
  segments = jnp.concatenate(segments, axis=0)
  input_mask = jnp.array(input_mask, dtype=bool)
  total_frames = jnp.array(
      jnp.full((segments.shape[0], num_classes), total_frames)
  )
  segments = segments.at[..., 0].set(jnp.maximum(segments[..., 0], 0))
  segments = segments.at[..., 1].set(
      jnp.minimum(segments[..., 1], total_frames)
  )
  segments = segments * secs_per_timestep
  class_probs = jnp.where(input_mask[:, None], class_probs, -1)
  mask = class_probs >= score_threshold
  inds, class_indices = mask.nonzero(size=mask.shape[0], fill_value=-1)
  class_probs = jnp.where(
      inds >= 0, jnp.take_along_axis(class_probs[:, 0], inds, axis=0), -1
  )
  segments = jnp.where(
      inds[:, None] >= 0,
      jnp.take_along_axis(segments[:, 0], inds[:, None], axis=0),
      -1,
  )

  return class_indices, class_probs, segments
