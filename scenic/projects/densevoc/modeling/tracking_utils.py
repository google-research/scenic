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

"""Tracking utils."""

import jax
import jax.numpy as jnp


def greedy_extract_trajectories(
    asso_scores, num_frames, thresh=0.3):
  """Greedily convert an association matrics to discrete tracking IDs.

  Decribed in Algorithm 1 of https://arxiv.org/pdf/2306.11729.pdf.

  Args:
    asso_scores: (num_tot_objs, num_tot_objs)
    num_frames: int
    thresh: float
  Returns:
    ids: (num_tot_objs,)
  """
  num_tot_objs = asso_scores.shape[0]
  assert num_frames > 0 and num_tot_objs % num_frames == 0
  num_objs = num_tot_objs // num_frames
  # Don't merge objects in the same frame.
  mask = (jnp.arange(num_tot_objs)[:, None] // num_objs != (
      jnp.arange(num_tot_objs)[None] // num_objs))
  mask = mask | jnp.eye(num_tot_objs, dtype=bool)
  asso_scores = asso_scores * mask
  ids = jnp.zeros(num_tot_objs, dtype=jnp.int32)
  def body_fn(state):
    asso_scores, ids = state
    can_merge = asso_scores >= thresh  # (num_tot_objs, num_tot_objs)
    num_merges = can_merge.sum(axis=1)  # num_tot_objs
    ind = num_merges.argmax()  # int
    id_count = ids.max() + 1  # int
    merge_inds = can_merge[ind]  # num_tot_objs
    # Don't merge with two objects in the same frame.
    max_ind_in_frame = asso_scores[ind].reshape(
        num_frames, num_objs).argmax(axis=1)  # num_frames
    is_max_score = jax.nn.one_hot(max_ind_in_frame, num_objs).astype(bool)
    merge_inds = merge_inds & is_max_score.reshape(num_tot_objs)
    ids = ids + merge_inds * id_count  # num_tot_objs
    asso_scores = asso_scores * (1. - merge_inds[None]) * (
        1. - merge_inds[:, None])
    return (asso_scores, ids)
  _, ids = jax.lax.while_loop(
      lambda s: s[0].max() >= thresh,
      body_fn,
      (asso_scores, ids))
  ids = jax.lax.stop_gradient(ids)
  return ids


def get_track_features(
    object_features, track_ids, max_num_tracks, hard_tracking_frames):
  """Features for each track.

  Args:
    object_features: (batch_size, num_objs, res, res, D). Note we assume all
      images in the batch are from the same video.
    track_ids: (num_tot_objs,). num_tot_objs = batch_size * num_objs.
    max_num_tracks: int
    hard_tracking_frames: int
  Returns:
    track_feats: (max_num_tracks, num_frames * res * res, D)
    track_feature_mask: (max_num_tracks, num_frames * res * res)
    track_matrix: (max_num_tracks, num_tot_objs)
  """
  batch_size, num_objs, res = object_features.shape[:3]
  num_tot_objs = batch_size * num_objs
  num_input_frames = batch_size

  padded_object_features = object_features.reshape(
      num_tot_objs, res ** 2, -1)
  padded_object_features = jnp.concatenate(
      [padded_object_features,
       jnp.zeros((num_tot_objs, res ** 2, object_features.shape[-1]))],
      axis=0)  # (num_tot_objs + 1, res * res, D)
  track_matrix = (jnp.arange(max_num_tracks) + 1)[
      :, None] == track_ids[None, :]  # (max_num_tracks, num_tot_objs)
  track_feats = []
  for i in range(max_num_tracks):
    track_ind = jnp.nonzero(
        track_matrix[i], size=num_input_frames, fill_value=-1,
    )[0]  # (num_input_frames,)

    num_valid_frames = (track_ind >= 0).sum()
    track_ind = jnp.where(
        num_valid_frames > hard_tracking_frames,
        track_ind[jnp.linspace(
            0, num_valid_frames, hard_tracking_frames,
            endpoint=False, dtype=jnp.int32)],
        track_ind[:hard_tracking_frames],
    )  # (hard_tracking_frames,)

    track_feat = jnp.take_along_axis(
        padded_object_features, track_ind[:, None, None], axis=0,
    )  # (num_frames, res * res, D)
    track_feats.append(track_feat)
  track_feats = jnp.stack(
      track_feats, axis=0)  # (max_num_tracks, num_frames, res ** 2, D)
  track_feats = track_feats.reshape(
      max_num_tracks, hard_tracking_frames * res ** 2, -1,
  )  # (max_num_tracks, hard_tracking_frames * res * res, D)
  track_feature_mask = jax.lax.stop_gradient(
      (track_feats ** 2).max(axis=-1) > 0
  )  # (max_num_tracks, tracking_frames * res * res)
  return track_feats, track_feature_mask, track_matrix


def get_track_texts(matched_text, track_matrix, hard_tracking_frames):
  """Texts for each track.

  Args:
    matched_text: (batch_size, num_objs, max_cap_len)
    track_matrix: (max_num_tracks, num_tot_objs)
    hard_tracking_frames: int
  Returns:
    track_texts: (max_num_tracks, max_cap_len)
  """

  batch_size, num_objs = matched_text.shape[:2]
  max_num_tracks, num_tot_objs = track_matrix.shape
  assert num_tot_objs == batch_size * num_objs

  padded_matched_text = jnp.concatenate(
      [matched_text.reshape(num_tot_objs, -1),
       jnp.zeros((num_tot_objs, matched_text.shape[-1]), jnp.int32)],
      axis=0)  # (num_tot_objs + 1, max_cap_len)
  track_texts = []
  for i in range(max_num_tracks):
    track_ind = jnp.nonzero(
        track_matrix[i], size=batch_size, fill_value=-1,
    )[0]  # (num_frames,)
    num_valid_frames = (track_ind >= 0).sum()
    track_ind = jnp.where(
        num_valid_frames > hard_tracking_frames,
        track_ind[jnp.linspace(
            0, num_valid_frames, hard_tracking_frames,
            endpoint=False, dtype=jnp.int32)],
        track_ind[:hard_tracking_frames],
    )  # (hard_tracking_frames,)

    track_text = jnp.take_along_axis(
        padded_matched_text, track_ind[:, None], axis=0,
    )  # (num_frames, max_cap_len)
    # track_text_count: how many objects in the track have the same text.
    track_text_count = ((
        (track_text[None, :] - track_text[:, None]) ** 2).sum(
            axis=-1) == 0).sum(axis=1)  # (num_frames,)
    track_text_count = track_text_count * (
        track_ind >= 0)  # remove padding
    track_text = track_text[track_text_count.argmax()]
    track_texts.append(track_text)
  track_texts = jnp.stack(
      track_texts, axis=0)  # (max_num_tracks, max_cap_len)
  return track_texts
