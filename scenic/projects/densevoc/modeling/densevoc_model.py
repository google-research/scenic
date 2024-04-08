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

"""Implementation of Dense Video Object Captioning (https://arxiv.org/pdf/2306.11729.pdf)."""

import dataclasses
from typing import Any, Dict, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from scenic.projects.baselines.centernet.modeling import iou_assignment
from scenic.projects.baselines.centernet.modeling import roi_head_utils
from scenic.projects.densevoc.modeling import grit
from scenic.projects.densevoc.modeling import tracking_layers
from scenic.projects.densevoc.modeling import tracking_utils

Assignment = iou_assignment.Assignment
ArrayDict = Dict[str, jnp.ndarray]
MetricsDict = Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]

DETECTION_LOSS_KEYS = [
    'pos_loss', 'neg_loss', 'reg_loss',
    'stage0_roi_cls_loss', 'stage0_roi_reg_loss']


class DenseVOCDetector(grit.GRiTDetector):
  """Video-based detector."""
  bg_proposal_thresh: float = 0.1
  track_loss_score_thresh: float = 0.0

  with_tracking: bool = False
  tracking_loss_weight: float = 1.0
  tracking_iou_thresh: float = 0.6
  propagate_asso_scores: float = -1.

  caption_with_track: bool = False
  use_tracked_object_features: bool = False
  asso_windows: int = -1

  hard_tracking: bool = False
  hard_tracking_test: bool = False
  tracking_score_thresh: float = 0.3
  max_num_tracks: int = 4
  hard_tracking_frames: int = 6

  flatten_video_input: bool = False
  with_global_video_caption: bool = False
  global_text_loss_weight: float = 1.0
  num_frames: int = -1
  skip_global_caption_test: bool = False
  frame_fuse_fn: str = 'concat'

  use_loss_masks: bool = False
  consistent_soft_track: bool = False

  def setup(self):
    super().setup()
    if self.with_tracking:
      self.tracking_layers = tracking_layers.GTRAssoHead()
      self.tracking_transformer = tracking_layers.GTRTransformer()

  def flatten_time_to_batch(
      self, inputs, gt_boxes, gt_classes, gt_text_tokens, gt_track_ids,
      image_caption_tokens):
    b, t = inputs.shape[0], inputs.shape[1]
    inputs = inputs.reshape(
        (b * t,) + inputs.shape[2:])
    if gt_boxes is not None:
      gt_boxes = gt_boxes.reshape((b * t,) + gt_boxes.shape[2:])
    if gt_classes is not None:
      gt_classes = gt_classes.reshape((b * t,) + gt_classes.shape[2:])
    if gt_text_tokens is not None:
      gt_text_tokens = gt_text_tokens.reshape(
          (b * t,) + gt_text_tokens.shape[2:])
    if gt_track_ids is not None:
      gt_track_ids = gt_track_ids.reshape((b * t,) + gt_track_ids.shape[2:])
    if image_caption_tokens is not None:
      image_caption_tokens = image_caption_tokens.reshape(
          (b * t,) + image_caption_tokens.shape[2:])
    return (inputs, gt_boxes, gt_classes, gt_text_tokens,
            gt_track_ids, image_caption_tokens)

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               gt_boxes: Optional[jnp.ndarray] = None,
               gt_classes: Optional[jnp.ndarray] = None,
               gt_text_tokens: Optional[jnp.ndarray] = None,
               gt_track_ids: Optional[jnp.ndarray] = None,
               video_caption_tokens: Optional[jnp.ndarray] = None,
               image_caption_tokens: Optional[jnp.ndarray] = None,
               train: bool = False,
               preprocess: bool = False,
               *,
               padding_mask: Optional[jnp.ndarray] = None,
               debug: bool = False) -> Any:
    """Applies DenseVOC model on the input.

    Args:
      inputs: array of the preprocessed input images, in shape B x H x W x 3.
      gt_boxes: B x N x 4. Only used in training.
      gt_classes: B x N. Only used in training.
      gt_text_tokens: B x N x max_caption_length. Only used in training.
      gt_track_ids: B x N. int; Only used in training.
      video_caption_tokens: 1 x N x max_caption_length
      image_caption_tokens: B x max_num_captions x max_caption_length
      train: Whether it is training.
      preprocess: If using the build-in preprocessing functions on inputs.
      padding_mask: Binary matrix with 0 at padded image regions.
      debug: Whether the debug mode is enabled. debug=True enables model
        specific logging/storing some values using jax.host_callback.

    Returns:
      if train == False:
        detections: a dict with the following keys:
          'detection_boxes': batch_size x max_detections x 4
          'detection_scores': batch_size x max_detections
          'detection_classes': batch_size x max_detections, 0-based int.
          'object_features': batch_size x max_detections x 196 x 256
          'begin_tokens': batch_size x max_detections x max_caption_length
      if train == True:
        TODO(zhouxy): complete after implementing training.
    """
    # the input is a video, reshape it.
    if self.flatten_video_input and inputs.ndim == 5:
      # assert inputs.ndim == 5
      (inputs, gt_boxes, gt_classes, gt_text_tokens,
       gt_track_ids, unused_image_caption_tokens) = self.flatten_time_to_batch(
           inputs, gt_boxes, gt_classes, gt_text_tokens, gt_track_ids,
           image_caption_tokens)
    if preprocess:
      inputs = self.preprocess(inputs, padding_mask)
    backbone_features = self.backbone(inputs, train=train)
    outputs = self.proposal_generator(
        backbone_features, train=train)

    detections, metrics, rpn_features, gt_classes, image_shape = (
        self.forward_detection(
            inputs, outputs, backbone_features, gt_classes, gt_boxes,
            train=train, debug=debug))

    strides = sorted(self.roi_heads.input_strides.items(), key=lambda x: x[1])
    features = [rpn_features[s[0]] for s in strides]  # Sorted features
    outputs, aux = self.forward_object_caption(
        detections, metrics, outputs, features,
        gt_classes, gt_boxes, gt_text_tokens, train=train)
    (metrics, object_features, last_proposals, unused_text_tokens,
     matched_text, matched) = aux

    # The proposals have both positive and negative boxes.
    # The following losses are only applied on positive objects.
    # Marking the background objects by setting their features to 0.
    # The followup process will ignore them.
    if train:
      object_features = object_features * matched[:, :, None, None, None]
    else:
      object_features = object_features * (detections[
          'detection_scores'][:, :, None, None] >= self.bg_proposal_thresh)
      detections['object_features'] = object_features
    res = self.object_feat_res

    # Forward the tracking head and get the "asso_score" outputs.
    if self.with_tracking:
      outputs = self.forward_tracking(
          last_proposals,
          object_features.reshape(
              object_features.shape[:2] + (res, res, -1)),
          gt_boxes, gt_track_ids,
          outputs, metrics, debug=debug)

    # use the "asso_score" outputs to produce new captions.
    if self.caption_with_track:
      assert self.with_tracking
      outputs = self.forward_caption_with_track(
          object_features.reshape(
              object_features.shape[:2] + (res, res, -1)),
          matched_text, matched,
          outputs, metrics, train=train, debug=debug)

    if self.with_global_video_caption and not (
        self.skip_global_caption_test and not train):
      outputs = self.forward_global_video_caption(
          features, video_caption_tokens,
          image_shape, outputs, metrics, train=train, debug=debug,
      )

    return outputs

  def forward_global_video_caption(
      self, features, video_caption_tokens,
      image_shape, outputs, metrics,
      train=False, debug=False):
    """Forward global video captioning.

    Args:
      features: list of arrays: FPN features.
      video_caption_tokens: 1 x N x max_caption_length. Only used in training.
      image_shape: B x 2, in order (height, width).
      outputs: dict of arrays.
      metrics: dict of floats.
      train: bool.
      debug: bool.
    Returns:
      updated outputs with additional keys:
        when train==False:
          'global_begin_tokens': (video_batch_size, max_cap_len)
          'global_features': (video_batch_size, num_tokens, C)
        when train==True:
          'global_text_loss': float
          'global_num_valid_tokens': float
    """
    del debug
    batch_size = image_shape.shape[0]
    assert self.num_frames > 0, self.num_frames
    assert batch_size % self.num_frames == 0, batch_size
    video_batch_size = batch_size // self.num_frames
    # assert video_batch_size == 1
    feat_res = self.object_feat_res
    image_box = jnp.concatenate(
        [jnp.zeros((batch_size, 2), jnp.float32),
         image_shape[:, 1:2], image_shape[:, 0:1]],
        axis=1)[:, None]  # (batch_size, 1, 4)
    global_features = self.roi_heads.roi_align(
        features, image_box, feat_res,
    )  # (batch_size, 1, feat_res, feat_res, C)
    if self.frame_fuse_fn == 'concat':
      global_features = global_features.reshape(
          video_batch_size, self.num_frames * feat_res ** 2, -1,
      )  # (video_batch_size, self.num_frames * feat_res ** 2, C)
    else:
      assert self.frame_fuse_fn == 'mean', self.frame_fuse_fn
      global_features = global_features.reshape(
          video_batch_size, self.num_frames, feat_res ** 2, -1,
      ).mean(axis=1)  # (video_batch_size, feat_res ** 2, C)
    if video_caption_tokens is None:  # evaluation
      assert not train
      global_text_tokens = jnp.full(
          (video_batch_size, self.max_caption_length),
          self.end_token_id,
          dtype=jnp.int32)  # (video_batch_size, max_cap_len)
      global_text_tokens = global_text_tokens.at[:, 0].set(
          self.begin_token_id)  # (video_batch_size, max_cap_len)
      global_text_feature = self.text_decoder(
          global_text_tokens,
          global_features,
          train=train,
      )  # (video_batch_size, max_caption_length, vocab_size)
      del global_text_feature
      outputs['global_begin_tokens'] = global_text_tokens
      outputs['global_features'] = global_features
    else:  # training
      assert train
      # video_caption_tokens: 1 x N x max_caption_length
      # Expand feature (batch_size, 1, object_feat_res, object_feat_res, C)
      num_caption_per_video = video_caption_tokens.shape[1]
      text_batch_size = video_batch_size * num_caption_per_video
      video_caption_tokens = video_caption_tokens.reshape(
          text_batch_size, self.max_caption_length)
      global_features = jnp.broadcast_to(
          global_features[:, None],
          (video_batch_size, num_caption_per_video) + global_features.shape[1:])
      global_features = global_features.reshape(
          (text_batch_size,) + global_features.shape[2:])
      text_outputs = self.text_decoder(
          video_caption_tokens, global_features, train=train,
      )  # (text_batch_size, max_caption_length, vocab_size)
      mask = (video_caption_tokens[:, 0] != self.end_token_id) & (
          video_caption_tokens[:, 0] > 0)
      text_loss, num_valid_tokens = self.text_loss(
          text_outputs,
          video_caption_tokens,
          mask=mask)
      metrics['global_text_loss'] = text_loss
      metrics['global_num_valid_tokens'] = num_valid_tokens
      outputs['metrics'] = metrics
    return outputs

  def forward_caption_with_track(
      self, object_features, matched_text, matched,
      outputs, metrics, train=False, debug=False):
    """Generate captions using augmented features from tracking.

    Args:
      object_features: (batch_size, num_objs, res, res, D)
      matched_text: (batch_size, num_objs, max_cap_len)
      matched: (batch_size, num_objs): 1: matched; 0: not matched.
      outputs: dict of arrays.
      metrics: dict of floats.
      train: bool.
      debug: bool.
    Returns:
      updated outputs
    """

    batch_size, num_objs = object_features.shape[:2]
    num_tot_objs = batch_size * num_objs
    asso_scores = jax.lax.stop_gradient(
        outputs['asso_scores'][0])  # (num_tot_objs, num_tot_objs)
    track_ids = tracking_utils.greedy_extract_trajectories(
        asso_scores,
        num_frames=batch_size,
        thresh=self.tracking_score_thresh,
    )  # (num_tot_objs,)
    outputs['track_ids'] = track_ids.reshape(batch_size, num_objs)
    if self.asso_windows >= 0:
      # Do not associate with too faraway frames.
      windows_mask = jnp.abs((jnp.arange(num_tot_objs)[:, None] // num_objs - (
          jnp.arange(num_tot_objs)[None] // num_objs))) <= self.asso_windows
      # Do not associate with other objects in the same frame.
      # With this and asso_windows == 0, there should be no association at all.
      frame_mask = (jnp.arange(num_tot_objs)[:, None] // num_objs != (
          jnp.arange(num_tot_objs)[None] // num_objs)) | jnp.eye(
              num_tot_objs, dtype=bool)
      asso_scores = asso_scores * (windows_mask & frame_mask)

    if self.hard_tracking:  # "hard" tracking
      track_feats, track_feature_mask, track_matrix = (
          tracking_utils.get_track_features(
              object_features, track_ids,
              max_num_tracks=self.max_num_tracks,
              hard_tracking_frames=self.hard_tracking_frames))
      if not train:
        if self.hard_tracking_test:
          outputs['track_features'] = track_feats
          outputs['track_feature_mask'] = track_feature_mask
      else:
        gt_track_texts = tracking_utils.get_track_texts(
            matched_text, track_matrix,
            hard_tracking_frames=self.hard_tracking_frames)
        text_outputs = self.text_decoder(
            gt_track_texts, track_feats,
            feature_valid_mask=track_feature_mask,
            train=train,
        )  # (max_num_tracks, max_caption_length, vocab_size)
        text_loss, _ = self.text_loss(
            text_outputs,
            gt_track_texts,
            track_feature_mask.any(axis=1))
        metrics['tracked_text_loss'] = text_loss
        outputs['metrics'] = metrics
    else:  # "soft" tracking
      if self.consistent_soft_track:
        asso_scores = (track_ids[None, :] == track_ids[:, None]).astype(
            jnp.float32)
      normalized_asso_scores = asso_scores / (
          asso_scores.sum(axis=1)[:, None] + 1e-6)
      if debug:
        outputs['normalized_asso_scores'] = normalized_asso_scores
      tracked_object_features = jnp.matmul(
          normalized_asso_scores,
          object_features.reshape(batch_size * num_objs, -1),
      )  # (num_tot_objs, res * res * D)
      if not train:
        outputs['tracked_object_features'] = tracked_object_features.reshape(
            batch_size, num_objs, -1, object_features.shape[-1],
        )
      else:
        text_batch_size = batch_size * num_objs
        text_outputs = self.text_decoder(
            matched_text.reshape(text_batch_size, self.max_caption_length),
            tracked_object_features.reshape(
                text_batch_size, -1, object_features.shape[-1]),
            train=train,
        )  # (text_batch_size, max_caption_length, vocab_size)
        text_loss, _ = self.text_loss(
            text_outputs,
            matched_text.reshape(text_batch_size, self.max_caption_length),
            matched.reshape(text_batch_size,))
        metrics['tracked_text_loss'] = text_loss
        outputs['metrics'] = metrics
    return outputs

  def forward_tracking(
      self, last_proposals, object_features, gt_boxes, gt_track_ids,
      outputs, metrics, debug=False):
    """Forward tracking head.

    The images from the batch are from the same video.

    Args:
      last_proposals: (batch_size, num_objs, 4)
      object_features: (batch_size, num_objs, res, res, D)
      gt_boxes: (batch_size, num_gt_objs, 4)
      gt_track_ids: (batch_size, num_gt_objs)
      outputs: dict of arrays.
      metrics: dict of floats.
      debug: bool.
    Returns:
      outputs
    """
    asso_features = self.tracking_layers(
        object_features)  # (batch_size, num_objs, D)
    num_frames, num_objs = asso_features.shape[:2]
    num_tot_objs = num_frames * num_objs
    asso_features = asso_features.reshape(
        1, num_tot_objs, -1)  # (1, num_tot_objs, D)
    asso_scores = self.tracking_transformer(
        asso_features)  # (1, num_tot_objs, num_tot_objs)
    outputs['asso_scores'] = nn.sigmoid(asso_scores)
    # TODO(zhouxy): check if we can merge with valid_mask below.
    valid_object_mask = ((
        object_features ** 2).sum(axis=(2, 3, 4)) > 0).reshape(num_tot_objs)
    outputs['asso_scores'] = outputs['asso_scores'] * (
        valid_object_mask[None, None, :] * valid_object_mask[None, :, None])
    if gt_track_ids is not None:  # training
      matched_ids = self.match_tracking_ids(
          last_proposals, gt_boxes, gt_track_ids,
          self.tracking_iou_thresh,
      )  # (batch_size, num_objs)
      matched_ids = matched_ids.reshape(num_tot_objs)
      tracking_gt = (matched_ids[None, :] == matched_ids[
          :, None])[None]  # (1, num_tot_objs, num_tot_objs)
      valid_mask = (matched_ids[None, :] > 0) & (
          matched_ids[:, None] > 0)[None]  # (1, num_tot_objs, num_tot_objs)
      tracking_loss = optax.sigmoid_binary_cross_entropy(
          asso_scores, tracking_gt.astype(jnp.float32),
      ) * valid_mask.astype(jnp.float32)  # (1, num_tot_objs, num_tot_objs)
      tracking_loss = tracking_loss.sum() / (valid_mask.sum() + 1e-6)
      metrics['tracking_loss'] = tracking_loss
      outputs['metrics'] = metrics
      if debug:
        metrics['tracking_gt'] = tracking_gt
        metrics['tracking_mask'] = valid_mask
    return outputs

  def match_tracking_ids(
      self, proposals, gt_boxes, gt_track_ids, thresh):
    """Match proposals and their texts based on bounding box IoU.

    Args:
      proposals: Boxes with array (B, num_objs, 4).
      gt_boxes: Boxes with array (B, max_gt_boxes, 4).
      gt_track_ids: (B, max_gt_boxes). 0 for padded objects.
      thresh: float.
    Returns:
      matched_ids: shape (B, num_objs).
    """
    def _impl(proposals, gt_boxes, gt_track_ids):
      iou = roi_head_utils.pairwise_iou(gt_boxes, proposals)
      matched_idxs, assignments = iou_assignment.label_assignment(
          iou, [thresh], [Assignment.NEGATIVE, Assignment.POSITIVE])
      matched_classes = gt_track_ids[matched_idxs]
      matched_classes = jnp.where(
          assignments != Assignment.POSITIVE, 0, matched_classes)
      return matched_classes
    matched_ids = jax.vmap(_impl, in_axes=0)(proposals, gt_boxes, gt_track_ids)
    return matched_ids

  def update_object_feature_with_track(self, predictions, t):
    mask = None
    tracked_features = predictions['object_features']
    assert not (self.use_tracked_object_features and self.hard_tracking_test), (
        'Soft and hard tracking can not be used together.')
    if self.use_tracked_object_features:
      tracked_features = predictions['tracked_object_features']
    elif self.hard_tracking_test:
      num_objs = predictions['object_features'].shape[1]
      num_tracks = predictions['track_features'].shape[0]
      predictions['track_ids'] = predictions['track_ids'].reshape(t, num_objs)
      track_ids = jnp.maximum(jnp.minimum(
          predictions['track_ids'] - 1, num_tracks - 1), 0)  # (t, num_objs)
      object_track_feature = jnp.take_along_axis(
          predictions['track_features'][None],
          track_ids[:, :, None, None], axis=1)  # (t, num_objs, t * res**2, D)
      mask = jnp.take_along_axis(
          predictions['track_feature_mask'][None],
          track_ids[:, :, None], axis=1)  # (t, num_objs, t * res**2)
      res2 = predictions['object_features'].shape[2]
      feature_dim = predictions['object_features'].shape[3]
      hard_tracking_frames = object_track_feature.shape[2] // res2
      tracked_features = jnp.where(
          predictions['track_ids'][:, :, None, None] > 0,
          object_track_feature,
          jnp.broadcast_to(
              predictions['object_features'][:, :, None],
              (t, num_objs, hard_tracking_frames, res2, feature_dim),
          ).reshape(t, num_objs, hard_tracking_frames * res2, feature_dim),
      )
      single_mask = jnp.concatenate([
          jnp.ones((t, num_objs, res2), dtype=bool),
          jnp.zeros(
              (t, num_objs, (hard_tracking_frames - 1) * res2), dtype=bool),
      ], axis=2)
      mask = jnp.where(
          predictions['track_ids'][:, :, None] > 0, mask, single_mask)
    return tracked_features, mask

  def loss_function(
      self,
      outputs: Any,
      batch: Any,
  ):
    """Loss functions.

    Args:
      outputs: dict of arrays.
      batch: dict that has 'inputs' and 'label' (ground truth).

    Returns:
      total_loss: Total loss weighted appropriately.
      metrics: auxiliary metrics for debugging and visualization.
    """
    if self.flatten_video_input:
      (_, batch['label']['boxes'], batch['label']['labels'],
       batch['label']['text_tokens'], batch['label']['track_ids'],
       batch['label']['image_caption_tokens']) = self.flatten_time_to_batch(
           batch['inputs'], batch['label']['boxes'], batch['label']['labels'],
           batch['label']['text_tokens'], batch['label']['track_ids'],
           batch['label']['image_caption_tokens'])

    detection_loss, metrics = super(
        DenseVOCDetector, self).loss_function(outputs, batch)
    del metrics['total_loss']
    if self.use_loss_masks:
      det_loss_mask = (batch['loss_masks']['det_loss_mask'] > 0).any(
          ).astype(jnp.float32)
      objcap_loss_mask = (batch['loss_masks']['objcap_loss_mask']).any(
          ).astype(jnp.float32)
      vidcap_loss_mask = (batch['loss_masks']['vidcap_loss_mask'] > 0).any(
          ).astype(jnp.float32)
      track_loss_mask = (batch['loss_masks']['track_loss_mask'] > 0).any(
          ).astype(jnp.float32)
      trackcap_loss_mask = (batch['loss_masks']['trackcap_loss_mask'] > 0).any(
          ).astype(jnp.float32)
    else:
      det_loss_mask = (batch['label']['boxes'] > 0).any().astype(jnp.float32)
      objcap_loss_mask = (
          (batch['label']['text_tokens'] > 0)
          & (batch['label']['text_tokens'] != self.end_token_id)
          & (batch['label']['text_tokens'] != self.begin_token_id)
      ).any().astype(jnp.float32)
      vidcap_loss_mask = (
          (batch['label']['video_caption_tokens'] > 0)
          & (batch['label']['video_caption_tokens'] != self.end_token_id)
          & (batch['label']['video_caption_tokens'] != self.begin_token_id)
      ).any().astype(jnp.float32)
      track_loss_mask = (batch['label']['track_ids'] > 0).any().astype(
          jnp.float32)
      trackcap_loss_mask = track_loss_mask * objcap_loss_mask
    for key in DETECTION_LOSS_KEYS:
      if key in metrics:
        metrics[key] = metrics[key] * det_loss_mask
    total_loss = detection_loss * det_loss_mask

    metrics['text_loss'] *= objcap_loss_mask
    total_loss += self.text_loss_weight * metrics['text_loss']

    if self.with_global_video_caption:
      metrics['global_text_loss'] *= vidcap_loss_mask
      total_loss += self.global_text_loss_weight * metrics['global_text_loss']

    if self.with_tracking:
      metrics['tracking_loss'] *= track_loss_mask
      total_loss += self.tracking_loss_weight * (metrics['tracking_loss'])

    if self.caption_with_track:
      metrics['tracked_text_loss'] *= trackcap_loss_mask
      total_loss += self.text_loss_weight * (metrics['tracked_text_loss'])

    metrics['total_loss'] = total_loss
    return total_loss, metrics


class DenseVOCModel(grit.GRiTModel):
  """Scenic Model Wrapper."""

  def build_flax_model(self):
    fields = set(x.name for x in dataclasses.fields(DenseVOCDetector))
    config_dict = {
        k: v for k, v in self.config.model.items() if k in fields}
    return DenseVOCDetector(**config_dict)
