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

"""Streaming caption model."""
import dataclasses
from typing import Any, Optional

from absl import logging
from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from scenic.projects.streaming_dvc.modeling import model as elcap_model
from scenic.projects.streaming_dvc.modeling import streaming_utils


class StreamingCaptioningFlaxModel(elcap_model.CaptioningFlaxModel):
  """Streaming captioning model.

  Attributes:
    streaming_buffer_size: int; constent memory size used in ToMe-based memory
      modules.
    streaming_method: str;
    ema_decay: float; parameter if the streaming_method == 'ema'
  """
  streaming_buffer_size: int = -1
  streaming_method: str = 'none'
  ema_decay: float = 0.9
  kmeans_num_iters: int = -1

  @nn.compact
  def __call__(
      self, images,
      context_tokens=None,
      gt_text_tokens=None,
      preprocess=True, train=False, debug=False):
    """forward model. See CaptioningFlaxModel for args."""
    del debug
    assert self.num_frames > 0
    assert images.ndim == 5
    images = images.reshape(
        (images.shape[0] * images.shape[1],) + images.shape[2:])

    if preprocess:
      images = self.preprocess(images)

    visual_features = self.get_visual_features(
        images, train=train)  # (video_batch_size, num_tokens, dim)
    logging.info('Visual features: %s', visual_features.shape)

    visual_features = self.get_streaming_features(
        visual_features, train=train,
    )  # (video_batch_size, new_num_vis_tokens, proj_dim)
    logging.info('Streaming features: %s', visual_features.shape)

    # maybe_project_visual_feature is only used for BLIP2 models.
    visual_features = self.maybe_project_visual_feature(
        visual_features, train=train,
    )  # (video_batch_size, num_vis_tokens, proj_dim)
    logging.info('Streaming features after proj.: %s', visual_features.shape)

    text_tokens, visual_features, context_tokens = (
        self.get_text_tokens_and_pad_visual_features(
            visual_features, gt_text_tokens, context_tokens))
    # text_tokens: (text_batch_size, max_cap_len)
    # visual_features: (text_batch_size, new_num_vis_tokens, proj_dim)
    # context_tokens: (text_batch_size, num_context_tokens)
    logging.info('Text tokens: %s \nVisual features: %s\n Context tokens: %s',
                 text_tokens.shape, visual_features.shape,
                 context_tokens.shape if context_tokens is not None else 'None')

    text_outputs = self.textual(
        text_tokens,
        visual_features,
        context_tokens=context_tokens,
        train=train,
    )  # (text_batch_size, max_cap_len, vocab_size)
    logging.info('Text outputs: %s', text_outputs.shape)

    if train:
      ret = {'text_outputs': text_outputs}
    else:
      ret = {
          'visual_features': visual_features,
          'begin_tokens': text_tokens,
          'context_tokens': context_tokens,
          'text_outputs': text_outputs,
      }
    return ret

  def get_streaming_features(self, features, train):
    """Get streaming features.

    Args:
      features: (video_batch_size, num_tot_tokens, dim)
      train: bool
    Returns:
      streaming_features: (video_batch_size, num_streaming_tokens, dim)
    """
    # NOTE: We can also implement this under
    # CaptioningFlaxModel.pool_video_feature. Put them here in a separate
    # StreamingCaptioningFlaxModel to make it less entangled with existing code.
    # The behaviours of temporal_mean_pool/ spatial_mean_pool are the same as
    # setting "frame_fuse_fn" in CaptioningFlaxModel.
    unused_video_batch_size, _, dim = features.shape

    del train
    def streaming_feature_extractor(feature):
      # Shape of feature is [n_total_tokens, dim]
      if self.streaming_method == 'temporal_mean_pool':
        streaming_feature = feature.reshape(
            self.num_frames, -1, dim).mean(axis=0)  # (hw, dim)
      elif self.streaming_method == 'spatial_mean_pool':
        streaming_feature = feature.reshape(
            self.num_frames, -1, dim).mean(axis=1)  # (t, dim)
      elif self.streaming_method == 'ema':
        streaming_feature = feature.reshape(
            self.num_frames, -1, dim)  # (t, hw, dim)
        buffer = streaming_feature[0]
        for t in range(1, self.num_frames):
          buffer = buffer * self.ema_decay + streaming_feature[t] * (
              1. - self.ema_decay)
        streaming_feature = buffer
      elif self.streaming_method == 'adjacent_tome':
        assert self.streaming_buffer_size > 0
        buffer = feature[:self.streaming_buffer_size]
        weights = jnp.ones((buffer.shape[0],), dtype=jnp.int32)
        streaming_feature = streaming_utils.adjacent_merge(
            buffer, feature[self.streaming_buffer_size:], weights=weights)[0]
      elif self.streaming_method == 'kmeans':
        assert self.kmeans_num_iters > 0
        centers = feature[:self.streaming_buffer_size]
        weights = jnp.ones((feature.shape[0],), dtype=jnp.int32)
        streaming_feature = streaming_utils.kmeans(
            centers, feature, weights=weights,
            num_iters=self.kmeans_num_iters)[0]
      else:
        assert self.streaming_method == 'none'
        streaming_feature = feature
      return streaming_feature

    streaming_features = jax.vmap(
        streaming_feature_extractor,
        in_axes=0, out_axes=0, axis_name='batch')(features)
    return streaming_features


class StreamingCaptioningModel(elcap_model.CaptioningModel):
  """Scenic Model Wrapper."""

  def get_dict_from_config(self):
    config_dict = super().get_dict_from_config()
    config_dict.update(dict(
        streaming_buffer_size=self.config.model.get(
            'streaming_buffer_size', -1),
        streaming_method=self.config.model.get('streaming_method', 'none'),
        ema_decay=self.config.model.get('ema_decay', 0.9),
        kmeans_num_iters=self.config.model.get('kmeans_num_iters', -1),
    ))
    return config_dict

  def build_flax_model(self):
    return StreamingCaptioningFlaxModel(**self.get_dict_from_config())


class DenseStreamingCaptioningFlaxModel(elcap_model.CaptioningFlaxModel):
  """Streaming captioning model with intermediate outputs.

  Attributes:
    num_dense_outputs: int; the number of intermediate outputs. This changes
      the output shape.
    streaming_buffer_size: int; constent memory size used in ToMe-based memory
      modules.
    streaming_method: str;
    ema_decay: float; parameter if the streaming_method == 'ema'
    early_segments_as_context: bool; If True (which is the full case), we also
      provide supervisions in earlier checkpoints as context. Here each
      checkpoint is in charge of segments ending between last checkpoint to this
      checkpoint. If it is False, every checkpoint is in charge of all captions
      from 0 to the checkpoint step. Here we can study the effect of
      intermediate supervision.
    normalize_early_timestamps: if False, we just split the original dense
      captioning segments into intermediate checkpoints as is; if True, we
      normalize the timestamps in early checkpoits to between 0 to num_bins,
      so that each intermediate checkpoint is a full densecaptioning task.
    copy_context: only used when early_segments_as_context is True. When True,
      always predict the context tokens in predictions.
    dense_outputs_weight: List of floats, with length self.num_dense_outputs.
      Setting different loss weights to different decoding points.
    remove_segments_from_wrong_checkpoint: bool
    streaming_feature_implementation: str; different implementations (legacy)
      of streaming functions.
    no_timestamp_in_context: bool; By default the time tokens are in the
      prefix from earlier decoding point. Remove them if this is True.
    num_dense_outputs_test: The number of intermediate outputs when testing.
      This can be different to the number used during training.
    kmeans_num_iters: int; parameter for k-means streaming method.
    ttm_output: parameters for ttm streaming method.
    ttm_config: configs for ttm streaming method.
    no_momentum_in_memory: bool; if we want to use the momentum term in memory.
  """
  num_dense_outputs: int = -1
  streaming_buffer_size: int = -1
  streaming_method: str = 'none'
  ema_decay: float = 0.9
  early_segments_as_context: bool = False
  normalize_early_timestamps: bool = False
  copy_context: bool = False
  dense_outputs_weight: Any = dataclasses.field(default_factory=tuple)
  remove_segments_from_wrong_checkpoint: bool = False
  streaming_feature_implementation: str = 'fixed_checkpoints'
  no_timestamp_in_context: bool = False
  num_dense_outputs_test: int = -1
  kmeans_num_iters: int = 2
  ttm_output: str = 'output'
  ttm_config: Optional[ml_collections.ConfigDict] = None
  no_momentum_in_memory: bool = False

  @nn.compact
  def __call__(
      self, images,
      context_tokens=None,
      gt_text_tokens=None,
      checkpoint_inds=None,
      preprocess=True, train=False, debug=False):
    """Forward model.

    Args:
      images: (batch_size, height, width, 3) for images or
        (batch_size, t, height, width, 3) for videos (when self.num_frames > 0).
      context_tokens: (batch_size, num_caps_per_image, max_context_len).
        Optional context tokens. E.g., the question in QA,
      gt_text_tokens: (batch_size, num_caps_per_image, max_cap_len)
      checkpoint_inds: (batch_size, num_caps_per_image) or None
      preprocess: bool
      train: bool
      debug: bool
    Returns:
      ret: dict of arrays.
        if train == True, return
          'text_outputs': (text_batch_size, max_cap_len, vocab_size)
        if train == False, return
          'visual_features': (text_batch_size, feature_len, feature_dim)
          'begin_tokens': (batch_size, num_caps_per_image, max_cap_len)
          'context_tokens': (batch_size, max_cap_len)
          'text_outputs': (batch_size, max_cap_len, vocab_size)
          'raw_streaming_feature':
            (batch_size, num_frames, num_streaming_tokens, dim)
        "batch_size" here are all the batch_size of videos.
    """
    del debug
    assert self.num_frames > 0
    assert images.ndim == 5
    assert self.project_layers_name == 'none'
    images = images.reshape(
        (images.shape[0] * images.shape[1],) + images.shape[2:])

    if preprocess:
      images = self.preprocess(images)

    visual_features = self.get_visual_features(
        images, train=train)  # (video_batch_size, num_tokens, dim)

    raw_streaming_feature, visual_features = self.get_dense_streaming_features(
        visual_features, train=train, checkpoint_inds=checkpoint_inds,
    )  # (video_batch_size, num_dense_outputs, new_num_vis_tokens, proj_dim)

    text_tokens, visual_features, context_tokens = (
        self.get_text_tokens_and_reshape_visual_features(
            visual_features, gt_text_tokens, context_tokens))
    # text_batch_size == video_batch_size * num_dense_outputs
    # text_tokens: (text_batch_size, max_cap_len)
    # visual_features: (text_batch_size, new_num_vis_tokens, proj_dim)
    # context_tokens: (text_batch_size, max_cap_len)

    if train:
      text_outputs = self.textual(
          text_tokens,
          visual_features,
          context_tokens=context_tokens,
          train=train,
      )  # (text_batch_size, max_cap_len, vocab_size)
      ret = {'text_outputs': text_outputs}
    else:
      text_outputs = self.textual(
          text_tokens,
          visual_features[:, 0],
          context_tokens=context_tokens,
          train=train,
      )  # (text_batch_size, max_cap_len, vocab_size)
      ret = {
          'visual_features': visual_features,
          'begin_tokens': text_tokens,
          'context_tokens': context_tokens,
          'text_outputs': text_outputs,
          'raw_streaming_feature': raw_streaming_feature,
      }
    return ret

  def get_dense_streaming_features(
      self, features, checkpoint_inds=None, train=False):
    """A wrapper function of different streaming function implementation.

    This is the place where we convert per-frame features (the input "features")
    to streaming feature at each intermediate decoding point.

    Args:
      features: (batch_size, num_tot_tokens, dim), where num_tot_tokens is
        num_frames * num_tokens_per_frame.
      checkpoint_inds: only needed when streaming_feature_implementation is
        'given_checkpoints'. Shape: (batch_size, num_checkpoints)
      train: bool
    Returns:
      streaming_features_per_frame:
        (batch_size, num_frames, num_streaming_tokens, dim)
      stteaming_features:
        (batch_size, num_checkpoints, num_streaming_tokens, dim)
    """
    num_dense_outputs = self.num_dense_outputs if train or (
        self.num_dense_outputs_test < 0) else self.num_dense_outputs_test
    if self.streaming_feature_implementation == 'fixed_checkpoints':
      return None, self.get_dense_streaming_features_fixed_checkpoints(
          features, train=train)
    elif self.streaming_feature_implementation == 'per_frame_and_gather':
      streaming_features_per_frame = self.get_dense_streaming_features_perframe(
          features, train=train)
      checkpoint_stride = self.num_frames // num_dense_outputs
      streaming_features = streaming_features_per_frame[
          :, (jnp.arange(num_dense_outputs) + 1) * checkpoint_stride - 1]
      return streaming_features_per_frame, streaming_features
    elif self.streaming_feature_implementation == 'given_checkpoints':
      # Here the checkpoint locations are variable and given in batch data.
      assert (checkpoint_inds is not None) or not train
      # checkpoint_inds: (video_batch_size, num_checkpoints)
      streaming_features_per_frame = self.get_dense_streaming_features_perframe(
          features, train=train,
      )  # (video_batch_size, num_frames, num_streaming_tokens, dim)
      if train:
        streaming_features = jnp.take_along_axis(
            streaming_features_per_frame, checkpoint_inds[:, :, None, None],
            axis=1,
        )  #  (video_batch_size, num_checkpoints, num_streaming_tokens, dim)
      else:
        # stride sampling
        checkpoint_stride = self.num_frames // num_dense_outputs
        streaming_features = streaming_features_per_frame[
            :, (jnp.arange(num_dense_outputs) + 1) * checkpoint_stride - 1]
      return streaming_features_per_frame, streaming_features
    else:
      raise NotImplementedError

  def get_dense_streaming_features_fixed_checkpoints(
      self, features, train=False):
    """Get streaming features with intermadiate outputs.

    With N = self.num_dense_outputs, we currently forward the memory module N
    times, each time with the features from 0 to k / N ratio of the video.
    This is thus inefficient in runtime (duplicate computing early features)
    and can be optimized when needed.

    Args:
      features: (video_batch_size, num_tot_tokens, dim)
      train: bool
    Returns:
      streaming_features:
        (video_batch_size, num_dense_outputs, num_streaming_tokens, dim)
    """
    num_dense_outputs = self.num_dense_outputs if train or (
        self.num_dense_outputs_test < 0) else self.num_dense_outputs_test
    video_batch_size, num_tot_tokens, dim = features.shape
    num_tokens_per_checkpoint = num_tot_tokens // num_dense_outputs
    streaming_features = []
    # TODO(zhouxy): implement this using vmap.
    for b in range(video_batch_size):
      video_streaming_feature = []
      for k in range(num_dense_outputs):
         # feature: (num_tokens_per_checkpoint * (k + 1), dim)
        feature = features[b, :(k + 1) * num_tokens_per_checkpoint]
        if self.streaming_method == 'temporal_mean_pool':
          streaming_feature = feature.reshape(
              k + 1, num_tokens_per_checkpoint, dim).mean(axis=0)  # (hw, dim)
        elif self.streaming_method == 'ema':
          streaming_feature = feature.reshape(
              self.num_frames, -1, dim)  # (t, hw, dim)
          buffer = streaming_feature[0]
          for t in range(1, self.num_frames):
            buffer = buffer * self.ema_decay + streaming_feature[t] * (
                1. - self.ema_decay)
          streaming_feature = buffer
        elif self.streaming_method == 'adjacent_tome':
          assert self.streaming_buffer_size > 0
          buffer = feature[:self.streaming_buffer_size]
          weights = jnp.ones((buffer.shape[0],), dtype=jnp.int32)
          streaming_feature = streaming_utils.adjacent_merge(
              buffer, feature[self.streaming_buffer_size:], weights=weights)[0]
        elif self.streaming_method == 'kmeans':
          assert self.streaming_buffer_size > 0
          weights = jnp.ones((feature.shape[0],), dtype=jnp.int32)
          init_centers = feature[:self.streaming_buffer_size]
          streaming_feature, _ = streaming_utils.kmeans(
              init_centers, feature, weights=weights,
              num_iters=self.kmeans_num_iters)
        else:
          assert self.streaming_method == 'none'
          streaming_feature = feature
        video_streaming_feature.append(streaming_feature)
      video_streaming_feature = jnp.stack(
          video_streaming_feature,
          axis=0)  # (num_dense_outputs, num_streaming_tokens, dim)
      streaming_features.append(video_streaming_feature)
    streaming_features = jnp.stack(streaming_features, axis=0)
    return streaming_features

  def get_dense_streaming_features_perframe(self, features, train=False):
    """Get streaming features with intermadiate outputs for all frames.

    Args:
      features: (video_batch_size, num_tot_tokens, dim)
      train: bool
    Returns:
      streaming_features:
        (video_batch_size, num_frames, num_streaming_tokens, dim)
    """
    del train
    _, num_tot_tokens, dim = features.shape
    num_token_per_frame = num_tot_tokens // self.num_frames
    assert self.streaming_buffer_size % num_token_per_frame == 0
    def process_video(video_features):
      # video_features: (num_tot_tokens, dim)
      video_features = video_features.reshape(
          self.num_frames, num_token_per_frame, dim)
      num_start_frames = self.streaming_buffer_size // num_token_per_frame
      if self.streaming_method == 'temporal_mean_pool':
        ret = jnp.cumsum(video_features, axis=0) / (jnp.arange(
            self.num_frames)[:, None, None] + 1)
        return ret
      elif self.streaming_method == 'kmeans':
        assert self.kmeans_num_iters > 0
        centers = video_features[:num_start_frames].reshape(-1, dim)
        counts = jnp.ones((centers.shape[0],), dtype=jnp.int32)
        ret = [centers for _ in range(num_start_frames)]
        for t in range(num_start_frames, self.num_frames):
          data = jnp.concatenate(
              [centers, video_features[t]], axis=0)
          weights = jnp.concatenate(
              [counts, jnp.ones((num_token_per_frame,), dtype=jnp.int32)],
              axis=0)
          if self.no_momentum_in_memory:
            weights = weights * 0 + 1
          centers, counts = streaming_utils.kmeans(
              centers, data, weights=weights, num_iters=self.kmeans_num_iters)
          ret.append(centers)
        ret = jnp.stack(ret, axis=0)
        return ret
      else:
        raise NotImplementedError(self.streaming_method)
    streaming_features = jax.vmap(process_video)(features)
    return streaming_features

  def get_text_tokens_and_reshape_visual_features(
      self, visual_features, gt_text_tokens, context_tokens):
    """Get inputs to the text decoder.

    In evaluation, we create the zero-padded text-token with the first token
      being BOS.
    In training, the visual_features and gt_text_tokens should have the same
      shape[1], and we consider them as aligned features and captions.

    Note the output dimension in evaluation changed compare to regular caption
    models, and thus we must use a separate eval step.

    Args:
      visual_features: (batch_size, num_dense_outputs, num_tokens, dim)
      gt_text_tokens: (batch_size, num_caps_per_image, max_cap_len) or None.
      context_tokens: (batch_size, num_caps_per_image, max_cap_len) or None.
    Returns:
      in evaluation (gt_text_tokens is None):
        text_tokens: (batch_size, max_cap_len).
        visual_features: (batch_size, num_dense_outputs, num_tokens, dim)
        context_tokens: None
      in training (gt_text_tokens is not None):
        text_tokens: (text_batch_size, max_cap_len). text_batch_size =
          batch_size * num_caps_per_image.
        visual_features: (text_batch_size, num_tokens, dim)
        context_tokens: (text_batch_size, max_cap_len)
    """
    if gt_text_tokens is None:  # Evaluation, create BOS tokens.
      text_tokens = jnp.full(
          (visual_features.shape[0],
           self.max_caption_length),
          self.end_token_id, dtype=jnp.int32)  # (B, max_cap_len)
      text_tokens = text_tokens.at[:, 0].set(
          self.begin_token_id)  # (text_batch_size, max_cap_len)
      context_tokens = None
    else:  # Training
      batch_size, num_caps_per_image = gt_text_tokens.shape[:2]
      assert (num_caps_per_image == self.num_dense_outputs) or (
          self.streaming_feature_implementation == 'given_checkpoints')
      text_tokens = gt_text_tokens.reshape(
          batch_size * num_caps_per_image,
          gt_text_tokens.shape[2],
      )  # (batch_size, num_caps_per_image, max_cap_len)
      visual_features = visual_features.reshape(
          (batch_size * num_caps_per_image,) + visual_features.shape[2:])
      if context_tokens is not None:
        context_tokens = context_tokens.reshape(-1, context_tokens.shape[-1])
    return text_tokens, visual_features, context_tokens

  def loss_function(self, outputs, batch):
    """Additionally support different weights for each intermediate output.

    We assume that all ground truth tokens are positive, and gt <= 0 is padding.

    Args:
      outputs: dict
        'text_outputs':
          (batch_size * num_caps_per_image, max_cap_len, vocab_size)
      batch: dict
        'text_tokens': (batch_size, num_caps_per_image, max_cap_len)
    Returns:
      loss: float
    """
    text_outputs = outputs['text_outputs']
    gt_text = batch['label']['text_tokens']
    batch_size = gt_text.shape[0]
    gt_text = gt_text.reshape(
        gt_text.shape[0] * gt_text.shape[1], gt_text.shape[2],
    )  # (batch_size * num_caps_per_image, max_cap_len)
    text_outputs = text_outputs[:, :-1]  # Move gt 1 word to the right.
    gt_text = gt_text[:, 1:]  # No need to predict BOS
    # valid: (text_batch_size, max_cap_len - 1)
    valid = (gt_text > 0).astype(jnp.float32)
    if self.ignore_empty_data:
      # Ignore samples with empty ground truth.
      valid = (valid.astype(bool) & (
          gt_text[:, 0] != self.end_token_id)[:, None]).astype(jnp.float32)
    # gt: (text_batch_size, max_cap_len - 1, vocab_size)
    gt = jax.nn.one_hot(gt_text, self.vocab_size)
    # customized label smoothing following GRiT
    #   https://github.com/JialianW/GRiT/blob/master/grit/modeling/text/
    #   text_decoder.py#L668
    gt = gt * (1. - self.label_smooth) + (
        1. - gt) * self.label_smooth / (self.vocab_size - 1)
    # loss:  (text_batch_size, max_cap_len - 1)
    gt = jax.lax.stop_gradient(gt)
    loss = optax.softmax_cross_entropy(text_outputs, gt)
    if self.dense_outputs_weight:
      assert len(self.dense_outputs_weight) == self.num_dense_outputs
      loss_weights = jnp.broadcast_to(
          jnp.asarray(self.dense_outputs_weight, jnp.float32)[None, :],
          (batch_size, self.num_dense_outputs)).reshape(-1)
      # (text_batch_size,)
      loss = loss * loss_weights[:, None]  # (text_batch_size, max_cap_len - 1)
    loss_dict = {}
    # TODO(zhouxy): Create a new DensecapModel class and move this code there.
    if self.show_densecap_loss or self.loc_loss_weight >= 0.0:
      thresh = self.vocab_size - self.num_bins
      cap_idx = ((gt_text < thresh) & (valid > 0)).astype(jnp.float32)
      loc_idx = ((gt_text >= thresh) & (valid > 0)).astype(jnp.float32)
      loss_dict['cap_loss'] = (loss * cap_idx).sum() / (cap_idx.sum() + 1e-8)
      loss_dict['loc_loss'] = (loss * loc_idx).sum() / (loc_idx.sum() + 1e-8)
      loss_dict['num_cap_tokens'] = cap_idx.sum() / cap_idx.shape[0]
      loss_dict['num_loc_tokens'] = loc_idx.sum() / loc_idx.shape[0]
    loss = (loss * valid).sum() / (valid.sum() + 1e-8)
    if self.loc_loss_weight >= 0.0:
      loss = loss_dict['cap_loss'] + (
          loss_dict['loc_loss'] * self.loc_loss_weight)
    loss_dict['total_loss'] = loss
    return loss, loss_dict


class DenseStreamingCaptioningModel(elcap_model.CaptioningModel):
  """Scenic Model Wrapper."""

  def get_dict_from_config(self):
    config_dict = super().get_dict_from_config()
    config_dict.update(dict(
        num_dense_outputs=self.config.model.get('num_dense_outputs', -1),
        streaming_buffer_size=self.config.model.get(
            'streaming_buffer_size', -1),
        streaming_method=self.config.model.get('streaming_method', 'none'),
        ema_decay=self.config.model.get('ema_decay', 0.9),
        early_segments_as_context=self.config.model.get(
            'early_segments_as_context', True),
        normalize_early_timestamps=self.config.model.get(
            'normalize_early_timestamps', False),
        copy_context=self.config.model.get('copy_context', False),
        dense_outputs_weight=self.config.model.get('dense_outputs_weight', ()),
        remove_segments_from_wrong_checkpoint=self.config.model.get(
            'remove_segments_from_wrong_checkpoint', False),
        streaming_feature_implementation=self.config.model.get(
            'streaming_feature_implementation', 'fixed_checkpoints'),
        no_timestamp_in_context=self.config.model.get(
            'no_timestamp_in_context', False),
        num_dense_outputs_test=self.config.model.get(
            'num_dense_outputs_test', -1),
        kmeans_num_iters=self.config.model.get('kmeans_num_iters', -1),
        ttm_output=self.config.model.get('ttm_output', 'output'),
        ttm_config=self.config.model.get('ttm_config'),
    ))
    return config_dict

  def build_flax_model(self):
    return DenseStreamingCaptioningFlaxModel(**self.get_dict_from_config())
