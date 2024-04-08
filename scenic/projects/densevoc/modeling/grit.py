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

"""Implementation of dense object captioning in the GRiT model.

Reference: https://arxiv.org/pdf/2212.00280.pdf
"""

import dataclasses
import math
from typing import Any, Dict, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from scenic.projects.baselines.centernet.modeling import centernet2
from scenic.projects.baselines.centernet.modeling import centernet_head
from scenic.projects.baselines.centernet.modeling import iou_assignment
from scenic.projects.baselines.centernet.modeling import roi_head_utils
from scenic.projects.baselines.centernet.modeling import roi_heads
from scenic.projects.baselines.centernet.modeling import vitdet
from scenic.projects.densevoc.modeling import auto_regressive_decode
from scenic.projects.densevoc.modeling import text_decoder


Assignment = iou_assignment.Assignment
ArrayDict = Dict[str, jnp.ndarray]
MetricsDict = Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]


class GRiTDetector(centernet2.CenterNet2Detector):
  """GRiT detector."""
  begin_token_id: int = 101  # tokenizer.cls_token_id == 101
  end_token_id: int = 102  # tokenizer.sep_token_id == 102
  vocab_size: int = 30522  # size of BertTokenizer
  max_caption_length: int = 40
  object_feat_res: int = 14
  text_iou_thresh: float = 0.8
  label_smooth: float = 0.1
  text_loss_weight: float = 1.0
  num_text_proposals: int = 128
  mult_caption_score: bool = False
  num_decoder_layers: int = 6
  use_roi_box_in_training: bool = False
  grounding_method: str = 'sumlogprob'

  def setup(self):
    self.backbone = vitdet.SimpleFeaturePyramid(
        backbone_args=self.backbone_args,
        scale_factors=self.vitdet_scale_factors,
        num_top_blocks=self.vitdet_num_top_blocks,
        dtype=self.dtype,
        name='backbone')

    self.proposal_generator = centernet_head.CenterNetHead(
        num_classes=self.num_classes, dtype=self.dtype,
        num_levels=len(self.strides),
        name='proposal_generator')

    self.roi_heads = roi_heads.CascadeROIHeads(
        input_strides={str(int(math.log2(s))): s for s in self.strides},
        num_classes=self.roi_num_classes,
        conv_dims=self.roi_conv_dims,
        conv_norm=self.roi_conv_norm,
        fc_dims=self.roi_fc_dims,
        samples_per_image=self.roi_samples_per_image,
        positive_fraction=self.roi_positive_fraction,
        matching_threshold=self.roi_matching_threshold,
        nms_threshold=self.roi_nms_threshold,
        class_box_regression=self.roi_class_box_regression,
        mult_proposal_score=self.roi_mult_proposal_score,
        scale_cascade_gradient=self.roi_scale_cascade_gradient,
        use_sigmoid_ce=self.roi_use_sigmoid_ce,
        add_box_pred_layers=self.roi_add_box_pred_layers,
        return_last_proposal=True,
        return_detection_in_training=self.use_roi_box_in_training,
        score_threshold=self.roi_score_threshold,
        post_nms_num_detections=self.roi_post_nms_num_detections,
    )
    self.text_decoder = text_decoder.TransformerDecoderTextualHead(
        num_layers=self.num_decoder_layers,
        name='roi_heads.text_decoder.textual')

  def decode_text(
      self, text_tokens, object_features,
      feature_valid_mask=None, return_feat=False):
    """Generate logits of a single word.

    Args:
      text_tokens: (batch_size, caption_length).
      object_features: (batch_size, feature_length, object_feat_size).
      feature_valid_mask: bool (batch_size, feature_length); False if padded.
      return_feat: bool; if True, return shape will be (
          batch_size, caption_length, hidden_size).
    Returns:
      output_logits: (batch_size, caption_length, vocab_size).
    """
    return self.text_decoder(
        text_tokens, object_features,
        feature_valid_mask=feature_valid_mask,
        return_feat=return_feat, train=False)

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               gt_boxes: Optional[jnp.ndarray] = None,
               gt_classes: Optional[jnp.ndarray] = None,
               gt_text_tokens: Optional[jnp.ndarray] = None,
               train: bool = False,
               preprocess: bool = False,
               *,
               padding_mask: Optional[jnp.ndarray] = None,
               debug: bool = False) -> Any:
    """Applies GRiT model on the input.

    Args:
      inputs: array of the preprocessed input images, in shape B x H x W x 3.
      gt_boxes: B x N x 4. Only used in training.
      gt_classes: B x N. Only used in training.
      gt_text_tokens: B x N x max_caption_length. Only used in training.
      train: Whether it is training.
      preprocess: If using the build-in preprocessing functions on inputs.
      padding_mask: Binary matrix with 0 at padded image regions.
      debug: Whether the debug mode is enabled. debug=True enables model
        specific logging/storing some values using jax.host_callback.

    Returns:
      outputs: a dict with the following keys:
        'detection_boxes' or 'last_proposals': batch_size x max_detections x 4
        'object_features': batch_size x max_detections x 196 x 256
        'begin_tokens': batch_size x max_detections x max_caption_length
        if train == False:
          'detection_scores': batch_size x max_detections
          'detection_classes': batch_size x max_detections, 0-based int.
        if train == True:
          'metrics': dict of losses.
    """
    if preprocess:
      inputs = self.preprocess(inputs, padding_mask)
    backbone_features = self.backbone(inputs, train=train)
    outputs = self.proposal_generator(
        backbone_features, train=train)

    detections, metrics, rpn_features, gt_classes, _ = self.forward_detection(
        inputs, outputs, backbone_features, gt_classes, gt_boxes,
        train=train, debug=debug)
    strides = sorted(self.roi_heads.input_strides.items(), key=lambda x: x[1])
    features = [rpn_features[s[0]] for s in strides]  # Sorted features
    outputs, _ = self.forward_object_caption(
        detections, metrics, outputs, features,
        gt_classes, gt_boxes, gt_text_tokens, train=train)
    return outputs

  def forward_object_caption(
      self, detections, metrics, outputs, features,
      gt_classes, gt_boxes, gt_text_tokens, train=False):
    if self.use_roi_box_in_training or not train:
      # Apply the text loss to the second stage outputs (vs. to the proposal).
      # This needs the second stage to be pretrained.
      # Otherwise the training easily goes NaN.
      last_proposals = detections['detection_boxes'][
          :, :self.num_text_proposals]  # (batch, num_text_proposals, 4)
    else:
      last_proposals = detections['last_proposals'][
          :, :self.num_text_proposals]  # (batch, num_text_proposals, 4)

    object_feat_res = self.object_feat_res
    if gt_text_tokens is None:  # evaluation
      detection_boxes = detections['detection_boxes']
      object_features = self.roi_heads.roi_align(
          features, detection_boxes, object_feat_res)
      object_features = object_features.reshape(
          object_features.shape[0], object_features.shape[1],
          object_feat_res ** 2, -1,
      )  # (batch_size, post_nms_num_detections, object_feat_res ** 2, C)
      text_tokens = jnp.full(
          (object_features.shape[0], object_features.shape[1],
           self.max_caption_length),
          self.end_token_id, dtype=jnp.int32)
      text_tokens = text_tokens.at[:, :, 0].set(self.begin_token_id)
      text_batch_size = object_features.shape[0] * object_features.shape[1]
      text_outputs = self.text_decoder(
          text_tokens.reshape(text_batch_size, self.max_caption_length),
          object_features.reshape(text_batch_size, object_feat_res ** 2, -1,),
          train=train,
      )  # (text_batch_size, max_caption_length, vocab_size)
      detections['detection_classes'] = (
          detections['detection_classes'] - 1).astype(jnp.int32)
      detections['object_features'] = object_features
      del text_outputs
      detections['begin_tokens'] = text_tokens
      outputs = detections
      matched_text, matched = None, None
    else:  # training
      # matched_text: (batch_size, num_text_proposals, max_caption_length)
      # matched: (batch_size, num_text_proposals): 1: matched; 0: not matched.
      matched_text, matched = self.match_texts(
          last_proposals, gt_boxes, gt_classes, gt_text_tokens,
          self.text_iou_thresh)
      object_features = self.roi_heads.roi_align(
          features, last_proposals, object_feat_res)
      # (batch_size, num_text_proposals, object_feat_res,  object_feat_res, C)
      # The text losses are only applied to foreground objects.
      text_batch_size = object_features.shape[0] * object_features.shape[1]
      text_outputs = self.text_decoder(
          matched_text.reshape(text_batch_size, self.max_caption_length),
          object_features.reshape(text_batch_size, object_feat_res ** 2, -1,),
          train=train,
      )  # (text_batch_size, max_caption_length, vocab_size)
      text_loss, num_valid_tokens = self.text_loss(
          text_outputs,
          matched_text.reshape(text_batch_size, self.max_caption_length),
          matched.reshape(text_batch_size,))
      metrics['text_loss'] = text_loss
      metrics['num_valid_tokens'] = num_valid_tokens
      outputs['metrics'] = metrics
      text_tokens = None
    return outputs, (metrics, object_features, last_proposals,
                     text_tokens, matched_text, matched)

  def forward_detection(
      self, inputs, outputs, backbone_features, gt_classes, gt_boxes,
      train=False, debug=False):
    """Forward second stage detection and get object features."""
    pre_nms_topk = self.pre_nms_topk_train if train else self.pre_nms_topk_test
    post_nms_topk = (
        self.post_nms_topk_train if train else self.post_nms_topk_test)
    boxes, scores, classes = self.extract_peaks(
        outputs, pre_nms_topk=pre_nms_topk)
    proposals = self.nms(
        boxes, scores, classes, post_nms_topk=post_nms_topk)
    proposal_boxes = jnp.stack(
        [x[0] for x in proposals], axis=0)  # B x num_prop x 4
    proposal_boxes = jnp.maximum(proposal_boxes, 0)
    proposal_boxes = jnp.minimum(
        proposal_boxes, max(inputs.shape[1], inputs.shape[2]))
    proposal_scores = jnp.stack(
        [x[1] for x in proposals], axis=0)  # B x num_propq
    rpn_features = {str(int(math.log2(s))): v for s, v in zip(
        self.strides, backbone_features)}
    # TODO(zhouxy): modify class format in the dataloader.
    # scenic dataloader loads classes in range [0, num_class - 1], and
    #  dpax RoI heads assume gt_classes in range [1, num_class]. Add 1 to valid
    #  gt objects (indicated by any box axis > 0).
    if gt_classes is not None and gt_boxes is not None:
      gt_classes = gt_classes + (gt_boxes.max(axis=2) > 0)
    image_shape = jnp.concatenate([
        jnp.ones((inputs.shape[0], 1), jnp.float32) * inputs.shape[1],
        jnp.ones((inputs.shape[0], 1), jnp.float32) * inputs.shape[2],
    ], axis=1)  # B x 2, in order (height, width)
    detections, metrics = self.roi_heads(
        rpn_features, image_shape,
        gt_boxes, gt_classes,
        proposal_boxes, proposal_scores,
        training=train, postprocess=True, debug=debug)
    return detections, metrics, rpn_features, gt_classes, image_shape

  def match_texts(
      self, proposals, gt_boxes, gt_classes, gt_text_tokens, thresh):
    """Match proposals and their texts based on bounding box IoU.

    Args:
      proposals: Boxes with array (B, samples_per_image, 4).
      gt_boxes: Boxes with array (B, max_gt_boxes, 4).
      gt_classes: (B, max_gt_boxes). This is needed for background padding.
      gt_text_tokens: (B, max_gt_boxes, max_caption_length)
      thresh: float.
    Returns:
      matched_text: shape (B, samples_per_image, max_caption_length).
      matched: shape (B, samples_per_image): 0 or 1.
    """
    def _impl(proposals, gt_boxes, gt_classes):
      iou = roi_head_utils.pairwise_iou(gt_boxes, proposals)
      matched_idxs, assignments = iou_assignment.label_assignment(
          iou, [thresh], [Assignment.NEGATIVE, Assignment.POSITIVE])
      matched_classes = gt_classes[matched_idxs]
      matched_classes = jnp.where(
          assignments != Assignment.POSITIVE, 0, matched_classes)
      return matched_idxs, matched_classes
    matched_idxs, matched_classes = jax.vmap(_impl, in_axes=0)(
        proposals, gt_boxes, gt_classes)
    matched_texts = jnp.take_along_axis(
        gt_text_tokens,
        matched_idxs[..., None],
        axis=1,
        mode='promise_in_bounds')
    return matched_texts, matched_classes

  def text_loss(self, text_outputs, matched_text, mask):
    """Text loss with label smoothing.

    Args:
      text_outputs: (text_batch_size, max_caption_length, vocab_size)
      matched_text: (text_batch_size, max_caption_length)
      mask: (text_batch_size,)

    Returns:
      loss: float
      num_valid_tokens: float
    """
    text_outputs = text_outputs[:, :-1]  # Move gt 1 word to the right.
    matched_text = matched_text[:, 1:]  # No need to predict BOS
    # valid: (text_batch_size, max_caption_length - 1)
    valid = ((matched_text > 0) & (mask[:, None] > 0))
    # Ignore samples with empty ground truth (from padding).
    valid = valid & (matched_text[:, 0] != self.end_token_id)[:, None]
    valid = valid.astype(jnp.float32)
    # gt: (text_batch_size, max_caption_length - 1, vocab_size)
    gt = jax.nn.one_hot(matched_text, self.vocab_size)
    # customized label smoothing following GRiT
    #   https://github.com/JialianW/GRiT/blob/master/grit/modeling/text/
    #   text_decoder.py#L668
    gt = gt * (1. - self.label_smooth) + (
        1. - gt) * self.label_smooth / (self.vocab_size - 1)
    # loss:  (text_batch_size, max_caption_length - 1)
    gt = jax.lax.stop_gradient(gt)
    loss = optax.softmax_cross_entropy(text_outputs, gt)
    loss = (loss * valid[:, :]).sum() / (valid.sum() + 1e-8)
    num_valid_tokens = valid.sum() / (mask.sum() + 1e-8)
    return loss, num_valid_tokens

  def loss_function(
      self,
      outputs: Any,
      batch: Any,
  ):
    """Loss function of GRiT.

    Args:
      outputs: dict of 'heatmaps' and `box_regs`. Both are list of arrays from
        different FPN levels, in shape L x [B, hl, wl, C']. L is the number
        of FPN levels, hl, wl are the shape in FPN level l.
      batch: dict that has 'inputs', 'batch_mask' and, 'label' (ground truth).
        batch['label'] is a dict with the following keys and shape:
          'boxes': B x max_boxes x 4
          'labels': B x max_boxes
    Returns:
      total_loss: Total loss weighted appropriately.
      metrics: auxiliary metrics for debugging and visualization.
    """
    detection_loss, metrics = super().loss_function(outputs, batch)
    total_loss = detection_loss + self.text_loss_weight * metrics['text_loss']
    metrics['total_loss'] = total_loss
    return total_loss, metrics


class GRiTModel(centernet2.CenterNet2Model):
  """Scenic Model Wrapper."""

  def build_flax_model(self):
    fields = set(x.name for x in dataclasses.fields(GRiTDetector))
    config_dict = {
        k: v for k, v in self.config.model.items() if k in fields}
    return GRiTDetector(**config_dict)

  def autoregressive_predict(
      self, params, detections, mask=None,
      feature_key='object_features',
      begin_token_key='begin_tokens',
      output_key='text_tokens'):
    """Generate caption from object features in an auto-agressive way.

    Args:
      params: pytree of network parameters.
      detections: dict with keys:
          'object_features': (batch_size, n, feature_length, object_feat_size)
          'begin_tokens': (batch_size, n, max_caption_length)
      mask: (batch_size, n, feature_length)
      feature_key: str
      begin_token_key: str
      output_key: str
    Returns:
      Updated detections with updated keys:
        'detections': int array (batch_size, n, max_caption_length),
            whose values are in range vocab_size
        'detection_scores': (batch_size, n)
    """
    batch_size, num_objects = detections[feature_key].shape[:2]
    text_batch_size = batch_size * num_objects
    object_features = detections[feature_key].reshape(
        text_batch_size, detections[feature_key].shape[2], -1)
    begin_tokens = detections[begin_token_key].reshape(
        text_batch_size, self.flax_model.max_caption_length)
    if mask is not None:
      mask = mask.reshape(
          text_batch_size, object_features.shape[1])
    # pylint: disable=g-long-lambda
    # (text_batch_size, max_caption_length) ->
    #   (text_batch_size, max_caption_length, vocab_size)
    tokens_to_logits = lambda x: self.flax_model.apply(
        variables={'params': params},
        text_tokens=x,
        object_features=object_features,
        feature_valid_mask=mask,
        method=self.flax_model.decode_text,
    )
    text_tokens, log_probs = auto_regressive_decode.auto_regressive_decode(
        begin_tokens, tokens_to_logits,
        max_steps=self.flax_model.max_caption_length,
        eos_index=self.flax_model.end_token_id)
    detections[output_key] = text_tokens.reshape(
        batch_size, num_objects, self.flax_model.max_caption_length)
    if self.flax_model.mult_caption_score:
      detections['detection_scores'] = (detections['detection_scores'].reshape(
          batch_size, num_objects) * jnp.exp(log_probs)) ** 0.5
    return detections

  def compute_sentence_likelihood(self, params, detections, sentence_tokens):
    """Compute likelihood of a given tokenized sentence.

    This implements section 3.5 in the Dense VOC paper
    https://arxiv.org/pdf/2306.11729.pdf

    Args:
      params: pytree of network parameters.
      detections: dict with keys:
          'object_features': (batch_size, n, feature_length, object_feat_size)
      sentence_tokens: (batch_size, max_caption_length), the first token should
          be BOS. Last valid token should be EOS. Padding tokens should be 0.
    Returns:
      Updated detections with updated keys:
          'likelihood': (batch_size, n)
    """
    object_features = detections['object_features']
    batch_size, num_objects = object_features.shape[:2]
    text_batch_size = batch_size * num_objects
    cap_len = self.flax_model.max_caption_length
    text_tokens = jnp.broadcast_to(
        sentence_tokens.reshape(batch_size, 1, cap_len),
        (batch_size, num_objects, cap_len)).reshape(text_batch_size, cap_len)
    object_features = object_features.reshape(
        text_batch_size, object_features.shape[2], object_features.shape[3])

    text_outputs = self.flax_model.apply(
        variables={'params': params},
        text_tokens=text_tokens,
        object_features=object_features,
        method=self.flax_model.decode_text,
    )  # (text_batch_size, max_caption_length, vocab_size)

    text_tokens = text_tokens[:, 1:]  # Shift GT sentence 1 word to right.
    text_outputs = text_outputs[:, :-1]  # Shift predicted sentence to align GT.
    mask = text_tokens > 0  # (text_batch_size, cap_len - 1)
    prob = jax.nn.softmax(text_outputs)  # (text_batch_size, cap_len - 1, vocab)
    prob = jnp.take_along_axis(
        prob, text_tokens[:, :, None],
        axis=2)[..., 0]  # (text_batch_size, cap_len - 1)
    if self.flax_model.grounding_method == 'sumprob':
      likelihood = (prob * mask).sum(axis=1) / (mask.sum(axis=1) + 1e-8)
    elif self.flax_model.grounding_method == 'sumlogprob':
      likelihood = jnp.exp(
          (jnp.log(prob) * mask).sum(axis=1) / (mask.sum(axis=1) + 1e-8))
    else:
      raise ValueError(
          f'Unknown grounding method: {self.flax_model.grounding_method}')
    likelihood = likelihood.reshape(
        batch_size, num_objects)
    likelihood = jnp.maximum(
        likelihood * detections['detection_scores'], 0) ** 0.5
    detections['likelihood'] = likelihood  # (batch_size, num_objects)
    return detections
