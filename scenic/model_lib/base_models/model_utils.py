# Copyright 2021 The Scenic Authors.
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

"""Utilities for models."""

from typing import Optional, Any, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


PyTree = Any
PyModule = Any
Array = Union[jnp.ndarray, np.ndarray]


def psum_metric_normalizer(metrics: Tuple[jnp.ndarray, jnp.ndarray]
                           ) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Applies psum over the given tuple of (metric, normalizer)."""
  psumed_metric = jnp.sum(jax.lax.psum(metrics[0], axis_name='batch'))
  psumed_normalizer = jnp.sum(
      jax.lax.psum(metrics[1], axis_name='batch'))
  return (psumed_metric, psumed_normalizer)


def num_examples(logits: jnp.ndarray,
                 one_hot_targets: jnp.ndarray,
                 weights: Optional[jnp.ndarray] = None
                 ) -> Union[jnp.ndarray, int]:
  del logits
  if weights is None:
    return one_hot_targets.shape[0]
  return weights.sum()


def apply_weights(output: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
  """Applies given weights of the inputs in the minibatch to outputs.

  Note that weights can be per example (i.e. of shape `[batch_size,]`) or per
  pixel/token (i.e. of shape `[batch_size, height, width]` or
  `[batch_size, len]`) so we need to broadcast it to the output shape.

  Args:
    output: Computed output, which can be loss or the correctly
      classified examples, etc.
    weights: Weights of inputs in the batch, which can be None or
      array of shape [batch, ...].

  Returns:
    Weighted output.
  """
  desired_weights_shape = weights.shape + (1,) * (output.ndim - weights.ndim)
  weights = jax.lax.broadcast_in_dim(
      weights,
      shape=desired_weights_shape,
      broadcast_dimensions=tuple(range(weights.ndim)))
  # scale the outputs with weights
  return output * weights


def weighted_correctly_classified(
    logits: jnp.ndarray,
    one_hot_targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Compute weighted number of correctly classified over the given batch.

  This computes the weighted number of correctly classified examples/pixels in a
  single, potentially padded minibatch. If the minibatch/inputs is padded (i.e.,
  it contains null examples/pad pixels) it is assumed that weights is a binary
  mask where 0 indicates that the example/pixel is null/padded. We assume the
  trainer will aggregate and divide by number of samples.

  Args:
   logits: Output of model in shape [batch, ..., num_classes].
   one_hot_targets: One hot vector of shape [batch, ..., num_classes].
   weights: None or array of shape [batch, ...] (rank of one_hot_targets -1).

  Returns:
    The number of correctly classified examples in the given batch.
  """
  if logits.ndim != one_hot_targets.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s one_hot_targets' %
        (str(logits.shape), str(one_hot_targets.shape)))
  preds = jnp.argmax(logits, axis=-1)
  targets = jnp.argmax(one_hot_targets, axis=-1)
  correct = jnp.equal(preds, targets)

  if weights is not None:
    correct = apply_weights(correct, weights)

  return correct.astype(jnp.int32)


def weighted_top_one_correctly_classified(
    logits: jnp.ndarray,
    multi_hot_targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Compute weighted number of correctly classified, given top 1 class.

  This computes the weighted number of correctly classified examples/pixels in a
  single, potentially padded minibatch, given top-one prediction. If the
  minibatch/inputs is padded (i.e., it contains null examples/pad pixels) it is
  assumed that weights is a binary mask where 0 indicates that the example/pixel
  is null/padded. We assume the trainer will aggregate and divide by number of
  samples.

  Args:
   logits: Output of model in shape [batch, ..., num_classes].
   multi_hot_targets: Multi hot vector of shape [batch, ..., num_classes].
   weights: None or array of shape [batch, ...] (rank of one_hot_targets -1).

  Returns:
    The number of correctly classified examples in the given batch, given top
    one prediction.
  """
  if logits.ndim != multi_hot_targets.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s multi_hot_targets' %
        (str(logits.shape), str(multi_hot_targets.shape)))

  top1_idx = jnp.argmax(logits, axis=-1)
  # extracts the label at the highest logit index for each inputs
  top1_correct = jnp.take_along_axis(
      multi_hot_targets, top1_idx[:, None], axis=-1)[:, 0]
  if weights is not None:
    top1_correct = apply_weights(top1_correct, weights)

  return top1_correct


def weighted_topk_correctly_classified(logits: jnp.ndarray,
                                       multi_hot_target: jnp.ndarray,
                                       weights: Optional[jnp.ndarray] = None,
                                       k: int = 5) -> jnp.ndarray:
  """Compute weighted number of correctly classified given the top k prediction.

  This computes the weighted number of correctly classified examples/pixels in a
  single, potentially padded minibatch, given the top-k prediction. In the
  multi-hot target case, the sample is considered correct when any of the top-k
  predictions matches any of the multi-hot targets. If the minibatch/inputs is
  padded (i.e., it contains null examples/pad pixels) it is assumed that weights
  is a binary mask where 0 indicates that the example/pixel is null/padded. We
  assume the trainer will aggregate and divide by number of
  samples.

  Args:
    logits: Output of model in shape [batch, ..., num_classes].
    multi_hot_target: Multi hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch, ...] (rank of one_hot_target -1).
    k: Number of top prediction to consider.

  Returns:
    The number of correctly classified examples in the given batch, given top
    k prediction.
  """
  if logits.ndim != multi_hot_target.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s one_hot_target' %
        (str(logits.shape), str(multi_hot_target.shape)))
  if k <= 0 or k > logits.shape[-1]:
    raise ValueError('Incorrect k. k must be in [1,%s]' %
                     str(logits.shape[-1]))

  topk_pred = jax.lax.top_k(logits, k)[1]

  num_classes = logits.shape[-1]
  multi_hot_pred = jnp.sum(
      jax.nn.one_hot(topk_pred, num_classes=num_classes), axis=-2)
  correct = jnp.any(multi_hot_pred * multi_hot_target,
                    axis=-1).astype(jnp.float32)

  if weights is not None:
    correct = apply_weights(correct, weights)

  return correct.astype(jnp.int32)


def weighted_recall(logits: Array, multi_hot_target: Array,
                    weights: Optional[Array] = None) -> Array:
  """Compute weighted recall given the top k prediction.

  This computes the weighted number of correctly recalled examples/pixels in a
  single, potentially padded minibatch, given the top-k prediction. Per sample,
  k is the number of gt labels in that sample. If the minibatch/inputs is padded
  (i.e., it contains null examples/pad pixels) it is assumed that weights is a
  binary mask where 0 indicates that the example/pixel is null/padded. We assume
  the trainer will aggregate and divide by number of samples.

  Args:
    logits: float array; Output of model in shape [batch, ..., num_classes].
    multi_hot_target: Multi hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch, ...] (rank of multi_hot_target -1).

  Returns:
    The fraction of correctly recalled labels.
  """
  if logits.ndim != multi_hot_target.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s one_hot_target' %
        (str(logits.shape), str(multi_hot_target.shape)))

  num_classes = multi_hot_target.shape[-1]

  indices_top = jnp.argsort(logits, axis=-1)[..., ::-1]
  predictions_at_top = jax.nn.one_hot(indices_top, num_classes)
  correct_at_top = jnp.sum(
      predictions_at_top * jnp.expand_dims(multi_hot_target, axis=-2), axis=-1)

  # Mask out (in)correct predictions that are not in top k, where k is the
  # number of gt labels.
  num_gt_labels = jnp.sum(multi_hot_target, axis=-1, keepdims=True)
  mask = (num_gt_labels > jnp.arange(num_classes)).astype(jnp.int32)

  recall = jnp.sum(correct_at_top * mask, axis=-1) / (
      jnp.sum(multi_hot_target, axis=-1) + 1E-12)

  if weights is not None:
    recall = apply_weights(recall, weights)

  return recall


def apply_label_smoothing(one_hot_targets: jnp.ndarray,
                          label_smoothing: Optional[float]) -> jnp.ndarray:
  """Apply label smoothing to the one-hot targets.

  Applies label smoothing such that the on-values are transformed from 1.0 to
  `1.0 - label_smoothing + label_smoothing / num_classes`, and the off-values
  are transformed from 0.0 to `label_smoothing / num_classes`.
  https://arxiv.org/abs/1512.00567

  Note that another way of performing label smoothing (which we don't use here)
  is to take `label_smoothing` mass from the on-values and distribute it to the
  off-values; in other words, transform the on-values to `1.0 - label_smoothing`
  and the  off-values to `label_smoothing / (num_classes - 1)`.
  http://jmlr.org/papers/v20/18-789.html


  Args:
    one_hot_targets: One-hot targets for an example, a [batch, ..., num_classes]
      float array.
    label_smoothing: A scalar in [0, 1] used to smooth the labels.

  Returns:
    A float array of the same shape as `one_hot_targets` with smoothed label
    values.
  """
  on_value = 1.0 - label_smoothing
  num_classes = one_hot_targets.shape[-1]
  off_value = label_smoothing / num_classes
  one_hot_targets = one_hot_targets * on_value + off_value
  return one_hot_targets


def weighted_unnormalized_softmax_cross_entropy(
    logits: jnp.ndarray,
    one_hot_targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_smoothing: Optional[float] = None,
    label_weights: Optional[jnp.ndarray] = None,
    logits_normalized: bool = False) -> jnp.ndarray:
  """Compute weighted softmax cross entropy give logits and targets.

  This computes sum_(x,y) softmax-ce(x, y) for a single, potentially padded
  minibatch. If the minibatch is padded (that is it contains null examples)
  it is assumed that weights is a binary mask where 0 indicates that the
  example is null.

  Args:
    logits: Output of model in shape [batch, ..., num_classes].
    one_hot_targets: One hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).
    label_smoothing: Scalar to use to smooth the one-hot labels.
    label_weights: Weight per label of shape [num_classes].
    logits_normalized: If True, the logits are assumed to already be
      normalized.

  Returns:
    The softmax cross entropy of the examples in the given batch.
  """
  if logits.ndim != one_hot_targets.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s one_hot_targets' %
        (str(logits.shape), str(one_hot_targets.shape)))

  # optionally apply label smoothing
  if label_smoothing is not None:
    one_hot_targets = apply_label_smoothing(one_hot_targets, label_smoothing)

  # optionally apply label weights
  if label_weights is not None:
    one_hot_targets *= label_weights

  if not logits_normalized:
    logits = nn.log_softmax(logits)
  loss = -jnp.einsum('...k,...k->...', one_hot_targets, logits)
  if weights is not None:
    loss = apply_weights(loss, weights)

  return loss


def weighted_unnormalized_sigmoid_cross_entropy(
    logits: jnp.ndarray,
    multi_hot_targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_weights: Optional[jnp.ndarray] = None,
    label_smoothing: Optional[float] = None,
    logits_normalized: bool = False) -> jnp.ndarray:
  """Compute weighted sigmoid cross entropy given logits and targets.

  This also called Binary Cross-Entropy Loss and it measures the probability
  error in discrete classification tasks in which each class is independent and
  not mutually exclusive.
  This computes sum_(x,y) sigmoid-ce(x, y) for a single, potentially padded
  minibatch. If the minibatch is padded (that is it contains null examples)
  it is assumed that weights is a binary mask where 0 indicates that the
  example is null.

  Args:
    logits: Output of model in shape [batch, ..., num_classes].
    multi_hot_targets: Multi-hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch x ...]
      (rank of one_hot_targets -1). This is the weight to apply to the loss
      computed for each example in the batch. Can be used to ignore padded
      examples in the batch.
    label_weights: None or array of shape broadcastable to the shape of logits.
      Typically this would be [num_classes] and is the weight to apply to each
      label.
    label_smoothing: Scalar to use to smooth the one-hot labels.
    logits_normalized: If True, the logits are assumed to be log probs.

  Returns:
    The sigmoid cross entropy of the examples in the given batch.
  """
  if logits.ndim != multi_hot_targets.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s multi_hot_targets' %
        (str(logits.shape), str(multi_hot_targets.shape)))

  # optionally apply label smoothing
  if label_smoothing is not None:
    multi_hot_targets = apply_label_smoothing(multi_hot_targets,
                                              label_smoothing)

  if logits_normalized:
    log_p, prob = logits, jnp.exp(logits)
    log_not_p = jnp.log((1 + 1e-6) - prob)
  else:
    log_p, log_not_p = jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits)
    prob = nn.sigmoid(logits)

  loss = -(multi_hot_targets * log_p +
           (1. - multi_hot_targets) * log_not_p)

  if label_weights is not None:
    loss = loss * label_weights

  if weights is not None:
    loss = apply_weights(loss, weights)
  return loss.sum(axis=-1)


def weighted_softmax_cross_entropy(
    logits: jnp.ndarray,
    one_hot_targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_smoothing: Optional[float] = None,
    label_weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Same as weighted_unnormalized, but additionally takes a mean.

  Args:
    logits: Output of model in shape [batch, ..., num_classes].
    one_hot_targets: One hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).
    label_smoothing: float scalar to use to smooth the one-hot labels.
    label_weights: Weight per label of shape [num_classes].

  Returns:
    The mean cross entropy of the examples in the given batch as a scalar.
  """
  if weights is not None:
    normalization = weights.sum()
  else:
    normalization = np.prod(one_hot_targets.shape[:-1])

  unnormalized_softmax_ce = weighted_unnormalized_softmax_cross_entropy(
      logits, one_hot_targets, weights, label_smoothing, label_weights)
  return jnp.sum(unnormalized_softmax_ce) / (normalization + 1e-8)


def weighted_sigmoid_cross_entropy(
    logits: jnp.ndarray,
    multi_hot_targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_weights: Optional[jnp.ndarray] = None,
    label_smoothing: Optional[float] = None) -> jnp.ndarray:
  """Compute weighted sigmoid cross entropy given logits and targets.

  Args:
    logits: Output of model in shape [batch, ..., num_classes].
    multi_hot_targets: Multi-hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).
    label_weights: None or array of shape broadcastable to the shape of logits.
      Typically this would be [num_classes] and is the weight to apply to each
      label.
    label_smoothing: Scalar to use to smooth the one-hot labels.

  Returns:
    The mean cross entropy of the examples in the given batch as a scalar.
  """
  if weights is not None:
    normalization = weights.sum()
  else:
    normalization = np.prod(multi_hot_targets.shape[:-1])

  unnormalized_sigmoid_ce = weighted_unnormalized_sigmoid_cross_entropy(
      logits,
      multi_hot_targets,
      weights=weights,
      label_weights=label_weights,
      label_smoothing=label_smoothing)
  return jnp.sum(unnormalized_sigmoid_ce) / normalization


def l2_regularization(params: PyTree):
  """Calculate the L2 loss (square L2 norm), given parameters of the model.

  Args:
    params: Parameters of the model.

  Returns:
    L2 norm.

  """
  weight_penalty_params = jax.tree_leaves(params)
  return sum([jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])


def weighted_l1_loss(x: jnp.ndarray,
                     y: jnp.ndarray,
                     weights: Optional[jnp.ndarray] = None,
                     reduction: Optional[str] = None) -> jnp.ndarray:
  """L1 loss with optional reduction specified.

  Args:
    x: Input array of any shape.
    y: Input array of shape broadcastable to that of x.
    weights: Weights to apply to the loss.
    reduction: Type of reduction, which is from [None, 'mean'].

  Returns:
    reduction(jnp.abs(x - y)). 'mean' reduction takes the global mean. To use
    customized normalization use 'none' reduction and scale loss in the caller.
  """
  abs_diff = jnp.abs(x - y)
  if weights is not None:
    abs_diff = apply_weights(abs_diff, weights)
  if not reduction:
    return abs_diff
  elif reduction == 'mean':
    return abs_diff.mean()


############################## Focal Loss ######################################


def focal_softmax_cross_entropy(
    logits: jnp.ndarray,
    one_hot_targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_smoothing: Optional[float] = None,
    label_weights: Optional[jnp.ndarray] = None,
    logits_normalized: bool = False,
    gamma: Optional[float] = 2.0) -> jnp.ndarray:
  """Compute focal softmax cross-entropy given logits and targets.

  This computes focal loss: (1-p_t)**gamma -log p_t, where p_t is the softmax
  probability of the target.

  Args:
    logits: Output of model in shape [batch, ..., num_classes].
    one_hot_targets: One hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).
    label_smoothing: Scalar to use to smooth the one-hot labels.
    label_weights: Weight per label of shape [num_classes].
    logits_normalized: If True, the logits are assumed to be log probs.
    gamma: Modulating factor of the focal loss.

  Returns:
    The loss of the examples in the given batch.
  """
  loss = weighted_unnormalized_softmax_cross_entropy(
      logits, one_hot_targets, weights=None, label_smoothing=label_smoothing,
      label_weights=label_weights, logits_normalized=logits_normalized)
  prob = jnp.exp(-loss)  # Loss is -log(p_t)
  loss *= (1. - prob)**gamma
  if weights is not None:
    loss = apply_weights(loss, weights)

  return loss


def focal_sigmoid_cross_entropy(
    logits: jnp.ndarray,
    multi_hot_targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_smoothing: Optional[float] = None,
    label_weights: Optional[jnp.ndarray] = None,
    logits_normalized: bool = False,
    alpha: Optional[float] = 0.5,
    gamma: Optional[float] = 2.0) -> jnp.ndarray:
  """Compute focal softmax cross-entropy given logits and targets.

  Focal loss assuming y is the binary target vector:
  alpha * (1-p_t)**gamma -log p_t, if y_t = 1, and
  (1.-alpha) * p_t**gamma -log (1 - p_t), if y_t = 0,
  and p_t is the sigmoid probability at index t.

  Args:
    logits: Output of model in shape [batch, ..., num_classes].
    multi_hot_targets: Multi-hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).
    label_smoothing: Scalar to use to smooth the one-hot labels.
    label_weights: Weight per label of shape [num_classes].
    logits_normalized: If True, the logits are assumed to be log probs.
    alpha: Balancing factor of the focal loss.
    gamma: Modulating factor of the focal loss.

  Returns:
    The loss of the examples in the given batch.
  """
  # optionally apply label smoothing
  if label_smoothing is not None:
    multi_hot_targets = apply_label_smoothing(multi_hot_targets,
                                              label_smoothing)
  if logits_normalized:
    log_p, prob = logits, jnp.exp(logits)
    log_not_p = jnp.log((1 + 1e-6) - prob)
  else:
    log_p, log_not_p = jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits)

  loss = -(multi_hot_targets * log_p + (1. - multi_hot_targets) * log_not_p)

  p_t = jnp.exp(-loss)
  loss *= (1 - p_t)**gamma
  loss *= alpha * multi_hot_targets + (1 - alpha) * (1 - multi_hot_targets)

  if label_weights is not None:
    loss = loss * label_weights

  if weights is not None:
    loss = apply_weights(loss, weights)
  return loss.sum(axis=-1)


############################## Misc ######################################


@jax.partial(jax.vmap, in_axes=[0, 0], out_axes=0)
def simple_gather(x: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
  """Gathers `x` using the indices in `idx`.

  `output[i] = x[i, idx[i]]` . This simple gather operation assumes that the
  first dimension is the batch dimension. The indices index into the second
  dimension. The rest of the dimensions are copied as is from `x` into output.
  Note that the implementation below only handles a single element in the batch.
  `jax.vmap` extends this to the batch dimension.

  Args:
    x: Inputs of shape [bs, n, d].
    idx: An array of shape [bs, m] and dtype jnp.int32 or int64 that
      specifies indexes we want to gather from x.

  Returns:
    Gathered output of shape [bs, m, d].
  """
  return x[idx]

# box utils implemented based on:
# https://github.com/facebookresearch/detr/blob/master/util/box_ops.py


def box_cxcywh_to_xyxy(x: Array,
                       np_backbone: PyModule = jnp) -> Array:
  """Converts boxes from [cx, cy, w, h] format into [x, y, x', y'] format."""
  x_c, y_c, w, h = np_backbone.split(x, 4, axis=-1)
  b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
  return np_backbone.concatenate(b, axis=-1)


def box_cxcywh_to_yxyx(x: Array,
                       np_backbone: PyModule = jnp) -> Array:
  """Converts boxes from [cx, cy, w, h] format into [y, x, y', x'] format."""
  x_c, y_c, w, h = np_backbone.split(x, 4, axis=-1)
  b = [(y_c - 0.5 * h), (x_c - 0.5 * w), (y_c + 0.5 * h), (x_c + 0.5 * w)]
  return np_backbone.concatenate(b, axis=-1)


def box_xyxy_to_cxcywh(x: Array,
                       np_backbone: PyModule = jnp) -> Array:
  """Converts boxes from [x, y, x', y'] format into [cx, cy, w, h] format."""
  x0, y0, x1, y1 = np_backbone.split(x, 4, axis=-1)
  b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
  return np_backbone.concatenate(b, axis=-1)


def box_yxyx_to_cxcywh(x: Array,
                       np_backbone: PyModule = jnp) -> Array:
  """Converts boxes from [y, x, y', x'] format into [cx, cy, w, h] format."""
  y0, x0, y1, x1 = np_backbone.split(x, 4, axis=-1)
  b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
  return np_backbone.concatenate(b, axis=-1)


def box_iou(boxes1: Array,
            boxes2: Array,
            np_backbone: PyModule = jnp,
            all_pairs: bool = True) -> Array:
  """Computes IoU between two sets of boxes.

  Boxes are in [x, y, x', y'] format [x, y] is top-left, [x', y'] is bottom
  right.

  Args:
    boxes1: Predicted bounding-boxes in shape [bs, n, 4].
    boxes2: Target bounding-boxes in shape [bs, m, 4]. Can have a
      different number of boxes if all_pairs is True.
    np_backbone: numpy module: Either the regular numpy package or jax.numpy.
    all_pairs: Whether to compute IoU between all pairs of boxes or not.
  Returns:
    If all_pairs == True, returns the pairwise IoU cost matrix of shape
    [bs, n, m].  If all_pairs == False, returns the IoU between corresponding
    boxes. The shape of the return value is then [bs, n].
  """

  # first compute box areas. These will be used later for computing the union
  wh1 = boxes1[..., 2:] - boxes1[..., :2]
  area1 = wh1[..., 0] * wh1[..., 1]  # [bs, n]

  wh2 = boxes2[..., 2:] - boxes2[..., :2]
  area2 = wh2[..., 0] * wh2[..., 1]  # [bs, m]

  if all_pairs:
    # compute pairwise top-left and bottom-right corners of the intersection
    # of the boxes
    lt = np_backbone.maximum(boxes1[..., :, None, :2],
                             boxes2[..., None, :, :2])  # [bs, n, m, 2].
    rb = np_backbone.minimum(boxes1[..., :, None, 2:],
                             boxes2[..., None, :, 2:])  # [bs, n, m, 2].

    # intersection = area of the box defined by [lt, rb]
    wh = (rb - lt).clip(0.0)  # [bs, n, m, 2]
    intersection = wh[..., 0] * wh[..., 1]  # [bs, n, m]

    # union = sum of areas - intersection
    union = area1[..., :, None] + area2[..., None, :] - intersection

    iou = intersection / (union + 1e-6)

  else:
    # compute top-left and bottom-right corners of the intersection between
    # corresponding boxes
    assert boxes1.shape[1] == boxes2.shape[1], (
        'Different number of boxes when all_pairs is False')
    lt = np_backbone.maximum(boxes1[..., :, :2],
                             boxes2[..., :, :2])  # [bs, n, 2]
    rb = np_backbone.minimum(boxes1[..., :, 2:], boxes2[..., :,
                                                        2:])  # [bs, n, 2]

    # intersection = area of the box defined by [lt, rb]
    wh = (rb - lt).clip(0.0)  # [bs, n, 2]
    intersection = wh[..., :, 0] * wh[..., :, 1]  # [bs, n]

    # union = sum of areas - intersection.
    union = area1 + area2 - intersection

    # somehow the pytorch implementation does not use + 1e-6 to avoid 1/0 cases
    iou = intersection / (union + 1e-6)

  return iou, union


def generalized_box_iou(boxes1: Array,
                        boxes2: Array,
                        np_backbone: PyModule = jnp,
                        all_pairs: bool = True) -> Array:
  """Generalized IoU from https://giou.stanford.edu/.

  The boxes should be in [x, y, x', y'] format specifying top-left and
  bottom-right corners.

  Args:
    boxes1: Predicted bounding-boxes in shape [..., n, 4].
    boxes2: Target bounding-boxes in shape [..., m, 4].
    np_backbone: Numpy module: Either the regular numpy package or jax.numpy.
    all_pairs: Whether to compute generalized IoU from between all-pairs of
      boxes or not. Note that if all_pairs == False, we must have m==n.

  Returns:
    If all_pairs == True, returns a [bs, n, m] pairwise matrix, of generalized
    ious. If all_pairs == False, returns a [bs, n] matrix of generalized ious.
  """
  # degenerate boxes gives inf / nan results, so do an early check
  # TODO(b/166344282): Figure out how to enable asserts on inputs with jitting:
  #  assert (boxes1[:, :, 2:] >= boxes1[:, :, :2]).all()
  #  assert (boxes2[:, :, 2:] >= boxes2[:, :, :2]).all()
  iou, union = box_iou(
      boxes1, boxes2, np_backbone=np_backbone, all_pairs=all_pairs)

  # generalized iou has an extra term which takes into account the area of
  # the box containing both of these boxes. The following code is very similar
  # to that for computing intersection but the min-max are flipped
  if all_pairs:
    lt = np_backbone.minimum(boxes1[..., :, None, :2],
                             boxes2[..., None, :, :2])  # [bs, n, m, 2]
    rb = np_backbone.maximum(boxes1[..., :, None, 2:],
                             boxes2[..., None, :, 2:])  # [bs, n, m, 2]

  else:
    lt = np_backbone.minimum(boxes1[..., :, :2],
                             boxes2[..., :, :2])  # [bs, n, 2]
    rb = np_backbone.maximum(boxes1[..., :, 2:], boxes2[..., :,
                                                        2:])  # [bs, n, 2]

  # now to compute the covering box's area
  wh = (rb - lt).clip(0.0)  # Either [bs, n, 2] or [bs, n, m, 2]
  area = wh[..., 0] * wh[..., 1]  # Either [bs, n] or [bs, n, m]

  # finally generalized IoU from IoU, union, and area
  # somehow the pytorch implementation does not use + 1e-6 to avoid 1/0 cases
  return iou - (area - union) / (area + 1e-6)


def confusion_matrix(y_true: Array,
                     y_pred: Array,
                     num_classes: int,
                     weights: Optional[Array] = None,
                     np_backbone: PyModule = jnp) -> Array:
  """Computes the confusion matrix between y_true and y_pred.

  Args:
    y_true: Array of true labels.
    y_pred: Array of predicted labels.
    num_classes: Number of classes.
    weights: nd-array, Weight of each datapoint (e.g. for masking).
    np_backbone: numpy module: Either the regular numpy package or jax.numpy.

  Returns:
    A [num_classes, num_classes] confusion matrix, normalized by the number of
      elements in y_true/y_pred.
  """
  assert y_true.shape == y_pred.shape
  if weights is None:
    weights = np_backbone.ones_like(y_true)
  else:
    assert y_true.shape == weights.shape

  # If weights are all zero, histogram2d returns NaN. To avoid this, set weights
  # to 1 and then set output to zero below:
  weights_all_zero = 1.0 - np_backbone.any(weights).astype(np_backbone.float32)
  weights = weights + weights_all_zero

  cm, *_ = np_backbone.histogram2d(
      y_true.ravel(),
      y_pred.ravel(),
      bins=np_backbone.arange(num_classes + 1),
      weights=None if weights is None else weights.ravel())

  # If weights are all zero, set the confusion matrix to zero:
  cm = cm * (1.0 - weights_all_zero)
  return cm


def mean_iou(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Computes the mean intersection-over-union, given a confusion matrix.

  Args:
    cm: array_like; [num_classes, num_classes] confusion matrix.

  Returns:
    Scalar mean intersection-over-union score.
  """
  # TODO(mjlm): Check the mean IoU computation for correctness (end to end).
  # Based on experimental/brain/off_the_grid/lib/metrics.py:

  sum_over_row = np.sum(cm, axis=0)
  sum_over_col = np.sum(cm, axis=1)
  true_positives = np.diag(cm)

  # sum_over_row + sum_over_col =
  #     2 * true_positives + false_positives + false_negatives.
  denominator = sum_over_row + sum_over_col - true_positives

  # The mean is only computed over classes that appear in the
  # label or prediction tensor. If the denominator is 0, we need to
  # ignore the class.
  iou_per_class = true_positives / denominator
  return (np.nan_to_num(np.nanmean(iou_per_class)),
          np.nan_to_num(iou_per_class))


def dice_loss(inputs: jnp.ndarray,
              targets: jnp.ndarray,
              weights: Optional[jnp.ndarray] = None,
              all_pairs: bool = False,
              eps: float = 1.0,
              interpolation: str = 'nearest') -> jnp.ndarray:
  """Computes the Dice loss given panoptic segmentation logits and targets.

  This loss is based on the Dice coefficient (F-1 score). For details, see
  https://arxiv.org/abs/2005.12872 and https://arxiv.org/pdf/1606.04797.pdf.

  Args:
    inputs: Predicted mask logits with shape [batch, num_objects, H, W].
    targets: Target masks with shape [batch, num_objects, H, W].
    weights: Array of shape [batch, ...].
    all_pairs: Whether to compute the loss for all object pairs or not.
    eps: Epsilon for numerical stability.
    interpolation: Method to use for upsampling inputs to target size.

  Returns:
    If all_pairs == True, returns a [bs, n, m] pairwise matrix, of dice loss.
    If all_pairs == False, returns a [bs, n] matrix of dice loss.
  """
  _, n, h, w = inputs.shape
  b, m, _, _ = targets.shape

  # Downsample targets to match prediction:
  # TODO(mjlm): Check if it would be better to upsample predictions.
  # For now, we downsample targets to save memory.
  targets = jax.image.resize(
      targets, shape=[b, m, h, w], method=interpolation, antialias=True)

  # TODO(mjlm): Also try softmax instead of sigmoid:
  # As in MaX-DeepLab:
  inputs = jax.nn.sigmoid(inputs)

  inputs = jnp.reshape(inputs, [b, n, h * w])
  targets = jnp.reshape(targets, [b, m, h * w])
  if all_pairs:
    numerator = 2 * jnp.einsum('bnp,bkp->bnk', inputs, targets)
    denominator = (jnp.sum(inputs[:, :, None, :], axis=-1) +
                   jnp.sum(targets[:, None, :, :], axis=-1))
  else:
    assert n == m
    numerator = 2 * jnp.einsum('bnp,bnp->bn', inputs, targets)
    denominator = jnp.sum(inputs + targets, axis=-1)
  loss = 1.0 - (numerator + eps) / (denominator + eps)

  if weights is not None:
    loss = apply_weights(loss, weights)

  return loss
