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

"""Utilities for models."""

import functools
from typing import Optional, Any, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


PyTree = Any
PyModule = Any
Array = Union[jnp.ndarray, np.ndarray]


def psum_metric_normalizer(
    metrics: Tuple[jnp.ndarray, jnp.ndarray],
    axis_name: Union[str,
                     Tuple[str]] = 'batch') -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Applies psum over the given tuple of (metric, normalizer)."""
  psumed_metric = jax.lax.psum(jnp.sum(metrics[0]), axis_name=axis_name)
  psumed_normalizer = jax.lax.psum(jnp.sum(metrics[1]), axis_name=axis_name)
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

  Note that weights can be per example (i.e. of shape `[batch,]`) or per
  pixel/token (i.e. of shape `[batch, height, width]` or
  `[batch, len]`) so we need to broadcast it to the output shape.

  Args:
    output: Computed output, which can be loss or the correctly classified
      examples, etc.
    weights: Weights of inputs in the batch, which can be None or array of shape
      [batch, ...].

  Returns:
    Weighted output.
  """
  if output.ndim < weights.ndim:
    raise ValueError('Output rank should be higher or equal to weights rank.')
  desired_weights_shape = weights.shape + (1,) * (output.ndim - weights.ndim)
  weights = jax.lax.broadcast_in_dim(
      weights,
      shape=desired_weights_shape,
      broadcast_dimensions=tuple(range(weights.ndim)))
  # Scale the outputs with weights.
  return output * weights


def weighted_correctly_classified(
    logits: jnp.ndarray,
    one_hot_targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Computes weighted number of correctly classified over the given batch.

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
  """Computes weighted number of correctly classified, given top 1 class.

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

  top1_idx = jnp.argmax(logits, axis=-1)[..., None]
  # Extracts the label at the highest logit index for each input.
  top1_correct = jnp.take_along_axis(multi_hot_targets, top1_idx, axis=-1)
  if weights is not None:
    top1_correct = apply_weights(top1_correct, weights)

  return top1_correct


def weighted_topk_correctly_classified(logits: jnp.ndarray,
                                       multi_hot_target: jnp.ndarray,
                                       weights: Optional[jnp.ndarray] = None,
                                       k: int = 5) -> jnp.ndarray:
  """Computes weighted number of correctly classified given the top k prediction.

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
  correct = jnp.any(
      multi_hot_pred * multi_hot_target, axis=-1, keepdims=True
  ).astype(jnp.float32)

  if weights is not None:
    correct = apply_weights(correct, weights)

  return correct.astype(jnp.int32)


def weighted_precision_at_k(logits: jnp.ndarray,
                            multi_hot_target: jnp.ndarray,
                            weights: Optional[jnp.ndarray] = None,
                            k: int = 5) -> jnp.ndarray:
  """Computes fraction of correct predictions among the top k predictions.

  This computes the weighted precision-at-k (i.e. the fraction of true positives
  among the top k predicted classes) in a single, potentially padded minibatch.
  If the minibatch/inputs is padded (i.e., it contains null examples/pad pixels)
  it is assumed that weights is a binary mask where 0 indicates that the
  example/pixel is null/padded. We assume the trainer will aggregate and divide
  by number of samples.

  Args:
    logits: Output of model in shape [batch, ..., num_classes].
    multi_hot_target: Multi hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch, ...] (rank of one_hot_target -1).
    k: Number of top predictions to consider.

  Returns:
    The precision for each example in the batch, given top k predictions.
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

  true_positive = jnp.sum(
      multi_hot_pred * multi_hot_target, axis=-1).astype(jnp.float32)
  # Above, the model is forced to predict exactly k positive classes, so the sum
  # of true and false positives is equal to k:
  precision = true_positive / k

  if weights is not None:
    precision = apply_weights(precision, weights)

  return precision


def weighted_recall(logits: Array, multi_hot_target: Array,
                    weights: Optional[Array] = None) -> Array:
  """Computes weighted recall given the top k prediction.

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
    logits_normalized: bool = False,
    keep_label_dimension: bool = False) -> jnp.ndarray:
  """Computes weighted softmax cross entropy give logits and targets.

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
    logits_normalized: If True, the logits are assumed to already be normalized.
    keep_label_dimension: If True, the class dimension of the output loss is not
      summed over.

  Returns:
    The softmax cross entropy of the examples in the given batch.
  """
  if logits.ndim != one_hot_targets.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s one_hot_targets' %
        (str(logits.shape), str(one_hot_targets.shape)))

  # Optionally apply label smoothing.
  if label_smoothing is not None:
    one_hot_targets = apply_label_smoothing(one_hot_targets, label_smoothing)

  # Optionally apply label weights.
  if label_weights is not None:
    one_hot_targets *= label_weights

  if not logits_normalized:
    logits = nn.log_softmax(logits)
  loss = -one_hot_targets * logits
  if weights is not None:
    loss = apply_weights(loss, weights)

  if not keep_label_dimension:
    loss = loss.sum(axis=-1)

  return loss


def weighted_unnormalized_sigmoid_cross_entropy(
    logits: jnp.ndarray,
    multi_hot_targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_weights: Optional[jnp.ndarray] = None,
    label_smoothing: Optional[float] = None,
    logits_normalized: bool = False) -> jnp.ndarray:
  """Computes weighted sigmoid cross entropy given logits and targets.

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
    weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).
      This is the weight to apply to the loss computed for each example in the
      batch. Can be used to ignore padded examples in the batch.
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

  # Optionally apply label smoothing.
  if label_smoothing is not None:
    multi_hot_targets = apply_label_smoothing(multi_hot_targets,
                                              label_smoothing)

  if logits_normalized:
    log_p, prob = logits, jnp.exp(logits)
    log_not_p = jnp.log((1 + 1e-6) - prob)
  else:
    log_p, log_not_p = jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits)

  loss = -(multi_hot_targets * log_p +
           (1. - multi_hot_targets) * log_not_p)

  if label_weights is not None:
    loss = loss * label_weights

  if weights is not None:
    loss = apply_weights(loss, weights)

  return loss


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
  """Computes weighted sigmoid cross entropy given logits and targets.

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
  return jnp.sum(unnormalized_sigmoid_ce) / (normalization + 1e-8)


def l2_regularization(params: PyTree):
  """Calculate the L2 loss (square L2 norm), given parameters of the model.

  Args:
    params: Parameters of the model.

  Returns:
    L2 norm.

  """
  weight_penalty_params = jax.tree_util.tree_leaves(params)
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
    return abs_diff.mean()  # pytype: disable=bad-return-type  # jax-ndarray


def weighted_box_l1_loss(
    pred: jnp.ndarray,
    tgt: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    reduction: Optional[str] = None,
    tight: bool = True,
) -> jnp.ndarray:
  """L1 loss for bounding box with optional reduction specified.

  Args:
    pred: Prediction boxes of shape (..., 4), where the last dimension has form
      (x_min, y_min, x_max, y_max).
    tgt: Target boxes of shape (..., 4), where the last dimension has form
      (x_min, y_min, x_max, y_max).
    weights: Weights to apply to the loss.
    reduction: Type of reduction, which is from [None, 'mean'].
    tight: If True, returns the vanilla L1 loss on the bounding box coordinates.
      If False, returns loose bounding-box L1 loss, where prediction edges only
      generate loss when they stretch outside the target box, but not when they
      are within it.

  Returns:
    reduction(jnp.abs(src - tgt)). 'mean' reduction takes the global mean. To
    use customized normalization use 'none' reduction and scale loss in the
    caller.
  """
  if pred.shape[-1] != 4:
    raise ValueError(
        f'The last dimension of the prediction boxes must be 4.'
        f' Got shape {pred.shape}.'
    )
  if tgt.shape[-1] != 4:
    raise ValueError(
        f'The last dimension of the target boxes must be 4.'
        f' Got shape {tgt.shape}.'
    )
  if tight:
    abs_diff = jnp.abs(pred - tgt)
  else:
    xy1, xy2 = jnp.split(pred - tgt, 2, axis=-1)
    xy1 = jnp.minimum(xy1, 0.)
    xy2 = jnp.maximum(xy2, 0.)
    abs_diff = jnp.abs(jnp.concatenate([xy1, xy2], axis=-1))
  if weights is not None:
    abs_diff = apply_weights(abs_diff, weights)
  if not reduction:
    return abs_diff
  elif reduction == 'mean':
    return abs_diff.mean()
  else:
    raise ValueError(f'Unknown reduction: {reduction}')


############################## Regression Loss #################################


def weighted_squared_error(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    axis: Optional[Union[int, Tuple[int, ...]]] = None) -> jnp.ndarray:
  """Computes weighted squared error given predictions and targets.

  This computes the squared_error of examples in a single, potentially
  padded minibatch. If the minibatch is padded (that is it contains null
  examples) it is assumed that weights is a binary mask where 0 indicates that
  the example is null.

  Args:
    predictions: Output of model in shape shape [batch, ..., n_features].
    targets: Array of shape [batch, ..., n_features].
    weights:  None or array of shape [batch,] This is the weight to apply to the
      loss computed for each example in the batch. Can be used to ignore padded
      examples in the batch.
    axis: The axis (or axes) to compute the loss over. If not specified, all
      dimensions besides the leading batch dimension are used.

  Returns:
    The mean squared error for each example in the given batch. The output shape
    is [batch,].
  """
  if predictions.ndim != targets.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s predictions and %s targets' %
        (str(predictions.shape), str(targets.shape)))
  if axis is None:
    # Sum over all features in each example in the batch:
    axis = tuple(range(1, predictions.ndim))

  error = targets - predictions
  loss = jnp.square(error)
  loss = jnp.sum(loss, axis=axis)
  if weights is not None:
    loss = apply_weights(loss, weights)
  return loss


def weighted_mean_squared_error(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    axis: Optional[Union[int, Tuple[int, ...]]] = None) -> jnp.ndarray:
  """Weighted mean of weighted_squared_error.

  Args:
    predictions: Output of model in shape [batch, ..., num_features].
    targets: Targets of shape [batch, ..., num_features].
    weights:  None or array of shape [batch,] This is the weight to apply to the
      loss  computed for each example in the batch. Can be used to ignore padded
      examples in the batch.
    axis: The axis (or axes) to compute the loss over. If not specified, all
      dimensions besides the leading batch dimension are used.

  Returns:
    The averaged mean squared error of all the examples in the given batch as a
    scalar.
  """
  unnormalized_mse = weighted_squared_error(
      predictions=predictions, targets=targets, weights=weights, axis=axis)

  if weights is not None:
    # Divide by sum of the broadcasted weights:
    broadcasted_shape = weights.shape + (1,) * (
        unnormalized_mse.ndim - weights.ndim)
    broadcasted_weights = jax.lax.broadcast_in_dim(
        weights,
        shape=broadcasted_shape,
        broadcast_dimensions=tuple(range(weights.ndim)))
    normalization = jnp.sum(broadcasted_weights *
                            jnp.ones(unnormalized_mse.shape))
  else:
    # Divide by number of examples:
    normalization = unnormalized_mse.size
  return jnp.sum(unnormalized_mse) / (normalization + 1e-8)


def weighted_absolute_error(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Computes weighted absolute error given predictions and targets.

  This computes the absolute_error of examples in a single, potentially
  padded minibatch. If the minibatch is padded (that is it contains null
  examples) it is assumed that weights is a binary mask where 0 indicates that
  the example is null.

  Args:
    predictions: Output of model in shape shape [batch, ..., n_features].
    targets: Array of shape [batch, ..., n_features].
    weights:  None or array of shape [batch,] This is the weight to apply to the
      loss  computed for each example in the batch. Can be used to ignore padded
      examples in the batch.

  Returns:
    The mean absolute error for each example in the given batch. The output
    shape is [batch,].
  """
  if predictions.ndim != targets.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s predictions and %s targets' %
        (str(predictions.shape), str(targets.shape)))

  error = targets - predictions
  loss = jnp.absolute(error)
  # Sum over all features in each example in the batch:
  loss = jnp.sum(loss, axis=tuple(range(1, predictions.ndim)))
  if weights is not None:
    loss = apply_weights(loss, weights)
  return loss


def weighted_mean_absolute_error(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Weighted mean of weighted_unnormalized_mean_absolute_error.

  Args:
    predictions: Output of model in shape [batch, ..., num_features].
    targets: Targets of shape [batch, ..., num_features].
    weights:  None or array of shape [batch] This is the weight to apply to the
      loss  computed for each example in the batch. Can be used to ignore padded
      examples in the batch.

  Returns:
    The averaged mean absolute error of all the examples in the given batch as
    a scalar.
  """
  unnormalized_mae = weighted_absolute_error(
      predictions=predictions, targets=targets, weights=weights)

  if weights is not None:
    # Divide by sum of weights:
    normalization = weights.sum()
  else:
    # Divide by batch size:
    normalization = unnormalized_mae.shape[0]
  return jnp.sum(unnormalized_mae) / (normalization + 1e-8)


############################## Focal Loss ######################################


def focal_softmax_cross_entropy(
    logits: jnp.ndarray,
    one_hot_targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    label_smoothing: Optional[float] = None,
    label_weights: Optional[jnp.ndarray] = None,
    logits_normalized: bool = False,
    gamma: Optional[float] = 2.0,
    keep_label_dimension: bool = False) -> jnp.ndarray:
  """Computes focal softmax cross-entropy given logits and targets.

  Focal loss as defined in https://arxiv.org/abs/1708.02002. Assuming y is the
  target vector and p is the predicted probability for the class, then:

  p_t = p if y == 1 and 1-p otherwise
  Focal loss = -(1-p_t)**gamma * log(p_t)

  NOTE: this is weighted unnormalized computation of loss that returns the loss
  of examples in the batch. If you are using it as a loss function, you can
  use the normalilzed version as:
  ```
    unnormalized_loss = focal_softmax_cross_entropy(...)
    if weights is not None:
      normalization = weights.sum()
    else:
      normalization = np.prod(one_hot_targets.shape[:-1])
    loss = jnp.sum(unnormalized_loss) / (normalization + 1e-8)
  ```

  Args:
    logits: Output of model in shape [batch, ..., num_classes].
    one_hot_targets: One hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch, ...] (rank of one_hot_targets -1).
    label_smoothing: Scalar to use to smooth the one-hot labels.
    label_weights: Weight per label of shape [num_classes].
    logits_normalized: If True, the logits are assumed to be log probs.
    gamma: Modulating factor of the focal loss.
    keep_label_dimension: If True, the class dimension of the output loss is not
      summed over.

  Returns:
    The loss of the examples in the given batch.
  """
  loss = weighted_unnormalized_softmax_cross_entropy(
      logits, one_hot_targets, weights=None, label_smoothing=label_smoothing,
      label_weights=label_weights, logits_normalized=logits_normalized,
      keep_label_dimension=True)
  prob = jnp.exp(logits) if logits_normalized else jax.nn.softmax(logits)
  prob = (prob * one_hot_targets).sum(axis=-1, keepdims=True)
  loss *= (1. - prob)**gamma
  if weights is not None:
    loss = apply_weights(loss, weights)

  if not keep_label_dimension:
    loss = loss.sum(axis=-1)

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
  """Computes focal softmax cross-entropy given logits and targets.

  Focal loss as defined in https://arxiv.org/abs/1708.02002. Assuming y is the
  target vector and p is the predicted probability for the class, then:

  p_t = p if y == 1 and 1-p otherwise
  alpha_t = alpha if y == 1 and 1-alpha otherwise

  Focal loss = -alpha_t * (1-p_t)**gamma * log(p_t)

  NOTE: this is weighted unnormalized computation of loss that returns the loss
  of examples in the batch. If you are using it as a loss function, you can
  use the normalilzed version as:
  ```
    unnormalized_loss = focal_sigmoid_cross_entropy(...)
    if weights is not None:
      normalization = weights.sum()
    else:
      normalization = np.prod(multi_hot_targets.shape[:-1])
    loss = jnp.sum(unnormalized_loss) / (normalization + 1e-8)
  ```

  Args:
    logits: Output of model in shape [batch, ..., num_classes].
    multi_hot_targets: Multi-hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch, ...] (rank of one_hot_targets -1).
    label_smoothing: Scalar to use to smooth the one-hot labels.
    label_weights: Weight per label of shape [num_classes].
    logits_normalized: If True, the logits are assumed to be log probs.
    alpha: Balancing factor of the focal loss.
    gamma: Modulating factor of the focal loss.

  Returns:
    The loss of the examples in the given batch.
  """
  # Optionally apply label smoothing.
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
  return loss


############################## Misc ######################################


@functools.partial(jax.vmap, in_axes=[0, 0], out_axes=0)
def simple_gather(x: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
  """Gathers `x` using the indices in `idx`.

  `output[i] = x[i, idx[i]]` . This simple gather operation assumes that the
  first dimension is the batch dimension. The indices index into the second
  dimension. The rest of the dimensions are copied as is from `x` into output.
  Note that the implementation below only handles a single element in the batch.
  `jax.vmap` extends this to the batch dimension.

  Args:
    x: Inputs of shape [bs, n, d].
    idx: An array of shape [bs, m] and dtype jnp.int32 or int64 that specifies
      indexes we want to gather from x.

  Returns:
    Gathered output of shape [bs, m, d].
  """
  return x[idx]


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
