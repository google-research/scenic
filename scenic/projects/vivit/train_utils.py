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

"""Training Utilities for ViViT."""

import functools
from typing import Callable, Dict, List, Optional, Tuple, Union

from absl import logging
from flax import jax_utils
import flax.linen as nn
import jax
from jax.example_libraries.optimizers import clip_grads
import jax.numpy as jnp
import jax.profiler
import matplotlib.pyplot as plt
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import model_utils
from scenic.train_lib_deprecated import optimizers
from scenic.train_lib_deprecated import train_utils
import seaborn as sns

# Aliases for custom types:
Array = Union[jnp.ndarray, np.ndarray]
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]


def to_cpu(array: jnp.ndarray):
  """Transfers array (replicated on multiple hosts) to a single host.

  Args:
    array: Replicated array of shape
      [num_hosts, num_devices, local_batch_size, ...]

  Returns:
    array of shape [global_batch_size, ...] where
      global_batch_size = num_devices * local_batch_size
  """
  return jax.device_get(dataset_utils.unshard(jax_utils.unreplicate(array)))


def train_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    learning_rate_fn: Callable[[int], float],
    loss_fn: LossFn,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False
) -> Tuple[train_utils.TrainState, Dict[str, Tuple[float, int]], float]:
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Note that in this code, the buffers of the first (train_state) and second
  (batch) arguments are donated to the computation.

  Args:
    train_state: The state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    flax_model: A Flax model.
    learning_rate_fn: learning rate scheduler which give the global_step
      generates the learning rate.
    loss_fn: A loss function that given logits, a batch, and parameters of the
      model calculates the loss.
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    config: Configuration of the experiment.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training, computed metrics, and learning rate for logging.
  """
  new_rng, rng = jax.random.split(train_state.rng)

  if config.get('mixup') and config.mixup.alpha:
    mixup_rng, rng = jax.random.split(rng, 2)
    mixup_rng = train_utils.bind_rng_to_host_device(
        mixup_rng,
        axis_name='batch',
        bind_to=config.mixup.get('bind_to', 'device'))
    batch = dataset_utils.mixup(
        batch,
        config.mixup.alpha,
        config.mixup.get('image_format', 'NTHWC'),
        rng=mixup_rng)

  # Bind the rng to the host/device we are on for dropout.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device')

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    logits, new_model_state = flax_model.apply(
        variables,
        batch['inputs'],
        mutable=['batch_stats'],
        train=True,
        rngs={'dropout': dropout_rng},
        debug=debug)
    loss = loss_fn(logits, batch, variables['params'])
    return loss, (new_model_state, logits)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  step = train_state.global_step
  lr = learning_rate_fn(step)
  if config.get('sam_rho', None) is None:
    # Normal training
    (train_cost,
     (new_model_state,
      logits)), grad = compute_gradient_fn(train_state.optimizer.target)
  else:
    # SAM training, taken from cl/373487774
    def dual_vector(y: jnp.ndarray) -> jnp.ndarray:
      """Returns the solution of max_x y^T x s.t. ||x||_2 <= 1."""
      gradient_norm = jnp.sqrt(sum(
          [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)]))
      normalized_gradient = jax.tree_util.tree_map(
          lambda x: x / (gradient_norm + 1e-7), y)
      return normalized_gradient

    g_sam, _ = jax.grad(training_loss_fn, has_aux=True)(
        train_state.optimizer.target)
    g_sam = dual_vector(g_sam)
    target_sam = jax.tree_util.tree_map(
        lambda a, b: a + config.get('sam_rho') * b,
        train_state.optimizer.target, g_sam)
    (train_cost,
     (new_model_state,
      logits)), grad = compute_gradient_fn(target_sam)

  # TODO(dehghani,aarnab): Check how to move this after the pmeam.
  if config.get('max_grad_norm', None) is not None:
    grad = clip_grads(grad, config.max_grad_norm)

  del train_cost
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')
  new_optimizer = train_state.optimizer.apply_gradient(grad, learning_rate=lr)

  # Explicit weight decay, if necessary.
  if config.get('explicit_weight_decay', None) is not None:
    new_optimizer = new_optimizer.replace(
        target=optimizers.tree_map_with_names(
            functools.partial(
                optimizers.decay_weight_fn,
                lr=lr,
                decay=config.explicit_weight_decay),
            new_optimizer.target,
            match_name_fn=lambda name: 'kernel' in name))

  metrics = metrics_fn(logits, batch)
  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=step + 1,
      optimizer=new_optimizer,
      model_state=new_model_state,
      rng=new_rng)
  return new_train_state, metrics, lr


def eval_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: MetricFn,
    return_logits_and_labels: bool = False,
    return_confusion_matrix: bool = False,
    debug: Optional[bool] = False,
) -> Union[
    Tuple[Dict[str, Tuple[float, int]], jnp.ndarray, jnp.ndarray],
    Tuple[Dict[str, Tuple[float, int]], jnp.ndarray],
    Dict[str, Tuple[float, int]],
]:
  """Runs a single step of training.

  Note that in this code, the buffer of the second argument (batch) is donated
  to the computation.

  Assumed API of metrics_fn is:
  ```metrics = metrics_fn(logits, batch)
  where batch is yielded by the batch iterator, and metrics is a dictionary
  mapping metric name to a vector of per example measurements. eval_step will
  aggregate (by summing) all per example measurements and divide by the
  aggregated normalizers. For each given metric we compute:
  1/N sum_{b in batch_iter} metric(b), where  N is the sum of normalizer
  over all batches.

  Args:
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data. a metrics function, that given logits and
      batch of data, calculates the metrics as well as the loss.
    flax_model: A Flax model.
    metrics_fn: A metrics function, that given logits and batch of data,
      calculates the metrics as well as the loss.
    return_logits_and_labels: If true, returns logits and labels. Can be used
      for calculating the Mean Average Precision for multi-label problems.
      Only one of "return_logits_and_labels" and "return_confusion_matrix"
      should be true, with the latter taking precedence if both are set as true.
    return_confusion_matrix: If true, returns confusion matrix. Can be used
      to calculate additional metrics for k-way classification problems.
    debug: Whether the debug mode is enabled during evaluation.
      `debug=True` enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Calculated metrics [and optionally logits or confusion matrix].
  """
  variables = {
      'params': train_state.optimizer.target,
      **train_state.model_state
  }
  logits = flax_model.apply(
      variables, batch['inputs'], train=False, mutable=False, debug=debug)
  metrics = metrics_fn(logits, batch)

  if return_confusion_matrix:
    confusion_matrix = get_confusion_matrix(
        labels=batch['label'], logits=logits, batch_mask=batch['batch_mask'])
    confusion_matrix = jax.lax.all_gather(confusion_matrix, 'batch')
    return metrics, confusion_matrix

  if return_logits_and_labels:
    logits = jax.lax.all_gather(logits, 'batch')
    labels = jax.lax.all_gather(batch['label'], 'batch')
    return metrics, logits, labels

  return metrics


def test_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: MetricFn,
    n_clips: int = 2,
    return_logits_and_labels: bool = False,
    softmax_logits: bool = False,
    debug: bool = False,
) -> Union[
    Dict[str, Tuple[float, int]],
    Tuple[Dict[str, Tuple[float, int]], jnp.ndarray, jnp.ndarray],
]:
  """Runs a single step of testing.

  For multi-crop testing, we assume that num_crops consecutive entries in the
  batch are from the same example. And we average the logits over these examples

  We assume that the batch contains different crops of the same original
  example. Therefore, we can average all the logits of it.
  This assumption is true when local_batch_size = num_local_devices

  Args:
    train_state: The state of training including the current
      global_step, model_state, rng, and optimizer, and other metadata.
    batch: Dictionary with keys 'inputs', 'labels', 'batch_mask'. We assume that
      all the inputs correspond to the same original example in the test set.
      The input shapes to this function are batch['inputs'] = [num_crops, t, h,
      w, c] batch['labels'] = [num_crops, num_classes] However, for
      classification, the labels for all the crops are the same.
      batch['batch_mask'] = [num_crops]
    flax_model: A Flax model.
    metrics_fn: Metrics function for the model.
    n_clips: The number of clips to process at a time by each device. Set
      due to memory constraints.
    return_logits_and_labels: Whether return logits of the model or not.
    softmax_logits: Whether to softmax-normalise the logits before
      averaging
    debug: Whether the debug mode is enabled during evaluation.
      `debug=True` enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Calculated metrics [and optionally averaged logits that are of
    shape `[1, num_classes]`].
  """

  all_logits = jnp.zeros(batch['label'].shape[1])
  assert len(batch['batch_mask'].shape) == 1, (
      'Spatial padding is not supported in multi-crop evaluation.')

  num_crops = batch['inputs'].shape[0]

  variables = {
      'params': train_state.optimizer.target,
      **train_state.model_state
  }
  for idx in range(0, num_crops, n_clips):
    temp_input = batch['inputs'][idx:idx + n_clips]
    logits = flax_model.apply(
        variables, temp_input, train=False, mutable=False, debug=debug)
    if softmax_logits:
      logits = nn.softmax(logits, axis=-1)
    logits = jnp.sum(logits, axis=0)
    all_logits = all_logits + logits

  all_logits = all_logits / num_crops
  all_logits = jnp.expand_dims(all_logits, axis=0)
  batch['label'] = jnp.expand_dims(batch['label'][0], axis=0)
  batch['batch_mask'] = jnp.expand_dims(batch['batch_mask'][0], axis=0)
  metrics = metrics_fn(all_logits, batch)
  if return_logits_and_labels:
    return metrics, all_logits, batch['label']
  return metrics


def get_confusion_matrix(labels: Array, logits: Array,
                         batch_mask: Array) -> Array:
  """Computes confusion matrix from predictions.

  Args:
    labels: [n_batch] or [n_batch, n_classes] array. In the latter case, labels
      are assumed to be one-hot, since the confusion matrix is only defined when
      each example has one label.
    logits: [n_batch, n_classes] array, which are the predictions of the model.
    batch_mask: [n_batch] array. Entries should be 1 or 0, and indicate if the
      example is valid or not.

  Returns:
    confusion_matrix of shape [1, n_classes, n_classes]
  """
  if labels.ndim == logits.ndim:  # one-hot targets
    y_true = jnp.argmax(labels, axis=-1)
  else:
    y_true = labels
  y_pred = jnp.argmax(logits, axis=-1)

  # Prepare sample weights for confusion matrix:
  weights = batch_mask.astype(jnp.float32)

  confusion_matrix = model_utils.confusion_matrix(
      y_true=y_true,
      y_pred=y_pred,
      num_classes=logits.shape[-1],
      weights=weights)
  confusion_matrix = confusion_matrix[jnp.newaxis, ...]  # Dummy batch dim.
  return confusion_matrix


def render_confusion_matrices(confusion_matrices: List[Array],
                              normalization_method: str = 'cols',
                              figsize: Tuple[int, int] = (12, 12),
                              dpi: int = 100,
                              font_scale: int = 3) -> Array:
  """Render confusion matrix so that it can be logged to Tensorboard.

  Args:
    confusion_matrices: List of [n_batch, n_class, n_class] confusion matrices.
      The first two dimensions will be summed over to get an [n_class, n_class]
      matrix for rendering.
    normalization_method: Method of normalizing the confusion matrix before
      plotting. Supported values are one of "cols", "rows" and "none".
      If any other value, no normalization is performed.
    figsize: The figure size used by matplotlib and seaborn.
    dpi: The dpi used by matplotlib and seaborn.
    font_scale: The font scale used by seaborn.

  Returns:
    image: Rendered image of the confusion matrix for plotting. Data type is
      uint8 and values are in range [0, 255]. Shape is
      [1, figsize * dpi, figsize * dpi, 3]
  """
  conf_matrix = np.sum(confusion_matrices, axis=0)  # Sum over eval batches.
  if conf_matrix.ndim != 3:
    raise AssertionError(
        'Expecting confusion matrix to have shape '
        f'[batch_size, num_classes, num_classes], got {conf_matrix.shape}.')
  conf_matrix = np.sum(conf_matrix, axis=0)  # Sum over batch dimension.

  if normalization_method not in {'rows', 'cols', 'none'}:
    logging.warning('Normalizer must be one of {rows, cols, none}.'
                    'Defaulting to none.')

  sns.set(font_scale=font_scale)
  fig = plt.figure(figsize=figsize, dpi=dpi)

  # Normalize entries of the confusion matrix.
  if normalization_method == 'rows':
    normalizer = conf_matrix.sum(axis=1)[:, np.newaxis]
  elif normalization_method == 'cols':
    normalizer = conf_matrix.sum(axis=0)[np.newaxis, :]
  else:
    normalizer = 1
  normalized_matrix = np.nan_to_num(conf_matrix / normalizer)

  if np.sum(normalized_matrix) > 0:
    sns.heatmap(
        normalized_matrix,
        annot=True,
        linewidths=0.5,
        square=True,
        cbar=False,
        cmap='jet',
        annot_kws={'size': 18})
    fig.tight_layout(pad=0.0)

  fig.canvas.draw()
  ncols, nrows = fig.canvas.get_width_height()
  image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image = image.reshape(nrows, ncols, 3)
  return np.expand_dims(image, axis=0)
