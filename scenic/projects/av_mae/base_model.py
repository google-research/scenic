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

"""Base model definition."""

import functools
from typing import Dict, Optional, Tuple, Union

from immutabledict import immutabledict
import jax
import jax.numpy as jnp
import numpy as np

from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils
from scenic.model_lib.base_models import regression_model
from scenic.model_lib.layers import nn_ops


# TODO(aarnab): Compute validation metrics.
_REGRESSION_METRICS = immutabledict({
    'mean_squared_error':
        (functools.partial(model_utils.weighted_squared_error, axis=-1),
         model_utils.num_examples)
})

Patch = Union[Tuple[int, int], Tuple[int, int, int]]


class FeatureTargets():
  RGB = 'rgb'
  SPECTROGRAM = 'spectrogram'


def get_output_shapes(feature_target: str,
                      patch_size: Patch,
                      select_central_frame: Optional[bool] = None,
                      channels: int = 3):
  """Returns the output shape, depending on the feature regression target."""
  if feature_target == FeatureTargets.RGB:
    if len(patch_size) == 3 and select_central_frame:
      output_elements = patch_size[:2] + (1, channels)
    else:
      output_elements = patch_size + (channels,)
    return np.prod(np.array(output_elements))
  elif feature_target == FeatureTargets.SPECTROGRAM:
    output_elements = patch_size + (channels,)
    return np.prod(np.array(output_elements))
  else:
    raise NotImplementedError('Other feature targets not implemented yet.')


def extract_tubelets_from_video(
    x: jnp.ndarray,
    tubelet_size: Tuple[int, int, int],
    select_central_frame: bool) -> jnp.ndarray:
  """Extracts tubelets from videos for use as regression targets.

  Args:
    x: Input tensor of shape [batch, time, height, width, channels]
    tubelet_size: Tuple containing tubelet/patch size parameterised as
      [ph, pw, pt].
    select_central_frame: If True, select the central frame as the feature
      regression target.

  Returns:
    Tensor of shape [n, gt * gh * gw, pt * ph * pw * c] where
      gt = t // pt, gh = h // ph, gw = w // pw.
  """
  ph, pw, pt = tubelet_size
  n, t, h, w, c = x.shape
  gt, gh, gw = t // pt, h // ph, w // pw
  x = x.reshape([n, gt, pt, gh, ph, gw, pw, c])
  # Shape is then [n, gt, gh, gw, pt, ph, pw, c].
  x = jnp.transpose(x, axes=[0, 1, 3, 5, 2, 4, 6, 7])
  if select_central_frame:
    x = x[:, :, :, :, pt // 2, :, :, :]
    pt = 1
  return x.reshape([n, gt * gh * gw, pt * ph * pw * c])


def get_rgb_targets(inputs: jnp.ndarray,
                    patch_size: Patch,
                    select_central_frame: Optional[bool] = None,
                    reconstruct_grayscale: bool = False,
                    standardise_per_patch: bool = False,
                    standardise_per_patch_channels: bool = False
                    ) -> jnp.ndarray:
  """Get RGB targets to use for feature regression.

  Here, the targets are the raw rgb patches of the image.

  Args:
    inputs: Tensor of shape [b, h, w, c] or [b, t, h, w, c]. The former are
      images, and the later video.
    patch_size: The shape of the patch, defined as [ph, pw] for images, and
      [ph, pw, pt] for video.
    select_central_frame: If video and True, select the central frame as the
      feature regression target.
    reconstruct_grayscale: If True, the target patch is in grayscale rather
      than rgb.
    standardise_per_patch: If true, standardise each patch by subtracting the
      mean and dividing by the standard deviation of that patch.
    standardise_per_patch_channels: If true, standardise each patch by
    subtracting the mean and dividing by the standard deviation of that patch
    per channels.

  Returns:
    Patched inputs. For images, shape is [b, gh * gw, ph * pw * c] where
      gh = h // ph and gw = w // pw.
      For video, shape is [b, gt * gh * gw, pt * ph * pw * c].
  """
  if not (inputs.ndim == 4 or inputs.ndim == 5):
    raise ValueError('Inputs should be 4D (images) or 5D (video).')

  if reconstruct_grayscale:
    # Reference for converting between RGB and grayscale.
    # https://en.wikipedia.org/wiki/Luma_%28video%29
    # Also used in tf.image.rgb_to_grayscale
    rgb_weights = jnp.tile(jnp.array([[0.2989, 0.5870, 0.1140]]), (3, 1)).T
    inputs = jnp.matmul(inputs, rgb_weights)

  if inputs.ndim == 4:
    batch = inputs.shape[0]
    # Shape is [batch, ht, wt, hp, wp, c]
    patched_image = nn_ops.patch_image(inputs, inputs_shape=None,
                                       patch_size=patch_size)
    num_tokens = patched_image.shape[1] * patched_image.shape[2]
    patched_input = jnp.reshape(patched_image, (batch, num_tokens, -1))
  elif inputs.ndim == 5:
    if select_central_frame is None:
      raise ValueError('`select_central_frame` must be defined.')
    patched_input = extract_tubelets_from_video(
        inputs,
        patch_size,
        select_central_frame)

  if standardise_per_patch:
    patched_input = jax.nn.standardize(patched_input, axis=-1, epsilon=1e-6)
  elif standardise_per_patch_channels:
    old_shape = patched_input.shape
    batch, num_tokens = patched_input.shape[:2]
    num_channels = inputs.shape[-1]
    patched_input = jnp.reshape(patched_input,
                                (batch, num_tokens, -1, num_channels))
    patched_input = jax.nn.standardize(patched_input, axis=2, epsilon=1e-6)
    patched_input = jnp.reshape(patched_input, old_shape)

  return patched_input


def get_spectogram_targets(inputs: jnp.ndarray,
                           patch_size: Patch,
                           standardise_per_patch: bool = False
                           ) -> jnp.ndarray:
  """Get spectogram targets to use for feature regression.

  Here, the targets are the raw spectogram patches of the image.

  Args:
    inputs: Tensor of shape [b, h, w, c].
    patch_size: The shape of the patch, defined as [ph, pw].
    standardise_per_patch: If true, standardise each patch by subtracting the
      mean and dividing by the standard deviation of that patch.

  Returns:
    Patched inputs. Shape is [b, gh * gw, ph * pw * c].
  """
  if inputs.ndim != 4:
    raise ValueError('Inputs should be 4D.')

  if inputs.ndim == 4:
    batch = inputs.shape[0]
    # Shape is [batch, ht, wt, hp, wp, c]
    patched_image = nn_ops.patch_image(inputs, inputs_shape=None,
                                       patch_size=patch_size)
    num_tokens = patched_image.shape[1] * patched_image.shape[2]
    patched_input = jnp.reshape(patched_image, (batch, num_tokens, -1))

  if standardise_per_patch:
    patched_input = jax.nn.standardize(patched_input, axis=-1, epsilon=1e-6)

  return patched_input


def feature_regression_metrics_function(
    predictions: jnp.ndarray,
    prediction_masks: jnp.ndarray,
    batch: base_model.Batch,
    feature_target: str,
    metrics: base_model.MetricNormalizerFnDict = _REGRESSION_METRICS,
) -> Dict[str, Tuple[float, int]]:
  """Calculate metrics for the feature regression task.

  Currently we assume each metric_fn has the API:
    ```metric_fn(predictions, targets, weights)```
  and returns an array of shape [batch,]. We also assume that to compute
  the aggregate metric, one should sum across all batches, then divide by the
  total samples seen. In this way we currently only support metrics of the 1/N
  sum f(inputs, targets). Note, the caller is responsible for dividing by
  the normalizer when computing the mean of each metric.

  Args:
   predictions: Output of model in shape [batch, length].
   prediction_masks: Which of the predictions are valid.
   batch: Batch (dict) with keys 'targets' and optionally 'batch_mask'.
   feature_target: The feature target used for feature regression.
   metrics: The regression metrics to evaluate. The key is the
     name of the  metric, and the value is the metrics function.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """
  if feature_target == FeatureTargets.RGB:
    targets = batch['target_rgb']
  else:
    raise NotImplementedError(
        f'Feature target {feature_target} not implemented')

  batch_mask = batch.get('batch_mask')
  if batch_mask is None:
    batch_mask = jnp.ones(prediction_masks.shape)
  if batch_mask.ndim == 1:
    n_batch = predictions.shape[0]
    batch_mask = jnp.reshape(batch_mask, (n_batch, 1))
  weights = batch_mask * prediction_masks

  evaluated_metrics = {}
  for key, val in metrics.items():
    evaluated_metrics[key] = model_utils.psum_metric_normalizer(
        (val[0](predictions, targets, weights), val[1](predictions, targets,  # pytype: disable=wrong-arg-types  # jax-ndarray
                                                       weights)))
  return evaluated_metrics  # pytype: disable=bad-return-type  # jax-ndarray


class MaskedFeatureRegressionModel(regression_model.RegressionModel):
  """Defines commonalities between all masked self-supervised models."""

  def loss_function(self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
                    predictions: jnp.ndarray,
                    prediction_masks: jnp.ndarray,
                    batch: base_model.Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Returns the (weighted) mean squared error.

    Args:
      predictions: Output of model in shape [batch, num_tokens, channels].
      prediction_masks: The tokens to compute the loss on. Shape is
        [batch, num_tokens]
      batch: Batch (dict) with keys 'targets' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        L2 regularization.

    Returns:
      The (weighted) mean squared error.
    """
    batch_mask = batch.get('batch_mask')
    if batch_mask is None:
      batch_mask = jnp.ones(prediction_masks.shape)
    if batch_mask.ndim == 1:
      batch_mask = jnp.expand_dims(batch_mask, axis=-1)
    if self.config.masked_feature_loss.get('loss_unmasked_tokens', False):
      loss_weights = batch_mask
    else:
      loss_weights = batch_mask * prediction_masks

    feature_target = self.config.masked_feature_loss.target
    if feature_target == FeatureTargets.RGB:
      targets = batch[f'target_{feature_target}']
    else:
      raise NotImplementedError(
          f'Feature target {feature_target} not implemented.')

    total_loss = model_utils.weighted_mean_squared_error(
        predictions, targets, loss_weights, axis=-1)

    # Mean squared error is normalised by the number of tokens.
    # If this option is enabled, we normalise further by the number of features
    # we are regressing to.
    if self.config.masked_feature_loss.get('normalise_by_output_dimension',
                                           False):
      output_dimension = predictions.shape[-1]
      total_loss = total_loss / output_dimension

    if self.config.get('l2_decay_factor'):
      l2_loss = model_utils.l2_regularization(model_params)
      total_loss += 0.5 * self.config.l2_decay_factor * l2_loss
    return total_loss  # pytype: disable=bad-return-type  # jax-ndarray

  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    By default, we return the same metric for each split.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API:
    ```metrics_fn(predictions, batch)```
    """

    del split  # Same function for all splits.
    return functools.partial(
        feature_regression_metrics_function,
        feature_target=self.config.masked_feature_loss.target,
        metrics=_REGRESSION_METRICS)
