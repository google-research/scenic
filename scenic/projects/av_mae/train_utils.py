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

"""Utility functions for the training loop."""
import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union

from absl import logging
from clu import platform

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scenic.model_lib.base_models import model_utils
from scenic.model_lib.layers import nn_ops
from scenic.train_lib import train_utils
from scenic.train_lib_deprecated import train_utils as train_utils_deprecated


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]
MetricFnEval = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                        Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, jnp.ndarray, Batch, Optional[jnp.ndarray]],
                  float]
# TODO(scenic-dev) JAX team is working on type annotation for pytree:
# https://github.com/google/jax/issues/1555
PyTree = Any


def generate_image_grid(target: jnp.ndarray,
                        prediction: jnp.ndarray,
                        prediction_masks: jnp.ndarray,
                        patch_size: Tuple[int, int],
                        n_columns: int,
                        input_size: Tuple[int, int, int],
                        modality: str = 'rgb'):
  """Generates a grid of images for logging summaries.

  Args:
   target: Array of shape [batch, num_tokens, patch_size * patch_size *
     channels]. Values are in the range [-1, 1].
   prediction: Array of same shape as target. It is possible that values are
     outside of the range [-1, 1].
   prediction_masks: Array of shape [batch, num_tokens]. Each entry is binary,
     ie 0 or 1. Used to mask out patches in the input and predictions.
   patch_size: The patch size of the model.
   n_columns: The number of columns to generate in the summary.
   input_size: The size of the input, specified as [height, width, channels].
   modality: The modality which is shown.
  Returns:
    An image grid. The shape is
    [batch_size / n_columns * 2 * height, n_columns * width, channels].
    Data type is uint8, and values are in the range [0, 255].
  """
  if target.shape != prediction.shape:
    raise ValueError('Shape of target and prediction must match.'
                     f'{target.shape} vs {prediction.shape}.')
  # TODO(aarnab): Correctly handle the case where the input is normalised by
  # mean and std-dev instead of [-1, 1].
  # if jnp.max(target) > 1 or jnp.min(target) < -1:
  #   raise ValueError('Invalid ranges in target.')
  if modality == 'spectrogram':
    prediction_clipped = prediction
  else:
    prediction_clipped = jnp.clip(prediction, a_min=-1, a_max=1)

  # Normalise to uint8 in range [0, 255] for summary-writing.
  def normalise(tensor: jnp.ndarray, offset: float = 127.5) -> jnp.ndarray:
    if modality == 'spectrogram':
      return tensor
    else:
      return tensor * offset + offset

  target_normalised = normalise(target)
  prediction_normalised = normalise(prediction_clipped)

  # if prediction_masks is not None:
  # Mask out corresponding part of the input.
  mask_image = jnp.zeros(target.shape)
  pred_mask = jnp.expand_dims(prediction_masks, axis=-1)
  input_normalized = target_normalised

  target_normalised = ((1 - pred_mask) * target_normalised +
                       pred_mask * mask_image)
  pred_unmasked_tokens = ((1 - pred_mask) * prediction_normalised +
                          pred_mask * mask_image)
  pred_masked_tokens = ((1 - pred_mask) * mask_image +
                        pred_mask * prediction_normalised)
  pred_with_unmasked_replaced = (
      (1 - pred_mask) * input_normalized + pred_mask * prediction_normalised)

  batch_size = target.shape[0]
  height, width, channels = input_size
  n_rows = batch_size // n_columns
  if batch_size != n_rows * n_columns:
    raise ValueError('`n_columns` must divide the number of images evenly')

  def unpatch_to_images(tensor: jnp.ndarray) -> jnp.ndarray:
    n_patches_h = height // patch_size[0]
    n_patches_w = width // patch_size[1]
    # Reshape the tensor as expected by the `patch_image` function.
    tensor_reshaped = jnp.reshape(
        tensor, (batch_size, n_patches_h, n_patches_w, patch_size[0],
                 patch_size[1], channels))
    return nn_ops.patch_image(tensor_reshaped,
                              inputs_shape=(batch_size, height,
                                            width, channels),
                              patch_size=patch_size,
                              mode='p2i')

  input_images = unpatch_to_images(input_normalized)
  target_images = unpatch_to_images(target_normalised)
  prediction_images = unpatch_to_images(prediction_normalised)
  masked_token_images = unpatch_to_images(pred_masked_tokens)
  unmasked_token_images = unpatch_to_images(pred_unmasked_tokens)
  pred_with_unmasked_replaced_images = unpatch_to_images(
      pred_with_unmasked_replaced)

  num_images = 6
  if modality == 'spectrogram':
    images_concat = jnp.array(
        [jnp.concatenate([x1, x2, x3, x4, x5, x6], axis=1)  # pylint: disable= g-complex-comprehension
         for x1, x2, x3, x4, x5, x6 in zip(
             input_images, target_images, prediction_images,
             masked_token_images, unmasked_token_images,
             pred_with_unmasked_replaced_images)])
    image_grid = images_concat.reshape(
        n_rows, n_columns, height, num_images * width, channels).swapaxes(1, 2)

    image_grid = image_grid.reshape(
        height * n_rows, num_images * width * n_columns, channels)
  else:
    images_concat = jnp.array(
        [jnp.concatenate([x1, x2, x3, x4, x5, x6], axis=0)  # pylint: disable= g-complex-comprehension
         for x1, x2, x3, x4, x5, x6 in zip(
             input_images, target_images, prediction_images,
             masked_token_images, unmasked_token_images,
             pred_with_unmasked_replaced_images)])

    image_grid = images_concat.reshape(n_rows, n_columns, num_images * height,
                                       width, channels).swapaxes(1, 2)

    image_grid = image_grid.reshape(
        num_images * height * n_rows, width * n_columns, channels
        ).astype(jnp.uint8)

  if modality == 'spectrogram':
    # Normalize the entire image
    image_grid = image_grid - image_grid.min()
    image_grid = image_grid / image_grid.max()

    cm = plt.get_cmap('viridis')
    # plt.get_cmap expects uint8 as input
    image_grid = image_grid * 255
    image_grid = cm(image_grid[:, :, 0].astype(jnp.uint8))
    # image_grid is between [0, 1]
    image_grid = (image_grid[:, :, :3] * 255).astype(jnp.uint8)

  return image_grid


def generate_image_grid_from_video(target: jnp.ndarray,
                                   prediction: jnp.ndarray,
                                   prediction_masks: jnp.ndarray,
                                   patch_size: Tuple[int, int, int],
                                   input_size: Tuple[int, int, int, int],
                                   select_central_frame: bool,
                                   n_columns: int = 1,
                                   num_img_in_column: int = 1):
  """Generates a grid of images for a video for logging summaries.

  When select_central_frame=True, all the reconstructed frames
  ( = video length / temporal size) will be stacked and shown onto a single row.
  When select_central_frame=False, the reconstructed frames
  ( = video length) will be split into multiple rows by num_img_in_column. It is
  highly recommended to set n_columns to 1 if the video is long.

  Args:
   target: Array of shape [batch, num_tokens, patch_size * patch_size *
     channels] or [batch, num_tokens, patch_size_t * patch_size_h * patch_size_w
     channels] depending on the value of select_central_frame.
     Values are in the range [-1, 1].
   prediction: Array of same shape as target. It is possible that values are
     outside of the range [-1, 1].
   prediction_masks: Array of shape [batch, num_tokens]. Each entry is binary,
     ie 0 or 1. Used to mask out patches in the input and predictions.
   patch_size: The patch size of the model.
   input_size: The size of the input TxHxWxC without the batch dimension.
   select_central_frame: If only the central frame is used for reconstruction.
   n_columns: The number of columns to generate in the summary. When
   select_central_frame=False is highly recommended to set it to 1.
   num_img_in_column: This is used only when select_central_frame=False. It
   splits the frames from an example onto multiple rows instead of stacking them
   onto a single row. It must be a multiple of the temporal size.
  Returns:
    An image grid. The shape is [height, width, channels].
    Data type is uint8, and values are in the range [0, 255].
  """

  if target.shape != prediction.shape:
    raise ValueError('Shape of target and prediction must match.'
                     f'{target.shape} vs {prediction.shape}.')

  # TODO(lgeorgescu): Correctly handle the case where the input is normalised by
  # mean and std-dev instead of [-1, 1].
  # if jnp.max(target) > 1 or jnp.min(target) < -1:
  #   raise ValueError('Invalid ranges in target.')
  prediction_clipped = jnp.clip(prediction, a_min=-1, a_max=1)

  # Normalise to uint8 in range [0, 255] for summary-writing.
  def normalise(tensor: jnp.ndarray, offset: float = 127.5) -> jnp.ndarray:
    return tensor * offset + offset

  target_normalised = normalise(target)
  prediction_normalised = normalise(prediction_clipped)

  # if prediction_masks is not None:
  # Mask out corresponding part of the input.
  mask_image = jnp.zeros(target.shape)
  pred_mask = jnp.expand_dims(prediction_masks, axis=-1)
  input_normalised = ((1 - pred_mask) * target_normalised +
                      pred_mask * mask_image)
  pred_unmasked_tokens = ((1 - pred_mask) * prediction_normalised +
                          pred_mask * mask_image)
  pred_masked_tokens = ((1 - pred_mask) * mask_image +
                        pred_mask * prediction_normalised)

  pred_with_unmasked_replaced = (
      (1 - pred_mask) * input_normalised + pred_mask * prediction_normalised)

  batch_size = target.shape[0]
  if select_central_frame:
    channels = int(target.shape[2] / (patch_size[0] * patch_size[1]))
  else:
    channels = int(target.shape[2] / (patch_size[0] * patch_size[1]
                                      * patch_size[2]))

  height = input_size[1]
  width = input_size[2]
  temporal_dims = input_size[0] // patch_size[2]
  n_rows = batch_size // n_columns
  if batch_size != n_rows * n_columns:
    raise ValueError('`n_columns` must divide the number of images evenly')

  def unpatch_to_images(tensor: jnp.ndarray) -> jnp.ndarray:
    n_patches_h = height // patch_size[0]
    n_patches_w = width // patch_size[1]

    if select_central_frame:
      # Reshape the tensor as expected by the `patch_image` function.
      tensor_reshaped = jnp.reshape(tensor, (batch_size, temporal_dims,
                                             n_patches_h, n_patches_w,
                                             patch_size[0], patch_size[1], 3))

      images_list = jax.vmap(
          functools.partial(nn_ops.patch_image,
                            inputs_shape=(batch_size, height, width, 3),
                            patch_size=patch_size[:2],
                            mode='p2i'),
          in_axes=1, out_axes=0)(tensor_reshaped)

      final_image = jnp.concatenate(images_list, axis=2)
      return final_image

    else:
      # Reshape the tensor as expected by the `patch_image` function.
      tensor_reshaped = jnp.reshape(tensor, (batch_size, temporal_dims,
                                             n_patches_h, n_patches_w,
                                             patch_size[2], patch_size[0],
                                             patch_size[1], 3))
      images_list = []
      for temporal_video_idx in range(temporal_dims):
        images_list_patch = jax.vmap(
            functools.partial(nn_ops.patch_image,
                              inputs_shape=(batch_size, height, width, 3),
                              patch_size=patch_size[:2],
                              mode='p2i'),
            in_axes=3, out_axes=0)(tensor_reshaped[:, temporal_video_idx])

        images_list.extend(images_list_patch)

      if num_img_in_column % patch_size[2] != 0:
        raise ValueError('`patch_size[2]` must divide'
                         'the `num_img_in_column` evenly!')

      grouped_images_list = np.array_split(
          images_list, max(1, len(images_list) // num_img_in_column))

      final_image = jnp.concatenate(
          [jnp.concatenate(sub_images_list, axis=2)
           for sub_images_list in grouped_images_list], axis=1)
      return final_image

  input_images = unpatch_to_images(input_normalised)
  pred_images = unpatch_to_images(prediction_normalised)
  masked_token_images = unpatch_to_images(pred_masked_tokens)
  unmasked_token_images = unpatch_to_images(pred_unmasked_tokens)
  target_images = unpatch_to_images(target_normalised)
  pred_with_unmasked_replaced_images = unpatch_to_images(
      pred_with_unmasked_replaced)

  images_concat = jnp.array([
      jnp.concatenate([x1, x2, x3, x4, x5, x6], axis=0)  # pylint: disable= g-complex-comprehension
      for x1, x2, x3, x4, x5, x6 in zip(target_images, input_images,
                                        pred_images,
                                        masked_token_images,
                                        unmasked_token_images,
                                        pred_with_unmasked_replaced_images)
  ])
  # add margin 2% of the image height
  margin_size = max(int(0.02 * images_concat.shape[1]), 10)
  margin = np.ones((images_concat.shape[0], margin_size,
                    images_concat.shape[2], images_concat.shape[3]),
                   dtype=np.uint8) * 255
  images_concat = jax.numpy.concatenate((images_concat, margin), axis=1)

  new_height = unmasked_token_images.shape[1]
  new_width = unmasked_token_images.shape[2]
  image_grid = images_concat.reshape(n_rows, n_columns,
                                     6 * new_height + margin_size,
                                     new_width, channels).swapaxes(1, 2)
  image_grid = image_grid.reshape((6 * new_height * n_rows +
                                   batch_size * margin_size),
                                  new_width * n_columns,
                                  channels).astype(jnp.uint8)
  return image_grid


def get_random_bounding_box(
    image_shape: Tuple[int, int],
    ratio: jnp.ndarray,
    rng: Any,
    margin: float = 0.) -> Tuple[int, int, int, int]:
  """Returns a random bounding box for Cutmix.

  Based on the implementation in timm:
  https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py

  Args:
    image_shape: The shape of the image, specified as [height, width].
    ratio: Ratio of the input height/width to use as the maximum dimensions of
      the randomly sampled bounding box.
    rng: JAX rng key.
    margin: Percentage of bounding box dimension to enforce as the margin.
      This reduces the amount of the bounding box outside the image.

  Returns:
    The bounding box parameterised as y_min, y_max, x_min, x_max.
  """
  img_h, img_w = image_shape
  cut_h, cut_w = (img_h * ratio).astype(int), (img_w * ratio).astype(int)
  margin_y, margin_x = (margin * cut_h).astype(int), (margin *
                                                      cut_w).astype(int)
  rngx, rngy = jax.random.split(rng)
  cy = jax.random.randint(rngy, [1], 0 + margin_y, img_h - margin_y)
  cx = jax.random.randint(rngx, [1], 0 + margin_x, img_w - margin_x)

  y_min = jnp.clip(cy - cut_h // 2, 0, img_h)[0]
  y_max = jnp.clip(cy + cut_h // 2, 0, img_h)[0]
  x_min = jnp.clip(cx - cut_w // 2, 0, img_w)[0]
  x_max = jnp.clip(cx + cut_w // 2, 0, img_w)[0]
  return y_min, y_max, x_min, x_max  # pytype: disable=bad-return-type  # jnp-type


def _do_mixup(inputs: jnp.ndarray, rng: Any,
              alpha: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Performs Mixup.

  Args:
    inputs: The input images of shape NHWC or NTHWC. Mixup is always performed
      along the leading axis of the array, i.e., along the "N" dimension.
    rng: A PRNGKey. Will be consumed by this function.
    alpha: The alpha value for mixup.

  Returns:
    The modified images and label weights.
  """
  batch_size = inputs.shape[0]
  weight = jax.random.beta(rng, alpha, alpha)
  weight *= jnp.ones((batch_size, 1))

  # Mixup inputs.
  # Shape calculations use np to avoid device memory fragmentation:
  image_weight_shape = np.ones(inputs.ndim, np.int32)
  image_weight_shape[0] = batch_size
  image_weight = weight.reshape(image_weight_shape)
  reverse = tuple(
      slice(inputs.shape[i]) if i > 0 else slice(-1, None, -1)
      for i in range(inputs.ndim))
  result_img = (image_weight * inputs + (1.0 - image_weight) * inputs[reverse])
  return result_img, weight


def _do_cutmix(inputs: jnp.ndarray, rng: Any,
               alpha: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Performs Cutmix.

  Args:
    inputs: The input images of shape NHWC or NTHWC.
    rng: A PRNGKey. Will be consumed by this function.
    alpha: The alpha value for cutmix.

  Returns:
    The modified images and label weights.
  """
  rng, beta_key = jax.random.split(rng)
  cutmix_lambda = jax.random.beta(beta_key, alpha, alpha)
  ratio = jnp.sqrt(1 - cutmix_lambda)

  # TODO(unterthiner): we are using the same bounding box for the whole batch
  y_min, y_max, x_min, x_max = get_random_bounding_box(
      inputs.shape[-3:-1], ratio, rng)

  height, width = inputs.shape[-3], inputs.shape[-2]
  y_idx = jnp.arange(height)
  x_idx = jnp.arange(width)
  mask0 = (y_min <= y_idx) & (y_idx < y_max)
  mask1 = (x_min <= x_idx) & (x_idx < x_max)
  mask = (~jnp.outer(mask0, mask1)).astype(int)
  if inputs.ndim == 4:  # image format NWHC
    mask = jnp.expand_dims(mask, axis=(0, -1))
  elif inputs.ndim == 5:  # image format NTWHC
    mask = jnp.expand_dims(mask, axis=(0, 1, -1))
  else:
    raise ValueError('Invalid image format')

  result_img = (inputs * mask + jnp.flip(inputs, axis=0) * (1.0 - mask))
  box_area = (y_max - y_min) * (x_max - x_min)
  weight = 1.0 - box_area / float(height * width)
  weight *= jnp.ones((inputs.shape[0], 1))
  return result_img, weight


def mixup_cutmix(batch: Dict['str', jnp.ndarray],
                 rng: Any,
                 mixup_alpha: float = 1.0,
                 cutmix_alpha: float = 0.,
                 switch_prob: float = 0.5,
                 label_smoothing: float = 0.0) -> Dict['str', jnp.ndarray]:
  """Performs Mixup or Cutmix within a single batch.

  For more details on Mixup, please see https://arxiv.org/abs/1710.09412.
  And for details on Cutmix, refer to https://arxiv.org/abs/1905.04899.

  Based on the implementation from:
  https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py

  This function supports `jax.numpy` to do mixup within a jitted/pmapped
  function (e.g. within a pmapped train step to apply mixup on device patch).

  Results in a batch with:
    mixed_images[idx] = weight * images[idx] + (1-weight) * images[-(idx+1)],
    where weight is sampled from a beta distribution with parameter alpha.

  Args:
    batch: dict; A batch of data with 'inputs' and 'label'. 'inputs' is expected
      to have shape [batch, height, width, channels] or NTWHC.
    rng: JAX rng key. This key will be consumed by the function call.
    mixup_alpha: The alpha parameter of the beta distribution that the weight is
      sampled from.
    cutmix_alpha: The alpha parameter of the beta distribution that the cutmix
      weight is sampled from.
    switch_prob: The probability of switching to cutmix when both mixup and
      cutmix are enabled.
    label_smoothing: The co-efficient for label-smoothing. If using mixup or
      cutmix, this is done before mixing the labels.

  Returns:
    Tuple (mixed_images, mixed_labels).
  """

  if cutmix_alpha <= 0 and mixup_alpha <= 0:
    return batch

  images, labels = batch['inputs'], batch['label']
  if labels.shape[-1] == 1:
    raise ValueError('Mixup requires one-hot targets.')
  if images.ndim not in (4, 5):
    raise ValueError(f'Unexpected shape: {images.shape}, wanted 4 or 5 dims.')

  rng, rng_coinflip = jax.random.split(rng)
  coin_flip = jax.random.bernoulli(rng_coinflip, p=switch_prob)
  pick_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or coin_flip)

  alpha = jax.lax.cond(pick_cutmix, lambda: cutmix_alpha, lambda: mixup_alpha)
  batch['inputs'], label_weight = jax.lax.cond(pick_cutmix, _do_cutmix,
                                               _do_mixup, images, rng, alpha)

  if label_smoothing > 0:
    labels = model_utils.apply_label_smoothing(labels, label_smoothing)

  batch['label'] = label_weight * labels + (1.0 - label_weight) * labels[::-1]
  return batch


def log_note(note: str):
  """Logging function."""
  if jax.process_index() == 0:  # Only perform on the lead_host
    logging.info(note)
    platform.work_unit().set_notes(note)


def compute_max_norm(tensors: PyTree) -> float:
  """Compute the maximum norm in a pytree of tensors."""
  leaves, _ = jax.tree_util.tree_flatten(tensors)
  norms = jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))
  max_norm = jnp.max(norms)
  return max_norm  # pytype: disable=bad-return-type  # jnp-type


def test_step(
    train_state: Union[
        train_utils.TrainState, train_utils_deprecated.TrainState
    ],
    batch: Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: MetricFnEval,
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
  num_crops = batch['inputs'].shape[0]
  if isinstance(train_state, train_utils.TrainState):
    variables = {
        'params': train_state.params,
        **train_state.model_state
    }
  elif isinstance(train_state, train_utils_deprecated.TrainState):
    variables = {
        'params': train_state.optimizer.target,  # pytype: disable=attribute-error
        **train_state.model_state
    }
  else:
    raise ValueError('Unknown train_state type.')

  # TODO(aarnab): Implement this with jax.scan to improve efficiency.
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


def test_step_multimodal(
    train_state: Union[
        train_utils.TrainState, train_utils_deprecated.TrainState
    ],
    batch: Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: MetricFnEval,
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

  if isinstance(train_state, train_utils.TrainState):
    variables = {
        'params': train_state.params,
        **train_state.model_state
    }
  elif isinstance(train_state, train_utils_deprecated.TrainState):
    variables = {
        'params': train_state.optimizer.target,  # pytype: disable=attribute-error
        **train_state.model_state
    }
  else:
    raise ValueError('Unknown train_state type.')

  for modality in batch['inputs']:
    num_crops = batch['inputs'][modality].shape[0]
  for idx in range(0, num_crops, n_clips):
    current_input = {}
    for modality in batch['inputs']:
      current_input[modality] = batch['inputs'][modality][idx:idx + n_clips]
    logits = flax_model.apply(
        variables, current_input, train=False, mutable=False, debug=debug)

    if softmax_logits:
      logits = nn.softmax(logits, axis=-1)
    logits = jnp.sum(logits, axis=0)
    all_logits = all_logits + logits

  # Average logits accross all views (segments) within the clip.
  all_logits = all_logits / num_crops
  all_logits = jnp.expand_dims(all_logits, axis=0)
  batch['label'] = jnp.expand_dims(batch['label'][0], axis=0)
  batch['batch_mask'] = jnp.expand_dims(batch['batch_mask'][0], axis=0)
  metrics = metrics_fn(all_logits, batch)
  if return_logits_and_labels:
    # Here we have the logits predicted for one eval clip, but mAP is computed
    # over the entire eval set. So we
    # 1) Gather & return logits and labels for the N eval clips processed in
    # N hosts during this test_step,
    # 2) Repeat for M test_batches (steps_per_test) needed to traverse eval set,
    # 3) Once we gathered logits and labels for the entire eval set,
    # we compute the mAP.
    all_logits = jax.lax.all_gather(all_logits, 'batch')
    labels = jax.lax.all_gather(batch['label'], 'batch')
    return metrics, all_logits, labels
  return metrics


def mixup_modalities(batch: Dict[str, Any],
                     alpha: float = 1.0,
                     batch_first: bool = True,
                     mixmod: bool = False,
                     rng: Optional[Any] = None) -> Dict['str', jnp.ndarray]:
  """Mixes multimodal inputs and labels within a single batch.

  For more details, please see https://arxiv.org/abs/1710.09412.

  This function supports both using `numpy` to do mixup in the input-pipeline
  and `jax.numpy` to do mixup within a jitted/pmapped function (e.g. within
  a pmapped train step to apply mixup on device patch).

  Results in a batch with:
    mixed_inputs[idx] = weight * inputs[idx] + (1-weight) * inputs[-(idx+1)],
    where weight is sampled from a beta distribution with parameter alpha.

  Args:
    batch: dict; A batch of data with 'inputs' and 'label'. batch['inputs'] has
      field like 'rgb', 'flow', spectrogram', 'waveform' or 'text'.
    alpha: float; Used to control the beta distribution that weight is sampled
      from.
    batch_first: bool; Batch is the first dimension or the last dimension.
    mixmod: bool; If True, applies mixup to each modality separately.
    rng: JAX rng key. If given, JAX numpy will be used as the backend, and if
      None (default value), normal numpy will be used.

  Returns:
    Tuple (mixed_images, mixed_labels).
  """
  inputs, labels = batch['inputs'], batch['label']
  batch['label'] = {}
  num_modalities = len(inputs)

  if labels.shape[-1] == 1:
    raise ValueError('Mixup requires one-hot targets.')

  batch_size = labels.shape[0]

  if mixmod:
    weights = list(jax.random.beta(rng, alpha, alpha, shape=[num_modalities]))
  else:
    weights = [jax.random.beta(rng, alpha, alpha)] * num_modalities
  for i in range(num_modalities):
    weights[i] *= jnp.ones((batch_size, 1))

  # Mixup inputs.
  # Shape calculations use np to avoid device memory fragmentation:
  for modality, values in inputs.items():
    weight = weights[len(batch['label'])]
    # Mixup labels.
    batch['label'][modality] = weight * labels + (1.0 - weight) * labels[::-1]
    weight_shape = np.ones((values.ndim))
    if batch_first:
      weight_shape[0] = batch_size
    else:
      weight_shape[-1] = batch_size
    weight = jnp.reshape(weight, weight_shape.astype(jnp.int32))
    reverse = []
    for i in range(values.ndim):
      if (i == 0 and batch_first) or (i == values.ndim - 1 and not batch_first):
        reverse.append(slice(-1, None, -1))
      else:
        reverse.append(slice(values.shape[i]))
    batch['inputs'][modality] = (weight * values +
                                 (1.0 - weight) * values[tuple(reverse)])
  if num_modalities == 1 or not mixmod:
    batch['label']['all'] = weights[0] * labels + (1.0 -
                                                   weights[0]) * labels[::-1]

  return batch
