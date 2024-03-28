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

"""Utility functions for mixup-cutmix augmentation."""
from typing import Any, Dict, Tuple

from absl import logging

import jax
import jax.numpy as jnp
import numpy as np
from scenic.model_lib.base_models import model_utils

PyTree = Any


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
  """Write note to XManager and also log to the console."""
  if jax.process_index() == 0:  # Only perform on the lead_host
    logging.info(note)


def compute_max_norm(tensors: PyTree) -> float:
  """Compute the maximum norm in a pytree of tensors."""
  leaves, _ = jax.tree_util.tree_flatten(tensors)
  norms = jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))
  max_norm = jnp.max(norms)
  return max_norm  # pytype: disable=bad-return-type  # jnp-type
