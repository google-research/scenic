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

"""Model utils functions."""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from absl import logging
from clu import preprocess_spec
from flax import core as flax_core
import jax
import jax.numpy as jnp
import numpy as np
from scenic.train_lib import train_utils
import scipy


def resize_posemb(posemb, target_size):
  """Resizes position embeddings to new resolution."""
  if target_size == posemb.shape[-2]:
    return posemb
  posemb = jax.device_get(posemb)
  ndim = posemb.ndim
  if ndim == 3:
    posemb = posemb[0]

  gs_old = int(np.sqrt(posemb.shape[0]))
  gs_new = int(np.sqrt(target_size))

  posemb_tok = None
  if gs_old**2 == posemb.shape[0]:  # No CLS token.
    posemb_grid = posemb
  elif gs_old**2 == posemb.shape[0] - 1:  # Prepended CLS token.
    posemb_tok, posemb_grid = posemb[:1], posemb[1:]
  else:
    raise ValueError(
        'Posemb shape must be a perfect square (maybe with CLS token), but '
        f'got posemb of shape {posemb.shape}.')

  logging.info('Posemb: grid-size from %s to %s', gs_old, gs_new)
  posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

  zoom = (gs_new / gs_old, gs_new / gs_old, 1)
  posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
  posemb = posemb_grid.reshape(gs_new * gs_new, -1)
  if posemb_tok is not None:
    posemb = np.concatenate([posemb_tok, posemb], axis=0)

  return jnp.array(posemb[np.newaxis] if ndim == 3 else posemb)


def seq2img(original_img: jnp.ndarray, features: jnp.ndarray) -> jnp.ndarray:
  """Reshapes 1D sequence to 2D image features."""
  if original_img.shape[1] == original_img.shape[2]:
    h = w = int(np.sqrt(features.shape[1]))
  else:
    stride = np.ceil(np.sqrt(
        original_img.shape[1] * original_img.shape[2] / features.shape[1]))
    h = np.ceil(original_img.shape[1] / stride)
    w = np.ceil(original_img.shape[2] / stride)
  return features.reshape(features.shape[0], int(h), int(w), -1)


def normalized_grid_corner_coordinates(
    feature_map: jnp.ndarray, padding_mask: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
  """Computes normalized xy corner coords from feature_map or padding_mask."""
  # Note 1: it computes not the centers of grid patches, but the patch corner
  # coordinates (for a grid patch from 0 to 0.1, it returns 0.1 not 0.05).
  # Note 2: behavior is quite different for feature_map and padding_mask inputs.
  if padding_mask is None:
    assert feature_map.ndim == 4  # [B, H, W, C]
    h, w = feature_map.shape[1:3]
    xy = np.stack(
        np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1)),
        axis=-1).astype(np.float32)
    xy /= np.array([w, h], np.float32)
  else:
    assert padding_mask.ndim == 3  # [B, H, W]
    y = jnp.cumsum(padding_mask, axis=1)
    x = jnp.cumsum(padding_mask, axis=2)
    xy = jnp.stack([x / (x[:, :, -1:] + 1e-6), y / (y[:, -1:] + 1e-6)], axis=-1)
  # Flatten h, w dimensions.
  return xy.reshape(*(xy.shape[:-3] + (-1, 2)))


def compute_box_bias(
    feature_map: jnp.ndarray,
    padding_mask: Optional[jnp.ndarray] = None,
    kind: str = 'both') -> jnp.ndarray:
  """Computes spatial bias for grid."""
  # The box center is biased to its position on the feature grid:
  xy = normalized_grid_corner_coordinates(feature_map, padding_mask)
  xy = jnp.clip(xy, 0.0, 1.0)

  if kind in ['both', 'location']:
    # Unnormalize xy (i.e., apply logit function/sigmoid^-1).
    xy_bias = logit(xy)
  else:
    xy_bias = jnp.zeros_like(xy)

  if kind in ['both', 'size']:
    # The box size is biased to the patch size:
    wh_bias = logit(jnp.full_like(xy_bias, 1.0 / feature_map.shape[-2]))
  else:
    wh_bias = jnp.zeros_like(xy_bias)

  return jnp.concatenate([xy_bias, wh_bias], axis=-1)


def logit(x, eps=1e-4):
  """Logit (inverse sigmoid) function (https://en.wikipedia.org/wiki/Logit)."""
  return jnp.log(x + eps) - jnp.log1p(-x + eps)


def init_classification_bias(bias: jnp.ndarray,
                             prior_prob: float) -> jnp.ndarray:
  return jnp.full(bias.shape, np.log(prior_prob) - np.log1p(-prior_prob))


def dot_product_similarity(x: jnp.ndarray,
                           y: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  return jnp.einsum('bnd,bmd->bnm', x, x if y is None else y)


def l2_norm(tree):
  """Computes the l2 norm of a pytree of arrays."""
  leaves = jax.tree_util.tree_leaves(tree)
  return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))


def _sorted_items(x):
  """Returns items of a dict ordered by keys."""
  return sorted(x.items())


def _get_params_dict(inputs):
  if isinstance(inputs, (dict, flax_core.FrozenDict)):
    return flax_core.unfreeze(inputs)
  else:
    raise ValueError(
        'Can only traverse a flax Model instance or a nested dict, not '
        f'{type(inputs)}')


def find_op(
    ops: Sequence[preprocess_spec.PreprocessOp],
    target_op_class: Union[Type[preprocess_spec.PreprocessOp],
                           Tuple[Type[preprocess_spec.PreprocessOp], ...]]
) -> preprocess_spec.PreprocessOp:
  """Find an op of the target class in a sequence of op instances."""
  result = [op for op in ops if isinstance(op, target_op_class)]
  if len(result) == 1:
    return result[0]
  elif len(result) > 1:
    raise ValueError(
        f'Found multiple candidate ops, please disambiguate: {result}')
  else:
    raise ValueError(f'Op not found: {target_op_class}')


def normalize_metrics_summary(metrics_summary: Dict[str, Any], split: str,
                              object_detection_loss_keys: List[str]):
  """Normalizes the metrics in the given metrics summary.

  Note that currently we only support metrics of the form 1/N sum f(x_i).

  Args:
    metrics_summary: Each value is a sum of a calculated metric over all
      examples.
    split: Split for which we normalize the metrics. Used for logging.
    object_detection_loss_keys: A loss key used for computing the object
      detection loss.

  Returns:
    Normalized metrics summary.

  Raises:
    TrainingDivergedError: Due to observing a NaN in the metrics.
  """
  for key, val in metrics_summary.items():
    metrics_summary[key] = val[0] / val[1]
    if np.isnan(metrics_summary[key]):
      raise train_utils.TrainingDivergedError(
          'NaN detected in {}'.format(f'{split}_{key}'))

  # Compute and add object_detection_loss using globally normalize terms:
  object_detection_loss = 0
  for loss_term_key in object_detection_loss_keys:
    object_detection_loss += metrics_summary[loss_term_key]
  metrics_summary['object_detection_loss'] = object_detection_loss

  return metrics_summary
