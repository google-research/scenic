"""Model utils functions."""
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


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
  leaves = jax.tree_leaves(tree)
  return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))
