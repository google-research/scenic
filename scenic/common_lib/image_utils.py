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

"""Image-related utility functions."""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


def compress_masks(mask_probs, k=3):
  """At each pixel, stores the largest k probabilities and their indices."""
  if mask_probs.ndim == 5:
    mask_probs = jnp.squeeze(mask_probs, axis=-1)  # Remove channel dim.
  # Input shape should be [b, num_queries, out_h, out_w].
  assert mask_probs.ndim == 4, f'Expected 4-D input, got {mask_probs.shape}'
  mask_probs = jnp.transpose(mask_probs, [0, 2, 3, 1])
  vals, inds = jax.lax.top_k(mask_probs, k=k)
  # Back to [b, k, out_h, out_w]
  vals = jnp.transpose(vals, [0, 3, 1, 2])
  inds = jnp.transpose(inds, [0, 3, 1, 2])
  return vals, inds


def decompress_masks(compressed_masks, num_queries):
  """Reconstructs the uncompressed mask representation."""
  vals, inds = compressed_masks
  b, _, h, w = vals.shape
  mask_probs = np.zeros((b, num_queries, h, w))
  ib, _, ih, iw = np.meshgrid(
      range(b), range(1), range(h), range(w), indexing='ij')
  mask_probs[ib, inds, ih, iw] = vals
  return mask_probs


def resize_pil(image_or_batch: np.ndarray,
               *,
               out_h: int,
               out_w: int,
               num_batch_dims: Optional[int] = None,
               method: str = 'linear') -> np.ndarray:
  """Resizes an image or batch of images using PIL.

  This function handles images with or without channel dimension, but requires
  any leading batch dimensions to be specified explicitly to avoid ambiguities.

  Args:
    image_or_batch: Image or batch of images.
    out_h: Image height after resizing.
    out_w: Image width after resizing.
    num_batch_dims: Number of leading dimensions that are to be treated as batch
      dimensions, e.g. 0 for single images or 1 for simple batches. If None, the
      input is assumed to be a single image.
    method: String indicating the resizing method. One of "linear" or "nearest".

  Returns:
    Resized image or batch of images.
  """
  if num_batch_dims is None:
    num_batch_dims = 0
    if image_or_batch.ndim > 3 or (image_or_batch.ndim == 3 and
                                   image_or_batch.shape[-1] not in [3, 4]):
      raise ValueError('If a batch of images is supplied, num_batch_dims must '
                       'be specified.')

  if method == 'linear':
    resample = Image.Resampling.BILINEAR
  elif method == 'nearest':
    resample = Image.Resampling.NEAREST
  elif method == 'lanczos':
    resample = Image.Resampling.LANCZOS
  else:
    raise NotImplementedError(f'Method not implemented: {method}')

  batch_dims = image_or_batch.shape[:num_batch_dims]
  image_dims = image_or_batch.shape[num_batch_dims:]
  batch = np.reshape(image_or_batch, (-1,) + image_dims)

  pil_size = [out_w, out_h]
  resized = np.stack([
      np.asarray(Image.fromarray(image).resize(pil_size, resample))
      for image in batch
  ])

  return np.reshape(resized, batch_dims + resized.shape[1:])
