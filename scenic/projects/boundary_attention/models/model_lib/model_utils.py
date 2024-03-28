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

"""Custom unfolding for image patches."""

from typing import Tuple

from jax import lax
import jax.numpy as jnp


def custom_unfold(im, patchsize, stride, hpatches, wpatches):
  """Extract patches from an image.

  Args:
    im: Array of shape [N, H, W, C]
    patchsize: Tuple of integers representing the filter shape.
    stride: Integer representing the stride or step size.
    hpatches: Integer representing how many vertical patches.
    wpatches: Integer representing how many horizontal patches.

  Returns:
    Array of shape [N, C, R, R, H', W'] containing all image patches.
    E.g. [k,l,:,:,i,j] is the lth channel of the (i,j)th patch of the kth image
  """

  patches = lax.conv_general_dilated_patches(
      im, filter_shape=patchsize,
      window_strides=[stride, stride],
      padding='SAME',
      dimension_numbers=('NHWC', 'HWIO', 'NHWC'))

  return patches.reshape([-1, hpatches, wpatches,
                          im.shape[-1], patchsize[0],
                          patchsize[1]]).transpose(0, 1, 2, 4, 5, 3)


def extract_patches(image: jnp.ndarray,
                    patchsize: Tuple[int, int],
                    stride: int):
  """Extracts patches from the input image.

  Args:
      image: The input image of shape (batch, height, width, channels).
      patchsize (int): The size of the patches to extract.
      stride (int): The stride or step size to move the window for each patch.

  Returns:
    Extracted patches of shape (batch, out_height, out_width,
    patch_size, patch_size, channels).
  """

  hpatches = image.shape[1]
  wpatches = image.shape[2]

  # Create patches using einops
  patches = custom_unfold(image, patchsize, stride, hpatches, wpatches)
  mask = custom_unfold(jnp.ones_like(image)[:, :, :, 0:1], patchsize, stride,
                       hpatches, wpatches)

  return patches, mask
