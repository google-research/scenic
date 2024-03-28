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

"""Transforms which we can attack through."""
import jax
import jax.numpy as jnp


def _to_grid(x, patch_sidelength):
  """To grid."""
  n, h, w, three = x.shape
  assert three == 3
  gh = h // patch_sidelength
  gw = w // patch_sidelength
  fh, fw = patch_sidelength, patch_sidelength

  x = jnp.reshape(x, [n, gh, fh, gw, fw, 3])
  x = jnp.transpose(x, [0, 1, 3, 2, 4, 5])
  x_grid = jnp.reshape(x, [n, gh, gw, fh, fw, 3])
  return x_grid


def _from_grid(x, patch_sidelength):
  """From grid."""
  (n, gh, gw, fh, fw, _) = x.shape
  fh, fw = patch_sidelength, patch_sidelength

  x = x.reshape([n, gh, gw, fh, fw, 3])
  x = x.transpose([0, 1, 3, 2, 4, 5])
  x = x.reshape([n, gh*fh, gw*fw, 3])
  return x


def fast_color_perturb(input_image, aug_params):
  return jnp.clip(input_image + aug_params.reshape(1, 1, 3), -1, 1)


def patched_color_jitter(input_image, aug_params, aug_fn=fast_color_perturb):
  """Color jitter applied to patch granularity."""
  local_batch_size, num_patches, num_patches_1, three = aug_params.shape
  assert three == 3
  assert num_patches == num_patches_1

  local_batch_size_1, h, w, three = input_image.shape
  assert three == 3
  assert h == w
  # assert h == 224
  assert local_batch_size_1 == local_batch_size

  patch_sidelength = h // num_patches
  grid = _to_grid(input_image, patch_sidelength)

  fast_color_jitter_vmapped = jax.vmap(aug_fn, in_axes=0, out_axes=0)
  fast_color_jitter_vmapped = jax.vmap(
      fast_color_jitter_vmapped, in_axes=1, out_axes=1)
  fast_color_jitter_vmapped = jax.vmap(
      fast_color_jitter_vmapped, in_axes=2, out_axes=2)

  jittered_grid = fast_color_jitter_vmapped(grid, aug_params)
  jittered = _from_grid(jittered_grid, patch_sidelength)
  return jittered

