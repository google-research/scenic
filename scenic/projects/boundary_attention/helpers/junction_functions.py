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

"""Maps Junctions To Images."""

import einops
import jax
import jax.numpy as jnp
import ml_collections
from scenic.projects.boundary_attention.helpers import render_junctions


class JunctionFunctions(render_junctions.JunctionRenderer):
  """Maps Junctions To Images."""

  def __init__(self,
               opts: ml_collections.ConfigDict,
               input_opts: ml_collections.ConfigDict):

    # First, define parameters that depend on opts
    self.patchmin = opts.patchmin
    self.patchmax = opts.patchmax
    self.patchsize = opts.patchsize
    self.stride = opts.stride
    self.num_wedges = opts.num_wedges
    self.delta = opts.delta
    self.eta = opts.eta

    self.mask_shape = opts.mask_shape
    self.jparameterization = opts.jparameterization
    self.bparameterization = opts.bparameterization
    self.patch_scales = opts.patch_scales

    # Next, define parameters that depend on input_opts
    self.height = input_opts.height
    self.width = input_opts.width
    self.channels = input_opts.channels
    self.hpatches = input_opts.hpatches
    self.wpatches = input_opts.wpatches

    self.val_features = [1, self.channels, self.height, self.width]
    self.val_boundaries = [1, 1, self.height, self.width]
    self.patch_density = self.fold(jnp.ones([1, 1,
                                             self.patchsize, self.patchsize,
                                             self.hpatches, self.wpatches]),
                                   self.val_boundaries)

  def unfold(self,
             im: jnp.ndarray,
             patchsize: int = 17,
             stride: int = 1,
             ) -> jnp.ndarray:
    """Extract patches from an image.

    Args:
      im: Array of shape [N, C, H, W] to be unfolded into patches
      patchsize: Size of extracted patches
      stride: Stride of extracted patches

    Returns:
      Array of shape [N, C, R, R, H', W'] containing all extracted patches.
      E.g. [k,l,:,:,i,j] is the lth channel of the (i,j)th patch of
      the kth image
    """
    _, channels, height, width = im.shape

    # Define patchsize and stride and use to calculate the number of patches for
    # each axis
    patchsize = self.patchsize if patchsize is None else patchsize
    stride = self.stride if stride is None else stride

    # If both patchsize and stride are undefined, use default
    hpatches = self.hpatches if (patchsize is None) and (stride is None) else (
        (height - patchsize) // stride + 1)
    wpatches = self.wpatches if (patchsize is None) and (stride is None) else (
        (width - patchsize) // stride + 1)

    patches = jax.lax.conv_general_dilated_patches(im,
                                                   filter_shape=(patchsize,
                                                                 patchsize),
                                                   window_strides=[stride,
                                                                   stride],
                                                   padding='VALID',
                                                   dimension_numbers=('NCHW',
                                                                      'HWIO',
                                                                      'NCHW'))
    patches = patches.reshape([-1, channels, patchsize, patchsize,
                               hpatches, wpatches])

    return patches

  def fold(self,
           unfolded: jnp.ndarray,
           output_shape: list[int],
           fn: str = 'sum',
           stride: int = 1,
           ) -> jnp.ndarray:
    """Fold patches of an image using function fn.

    Args:
      unfolded: Array of shape [N, C, R, R, H', W'] containing all unfolded
        patches.
      output_shape: Shape of folded array.
      fn: Function to fold with (example: mean or mode)
      stride: Stride of unfolded patches.

    Returns:
      Array of shape [N, C, H, W]
    """
    stride = self.stride if stride is None else stride

    kernel_size = (unfolded.shape[2], unfolded.shape[3])
    dilation = (1, 1)

    if isinstance(stride, int):
      stride = (stride, stride)

    # Calculate indices for each spatial location
    idx_h = jnp.arange(0, output_shape[2] - kernel_size[0] * dilation[0] + 1,
                       stride[0])
    idx_w = jnp.arange(0, output_shape[3] - kernel_size[1] * dilation[1] + 1,
                       stride[1])

    # Create a meshgrid for height and width indices
    grid_h, grid_w = jnp.meshgrid(idx_h, idx_w, indexing='ij')

    # Expand dimensions for broadcasting
    grid_h = grid_h[None, None, None, :, :]
    grid_w = grid_w[None, None, None, :, :]

    # Compute the window indices for height and width
    window_h_indices = jnp.arange(0, kernel_size[0] * dilation[0],
                                  dilation[0])[None, :, None, None, None]
    window_w_indices = jnp.arange(0, kernel_size[1] * dilation[1],
                                  dilation[1])[None, None, :, None, None]

    # Add the window indices to the grid indices
    h_indices = grid_h + window_h_indices
    w_indices = grid_w + window_w_indices

    # Compute indices for segment_sum
    batch_indices = jnp.arange(output_shape[0] *
                               output_shape[1])[:, None, None, None, None]

    indices = (batch_indices * output_shape[2] *
               output_shape[3] + h_indices * output_shape[3]+ w_indices)

    flattened_indices = indices.flatten()

    # Flatten unfolded array for segment_sum
    unfolded_flat = unfolded.flatten()

    if fn == 'sum':
      fold_fn = jax.ops.segment_sum
    elif fn == 'prod':
      fold_fn = jax.ops.segment_prod
    elif fn == 'min':
      fold_fn = jax.ops.segment_min
    elif fn == 'max':
      fold_fn = jax.ops.segment_max
    else:
      # Default to segment sum
      fold_fn = jax.ops.segment_sum

    # Use function to accumulate the values
    folded_flat = fold_fn(unfolded_flat, flattened_indices,
                          num_segments=output_shape[0] * output_shape[1] *
                          output_shape[2] * output_shape[3])

    # Reshape the result to the original shape
    folded = folded_flat.reshape(output_shape)

    return folded

  def local2global(self,
                   local_features: jnp.ndarray,
                   patch_density: jnp.ndarray,
                   stride: int = 1,
                   ) -> jnp.ndarray:
    """Takes feature patches and folds and normalizes to form global features.

    Args:
      local_features: Array of shape [N, C, R, R, H', W'].
      patch_density: Number of patches that overlap each pixel.
        Used for normalization.
      stride: Stride of the patches.

    Returns:
      jnp.ndarray containing folded global features
    """

    stride = self.stride if stride is None else stride

    batch, channels, patchsize, _, hpatches, wpatches = local_features.shape

    height = hpatches * stride + patchsize - 1
    width = wpatches * stride + patchsize - 1

    val_outputs = [batch, channels, height, width]

    # Calculate global features and boundaries
    global_outputs = self.fold(local_features, val_outputs,
                               stride=stride)/(patch_density + 1e-5)

    return global_outputs

  def get_avg_wedge_feature(self,
                            input_features,
                            global_features,
                            wedges,
                            patchsize=None,
                            stride=None,
                            lmbda_wedge_mixing=0.0):
    """Find smoothed patches of the image along with wedge colors.

    Args:
      input_features: Input features with shape [N, C, H, W]
      global_features: Current estimate of globally smoothed image with shape
      [N, C, H, W]
      wedges: Array with shape [N, M, R, R, H', W'] containing rendered wedges
      patchsize:  Patchsize of each patch.
      stride: Patch stride.
      lmbda_wedge_mixing: Mixing parameter. Determines how much to weigh current
      junction parameters versus new parameter estimates when determining wedge
      colors.

    Returns:
      patches: Array of shape [N, C, R, R, H', W'] containing wedges with
      average feature superimposed
      wedge_colors: Array of shape [N, C, M, H', W'] with wedge average feature
      for each wedge of each patch
    """

    patchsize = self.patchsize if patchsize is None else patchsize
    stride = self.stride if stride is None else stride

    input_feature_patches = self.unfold(input_features, patchsize, stride)
    current_global_feature_patches = self.unfold(global_features, patchsize,
                                                 stride)

    numerator = (jnp.expand_dims(input_feature_patches + lmbda_wedge_mixing *
                                 current_global_feature_patches,
                                 2) * jnp.expand_dims(wedges, 1)).sum([3, 4])
    denominator = (1.0 + lmbda_wedge_mixing) * jnp.expand_dims(wedges.sum([2,
                                                                           3]),
                                                               1)

    wedge_colors = numerator / (denominator + 1e-10)

    # Fill wedges with optimal colors
    patches = jnp.sum(jnp.expand_dims(wedges, 1) * jnp.expand_dims(wedge_colors,
                                                                   [3, 4]),
                      axis=2)

    return patches, wedge_colors

  def dist2bdry(self, dist_boundaries, delta=None):
    """Convert a distance map into a boundary map."""

    delta = self.delta if delta is None else delta

    return 1 / (1 + (dist_boundaries/delta)**2)

  def make_square_patch_masks(self, rf_size, patchsize=None):
    """Make square patch masks."""

    patchsize = self.patchsize if patchsize is None else patchsize

    xy = jnp.linspace(-jnp.floor((patchsize-1)/2), jnp.floor((patchsize-1)/2),
                      patchsize)
    xlim, ylim = jnp.meshgrid(xy, xy)
    mask = jnp.where((jnp.abs(xlim) < rf_size/2) &
                     (jnp.abs(ylim) < rf_size/2), 1, 0)

    return mask

  def make_circle_patch_masks(self, rf_size, patchsize=None):
    """Make circle patch masks."""

    patchsize = self.patchsize if patchsize is None else patchsize

    xy = jnp.linspace(-jnp.floor((patchsize-1)/2), jnp.floor((patchsize-1)/2),
                      patchsize)

    xlim, ylim = jnp.meshgrid(xy, xy)
    mask = jnp.where(jnp.sqrt(xlim**2 + ylim**2) <= rf_size/2, 1, 0)

    return mask

  def get_scale_masks(self, scales, mask_shape=None, patchsize=None):
    """Get scale masks for patches with variable patchsizes."""

    mask_shape = self.mask_shape if mask_shape is None else mask_shape
    patchsize = self.patchsize if patchsize is None else patchsize

    if mask_shape == 'square':
      flat_scale = einops.rearrange(scales, 'n f h w -> (n h w) f')
      masks = jax.vmap(self.make_square_patch_masks, in_axes=(0, None),
                       out_axes=0)(flat_scale, patchsize)
      masks = einops.rearrange(masks, '(n h w) i j -> n i j h w',
                               n=scales.shape[0], h=scales.shape[2],
                               w=scales.shape[3])

    elif mask_shape == 'circle':
      flat_scale = einops.rearrange(scales, 'n f h w -> (n h w) f')
      masks = jax.vmap(self.make_circle_patch_masks, in_axes=(0, None),
                       out_axes=0)(flat_scale, patchsize)
      masks = einops.rearrange(masks, '(n h w) i j -> n i j h w',
                               n=scales.shape[0], h=scales.shape[2],
                               w=scales.shape[3])

    else:
      raise NotImplementedError('%s not a valid mask shape') % mask_shape

    return masks

  def get_patch_density_and_masks(self, scales, mask_shape=None, patchsize=None,
                                  height=None, width=None):
    """Get patch density and masks."""

    height = self.height if height is None else height
    width = self.width if width is None else width
    patchsize = self.patchsize if patchsize is None else patchsize
    mask_shape = self.mask_shape if mask_shape is None else mask_shape

    scale_masks = self.get_scale_masks(scales, mask_shape, patchsize)

    return scale_masks, self.fold(scale_masks, [scale_masks.shape[0],
                                                scale_masks.shape[1], height,
                                                width])

  def get_alpha_omega_vertex(self, jparams, jparameterization=None,
                             num_wedges=None):
    """Maps output of model to alpha, omega, vertex."""

    num_wedges = self.num_wedges if num_wedges is None else num_wedges
    jparameterization = self.jparameterization if (jparameterization is
                                                   None) else jparameterization

    if jparameterization == 'standard':
      # default parameterization: (cos(alpha), sin(alpha), omega1, omega2,
      # omega3, u, v))
      alpha = jnp.expand_dims(jnp.arctan2(jparams[..., 1], jparams[..., 0]), 1)
      omega = jparams[..., 2:num_wedges+2].transpose(0, -1, 1, 2)
      vertex = jparams[..., num_wedges+2:].transpose(0, -1, 1, 2)

      # Normalize omega
      omega = omega*(2*jnp.pi)/jnp.expand_dims(jnp.sum(omega, 1), 1)
    else:
      raise NotImplementedError('%s not a valid parameterization.'
                                '' % jparameterization)

    return alpha, omega, vertex

  def jparams2patches(self, jparams, jparameterization=None, num_wedges=None,
                      patchmin=None, patchmax=None, patchsize=None, delta=None,
                      eta=None):
    """Render boundary and wedge patches."""

    jparameterization = self.jparameterization if (jparameterization is
                                                   None) else jparameterization
    num_wedges = self.num_wedges if num_wedges is None else num_wedges
    patchmin = self.patchmin if patchmin is None else patchmin
    patchmax = self.patchmax if patchmax is None else patchmax
    patchsize = self.patchsize if patchsize is None else patchsize
    delta = self.delta if delta is None else delta
    eta = self.eta if eta is None else eta

    alpha, omega, vertex = self.get_alpha_omega_vertex(jparams,
                                                       jparameterization,
                                                       num_wedges)

    return self.get_local_maps(alpha, omega, vertex, patchmin, patchmax,
                               patchsize, delta=delta, eta=eta)

  def get_local_maps(self, alpha, omega, vertex, patchmin=None, patchmax=None,
                     patchsize=None, delta=None, eta=None):
    """Render boundary and wedge patches."""

    patchmin = self.patchmin if patchmin is None else patchmin
    patchmax = self.patchmax if patchmax is None else patchmax
    patchsize = self.patchsize if patchsize is None else patchsize
    delta = self.delta if delta is None else delta
    eta = self.eta if eta is None else eta

    padding = [(0, 0)] + [(1, 0)] + [(0, 0)] + [(0, 0)]

    # Compute wedge central angles
    centralangles = jnp.expand_dims(alpha +
                                    omega/2 +
                                    jnp.pad(jnp.cumsum(omega,
                                                       axis=1)[:, :-1, ...],
                                            padding), (2, 3))

    # Compute wedge angles
    wedgeangles = jnp.expand_dims(omega*(2*jnp.pi)/
                                  jnp.expand_dims(jnp.sum(omega, 1), 1), (2, 3))

    # Compute wedge boundary angles
    boundaryangles = jnp.expand_dims(alpha +
                                     jnp.pad(jnp.cumsum(omega,
                                                        axis=1)[:,:-1, ...],
                                             padding), (2, 3))

    # Render and return boundary and feature patches
    feature_patches = self.render_wedges(vertex, centralangles, wedgeangles,
                                         patchmin, patchmax, patchsize, eta)
    distance_patches = self.render_distance(vertex, boundaryangles, patchmin,
                                            patchmax, patchsize)
    boundary_patches = self.render_boundaries(vertex, boundaryangles, patchmin,
                                              patchmax, patchsize, delta)

    return feature_patches, distance_patches, boundary_patches
