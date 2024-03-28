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

"""Maps Junction Parameters to Images."""
import jax.numpy as jnp
import ml_collections
from scenic.projects.boundary_attention.helpers import junction_functions


class Params2Maps(junction_functions.JunctionFunctions):
  """Maps Junction Parameters to Images."""

  def __call__(self,
               jparams: jnp.ndarray,
               patchsize_distribution: jnp.ndarray,
               img: jnp.ndarray,
               global_features: jnp.ndarray,
               train_opts: ml_collections.ConfigDict) -> dict[str, jnp.ndarray]:

    if patchsize_distribution is not None:
      scales = jnp.expand_dims(jnp.array(self.patch_scales), (0, 1, 2))
      all_patch_masks = jnp.expand_dims(
          self.get_scale_masks(scales, self.mask_shape), axis=(1, 4)
      )
      all_patch_masks = jnp.mean(
          all_patch_masks * jnp.expand_dims(patchsize_distribution, (1, 2, 3)),
          axis=-1,
      )

      patch_density = self.fold(
          all_patch_masks,
          [patchsize_distribution.shape[0], 1, self.height, self.width],
      )
    else:
      all_patch_masks = 1
      patch_density = self.patch_density

    local_wedges, local_distance_branches, local_boundary_branches = (
        self.jparams2patches(
            jparams, delta=train_opts.delta, eta=train_opts.eta
        )
    )

    # Apply patch masks
    masked_local_wedges = local_wedges*all_patch_masks
    masked_distance_patches = local_distance_branches * all_patch_masks
    masked_boundary_patches = local_boundary_branches * all_patch_masks

    # Find image patches, wedge_colors, and global_image
    feature_patches, wedge_colors = self.get_avg_wedge_feature(
        img,
        global_features,
        masked_local_wedges,
        lmbda_wedge_mixing=train_opts.lmbda_wedge_mixing
    )

    # Make global maps from patches
    all_patches = jnp.concatenate([feature_patches, masked_distance_patches,
                                   masked_boundary_patches], axis=1)
    global_maps = self.local2global(all_patches, patch_density)

    # Split into individual maps
    global_features, global_distances, global_boundaries = jnp.split(
        global_maps, [self.channels, self.channels + 1], axis=1
    )

    return dict(jparams=jparams,
                patchsize_distribution=patchsize_distribution,
                global_features=global_features,
                global_distances=global_distances,
                global_boundaries=global_boundaries,
                feature_patches=feature_patches,
                distance_patches=masked_distance_patches,
                boundary_patches=masked_boundary_patches,
                wedge_colors=wedge_colors,
                wedges=local_wedges,
                patch_masks=all_patch_masks,
                patch_density=patch_density)
