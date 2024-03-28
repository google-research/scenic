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

"""Loss Functions for Boundary Attention."""

import jax
import jax.numpy as jnp
import ml_collections
from scenic.projects.boundary_attention.helpers import params2maps as params2maps_lib


def pytrees_stack(pytrees, axis=0):
  results = jax.tree_util.tree_map(
      lambda *values: jnp.stack(values, axis=axis), *pytrees
  )
  return results


class BoundaryAttentionLoss:
  """Loss functions for Boundary Attention."""

  def __init__(
      self,
      config: ml_collections.ConfigDict,
      params2maps: params2maps_lib.Params2Maps,
  ):
    self.config = config
    self.params2maps = params2maps
    self.loss_opts = self.get_loss_opts()

  def get_loss_opts(self):
    return self.config.model.loss_opts

  def get_patch_supervision_loss(self, local_est, global_truth):
    """Calculates the loss for patch supervision."""
    local_boundaries_gt = self.params2maps.unfold(global_truth)
    loss = jnp.sum((local_est - local_boundaries_gt)**2, axis=1, keepdims=True)
    return loss

  def get_global_supervision_loss(self, global_est, global_gt):
    """Calculates the loss for global supervision."""
    return jnp.sum((global_est - global_gt)**2, axis=1, keepdims=True)

  def get_patch_consistency_loss(self, local_est, global_est):
    """Calculates the loss for patch consistency."""
    global_patches_est = jax.lax.stop_gradient(
        self.params2maps.unfold(global_est)
    )
    return jnp.sum((local_est - global_patches_est)**2, axis=1, keepdims=True)

  def get_per_pixel_importance(
      self,
      gt_distances,
      patch_density=1,
      iv_masks=1,
      beta=0.1,
      delta=1.0,
      const=1.0,
  ):
    """Calculates the per-pixel importance function."""

    global_scale_mask = patch_density > 0
    return (
        jnp.exp(-beta * (gt_distances + delta))
        * (iv_masks + const)
        * global_scale_mask
    )

  def get_per_patch_importance(
      self, gt_distances, patch_masks, pixel_importance, delta=1.0
  ):
    """Calculates the per-patch importance function."""

    normalized_patch_masks = patch_masks / (
        jnp.sum(patch_masks, axis=(1, 2, 3), keepdims=True) + 1e-4
    )
    distance_weight = 1 / jnp.sum(
        self.params2maps.unfold(gt_distances) + delta,
        axis=(2, 3),
        keepdims=True,
    )
    patched_pixel_importance = self.params2maps.unfold(pixel_importance)
    return distance_weight * normalized_patch_masks * patched_pixel_importance

  def get_loss(self, outputs, inputs):
    """Calculates the loss function."""

    num_losses = 2
    weights = (0.3) ** (num_losses - jnp.arange(num_losses))
    weights = weights/jnp.sum(weights)

    output_loss = jnp.expand_dims(weights,
                                  (1, 2, 3, 4)) * jax.vmap(
                                      self.get_layer_loss,
                                      in_axes=(0, None))(
                                          pytrees_stack(outputs[-2:]), inputs)

    return output_loss

  def get_layer_loss(self, outputs, inputs):
    """Calculates the loss function for a single layer."""

    global_distances = outputs['global_distances']
    global_features = outputs['global_features']
    global_boundaries = outputs['global_boundaries']
    distance_patches = outputs['distance_patches']
    boundary_patches = outputs['boundary_patches']
    feature_patches = outputs['feature_patches']
    patch_density = outputs['patch_density']
    patch_masks = outputs['patch_masks']

    pixel_importance = self.get_per_pixel_importance(
        inputs['distances'],
        patch_density,
        inputs['iv_mask'].transpose(0, 3, 1, 2),
        beta=self.loss_opts.beta,
        delta=1.0,
        const=self.loss_opts.loss_constant,
    )
    patch_importance = self.get_per_patch_importance(
        inputs['distances'],
        patch_masks,
        pixel_importance,
        delta=self.loss_opts.loss_constant,
    )

    # --- Global Supervision Losses
    global_distance_supervision_loss = (
        self.get_global_supervision_loss(global_distances, inputs['distances'])
        * pixel_importance
    )
    global_feature_supervision_loss = (
        self.get_global_supervision_loss(global_features, inputs['clean_image'])
        * pixel_importance
    )

    # ---- Patchwise Supervision Losses
    patch_distance_supervision_loss = (
        self.get_patch_supervision_loss(distance_patches, inputs['distances'])
        * patch_importance
    )
    patch_feature_supervision_loss = (
        self.get_patch_supervision_loss(feature_patches, inputs['clean_image'])
        * patch_importance
    )

    # ---- Patchwise consistency Losses
    patch_boundary_consistency_loss = (
        self.get_patch_consistency_loss(boundary_patches, global_boundaries)
        * patch_importance
    )
    patch_feature_consistency_loss = (
        self.get_patch_consistency_loss(feature_patches, global_features)
        * patch_importance
    )

    # Fold patched losses so that output shapes are consistent
    folded_shape = global_distances.shape

    patch_distance_supervision_loss = self.params2maps.fold(
        patch_distance_supervision_loss, folded_shape
    )
    patch_feature_supervision_loss = self.params2maps.fold(
        patch_feature_supervision_loss, folded_shape
    )
    patch_boundary_consistency_loss = self.params2maps.fold(
        patch_boundary_consistency_loss, folded_shape
    )
    patch_feature_consistency_loss = self.params2maps.fold(
        patch_feature_consistency_loss, folded_shape
    )

    return (
        self.loss_opts.beta_GDS * global_distance_supervision_loss +
        self.loss_opts.beta_GFS * global_feature_supervision_loss +
        self.loss_opts.beta_PDS * patch_distance_supervision_loss +
        self.loss_opts.beta_PFS * patch_feature_supervision_loss +
        self.loss_opts.beta_BC * patch_boundary_consistency_loss +
        self.loss_opts.beta_FC * patch_feature_consistency_loss
        )

  def standard_metric(self, outputs, inputs):
    """Define objective loss function."""

    # The simplest metric is the difference between the global ground truth
    # distance map and the predicted distance map
    standard_metric = self.get_global_supervision_loss(
        outputs[-1]['global_distances'], inputs['distances']
    )

    return {'standard_metric': standard_metric}
