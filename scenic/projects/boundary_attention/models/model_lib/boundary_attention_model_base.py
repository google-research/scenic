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

"""Base class for boundary attention models."""

from flax import linen as nn
import jax.numpy as jnp
import ml_collections
from scenic.projects.boundary_attention.helpers import params2maps
from scenic.projects.boundary_attention.models.model_lib import deformable_refinement_blocks
from scenic.projects.boundary_attention.models.model_lib import initialization_blocks
from scenic.projects.boundary_attention.models.model_lib import misc_blocks
from scenic.projects.boundary_attention.models.model_lib import refinement_blocks


class BoundaryAttentionModelBase(nn.Module):
  """Base class for boundary attention models."""

  config: ml_collections.ConfigDict
  params2maps: params2maps.Params2Maps

  def setup(self):

    if self.config.model_name == 'boundary_attention':
      self.initialization = initialization_blocks.PatchInitializer(
          init_opts=self.config.model.model_opts.init_opts,
          train_opts=self.config.model.train_opts,
          params2maps=self.params2maps,
          name='PatchInitializer',
      )
      refinement_block_1 = refinement_blocks.BaseRefinementBlock(
          refine_opts=self.config.model.model_opts.refine_opts,
          train_opts=self.config.model.train_opts,
          params2maps=self.params2maps,
          name='PatchRefinement_0',
      )
      refinement_block_2 = refinement_blocks.BaseRefinementBlock(
          refine_opts=self.config.model.model_opts.refine_opts,
          train_opts=self.config.model.train_opts,
          params2maps=self.params2maps,
          name='PatchRefinement_1',
      )
      self.refinement = refinement_blocks.BaseRefinement(
          [refinement_block_1, refinement_block_2]
      )

    elif self.config.model_name == 'deformable_boundary_attention':
      self.hidden2outputs = misc_blocks.Hidden2OutputsBlock(
          num_wedges=self.params2maps.num_wedges,
          parameterization=self.params2maps.jparameterization,
          params2maps=self.params2maps,
          name='Hidden2OutputsBlock',
      )
      self.initialization = initialization_blocks.PatchInitializer(
          init_opts=self.config.model.model_opts.init_opts,
          train_opts=self.config.model.train_opts,
          params2maps=self.params2maps,
          name='PatchInitializer',
      )
      refinement_block = (
          deformable_refinement_blocks.DeformableRefinementBlock(
              refine_opts=self.config.model.model_opts.refine_opts,
              train_opts=self.config.model.train_opts,
              params2maps=self.params2maps,
              hidden2outputs=self.hidden2outputs,
              name='PatchRefinement_1',
          )
      )
      self.refinement = deformable_refinement_blocks.DeformableRefinement(
          [refinement_block]
      )

    elif self.config.model_name == 'deformable_boundary_attention_v0':
      self.hidden2outputs = misc_blocks.Hidden2OutputsBlock(
          num_wedges=self.params2maps.num_wedges,
          parameterization=self.params2maps.jparameterization,
          params2maps=self.params2maps,
          name='Hidden2OutputsBlock',
      )
      self.initialization = initialization_blocks.PatchInitializer(
          init_opts=self.config.model.model_opts.init_opts,
          train_opts=self.config.model.train_opts,
          params2maps=self.params2maps,
          name='PatchInitializer',
      )
      refinement_block_1 = refinement_blocks.BaseRefinementBlock(
          refine_opts=self.config.model.model_opts.refine_opts,
          train_opts=self.config.model.train_opts,
          params2maps=self.params2maps,
          hidden2outputs=self.hidden2outputs,
          use_deformable_attention=True,
          name='PatchRefinement_0',
      )
      refinement_block_2 = refinement_blocks.BaseRefinementBlock(
          refine_opts=self.config.model.model_opts.refine_opts,
          train_opts=self.config.model.train_opts,
          params2maps=self.params2maps,
          hidden2outputs=self.hidden2outputs,
          use_deformable_attention=True,
          name='PatchRefinement_1',
      )
      self.refinement = refinement_blocks.BaseRefinement(
          [refinement_block_1, refinement_block_2]
      )

    else:
      raise NameError('No valid boundary attention model found')

  def __call__(self,
               image: jnp.ndarray,
               *,
               train: bool = True,
               debug: bool = False):

    init_outputs = self.initialization(image, train=train, debug=debug)
    outputs = self.refinement(init_outputs, image, train=train, debug=debug)

    return outputs
