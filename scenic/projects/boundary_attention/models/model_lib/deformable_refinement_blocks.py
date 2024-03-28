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

"""Deformable Refinement Network."""

from collections.abc import Sequence
from typing import Any
from typing import Optional
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from scenic.projects.boundary_attention.models.model_lib import deformable_attention_blocks
from scenic.projects.boundary_attention.models.model_lib import misc_blocks


def pytrees_stack(pytrees, axis=0):
  """Stacks a list of pytrees along a given axis."""
  results = jax.tree_util.tree_map(
      lambda *values: jnp.stack(values, axis=axis), *pytrees
  )
  return results


class DeformableRefinement(nn.Module):
  """Base Refinement Network."""

  refinement_blocks: Sequence[nn.Module]

  @nn.compact
  def __call__(self,
               init_hidden_state: jnp.ndarray,
               image: jnp.ndarray,
               *,
               train: bool = True,
               debug: bool = False):

    # Initialize inputs
    hidden_state = init_hidden_state.copy()
    patchsize_tokens = None
    offset_tokens = None
    global_features = image

    all_outputs = []
    # Do refinement
    for block in self.refinement_blocks:
      outputs = block(hidden_state,
                      init_hidden_state,
                      patchsize_tokens,
                      offset_tokens,
                      global_features,
                      image,
                      train=train,
                      debug=debug)

      # Update inputs after each block
      hidden_state = outputs[-1]['hidden_state']
      global_features = outputs[-1]['global_features']
      patchsize_tokens = outputs[-1]['patchsize_tokens']

      if outputs[-1]['offset_tokens'] is not None:
        offset_tokens = outputs[-1]['offset_tokens']

      # Save outputs
      all_outputs.extend(outputs)

    return all_outputs


class DeformableRefinementBlock(nn.Module):
  """Boundary Attention Block."""

  refine_opts: ml_collections.ConfigDict
  train_opts: ml_collections.ConfigDict
  params2maps: Any
  hidden2outputs: Optional[nn.Module] = None

  def setup(self):

    self.niters = self.refine_opts.niters

    if self.hidden2outputs is None:
      self.hidden2outputs = misc_blocks.Hidden2OutputsBlock(
          num_wedges=self.params2maps.num_wedges,
          parameterization=self.params2maps.jparameterization,
          params2maps=self.params2maps,
          name='Hidden2OutputsBlock')

    self.attention_block = (
        deformable_attention_blocks.DeformableTransformerBlock(
            deformation_type=self.refine_opts.get('deformation_type', 'simple'),
            offset_fn=self.refine_opts.get('offset_fn', 'dense'),
            max_offset=self.refine_opts.get('max_offset', 30),
            num_samples=self.refine_opts.get('num_samples', 16),
            M=self.params2maps.num_wedges,
            C=self.params2maps.C,
            patchsize=self.refine_opts.attention_patch_size,
            encoding_dim=self.refine_opts.encoding_dim,
            num_heads=self.refine_opts.num_attention_heads,
            num_layers=self.refine_opts.num_transformer_layers,
            attn_dropout_prob=self.refine_opts.attn_dropout_prob,
            name='TransformerBlock'))

    self.residual_block = misc_blocks.ResidualBlock(
        hidden_dim=self.refine_opts.hidden_dim,
        name='ResidualBlock')

    self.ps_token = self.param('PatchsizeToken',
                               nn.initializers.lecun_normal(),
                               (1, 1, 1, self.refine_opts.ps_token_dim))
    self.est_maxps = nn.Dense(
        3,
        kernel_init=nn.initializers.constant(1),
        name='EstMaxPatchsize')

    if self.refine_opts.get('deformation_type', 'simple') == 'token':
      self.offset_token = self.param(
          'OffsetToken',
          nn.initializers.lecun_normal(),
          (1, 1, 1, self.refine_opts.get('offset_token_dim', 8)))

  def __call__(self,
               hidden_state: jnp.ndarray,
               init_hidden_state: jnp.ndarray,
               patchsize_tokens: Any,
               offset_tokens: Any,
               global_features: jnp.ndarray,
               input_image: jnp.ndarray,
               *,
               train: bool = True,
               debug: bool = False):

    refine_outputs = []

    if patchsize_tokens is None:
      patchsize_tokens = jnp.tile(
          self.ps_token,
          (hidden_state.shape[0], self.params2maps.H_patches,
           self.params2maps.W_patches, 1))
    if (offset_tokens is None) and (
        self.refine_opts.get('deformation_type', 'simple') == 'token'):
      offset_tokens = jnp.tile(
          self.offset_token,
          (hidden_state.shape[0], self.params2maps.H_patches,
           self.params2maps.W_patches, 1))

    for _ in range(self.niters):

      # Embed initial hidden state
      hidden_state = self.residual_block(hidden_state, init_hidden_state)

      # Add a patchsize token to the hidden state
      full_hidden_state = jnp.concatenate((hidden_state, patchsize_tokens), -1)

      if self.refine_opts.get('deformation_type', 'simple') == 'token':
        full_hidden_state = jnp.concatenate((full_hidden_state, offset_tokens),
                                            -1)

      # Do cross attention
      output_hidden_state_with_ps_token = self.attention_block(
          full_hidden_state, full_hidden_state, train=train)

      # Separate updated hidden state with patch size token
      hidden_dim = hidden_state.shape[-1]
      hidden_state = output_hidden_state_with_ps_token[:, :, :, :hidden_dim]
      ps_token = output_hidden_state_with_ps_token[:, :, :, hidden_dim:]

      if self.refine_opts.get('deformation_type', 'simple') == 'token':
        ps_token_dim = self.refine_opts.ps_token_dim
        ps_token = ps_token[:ps_token_dim]
        offset_tokens = ps_token[ps_token_dim:]
        # to try: try stopping the gradients of the patchsize and offset tokens

      # Estimate patchsize distribution
      patchsize_distribution = nn.softmax(self.est_maxps(ps_token), axis=-1)

      # Gather and save outputs
      outputs = self.hidden2outputs(hidden_state, patchsize_distribution,
                                    input_image, global_features,
                                    self.train_opts, train=train)
      outputs['patchsize_tokens'] = patchsize_tokens
      outputs['offset_tokens'] = offset_tokens

      refine_outputs.append(outputs)

    return refine_outputs
