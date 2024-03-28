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

"""Refinement networks."""

from collections.abc import Sequence
from typing import Any, Dict, Optional

import einops
import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections
from scenic.projects.boundary_attention.models.model_lib import attention_blocks
from scenic.projects.boundary_attention.models.model_lib import deformable_attention_blocks
from scenic.projects.boundary_attention.models.model_lib import misc_blocks


def pytrees_stack(pytrees, axis=0):
  results = jax.tree_util.tree_map(
      lambda *values: jnp.stack(values, axis=axis), *pytrees
  )
  return results


def pytrees_cat(pytrees, axis=0):
  results = jax.tree_util.tree_map(
      lambda *values: jnp.concatenate(values, axis=axis), *pytrees
  )
  return results


def pytrees_list(pytrees):
  results = jax.tree_util.tree_map(lambda *values: list(values), *pytrees)
  return results


class BaseRefinement(nn.Module):
  """Base Refinement Network.

  This is the base class for all refinement networks.
  """

  refinement_blocks: Sequence[nn.Module]

  @nn.compact
  def __call__(
      self,
      init_outputs: Dict[str, jnp.ndarray],
      image: jnp.ndarray,
      *,
      train: bool = True,
      debug: bool = False
  ):
    init_hidden_state = init_outputs['hidden_state'].copy()
    hidden_state = init_outputs['hidden_state']
    global_features = init_outputs['global_features']
    all_outputs = [init_outputs]
    patchsize_tokens = None

    for block in self.refinement_blocks:
      outputs = block(hidden_state,
                      init_hidden_state,
                      patchsize_tokens,
                      global_features,
                      image,
                      train=train,
                      debug=debug)
      all_outputs.extend(outputs)

      init_hidden_state = outputs[0]['hidden_state']
      hidden_state = outputs[-1]['hidden_state']
      global_features = outputs[-1]['global_features']
      patchsize_tokens = outputs[-1]['patchsize_tokens']

    return all_outputs


class BaseRefinementBlock(nn.Module):
  """Base Refinement Block.

  This is the base class for all refinement blocks.
  """

  refine_opts: ml_collections.ConfigDict
  train_opts: ml_collections.ConfigDict
  params2maps: Any
  hidden2outputs: Optional[nn.Module] = None
  use_deformable_attention: Optional[bool] = False

  def setup(self):

    self.niters = self.refine_opts.niters

    if self.hidden2outputs is None:
      self.hidden2outputs_block = misc_blocks.Hidden2OutputsBlock(
          num_wedges=self.params2maps.num_wedges,
          parameterization=self.params2maps.jparameterization,
          params2maps=self.params2maps,
          name='Hidden2OutputsBlock')
    else:
      self.hidden2outputs_block = self.hidden2outputs

    if self.use_deformable_attention:
      self.attention_block = (
          deformable_attention_blocks.DeformableTransformerBlock(
              deformation_type=self.refine_opts.get(
                  'deformation_type', 'simple'
              ),
              offset_fn=self.refine_opts.get('offset_fn', 'dense'),
              max_offset=self.refine_opts.get('max_offset', 30),
              num_samples=self.refine_opts.get('num_samples', 16),
              M=self.params2maps.num_wedges,
              C=self.params2maps.channels,
              patchsize=self.refine_opts.attention_patch_size,
              encoding_dim=self.refine_opts.encoding_dim,
              num_heads=self.refine_opts.num_attention_heads,
              num_layers=self.refine_opts.num_transformer_layers,
              attn_dropout_prob=self.refine_opts.attn_dropout_prob,
              name='TransformerBlock',
          )
      )
    else:
      self.attention_block = attention_blocks.TransformerBlock(
          M=self.params2maps.num_wedges,
          C=self.params2maps.channels,
          patchsize=self.refine_opts.attention_patch_size,
          encoding_dim=self.refine_opts.encoding_dim,
          num_heads=self.refine_opts.num_attention_heads,
          num_layers=self.refine_opts.num_transformer_layers,
          attn_dropout_prob=self.refine_opts.attn_dropout_prob,
          name='TransformerBlock',
      )

    self.residual_block = misc_blocks.ResidualBlock(
        hidden_dim=self.refine_opts.hidden_dim,
        name='ResidualBlock')
    self.ps_token = self.param('PatchsizeToken',
                               nn.initializers.lecun_normal(),
                               (1, 1, 1, self.refine_opts.ps_token_dim))
    self.est_maxps = nn.Dense(3, kernel_init=nn.initializers.constant(1),
                              name='EstMaxPatchsize')

  def __call__(self,
               hidden_state: jnp.ndarray,
               init_hidden_state: jnp.ndarray,
               patchsize_tokens: Any,
               global_features: jnp.ndarray,
               input_image: jnp.ndarray,
               *,
               train: bool = True,
               debug: bool = False):

    # Crop the input image
    start = self.params2maps.patchsize // 2
    if self.params2maps.stride == 1:
      input_image_valid = lax.dynamic_slice(
          input_image,
          (0, 0, start, start),
          (
              input_image.shape[0],
              input_image.shape[1],
              self.params2maps.hpatches,
              self.params2maps.wpatches,
          ),
      )
    else:
      sliced_input = jax.lax.dynamic_slice(
          input_image,
          (0, 0, start, start),
          (
              input_image.shape[0],
              input_image.shape[1],
              self.params2maps.stride * self.params2maps.hpatches,
              self.params2maps.stride * self.params2maps.wpatches,
          ),
      )
      input_image_valid = jax.lax.slice(sliced_input, (0, 0, 0, 0),
                                        (sliced_input.shape[0],
                                         sliced_input.shape[1],
                                         sliced_input.shape[2],
                                         sliced_input.shape[3]),
                                        strides=(1, 1, self.params2maps.stride,
                                                 self.params2maps.stride))

    refine_outputs = []
    for _ in range(self.niters):
      # Embed initial hidden state
      hidden_state = self.residual_block(hidden_state, init_hidden_state)

      if (patchsize_tokens is None) or (not self.refine_opts.reuse_token):
        ps_token = jnp.tile(
            self.ps_token,
            (
                hidden_state.shape[0],
                self.params2maps.hpatches,
                self.params2maps.wpatches,
                1,
            ),
        )
      else:
        ps_token = patchsize_tokens

      # Add a patchsize token to the hidden state
      hidden_state_with_ps_token = jnp.concatenate((hidden_state, ps_token), -1)

      # First, crop the global_features:
      if self.params2maps.stride == 1:
        global_features_valid = lax.dynamic_slice(
            global_features,
            (0, 0, start, start),
            (
                global_features.shape[0],
                global_features.shape[1],
                self.params2maps.hpatches,
                self.params2maps.wpatches,
            ),
        )
      else:
        sliced_features = jax.lax.dynamic_slice(
            global_features,
            (0, 0, start, start),
            (
                global_features.shape[0],
                global_features.shape[1],
                self.params2maps.stride * self.params2maps.hpatches,
                self.params2maps.stride * self.params2maps.wpatches,
            ),
        )
        global_features_valid = jax.lax.slice(
            sliced_features, (0, 0, 0, 0),
            (sliced_features.shape[0], sliced_features.shape[1],
             sliced_features.shape[2], sliced_features.shape[3]),
            strides=(
                1, 1, self.params2maps.hpatches, self.params2maps.wpatches))

      hidden_state_proposal = jnp.concatenate(
          (hidden_state,
           einops.rearrange(global_features_valid, 'b f h w -> b h w f'),
           einops.rearrange(
               global_features_valid[:, 0:1, :, :], 'b f h w -> b h w f'),
           einops.rearrange(input_image_valid, 'b f h w -> b h w f'),
          ),
          -1,
      )

      # Do cross attention
      output_hidden_state_with_ps_token = self.attention_block(
          hidden_state_with_ps_token, hidden_state_proposal, train=train)

      # Separate updated hidden state with patch size token
      hidden_dim = hidden_state.shape[-1]
      hidden_state = output_hidden_state_with_ps_token[:, :, :, :hidden_dim]
      ps_token = output_hidden_state_with_ps_token[:, :, :, hidden_dim:]
      patchsize_distribution = nn.softmax(self.est_maxps(ps_token), axis=-1)

      outputs = self.hidden2outputs_block(hidden_state, patchsize_distribution,
                                          input_image, global_features,
                                          self.train_opts, train=train)
      outputs['patchsize_tokens'] = patchsize_tokens

      # Update global features
      global_features = outputs['global_features']

      refine_outputs.append(outputs)

    return refine_outputs
