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

"""Initialization blocks: patch mixer."""
from typing import Optional, Any

import flax.linen as nn
from jax import lax
import jax.numpy as jnp
import ml_collections
from scenic.projects.boundary_attention.models.model_lib import misc_blocks
from scenic.projects.boundary_attention.models.model_lib import patch_mixer_blocks


class PatchInitializer(nn.Module):
  """Junction Predictor."""

  init_opts: ml_collections.ConfigDict
  train_opts: ml_collections.ConfigDict
  params2maps: Any
  hidden2outputs: Optional[nn.Module] = None

  @nn.compact
  def __call__(self,
               image: jnp.ndarray,
               extras: Optional[Any] = None,
               *,
               train: bool = True,
               debug: bool = False):

    if not(self.hidden2outputs):
      hidden2outputs_block = misc_blocks.Hidden2OutputsBlock(
          num_wedges=self.params2maps.num_wedges,
          parameterization=self.params2maps.jparameterization,
          params2maps=self.params2maps,
          name='Hidden2OutputsBlock',
      )
    else:
      hidden2outputs_block = self.hidden2outputs

    # Define Junction-Mixer
    junction_mixer = patch_mixer_blocks.PatchMixer(
        tokens_rf=self.init_opts.junction_mixer_rf,
        num_blocks=self.init_opts.num_junction_mixer_blocks,
        hidden_dim=self.init_opts.hidden_dim,
        tokens_conv_dim=self.init_opts.token_conv_dim,
        channels_conv_dim=self.init_opts.channels_conv_dim,
        padding=self.init_opts.junction_mixer_padding,
        stride=self.init_opts.stride,
    )

    init_junction_block = InitJunctionBlock(
        junction_mixer=junction_mixer,
        patchsize=self.params2maps.patchsize,
        hpatches=self.params2maps.hpatches,
        wpatches=self.params2maps.wpatches,
        crop_output=self.init_opts.get('crop_output', True),
    )

    if self.init_opts.get('normalize_input', True):
      im_min = jnp.min(image, axis=(-2, -1), keepdims=True)
      im_max = jnp.max(image, axis=(-2, -1), keepdims=True)
      image_norm = (image - im_min) / (im_max - im_min)
    else:
      image_norm = image

    # Next, pass through the local block to get initial junction parameter
    # estimates for each iteration of the patch mixer
    init_hidden_state = init_junction_block(image_norm)

    init_outputs = hidden2outputs_block(
        init_hidden_state, None, image, image, self.train_opts, train=train
    )

    return init_outputs


class InitJunctionBlock(nn.Module):
  """Finds initial hidden_state given input image."""

  junction_mixer: nn.Module
  patchsize: int
  hpatches: int
  wpatches: int
  crop_output: bool = True

  @nn.compact
  def __call__(self, input_img: jnp.ndarray):

    hidden_states = self.junction_mixer(input_img.transpose(0, 2, 3, 1))

    if self.crop_output:
      # Extract valid estimates (ones that aren't zero-padded)
      start = self.patchsize // 2
      hidden_states = lax.dynamic_slice(hidden_states, (0, start, start, 0),
                                        (hidden_states.shape[0],
                                         self.hpatches, self.wpatches,
                                         hidden_states.shape[-1]))

    return hidden_states
