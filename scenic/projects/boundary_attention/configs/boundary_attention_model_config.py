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

"""Config for boundary attention model."""

import ml_collections


def get_boundary_attention_model_config():
  """Returns config for boundary attention model."""

  model_config = ml_collections.ConfigDict()
  model_config.name = 'boundary_attention'
  model_config.learning_rate = .000005

  # General model options
  model_config.opts = ml_collections.ConfigDict()
  model_config.opts.num_wedges = 3
  model_config.opts.patchsize = 17
  model_config.opts.patchmin = -1
  model_config.opts.patchmax = 1
  model_config.opts.stride = 1
  model_config.opts.jparameterization = 'standard'
  model_config.opts.bparameterization = 'standard'

  model_config.opts.delta = 0.001
  model_config.opts.eta = 0.0001
  model_config.opts.mask_shape = 'square'
  model_config.opts.patch_scales = [3, 9, 17]

  # Model architecture choices
  model_config.model_opts = ml_collections.ConfigDict()
  model_config.model_opts.input_feature_dim = 3
  model_config.model_opts.hidden_dim = 64

  # Initialization options
  model_config.model_opts.init_opts = ml_collections.ConfigDict()
  model_config.model_opts.init_opts.normalize_input = True
  model_config.model_opts.init_opts.hidden_dim = (
      model_config.model_opts.hidden_dim
  )
  model_config.model_opts.init_opts.token_conv_dim = 96
  model_config.model_opts.init_opts.channels_conv_dim = 64
  model_config.model_opts.init_opts.junction_mixer_rf = 3
  model_config.model_opts.init_opts.num_junction_mixer_blocks = 2
  model_config.model_opts.init_opts.junction_mixer_padding = 'SAME'
  model_config.model_opts.init_opts.stride = 1

  # Refinement options
  model_config.model_opts.refine_opts = ml_collections.ConfigDict()
  model_config.model_opts.refine_opts.hidden_dim = (
      model_config.model_opts.hidden_dim
  )
  model_config.model_opts.refine_opts.niters = 4
  model_config.model_opts.refine_opts.add_linear_residual = True
  model_config.model_opts.refine_opts.use_transformer = True
  model_config.model_opts.refine_opts.encoding_dim = 128
  model_config.model_opts.refine_opts.num_transformer_layers = 2
  model_config.model_opts.refine_opts.attention_patch_size = 11
  model_config.model_opts.refine_opts.num_attention_heads = 4
  model_config.model_opts.refine_opts.attn_dropout_prob = 0.1
  model_config.model_opts.refine_opts.ps_token_dim = 8
  model_config.model_opts.refine_opts.estimate_distribution = True
  model_config.model_opts.refine_opts.reuse_token = False

  # Training options
  model_config.train_opts = ml_collections.ConfigDict()
  model_config.train_opts.lmbda_wedge_mixing = 0
  model_config.train_opts.delta = model_config.opts.delta
  model_config.train_opts.eta = model_config.opts.eta

  # Loss options
  model_config.loss_opts = ml_collections.ConfigDict()

  model_config.loss_opts.beta = 0.1
  model_config.loss_opts.loss_constant = 0.3

  model_config.loss_opts.beta_PDS = 1e-3  # Patch distance supervision loss
  model_config.loss_opts.beta_PFS = 1.0  # Patch clean feature supervision loss
  model_config.loss_opts.beta_GDS = 1e-3  # Global distance supervision loss
  model_config.loss_opts.beta_GFS = 1.0  # Global clean feature supervision loss
  model_config.loss_opts.beta_BC = 1e-2  # Boundary consistency loss
  model_config.loss_opts.beta_FC = 20.0  # Feature consistency loss

  return model_config
