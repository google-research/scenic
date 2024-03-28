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

"""Returns model input_opts given input shape."""

import ml_collections


def get_receptive_field(kernel_size, num_mixer_layers):
  receptive_field = 1 + (num_mixer_layers*2) * 1 * (kernel_size - 1)
  return receptive_field


def get_input_opts(input_shape, opts):
  """Returns input shape dependent model opts."""

  height, width, channels = input_shape

  input_opts = ml_collections.ConfigDict()
  input_opts.patchsize = opts.get('patchsize', 17)
  input_opts.height = height
  input_opts.width = width
  input_opts.channels = channels
  input_opts.hpatches = (height - input_opts.patchsize) // opts.stride + 1
  input_opts.wpatches = (width - input_opts.patchsize) // opts.stride + 1

  return input_opts
