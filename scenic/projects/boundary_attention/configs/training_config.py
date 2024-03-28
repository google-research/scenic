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

"""Training config for boundary_attention."""

from scenic.projects.boundary_attention.configs import base_config


def get_config(args):
  return base_config.get_config(args)


def get_hyper(hyper):
  """Returns the hyperparameter sweep."""

  eta = [.001]

  hyper1 = hyper.sweep(
      'config.model.opts.eta',
      eta)

  hyper_val = hyper.chainit([hyper.product([hyper1])])

  return hyper_val
