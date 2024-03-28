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

"""Defines the models for the boundary attention project."""

import immutabledict
from scenic.projects.boundary_attention.models import boundary_attention


ALL_MODELS = immutabledict.immutabledict({
    'boundary_attention': boundary_attention.BoundaryAttention,
    })


def get_model_cls(model_name):
  """Returns the model class for the given model name."""

  if model_name not in ALL_MODELS.keys():
    raise NotImplementedError('Unrecognized model: {}'.format(model_name))
  return ALL_MODELS[model_name]
