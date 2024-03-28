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

"""Runs the field of junctions optimization with verbose logging."""

import types
from scenic.projects.boundary_attention.field_of_junctions_jax import field_of_junctions


def get_opts(patchsize=21, lambda_boundary_final=0.5, lambda_color_final=0.1):
  """Returns the optimization options for the field of junctions model."""
  opts = types.SimpleNamespace()

  opts.patchsize = patchsize
  opts.stride = 1
  opts.eta = 0.01
  opts.delta = 0.05
  opts.lr_angles = 0.003
  opts.lr_x0y0 = 0.03
  opts.lambda_boundary_final = lambda_boundary_final
  opts.lambda_color_final = lambda_color_final
  opts.nvals = 31
  opts.num_initialization_iters = 30
  opts.num_refinement_iters = 1000
  opts.greedy_step_every_iters = 50
  opts.parallel_mode = True

  return opts


def foj_optimize_verbose(img, opts):
  """Runs the field of junctions optimization with verbose logging."""
  foj = field_of_junctions.FieldOfJunctions(img, opts)
  angles = foj.angles
  x0y0 = foj.x0y0
  global_image = foj.global_image
  global_boundaries = foj.global_boundaries
  opt_states = foj.opt_states

  for i in range(foj.num_iters):
    if i == 0:
      print("Beginning initialization...")
    if i == opts.num_initialization_iters:
      print("Initialization done. Beginning refinement...")
    if i < opts.num_initialization_iters:
      if i % 5 == 0:
        print(f"Initialization iteration {i}/{opts.num_initialization_iters}")
    else:
      if i % 100 == 0:
        print(f"Refinement iteration {i}/{opts.num_refinement_iters}")

    angles, x0y0, global_image, global_boundaries, opt_states = foj.step(
        i, angles, x0y0, global_image, global_boundaries, opt_states
    )

  return angles, x0y0, global_image, global_boundaries
