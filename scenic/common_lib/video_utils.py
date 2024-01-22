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

"""Video-related utility functions."""

import jax.numpy as jnp


def sample_frames_uniformly(x: jnp.ndarray,
                            n_sampled_frames: int) -> jnp.ndarray:
  """Sample frames from the input video."""
  if x.ndim != 5:
    raise ValueError('Input shape should be [bs, t, h, w, c].')
  num_frames = x.shape[1]
  if n_sampled_frames < num_frames:
    t_start_idx = num_frames / (n_sampled_frames + 1)
    t_step = t_start_idx
  else:
    t_start_idx = 0
    t_step = 1
  t_end_idx = num_frames
  temporal_indices = jnp.arange(t_start_idx, t_end_idx, t_step)
  temporal_indices = jnp.round(temporal_indices).astype(jnp.int32)
  temporal_indices = jnp.minimum(temporal_indices, num_frames - 1)
  return x[:, temporal_indices]  # [n, t_s, in_h, in_w, c]
