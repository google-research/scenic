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

"""OmniNet models utilities."""

import jax.numpy as jnp


def grid_restack(all_vecs):
  """Stack layers with respect to the grid shape of positions.

  Given multiple sequences (lists) of batch x len x dim reshape this such
  that all positions are side by side.

  for example (for illustrative purposes):

  inputs: [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]
  outputs: [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

  Args:
    all_vecs: list of sequences of batch x len x dim

  Returns:
    Array of batch x (length x num_items) x dim.
  """
  cat_output = []
  for pos in range(all_vecs[0].shape[1]):
    pos_vecs = [x[:, None, pos, :] for x in all_vecs]
    cat_output += pos_vecs
  x2 = jnp.concatenate(cat_output, 1)
  return x2
