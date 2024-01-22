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

"""Hungarian matching based on Scipy."""

import numpy as np
from scenic.model_lib.matchers.common import cpu_matcher
import scipy.optimize as sciopt


@cpu_matcher
def hungarian_matcher(cost):
  """Computes Hungarian Matching given a single cost matrix.

  Relevant DETR code:
  https://github.com/facebookresearch/detr/blob/647917626d5017e63c1217b99537deb2dcb370d6/models/matcher.py#L35

  Args:
    cost: Matching cost matrix of shape [N, M].

  Returns:
    Array of shape [min(N, M), 2] where each row contains a matched pair of
    indices into the rows (N) and columns (M) of the cost matrix.
  """
  # Matrix is transposed to maintain the convention of other matchers:
  col_ind, row_ind = sciopt.linear_sum_assignment(cost.T)
  return np.stack([row_ind, col_ind]).astype(np.int32)
