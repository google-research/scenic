# Copyright 2022 The Scenic Authors.
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
  """Computes Hungarian Matching on cost matrix for a batch of datapoints.

  Uses a map over linear_sum_assignment, which computes the matching between
  predictions for a single datapoint.

  Predicted boxes that were not matched to any non-empty target box will be
  given a dummy matching to the last target box, which is assumed to be empty.
  Due to this padding of indices, the number of indices for all datapoints are
  all equal to num_queries. The cost matrix should be already padded to the
  correct shape with dummy targets.

  Relevant DETR code:
  https://github.com/facebookresearch/detr/blob/647917626d5017e63c1217b99537deb2dcb370d6/models/matcher.py#L35

  Args:
    cost: np.ndarray; Batch of matching cost matrices; should be on CPU
      already.

  Returns:
    An np.ndarray batch_size, containing indices (index_i, index_j) where:
        - index_i is the indices of the selected predictions (in order).
        - index_j is the indices of the corresponding selected targets (in
        order).
      Each index_i and index_j is an np.array of shape [num_queries].
  """
  return np.array(
      list(map(lambda x: tuple(sciopt.linear_sum_assignment(x)), cost)))
