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

"""Unit tests for functions in image_utils.py."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from scenic.common_lib import image_utils


class ResizePilTest(parameterized.TestCase):
  """Test resize_pil."""

  @parameterized.parameters([
      # pylint: disable=line-too-long
      # No batch dims:
      {'input_shape': [4, 5], 'num_batch_dims': 0, 'h': 2, 'w': 3, 'output_shape': [2, 3]},
      {'input_shape': [4, 5, 3], 'num_batch_dims': 0, 'h': 2, 'w': 3, 'output_shape': [2, 3, 3]},
      # One batch dim:
      {'input_shape': [7, 4, 5], 'num_batch_dims': 1, 'h': 2, 'w': 3, 'output_shape': [7, 2, 3]},
      {'input_shape': [7, 4, 5, 3], 'num_batch_dims': 1, 'h': 2, 'w': 3, 'output_shape': [7, 2, 3, 3]},
      # Two batch dims:
      {'input_shape': [6, 7, 4, 5], 'num_batch_dims': 2, 'h': 2, 'w': 3, 'output_shape': [6, 7, 2, 3]},
      {'input_shape': [6, 7, 4, 5, 3], 'num_batch_dims': 2, 'h': 2, 'w': 3, 'output_shape': [6, 7, 2, 3, 3]},
      # pylint: enable=line-too-long
  ])
  def test_resize_pil(self, input_shape, num_batch_dims, h, w, output_shape):
    """Test image resizing."""
    self.assertSequenceEqual(
        output_shape,
        image_utils.resize_pil(
            np.zeros(input_shape, dtype=np.uint8),
            num_batch_dims=num_batch_dims,
            out_h=h,
            out_w=w).shape)


if __name__ == '__main__':
  absltest.main()
