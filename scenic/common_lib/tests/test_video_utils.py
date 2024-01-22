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

"""Unit tests for functions in video_utils.py."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from scenic.common_lib import video_utils


class VideoUtilsTest(parameterized.TestCase):
  """Test video utils."""

  @parameterized.named_parameters([
      ('n_sampled_frames_32', 32, (32, 32, 224, 224, 3)),
      ('n_sampled_frames_18', 18, (32, 19, 224, 224, 3)),
  ])
  def test_sample_frames_uniformly(self, n_sampled_frames, output_shape):
    """Test frame sampling."""
    input_shape = (32, 128, 224, 224, 3)
    self.assertSequenceEqual(
        output_shape,
        video_utils.sample_frames_uniformly(
            np.zeros(input_shape), n_sampled_frames).shape)


if __name__ == '__main__':
  absltest.main()
