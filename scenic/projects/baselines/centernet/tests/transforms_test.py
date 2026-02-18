# Copyright 2026 The Scenic Authors.
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

"""Unit tests for transforms.py."""

from absl.testing import absltest
from absl.testing import parameterized
from scenic.projects.baselines.centernet import transforms


class GetSizeWithAspectRatioTest(parameterized.TestCase):
  """Unit tests for get_size_with_aspect_ratio."""

  @parameterized.parameters(
      # One off.
      ((427, 640), 512, 512, False, (341, 511)),
      # Portrait (h > w), small size.
      ((800, 600), 400, None, False, (533, 400)),
      # Landscape (w > h), small size.
      ((600, 800), 400, None, False, (400, 533)),
      # Square, small size.
      ((500, 500), 250, None, False, (250, 250)),
      # Portrait, max_size constraint hit.
      ((1000, 500), 600, 800, False, (800, 400)),
      # Landscape, max_size constraint hit.
      ((500, 1000), 600, 800, False, (400, 800)),
      # size == max_size.
      ((1000, 500), 800, 800, False, (800, 400)),
      # size < max_size, max_size not hit.
      ((1000, 500), 400, 1000, False, (800, 400)),
      # size < max_size, max_size hit.
      ((2000, 1000), 800, 1200, False, (1200, 600)),
      # size smaller than original size.
      ((1000, 1000), 500, None, False, (500, 500)),
      # size larger than original size.
      ((100, 100), 200, None, False, (200, 200)),
      # One off, now correct rounding (note that h is now 512, not 511).
      ((427, 640), 512, 512, True, (341, 512)),
      # Portrait (h > w), small size.
      ((800, 600), 400, None, True, (533, 400)),
      # Landscape (w > h), small size.
      ((600, 800), 400, None, True, (400, 533)),
      # Square, small size.
      ((500, 500), 250, None, True, (250, 250)),
      # Portrait, max_size constraint hit.
      ((1000, 500), 600, 800, True, (800, 400)),
      # Landscape, max_size constraint hit.
      ((500, 1000), 600, 800, True, (400, 800)),
      # size == max_size.
      ((1000, 500), 800, 800, True, (800, 400)),
      # size < max_size, max_size not hit.
      ((1000, 500), 400, 1000, True, (800, 400)),
      # size < max_size, max_size hit.
      ((2000, 1000), 800, 1200, True, (1200, 600)),
      # size smaller than original size.
      ((1000, 1000), 500, None, True, (500, 500)),
      # size larger than original size.
      ((100, 100), 200, None, True, (200, 200)),
  )
  def test_get_size_with_aspect_ratio(
      self, image_size, size, max_size, correct_rounding_max_size, expected_size
  ):
    """Checks resizing logic with aspect ratio preservation."""
    actual_size = transforms.get_size_with_aspect_ratio(
        image_size, size, max_size, correct_rounding_max_size
    )
    self.assertEqual(actual_size, expected_size)


if __name__ == '__main__':
  absltest.main()
