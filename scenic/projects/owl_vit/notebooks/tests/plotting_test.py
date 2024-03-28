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

"""Tests for plotting."""

from absl.testing import absltest
from bokeh import layouts as bk_layouts
import numpy as np
from scenic.projects.owl_vit.notebooks import plotting


class PlottingTest(absltest.TestCase):

  def test_create_text_conditional_figure(self):
    out = plotting.create_text_conditional_figure(
        image=np.zeros((128, 64, 3), dtype=np.uint8), boxes=np.zeros((5, 4)))
    self.assertIsInstance(out, bk_layouts.LayoutDOM)

  def test_create_image_conditional_figure(self):
    out = plotting.create_image_conditional_figure(
        query_image=np.zeros((128, 64, 3), dtype=np.uint8),
        target_image=np.zeros((128, 128, 3), dtype=np.uint8),
        target_boxes=np.zeros((5, 4)))
    self.assertIsInstance(out, bk_layouts.LayoutDOM)


if __name__ == '__main__':
  absltest.main()
