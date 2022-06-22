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
