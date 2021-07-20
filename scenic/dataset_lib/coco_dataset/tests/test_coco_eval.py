# Copyright 2021 The Scenic Authors.
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

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for functions in coco_eval.py."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from scenic.dataset_lib.coco_dataset import coco_eval

NUM_PANOPTIC_CLASSES = 132 + 1


class CocoPanopticEvalTest(parameterized.TestCase):
  """Test COCO panoptic evaluation."""

  def test_add_panoptic_annotation(self):
    """Test post_process_panoptic."""
    # Manually create a single example that has known results.
    # A single box, values taken from ground-truth annotations and manually
    # converted to relative [cx, cy, w, h] format:
    h, w = 480, 640
    num_queries = 6
    bx, by, bw, bh = 217.62, 240.54, 38.99, 57.75

    annotation = {}
    annotation['pred_boxes'] = np.array([
        [(bx + bw / 2) / w, (by + bh / 2) / h, bw / w, bh / h],
    ] * num_queries)
    annotation['pred_logits'] = np.zeros((num_queries, NUM_PANOPTIC_CLASSES))
    annotation['pred_logits'][0:num_queries//2, 1] = 100.0
    annotation['pred_masks'] = np.ones((num_queries, h // 4, w // 4))
    annotation['orig_size'] = np.array([h, w])
    annotation['image/id'] = 0

    evaluator = coco_eval.PanopticEvaluator()

    # Check that passing non-resized masks raises error:
    self.assertRaises(ValueError, evaluator.add_panoptic_annotation, annotation)

    # Try again with resized masks:
    annotation['pred_masks'] = np.ones((num_queries, h, w))
    evaluator.add_panoptic_annotation(annotation)

    self.assertSameElements(
        ['file_name', 'image_id', 'segments_info'],
        evaluator.pano_annotations[0].keys())

    self.assertSameElements(
        ['id', 'isthing', 'category_id', 'area'],
        evaluator.pano_annotations[0]['segments_info'][0].keys())


if __name__ == '__main__':
  absltest.main()
