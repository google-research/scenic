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

"""Unit tests for functions in train_utils.py."""

import collections
import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from scenic.projects.baselines.detr import train_utils as detr_train_utils

NUM_COCO_CLASSES = 81  # COCO (non-panoptic) uses 80 classes, + background.


def sample_cxcywh_bbox(key, batch_shape):
  """Samples a bounding box in the [cx, cy, w, h] in [0, 1] range format."""
  frac = 0.8
  sample = jax.random.uniform(key, shape=(*batch_shape, 4)) * frac
  cx, cy, w, h = jnp.split(sample, indices_or_sections=4, axis=-1)
  # Make sure the bounding box doesn't cross the right and top image borders
  w = jnp.where(cx + w / 2. >= 1., frac * 2. * (1. - cx), w)
  h = jnp.where(cy + h / 2. >= 1., frac * 2. * (1. - cy), h)
  # Make sure the bounding box doesn't cross the left and bottom image borders
  w = jnp.where(cx - w / 2. <= 0., frac * 2. * cx, w)
  h = jnp.where(cy - h / 2. <= 0., frac * 2. * cy, h)

  bbox = jnp.concatenate([cx, cy, w, h], axis=-1)
  return bbox


class TrainUtilsTest(parameterized.TestCase):
  """Test train utilities."""

  def setUp(self):
    """Setup sample output predictions and target labels and bounding boxes."""
    super().setUp()

    self.batchsize = 4
    self.num_classes = 81
    self.max_num_boxes = 63
    self.num_preds = 100

    key = jax.random.PRNGKey(0)

    # Create fake predictions and targets
    key, subkey = jax.random.split(key)
    # set probabilities for class 0 higher than others
    p_logits = jnp.ones(self.num_classes).at[0].set(5.)
    p = jax.nn.softmax(p_logits)
    tgt_labels = jax.random.choice(
        subkey,
        self.num_classes,
        shape=(self.batchsize, self.max_num_boxes),
        replace=True,
        p=p)
    # Ensure last target is dummy empty target.
    tgt_labels = tgt_labels.at[:, -1].set(0)
    onehot_tgt_labels = jax.nn.one_hot(tgt_labels, self.num_classes)

    key, subkey = jax.random.split(key)
    pred_logits = jax.random.normal(
        subkey, shape=(self.batchsize, self.num_preds, self.num_classes))
    pred_probs = jax.nn.softmax(pred_logits, axis=-1)

    key, subkey = jax.random.split(key)
    pred_bbox = sample_cxcywh_bbox(
        subkey, batch_shape=(self.batchsize, self.num_preds))

    key, subkey = jax.random.split(key)
    tgt_bbox = sample_cxcywh_bbox(
        subkey, batch_shape=(self.batchsize, self.max_num_boxes))

    self.outputs = {'pred_probs': pred_probs, 'pred_boxes': pred_bbox}
    self.targets = {'labels': tgt_labels, 'boxes': tgt_bbox}
    self.onehot_targets = {'labels': onehot_tgt_labels, 'boxes': tgt_bbox}

  def test_coco_global_val_metrics_function_results(self):
    """Test coco_global_metrics_function correctness on a single box."""
    data_dir = os.path.join(
        os.path.normpath(os.path.dirname(__file__) + '/../../../../'),
        'dataset_lib', 'coco_dataset', 'data')
    test_annotations_path = os.path.join(
        data_dir, 'instances_val2017_unittest.json')
    outputs, targets = _get_fake_detection_example()

    global_metrics_evaluator = detr_train_utils.DetrGlobalEvaluator(
        'coco_detr_detection', annotations_loc=test_annotations_path)
    global_metrics_evaluator.add_example(prediction=outputs, target=targets)
    metrics = global_metrics_evaluator.compute_metrics()

    self.assertAlmostEqual(metrics['AP'], 1.0, 5)
    self.assertAlmostEqual(metrics['AP_50'], 1.0, 5)
    self.assertAlmostEqual(metrics['AP_75'], 1.0, 5)
    self.assertAlmostEqual(metrics['AP_small'], -1.0, 5)
    self.assertAlmostEqual(metrics['AP_medium'], 1.0, 5)
    self.assertAlmostEqual(metrics['AP_large'], -1.0, 5)
    self.assertAlmostEqual(metrics['AR_max_1'], 1.0, 5)
    self.assertAlmostEqual(metrics['AR_max_10'], 1.0, 5)
    self.assertAlmostEqual(metrics['AR_max_100'], 1.0, 5)
    self.assertAlmostEqual(metrics['AR_small'], -1.0, 5)
    self.assertAlmostEqual(metrics['AR_medium'], 1.0, 5)
    self.assertAlmostEqual(metrics['AR_large'], -1.0, 5)

  def test_coco_global_val_metrics_function_subset(self):
    """Test that evaluation on part of the dataset works."""
    # Instatiate evaluator with default annotations file:
    outputs, targets = _get_fake_detection_example()
    included_image_ids = {int(targets['image/id'])}
    global_metrics_evaluator = detr_train_utils.DetrGlobalEvaluator(
        'coco_detr_detection')
    global_metrics_evaluator.add_example(prediction=outputs, target=targets)
    global_metrics_evaluator.compute_metrics(
        included_image_ids=included_image_ids)


class DETRVisualizationsTest(parameterized.TestCase):
  """Test visualization functions."""

  @classmethod
  def _sample_cxcywh_bbox(cls, key, batch_shape):
    """Samples a bounding box in the [cx, cy, w, h] in [0, 1] range format."""
    frac = 0.8
    sample = jax.random.uniform(key, shape=(*batch_shape, 4)) * frac
    cx, cy, w, h = jnp.split(sample, indices_or_sections=4, axis=-1)
    # Make sure the bounding box doesn't cross the right and top image borders
    w = jnp.where(cx + w/2. >= 1., frac * 2. * (1. - cx), w)
    h = jnp.where(cy + h/2. >= 1., frac * 2. * (1. - cy), h)
    # Make sure the bounding box doesn't cross the left and bottom image borders
    w = jnp.where(cx - w/2. <= 0., frac * 2. * cx, w)
    h = jnp.where(cy - h/2. <= 0., frac * 2. * cy, h)

    bbox = jnp.concatenate([cx, cy, w, h], axis=-1)
    return bbox

  def _generate_visualization_inputs(self):
    """Returns fake inputs for visualization tests."""
    batchsize = 4
    num_classes = 81
    max_num_boxes = 3
    num_preds = 5

    key = jax.random.PRNGKey(0)

    # Create fake predictions and targets
    key, subkey = jax.random.split(key)
    # set probabilities for class 0 higher than others
    p_logits = jnp.ones(num_classes).at[0].set(5.)
    p = jax.nn.softmax(p_logits)
    tgt_labels = jax.random.choice(subkey,
                                   num_classes,
                                   shape=(batchsize, max_num_boxes),
                                   replace=True,
                                   p=p)
    # Ensure last target is dummy empty target.
    tgt_labels = tgt_labels.at[:, -1].set(0)

    key, subkey = jax.random.split(key)
    pred_logits = jax.random.normal(subkey,
                                    shape=(batchsize, num_preds, num_classes))

    key, subkey = jax.random.split(key)
    pred_bbox = self._sample_cxcywh_bbox(subkey,
                                         batch_shape=(batchsize, num_preds))

    key, subkey = jax.random.split(key)
    tgt_bbox = self._sample_cxcywh_bbox(subkey,
                                        batch_shape=(batchsize, max_num_boxes))
    key, subkey = jax.random.split(key)
    iscrowd = jax.random.uniform(subkey, shape=(batchsize, max_num_boxes)) > 0.5

    pred = {'pred_logits': pred_logits, 'pred_boxes': pred_bbox}
    tgt = {'labels': tgt_labels,
           'boxes': tgt_bbox,
           'is_crowd': iscrowd,
           'size': jnp.array([[128, 256], [256, 128], [224, 224], [256, 256]])}

    mean_rgb = np.reshape(np.array([0.48, 0.456, 0.406]), [1, 1, 1, 3])
    std_rgb = np.reshape(np.array([0.229, 0.224, 0.225]), [1, 1, 1, 3])

    imgs = np.zeros((batchsize, 256, 256, 3))
    for i, sz in enumerate(tgt['size']):
      img = np.random.uniform(size=(sz[0], sz[1], 3))
      imgs[i, :sz[0], :sz[1], :] = img

    imgs = (imgs - mean_rgb) / std_rgb
    batch = {
        'inputs': jnp.array(imgs),
        'label': tgt,
    }

    return jax.device_get(pred), jax.device_get(batch)

  def test_draw_boxes_side_by_side(self):
    """Test draw_boxes_side_by_side."""
    pred, batch = self._generate_visualization_inputs()
    viz = detr_train_utils.draw_boxes_side_by_side(
        pred, batch, collections.defaultdict(lambda: ''))
    self.assertSequenceEqual(viz.shape, [4, 256, 512, 3])


def _get_fake_detection_example(dataset='coco'):
  """Manually create a single example that has known results."""
  # A single box, values taken from ground-truth annotations and manually
  # converted to relative [cx, cy, w, h] format:
  h, w = 427, 640
  bx, by, bw, bh = 217.62, 240.54, 38.99, 57.75

  outputs = {}
  outputs['pred_boxes'] = np.array([
      [(bx + bw / 2) / w, (by + bh / 2) / h, bw / w, bh / h],
  ])
  outputs['pred_boxes'] = jnp.asarray(outputs['pred_boxes'])
  outputs['pred_logits'] = np.zeros((1, NUM_COCO_CLASSES))
  outputs['pred_logits'][0, 1] = 100.0

  targets = {}
  targets['size'] = np.array([h, w])
  targets['orig_size'] = np.array([h, w])
  targets['image/id'] = np.array([397133])
  targets['boxes'] = outputs['pred_boxes']
  targets['is_crowd'] = np.array([0])

  if dataset == 'coco':
    targets['labels'] = np.array([44])
  elif dataset == 'lvis':
    targets['labels'] = np.array([1])
    targets['not_exhaustive_category_ids'] = np.array([])
    targets['neg_category_ids'] = np.array([2, 3])
  return outputs, targets


if __name__ == '__main__':
  absltest.main()
