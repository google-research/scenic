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

"""Unit test for the components in detr_base_model.py."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.projects.baselines.detr import detr_base_model

NUM_CLASSES = 81  # Lets say 80 classes + background.
BBOX_LOSS_COEF = 1.0
GIOU_LOSS_COEF = 0.1
CE_LOSS_COEF = 2.0
EOS_COEF = 0.1
NUM_AUX_OUTPUTS = 2


class MyObjectDetectionWithMatchingModel(
    detr_base_model.ObjectDetectionWithMatchingModel):
  """A dummy set detection model for testing purposes."""

  def __init__(self, config=None, dataset_meta_data=None):
    dataset_meta_data = {'num_classes': NUM_CLASSES, 'target_is_onehot': False}
    self.losses_and_metrics = ['labels', 'boxes']
    config = config or ml_collections.ConfigDict()
    config.aux_loss = True
    config.bbox_loss_coef = BBOX_LOSS_COEF
    config.giou_loss_coef = GIOU_LOSS_COEF
    config.class_loss_coef = CE_LOSS_COEF
    config.eos_coef = EOS_COEF
    self.loss_terms_weights = {
        'loss_class': config.class_loss_coef,
        'loss_bbox': config.bbox_loss_coef,
        'loss_giou': config.giou_loss_coef,
    }
    super().__init__(config, dataset_meta_data)

  def build_flax_model(self):
    pass


def fake_model_outputs_batch(num_boxes):
  """Generate fake data that resembles model `outputs` `batch` `indices`.

  See ObjectDetectionWithMatchingModel's loss_* functions for more details
  regarding the format of these dictionaries. The batch-size is hard-coded to
  3 because `indices` are manually synthesized.

  Args:
    num_boxes: int; Number of boxes in the data.

  Returns:
    `outputs`: dict; Dictionary of model predictions.
    `batch`: dict; Dictionary of None inputs and fake ground truth targets.
    `indices`: nd-array; Indices matching the structure returned by a matcher.
  """
  np.random.seed(num_boxes)

  outputs = {
      'pred_logits':
          jnp.array(
              np.random.normal(size=(3, num_boxes, NUM_CLASSES)),
              dtype=jnp.float32),
      'pred_boxes':
          jnp.array(
              np.random.uniform(size=(3, num_boxes, 4), low=0.0, high=1.0),
              dtype=jnp.float32),
      'pred_masks':
          jnp.array(
              np.random.uniform(size=(3, num_boxes, 8, 8), low=0.0, high=1.0),
              dtype=jnp.float32),
  }
  aux_outputs = [dict(outputs), dict(outputs)]
  outputs['aux_outputs'] = aux_outputs
  batch = {
      'inputs': None,
      'label': {
          'labels':
              jnp.array(np.random.randint(NUM_CLASSES, size=(3, num_boxes))),
          'boxes':
              jnp.array(
                  np.random.uniform(size=(3, num_boxes, 4), low=0.0, high=1.0),
                  dtype=jnp.float32),
          'masks':
              jnp.array(
                  np.argsort(
                      np.random.uniform(size=(3, num_boxes, 16, 16)),
                      axis=1) == 0,
                  dtype=jnp.float32),
          'image/id':
              jnp.array([87038, 348881, 143931]),
          'orig_size':
              jnp.array(
                  np.random.uniform(size=(3, 2), low=1, high=100),
                  dtype=jnp.int32),
      }
  }

  seq = np.arange(num_boxes, dtype=np.int32)
  seq_rev = seq[::-1]
  seq_21 = np.concatenate([seq[num_boxes // 2:], seq[:num_boxes // 2]])
  indices = np.array([(seq, seq_rev), (seq_rev, seq), (seq, seq_21)])

  return outputs, batch, indices


class TestObjectDetectionWithMatchingModel(parameterized.TestCase):
  """Test ObjectDetectionWithMatchingModel."""

  def is_valid(self, t):
    """Helper function to assert that tensor `t` does not have `nan`, `inf`."""
    self.assertFalse(jnp.isnan(t).any(), msg=f'Found nan\'s in {t}')
    self.assertFalse(jnp.isinf(t).any(), msg=f'Found inf\'s in {t}')

  def is_valid_loss(self, loss):
    """Helper function to assert that `loss` is of shape [] and `is_valid`."""
    self.assertSequenceEqual(loss.shape, [])
    self.is_valid(loss)

  @parameterized.named_parameters(
      ('num_boxes_8_without_log', 8, False, ['loss_class'], False),
      ('num_boxes_8_without_log_focal', 8, False, ['loss_class'], True),
      ('num_boxes_8_with_log', 8, True,
       ['loss_class', 'class_accuracy', 'class_accuracy_not_pad'], False),
      ('num_boxes_3_without_log', 3, False, ['loss_class'], False),
      ('num_boxes_3_without_log_focal', 3, False, ['loss_class'], True),
      ('num_boxes_3_with_log', 3, True,
       ['loss_class', 'class_accuracy', 'class_accuracy_not_pad'], False))
  def test_labels_losses_and_metrics(self, num_boxes, log, metrics_key,
                                     focal_loss):
    """Test loss_labels by checking its output's dictionary format.

    Args:
      num_boxes: int; Number of boxes used for creating the flax model.
      log: bool; Whether do logging or not in labels_losses_and_metrics.
      metrics_key: list; Expected metric keys.
      focal_loss: bool; Whether to use focal loss.
    """
    config = ml_collections.ConfigDict()
    config.focal_loss = focal_loss
    model = MyObjectDetectionWithMatchingModel(config=config)
    outputs, batch, indices = fake_model_outputs_batch(num_boxes)

    # Test loss function in the pmapped setup:
    def function_to_pmap(outputs, batch):
      return model.labels_losses_and_metrics(outputs, batch, indices, log)  # pytype: disable=wrong-arg-types  # jax-ndarray

    labels_lm_pmapped = jax.pmap(function_to_pmap, axis_name='batch')

    outputs, batch = (jax_utils.replicate(outputs), jax_utils.replicate(batch))
    losses, metrics = labels_lm_pmapped(outputs, batch)
    losses = jax_utils.unreplicate(losses)
    metrics = jax_utils.unreplicate(metrics)
    self.assertSameElements(['loss_class'], losses.keys())
    self.is_valid_loss(losses['loss_class'])
    self.assertSameElements(metrics_key, metrics.keys())
    for mk in metrics_key:
      self.is_valid(metrics[mk][0])
      self.is_valid(metrics[mk][1])

  @parameterized.named_parameters(('num_boxes_4', 4), ('num_boxes_9', 9))
  def test_boxes_losses_and_metrics(self, num_boxes):
    """Test loss_boxes by checking its output's dictionary format."""
    model = MyObjectDetectionWithMatchingModel()
    outputs, batch, indices = fake_model_outputs_batch(num_boxes)

    # Test loss function in the pmapped setup:
    boxes_lm_pmapped = jax.pmap(
        lambda o, b: model.boxes_losses_and_metrics(o, b, indices),
        axis_name='batch')

    outputs_replicate, batch_replicate = (jax_utils.replicate(outputs),
                                          jax_utils.replicate(batch))

    losses, metrics = boxes_lm_pmapped(outputs_replicate, batch_replicate)
    losses = jax_utils.unreplicate(losses)
    metrics = jax_utils.unreplicate(metrics)

    self.assertSameElements(['loss_bbox', 'loss_giou'], losses.keys())
    self.is_valid_loss(losses['loss_bbox'])
    self.is_valid_loss(losses['loss_giou'])

    self.assertSameElements(['loss_bbox', 'loss_giou'], metrics.keys())
    for i in range(2):  # metric and its normalizer
      self.is_valid(metrics['loss_bbox'][i])
      self.is_valid(metrics['loss_giou'][i])

    # Check whether hard-wiring boxes to match and hard-wiring indices to align
    # gives loss_bbox = 0.0 and loss_giou = 1.0:
    outputs['pred_boxes'] = batch['label']['boxes']
    indices = jnp.stack([indices[:, 0, :], indices[:, 0, :]], axis=1)
    outputs_replicate, batch_replicate = (jax_utils.replicate(outputs),
                                          jax_utils.replicate(batch))
    boxes_lm_pmapped = jax.pmap(
        lambda o, b: model.boxes_losses_and_metrics(o, b, indices),
        axis_name='batch')
    losses, metrics = boxes_lm_pmapped(outputs_replicate, batch_replicate)
    losses = jax_utils.unreplicate(losses)
    metrics = jax_utils.unreplicate(metrics)

    self.assertAlmostEqual(losses['loss_bbox'], 0.0, places=4)
    self.assertAlmostEqual(losses['loss_giou'], 0.0, places=4)

    self.assertAlmostEqual(
        metrics['loss_bbox'][0] / metrics['loss_bbox'][1], 0.0, places=5)
    self.assertAlmostEqual(
        metrics['loss_giou'][0] / metrics['loss_giou'][1], 0.0, places=4)

  @parameterized.named_parameters(('num_boxes_5', 5), ('num_boxes_6', 6))
  def test_loss_function(self, num_boxes):
    """Test loss_function by checking its output's dictionary format."""
    model = MyObjectDetectionWithMatchingModel()
    outputs, batch, indices = fake_model_outputs_batch(num_boxes)
    outputs_noaux = dict(outputs)
    outputs_noaux.pop('aux_outputs')

    outputs_replicated, batch_replicated = (jax_utils.replicate(outputs_noaux),
                                            jax_utils.replicate(batch))

    # Test loss function in the pmapped setup:
    loss_function_pmapped = jax.pmap(model.loss_function, axis_name='batch')
    outputs_replicated = jax_utils.replicate(outputs)

    indices_replicated = jax_utils.replicate(
        # Fake matching for the  final output +  2 aux outputs:
        [indices] * 3)
    total_loss, metrics_dict = loss_function_pmapped(outputs_replicated,
                                                     batch_replicated,
                                                     indices_replicated)

    total_loss, metrics_dict = (jax_utils.unreplicate(total_loss),
                                jax_utils.unreplicate(metrics_dict))

    # Collect what keys we expect to find in the metrics_dict:
    base = [
        'class_accuracy', 'loss_class', 'loss_bbox', 'loss_giou', 'total_loss',
        'loss_class_aux_0', 'loss_bbox_aux_0', 'loss_giou_aux_0',
        'loss_class_aux_1', 'loss_bbox_aux_1', 'loss_giou_aux_1',
        'class_accuracy_not_pad'
    ]
    base_unscaled = []
    for b in base:
      if b.split('_aux_')[0] in model.loss_terms_weights.keys():
        base_unscaled.append(b + '_unscaled')
      else:
        base_unscaled.append(b)
    base_scaled = [
        'loss_class',
        'loss_bbox',
        'loss_giou',
        'loss_class_aux_0',
        'loss_bbox_aux_0',
        'loss_giou_aux_0',
        'loss_class_aux_1',
        'loss_bbox_aux_1',
        'loss_giou_aux_1',
    ]
    expected_metrics_keys = base_unscaled + base_scaled
    self.assertSameElements(expected_metrics_keys, metrics_dict.keys())

    # Because weight decay is not used, the following must hold:
    object_detection_loss = 0
    for k in metrics_dict.keys():
      b = k.split('_aux_')[0]
      # If this loss is included in the total object detection loss...
      if '_unscaled' not in k and b in model.loss_terms_weights.keys():
        # ...get the normalizer for this loss:
        object_detection_loss += (
            # Already scaled loss term / loss term normalizer:
            metrics_dict[k][0] / metrics_dict[k][1])
    self.assertAlmostEqual(total_loss, object_detection_loss, places=5)

  def test_auxiliary_loss_consistency(self):
    """Test whether loss_function for the output and aux_outputs is same."""

    model = MyObjectDetectionWithMatchingModel()
    outputs, batch, indices = fake_model_outputs_batch(num_boxes=4)

    # Test loss function in the pmapped setup:
    loss_function_pmapped = jax.pmap(model.loss_function, axis_name='batch')

    indices_replicated = jax_utils.replicate(
        # Fake matching for the  final output +  2 aux outputs:
        [indices] * 3)
    outputs_replicated, batch_replicated = (jax_utils.replicate(outputs),
                                            jax_utils.replicate(batch))
    _, metrics_dict = loss_function_pmapped(outputs_replicated,
                                            batch_replicated,
                                            indices_replicated)

    metrics_dict = jax_utils.unreplicate(metrics_dict)

    for key in ['loss_class', 'loss_bbox', 'loss_giou']:
      for i in range(NUM_AUX_OUTPUTS):
        self.assertAlmostEqual(
            metrics_dict[key + '_unscaled'],
            metrics_dict[key + f'_aux_{i}_unscaled'],
            places=5)
        self.assertAlmostEqual(
            metrics_dict[key], metrics_dict[key + f'_aux_{i}'], places=5)


if __name__ == '__main__':
  absltest.main()
