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

"""Utilities for DeformableDETR trainer/evaluator."""

import copy
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scenic.dataset_lib.coco_dataset import coco_eval
from scenic.model_lib.base_models import box_utils
from scenic.projects.baselines.detr.train_utils import DetrGlobalEvaluator

import scipy


class DetectionEvaluator(coco_eval.DetectionEvaluator):
  """Changes DETR Evaluator bay assuming labels vary in label_id space."""

  def __init__(self,
               annotations_loc: Optional[str] = None,
               threshold: float = 0.,
               disable_output: bool = True):
    """Initializes a DetectionEvaluator object."""
    super().__init__(
        annotations_loc=annotations_loc,
        threshold=threshold,
        disable_output=disable_output)

    # Just override the label map.
    max_id = max([c['id'] for c in self.coco.dataset['categories']])
    # Just a mapping from 0-index to 1-index, to remove no-object label.
    self.label_to_coco_id = {i: i + 1 for i in range(max_id)}


class DeformableDetrGlobalEvaluator(DetrGlobalEvaluator):
  """An interface between DeformableDETR implementation and COCO."""

  def __init__(self, dataset_name: str, **kwargs):
    """Instantiate evaluator and override to use DeformableDETR Eval."""
    del dataset_name  # Unused.

    self.coco_evaluator = DetectionEvaluator(**kwargs)
    self._included_image_ids = set()
    self._num_examples_added = 0

  def add_example(
      self, prediction: Dict[str, np.ndarray], target: Dict[str, np.ndarray]):
    """Add a single example to the evaluator.

    Note that the postprocessing of the predictions is significantly different
    from the one used in DETR. In DETR, each query gives one predicted box. Here
    each query gives num_classes predicted boxes, one for each class, and all
    these num_queries * num_classes predicted boxes are mixed together and the
    top 100 of them are sent to the evaluator. This different postprocessing
    gives about 0.5 final AP increase.

    Args:
      prediction: Model prediction dictionary with keys 'pred_img_ids',
        'pred_probs' in shape of `[num_objects, num_classes]` and 'pred_boxes'
        in shape of `[num_objects, 4]`. Box coordinates should be in raw DETR
        format, i.e. [cx, cy, w, h] in range [0, 1].
      target: Target dictionary with keys 'orig_size', 'size', and 'image/id'.
        Must also contain 'padding_mask' if the input image was padded.
    """
    if 'pred_boxes' not in prediction:
      # Add dummy to make eval work:
      prediction = copy.deepcopy(prediction)
      prediction['pred_boxes'] = np.zeros(
          (prediction['pred_logits'].shape[0], 4)) + 0.5

    # Convert from DETR [cx, cy, w, h] to COCO [x, y, w, h] bounding box format:
    boxes = box_utils.box_cxcywh_to_xyxy(prediction['pred_boxes'])
    boxes = np.array(boxes)
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]

    # Scale from relative to absolute size:
    # Note that the padding is implemented such that such that the model's
    # predictions are [0,1] normalized to the non-padded image, so scaling by
    # `orig_size` will convert correctly to the original image coordinates. No
    # image flipping happens during evaluation.
    h, w = np.asarray(target['orig_size'])
    scale_factor = np.array([w, h, w, h])
    boxes = boxes * scale_factor[np.newaxis, :]
    boxes = np.asarray(boxes)

    # Get scores, excluding the background class:
    if 'pred_probs' in prediction:
      scores = prediction['pred_probs'][:, 1:]
    else:
      scores = scipy.special.softmax(prediction['pred_logits'], axis=-1)[:, 1:]
    scores = np.asarray(scores)

    num_classes = scores.shape[1]
    topk_indices = np.argsort(scores.flatten())[-1:-101:-1]
    topk_box_indices = topk_indices // num_classes
    topk_score_indices = topk_indices % num_classes

    for i in range(len(topk_indices)):
      # Add example to evaluator:
      img_id = int(target['image/id'])
      single_classification = {
          'image_id':
              img_id,
          'category_id':
              self.coco_evaluator.label_to_coco_id[topk_score_indices[i]],
          'bbox':
              boxes[topk_box_indices[i]].tolist(),
          'score':
              scores[topk_box_indices[i]][topk_score_indices[i]]
      }
      self.coco_evaluator.annotations.append(single_classification)
      self.coco_evaluator.annotated_img_ids.append(img_id)

    self._num_examples_added += 1


def prepare_coco_eval_dicts(
    batch: Dict[str, Any],
    predictions: Dict[str, jnp.ndarray],
    logits_to_probs_fn: Callable[[jnp.ndarray], jnp.ndarray],
    gather: bool = False
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
  """Prepare predictions and validations for COCO eval.

  Args:
    batch: Eval batch targets.
    predictions: Predictions from DeformableDETR.
    logits_to_probs_fn: Convert logits to probabilities.
    gather: Whether to perform gather if we are in a pmapped eval.

  Returns:
    Targets and predictions formatted for COCO eval.
  """
  pred_probs = logits_to_probs_fn(predictions['pred_logits'])
  # Collect necessary predictions and target information from all hosts.
  predictions_out = {
      'pred_probs': pred_probs,
      'pred_logits': predictions['pred_logits'],
      'pred_boxes': predictions['pred_boxes']
  }
  labels = {
      'image/id': batch['label']['image/id'],
      'size': batch['label']['size'],
      'orig_size': batch['label']['orig_size'],
  }
  to_copy = [
      'labels', 'boxes', 'not_exhaustive_category_ids', 'neg_category_ids'
  ]
  for name in to_copy:
    if name in batch['label']:
      labels[name] = batch['label'][name]

  targets = {'label': labels, 'batch_mask': batch['batch_mask']}

  if gather:
    predictions_out = jax.lax.all_gather(predictions_out, 'batch')
    targets = jax.lax.all_gather(targets, 'batch')
  return targets, predictions_out  # pytype: disable=bad-return-type  # jax-ndarray
