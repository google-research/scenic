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

"""Evaluator classes.

This file is modified from the scenic coco evaluator at
https://github.com/google-research/scenic/blob/main/scenic/dataset_lib/
coco_dataset/coco_eval.py
"""

import json
import os
from typing import Any, Dict, Optional
from absl import logging

import lvis
import numpy as np
from pycocotools import cocoeval
from scenic.dataset_lib.coco_dataset import coco_eval as coco_eval_wrapper
from tensorflow.io import gfile


class LvisEvaluator(coco_eval_wrapper.DetectionEvaluator):
  """LVIS evaluator."""

  def __init__(self, annotations_loc):
    """Initializes a LvisEvaluator object."""
    self.annotations = []
    self.annotated_img_ids = []
    self.coco = lvis.LVIS(json.load(gfile.GFile(annotations_loc, 'r')))
    self.label_to_coco_id = {
        i: cat['id'] for i, cat in enumerate(sorted(
            self.coco.dataset['categories'], key=lambda x: x['id']))}


class DetectionEvaluator(object):
  """Class that feeds model outputs to COCO evaluation api.

    The module is mostly the same as DetrGlobalEvaluator, except for CenterNet
      outputs are in different format.
  """

  def __init__(self, dataset_name: str = 'coco', annotations_loc=None):
    self.dataset_name = dataset_name
    if self.dataset_name == 'lvis':
      self.coco_evaluator = LvisEvaluator(annotations_loc=annotations_loc)
    else:
      assert self.dataset_name == 'coco', self.dataset_name
      self.coco_evaluator = coco_eval_wrapper.DetectionEvaluator(
          threshold=0.0, annotations_loc=annotations_loc)
    self._included_image_ids = set()
    self._num_examples_added = 0

  def add_example(self, prediction: Any, target: Dict[str, np.ndarray]):
    """Add a single example to the evaluator.

    Args:
      prediction: Model prediction tuple of 3 arrays: boxes, scores, classes.
        'boxes' is in shape of `[num_objects, 4]` and 'pred_boxes', 'classes'
        are botoh in shape of `[num_objects, num_classes]`.
        Box coordinates are absolute values in the input image coordinates.
        We need to scale them back to the original image coordinates using
        information in target.
      target: Target dictionary with keys 'orig_size', 'size', and 'image/id'.
    """
    boxes, scores, classes = prediction
    h, w = np.asarray(target['orig_size'])
    input_h, input_w = np.asarray(target['size'])
    scale_factor = np.array([w, h, w, h]) / np.array(
        [input_w, input_h, input_w, input_h])
    boxes = boxes * scale_factor[np.newaxis, :]
    boxes = np.maximum(boxes, 0)
    boxes[:, [0, 2]] = np.minimum(boxes[:, [0, 2]], w)
    boxes[:, [1, 3]] = np.minimum(boxes[:, [1, 3]], h)
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]
    boxes = np.asarray(boxes).tolist()
    img_id = int(target['image/id'])
    if img_id in self._included_image_ids:
      logging.info('Duplicate image %s not being added again', img_id)
      return
    self._included_image_ids.add(img_id)

    # coco_eval.DetectionEvaluator.add_annotation is specific for DETR formats
    #   and not compatible here.
    for bbox, label, score in zip(boxes, classes, scores):
      single_classification = {
          'image_id': img_id,
          'category_id': self.coco_evaluator.label_to_coco_id[label],
          'bbox': bbox,
          'score': score,
      }
      self.coco_evaluator.annotations.append(single_classification)
      self.coco_evaluator.annotated_img_ids.append(img_id)
    self._num_examples_added += 1

  def compute_metrics(
      self,
      clear_annotations: Optional[bool] = True,
      eval_class_agnostic: bool = False) -> Dict[str, Any]:
    """Computes the metrics for all added predictions."""
    if eval_class_agnostic:
      default_id = self.coco_evaluator.coco.dataset['categories'][0]['id']
      for x in self.coco_evaluator.coco.dataset['annotations']:
        x['category_id'] = default_id
    if self.dataset_name == 'lvis':
      lvis_dt = lvis.LVISResults(
          self.coco_evaluator.coco,
          self.coco_evaluator.annotations)
      lvis_eval = lvis.LVISEval(
          self.coco_evaluator.coco, lvis_dt, iou_type='bbox')
      lvis_eval.evaluate()
      lvis_eval.accumulate()
      lvis_eval.summarize()
      lvis_eval.print_results()
      results_dict = lvis_eval.results
    else:
      coco_eval = cocoeval.COCOeval(
          self.coco_evaluator.coco,
          self.coco_evaluator.coco.loadRes(  # pytype: disable=attribute-error
              self.coco_evaluator.annotations),
          'bbox')
      coco_eval.params.imgIds = self.coco_evaluator.annotated_img_ids
      coco_eval.evaluate()
      coco_eval.accumulate()
      coco_eval.summarize()
      results_dict = self.coco_evaluator.construct_result_dict(coco_eval.stats)
      recall50 = coco_eval.eval['recall']  # ious x classes x areas x max_dets
      recall50 = recall50[0, :, 0, -1]  # iou=0.5, area=all, max_det=100
      recall50 = recall50[recall50 >= 0].mean()
      results_dict['Recall50'] = recall50
    if clear_annotations:
      self.coco_evaluator.clear_annotations()
    return results_dict

  def write_pred_annotations_to_file(self,
                                     path: str,
                                     fname_app: Optional[str] = None,
                                     clear_annotations: Optional[bool] = True):
    """Writes predictions to file in JSON format.

    Args:
      path: Path to write the prediction annotation JSON file.
      fname_app: Optional string to append to the file name.
      clear_annotations: Clear annotations after they are stored in a file.
    """
    if not gfile.exists(path):
      gfile.makedirs(path)
    json_file_name = f"predictions{fname_app if fname_app else ''}.json"
    json_file_path = os.path.join(path, json_file_name)

    def _convert_to_serializable(obj):
      if isinstance(obj, np.ndarray):
        return obj.tolist()
      elif isinstance(obj, np.float32):
        return float(obj)
      else:
        raise TypeError(f'Unserializable object {obj} of type {type(obj)}')

    with gfile.GFile(json_file_path, 'w') as f:
      f.write(
          json.dumps(
              self.coco_evaluator.annotations,
              default=_convert_to_serializable))
    logging.info('Predicted annotations are stored in %s.', json_file_path)
    if clear_annotations:
      self.coco_evaluator.clear_annotations()

  def __len__(self):
    return self._num_examples_added

  def clear(self):
    self.coco_evaluator.clear_annotations()
    self._num_examples_added = 0
    self._included_image_ids = set()
