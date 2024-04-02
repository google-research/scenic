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

"""Caption evaluator."""

import json
import os
from typing import Any, Dict, Optional

from absl import logging
from coco_caption import coco
import numpy as np
# pylint: disable=g-import-not-at-top
try:
  from scenic.projects.streaming_dvc import cococap_eval
except ImportError:
  cococap_eval = None
import tensorflow as tf
# pylint: enable=g-import-not-at-top


class CaptionEvaluator(object):
  """Class that feeds model outputs to COCO caption evaluation api."""

  def __init__(self, annotations_loc, eval_meteor_spice=False,
               step: Optional[int] = None):
    self.annotations_loc = annotations_loc
    logging.info('Initializing evaluator.')
    if self.annotations_loc:
      self.coco = coco.COCO(self.annotations_loc)
    self.annotations = {
        'images': [], 'annotations': [], 'type': 'captions', 'info': {},
        'licenses': [], 'categories': [{'id': 1, 'name': 'object'}]}
    self.predictions = []
    self.pred_image_set = set()
    self.gt_image_set = set()
    self._num_examples_added = 0
    self._num_captions_added = 0
    self.eval_meteor_spice = eval_meteor_spice
    self.step = step

  def add_example(self, prediction: Any, target: Dict[str, np.ndarray]):
    """Add a single example to the evaluator.

    Args:
      prediction: Model prediction tuple of 3 arrays: boxes, scores, classes.
        'boxes' is in shape of `[num_objects, 4]` and 'pred_boxes', 'classes'
        are botoh in shape of `[num_objects, num_classes]`.
        Box coordinates are absolute values in the input image coordinates.
        We need to scale them back to the original image coordinates using
        information in target.
      target: Target dictionary with keys and 'image/id'.
    """
    self._num_examples_added += 1
    if self.annotations_loc:
      # We will use image_id that matches the annotation file.
      img_id = int(target['image/id'])
    else:
      # we will create image_id on the fly
      if 'media_id' in target:
        img_id = ''.join([chr(x) for x in target['media_id'] if x])
      else:
        img_id = self._num_examples_added
      if img_id not in self.gt_image_set:
        # Avoid adding the same image twice due to repeated sampling.
        has_annotation = False
        for x in target['captions']:
          if x:  # remove empty captions from padding.
            self._num_captions_added += 1
            self.annotations['annotations'].append({
                'id': self._num_captions_added,
                'image_id': img_id,
                'caption': x})
            has_annotation = True
        if has_annotation:
          self.annotations['images'].append({'id': img_id})
          self.gt_image_set.add(img_id)
        else:
          logging.info('Skipping %s. No annotations found', img_id)
          return
    single_prediction = {
        'image_id': img_id,
        'caption': prediction,
    }

    if img_id not in self.pred_image_set:
      self.predictions.append(single_prediction)
    else:
      logging.info('Duplicate image %s not being added again', img_id)
    self.pred_image_set.add(img_id)

  def compute_metrics(
      self,
      save_dir: str,
      clear_annotations: Optional[bool] = True,
      skip_evaluate=False):
    """Computes the metrics for all added predictions."""
    json_file_path = self.write_pred_annotations_to_file(save_dir)
    if skip_evaluate:
      return {}
    if not self.annotations_loc:
      gt_file_path = self.write_pred_annotations_to_file(
          save_dir, is_groundtruth=True)
      self.coco = coco.COCO(gt_file_path)
    coco_res = self.coco.loadRes(json_file_path)
    coco_eval = cococap_eval.CustomCOCOEvalCap(  # pytype: disable=attribute-error
        self.coco, coco_res, eval_meteor_spice=self.eval_meteor_spice)
    coco_eval.params['image_id'] = coco_res.getImgIds()
    coco_eval.evaluate()
    results = coco_eval.eval
    if clear_annotations:
      self.clear()
    return results

  def clear(self):
    self.predictions = []
    self._num_examples_added = 0
    self._num_captions_added = 0

  def __len__(self):
    return self._num_examples_added

  def write_pred_annotations_to_file(self,
                                     path: str,
                                     is_groundtruth: bool = False):
    """Writes predictions to file in JSON format.

    Args:
      path: Path to write the prediction annotation JSON file.
      is_groundtruth: bool; if the file is ground truth or prediction.
    Returns:
      json_file_path: path to the saved json
    """
    if not tf.io.gfile.exists(path):
      tf.io.gfile.makedirs(path)
    fname_app = 'predictions' if not is_groundtruth else 'annotations'
    if self.step:
      json_file_name = f'caption_{fname_app}_{self.step}.json'
    else:
      json_file_name = f'caption_{fname_app}.json'
    json_file_path = os.path.join(path, json_file_name)
    logging.info('Saving predictions to %s.', json_file_path)
    def _convert_to_serializable(obj):
      if isinstance(obj, np.ndarray):
        return obj.tolist()
      elif isinstance(obj, np.float32):
        return float(obj)
      else:
        raise TypeError(f'Unserializable object {obj} of type {type(obj)}')

    with tf.io.gfile.GFile(json_file_path, 'w') as f:
      f.write(
          json.dumps(
              self.predictions if not is_groundtruth else self.annotations,
              default=_convert_to_serializable))
    logging.info('Predicted annotations are stored in %s.', json_file_path)
    return json_file_path
