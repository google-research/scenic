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

"""COCO evaluation metrics based on pycocotools.

Implementation is based on
https://github.com/google/flax/blob/ac5e46ed448f4c6801c35d15eb15f4638167d8a1/examples/retinanet/coco_eval.py

"""

import collections
import contextlib
import functools
import io
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Set
import zipfile

from absl import logging
import jax
import numpy as np
import PIL
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


from tensorflow.io import gfile

COCO_ANNOTATIONS_PATH = os.path.join(
    os.path.dirname(__file__),
    'data',
    'instances_val2017.json')

PANOPTIC_ANNOTATIONS_PATH = os.path.join(
    os.path.dirname(__file__),
    'data',
    'panoptic_val2017.json')

PANOPTIC_CATEGORIES_PATH = os.path.join(
    os.path.dirname(__file__),
    'data',
    'panoptic_coco_categories.json')

PANOPTIC_ANNOTATIONS_DIR = (
    'panoptic_annotations_trainval2017')


@functools.lru_cache(maxsize=1)
def _load_json(path, mode='r'):
  """Loading json file."""


  with gfile.GFile(path, mode) as f:
    return json.load(f)


class UniversalCOCO(COCO):
  """Extends the COCO API to (optionally) support panoptic annotations."""

  def __init__(self, annotation_file: Optional[str] = None):
    """Constructor of Microsoft COCO helper class.

    Args:
      annotation_file: path to annotation file.
    """
    self.annotation_file = annotation_file
    self.reload_ground_truth()

  def reload_ground_truth(self, included_image_ids: Optional[List[int]] = None):
    """Reload GT annotations, optionally just a subset."""
    self.dataset, self.anns, self.cats, self.imgs = {}, {}, {}, {}
    self.imgToAnns = collections.defaultdict(list)  # pylint: disable=invalid-name
    self.catToImgs = collections.defaultdict(list)  # pylint: disable=invalid-name
    if self.annotation_file is not None:
      dataset = _load_json(self.annotation_file)
      assert isinstance(
          dataset, dict), 'annotation file format {} not supported'.format(
              type(dataset))

      if 'segments_info' in dataset['annotations'][0]:
        # Dataset is in panoptic format. Translate to standard format:
        dataset['annotations'] = _panoptic_to_standard_annotations(
            dataset['annotations'])

      if 'iscrowd' not in dataset['annotations'][0]:
        # Dataset is in LVIS format. Add missing 'iscrowd' field":
        for image_annotation in dataset['annotations']:
          image_annotation['iscrowd'] = 0

      # Subselect included image IDs:
      if included_image_ids is not None:
        included_image_ids = set(included_image_ids)
        logging.warn('Using only a subset of validation set: %s of %s images.',
                     len(included_image_ids), len(dataset['images']))
        dataset['images'] = [
            a for a in dataset['images'] if a['id'] in included_image_ids]
        dataset['annotations'] = [
            a for a in dataset['annotations']
            if a['image_id'] in included_image_ids]

      self.dataset = dataset
      self.createIndex()


def _panoptic_to_standard_annotations(annotations):
  """Translates panoptic annotations to standard annotations.

  Panoptic annotations have one extra level of nesting compared to
  detection annotations (see https://cocodataset.org/#format-data), which
  we remove here. Also see
  pycocotools/panopticapi/converters/panoptic2detection_coco_format.py
  for reference regarding the conversion. Here, we do not convert the
  segmentation masks, since they are not required for the detection
  metric.

  Args:
    annotations: Dict with panoptic annotations loaded from JSON.

  Returns:
    Updated annotations dict in standard COCO format.
  """

  object_annotations = []
  for image_annotation in annotations:
    for object_annotation in image_annotation['segments_info']:
      object_annotations.append({
          'image_id': image_annotation['image_id'],
          'id': object_annotation['id'],
          'category_id': object_annotation['category_id'],
          'iscrowd': object_annotation['iscrowd'],
          'bbox': object_annotation['bbox'],
          'area': object_annotation['area'],
      })
  return object_annotations


class DetectionEvaluator():
  """Main evaluator class."""

  def __init__(self,
               annotations_loc: Optional[str] = None,
               threshold: float = 0.05,
               disable_output: bool = True):
    """Initializes a DetectionEvaluator object.

    Args:
      annotations_loc: a path towards the .json files storing the COCO/2014
        ground truths for object detection. To get the annotations, please
        download the relevant files from https://cocodataset.org/#download
      threshold: a scalar which indicates the lower threshold (inclusive) for
        the scores. Anything below this value will be removed.
      disable_output: if True disables the output produced by the COCO API
    """
    self.annotations = []
    self.annotated_img_ids = []
    self.threshold = threshold
    self.disable_output = disable_output
    if annotations_loc is None:
      annotations_loc = COCO_ANNOTATIONS_PATH

    if self.disable_output:
      with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
          self.coco = UniversalCOCO(annotations_loc)
    else:
      self.coco = UniversalCOCO(annotations_loc)

    # Dict to translate model labels to COCO category IDs:
    self.label_to_coco_id = {
        i: cat['id'] for i, cat in enumerate(self.coco.dataset['categories'])}

  @staticmethod
  def construct_result_dict(coco_metrics):
    """Packs the COCOEval results into a dictionary.

    Args:
      coco_metrics: an array of length 12, as returned by `COCOeval.summarize()`
    Returns:
      A dictionary which contains all the COCO metrics. For more details,
      visit: https://cocodataset.org/#detection-eval.
    """
    return {
        'AP': coco_metrics[0],
        'AP_50': coco_metrics[1],
        'AP_75': coco_metrics[2],
        'AP_small': coco_metrics[3],
        'AP_medium': coco_metrics[4],
        'AP_large': coco_metrics[5],
        'AR_max_1': coco_metrics[6],
        'AR_max_10': coco_metrics[7],
        'AR_max_100': coco_metrics[8],
        'AR_small': coco_metrics[9],
        'AR_medium': coco_metrics[10],
        'AR_large': coco_metrics[11]
    }

  def clear_annotations(self):
    """Clears the annotations collected in this object.

    It is important to call this method either at the end or at the beginning
    of a new evaluation round (or both). Otherwise, previous model inferences
    will skew the results due to residual annotations.
    """
    self.annotations.clear()
    self.annotated_img_ids.clear()

  def extract_classifications(self, bboxes, scores):
    """Extracts the label for each bbox, and sorts the results by score.

    More specifically, after extracting each bbox's label, the bboxes and
    scores are sorted in descending order based on score. The scores which fall
    below `threshold` are removed.
    Args:
      bboxes: a matrix of the shape (|B|, 4), where |B| is the number of
        bboxes; each row contains the `[x1, y1, x2, y2]` of the bbox
      scores: a matrix of the shape (|B|, K), where `K` is the number of
        classes in the object detection task
    Returns:
      A tuple consisting of the bboxes, a vector of length |B| containing
      the label of each of the anchors, and a vector of length |B| containing
      the label score. All elements are sorted in descending order relative
      to the score.
    """
    # Extract the labels and max score for each anchor
    labels = np.argmax(scores, axis=1)

    # Get the score associated to each anchor's label
    scores = scores[np.arange(labels.shape[0]), labels]

    # Apply the threshold
    kept_idx = np.where(scores >= self.threshold)[0]
    scores = scores[kept_idx]
    labels = labels[kept_idx]
    bboxes = bboxes[kept_idx]

    # Sort everything in descending order and return
    sorted_idx = np.flip(np.argsort(scores, axis=0))
    scores = scores[sorted_idx]
    labels = labels[sorted_idx]
    bboxes = bboxes[sorted_idx]

    return bboxes, labels, scores

  def add_annotation(self, bboxes, scores, img_id):
    """Add a single inference example as COCO annotation for later evaluation.

    Labels should not include a background/padding class, but only valid object
    classes.

    Note that this method raises an exception if the `threshold` is too
    high and thus eliminates all detections.

    Args:
      bboxes: [num_objects, 4] array of bboxes in COCO format [x, y, w, h] in
        absolute image coorinates.
      scores: [num_objects, num_classes] array of scores (softmax outputs).
      img_id: scalar COCO image ID.
    """

    # Get the sorted bboxes, labels and scores (threshold is applied here):
    i_bboxes, i_labels, i_scores = self.extract_classifications(
        bboxes, scores)

    if not i_bboxes.size:
      raise ValueError('All objects were thresholded out.')

    # Iterate through the thresholded predictions and pack them in COCO format:
    for bbox, label, score in zip(i_bboxes, i_labels, i_scores):
      single_classification = {
          'image_id': img_id,
          'category_id': self.label_to_coco_id[label],
          'bbox': bbox.tolist(),
          'score': score
      }
      self.annotations.append(single_classification)
      self.annotated_img_ids.append(img_id)

  def get_annotations_and_ids(self):
    """Returns copies of `self.annotations` and `self.annotated_img_ids`.

    Returns:
      Copies of `self.annotations` and `self.annotated_img_ids`.
    """
    return self.annotations.copy(), self.annotated_img_ids.copy()

  def set_annotations_and_ids(self, annotations, ids):
    """Sets the `self.annotations` and `self.annotated_img_ids`.

    This method should only be used when trying to compute the metrics across
    hosts, where one host captures the data from everyone in an effort to
    produce the entire dataset metrics.
    Args:
      annotations: the new `annotations`
      ids: the new `annotated_img_ids`
    """
    self.annotations = annotations
    self.annotated_img_ids = ids

  def compute_coco_metrics(self, clear_annotations=False):
    """Compute the COCO metrics for the collected annotations.

    Args:
      clear_annotations: if True, clears the `self.annotations`
        parameter after obtaining the COCO metrics

    Returns:
      The COCO metrics as a dictionary, defining the following entries:
      ```
      Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
      Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]
      Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]
      Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
      Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
      Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
      Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]
      Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
      Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]
      ```
    """
    def _run_eval():
      # Create prediction object for producing mAP metric values
      pred_object = self.coco.loadRes(self.annotations)

      # Compute mAP
      coco_eval = COCOeval(self.coco, pred_object, 'bbox')
      coco_eval.params.imgIds = self.annotated_img_ids
      coco_eval.evaluate()
      coco_eval.accumulate()
      coco_eval.summarize()
      return coco_eval

    if self.disable_output:
      with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
          coco_eval = _run_eval()
    else:
      coco_eval = _run_eval()

    # Clear annotations if requested
    if clear_annotations:
      self.clear_annotations()

    # Pack the results
    return self.construct_result_dict(coco_eval.stats)


