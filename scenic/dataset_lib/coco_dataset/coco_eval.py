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
from pycocotools.panopticapi import evaluation as panoptic_eval
from pycocotools.panopticapi import utils as panoptic_utils
from tensorflow.io import gfile

COCO_ANNOTATIONS_PATH = os.path.join(
    '/readahead/128M/', os.path.dirname(__file__), 'data',
    'instances_val2017.json')

PANOPTIC_ANNOTATIONS_PATH = os.path.join(
    '/readahead/128M/', os.path.dirname(__file__), 'data',
    'panoptic_val2017.json')

PANOPTIC_CATEGORIES_PATH = os.path.join(
    '/readahead/128M/', os.path.dirname(__file__), 'data',
    'panoptic_coco_categories.json')

# TODO(mjlm): Move these to placer?
PANOPTIC_ANNOTATIONS_DIR = (
    'panoptic_annotations_trainval2017')


@functools.lru_cache(maxsize=1)
def _load_json(path):
  return json.load(gfile.GFile(path, 'r'))


class PanopticCOCO(COCO):
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
          self.coco = PanopticCOCO(annotations_loc)
    else:
      self.coco = PanopticCOCO(annotations_loc)

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


class PanopticEvaluator(DetectionEvaluator):
  """Wraps DetectionEvaluator and adds panoptic evaluation code."""

  def __init__(self,
               det_threshold: float = 0.05,
               pano_threshold: float = 0.85,
               disable_output: bool = False):
    """Panoptic evaluator class.

    Args:
      det_threshold: Scalar which indicates the lower threshold (inclusive) for
        detection scores. Detections below this value will be excluded from
        detection metrics computation.
      pano_threshold: Scalar which indicates the threshold for panoptic objects.
        Objects with scores below this value will be excluded from panoptic
        metrics computation.
      disable_output: if True disables the output produced by the COCO API.

    """
    super().__init__(
        annotations_loc=PANOPTIC_ANNOTATIONS_PATH,
        threshold=det_threshold,
        disable_output=disable_output)

    self.pano_threshold = pano_threshold
    self.pano_annotations = []

    # Update annotation dicts to panoptic versions:
    with gfile.GFile(PANOPTIC_CATEGORIES_PATH, 'r') as f:
      self.categories = json.load(f)
    self.is_thing_map = {c['id']: c['isthing'] for c in self.categories}
    self.label_to_coco_id = {
        i: c['id'] for i, c in enumerate(self.categories)}

    # Create local directories and files (required by pycocotools):
    self._local_gt_dir_obj = tempfile.TemporaryDirectory()
    self._local_pred_dir_obj = tempfile.TemporaryDirectory()

    self._copy_pano_annotations_to_local()

    image_zip_dest = os.path.join(self._local_gt_dir_obj.name, 'images')
    self.annotations_images_dir = os.path.join(
        image_zip_dest, 'panoptic_val2017')
    if not gfile.exists(image_zip_dest):
      remote_images_zip = os.path.join(
          PANOPTIC_ANNOTATIONS_DIR, 'panoptic_val2017.zip')
      local_images_zip = os.path.join(
          self._local_gt_dir_obj.name, 'panoptic_val2017.zip')
      gfile.copy(remote_images_zip, local_images_zip)
      gfile.mkdir(image_zip_dest)
      with zipfile.ZipFile(local_images_zip, 'r') as f:
        f.extractall(image_zip_dest)

  def get_color_generator(self):
    """Returns a function that generates visualization colors from labels.

    The returned function expects labels in the Scenic format, including the
    increment of labels due to the padding label.

    The function generates slightly different colors on subsequent calls to make
    different instances of the same class distinguishable.
    """
    category_dict = {category['id']: category for category in self.categories}
    id_generator = panoptic_utils.IdGenerator(category_dict)

    def color_generator(label):
      if not label:
        return 127, 127, 127  # Padding is gray.
      cat_id = self.label_to_coco_id[label - 1]  # -1 accounts for padding.
      return id_generator.get_color(cat_id)

    return color_generator

  def clear_annotations(self):
    super().clear_annotations()
    self.pano_annotations.clear()
    self._local_pred_dir_obj.cleanup()
    self._local_pred_dir_obj = tempfile.TemporaryDirectory()

  def compute_coco_metrics(
      self, clear_annotations: bool = False) -> Dict[str, Any]:
    detection_metrics = super().compute_coco_metrics(clear_annotations=False)
    panoptic_metrics = self.compute_panoptic_metrics()
    if clear_annotations:
      self.clear_annotations()
    return {**detection_metrics, **panoptic_metrics}

  def add_panoptic_annotation(self, annotation_dict: Dict[str, np.ndarray]):
    """Adds the model predictions for a single example."""
    expected_keys = {'pred_logits', 'pred_boxes', 'pred_masks', 'orig_size',
                     'image/id'}
    if set(annotation_dict.keys()) != expected_keys:
      raise ValueError(f'Annotation should be dict with keys {expected_keys}, '
                       f'got dict with keys {annotation_dict.keys()}')
    if annotation_dict['pred_masks'].shape[1:3] != tuple(
        annotation_dict['orig_size']):
      raise ValueError(
          'Predicted masks passed to this function should be formatted like '
          'the ground-truth PNG mask images, i.e. contain no padding and be of '
          'size `orig_size`. Received image with ID '
          f"{annotation_dict['image/id']} with shape "
          f"{annotation_dict['pred_masks'].shape}, but `orig_size` is "
          f"{annotation_dict['orig_size']}.")
    annotation_dict = self._post_process_panoptic_annotation(annotation_dict)

    # Write image to disk and store other data in the object:
    # TODO(mjlm): This can be put into postproc fun?
    save_path = os.path.join(
        self._local_pred_dir_obj.name, annotation_dict['file_name'])
    with gfile.GFile(save_path, 'wb') as f:
      f.write(annotation_dict.pop('png_string'))
    self.pano_annotations.append(annotation_dict)

  def compute_panoptic_metrics(self) -> Dict[str, Any]:
    """Computes PQ score on collected annotations."""
    json_data = {'annotations': self.pano_annotations}
    pred_json_file = os.path.join(
        self._local_pred_dir_obj.name, 'predictions.json')
    with gfile.GFile(pred_json_file, 'w') as f:
      f.write(json.dumps(json_data))

    # To have the same number of examples on all hosts, some val set examples
    # may be dropped. Here, we re-write the ground-truth annotations, excluding
    # the dropped examples:
    self._copy_pano_annotations_to_local(
        include_ids={a['image_id'] for a in self.pano_annotations})

    if not any(a['segments_info'] for a in self.pano_annotations):
      logging.info(
          'No predictions left in any of %d images after filtering. Skipping '
          'PQ eval.', len(self.pano_annotations))
      return {}

    metrics_fn = functools.partial(
        panoptic_eval.pq_compute,
        gt_json_file=self.annotations_json_path,
        gt_folder=self.annotations_images_dir,
        pred_json_file=pred_json_file,
        pred_folder=self._local_pred_dir_obj.name,
        cpu_num=1)  # Must be 1 because multiprocessing does not work on Borg.

    if self.disable_output:
      with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
          metrics_dict_nested = metrics_fn()
    else:
      metrics_dict_nested = metrics_fn()

    # Flatten the nested dict returned by panoptic_eval.pq_compute:
    return _flatten_nested_dict(metrics_dict_nested)

  def _copy_pano_annotations_to_local(
      self, include_ids: Optional[Set[int]] = None):
    """Dump a local copy of the panoptic annotations (optionally, a subset)."""
    with gfile.GFile(PANOPTIC_ANNOTATIONS_PATH, 'rb') as f:
      annotations = json.load(f)
    num_annotations = len(annotations['images'])
    if (include_ids is not None) and (len(include_ids) < num_annotations):
      annotations['images'] = [
          img for img in annotations['images'] if img['id'] in include_ids]
      annotations['annotations'] = [
          ann for ann in annotations['annotations']
          if ann['image_id'] in include_ids
      ]
      logging.warning('Evaluating on only %s of %s annotations.',
                      len(include_ids), num_annotations)
    self.annotations_json_path = os.path.join(
        self._local_gt_dir_obj.name, 'panoptic_val2017.json')
    with gfile.GFile(self.annotations_json_path, 'wb') as f:
      json.dump(annotations, f)

  def _post_process_panoptic_annotation(self, annotation):
    """Converts model output for a single example into panoptic API format."""
    # Remove padding and below-threshold detections:
    # TODO(mjlm): Should we remove pad class like in detection eval?
    probs = np.asarray(jax.nn.softmax(annotation['pred_logits'], axis=-1))
    scores = np.max(probs, axis=-1)
    labels = np.argmax(probs, axis=-1)
    keep = (labels != 0) & (scores > self.pano_threshold)
    scores = scores[keep]
    labels = labels[keep] - 1  # Don't count padding class.
    boxes = annotation['pred_boxes'][keep, ...]
    masks = annotation['pred_masks'][keep, ...]

    # Unpad mask and resize to original size:

    # Perform spatial softmax to normalize masks:
    num_objects, h, w = masks.shape
    masks_linear = np.reshape(masks, [num_objects, h * w])
    masks_linear = np.asarray(jax.nn.softmax(masks_linear, axis=-1))
    masks = np.reshape(masks, [num_objects, h, w])

    assert len(boxes) == len(labels)

    stuff_segment_lists = collections.defaultdict(list)
    for i, label in enumerate(labels):
      coco_id = self.label_to_coco_id[label]
      if not self.is_thing_map[coco_id]:
        stuff_segment_lists[label].append(i)

    seg_img, areas = _get_panoptic_img(masks, scores, stuff_segment_lists)

    if labels.size:
      # Iteratively filter empty/tiny masks:
      keep = areas > 4
      while not np.all(keep):
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep]
        seg_img, areas = _get_panoptic_img(masks, scores, stuff_segment_lists)
        keep = areas > 4
    else:
      labels = np.ones(1)  # TODO(mjlm): Is 1 the correct dummy label?

    segments_info = []
    for i, (label, area) in enumerate(zip(labels, areas)):
      coco_id = self.label_to_coco_id[label]
      segments_info.append({
          'id': int(i),
          'isthing': self.is_thing_map[coco_id],
          'category_id': coco_id,
          'area': int(area)
      })

    with io.BytesIO() as out:
      seg_img.save(out, format='PNG')
      processed = {
          'png_string': out.getvalue(),
          'segments_info': segments_info,
          'image_id': int(annotation['image/id']),
          'file_name': f'{annotation["image/id"]:012d}.png',
      }
    return processed


def _get_panoptic_img(masks, scores, stuff_segment_lists=None):
  """Creates final panoptic image and computes segment areas."""

  # `masks` has shape [num_objects, H, W]:
  num_objects, h, w = masks.shape

  if num_objects:
    label_masks = np.argmax(masks, axis=0)
  else:
    # No objects were detected:
    label_masks = np.zeros((h, w))

  if stuff_segment_lists is not None:
    # Merge masks corresponding to the same stuff class:
    for stuff_segment_list in stuff_segment_lists.values():
      if len(stuff_segment_list) > 1:
        for segment_id in stuff_segment_list:
          label_masks[label_masks == segment_id] = stuff_segment_list[0]

  seg_img = PIL.Image.fromarray(panoptic_utils.id2rgb(label_masks))
  label_mask = panoptic_utils.rgb2id(np.asarray(seg_img))
  area = np.bincount(label_mask.ravel(), minlength=len(scores))

  return seg_img, area


def _flatten_nested_dict(d, prefix=None):
  d_flat = {}
  for k, v in d.items():
    new_k = f'{prefix}/{k}' if prefix else k
    if isinstance(v, dict):
      d_flat.update(_flatten_nested_dict(v, prefix=new_k))
    else:
      d_flat[new_k] = v
  return d_flat
