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

"""Evaluation utils for PixelLLM."""
# pylint: disable=g-explicit-length-test

import json
import os
from typing import Any, Dict, Optional, List

from absl import logging
from coco_caption.coco import COCO as COCOCaption
import cv2
# pylint: disable=g-import-not-at-top
try:
  from coco_caption.eval import COCOEvalCap
  from coco_caption.bleu import Bleu
  from coco_caption.cider import Cider
  from coco_caption.meteor import Meteor
  from coco_caption.rouge import Rouge
  from coco_caption.upp_tokenizer import tokenize
except ImportError:
  COCOEvalCap = None
  Bleu = None
  Cider = None
  Meteor = None
  Rouge = None
  tokenize = None
import numpy as np
from pycocotools import mask as mask_api
from scenic.model_lib.base_models import box_utils
from scenic.projects.pixel_llm import densecap_evaluator

# Evaluator without METEOR and SPICE
# This import raises an error on colab.
# pylint: disable=g-import-not-at-top
try:
  from pix2seq.metrics.coco_caption_eval import COCOEvalCap as SimpleCOCOEvalCap
except ImportError:
  SimpleCOCOEvalCap = None

import tensorflow as tf


class PointEvaluator(object):
  """Class that evaluate the point prediction."""

  def __init__(
      self, dataset_name: Optional[str] = '', step: Optional[int] = None
  ):
    del dataset_name, step
    self.results = []
    self._num_examples_added = 0

  def add_example(self, prediction: Any, target: Dict[str, np.ndarray]):
    """Compute MSE."""

    self._num_examples_added += 1
    # [num_caps, max_text_tokens, num_gt_points, 2]
    gt_coords = target['points']
    # [num_caps, max_text_tokens]
    valid_token_mask = target.get(
        'token_padding_mask', target['text_tokens'] > 0
    )
    valid_token_mask *= gt_coords.max(axis=(-2, -1)) > 0

    # [num_caps, max_text_tokens, 2]
    # or [num_caps, max_text_tokens, num_pred_points, 2]
    pred_coords = prediction['point_coords']
    if pred_coords.ndim == 3:
      # [num_caps, max_text_tokens, 1, 2]
      pred_coords = pred_coords.reshape(gt_coords.shape[:-2] + (1, 2))

    # normalize coords
    height, width = target['size']
    gt_coords = gt_coords / np.array([width, height])
    pred_coords = pred_coords / np.array([width, height])

    # [num_caps, max_tokens, num_pred_points, num_gt_points]
    dist = np.mean(
        np.abs(
            np.expand_dims(pred_coords, axis=3)
            - np.expand_dims(gt_coords, axis=2)
        ),
        axis=-1,
    )
    # only count the dist to the closest GT
    # [num_caps, max_tokens, num_pred_points]
    dist = np.min(dist, axis=-1).mean(axis=-1)

    # [num_caps, max_tokens]
    dist *= valid_token_mask
    error = dist.sum() / (valid_token_mask.sum() + 1e-8)

    self.results.append(error)

  def __len__(self):
    return self._num_examples_added

  def clear(self):
    self.results = []
    self._num_examples_added = 0

  def compute_metrics(
      self,
      save_dir: str,
      clear_annotations: Optional[bool] = True,
      skip_evaluate=False,
      ):
    del save_dir, skip_evaluate
    result = np.array(self.results).mean()
    if clear_annotations:
      self.clear()
    return {'point_l1': result}


class CaptionEvaluator(object):
  """Class that feeds model outputs to COCO caption evaluation api."""

  def __init__(
      self, annotations_loc, eval_meteor_spice=False, step: Optional[int] = None
  ):
    self.annotations_loc = annotations_loc
    logging.info('Initializing evaluator.')
    if self.annotations_loc:
      logging.info('Loading annotations from %s.', self.annotations_loc)
      self.coco = COCOCaption(self.annotations_loc)
    self.annotations = {
        'images': [],
        'annotations': [],
        'type': 'captions',
        'info': {},
        'licenses': [],
        'categories': [{'id': 1, 'name': 'object'}],
    }
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
        are botoh in shape of `[num_objects, num_classes]`. Box coordinates are
        absolute values in the input image coordinates. We need to scale them
        back to the original image coordinates using information in target.
      target: Target dictionary with keys and 'image/id'.
    """
    if isinstance(prediction, dict):
      pred_caption = prediction['caption']
    else:
      pred_caption = prediction
    self._num_examples_added += 1
    id_key = 'image_id'
    empty_gt = False
    if self.annotations_loc:
      # we will use image_id that matches the annotation file
      img_id = int(target['image/id'])
    else:
      # we will create image_id on the fly
      img_id = self._num_examples_added
      if img_id not in self.gt_image_set:
        # avoid adding the same image twice due to repeated sampling.
        self.annotations['images'].append({'id': img_id})
        for x in target['captions']:
          # NOTE: if there is no gt  but pred for some images, coco raise error
          # we use `empty_gt` to mark these kind of images and ignore them
          if x:  # remove empty captions from padding.
            self._num_captions_added += 1
            self.annotations['annotations'].append(
                {'id': self._num_captions_added, id_key: img_id, 'caption': x}
            )
        # NOTE: this marks even img_id will be added into gt_image_set, there
        # is no gt for it, since it's filtered out above
        empty_gt = sum(len(t) for t in target['captions']) == 0
      self.gt_image_set.add(img_id)
    single_prediction = {
        id_key: img_id,
        'caption': pred_caption,
    }
    if img_id not in self.pred_image_set:
      if empty_gt:
        logging.warn('Image %s does not have any ground truth caption', img_id)
      else:
        self.predictions.append(single_prediction)
    else:
      logging.warn('Duplicate image %s not being added again', img_id)
    self.pred_image_set.add(img_id)

  def compute_metrics(
      self,
      save_dir: str,
      clear_annotations: Optional[bool] = True,
      skip_evaluate=False,
  ):
    """Computes the metrics for all added predictions."""
    json_file_path = self.write_pred_annotations_to_file(save_dir)
    if skip_evaluate:
      return {}
    if not self.annotations_loc:
      gt_file_path = self.write_pred_annotations_to_file(
          save_dir, is_groundtruth=True
      )
      self.coco = COCOCaption(gt_file_path)
    coco_res = self.coco.loadRes(json_file_path)
    evaluator_class = (
        COCOEvalCap if (self.eval_meteor_spice) else SimpleCOCOEvalCap
    )
    coco_eval = evaluator_class(self.coco, coco_res)
    coco_eval.params['image_id'] = coco_res.getImgIds()
    coco_eval.evaluate()
    results = coco_eval.eval
    if clear_annotations:
      self.clear()
    return results

  def clear(self):
    self.predictions = []
    self.pred_image_set = set()
    self._num_examples_added = 0
    self._num_captions_added = 0

  def __len__(self):
    return self._num_examples_added

  def write_pred_annotations_to_file(
      self, path: str, is_groundtruth: bool = False
  ):
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
              default=_convert_to_serializable,
          )
      )
    logging.info('Predicted annotations are stored in %s.', json_file_path)
    return json_file_path


def rescale_and_convert_boxes_to_xywh(boxes, input_size, orig_size):
  """Rescale boxes, and convert format to xywh."""
  h, w = orig_size
  input_h, input_w = np.asarray(input_size)
  scale_factor = np.array([w, h, w, h]) / np.array(
      [input_w, input_h, input_w, input_h])
  boxes = boxes * scale_factor[np.newaxis, :]
  boxes = np.maximum(boxes, 0)
  boxes[:, [0, 2]] = np.minimum(boxes[:, [0, 2]], w)
  boxes[:, [1, 3]] = np.minimum(boxes[:, [1, 3]], h)
  boxes[:, 2] -= boxes[:, 0]
  boxes[:, 3] -= boxes[:, 1]

  return boxes


def rescale_and_encode_masks(
    masks, input_size, padded_size, orig_size, mask_threshold
):
  """Rescale masks, and encode into COCO format."""
  input_h, input_w = input_size
  padded_h, padded_w = padded_size
  h, w = orig_size
  out_masks = []
  for mask in masks:
    mask_h, mask_w = mask.shape
    mask_input_h = int(input_h * (mask_h / padded_h))
    mask_input_w = int(input_w * (mask_w / padded_w))

    mask = (
        cv2.resize(
            mask[:mask_input_h, :mask_input_w],
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
        > mask_threshold
    )
    out_masks.append(mask_api.encode(
        np.asfortranarray(mask)
    ))

  return out_masks


def polygons_to_bitmask(
    polygons: List[np.ndarray], height: int, width: int
) -> np.ndarray:
  """Converts polygons to bitmask.

  Reference:
  https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/masks.py#L22

  Args:
    polygons(list[ndarray]): each array has shape (Nx2,)
    height(int):
    width(int):

  Returns:
    ndarray: a bool mask of shape (height, width)
  """
  if not len(polygons):
    # COCOAPI does not support empty polygons
    return np.zeros((height, width)).astype(bool)
  rles = mask_api.frPyObjects(polygons, height, width)
  rle = mask_api.merge(rles)
  return mask_api.decode(rle).astype(bool)


def decode_to_mask(segm, image_size):
  """Converts segmentation to mask."""
  if isinstance(segm, list):
    # polygon
    mask = polygons_to_bitmask(segm, *image_size)
  elif isinstance(segm, dict):
    # COCO RLE
    mask = mask_api.decode(segm)
  elif isinstance(segm, np.ndarray):
    assert (
        segm.ndim == 2
    ), 'Expect segmentation of 2 dimensions, got {}.'.format(segm.ndim)
    # mask array
    mask = segm
  else:
    raise ValueError(
        "Cannot convert segmentation of type '{}' to BitMasks!"
        'Supported types are: polygons as list[list[float] or ndarray],'
        ' COCO-style RLE as a dict, or a binary segmentation mask '
        ' in a 2D numpy array of shape HxW.'.format(type(segm))
    )
  return mask


def mask_to_box(mask):
  """Converts mask to box."""
  boxes = np.zeros((4,), dtype=np.float32)
  x_any = np.any(mask, axis=0)
  y_any = np.any(mask, axis=1)
  x = np.where(x_any)[0]
  y = np.where(y_any)[0]
  if len(x) and len(y):
    boxes = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float32)

  return boxes


class RefCocoEvaluator(object):
  """Class that evaluates the RefCOCO.

    Reference: https://github.com/ashkamath/mdetr/blob/main/datasets/refexp.py

  """

  def __init__(
      self,
      dataset_name: str,
      annotations_loc: str,
      k=(1,),
      iou_threshold=0.5,
      step: Optional[int] = None,
  ):
    self.dataset_name = dataset_name
    self.annotations_loc = annotations_loc
    if self.annotations_loc:
      logging.info('Loading refer annotations from %s.', self.annotations_loc)
      self.annotations = json.load(tf.io.gfile.GFile(self.annotations_loc))
    else:
      self.annotations = {
          'images': [],
          'annotations': [],
          'type': 'refer',
          'info': {},
          'licenses': [],
          'categories': [{'id': 1, 'name': 'object'}],
      }
    self.predictions = []
    self.pred_image_set = set()
    self.gt_image_set = set()

    self.k = k
    self.iou_threshold = iou_threshold
    self.mask_threshold = 0.
    # self.results = []
    self._num_examples_added = 0
    self.step = step

  def add_example(self, prediction: Any, target: Dict[str, np.ndarray]):
    """Compute Precision."""
    boxes = prediction['detection_boxes']
    masks = prediction.get('detection_masks', None)
    boxes = rescale_and_convert_boxes_to_xywh(
        boxes, target['size'], target['orig_size']
    )
    boxes = np.asarray(boxes).tolist()
    if masks is not None:
      masks = rescale_and_encode_masks(
          masks,
          target['size'],
          target['padded_size'],
          target['orig_size'],
          self.mask_threshold,
      )
    img_id = int(target['image/id'])

    if img_id in self.pred_image_set:
      logging.warn('Duplicate image %s not being added again', img_id)
      return
    self.pred_image_set.add(img_id)

    for i in range(len(boxes)):
      refexp_id = int(target['refexp_ids'][i])
      # [4], in XYXY abs format
      pred_box = boxes[i]
      caption = target['captions'][i]
      if not refexp_id > 0:
        continue

      self._num_examples_added += 1

      single_pred = {
          'id': refexp_id,
          'image_id': img_id,
          'bbox': pred_box,
          'refexp': caption,
      }
      if masks is not None:
        single_pred['segmentation'] = masks[i]
      self.predictions.append(single_pred)

    # create annotation json
    if not self.annotations_loc and img_id not in self.gt_image_set:
      # avoid adding the same image twice due to repeated sampling.
      self.annotations['images'].append({'id': img_id})
      gt_boxes = target['boxes']
      gt_boxes = rescale_and_convert_boxes_to_xywh(
          gt_boxes, target['size'], target['orig_size']
      )
      gt_boxes = np.asarray(gt_boxes).tolist()
      for i in range(len(gt_boxes)):
        gt_box = gt_boxes[i]
        refexp_id = int(target['refexp_ids'][i])
        if not refexp_id > 0:
          continue

        caption = target['captions'][i]
        self.annotations['annotations'].append({
            'id': refexp_id,
            'image_id': img_id,
            'bbox': gt_box,
            'refexp': caption,
        })
      self.gt_image_set.add(img_id)

  def __len__(self):
    return self._num_examples_added

  def clear(self):
    self.predictions = []
    self._num_examples_added = 0
    self.pred_image_set = set()
    self.gt_image_set = set()

  def write_pred_annotations_to_file(
      self, path: str, is_groundtruth: bool = False
  ):
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
      json_file_name = f'{self.dataset_name}_{fname_app}_{self.step}.json'
    else:
      json_file_name = f'{self.dataset_name}_{fname_app}.json'
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
              default=_convert_to_serializable,
          )
      )
    logging.info('Predicted annotations are stored in %s.', json_file_path)
    return json_file_path

  def compute_metrics(
      self,
      save_dir: str,
      clear_annotations: Optional[bool] = True,
      skip_evaluate=False,
  ) -> Dict[str, Any]:
    """Computes the metrics for all added predictions."""
    self.write_pred_annotations_to_file(save_dir)
    if not self.annotations_loc:
      self.write_pred_annotations_to_file(save_dir, is_groundtruth=True)
    if skip_evaluate:
      return {}

    pred_map = {d['id']: idx for idx, d in enumerate(self.predictions)}
    # NOTE(jiaruixu): handle coco style annotation
    if 'refexp_id' in self.annotations['annotations'][0]:
      gt_anno_map = {}
      for idx, d in enumerate(self.annotations['annotations']):
        refexp_ids = d['refexp_id']
        for refexp_id in refexp_ids:
          gt_anno_map[refexp_id] = idx
    else:
      gt_anno_map = {
          d['id']: idx for idx, d in enumerate(self.annotations['annotations'])
      }
    gt_image_map = {
        d['id']: idx for idx, d in enumerate(self.annotations['images'])
    }
    eval_seg = (
        'segmentation' in self.predictions[0]
        and 'segmentation' in self.annotations['annotations'][0]
    )
    box_tp_list = []

    seg_inter_list = []
    seg_union_list = []
    seg_box_tp_list = []

    for refexp_id in pred_map:
      pred = self.predictions[pred_map[refexp_id]]
      gt_anno = self.annotations['annotations'][gt_anno_map[refexp_id]]
      # single box
      pred_box = np.array(pred['bbox']).reshape(-1, 4)
      gt_box = np.array(gt_anno['bbox']).reshape(-1, 4)
      pred_box[:, 2:4] += pred_box[:, :2]
      gt_box[:, 2:4] += gt_box[:, :2]

      box_iou, _ = box_utils.box_iou(pred_box, gt_box, np_backbone=np)
      for k in self.k:
        box_tp_list.append(max(box_iou[:k]) > self.iou_threshold)
      if eval_seg:
        gt_image = self.annotations['images'][gt_image_map[gt_anno['image_id']]]
        image_size = (gt_image['height'], gt_image['width'])
        pred_mask = decode_to_mask(pred['segmentation'], image_size)
        gt_mask = decode_to_mask(gt_anno['segmentation'], image_size)
        cur_inter = (pred_mask & gt_mask).sum()
        cur_union = (pred_mask | gt_mask).sum()
        seg_inter_list.append(cur_inter)
        seg_union_list.append(cur_union)

        pred_seg_box = mask_to_box(pred_mask).reshape(-1, 4)

        seg_box_iou, _ = box_utils.box_iou(pred_seg_box, gt_box, np_backbone=np)
        for k in self.k:
          seg_box_tp_list.append(max(seg_box_iou[:k]) > self.iou_threshold)

    # compute mean over all refexp
    box_tp = (
        np.array(box_tp_list).reshape(len(pred_map), len(self.k)).mean(axis=0)
    )
    metrics = {
        f'box_Precision@{k}': result for k, result in zip(self.k, box_tp)
    }

    if eval_seg:
      # compute mean over all refexp
      seg_box_tp = (
          np.array(seg_box_tp_list)
          .reshape(len(pred_map), len(self.k))
          .mean(axis=0)
      )

      metrics.update(
          {
              f'seg_box_Precision@{k}': result
              for k, result in zip(self.k, seg_box_tp)
          }
      )

      seg_inter_list = np.array(seg_inter_list)
      seg_union_list = np.array(seg_union_list)

      metrics['seg_cIoU'] = seg_inter_list.mean() / (
          seg_union_list.mean() + 1e-5
      )
      metrics['seg_gIoU'] = (seg_inter_list / (seg_union_list + 1e-5)).mean()
      metrics['seg_AP'] = (
          (seg_inter_list / (seg_union_list + 1e-5)) > self.iou_threshold
      ).mean()

    if clear_annotations:
      self.clear()
    return metrics


class DensecapEvaluator(object):
  """DensecapEvaluator wrapper."""

  def __init__(self, dataset_name: str, annotations_loc, eval_meteor=True,
               ignore_empty_string=True,
               step: Optional[int] = None):
    self.dataset_name = dataset_name
    self.step = step
    self.evaluator = densecap_evaluator.DensecapEval(
        annotations_loc, eval_meteor=eval_meteor,
        ignore_empty_string=ignore_empty_string)
    self.predictions = []
    self._num_examples_added = 0
    self.pred_image_set = set()

  def add_example(self, prediction: Any, target: Dict[str, np.ndarray]):
    """Add prediction of a single image to the evaluator.

    Args:
      prediction: Model prediction tuple of 4 arrays: boxes, scores, classes,
        captions. 'boxes' is in shape of `[num_objects, 4]` and 'pred_boxes',
        'classes' are botoh in shape of `[num_objects, num_classes]`. 'captions'
        is a list of strings. Box coordinates are absolute values in the input
        image coordinates. We need to scale them back to the original image
        coordinates using information in target.
      target: Target dictionary with keys 'orig_size', 'size', and 'image/id'.
    """
    boxes = prediction['detection_boxes']
    scores = prediction['detection_scores']
    captions = prediction['captions']

    boxes = rescale_and_convert_boxes_to_xywh(
        boxes, target['size'], target['orig_size']
    )
    boxes = np.asarray(boxes).tolist()
    img_id = int(target['image/id'])

    if img_id in self.pred_image_set:
      logging.warn('Duplicate image %s not being added again', img_id)
      return
    self.pred_image_set.add(img_id)

    for bbox, score, caption in zip(
        boxes, scores, captions):
      single_classification = {
          'image_id': img_id,
          'category_id': 0,
          'bbox': bbox,
          'score': score,
          'caption': caption,
      }
      self.predictions.append(single_classification)
    self._num_examples_added += 1

# pytype: disable=signature-mismatch
  def compute_metrics(
      self,
      save_dir: str,
      clear_annotations: Optional[bool] = True,
      skip_evaluate=False,
  ) -> Dict[str, Any]:
# pytype: enable=signature-mismatch
    """Computes the metrics for all added predictions."""
    if self.step:
      fname_app = f'{self.dataset_name}_{self.step}.json'
    else:
      fname_app = f'{self.dataset_name}.json'
    self.write_pred_annotations_to_file(save_dir, fname_app=fname_app)
    if skip_evaluate:
      return {}
    results = self.evaluator.compute_metrics(self.predictions)
    if clear_annotations:
      self.clear()
    return results

  def clear(self):
    self.predictions = []
    self._num_examples_added = 0
    self.pred_image_set = set()

  def __len__(self):
    return self._num_examples_added

  def write_pred_annotations_to_file(self,
                                     path: str,
                                     fname_app: Optional[str] = None):
    """Writes predictions to file in JSON format.

    Args:
      path: Path to write the prediction annotation JSON file.
      fname_app: Optional string to append to the file name.
    """
    if not tf.io.gfile.exists(path):
      tf.io.gfile.makedirs(path)
    json_file_name = f"predictions{fname_app if fname_app else ''}.json"
    json_file_path = os.path.join(path, json_file_name)

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
              self.predictions,
              default=_convert_to_serializable))
    logging.info('Predicted annotations are stored in %s.', json_file_path)


class LocaEvaluator(object):
  """Location-conditioned Caption wrapper."""
  merge_gt_boxes_iou = 0.7

  def __init__(self, dataset_name: str,
               step: Optional[int] = None,
               merge_gt_boxes: Optional[bool] = False,
               meteor_jar_path: Optional[str] = None,
               java_jre_path: Optional[str] = None):
    self.dataset_name = dataset_name
    self.merge_gt_boxes = merge_gt_boxes
    self.step = step
    self.predictions = []
    self._num_examples_added = 0
    self._num_captions_added = 0
    self.pred_image_set = set()
    self.meteor_jar_path = meteor_jar_path
    self.java_jre_path = java_jre_path
    self.annotations = {
        'images': [],
        'annotations': [],
        'type': 'captions',
        'info': {},
        'licenses': [],
        'categories': [{'id': 1, 'name': 'object'}],
    }

  @staticmethod
  def merge_gt_anno(gts, iou_thresh, is_gt=True):
    """VG ground truth are overlaping. We need to merge them before evaluating.

    Original code:
      github.com/jcjohnson/densecap/blob/maste*/densecap/box_utils.lua#L590
      github.com/jcjohnson/densecap/blob/maste*/eval/eval_utils.lua#L105

    Args:
      gts: gts of a single image. list of dicts, each with the following keys:
        'bbox': list of 4 floats in order (l, t, w, h)
        'caption': a string.
        ...
      iou_thresh: float
      is_gt: bool
    Returns:
      new_gts: list of dicts. Might have different length from the input.
        'bbox': list of 4 floats in order (l, t, w, h)
        'captions': list of strings.
    """
    new_gts = []
    if not gts:
      return new_gts
    gt_boxes = np.asarray([x['bbox'] for x in gts], dtype=np.float32)
    ious, _ = box_utils.box_iou(gt_boxes, gt_boxes, np_backbone=np)  # N x N

    while True:
      can_merge = ious >= iou_thresh
      # Find the largest cluster and merge it.
      num_merges = can_merge.sum(axis=1)  # N
      ind = np.argmax(num_merges)  # int
      if num_merges[ind] == 0:
        break
      merge_inds = np.nonzero(can_merge[ind])[0]
      new_box = gt_boxes[merge_inds].mean(axis=0)
      all_captions = [gts[x]['caption'].replace('\n', '') for x in merge_inds]
      for merge_ind in merge_inds:
        if is_gt:
          new_gt = {
              'bbox': new_box,
              'captions': all_captions,
              'id': gts[merge_ind]['id'],
          }
        else:
          new_gt = {
              'bbox': new_box,
              'caption': gts[merge_ind]['caption'],
              'id': gts[merge_ind]['id'],
          }
        new_gts.append(new_gt)
      ious[merge_inds, :] = 0.0
      ious[:, merge_inds] = 0.0
    return new_gts

  def add_example(self, prediction: Any, target: Dict[str, np.ndarray]):
    """Add prediction of a single image to the evaluator.

    Args:
      prediction: Model prediction tuple of 4 arrays: boxes, scores, classes,
        captions. 'boxes' is in shape of `[num_objects, 4]` and 'pred_boxes',
        'classes' are botoh in shape of `[num_objects, num_classes]`. 'captions'
        is a list of strings. Box coordinates are absolute values in the input
        image coordinates. We need to scale them back to the original image
        coordinates using information in target.
      target: Target dictionary with keys 'orig_size', 'size', and 'image/id'.
    """
    captions = prediction['captions']
    boxes = prediction['detection_boxes']
    gt_captions = target['captions']
    gt_boxes = target['boxes']

    boxes = rescale_and_convert_boxes_to_xywh(
        boxes, target['size'], target['orig_size']
    )
    boxes = np.asarray(boxes).tolist()
    gt_boxes = rescale_and_convert_boxes_to_xywh(
        gt_boxes, target['size'], target['orig_size']
    )
    gt_boxes = np.asarray(gt_boxes).tolist()
    assert len(boxes) == len(captions)
    assert len(gt_boxes) == len(boxes)
    assert len(gt_captions) == len(captions)

    img_id = int(target['image/id'])

    if img_id in self.pred_image_set:
      logging.warn('Duplicate image %s not being added again', img_id)
      return
    self.pred_image_set.add(img_id)
    self.annotations['images'].append({'id': self._num_captions_added})

    cur_preds = []
    cur_annos = []
    for caption, box, gt_caption, gt_box in zip(
        captions, boxes, gt_captions, gt_boxes
    ):
      if max(gt_box) <= 0:
        continue
      single_classification = {
          'image_id': img_id,
          'id': self._num_captions_added,
          'category_id': 0,
          'bbox': box,
          'caption': caption,
      }
      single_annotation = {
          'image_id': img_id,
          'id': self._num_captions_added,
          'category_id': 0,
          'bbox': gt_box,
          'caption': gt_caption,
      }
      # self.annotations['annotations'].append(single_annotation)
      # self.predictions.append(single_classification)
      cur_preds.append(single_classification)
      cur_annos.append(single_annotation)
      self._num_captions_added += 1
    if self.merge_gt_boxes:
      cur_preds = self.merge_gt_anno(
          cur_preds, self.merge_gt_boxes_iou, is_gt=False
      )
      cur_annos = self.merge_gt_anno(cur_annos, self.merge_gt_boxes_iou)
    self.predictions.extend(cur_preds)
    self.annotations['annotations'].extend(cur_annos)
    self._num_examples_added += 1

  # pytype: disable=signature-mismatch
  def compute_metrics(
      self,
      save_dir: str,
      clear_annotations: Optional[bool] = True,
      skip_evaluate=False,
  ) -> Dict[str, Any]:
    # pytype: enable=signature-mismatch
    """Computes the metrics for all added predictions."""
    if self.step:
      fname_app = f'{self.dataset_name}_{self.step}.json'
    else:
      fname_app = f'{self.dataset_name}.json'
    self.write_pred_annotations_to_file(save_dir, fname_app=fname_app)
    if skip_evaluate:
      return {}
    res = {}
    gts = {}
    for pred in self.predictions:
      if 'captions' in pred:
        res[pred['id']] = [{'caption': c} for c in pred['captions']]
      else:
        res[pred['id']] = [pred]
    for anno in self.annotations['annotations']:
      if 'captions' in anno:
        gts[anno['id']] = [{'caption': c} for c in anno['captions']]
      else:
        gts[anno['id']] = [anno]

    res = tokenize(res)
    gts = tokenize(gts)

    scorers = [
        (Rouge(), 'ROUGE_L'),
        (Cider(), 'CIDEr'),
        (Bleu(), 'BLEU-4'),
        (Meteor(), 'Meteor'),
    ]
    results = {}
    for scorer, method in scorers:
      logging.info('computing %s score...', scorer.method())
      score, _ = scorer.compute_score(gts, res)
      results[method] = score
    if clear_annotations:
      self.clear()
    return results

  def clear(self):
    self.predictions = []
    self._num_examples_added = 0
    self._num_captions_added = 0
    self.pred_image_set = set()

  def __len__(self):
    return self._num_examples_added

  def write_pred_annotations_to_file(self,
                                     path: str,
                                     fname_app: Optional[str] = None):
    """Writes predictions to file in JSON format.

    Args:
      path: Path to write the prediction annotation JSON file.
      fname_app: Optional string to append to the file name.
    """
    if not tf.io.gfile.exists(path):
      tf.io.gfile.makedirs(path)
    json_file_name = f"predictions{fname_app if fname_app else ''}.json"
    json_file_path = os.path.join(path, json_file_name)

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
              self.predictions,
              default=_convert_to_serializable))
    logging.info('Predicted annotations are stored in %s.', json_file_path)
