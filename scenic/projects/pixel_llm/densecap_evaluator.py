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

"""Evaluator for dense object caption.

Modified from scenic.projects.baseline.detr.train_utils.DetrGlobalEvaluator,
but with a different underlying evaluator.
"""
import copy
import json
import logging

from absl import logging
import numpy as np
from coco_caption.meteor import Meteor
import tensorflow as tf


def box_iou(boxes1, boxes2):
  """Compute box IoU. Boxes in format [l, t, w, h].

  Args:
    boxes1: array in shape n x 4
    boxes2: array in shape m x 4
  Returns:
    iou: array in shape n x m
    union: array in shape n x m
  """
  wh1 = boxes1[:, 2:]
  wh2 = boxes2[:, 2:]
  area1 = wh1[:, 0] * wh1[:, 1]  # [n]
  area2 = wh2[:, 0] * wh2[:, 1]  # [m]
  lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [n, m, 2]
  rb = np.minimum(
      boxes1[:, None, 2:] + boxes1[:, None, :2],
      boxes2[None, :, 2:] + boxes2[None, :, :2])  # [n, m, 2]
  wh = (rb - lt).clip(0.0)  # [n, m, 2]
  intersection = wh[:, :, 0] * wh[:, :, 1]  # [n, m]
  union = area1[:, None] + area2[None, :] - intersection  # [n, m]
  iou = np.where(union > 0, intersection / union, 0)
  return iou


class DensecapEval(object):
  """Evaluator for dense caption.

  This class reproduce the official evaluation for dense caption:
    https://github.com/jcjohnson/densecap/blob/maste*/eval/eval_utils.lua
  """
  merge_gt_boxes_iou = 0.7
  iou_threshs = (0.3, 0.4, 0.5, 0.6, 0.7)
  meteor_threshs = (-1, 0, 0.05, 0.1, 0.15, 0.2, 0.25)
  meteor_jar_path = None
  java_jre_path = None

  def __init__(
      self,
      annotations_loc,
      merge_gt_boxes=True,
      eval_meteor=True,
      ignore_empty_string=True,
      eval_in_720=False,
      score_key='score',
      meteor_jar_path=None,
      java_jre_path=None,
  ):
    self.ignore_empty_string = ignore_empty_string
    self.eval_meteor = eval_meteor
    self.score_key = score_key
    if isinstance(annotations_loc, str):
      self.dataset = json.load(tf.io.gfile.GFile(annotations_loc, 'r'))
    else:
      self.dataset = annotations_loc
    self.image_ids = set([x['id'] for x in self.dataset['images']])
    self.gts = {x: [] for x in self.image_ids}
    self.meteor_jar_path = meteor_jar_path
    self.java_jre_path = java_jre_path

    for x in self.dataset['annotations']:
      self.gts[x['image_id']].append(x)
    if merge_gt_boxes:
      logging.info('Merging ground truth boxes...')
      num_boxes = sum(len(x) for x in self.gts.values())
      logging.info('Num boxes before merging: %d', num_boxes)
      for image_id in self.image_ids:
        self.gts[image_id] = self.merge_gt_boxes(
            self.gts[image_id], self.merge_gt_boxes_iou)
      num_boxes = sum(len(x) for x in self.gts.values())
      self.num_boxes_after_merging = num_boxes
      logging.info('Num boxes after merging: %d', num_boxes)
    if eval_in_720:
      for image_info in self.dataset['images']:
        image_id = image_info['id']
        height, width = image_info['height'], image_info['width']
        for x in self.gts[image_id]:
          x['bbox'] = (np.asarray(x['bbox'], dtype=np.float32) * 720 / max(
              height, width)).tolist()
    self.dts = {x: [] for x in self.image_ids}

  def compute_metrics(self, predictions):
    """Evaluate metrics.

    Args:
      predictions: list of dict. Each dict is a prediction of an *instance*,
        with keys 'image_id', 'bbox', 'caption', 'score'.

    Returns:
      results: a dict of string (metric name) to float.
    """
    predictions = copy.deepcopy(predictions)
    all_dts = {x['id']: [] for x in self.dataset['images']}
    for x in predictions:
      all_dts[x['image_id']].append(x)
    records = []
    logging.info('Computing metrics...')
    logging.info('ignore_empty_string %s', self.ignore_empty_string)
    for image_id in self.image_ids:
      dts = sorted(all_dts[image_id], key=lambda x: -x[self.score_key])
      gts = self.gts[image_id]
      dt_boxes = np.asarray([x['bbox'] for x in dts]).reshape(-1, 4)
      gt_boxes = np.asarray([x['bbox'] for x in gts]).reshape(-1, 4)
      ious = box_iou(dt_boxes, gt_boxes)
      gt_used = np.zeros(len(gts), dtype=bool)
      for i, dt in enumerate(dts):
        # Unlike COCO mAP evaluation, the official densecap evaluation does not
        # find the best "available" gt, but directly returns the best IoU gt.
        if len(ious[i]) > 0:  # pylint: disable=g-explicit-length-test
          max_iou = np.max(ious[i])
          matched_gt_ind = np.argmax(ious[i])
          matched_caps = gts[matched_gt_ind]['captions']
        else:
          max_iou = -1
          matched_gt_ind = -1
          matched_caps = ['']
        matched = max_iou > 0 and not gt_used[matched_gt_ind]
        if matched:
          gt_used[matched_gt_ind] = True
          if self.ignore_empty_string and '' in matched_caps:
            dt['caption'] = 'EMPTY'
            matched_caps = ['EMPTY']
        record = {
            'matched': matched,
            'iou': max_iou,
            'candidate': [dt['caption']],
            'references': matched_caps,
            'image_id': image_id,
            'score': dt[self.score_key],
        }
        records.append(record)

    if not self.ignore_empty_string:
      records = [x for x in records if x['candidate'][0] != 'EMPTY']
      num_pos = sum(len(
          [xx for xx in x if '' not in xx['captions']]
          ) for x in self.gts.values())
    else:
      num_pos = sum(len(x) for x in self.gts.values())
    records = sorted(records, key=lambda x: -x['score'])
    references = {i: x['references'] for i, x in enumerate(records)}
    candidate = {i: x['candidate'] for i, x in enumerate(records)}
    num_preds = len(records)

    if self.eval_meteor:
      logging.info('Computing METEOR...')
      meteor_evaluator = Meteor(
          meteor_jar_path=self.meteor_jar_path, java_jre_path=self.java_jre_path
      )
      _, meteor_scores = meteor_evaluator.compute_score(references, candidate)
      meteor_threshs = self.meteor_threshs
    else:
      meteor_scores = np.ones(num_preds, dtype=np.float32)
      meteor_threshs = (-1,)

    detection_results, results = {}, {}
    logging.info('Accumulating results...')
    for iou_thresh in self.iou_threshs:
      for meteor_thresh in meteor_threshs:
        tp = np.zeros(num_preds, dtype=np.float32)
        fp = np.zeros(num_preds, dtype=np.float32)
        for i, record in enumerate(records):
          if not record['references']:
            fp[i] = 1
          else:
            if record['matched'] and (record['iou'] >= iou_thresh) and (
                meteor_scores[i] > meteor_thresh):
              tp[i] = 1
            else:
              fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        rec = tp / num_pos
        prec = tp / (tp + fp)
        ap = 0
        for t in range(100):
          mask = rec >= t / 100.
          prec_masked = prec * mask
          if len(prec_masked) > 0:  # pylint: disable=g-explicit-length-test
            p = prec_masked.max()
          else:
            p = 0.
          ap += p
        ap = ap / 100.
        if meteor_thresh < 0:
          detection_results[f'mAP_detection_iou{iou_thresh:.1f}'] = ap
        else:
          results[f'mAP_iou{iou_thresh:.1f}_meteor{meteor_thresh:.2f}'] = ap
    if self.eval_meteor:
      results['mAP'] = sum(results.values()) / len(results)
      print('mAP', results['mAP'])
    results['mAP_detection'] = sum(
        detection_results.values()) / len(detection_results)
    results.update(detection_results)
    print('mAP_detection', results['mAP_detection'])

    return results

  @staticmethod
  def merge_gt_boxes(gts, iou_thresh):
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
    Returns:
      new_gts: list of dicts. Might have different length from the input.
        'bbox': list of 4 floats in order (l, t, w, h)
        'captions': list of strings.
    """
    new_gts = []
    if not gts:
      return new_gts
    gt_boxes = np.asarray([x['bbox'] for x in gts], dtype=np.float32)
    ious = box_iou(gt_boxes, gt_boxes)  # N x N

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
      new_gt = {'bbox': new_box, 'captions': all_captions}
      if 'track_id' in gts[merge_inds[0]]:
        new_gt['track_id'] = gts[merge_inds[0]]['track_id']
      new_gts.append(new_gt)
      ious[merge_inds, :] = 0.
      ious[:, merge_inds] = 0.
    return new_gts
