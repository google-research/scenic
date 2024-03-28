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

"""Tools for evaluating dense captions.

Reimplements evaluation metrics that agree with open-sourced methods at
https://github.com/ranjaykrishna/densevid_eval/blob/master/evaluate.py
"""

import collections
import logging
import random
import re
import string

import numpy as np
from scenic.projects.vid2seq.metrics.cider import Cider
from scenic.projects.vid2seq.metrics.meteor import Meteor
from scenic.projects.vid2seq.metrics.ptbtokenizer import PTBTokenizer


def convert_uint8_array_to_string(uint8_array):
  return uint8_array.tobytes().rstrip(b'\x00').decode('utf-8')


def convert_strings_to_uint8_arrays(str_tensor, max_str_len=None):
  """Convert string numpy array into uint8 arrays to transfer to TPUs.

  Given the input string array, outputs a uint8 tensor with an additional
  dimension at the end with the size of max_str_len.

  Args:
    str_tensor: The input string array.
    max_str_len: The maximum number of characters to keep in the converted uint8
      array. If None, it is set to the longest string length in the input array.

  Returns:
    Converted uint8 numpy array with an additional dim of size max_str_len.
  """
  # Make sure that the input str_tensor is an np.ndarray of bytes not of object.
  # An object array stores pointers only whereas a bytes array stores actual
  # string bytes
  str_tensor = np.array(str_tensor, dtype=bytes)
  uint8_tensor = np.frombuffer(str_tensor,
                               np.uint8).reshape(str_tensor.shape + (-1,))
  if max_str_len:
    to_pad = max(0, max_str_len - uint8_tensor.shape[-1])
    uint8_tensor = np.pad(uint8_tensor[..., :max_str_len],
                          [[0, 0]] * str_tensor.ndim + [[0, to_pad]])

  return uint8_tensor


def random_string(string_length):
  """Random string generator for unmatched captions."""
  letters = string.ascii_lowercase
  return ''.join(random.choice(letters) for i in range(string_length))


def chased_dp_assignment(scores):
  """Run dp matching as https://github.com/fujiso/SODA/blob/master/soda.py."""

  m, n = scores.shape
  dp = - np.ones((m, n))
  path = np.zeros((m, n))

  def transition(i, j):
    if dp[i, j] >= 0:
      return dp[i, j]
    elif i == 0 and j == 0:
      state = [-1, -1, scores[i, j]]
    elif i == 0:
      state = [-1, transition(i, j-1), scores[i, j]]
    elif j == 0:
      state = [transition(i-1, j), -1, scores[i, j]]
    else:
      state = [
          transition(i - 1, j),
          transition(i, j - 1),
          transition(i - 1, j - 1) + scores[i, j]
      ]
    dp[i, j] = np.max(state)
    path[i, j] = np.argmax(state)
    return dp[i, j]

  def get_pairs(i, j):
    p = np.where(path[i][:j+1] == 2)[0]
    # pylint: disable=g-explicit-length-test
    if i != 0 and not len(p):
      return get_pairs(i-1, j)
    elif i == 0 or p[-1] == 0:
      return [(i, p[-1])]
    else:
      return get_pairs(i-1, p[-1]-1) + [(i, p[-1])]
  n, m = scores.shape
  max_score = transition(n-1, m-1)
  pairs = get_pairs(n-1, m-1)
  return max_score, pairs


def iou(interval_1, interval_2):
  """Compute the IOU between two intervals.

  Args:
    interval_1: A tuple (start, end) containing the first interval.
    interval_2: A tuple (start, end) containing the second interval.

  Returns:
    The IOU of the two intervals.
  """
  start_1, end_1 = min(*interval_1), max(*interval_1)
  start_2, end_2 = min(*interval_2), max(*interval_2)

  intersection = max(0, min(end_1, end_2) - max(start_1, start_2))
  union = min(
      max(end_1, end_2) - min(start_1, start_2),
      end_1 - start_1 + end_2 - start_2)
  result = float(intersection) / (union + 1e-8)
  return result


def evaluate_detections(predicted_segments,
                        gt_segments,
                        splits,
                        iou_thresholds=(0.3, 0.5, 0.7, 0.9)):
  """Compute the mean P/R between the predicted and ground truth segments.

  Args:
    predicted_segments: A numpy array of shape [K x 2] containing the predicted
      segments.
    gt_segments: A numpy array of shape [S x 2] containing the ground truth
      segments.
    splits: A numpy array of shape [S] indicating the annotation set.
    iou_thresholds: The IOU thresholds to use for Precision/Recall calculations.

  Returns:
    precision: The mean precision of the predictions over the IOU thresholds.
    recall: The mean recall of the predictions over the IOU thresholds.
    best_miou: The mIoU.
    iou_matrices: dictionary mapping each split to the corresponding iou matrix.
  """
  # Recall is the percentage of ground truth that is covered by the predictions.
  # Precision is the percentage of predictions that are valid.

  best_recall = []
  best_precision = []
  iou_matrices = {}

  predicted_shape = predicted_segments.shape[0]

  for split in set(splits):
    metrics = {}
    for threshold in iou_thresholds:
      metrics[str(threshold)] = {
          'gt_covered': set(),
          'pred_covered': set(),
      }
    split_idx = np.where(splits == split)[0]
    split_gt_segments = np.array([gt_segments[idx] for idx in split_idx])

    gt_shape = split_gt_segments.shape[0]

    # Compute the IOUs for the segments.
    iou_matrix = np.zeros((gt_shape, max(predicted_shape, 1)))
    for idx_g, gt_segment in enumerate(split_gt_segments):
      cur_max_iou = 0
      for idx_p, segment in enumerate(predicted_segments):
        sample_iou = iou(segment, gt_segment)
        iou_matrix[idx_g, idx_p] = sample_iou
        cur_max_iou = max(cur_max_iou, sample_iou)
        for threshold in iou_thresholds:
          if sample_iou > threshold:
            metrics[str(threshold)]['pred_covered'].add(idx_p)
            metrics[str(threshold)]['gt_covered'].add(idx_g)

    # Compute the precisions and recalls for each IOU threshold.
    for threshold, m in metrics.items():
      pred_covered = m['pred_covered']
      gt_covered = m['gt_covered']

      # Avoid dividing by 0 for precision
      m['precision'] = float(len(pred_covered)) / max(
          float(predicted_shape), 1.0)
      m['recall'] = float(len(gt_covered)) / float(gt_shape)

    precision = [m['precision'] for m in metrics.values()]
    recall = [m['recall'] for m in metrics.values()]
    if best_precision:
      best_precision = [
          max(precision[i], best_precision[i]) for i in range(len(precision))
      ]
      best_recall = [max(recall[i], best_recall[i]) for i in range(len(recall))]
    else:
      best_precision, best_recall = precision, recall
    iou_matrices[int(split)] = iou_matrix

  return best_precision, best_recall, iou_matrices


def match_captions(predicted_segments,
                   gt_segments,
                   predicted_captions,
                   gt_captions,
                   iou_thresholds=(0.3, 0.5, 0.7, 0.9)):
  """Matches the predicted captions to ground truth using the IOU thresholds.

  Args:
   predicted_segments: A numpy array of shape [K x 2] containing the predicted
     segment intervals.
   gt_segments: A numpy array of shape [S x 2] containing the ground truth
     segment intervals.
   predicted_captions: A list of string of shape [K] containing the
     corresponding K predicted captions.
   gt_captions: A list of strings of shape [S] containing the corresponding S
     ground truth captions.
   iou_thresholds: A list of thresholds for IOU to average over.

  Returns:
   ground_truths_filtered: Filtered list of ground truth captions for all
    threshold.
   predictions_filtered: Matching list of predicted captions for all
    threshold.
   isxes: For each threshold, contains lists of isx of matches.
  """

  # Setup a set of dictionaries to hold the results.
  ground_truths_filtered = {str(threshold): {} for threshold in iou_thresholds}
  predictions_filtered = {str(threshold): {} for threshold in iou_thresholds}

  # Create GT lists for each of the IOU thresholds.
  isx = 0
  isxes = {str(threshold): [] for threshold in iou_thresholds}
  for idx_p, segment in enumerate(predicted_segments):
    pc_idxp = predicted_captions[idx_p]
    added = {str(threshold): False for threshold in iou_thresholds}
    for idx_g, gt_segment in enumerate(gt_segments):
      gt_idxg = gt_captions[idx_g]
      sample_iou = iou(segment, gt_segment)
      for threshold in iou_thresholds:
        if sample_iou >= threshold:
          key = str(isx)
          isxes[str(threshold)].append(isx)
          isx += 1
          ground_truths_filtered[str(threshold)][key] = [{'caption': gt_idxg}]
          predictions_filtered[str(threshold)][key] = [{'caption': pc_idxp}]
          added[str(threshold)] = True
    for threshold in iou_thresholds:
      if not added[str(threshold)]:
        key = str(isx)
        isxes[str(threshold)].append(isx)
        isx += 1
        # Set this to a random string with no match to the predictions to
        # get a zero score
        ground_truths_filtered[str(threshold)][key] = [
            {'caption': random_string(random.randint(10, 20))}
        ]
        predictions_filtered[str(threshold)][key] = [{'caption': pc_idxp}]

  return ground_truths_filtered, predictions_filtered, isxes


def evaluate_caption_scores(ground_truths_filtered,
                            predictions_filtered,
                            iou_thresholds=(0.3, 0.5, 0.7, 0.9),
                            scorers=None):
  """Compute the mean NLP metrics over the given IOU thresholds.

  Args:
   ground_truths_filtered: Filtered list of ground truth captions for each
    threshold.
   predictions_filtered: Matching list of predicted captions for each threshold.
   iou_thresholds: A list of thresholds for IOU to average over.
   scorers: A dictionary of scorers.

  Returns:
   metrics: dictionary with mean captioning score across the threshold set.
  """

  if scorers is None:
    scorers = {}

  # Compute the caption metrics.
  metrics = collections.defaultdict(list)
  for scorer_name, scorer in scorers.items():
    for threshold in iou_thresholds:
      # Handle the case where we have no overlapping truths
      if not ground_truths_filtered[str(threshold)]:
        metrics[scorer_name].append(0.0)
      elif not predictions_filtered[str(threshold)]:
        metrics[scorer_name].append(0.0)
      else:
        score = scorer.compute_score(ground_truths_filtered[str(threshold)],
                                     predictions_filtered[str(threshold)])
        score = np.nan_to_num(score[0])
        metrics[scorer_name].append(score)

  # Aggregate the caption metrics.
  for key, value in metrics.items():
    metrics[key] = np.mean(value)

  return metrics


def sodac(iou_matrices,
          scorer,
          predicted_captions,
          gt_captions,
          splits,
          iou_thresholds=(0.,)):
  """SODA_c from https://github.com/fujiso/SODA/."""
  if not predicted_captions:
    return {int(split): 0 for split in splits}

  res = {
      str(index): [p]
      for index, p in enumerate(predicted_captions)
  }
  unique_splits = set(splits)
  fs = {int(split): [0] * len(iou_thresholds) for split in unique_splits}
  for split in unique_splits:
    split_idx = np.where(splits == split)[0]
    split_gt_captions = [gt_captions[idx] for idx in split_idx]
    gts = [{index: [x]
            for index in res}
           for x in split_gt_captions]
    iou_matrix = iou_matrices[int(split)]
    score_matrix = np.array(
        [np.nan_to_num(scorer.compute_score(res, gt)[1]) for gt in gts])
    for i, threshold in enumerate(iou_thresholds):
      iou_cur = np.copy(iou_matrix)
      iou_cur[iou_cur < threshold] = 0.0
      max_score, _ = chased_dp_assignment(iou_cur * score_matrix)
      (n_g, n_p) = iou_cur.shape
      p = max_score / n_p
      r = max_score / n_g
      fs[int(split)][i] = 2 * p * r / (p + r) if p+r > 0 else 0
  for split in unique_splits:
    fs[int(split)] = np.mean(fs[int(split)])
  return fs


def evaluate_dense_captions(predicted_segments,
                            gt_segments,
                            predicted_captions,
                            gt_captions,
                            splits,
                            keys,
                            iou_thresholds=(0.3, 0.5, 0.7, 0.9),
                            soda=False,
                            tmponly=False):
  """Compute both the P/R and NLP metrics for the given predictions.

  This is the same as calling the above functions, however it aggregates the
  metrics generated by evaluate_detections and evaluate_caption_scores across
  a list of inputs.

  Args:
   predicted_segments: A list of numpy arrays, of shape [K x 2]
     containing the predicted segment intervals.
   gt_segments: A list of numpy arrays, of shape [S x 2]
     containing the ground truth segment intervals.
   predicted_captions: A list of lists, of string of shape [K]
     containing the corresponding K predicted captions.
   gt_captions: A list of lists, of strings of shape [S] containing the
     corresponding S ground truth captions.
   splits: A list of numpy arrays, of shape [S] indicating
     the annotation set (1/2 for ActivityNet).
   keys: A list of strings
   iou_thresholds: A list of thresholds for IOU to average over.
   soda: Whether to compute SODA or not.
   tmponly: In this case do not compute captioning metrics.

  Returns:
    (precision, recall): The precision and recall of the detections averaged
    over the IOU thresholds.
    metrics: The NLP metrics of the predictions averaged over the IOU
      thresholds.
  """

  # Handle if these are lists, or single samples.
  assert all([isinstance(p, list) for p in [predicted_segments, gt_segments]])
  # Only construct the scorers once, so that we don't have any issues with
  # overhead when running multiple evaluations.
  scorers = {
      'CIDER': Cider(),
      'METEOR': Meteor(),
  }
  tokenizer = PTBTokenizer()
  metric_tiou = collections.defaultdict(list)
  gts = {str(threshold): {} for threshold in iou_thresholds}
  preds = {str(threshold): {} for threshold in iou_thresholds}
  vid2isx = {str(threshold): {} for threshold in iou_thresholds}

  assert len(predicted_segments) == len(gt_segments) == len(
      predicted_captions) == len(gt_captions) == len(splits)

  # Compute matches
  for pred_seg, gt_seg, pred_cap, gt_cap, key in zip(
      predicted_segments,
      gt_segments,
      predicted_captions,
      gt_captions,
      keys,
  ):
    gt, pred, isxes = match_captions(
        pred_seg, gt_seg, pred_cap, gt_cap, iou_thresholds
    )
    # Flatten for tokenization
    for threshold in iou_thresholds:
      for k, v in gt[str(threshold)].items():
        gts[str(threshold)][key + '_' + str(k)] = v
      for k, v in pred[str(threshold)].items():
        preds[str(threshold)][key + '_' + str(k)] = v
      vid2isx[str(threshold)][key] = isxes[str(threshold)]

  # Call tokenization once
  for threshold in iou_thresholds:
    gts[str(threshold)] = tokenizer.tokenize(gts[str(threshold)])
    preds[str(threshold)] = tokenizer.tokenize(preds[str(threshold)])

  # Tokenize also the original lists for SODA computation
  predicted_captions_dict = {  # pylint: disable=g-complex-comprehension
      keys[i] + '_' + str(j): [{'caption': p}]
      for i, ps in enumerate(predicted_captions)
      for j, p in enumerate(ps)
  }
  gt_captions_dict = {  # pylint: disable=g-complex-comprehension
      keys[i] + '_' + str(j): [{'caption': g}]
      for i, gs in enumerate(gt_captions)
      for j, g in enumerate(gs)
  }
  predicted_captions_tok = tokenizer.tokenize(predicted_captions_dict)
  gt_captions_tok = tokenizer.tokenize(gt_captions_dict)
  predicted_captions_res = []
  gt_captions_res = []
  for i, ps in enumerate(predicted_captions):
    res = [
        predicted_captions_tok[keys[i] + '_' + str(j)][0]
        for j, _ in enumerate(ps)
    ]
    predicted_captions_res.append(res)
  for i, gs in enumerate(gt_captions):
    res = [gt_captions_tok[keys[i] + '_' + str(j)][0] for j, _ in enumerate(gs)]
    gt_captions_res.append(res)

  # Reshape
  final_gts = {str(threshold): {} for threshold in iou_thresholds}
  final_preds = {str(threshold): {} for threshold in iou_thresholds}
  for threshold in iou_thresholds:
    for key in keys:
      final_gts[str(threshold)][key] = {
          str(k): gts[str(threshold)][key + '_' + str(k)]
          for k in vid2isx[str(threshold)][key]
      }
      final_preds[str(threshold)][key] = {
          str(k): preds[str(threshold)][key + '_' + str(k)]
          for k in vid2isx[str(threshold)][key]
      }

  # Compute dense video captioning metrics at the video level
  for i, key in enumerate(keys):
    pred_filt_i = {str(t): final_preds[str(t)][key] for t in iou_thresholds}
    gt_filt_i = {str(t): final_gts[str(t)][key] for t in iou_thresholds}
    res = evaluate_single_dense_captions(
        predicted_segments[i],
        gt_segments[i],
        pred_filt_i,
        gt_filt_i,
        predicted_captions_res[i],
        gt_captions_res[i],
        splits[i],
        key,
        iou_thresholds,
        soda,
        tmponly,
        scorers,
    )
    for met in res:
      metric_tiou[met].append(res[met])
    if soda:
      if 'SODA_c_1' not in res:
        metric_tiou['SODA_c_1'].append(-1)
      if 'SODA_c_2' not in res:
        metric_tiou['SODA_c_2'].append(-1)

  logging.info('Closing Meteor')
  with scorers['METEOR'].lock:
    scorers['METEOR'].meteor_p.stdin.close()
    scorers['METEOR'].meteor_p.stdout.close()
    scorers['METEOR'].meteor_p.kill()
    scorers['METEOR'].meteor_p.wait()
  del scorers

  return metric_tiou


def evaluate_single_dense_captions(predicted_segments,
                                   gt_segments,
                                   predictions_filtered,
                                   ground_truths_filtered,
                                   predicted_captions,
                                   gt_captions,
                                   splits,
                                   keys,
                                   iou_thresholds=(0.3, 0.5, 0.7, 0.9),
                                   soda=False,
                                   tmponly=False,
                                   scorers=None):
  """Compute both the P/R and NLP metrics for the given predictions.

  Args:
   predicted_segments: A numpy arrays, of shape [K x 2]
     containing the predicted segment intervals.
   gt_segments: A numpy arrays, of shape [S x 2]
     containing the ground truth segment intervals.
   predictions_filtered: Matching list of predicted captions for each threshold.
   ground_truths_filtered: Filtered list of ground truth captions for each
    threshold.
   predicted_captions: A list, of string of shape [K]
     containing the corresponding K predicted captions.
   gt_captions: A list, of strings of shape [S] containing the
     corresponding S ground truth captions.
   splits: A numpy array, of shape [S] indicating
     the annotation set (1/2 for ActivityNet).
   keys: A string
   iou_thresholds: A list of thresholds for IOU to average over.
   soda: Whether to compute SODA or not.
   tmponly: In this case do not compute captioning metrics.
   scorers: dictionary mapping strings to scorers.

  Returns:
    (precision, recall): The precision and recall of the detections averaged
    over the IOU thresholds.
    metrics: The NLP metrics of the predictions averaged over the IOU
      thresholds.
  """
  if scorers is None:
    scorers = {}

  # Localization
  detection_precision, detection_recall, iou_matrices = (
      evaluate_detections(
          predicted_segments, gt_segments, splits, iou_thresholds
      )
  )

  # Captions
  n_preds = len(predicted_captions)
  if not tmponly:
    metric_tiou = evaluate_caption_scores(
        ground_truths_filtered, predictions_filtered,
        iou_thresholds, scorers)
    if soda:
      fs = sodac(iou_matrices, scorers['METEOR'],
                 predicted_captions, gt_captions, splits, (0.,))
  else:
    metric_tiou = {}

  mean_precision = sum(detection_precision) / len(detection_precision)
  mean_recall = sum(detection_recall) / len(detection_recall)
  for j, threshold in enumerate(iou_thresholds):
    metric_tiou[f'Precision@{threshold}'] = float(detection_precision[j])
    metric_tiou[f'Recall@{threshold}'] = float(detection_recall[j])
  metric_tiou['Precision_Mean'] = float(mean_precision)
  metric_tiou['Recall_Mean'] = float(mean_recall)
  metric_tiou['F1_Score'] = 2 * float(mean_recall) * float(mean_precision) / (
      float(mean_recall) + float(mean_precision)
  ) if float(mean_recall) + float(mean_precision) > 0 else 0
  if soda and not tmponly:
    for split in fs:
      metric_tiou[f'SODA_c_{split}'] = float(fs[split])
  metric_tiou['n_preds'] = n_preds
  metric_tiou['key'] = keys

  return metric_tiou


def parse_sent(sent):
  """Sentence preprocessor."""
  res = re.sub('[^a-zA-Z]', ' ', sent)
  res = res.strip().lower().split()
  return res


def evaluate_para(predicted_captions,
                  gt_captions):
  """Paragraph-level evaluation.

  Args:
   predicted_captions: A list of strings (paragraphs).
   gt_captions: A list of lists (multi-ref) of strings (paragraphs).

  Returns:
    metrics: The NLP metrics of the predictions computed at the corpus level.
  """
  scorers = {
      'CIDER': Cider(),
      'METEOR': Meteor(),
  }
  all_gts = {}
  all_preds = {}
  for i, (preds, gts) in enumerate(zip(predicted_captions, gt_captions)):
    all_preds[str(i)] = [' '.join(parse_sent(preds))]
    all_gts[str(i)] = [' '.join(parse_sent(gt)) for gt in gts]

  metrics = collections.defaultdict(list)
  for scorer_name, scorer in scorers.items():
    score = scorer.compute_score(all_gts, all_preds)
    score = np.nan_to_num(score[0])
    metrics['Para_' + scorer_name] = float(score)

  logging.info('Closing Meteor')
  with scorers['METEOR'].lock:
    scorers['METEOR'].meteor_p.stdin.close()
    scorers['METEOR'].meteor_p.stdout.close()
    scorers['METEOR'].meteor_p.kill()
    scorers['METEOR'].meteor_p.wait()
  del scorers

  return metrics
