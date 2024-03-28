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

"""Functions for evaluating action detection with ActivityNet metrics.

For proposal metrics, see:
DAPs: Deep Action Proposals for Action Understanding. V. Escorcia , F. C.
Heilbron, J. C. Niebles, B. Ghanem. ECCV 2016.

For detection metrics, see:
ActivityNet Challenge. http://activity-net.org/challenges/2017/.
"""

import logging

from activitynet.evaluation import eval_detection
import numpy as np
import pandas as pd


def compute_average_precision_detection_ng(
    ground_truth: pd.DataFrame,
    prediction: pd.DataFrame,
    tiou_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10)
) -> np.ndarray:
  """Computes average precision (detection task).

  Notes:
  The open source implementation is extremely slow, due to the way it iterates
  through the prediction dataframe.  A simple change of the dataframes to numpy
  arrays will see orders of magnitude of speed-up.  There are more ways to
  improve, e.g., grouping and processing the ground truth and predictions by
  videos, which will eliminate the double indexing in lock_gt.  Vectorization,
  though doable, is awkward given the greedy method quoted below:

    If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

  I have kept as much of the original code, as long as it doesn't impact the
  speed greatly.

  Args:
    ground_truth: df Data frame containing the ground truth instances. Required
      fields ['video-id', 't-start', 't-end']
    prediction: df Data frame containing the prediction instances. Required
      fields ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds: 1d array, Temporal intersection over union threshold.

  Returns:
      ap: Average precision scores.
  """
  npos = float(len(ground_truth))
  lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
  # Sort predictions by decreasing score order.
  sort_idx = prediction['score'].values.argsort()[::-1]
  prediction = prediction.loc[sort_idx].reset_index(drop=True)

  # Initialize true positive and false positive vectors.
  tp = np.zeros((len(tiou_thresholds), len(prediction)))
  fp = np.zeros((len(tiou_thresholds), len(prediction)))

  # Adaptation to query faster.
  # Notes:
  # Transform the dataframes to numpy arrays here. The indices are used
  # to match the ground truth segments only once, in a "winners take all"
  # manner.  We have to do the two-step dance (gt_dict to gt_indices
  # and gt_segments) because we need to keep "index" consistent
  # between the two.
  ground_truth_gbvn = ground_truth.groupby('video-id')
  gt_dict = {
      k: g.reset_index()[['index', 't-start', 't-end']].to_numpy()
      for k, g in ground_truth_gbvn
  }
  gt_indices = {}
  gt_segments = {}
  for k, g in gt_dict.items():
    gt_indices[k] = gt_dict[k][:, 0].astype(int)
    gt_segments[k] = gt_dict[k][:, 1:]

  pred_ids = prediction['video-id']
  pred_segments = prediction[['t-start', 't-end']].to_numpy()
  npred = pred_ids.shape[0]

  # Assigning true positive to truly grount truth instances.
  for idx in range(npred):

    try:
      # Check if there is at least one ground truth in the video associated.
      pred_id = pred_ids[idx]
      gt_index = gt_indices[pred_id]
      gt_segment = gt_segments[pred_id]
    except Exception:  # pylint: disable = broad-except
      fp[:, idx] = 1
      continue

    tiou_arr = eval_detection.segment_iou(pred_segments[idx], gt_segment)
    # We would like to retrieve the predictions with highest tiou score.
    tiou_sorted_idx = tiou_arr.argsort()[::-1]
    for tidx, tiou_thr in enumerate(tiou_thresholds):
      for jdx in tiou_sorted_idx:
        if tiou_arr[jdx] < tiou_thr:
          fp[tidx, idx] = 1
          break
        if lock_gt[tidx, gt_index[jdx]] >= 0:
          continue
        # Assign as true positive after the filters above.
        tp[tidx, idx] = 1
        lock_gt[tidx, gt_index[jdx]] = idx
        break

      if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
        fp[tidx, idx] = 1

  ap = np.zeros(len(tiou_thresholds))

  for tidx in range(len(tiou_thresholds)):
    # Computing prec-rec
    this_tp = np.cumsum(tp[tidx, :]).astype(np.float32)
    this_fp = np.cumsum(fp[tidx, :]).astype(np.float32)
    rec = this_tp / npos
    prec = this_tp / (this_tp + this_fp)
    ap[tidx] = eval_detection.interpolated_prec_rec(prec, rec)

  return ap


def evaluate_detection_results_anet(result_lists,
                                    num_classes,
                                    label_id_offset=0,
                                    excluded_classes=(),
                                    class_weights=None,
                                    ):
  """Computes ActivityNet detection metrics given groundtruth and detections.

  This function computes official ActivityNet detection metrics using the third
  party evaluation toolkit. This function by default takes detections and
  groundtruth segments encoded in result_lists and writes evaluation results to
  tf summaries which can be viewed on tensorboard.

  Args:
    result_lists: A dictionary holding lists of groundtruth and detection data
      corresponding to each sequence being evaluated.
      The following keys are required:
        'video_id': A list of string ids
        'detection_segments': A list of float32 numpy arrays of shape [N, 2].
        'detection_scores': A list of float32 numpy arrays of shape [N].
        'detection_classes': A list of int32 numpy arrays of shape [N].
        'groundtruth_segments': A list of float32 numpy arrays of shape [M, 2].
        'groundtruth_classes': A list of int32 numpy arrays of shape [M].
      Note that it is okay to have additional fields in result_lists --- they
      are simply ignored.
    num_classes: (int scalar) Number of classes excluding background.
    label_id_offset: An integer offset for the label space.
    excluded_classes: A list of (int) class indices to be excluded, after adding
      the label_id_offset.
    class_weights: A 1-d numpy array containing weights for the classes.

  Returns:
    A dictionary of metric names to scalar values.
      'eval_detection/mAP@0.5:0.05:0.95IOU': Mean Average Precision averaged
          over IOU interval [0.5, 0.95].
      'eval_detection/mAP@0.5:0.05:0.95IOU_present_classes': Mean Average
          Precision averaged over IOU interval [0.5, 0.95] only including
          classes that exist in the groundtruth.
      'eval_detection/mAP@IOU': Mean Average Precision versus IOU.
      'eval_detection_per_class/avgAP@': Per-class Average Precision averaged
          over IOU interval [0.05, 0.95].

  Raises:
    ValueError: If the set of keys in result_lists is not a superset of the
      expected list of keys. Unexpected keys are ignored.
    ValueError: If the lists in result_lists have inconsistent sizes.
  """
  # Check for expected keys in result_lists.
  expected_keys = [
      'detection_segments', 'detection_scores', 'detection_classes', 'video_id',
      'groundtruth_segments', 'groundtruth_classes'
  ]
  if not set(expected_keys).issubset(set(result_lists.keys())):
    raise ValueError('result_lists does not have expected key set: ' + str(
        set(expected_keys).difference(set(result_lists.keys()))))
  num_results = len(result_lists[expected_keys[0]])
  for key in expected_keys:
    if len(result_lists[key]) != num_results:
      raise ValueError('Inconsistent list sizes in result_lists')

  # TODO(ywchao): input categories instead of num_classes.

  logging.info('Computing ActivityNet detection metrics on results.')

  ground_truth = _convert_ground_truth(result_lists, num_results)
  detection = _convert_detection(result_lists, num_results)

  # Compute AP.
  tiou_thresholds = np.linspace(0.05, 0.95, 19)
  ap = np.zeros((len(tiou_thresholds), num_classes))
  for cidx in range(num_classes):
    gt_idx = ground_truth['label'] == cidx + label_id_offset
    pred_idx = detection['label'] == cidx + label_id_offset
    ap[:, cidx] = compute_average_precision_detection_ng(
        ground_truth.loc[gt_idx].reset_index(drop=True),
        detection.loc[pred_idx].reset_index(drop=True),
        tiou_thresholds=tiou_thresholds)
  # Exclude unwanted classes.
  keep = [i for i in range(num_classes) if i not in excluded_classes]

  # Gather APs for classes present in the groundtruth.
  classes_with_gt_ap = []
  for cidx in range(num_classes):
    gt_exists = np.sum(ground_truth['label'] == cidx)
    if gt_exists and cidx not in excluded_classes:
      classes_with_gt_ap.append(ap[:, cidx])
  classes_with_gt_ap = np.column_stack(classes_with_gt_ap)

  # Exclude NaN AP.
  mean_ap = np.nanmean(ap[:, keep], axis=1)

  metrics = {
      'eval_detection/mAP@0.5:0.05:0.95IOU':
          mean_ap[9:].mean(),
      'eval_detection/mAP@0.3:0.1:0.7IOU': mean_ap[5:14:2].mean(),
      'eval_detection_present_classes/mAP@0.5:0.05:0.95IOU':
          np.nanmean(classes_with_gt_ap, axis=1)[9:].mean()
  }
  if class_weights is not None:
    weighted_mean_ap = np.nansum(ap[:, keep] * class_weights[keep], axis=1)
    metrics['eval_detection/wmAP@0.5:0.05:0.95IOU'] = weighted_mean_ap[9:].mean(
    )
    for idx in range(ap.shape[0]):
      display_name = 'eval_detection/wmAP@{:0.2f}IOU'.format(
          tiou_thresholds[idx])
      metrics[display_name] = np.nansum(ap[idx, keep] * class_weights[keep])

  for idx in range(ap.shape[0]):
    display_name = 'eval_detection/mAP@{:0.2f}IOU'.format(tiou_thresholds[idx])
    metrics[display_name] = np.nanmean(ap[idx, keep])
    present_classes_display_name = (
        'eval_detection_present_classes/mAP@{:0.2f}IOU'.format(
            tiou_thresholds[idx]))
    metrics[present_classes_display_name] = np.nanmean(classes_with_gt_ap[idx])

  for idx in range(ap.shape[1]):
    display_name = 'eval_detection_per_class/avgAP@{:02d}'.format(idx)
    metrics[display_name] = np.nanmean(ap[:, idx])

  return metrics


def compute_ious(segments1, segments2):
  """Computes IOUs between two sets of segments.

  Args:
    segments1: First set of segments of size [N, 2]
    segments2: Second set of segments of size [M, 2]

  Returns:
    ious: All pairs IOU array of shape [N, M]
  """
  all_pairs_min_end = np.minimum(segments1[:, 1:], np.transpose(segments2[:,
                                                                          1:]))
  all_pairs_max_start = np.maximum(segments1[:, 0:1],
                                   np.transpose(segments2[:, 0:1]))
  intersection = np.maximum(0.0, all_pairs_min_end - all_pairs_max_start)
  union = (segments1[:, 1:] - segments1[:, 0:1]
           ) + np.transpose(segments2[:, 1:] - segments2[:, 0:1]) - intersection
  return intersection / (union + 1e-8)


def _convert_ground_truth(result_lists, num_results):
  """Converts the format of groundtruth segments for evaluation.

  Args:
    result_lists: A dictionary holding lists of groundtruth.
    num_results: Length of the lists.

  Returns:
    A pd.DataFrame containing the groundtruth instances.
  """
  videos, t_starts, t_ends, labels = [], [], [], []
  for vidx in range(num_results):
    for sidx in range(result_lists['groundtruth_segments'][vidx].shape[0]):
      videos.append(result_lists['video_id'][vidx])
      t_starts.append(result_lists['groundtruth_segments'][vidx][sidx][0])
      t_ends.append(result_lists['groundtruth_segments'][vidx][sidx][1])
      labels.append(result_lists['groundtruth_classes'][vidx][sidx])

  return pd.DataFrame({
      'video-id': videos,
      't-start': t_starts,
      't-end': t_ends,
      'label': labels
  })


def _convert_proposal(result_lists, num_results):
  """Converts the format of proposals for evaluation.

  Args:
    result_lists: A dictionary holding lists of proposals.
    num_results: Length of the lists.

  Returns:
    A pd.DataFrame containing the proposal instances.
  """
  videos, t_starts, t_ends = [], [], []
  score_lst = []
  for vidx in range(num_results):
    for sidx in range(result_lists['proposal_segments'][vidx].shape[0]):
      videos.append(result_lists['video_id'][vidx])
      t_starts.append(result_lists['proposal_segments'][vidx][sidx][0])
      t_ends.append(result_lists['proposal_segments'][vidx][sidx][1])
      score_lst.append(result_lists['proposal_scores'][vidx][sidx])

  return pd.DataFrame({
      'video-id': videos,
      't-start': t_starts,
      't-end': t_ends,
      'score': score_lst
  })


def _convert_detection(result_lists, num_results):
  """Converts the format of detections for evaluation.

  Args:
    result_lists: A dictionary holding lists of detections.
    num_results: Length of the lists.

  Returns:
    A pd.DataFrame containing the detection instances.
  """
  videos, t_starts, t_ends = [], [], []
  labels, score_lst = [], []
  for vidx in range(num_results):
    for sidx in range(result_lists['detection_segments'][vidx].shape[0]):
      videos.append(result_lists['video_id'][vidx])
      t_starts.append(result_lists['detection_segments'][vidx][sidx][0])
      t_ends.append(result_lists['detection_segments'][vidx][sidx][1])
      labels.append(result_lists['detection_classes'][vidx][sidx])
      score_lst.append(result_lists['detection_scores'][vidx][sidx])

  return pd.DataFrame({
      'video-id': videos,
      't-start': t_starts,
      't-end': t_ends,
      'label': labels,
      'score': score_lst
  })
