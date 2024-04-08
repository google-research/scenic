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

"""Evaluation of spatial-temporal grounding of VidSTG dataset.

The main body of the code is from TubeDETR
https://github.com/antoyang/TubeDETR/blob/main/datasets/vidstg_eval.py
"""
import json
import logging
import os
import numpy as np

from scenic.projects.densevoc import densevoc_evaluator
import tensorflow as tf

# pylint: disable=logging-fstring-interpolation


class VidSTGiouEvaluator:
  """VidSTG evaluator."""

  def __init__(
      self,
      annotations,
      subset="val",
      iou_thresholds=(0.3, 0.5),
      fps=5,
      video_max_len=200,
      tmp_loc=True,
  ):
    """Initialize.

    Args:
      annotations: VidSTG annotations
      subset: train, val or test
      iou_thresholds: IoU thresholds for the vIoU metrics
      fps: number of frames per second
      video_max_len: maximum number of frames to be extracted from a video
      tmp_loc: whether to evaluate temporal localization
    """

    assert subset in ["train", "test", "val"], f"Wrong VidSTG subset {subset}"

    self.iou_thresholds = iou_thresholds
    self.tmp_loc = tmp_loc
    self.anns = annotations
    # Map video_id to list of corresponding frames to forward and list of
    # corresponding frames in the GT tube.
    self.vid2imgids = {}
    self.vid2steds = {}  # map video_id to [start, end] of the GT tube
    self.img2box = {}  # map video_id + frame_id to bbox
    for video in self.anns["videos"]:  #
      video_id = int(video["video_id"])
      video_fps = video["fps"]  # used for extraction
      sampling_rate = fps / video_fps
      assert sampling_rate <= 1  # downsampling at fps
      start_frame = (
          video["start_frame"] if self.tmp_loc else video["tube_start_frame"])
      end_frame = (
          video["end_frame"] if self.tmp_loc else video["tube_start_frame"])
      frame_ids = [start_frame]
      for frame_id in range(start_frame, end_frame):
        if int(frame_ids[-1] * sampling_rate) < int(frame_id * sampling_rate):
          frame_ids.append(frame_id)
      # Temporal downsampling if there are too many frames.
      if len(frame_ids) > video_max_len:
        frame_ids = [
            frame_ids[(j * len(frame_ids)) // video_max_len]
            for j in range(video_max_len)
        ]
      inter_frames = []
      self.vid2steds[video_id] = [
          video["tube_start_frame"], video["tube_end_frame"]]
      for frame_id in frame_ids:
        if video["tube_start_frame"] <= frame_id < video["tube_end_frame"]:
          x1, y1, w, h = self.anns["trajectories"][video["original_video_id"]][
              str(video["target_id"])][str(frame_id)]["bbox"]
          x2 = x1 + w
          y2 = y1 + h
          self.img2box[f"{video_id}_{frame_id}"] = [[x1, y1, x2, y2]]
          inter_frames.append(f"{video_id}_{frame_id}")
      self.vid2imgids[video_id] = [frame_ids, inter_frames]

    logging.info(f"VidSTG subset contains {len(self.vid2imgids)} videos")

  def evaluate(self, predictions, video_predictions):
    """Evaluate on given predictions."""
    if len(video_predictions) < len(self.vid2imgids):
      num_miss = len(self.vid2imgids) - len(video_predictions)
      logging.info(f"{num_miss} video predictions missing")
    if len(predictions) < len(self.img2box):
      num_miss = len(self.img2box) - len(predictions)
      logging.info(f"{num_miss} box predictions missing")
    vid_metrics = {}
    for video_id, video_pred in video_predictions.items():
      if video_id in vid_metrics:
        logging.info(
            f"Warning, multiple predictions found for video {video_id}")
        continue
      if video_id not in self.vid2imgids:
        logging.info(
            f"Warning, no image predictions found for video {video_id}")
        continue
      if self.tmp_loc:
        gt_sted = self.vid2steds[video_id]
        pred_sted = video_pred["sted"]
      qtype = video_pred["qtype"]
      frame_ids, inter_frames = self.vid2imgids[video_id]

      # compute temporal iou
      if self.tmp_loc:
        max_start = max(gt_sted[0], pred_sted[0])
        min_end = min(gt_sted[1], pred_sted[1])
        min_start = min(gt_sted[0], pred_sted[0])
        max_end = max(gt_sted[1], pred_sted[1])
        if min_end <= max_start:
          tiou = 0
        else:
          intersection = min_end - max_start
          gt_span = gt_sted[1] - gt_sted[0]
          pred_span = pred_sted[1] - pred_sted[0]
          union = gt_span + pred_span - intersection
          tiou = intersection / union

        # compute viou and gt_viou
        vid_metrics[video_id] = {
            "gt_sted": gt_sted,
            "pred_sted": pred_sted,
            "tiou": tiou,
            "qtype": qtype,
            "img_metrics": {},
        }
        union_predgt = [
            frame_id
            for frame_id in frame_ids
            if min_start <= frame_id < max_end
        ]
        inter_predgt = set([
            frame_id
            for frame_id in frame_ids
            if max_start <= frame_id < min_end
        ])
        viou = 0
      else:
        vid_metrics[video_id] = {
            "qtype": qtype,
            "img_metrics": {},
        }
        union_predgt = frame_ids
        inter_predgt = frame_ids
      gt_viou = 0

      # Iterate on all frames of the annotated moment to update GT metrics
      for image_id in inter_frames:
        if image_id not in predictions:
          logging.info(f"No prediction for frame {image_id}")
          continue
        else:
          pred_boxes = predictions[image_id]["boxes"]
        gt_boxes = self.img2box[image_id]
        pred_boxes_np = np.array(pred_boxes)
        gt_boxes_np = np.array(gt_boxes)
        pred_boxes_np[:, 2:] -= pred_boxes_np[:, :2]
        gt_boxes_np[:, 2:] -= gt_boxes_np[:, :2]
        iou = densevoc_evaluator.box_iou(pred_boxes_np, gt_boxes_np)[0][0]
        frame_id = int(image_id.split("_")[1])
        vid_metrics[video_id]["img_metrics"][image_id] = {
            "iou": iou,
            "pred_box": pred_boxes[0],
            "gt_box": gt_boxes[0],
        }
        # Update viou if this frame is in the intersection between the
        # annotated moment and the predicted moment
        if (frame_id in inter_predgt and self.tmp_loc):
          viou += iou
        gt_viou += iou

      if self.tmp_loc:  # compute viou@R
        viou = viou / max(len(union_predgt), 1)
        vid_metrics[video_id]["viou"] = viou
        recalls = {thresh: 0 for thresh in self.iou_thresholds}
        for thresh in self.iou_thresholds:
          if viou > thresh:
            recalls[thresh] += 1
        vid_metrics[video_id].update({
            f"viou@{thresh}": recalls[thresh]
            for thresh in self.iou_thresholds
        })

      # compute gt_viou@R
      gt_viou = gt_viou / max(len(inter_frames), 1)
      vid_metrics[video_id]["gt_viou"] = gt_viou
      gt_recalls = {thresh: 0 for thresh in self.iou_thresholds}
      for thresh in self.iou_thresholds:
        if gt_viou > thresh:
          gt_recalls[thresh] += 1
      vid_metrics[video_id].update({
          f"gt_viou@{thresh}": gt_recalls[thresh]
          for thresh in self.iou_thresholds
      })

    return vid_metrics


class VidSTGEvaluator(object):
  """VidSTG evaluator."""

  def __init__(
      self,
      annotations_loc,
      iou_thresholds=(0.3, 0.5),
      fps=5,
      video_max_len=200,
      tmp_loc=True,
  ):
    """Init evaluator.

    Args:
      annotations_loc: path to VidSTG annotations
      iou_thresholds: IoU thresholds for the vIoU metrics
      fps: number of frames per second
      video_max_len: maximum number of frames to be extracted from a video
      tmp_loc: temporal localization
    """
    subset = annotations_loc[
        annotations_loc.rfind("/") + 1: annotations_loc.rfind("_")]
    annotations = json.load(tf.io.gfile.GFile(annotations_loc, "r"))
    annotations["videos"] = [
        x for x in annotations["videos"] if x["qtype"] == "declarative"]
    self.evaluator = VidSTGiouEvaluator(
        annotations,
        subset=subset,
        iou_thresholds=iou_thresholds,
        fps=fps,
        video_max_len=video_max_len,
        tmp_loc=tmp_loc,
    )
    self.predictions = {}
    self.video_predictions = {}
    self.results = None
    self.iou_thresholds = iou_thresholds
    self.tmp_loc = tmp_loc
    self.tsa_weights = {}
    self.text_weights = {}
    self.spatial_weights = {}
    self.pred_sted = {}

  def accumulate(self):
    pass

  def update(self, predictions):
    """Update per-frame localization predictions.

    Args:
      predictions: dict of image_id ('{video_id}_{frame_id}') to dict
        {'boxes': [[l, t, r, b]]}
    """
    self.predictions.update(predictions)

  def video_update(self, video_predictions):
    """Update per-video temporal localization predictions.

    Args:
      video_predictions: dict of video_id to dict {'sted': [st, ed]}
    """
    self.video_predictions.update(video_predictions)

  def compute_metrics(self):
    """Summarize results."""
    self.results = self.evaluator.evaluate(
        self.predictions, self.video_predictions)
    categories = set(x["qtype"] for x in self.results.values())
    metrics = {}
    counter = {}
    for category in categories:  # init metrics
      metrics[category] = {"gt_viou": 0}
      if self.tmp_loc:
        metrics[category].update({"tiou": 0, "viou": 0})
      for thresh in self.iou_thresholds:
        if self.tmp_loc:
          metrics[category][f"viou@{thresh}"] = 0
        metrics[category][f"gt_viou@{thresh}"] = 0
      counter[category] = 0
    for x in self.results.values():  # sum results
      qtype = x["qtype"]
      if self.tmp_loc:
        metrics[qtype]["tiou"] += x["tiou"]
        metrics[qtype]["viou"] += x["viou"]
      metrics[qtype]["gt_viou"] += x["gt_viou"]
      for thresh in self.iou_thresholds:
        if self.tmp_loc:
          metrics[qtype][f"viou@{thresh}"] += x[f"viou@{thresh}"]
        metrics[qtype][f"gt_viou@{thresh}"] += x[f"gt_viou@{thresh}"]
      counter[qtype] += 1
    for category in categories:  # average results per category
      for key in metrics[qtype]:
        metrics[category][key] = metrics[category][key] / counter[category]
        logging.info(f"{category} {key}: {metrics[category][key]:.4f}")
    out = {  # pylint: disable=g-complex-comprehension
        f"{qtype}_{name}": metrics[qtype][name]
        for qtype in metrics
        for name in metrics[qtype]
    }
    return out

  def write_pred_annotations_to_file(self, path):
    """Writes predictions to file in JSON format."""
    out = {}
    out["predictions"] = self.predictions
    out["video_predictions"] = self.video_predictions
    out["vid_metrics"] = self.results
    if not tf.io.gfile.exists(path):
      tf.io.gfile.makedirs(path)
    json_file_name = "vidstg_predictions.json"
    json_file_path = os.path.join(path, json_file_name)

    def _convert_to_serializable(obj):
      if isinstance(obj, np.ndarray):
        return obj.tolist()
      elif isinstance(obj, np.float32):
        return float(obj)
      else:
        raise TypeError(f"Unserializable object {obj} of type {type(obj)}")

    with tf.io.gfile.GFile(json_file_path, "w") as f:
      f.write(json.dumps(out, default=_convert_to_serializable))
    logging.info("Predicted annotations are stored in %s.", json_file_path)

  def clear(self):
    self.predictions = {}
    self.video_predictions = {}
    self.results = None
